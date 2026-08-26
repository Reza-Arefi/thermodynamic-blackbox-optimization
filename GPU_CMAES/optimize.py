"""
CMA-ES (Hansen) with black-box batch offspring evaluations.

Stopping criteria identical to GPU-PSO / GPU-DE.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional, Tuple

import numpy as np

_GIT = Path(__file__).resolve().parents[1]
if str(_GIT) not in sys.path:
    sys.path.insert(0, str(_GIT))

from common.gpu_stopping import check_gpu_stopping, relative_improvement
from common.objective import BatchObjective, Objective, as_batch
from common.params import default_bounds

from .settings import GPUCMAESSettings


class CMAParams(NamedTuple):
    n: int
    lam: int
    mu: int
    weights: np.ndarray
    mu_eff: float
    c_sigma: float
    d_sigma: float
    c_c: float
    c_1: float
    c_mu: float
    chi_n: float


def build_cma_params(n: int, lam: int) -> CMAParams:
    n = int(n)
    lam = max(int(lam), 4)
    mu = lam // 2
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1, dtype=float))
    weights = weights / np.sum(weights)
    mu_eff = float(1.0 / np.sum(weights ** 2))
    c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
    d_sigma = 1 + 2 * max(0.0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
    c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
    c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
    c_mu = min(1 - c_1, 2 * ((mu_eff - 2) + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
    chi_n = float(np.sqrt(n) * ((1 - 1 / (4 * n)) + 1 / (21 * n * n)))
    return CMAParams(
        n=n, lam=lam, mu=mu, weights=weights.astype(float), mu_eff=mu_eff,
        c_sigma=float(c_sigma), d_sigma=float(d_sigma), c_c=float(c_c),
        c_1=float(c_1), c_mu=float(c_mu), chi_n=chi_n,
    )


def eigendecompose_C(C: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    C = 0.5 * (C + C.T)
    eigvals, B = np.linalg.eigh(C)
    eigvals = np.maximum(eigvals, 1e-30)
    D = np.sqrt(eigvals)
    return B, D


def sample_offspring(
    mean: np.ndarray,
    sigma: float,
    B: np.ndarray,
    D: np.ndarray,
    lam: int,
    lo: np.ndarray,
    hi: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = mean.shape[0]
    z = rng.standard_normal(size=(lam, n))
    y = (B * D) @ z.T
    y = y.T
    x = mean[None, :] + sigma * y
    x = np.clip(x, lo, hi)
    return x, z, y


def cma_update(
    *,
    mean: np.ndarray,
    sigma: float,
    C: np.ndarray,
    pc: np.ndarray,
    ps: np.ndarray,
    B: np.ndarray,
    D: np.ndarray,
    y: np.ndarray,
    fitness: np.ndarray,
    params: CMAParams,
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(fitness)
    y_sel = y[order[: params.mu]]
    z_sel = (B.T @ y_sel.T).T / D
    y_w = params.weights @ y_sel
    z_w = params.weights @ z_sel
    mean_new = mean + sigma * y_w
    ps_new = (1 - params.c_sigma) * ps + np.sqrt(
        params.c_sigma * (2 - params.c_sigma) * params.mu_eff
    ) * (B @ z_w)
    ps_norm = float(np.linalg.norm(ps_new))
    sigma_new = sigma * float(
        np.exp((params.c_sigma / params.d_sigma) * (ps_norm / params.chi_n - 1))
    )
    if (not np.isfinite(sigma_new)) or sigma_new <= 0:
        sigma_new = sigma * 0.5
    sigma_new = float(np.clip(sigma_new, 1e-20, 1e6))
    denom = np.sqrt(1 - (1 - params.c_sigma) ** (2 * (params.n + 1)))
    h_sig = float(ps_norm / max(denom, 1e-20) < (1.4 + 2 / (params.n + 1)) * params.chi_n)
    pc_new = (1 - params.c_c) * pc + h_sig * np.sqrt(
        params.c_c * (2 - params.c_c) * params.mu_eff
    ) * y_w
    C_mu = (y_sel.T * params.weights) @ y_sel
    C_new = (
        (1 - params.c_1 - params.c_mu) * C
        + params.c_1
        * (np.outer(pc_new, pc_new) + (1 - h_sig) * params.c_c * (2 - params.c_c) * C)
        + params.c_mu * C_mu
    )
    C_new = 0.5 * (C_new + C_new.T)
    B_new, D_new = eigendecompose_C(C_new)
    return mean_new, sigma_new, C_new, pc_new, ps_new, B_new, D_new


def run_gpu_cmaes(
    objective: BatchObjective | Objective,
    x0: np.ndarray,
    settings: Optional[GPUCMAESSettings] = None,
    bounds: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    settings = settings or GPUCMAESSettings()
    obj = as_batch(objective)
    rng = np.random.default_rng(settings.seed)

    x0 = np.asarray(x0, dtype=float).ravel()
    n = x0.size
    bounds = default_bounds(x0) if bounds is None else np.asarray(bounds, dtype=float)
    lo, hi = bounds[:, 0], bounds[:, 1]
    half = 0.5 * (hi - lo)

    if settings.use_hansen_pop:
        lam = int(4 + np.floor(3 * np.log(n)))
    else:
        lam = settings.pop_size or int(min(5 * n, 50))
    lam = max(lam, 4)

    params = build_cma_params(n, lam)
    mean = x0.copy()
    sigma = float(settings.sigma0 * float(np.mean(half)))
    C = np.eye(n)
    pc = np.zeros(n)
    ps = np.zeros(n)
    B, D = eigendecompose_C(C)

    # initial eval at mean
    best_x = mean.copy()
    best_loss = float(obj.evaluate(mean))
    n_evals = 1

    def diversity_from_pop(p: np.ndarray) -> float:
        return float(np.mean(np.std(p, axis=0)))

    stag = 0
    prev_best = best_loss
    init_div = None
    history = {"iteration": [0], "loss": [best_loss], "contraction_ratio": [1.0]}
    stop_reason = ""
    converged = False
    scientific = False
    n_done = 0

    for it in range(1, settings.max_iter + 1):
        pop, z, y = sample_offspring(mean, sigma, B, D, lam, lo, hi, rng)
        fitness = np.asarray(obj.evaluate_many([pop[i] for i in range(lam)]), dtype=float)
        n_evals += lam

        i_best = int(np.argmin(fitness))
        if fitness[i_best] < best_loss:
            best_loss = float(fitness[i_best])
            best_x = pop[i_best].copy()

        mean, sigma, C, pc, ps, B, D = cma_update(
            mean=mean, sigma=sigma, C=C, pc=pc, ps=ps, B=B, D=D, y=y, fitness=fitness, params=params
        )
        mean = np.clip(mean, lo, hi)

        div = diversity_from_pop(pop)
        if init_div is None:
            init_div = max(div, 1e-30)
        cr = div / init_div
        imp = relative_improvement(prev_best, best_loss)
        tol = settings.stop.stagnation_tol if settings.stop.simple_stop else settings.stop.rel_improve_tol
        stag = 0 if imp > tol else stag + 1
        prev_best = best_loss

        history["iteration"].append(it)
        history["loss"].append(best_loss)
        history["contraction_ratio"].append(cr)
        n_done = it

        hit, reasons, sci = check_gpu_stopping(
            it=it,
            max_iter=settings.max_iter,
            stagnation_counter=stag,
            contraction_ratio=cr,
            best_loss=best_loss,
            settings=settings.stop,
        )
        if hit:
            converged = True
            scientific = sci
            stop_reason = " | ".join(reasons)
            break

    return {
        "method": "GPU-CMAES",
        "best_x": best_x,
        "best_loss": best_loss,
        "iterations": n_done,
        "evaluations": n_evals,
        "pop_size": lam,
        "converged": converged,
        "scientific_convergence": scientific,
        "stop_reason": stop_reason,
        "history": history,
    }
