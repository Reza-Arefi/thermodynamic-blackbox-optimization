"""
Differential Evolution (DE/rand/1/bin) with black-box batch evaluations.

Stopping criteria identical to GPU-PSO (Stop A/B/C or unified L/S/C).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

_GIT = Path(__file__).resolve().parents[1]
if str(_GIT) not in sys.path:
    sys.path.insert(0, str(_GIT))

from common.gpu_stopping import check_gpu_stopping, relative_improvement
from common.objective import BatchObjective, Objective, as_batch
from common.params import default_bounds

from .settings import GPUDESettings


def de_rand_1_bin_trials(
    population: np.ndarray,
    F: float,
    CR: float,
    lo: np.ndarray,
    hi: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    pop = np.asarray(population, dtype=float)
    Np, D = pop.shape
    if Np < 4:
        raise ValueError(f"DE/rand/1 needs pop_size >= 4, got {Np}")
    donors = np.empty((Np, 3), dtype=np.int64)
    all_idx = np.arange(Np)
    for i in range(Np):
        donors[i] = rng.choice(all_idx[all_idx != i], size=3, replace=False)
    r1, r2, r3 = donors[:, 0], donors[:, 1], donors[:, 2]
    mutants = pop[r1] + F * (pop[r2] - pop[r3])
    cross = rng.random((Np, D)) <= CR
    j_rand = rng.integers(0, D, size=Np)
    cross[np.arange(Np), j_rand] = True
    trials = np.where(cross, mutants, pop)
    return np.clip(trials, lo, hi)


def run_gpu_de(
    objective: BatchObjective | Objective,
    x0: np.ndarray,
    settings: Optional[GPUDESettings] = None,
    bounds: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    settings = settings or GPUDESettings()
    obj = as_batch(objective)
    rng = np.random.default_rng(settings.seed)

    x0 = np.asarray(x0, dtype=float).ravel()
    n = x0.size
    bounds = default_bounds(x0) if bounds is None else np.asarray(bounds, dtype=float)
    lo, hi = bounds[:, 0], bounds[:, 1]

    pop_size = settings.pop_size or int(min(5 * n, 50))
    pop_size = max(pop_size, 4)

    pop = rng.uniform(lo, hi, size=(pop_size, n))
    pop[0] = x0
    fitness = np.asarray(obj.evaluate_many([pop[i] for i in range(pop_size)]), dtype=float)
    n_evals = pop_size
    best_idx = int(np.argmin(fitness))
    best_x = pop[best_idx].copy()
    best_loss = float(fitness[best_idx])

    def diversity(p: np.ndarray) -> float:
        return float(np.mean(np.std(p, axis=0)))

    init_div = max(diversity(pop), 1e-30)
    stag = 0
    prev_best = best_loss
    history = {"iteration": [0], "loss": [best_loss], "contraction_ratio": [1.0]}
    stop_reason = ""
    converged = False
    scientific = False
    n_done = 0

    for it in range(1, settings.max_iter + 1):
        trials = de_rand_1_bin_trials(pop, settings.F, settings.CR, lo, hi, rng)
        trial_fit = np.asarray(
            obj.evaluate_many([trials[i] for i in range(pop_size)]), dtype=float
        )
        n_evals += pop_size
        better = trial_fit <= fitness
        pop[better] = trials[better]
        fitness[better] = trial_fit[better]

        best_idx = int(np.argmin(fitness))
        best_loss = float(fitness[best_idx])
        best_x = pop[best_idx].copy()

        cr = diversity(pop) / init_div
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
        "method": "GPU-DE",
        "best_x": best_x,
        "best_loss": best_loss,
        "iterations": n_done,
        "evaluations": n_evals,
        "pop_size": pop_size,
        "converged": converged,
        "scientific_convergence": scientific,
        "stop_reason": stop_reason,
        "history": history,
    }
