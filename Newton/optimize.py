"""
Damped Newton / Levenberg–Marquardt on a black-box objective.

Flash-free: only `objective.evaluate(x)`. FD gradient + Hessian, Armijo line search.
Faithful to `newton/newton_rPar.py` stopping and update logic.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

_GIT = Path(__file__).resolve().parents[1]
if str(_GIT) not in sys.path:
    sys.path.insert(0, str(_GIT))

from common.objective import Objective
from common.params import default_bounds
from common.stopping_unified import (
    UnifiedStopConfig,
    check_unified_stop,
    update_stagnation_counter,
)

from .settings import NewtonSettings
from .stopping import check_newton_native_stop


def _fd_delta(value: float, relative: float = 0.01, absolute_floor: float = 1e-6) -> float:
    delta = abs(value) * relative
    return absolute_floor if delta < absolute_floor else delta


def _gradient_fd(
    objective: Objective, x: np.ndarray, f0: float, settings: NewtonSettings
) -> Tuple[np.ndarray, int]:
    g = np.zeros_like(x)
    n_evals = 0
    for i in range(x.size):
        d = _fd_delta(x[i], settings.fd_relative, settings.fd_floor)
        xp = x.copy()
        xp[i] += d
        g[i] = (float(objective.evaluate(xp)) - f0) / d
        n_evals += 1
    return g, n_evals


def _hessian_fd(
    objective: Objective, x: np.ndarray, f0: float, settings: NewtonSettings
) -> Tuple[np.ndarray, int]:
    n = x.size
    H = np.zeros((n, n), dtype=float)
    deltas = [_fd_delta(x[i], settings.fd_relative, settings.fd_floor) for i in range(n)]
    n_evals = 0
    f_plus = np.zeros(n)
    for i in range(n):
        xp = x.copy()
        xp[i] += deltas[i]
        f_plus[i] = float(objective.evaluate(xp))
        n_evals += 1
    for i in range(n):
        for j in range(i, n):
            xp = x.copy()
            xp[i] += deltas[i]
            xp[j] += deltas[j]
            fij = float(objective.evaluate(xp))
            n_evals += 1
            H[i, j] = (fij - f_plus[i] - f_plus[j] + f0) / (deltas[i] * deltas[j])
            H[j, i] = H[i, j]
    return H, n_evals


def _newton_step(H: np.ndarray, g: np.ndarray, lam: float) -> Tuple[np.ndarray, float, float]:
    lam_used = max(lam, 0.0)
    eye = np.eye(len(g))
    for _ in range(8):
        H_reg = H + lam_used * eye
        try:
            try:
                L = np.linalg.cholesky(H_reg)
                step = np.linalg.solve(L.T, np.linalg.solve(L, g))
            except np.linalg.LinAlgError:
                step = np.linalg.solve(H_reg, g)
            eigvals = np.linalg.eigvalsh(H_reg)
            cond = float(abs(eigvals[-1]) / (abs(eigvals[0]) + 1e-16))
            if np.any(~np.isfinite(step)):
                raise np.linalg.LinAlgError("non-finite step")
            return step, lam_used, cond
        except np.linalg.LinAlgError:
            lam_used = 10.0 * (lam_used if lam_used > 0 else 1e-6)
    return g.copy(), lam_used, float("inf")


def run_newton(
    objective: Objective,
    x0: np.ndarray,
    settings: Optional[NewtonSettings] = None,
    bounds: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    settings = settings or NewtonSettings()
    x = np.asarray(x0, dtype=float).ravel().copy()
    bounds = default_bounds(x) if bounds is None else np.asarray(bounds, dtype=float)
    lo, hi = bounds[:, 0], bounds[:, 1]

    best_x = x.copy()
    best_loss = float(objective.evaluate(x))
    n_evals = 1
    prev_x = None
    no_improve = 0
    uni_stag = 0
    prev_best = best_loss
    current_lambda = settings.lm_lambda

    history = {
        "iteration": [],
        "loss": [],
        "max_grad": [],
        "param_disp": [],
        "alpha": [],
        "lm_lambda": [],
    }
    stop_reason = ""
    converged = False
    n_done = 0
    losses_hist: list[float] = []

    for it in range(1, settings.max_iter + 1):
        f_cur = float(objective.evaluate(x))
        n_evals += 1
        g, ne_g = _gradient_fd(objective, x, f_cur, settings)
        n_evals += ne_g
        H, ne_h = _hessian_fd(objective, x, f_cur, settings)
        n_evals += ne_h
        max_grad = float(np.max(np.abs(g)))

        step_vec, lam_used, _cond = _newton_step(H, g, current_lambda)
        current_lambda = lam_used
        g_dot_step = float(np.dot(g, step_vec))

        alpha = 1.0
        accepted = False
        new_x = x.copy()
        new_loss = f_cur
        for _ in range(settings.max_backtracks):
            trial = np.clip(x - alpha * step_vec, lo, hi)
            trial_loss = float(objective.evaluate(trial))
            n_evals += 1
            enough = trial_loss <= f_cur - settings.armijo_c * alpha * max(g_dot_step, 0.0)
            if enough or trial_loss < f_cur:
                new_x, new_loss, accepted = trial, trial_loss, True
                break
            alpha *= settings.backtrack_rho

        if not accepted:
            alpha = 1e-3
            new_x = np.clip(x - alpha * g, lo, hi)
            new_loss = float(objective.evaluate(new_x))
            n_evals += 1
            current_lambda = max(current_lambda * 10.0, 1e-2)
        else:
            current_lambda = max(current_lambda * 0.5, 1e-12)

        param_disp = (
            float(np.linalg.norm(new_x - prev_x)) if prev_x is not None else float("inf")
        )

        if new_loss < best_loss - 1e-15:
            best_loss = new_loss
            best_x = new_x.copy()
            no_improve = 0
        else:
            no_improve += 1

        losses_hist.append(f_cur)
        loss_stag_50 = False
        if it > 50:
            recent = losses_hist[-50:]
            loss_stag_50 = all(abs(v - f_cur) < settings.stag_loss_tol for v in recent)

        uni_stag = update_stagnation_counter(
            uni_stag, prev_best, best_loss, tol=settings.stagnation_tol, relative=True
        )
        prev_best = best_loss

        history["iteration"].append(it)
        history["loss"].append(best_loss)
        history["max_grad"].append(max_grad)
        history["param_disp"].append(param_disp if prev_x is not None else 0.0)
        history["alpha"].append(alpha)
        history["lm_lambda"].append(lam_used)
        n_done = it

        if settings.simple_stop:
            cfg = UnifiedStopConfig(
                loss_target=settings.loss_target,
                stagnation_window=settings.stagnation_window,
                stagnation_tol=settings.stagnation_tol,
            )
            hit, reasons, _ = check_unified_stop(
                it=it,
                max_iter=settings.max_iter,
                best_loss=best_loss,
                stagnation_counter=uni_stag,
                cfg=cfg,
            )
        else:
            hit, reasons = check_newton_native_stop(
                it=it,
                max_grad=max_grad,
                param_disp=param_disp if prev_x is not None else float("inf"),
                loss_stag_50=loss_stag_50,
                no_improve_count=no_improve,
                settings=settings,
            )

        prev_x = new_x.copy()
        x = new_x
        if hit:
            converged = True
            stop_reason = " | ".join(reasons)
            break

    return {
        "method": "Newton",
        "best_x": best_x,
        "best_loss": best_loss,
        "iterations": n_done,
        "evaluations": n_evals,
        "converged": converged,
        "stop_reason": stop_reason,
        "history": history,
    }
