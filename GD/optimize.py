"""
Gradient Descent on a black-box objective.

Flash / curve evaluation is injected via `objective.evaluate(x)`.
Finite-difference gradients + Adagrad-style adaptive step (AdG lineage).
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
from common.stopping_unified import (
    UnifiedStopConfig,
    check_unified_stop,
    update_stagnation_counter,
)

from .settings import GDSettings
from .stopping import check_gd_native_stop


def _fd_gradient(
    objective: Objective, x: np.ndarray, eps: float
) -> Tuple[np.ndarray, float, int]:
    x = np.asarray(x, dtype=float).ravel()
    f0 = float(objective.evaluate(x))
    g = np.zeros_like(x)
    for i in range(x.size):
        xp = x.copy()
        xp[i] += eps
        g[i] = (float(objective.evaluate(xp)) - f0) / eps
    return g, f0, 1 + x.size


def run_gd(
    objective: Objective,
    x0: np.ndarray,
    settings: Optional[GDSettings] = None,
) -> Dict[str, Any]:
    settings = settings or GDSettings()
    x = np.asarray(x0, dtype=float).ravel().copy()
    n = x.size
    cum_sq = np.zeros(n, dtype=float)

    best_x = x.copy()
    best_loss = float(objective.evaluate(x))
    n_evals = 1
    prev_loss = best_loss
    prev_best = best_loss
    loss_stag = 0
    uni_stag = 0

    history = {"iteration": [], "loss": [], "max_grad": [], "param_disp": []}
    stop_reason = ""
    converged = False
    n_done = 0

    for it in range(1, settings.max_iter + 1):
        g, f_cur, ne = _fd_gradient(objective, x, settings.fd_eps)
        n_evals += ne
        max_grad = float(np.max(np.abs(g)))
        cum_sq += g ** 2
        step = settings.learning_rate * g / (np.sqrt(cum_sq) + 1e-8)
        x_new = x - step
        f_new = float(objective.evaluate(x_new))
        n_evals += 1
        param_disp = float(np.linalg.norm(x_new - x))
        abs_dloss = abs(f_new - prev_loss)

        if f_new < best_loss:
            best_loss = f_new
            best_x = x_new.copy()

        if abs(f_new - prev_loss) < settings.stag_loss_tol:
            loss_stag += 1
        else:
            loss_stag = 0

        uni_stag = update_stagnation_counter(
            uni_stag,
            prev_best,
            best_loss,
            tol=settings.stagnation_tol,
            relative=True,
        )
        prev_best = best_loss

        history["iteration"].append(it)
        history["loss"].append(best_loss)
        history["max_grad"].append(max_grad)
        history["param_disp"].append(param_disp)
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
            hit, reasons = check_gd_native_stop(
                it=it,
                max_grad=max_grad,
                param_disp=param_disp,
                loss_stag_count=loss_stag,
                abs_dloss=abs_dloss,
                settings=settings,
            )

        x = x_new
        prev_loss = f_new
        if hit:
            converged = True
            stop_reason = " | ".join(reasons)
            break

    return {
        "method": "GD",
        "best_x": best_x,
        "best_loss": best_loss,
        "iterations": n_done,
        "evaluations": n_evals,
        "converged": converged,
        "stop_reason": stop_reason,
        "history": history,
    }
