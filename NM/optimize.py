"""
Nelder–Mead simplex method on a black-box objective.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

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

from .settings import NMSettings
from .stopping import check_nm_native_stop


def run_nm(
    objective: Objective,
    x0: np.ndarray,
    settings: Optional[NMSettings] = None,
) -> Dict[str, Any]:
    settings = settings or NMSettings()
    x0 = np.asarray(x0, dtype=float).ravel()
    n = x0.size

    # Initial simplex: x0 and x0 + step*e_i
    simplex = np.vstack([x0, x0 + np.eye(n) * settings.step])
    fvals = np.array([float(objective.evaluate(p)) for p in simplex])
    n_evals = n + 1

    best_loss = float(np.min(fvals))
    best_x = simplex[int(np.argmin(fvals))].copy()
    prev_best = best_loss
    loss_stag = 0
    uni_stag = 0
    complete_stag = 0
    prev_span = None

    history = {"iteration": [], "loss": [], "contraction": [], "max_dist": []}
    stop_reason = ""
    converged = False
    n_done = 0

    for it in range(1, settings.max_iter + 1):
        order = np.argsort(fvals)
        simplex = simplex[order]
        fvals = fvals[order]
        best = simplex[0]
        worst = simplex[-1]
        centroid = np.mean(simplex[:-1], axis=0)

        # Reflect
        xr = centroid + settings.alpha * (centroid - worst)
        fr = float(objective.evaluate(xr))
        n_evals += 1

        if fvals[0] <= fr < fvals[-2]:
            simplex[-1], fvals[-1] = xr, fr
        elif fr < fvals[0]:
            # Expand
            xe = centroid + settings.gamma * (xr - centroid)
            fe = float(objective.evaluate(xe))
            n_evals += 1
            if fe < fr:
                simplex[-1], fvals[-1] = xe, fe
            else:
                simplex[-1], fvals[-1] = xr, fr
        else:
            # Contract
            if fr < fvals[-1]:
                xc = centroid + settings.rho * (xr - centroid)
            else:
                xc = centroid + settings.rho * (worst - centroid)
            fc = float(objective.evaluate(xc))
            n_evals += 1
            if fc < min(fr, fvals[-1]):
                simplex[-1], fvals[-1] = xc, fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = best + settings.sigma * (simplex[i] - best)
                    fvals[i] = float(objective.evaluate(simplex[i]))
                    n_evals += 1

        order = np.argsort(fvals)
        simplex = simplex[order]
        fvals = fvals[order]
        best_x = simplex[0].copy()
        best_loss = float(fvals[0])

        dists = np.linalg.norm(simplex - best_x, axis=1)
        max_dist = float(np.max(dists))
        contraction = max_dist  # same spirit as paper simplex_contraction

        if abs(best_loss - prev_best) < settings.stag_loss_tol:
            loss_stag += 1
        else:
            loss_stag = 0

        if (
            abs(best_loss - prev_best) < settings.complete_stag_loss_tol
            and prev_span is not None
            and abs(contraction - prev_span) < 1e-15
        ):
            complete_stag += 1
        else:
            complete_stag = 0
        prev_span = contraction

        uni_stag = update_stagnation_counter(
            uni_stag, prev_best, best_loss, tol=settings.stagnation_tol, relative=True
        )
        prev_best = best_loss

        history["iteration"].append(it)
        history["loss"].append(best_loss)
        history["contraction"].append(contraction)
        history["max_dist"].append(max_dist)
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
            hit, reasons = check_nm_native_stop(
                it=it,
                simplex_contraction=contraction,
                max_dist=max_dist,
                loss_stag_count=loss_stag,
                complete_stag_count=complete_stag,
                settings=settings,
            )

        if hit:
            converged = True
            stop_reason = " | ".join(reasons)
            break

    return {
        "method": "NM",
        "best_x": best_x,
        "best_loss": best_loss,
        "iterations": n_done,
        "evaluations": n_evals,
        "converged": converged,
        "stop_reason": stop_reason,
        "history": history,
    }
