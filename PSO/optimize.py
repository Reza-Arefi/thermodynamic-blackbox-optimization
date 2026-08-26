"""
Particle Swarm Optimization on a black-box objective (serial particle evals).
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
from common.params import default_bounds
from common.stopping_unified import (
    UnifiedStopConfig,
    check_unified_stop,
    update_stagnation_counter,
)

from .settings import PSOSettings
from .stopping import check_pso_native_stop


def run_pso(
    objective: Objective,
    x0: np.ndarray,
    settings: Optional[PSOSettings] = None,
    bounds: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    settings = settings or PSOSettings()
    if settings.seed is not None:
        np.random.seed(settings.seed)

    x0 = np.asarray(x0, dtype=float).ravel()
    n = x0.size
    bounds = default_bounds(x0) if bounds is None else np.asarray(bounds, dtype=float)
    lo, hi = bounds[:, 0], bounds[:, 1]
    span = hi - lo
    mean_range = float(np.mean(span)) if np.mean(span) > 0 else 1.0

    n_part = settings.num_particles or int(min(5 * n, 50))
    n_part = max(n_part, 2)

    pos = np.random.uniform(lo, hi, size=(n_part, n))
    pos[0] = x0
    vel = np.random.uniform(-0.1 * span, 0.1 * span, size=(n_part, n))

    losses = np.array([float(objective.evaluate(p)) for p in pos])
    n_evals = n_part
    pbest_pos = pos.copy()
    pbest = losses.copy()
    g_idx = int(np.argmin(losses))
    gbest = pos[g_idx].copy()
    gbest_loss = float(losses[g_idx])

    prev_best = gbest_loss
    loss_stag = 0
    ext_stag = 0
    uni_stag = 0

    history = {
        "iteration": [0],
        "loss": [gbest_loss],
        "diversity": [],
        "norm_contraction": [],
    }
    stop_reason = ""
    converged = False
    n_done = 0

    def metrics(p: np.ndarray):
        div = float(np.mean(np.std(p, axis=0)))
        return div, div / mean_range

    div0, nc0 = metrics(pos)
    history["diversity"].append(div0)
    history["norm_contraction"].append(nc0)

    for it in range(1, settings.max_iter + 1):
        w = settings.w_max - (settings.w_max - settings.w_min) * (it / settings.max_iter)
        r1 = np.random.rand(n_part, n)
        r2 = np.random.rand(n_part, n)
        vel = (
            w * vel
            + settings.c1 * r1 * (pbest_pos - pos)
            + settings.c2 * r2 * (gbest - pos)
        )
        pos = np.clip(pos + vel, lo, hi)

        losses = np.array([float(objective.evaluate(p)) for p in pos])
        n_evals += n_part

        improved = losses < pbest
        pbest_pos[improved] = pos[improved]
        pbest[improved] = losses[improved]
        g_idx = int(np.argmin(pbest))
        if pbest[g_idx] < gbest_loss:
            gbest_loss = float(pbest[g_idx])
            gbest = pbest_pos[g_idx].copy()

        div, nc = metrics(pos)
        if abs(gbest_loss - prev_best) < settings.stag_loss_tol:
            loss_stag += 1
            ext_stag += 1
        else:
            loss_stag = 0
            ext_stag = 0

        uni_stag = update_stagnation_counter(
            uni_stag, prev_best, gbest_loss, tol=settings.stagnation_tol, relative=True
        )
        prev_best = gbest_loss

        history["iteration"].append(it)
        history["loss"].append(gbest_loss)
        history["diversity"].append(div)
        history["norm_contraction"].append(nc)
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
                best_loss=gbest_loss,
                stagnation_counter=uni_stag,
                cfg=cfg,
            )
        else:
            hit, reasons = check_pso_native_stop(
                it=it,
                normalized_contraction=nc,
                diversity=div,
                loss_stag_count=loss_stag,
                extended_stag_count=ext_stag,
                settings=settings,
            )

        if hit:
            converged = True
            stop_reason = " | ".join(reasons)
            break

    return {
        "method": "PSO",
        "best_x": gbest,
        "best_loss": gbest_loss,
        "iterations": n_done,
        "evaluations": n_evals,
        "num_particles": n_part,
        "converged": converged,
        "stop_reason": stop_reason,
        "history": history,
    }
