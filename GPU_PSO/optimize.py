"""
GPU-PSO: PSO updates on host; particle losses via black-box `evaluate_many`.

No Flash / pvtXpert coupling — inject any BatchObjective.
Stopping: Stop A/B/C (native) or unified L/S/C (`simple_stop`).
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

from .settings import GPUPSOSettings


def run_gpu_pso(
    objective: BatchObjective | Objective,
    x0: np.ndarray,
    settings: Optional[GPUPSOSettings] = None,
    bounds: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    settings = settings or GPUPSOSettings()
    obj = as_batch(objective)
    if settings.seed is not None:
        np.random.seed(settings.seed)

    x0 = np.asarray(x0, dtype=float).ravel()
    n = x0.size
    bounds = default_bounds(x0) if bounds is None else np.asarray(bounds, dtype=float)
    lo, hi = bounds[:, 0], bounds[:, 1]
    span = hi - lo

    n_part = settings.num_particles or int(min(5 * n, 50))
    n_part = max(n_part, 2)

    pos = np.random.uniform(lo, hi, size=(n_part, n))
    pos[0] = x0
    vel = np.random.uniform(-0.1 * span, 0.1 * span, size=(n_part, n))

    losses = np.asarray(obj.evaluate_many([pos[i] for i in range(n_part)]), dtype=float)
    n_evals = n_part
    pbest_pos = pos.copy()
    pbest = losses.copy()
    g_idx = int(np.argmin(losses))
    gbest = pos[g_idx].copy()
    gbest_loss = float(losses[g_idx])

    def diversity(p: np.ndarray) -> float:
        return float(np.mean(np.std(p, axis=0)))

    init_div = max(diversity(pos), 1e-30)
    stag = 0
    prev_best = gbest_loss
    history = {"iteration": [0], "loss": [gbest_loss], "contraction_ratio": [1.0]}
    stop_reason = ""
    converged = False
    scientific = False
    n_done = 0

    for it in range(1, settings.max_iter + 1):
        w = settings.w_max - (settings.w_max - settings.w_min) * (it / settings.max_iter)
        r1 = np.random.rand(n_part, n)
        r2 = np.random.rand(n_part, n)
        vel = w * vel + settings.c1 * r1 * (pbest_pos - pos) + settings.c2 * r2 * (gbest - pos)
        pos = np.clip(pos + vel, lo, hi)

        losses = np.asarray(obj.evaluate_many([pos[i] for i in range(n_part)]), dtype=float)
        n_evals += n_part
        improved = losses < pbest
        pbest_pos[improved] = pos[improved]
        pbest[improved] = losses[improved]
        g_idx = int(np.argmin(pbest))
        new_best = float(pbest[g_idx])
        if new_best < gbest_loss:
            gbest_loss = new_best
            gbest = pbest_pos[g_idx].copy()

        cr = diversity(pos) / init_div
        imp = relative_improvement(prev_best, gbest_loss)
        tol = settings.stop.stagnation_tol if settings.stop.simple_stop else settings.stop.rel_improve_tol
        stag = 0 if imp > tol else stag + 1
        prev_best = gbest_loss

        history["iteration"].append(it)
        history["loss"].append(gbest_loss)
        history["contraction_ratio"].append(cr)
        n_done = it

        hit, reasons, sci = check_gpu_stopping(
            it=it,
            max_iter=settings.max_iter,
            stagnation_counter=stag,
            contraction_ratio=cr,
            best_loss=gbest_loss,
            settings=settings.stop,
        )
        if hit:
            converged = True
            scientific = sci
            stop_reason = " | ".join(reasons)
            break

    return {
        "method": "GPU-PSO",
        "best_x": gbest,
        "best_loss": gbest_loss,
        "iterations": n_done,
        "evaluations": n_evals,
        "num_particles": n_part,
        "converged": converged,
        "scientific_convergence": scientific,
        "stop_reason": stop_reason,
        "history": history,
    }
