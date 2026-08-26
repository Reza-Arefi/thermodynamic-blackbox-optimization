"""
Shared Stop A / B / C / L for GPU-PSO, GPU-DE, GPU-CMA-ES.

Copied conceptually from GPU-*/run.py + config.py (paper implementations).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class GPUStopSettings:
    # Stop A
    stop_a_window: int = 10
    contraction_ratio_tol: float = 0.10
    # Stop B
    stop_b_window: int = 50
    # relative improvement tolerance used to count "flat" iters
    rel_improve_tol: float = 1e-6
    loss_stagnation_window: int = 10  # min-iter gate max(5, window)
    # Unified
    simple_stop: bool = False
    loss_target: float = 1.0
    stagnation_window: Optional[int] = None  # defaults to stop_b_window
    stagnation_tol: float = 1e-8


def relative_improvement(best_old: float, best_new: float) -> float:
    return float(abs(best_old - best_new) / max(1.0, abs(best_old)))


def check_stop_a(
    *, stagnation_counter: int, contraction_ratio: float, short_window: int, contraction_tol: float
) -> Tuple[bool, str]:
    if stagnation_counter >= int(short_window) and contraction_ratio < float(contraction_tol):
        return True, (
            f"Stop A (normal convergence): rel_imp flat {stagnation_counter} iters "
            f"(≥{short_window}) AND contraction_ratio={contraction_ratio:.4f} < {contraction_tol}"
        )
    return False, ""


def check_stop_b(*, stagnation_counter: int, long_window: int) -> Tuple[bool, str]:
    if stagnation_counter >= int(long_window):
        return True, (
            f"Stop B (long stagnation): rel_imp flat {stagnation_counter} iters (≥{long_window})"
        )
    return False, ""


def check_stop_c(*, it: int, max_iter: int) -> Tuple[bool, str]:
    if it >= int(max_iter):
        return True, f"Stop C (max_iter={max_iter})"
    return False, ""


def check_stop_l(*, best_loss: float, loss_target: float) -> Tuple[bool, str]:
    if best_loss < float(loss_target):
        return True, f"Stop L (loss target): best_loss={best_loss:.6g} < {loss_target}"
    return False, ""


def check_gpu_stopping(
    *,
    it: int,
    max_iter: int,
    stagnation_counter: int,
    contraction_ratio: float,
    best_loss: float,
    settings: GPUStopSettings,
) -> Tuple[bool, List[str], bool]:
    """
    Returns (should_stop, reasons, scientific_convergence).

    simple_stop: Stop L OR Stop S(=B with stagnation_window) OR Stop C; disables A.
    else: Stop A / B / C with min-iter gate for A/B.
    """
    reasons: List[str] = []
    scientific = False
    long_window = settings.stagnation_window or settings.stop_b_window

    if settings.simple_stop:
        hit_l, reason_l = check_stop_l(best_loss=best_loss, loss_target=settings.loss_target)
        if hit_l:
            scientific = True
            reasons.append(reason_l)
        min_it = max(5, int(long_window))
        if it >= min_it:
            hit_s, reason_s = check_stop_b(
                stagnation_counter=stagnation_counter, long_window=long_window
            )
            if hit_s:
                scientific = True
                reasons.append(reason_s)
        hit_c, reason_c = check_stop_c(it=it, max_iter=max_iter)
        if hit_c:
            reasons.append(reason_c)
        return bool(reasons), reasons, scientific

    min_it = max(5, int(settings.loss_stagnation_window))
    if it >= min_it:
        hit_a, reason_a = check_stop_a(
            stagnation_counter=stagnation_counter,
            contraction_ratio=contraction_ratio,
            short_window=settings.stop_a_window,
            contraction_tol=settings.contraction_ratio_tol,
        )
        if hit_a:
            scientific = True
            reasons.append(reason_a)
        hit_b, reason_b = check_stop_b(
            stagnation_counter=stagnation_counter, long_window=settings.stop_b_window
        )
        if hit_b:
            scientific = True
            reasons.append(reason_b)
    hit_c, reason_c = check_stop_c(it=it, max_iter=max_iter)
    if hit_c:
        reasons.append(reason_c)
    return bool(reasons), reasons, scientific
