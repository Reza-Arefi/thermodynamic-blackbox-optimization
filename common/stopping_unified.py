"""
Unified table stopping protocol (Stop L / S / C).

Used by all six methods when `--simple-stop` is enabled in the paper tables:
  L: best_loss < loss_target          (default 1.0)
  S: relative improvement flat for `stagnation_window` iters (default 50)
  C: iteration >= max_iter

Relative improvement: |ΔL| / max(1, |L_old|)  (GPU methods),
or absolute |ΔL| < stagnation_tol for sequential simple-stop (table runs
used the same CLI flags; see each method's stopping module).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class UnifiedStopConfig:
    loss_target: float = 1.0
    stagnation_window: int = 50
    stagnation_tol: float = 1e-8
    use_relative_improvement: bool = True


def relative_improvement(best_old: float, best_new: float) -> float:
    return float(abs(best_old - best_new) / max(1.0, abs(best_old)))


def check_unified_stop(
    *,
    it: int,
    max_iter: int,
    best_loss: float,
    stagnation_counter: int,
    cfg: UnifiedStopConfig,
) -> Tuple[bool, List[str], bool]:
    """
    Returns (should_stop, reasons, scientific_convergence).

    scientific_convergence is True for Stop L or S (not budget-only Stop C).
    """
    reasons: List[str] = []
    scientific = False

    if best_loss < cfg.loss_target:
        scientific = True
        reasons.append(
            f"Stop L (loss target): best_loss={best_loss:.6g} < {cfg.loss_target}"
        )

    min_it = max(5, int(cfg.stagnation_window))
    if it >= min_it and stagnation_counter >= int(cfg.stagnation_window):
        scientific = True
        reasons.append(
            f"Stop S (stagnation): flat {stagnation_counter} iters "
            f"(≥{cfg.stagnation_window})"
        )

    if it >= int(max_iter):
        reasons.append(f"Stop C (max_iter={max_iter})")

    return bool(reasons), reasons, scientific


def update_stagnation_counter(
    counter: int,
    best_old: float,
    best_new: float,
    *,
    tol: float = 1e-8,
    relative: bool = True,
) -> int:
    """Increment flat-counter when improvement is below tolerance."""
    if relative:
        imp = relative_improvement(best_old, best_new)
        improved = imp > tol
    else:
        improved = abs(best_old - best_new) > tol
    return 0 if improved else counter + 1
