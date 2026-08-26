"""Native PSO stopping criteria."""
from __future__ import annotations

from typing import List, Tuple

from .settings import PSOSettings


def check_pso_native_stop(
    *,
    it: int,
    normalized_contraction: float,
    diversity: float,
    loss_stag_count: int,
    extended_stag_count: int,
    settings: PSOSettings,
) -> Tuple[bool, List[str]]:
    """
    Native criteria (OR), require it >= 5:
      1. normalized_contraction < 1e-5
      2. swarm_diversity < 1e-5
      3. loss flat 50 iters AND normalized_contraction < 1e-4
      4. extended stag 10 iters AND normalized_contraction < 0.05
      5. max_iter
    """
    reasons: List[str] = []
    if it >= settings.min_iter_gate:
        if normalized_contraction < settings.norm_contraction_tol:
            reasons.append(
                f"norm_contraction={normalized_contraction:.3e} < {settings.norm_contraction_tol:.3e}"
            )
        if diversity < settings.diversity_tol:
            reasons.append(f"diversity={diversity:.3e} < {settings.diversity_tol:.3e}")
        if (
            loss_stag_count >= settings.stag_window
            and normalized_contraction < settings.stag_contraction_tol
        ):
            reasons.append(
                f"loss stagnation {loss_stag_count} + norm_contraction={normalized_contraction:.3e}"
            )
        if (
            extended_stag_count >= settings.extended_stag_window
            and normalized_contraction < settings.extended_contraction_tol
        ):
            reasons.append(
                f"extended stagnation {extended_stag_count} + "
                f"norm_contraction={normalized_contraction:.3e}"
            )
    if it >= settings.max_iter:
        reasons.append(f"max_iter={settings.max_iter}")
    return bool(reasons), reasons
