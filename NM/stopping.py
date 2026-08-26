"""Native Nelder–Mead stopping criteria."""
from __future__ import annotations

from typing import List, Tuple

from .settings import NMSettings


def check_nm_native_stop(
    *,
    it: int,
    simplex_contraction: float,
    max_dist: float,
    loss_stag_count: int,
    complete_stag_count: int,
    settings: NMSettings,
) -> Tuple[bool, List[str]]:
    """
    Native criteria (OR):
      1. simplex_contraction < 5e-5
      2. max_dist to best < 1e-5
      3. loss flat 50 iters (|ΔL|<1e-8) AND contraction < 1e-4
      4. complete stag 10 iters (|ΔL|<1e-10, geometry unchanged)
      5. max_iter
    """
    reasons: List[str] = []
    if simplex_contraction < settings.contraction_tol:
        reasons.append(
            f"simplex_contraction={simplex_contraction:.3e} < {settings.contraction_tol:.3e}"
        )
    if max_dist < settings.max_dist_tol:
        reasons.append(f"max_dist={max_dist:.3e} < {settings.max_dist_tol:.3e}")
    if (
        loss_stag_count >= settings.stag_window
        and simplex_contraction < settings.stag_contraction_tol
    ):
        reasons.append(
            f"loss stagnation {loss_stag_count} + contraction={simplex_contraction:.3e}"
        )
    if complete_stag_count >= settings.complete_stag_window:
        reasons.append(
            f"complete stagnation {complete_stag_count} iters"
        )
    if it >= settings.max_iter:
        reasons.append(f"max_iter={settings.max_iter}")
    return bool(reasons), reasons
