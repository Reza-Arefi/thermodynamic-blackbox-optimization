"""Native GD stopping criteria (paper / STOPPING_CRITERIA_SUMMARY.md)."""
from __future__ import annotations

from typing import List, Tuple

from .settings import GDSettings


def check_gd_native_stop(
    *,
    it: int,
    max_grad: float,
    param_disp: float,
    loss_stag_count: int,
    abs_dloss: float,
    settings: GDSettings,
) -> Tuple[bool, List[str]]:
    """
    Native criteria (OR):
      1. ||∇L||_∞ < 5e-5
      2. ||Δk||_2 < 1e-5
      3. |ΔL| < 1e-8 for 50 iters AND ||∇L||_∞ < 1e-4
      4. fallback |ΔL| < 1e-6 (after min iters)
      5. it >= max_iter
    """
    reasons: List[str] = []
    if max_grad < settings.grad_tol:
        reasons.append(f"max|∇L|={max_grad:.3e} < {settings.grad_tol:.3e}")
    if param_disp < settings.param_tol:
        reasons.append(f"||Δk||_2={param_disp:.3e} < {settings.param_tol:.3e}")
    if (
        loss_stag_count >= settings.stag_window
        and max_grad < settings.stag_grad_tol
    ):
        reasons.append(
            f"loss stagnation {loss_stag_count} iters + |∇L|_∞={max_grad:.3e}"
        )
    if it >= 5 and abs_dloss < settings.fallback_loss_tol and abs_dloss > 0:
        reasons.append(f"fallback |ΔL|={abs_dloss:.3e} < {settings.fallback_loss_tol:.3e}")
    if it >= settings.max_iter:
        reasons.append(f"max_iter={settings.max_iter}")
    return bool(reasons), reasons
