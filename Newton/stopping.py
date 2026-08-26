"""Native Newton stopping criteria (see newton/STOPPING_CRITERIA.md)."""
from __future__ import annotations

from typing import List, Tuple

from .settings import NewtonSettings


def check_newton_native_stop(
    *,
    it: int,
    max_grad: float,
    param_disp: float,
    loss_stag_50: bool,
    no_improve_count: int,
    settings: NewtonSettings,
) -> Tuple[bool, List[str]]:
    """
    Native criteria (OR):
      1. max|∇L|_∞ < 5e-5
      2. ||Δx||_2 < 1e-5
      3. loss flat 50 iters (|ΔL|<1e-8) AND max|∇L| < 1e-4
      4. no improve for 10 iters AND max|∇L| < 1e-3
      5. max_iter
    """
    reasons: List[str] = []
    if max_grad < settings.grad_tol:
        reasons.append(f"max|∇L|={max_grad:.3e} < {settings.grad_tol:.3e}")
    if it > 1 and param_disp < settings.param_tol:
        reasons.append(f"||Δx||_2={param_disp:.3e} < {settings.param_tol:.3e}")
    if loss_stag_50 and max_grad < settings.stag_grad_tol:
        reasons.append("loss_stagnation + small_gradient")
    if (
        it > settings.extended_stag_window
        and no_improve_count >= settings.extended_stag_window
        and max_grad < settings.extended_grad_tol
    ):
        reasons.append(
            f"extended_stagnation ({no_improve_count} iters, max_grad={max_grad:.3e})"
        )
    if it >= settings.max_iter:
        reasons.append(f"max_iter={settings.max_iter}")
    return bool(reasons), reasons
