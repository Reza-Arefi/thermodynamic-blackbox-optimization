"""Default settings for Gradient Descent (matches paper / table usage)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GDSettings:
    max_iter: int = 2000
    learning_rate: float = 0.01
    fd_eps: float = 1e-4
    # Native scientific stopping (OR logic)
    grad_tol: float = 5e-5          # ||∇L||_∞
    param_tol: float = 1e-5         # ||Δk||_2
    stag_window: int = 50
    stag_loss_tol: float = 1e-8
    stag_grad_tol: float = 1e-4
    fallback_loss_tol: float = 1e-6
    # Unified table protocol
    simple_stop: bool = False
    loss_target: float = 1.0
    stagnation_window: int = 50
    stagnation_tol: float = 1e-8
