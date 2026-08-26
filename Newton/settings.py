"""Default settings for damped Newton / LM (matches newton/newton_rPar.py)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NewtonSettings:
    max_iter: int = 2000
    lm_lambda: float = 1e-3
    armijo_c: float = 1e-4
    backtrack_rho: float = 0.5
    max_backtracks: int = 10
    fd_relative: float = 0.01
    fd_floor: float = 1e-6
    # Native stopping (aligned with GD / NM / PSO)
    grad_tol: float = 5e-5
    param_tol: float = 1e-5
    stag_window: int = 50
    stag_loss_tol: float = 1e-8
    stag_grad_tol: float = 1e-4
    extended_stag_window: int = 10
    extended_grad_tol: float = 1e-3
    # Unified table protocol
    simple_stop: bool = False
    loss_target: float = 1.0
    stagnation_window: int = 50
    stagnation_tol: float = 1e-8
