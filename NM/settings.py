"""Default settings for Nelder–Mead."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NMSettings:
    max_iter: int = 5000
    step: float = 0.05
    alpha: float = 1.0
    gamma: float = 2.0
    rho: float = 0.5
    sigma: float = 0.5
    # Native stopping
    contraction_tol: float = 5e-5
    max_dist_tol: float = 1e-5
    stag_window: int = 50
    stag_loss_tol: float = 1e-8
    stag_contraction_tol: float = 1e-4
    complete_stag_window: int = 10
    complete_stag_loss_tol: float = 1e-10
    # Unified table protocol
    simple_stop: bool = False
    loss_target: float = 1.0
    stagnation_window: int = 50
    stagnation_tol: float = 1e-8
