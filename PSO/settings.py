"""Default settings for CPU Particle Swarm Optimization."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class PSOSettings:
    max_iter: int = 2000
    num_particles: Optional[int] = None  # default min(5*d, 50)
    w_max: float = 1.0
    w_min: float = 0.1
    c1: float = 1.5
    c2: float = 1.5
    # Native stopping
    norm_contraction_tol: float = 1e-5
    diversity_tol: float = 1e-5
    stag_window: int = 50
    stag_loss_tol: float = 1e-8
    stag_contraction_tol: float = 1e-4
    extended_stag_window: int = 10
    extended_contraction_tol: float = 0.05
    min_iter_gate: int = 5
    # Unified table protocol
    simple_stop: bool = False
    loss_target: float = 1.0
    stagnation_window: int = 50
    stagnation_tol: float = 1e-8
    seed: Optional[int] = None
