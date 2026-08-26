"""Particle Swarm Optimization — optimization only (black-box objective)."""

from .optimize import run_pso
from .settings import PSOSettings
from .stopping import check_pso_native_stop

__all__ = ["PSOSettings", "run_pso", "check_pso_native_stop"]
