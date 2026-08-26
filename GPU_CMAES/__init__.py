"""GPU-CMA-ES — black-box batch objective."""

from .optimize import run_gpu_cmaes
from .settings import GPUCMAESSettings

__all__ = ["GPUCMAESSettings", "run_gpu_cmaes"]
