"""GPU-PSO algorithm package (black-box batch objective)."""

from .optimize import run_gpu_pso
from .settings import GPUPSOSettings

__all__ = ["GPUPSOSettings", "run_gpu_pso"]
