"""GPU-DE (DE/rand/1/bin) — black-box batch objective."""

from .optimize import run_gpu_de
from .settings import GPUDESettings

__all__ = ["GPUDESettings", "run_gpu_de"]
