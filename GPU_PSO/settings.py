"""Settings for GPU-PSO (from GPU-pso/config.py + CLI defaults)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import sys
from pathlib import Path

_GIT = Path(__file__).resolve().parents[1]
if str(_GIT) not in sys.path:
    sys.path.insert(0, str(_GIT))

from common.gpu_stopping import GPUStopSettings


@dataclass
class GPUPSOSettings:
    max_iter: int = 2000
    num_particles: Optional[int] = None  # min(5*d, 50)
    w_max: float = 1.0
    w_min: float = 0.1
    c1: float = 1.5
    c2: float = 1.5
    seed: Optional[int] = None
    stop: GPUStopSettings = field(default_factory=GPUStopSettings)
