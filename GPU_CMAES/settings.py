"""Settings for GPU-CMA-ES (from GPU-cmaes/config.py)."""
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
class GPUCMAESSettings:
    max_iter: int = 2000
    pop_size: Optional[int] = None  # default min(5*d, 50); or Hansen 4+floor(3 ln n)
    use_hansen_pop: bool = False
    sigma0: float = 0.3  # relative to bound half-width
    seed: Optional[int] = None
    stop: GPUStopSettings = field(default_factory=GPUStopSettings)
