"""Settings for GPU-DE (from GPU-de/config.py)."""
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
class GPUDESettings:
    max_iter: int = 2000
    pop_size: Optional[int] = None  # min(5*d, 50), at least 4
    F: float = 0.5
    CR: float = 0.9
    seed: Optional[int] = None
    stop: GPUStopSettings = field(default_factory=GPUStopSettings)
