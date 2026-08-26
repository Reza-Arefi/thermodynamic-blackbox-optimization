"""Parameter-index helpers used by all methods."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def select_random_parameters(num_params: int, dim: int = 210, seed: Optional[int] = None) -> List[int]:
    rng = np.random.default_rng(seed)
    num_params = int(num_params)
    if num_params < 1 or num_params > dim:
        raise ValueError(f"num_params must be in 1..{dim}, got {num_params}")
    return sorted(rng.choice(dim, size=num_params, replace=False).tolist())


def default_bounds(
    x0: np.ndarray,
    half_width: float = 0.5,
) -> np.ndarray:
    """Return (n, 2) array of [lo, hi] around each free coordinate of x0."""
    x0 = np.asarray(x0, dtype=float).ravel()
    lo = x0 - half_width
    hi = x0 + half_width
    return np.column_stack([lo, hi])


def to_full_vector(
    free: np.ndarray,
    param_indices: Sequence[int],
    base: np.ndarray,
) -> np.ndarray:
    """Embed free parameters into a full base vector (e.g. length-210 Kij pack)."""
    out = np.asarray(base, dtype=float).copy()
    free = np.asarray(free, dtype=float).ravel()
    for i, idx in enumerate(param_indices):
        out[int(idx)] = free[i]
    return out


def bounds_dict(
    param_indices: Sequence[int],
    base: np.ndarray,
    half_width: float = 0.5,
) -> Dict[int, Tuple[float, float]]:
    base = np.asarray(base, dtype=float)
    return {
        int(idx): (float(base[idx] - half_width), float(base[idx] + half_width))
        for idx in param_indices
    }
