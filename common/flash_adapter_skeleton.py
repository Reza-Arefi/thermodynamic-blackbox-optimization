"""
Example: plug your Flash/curve evaluator behind the black-box API.

Do NOT put proprietary Flash binaries in this repository.
Implement `evaluate` (and optionally `evaluate_many`) in your private
environment, then call the optimizers from Git/.
"""
from __future__ import annotations

import numpy as np


class FlashCurveObjective:
    """
    Skeleton only — replace the body with:
      pack free params → run flash → build bubble curve → MSE vs experiment
    using your private Flash stack.
    """

    def __init__(self, base_kij: np.ndarray, param_indices: list[int]):
        self.base = np.asarray(base_kij, dtype=float).copy()
        self.param_indices = list(param_indices)

    def _to_full(self, x: np.ndarray) -> np.ndarray:
        full = self.base.copy()
        for i, idx in enumerate(self.param_indices):
            full[idx] = float(x[i])
        return full

    def evaluate(self, x: np.ndarray) -> float:
        _full = self._to_full(x)
        # TODO: call private Flash + curve MSE here
        raise NotImplementedError("Connect your private Flash evaluator here")

    def evaluate_many(self, xs):
        return [self.evaluate(x) for x in xs]


# Usage sketch (private machine with Flash available):
#
#   from GD import run_gd, GDSettings
#   obj = FlashCurveObjective(base_kij, [203, 190, 150])
#   result = run_gd(obj, x0, GDSettings(simple_stop=True, loss_target=1.0))
