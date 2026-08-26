"""
Black-box objective interface.

Flash / envelope / curve-MSE live behind this API. Optimizers only see
`evaluate(x) -> float` (and optionally batched evaluation).
"""
from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import numpy as np


@runtime_checkable
class Objective(Protocol):
    """Scalar black-box loss on the free-parameter vector."""

    def evaluate(self, x: np.ndarray) -> float:
        """Return finite loss, or +inf if the evaluation fails."""
        ...


@runtime_checkable
class BatchObjective(Protocol):
    """Optional parallel / batched evaluation (GPU population methods)."""

    def evaluate(self, x: np.ndarray) -> float:
        ...

    def evaluate_many(self, xs: Sequence[np.ndarray]) -> list[float]:
        """Evaluate many free-parameter vectors; same order as `xs`."""
        ...


class QuadraticDemo:
    """
    Tiny analytic objective for smoke-testing optimizers without Flash.

    f(x) = ||x - x*||^2 with x* = ones(d).
    """

    def __init__(self, dim: int = 2, x_star: np.ndarray | None = None):
        self.dim = int(dim)
        self.x_star = (
            np.ones(self.dim, dtype=float)
            if x_star is None
            else np.asarray(x_star, dtype=float).ravel()
        )

    def evaluate(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float).ravel()
        return float(np.sum((x - self.x_star) ** 2))

    def evaluate_many(self, xs: Sequence[np.ndarray]) -> list[float]:
        return [self.evaluate(x) for x in xs]


def as_batch(objective: Objective) -> BatchObjective:
    """Wrap a scalar objective with a sequential `evaluate_many`."""

    class _Wrapped:
        def evaluate(self, x: np.ndarray) -> float:
            return objective.evaluate(x)

        def evaluate_many(self, xs: Sequence[np.ndarray]) -> list[float]:
            if hasattr(objective, "evaluate_many"):
                return objective.evaluate_many(xs)  # type: ignore[attr-defined]
            return [objective.evaluate(x) for x in xs]

    return _Wrapped()
