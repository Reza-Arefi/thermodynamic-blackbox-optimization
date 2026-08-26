"""Shared black-box objective interface and unified stopping helpers."""

from .objective import BatchObjective, Objective, QuadraticDemo
from .params import default_bounds, select_random_parameters, to_full_vector
from .stopping_unified import UnifiedStopConfig, check_unified_stop

__all__ = [
    "Objective",
    "BatchObjective",
    "QuadraticDemo",
    "select_random_parameters",
    "default_bounds",
    "to_full_vector",
    "UnifiedStopConfig",
    "check_unified_stop",
]
