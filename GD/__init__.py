"""Gradient Descent (finite-difference Adagrad-style) — optimization only."""

from .optimize import run_gd
from .settings import GDSettings
from .stopping import check_gd_native_stop

__all__ = ["GDSettings", "run_gd", "check_gd_native_stop"]
