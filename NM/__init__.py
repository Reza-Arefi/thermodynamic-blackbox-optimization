"""Nelder–Mead — optimization only (black-box objective)."""

from .optimize import run_nm
from .settings import NMSettings
from .stopping import check_nm_native_stop

__all__ = ["NMSettings", "run_nm", "check_nm_native_stop"]
