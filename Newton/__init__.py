"""Newton (damped / Levenberg–Marquardt) — black-box objective only."""

from .optimize import run_newton
from .settings import NewtonSettings
from .stopping import check_newton_native_stop

__all__ = ["NewtonSettings", "run_newton", "check_newton_native_stop"]
