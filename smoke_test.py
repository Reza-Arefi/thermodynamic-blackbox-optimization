"""
Smoke-test all optimizers on QuadraticDemo (no Flash).

Run from Git/:
  python3 smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from common.objective import QuadraticDemo
from common.gpu_stopping import GPUStopSettings
from GD import GDSettings, run_gd
from NM import NMSettings, run_nm
from PSO import PSOSettings, run_pso
from GPU_PSO import GPUPSOSettings, run_gpu_pso
from GPU_DE import GPUDESettings, run_gpu_de
from GPU_CMAES import GPUCMAESSettings, run_gpu_cmaes
from Newton import NewtonSettings, run_newton


def main() -> None:
    dim = 4
    # Optimum inside default ±0.5 bounds around x0
    x_star = np.full(dim, 0.25)
    obj = QuadraticDemo(dim=dim, x_star=x_star)
    x0 = np.zeros(dim)

    print("=== GD ===")
    r = run_gd(obj, x0, GDSettings(max_iter=300, learning_rate=0.5, simple_stop=True, loss_target=1e-8))
    print(r["best_loss"], r["iterations"], r["stop_reason"][:80])

    print("=== NM ===")
    r = run_nm(obj, x0, NMSettings(max_iter=200, simple_stop=True, loss_target=1e-8))
    print(r["best_loss"], r["iterations"], r["stop_reason"][:80])

    print("=== PSO ===")
    r = run_pso(obj, x0, PSOSettings(max_iter=100, seed=0, simple_stop=True, loss_target=1e-6))
    print(r["best_loss"], r["iterations"], r["stop_reason"][:80])

    stop = GPUStopSettings(simple_stop=True, loss_target=1e-6, stagnation_window=50)
    print("=== GPU-PSO ===")
    r = run_gpu_pso(obj, x0, GPUPSOSettings(max_iter=100, seed=0, stop=stop))
    print(r["best_loss"], r["iterations"], r["stop_reason"][:80])

    print("=== GPU-DE ===")
    r = run_gpu_de(obj, x0, GPUDESettings(max_iter=100, seed=0, stop=stop))
    print(r["best_loss"], r["iterations"], r["stop_reason"][:80])


    print("=== Newton ===")
    r = run_newton(obj, x0, NewtonSettings(max_iter=50, simple_stop=True, loss_target=1e-8))
    print(r["best_loss"], r["iterations"], r["stop_reason"][:80])

    print("=== GPU-CMAES ===")
    r = run_gpu_cmaes(obj, x0, GPUCMAESSettings(max_iter=100, seed=0, stop=stop))
    print(r["best_loss"], r["iterations"], r["stop_reason"][:80])

    print("OK")


if __name__ == "__main__":
    main()
