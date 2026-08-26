"""Configuration for GPU-cmaes (self-contained; does not touch GPU-pso/ or GPU-de/)."""
from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
ROOT_DIR = PACKAGE_DIR.parent  # pvtR/

PVTXPERT_BIN = ROOT_DIR / "pvtXpert.bin"
TEMPLATE_SRC = ROOT_DIR / "Problem" / "LiveOil1_Template.json"
DOCKER_IMAGE = os.environ.get("PVTXPERT_DOCKER_IMAGE", "nvidia-pytorch:24.03")

# Default: all 8 A100s for parallel offspring evaluations
DEFAULT_GPUS = list(range(8))

WORKSPACES_DIR = PACKAGE_DIR / "workspaces"
RESULTS_DIR = PACKAGE_DIR / "results"

ENVELOPE_NAME = "LiveOil1_3.731e+02_Envelope.dat"
BLOCK_SIZE = "32"

# CMA-ES defaults (Hansen). sigma0 scaled relative to bound half-width (0.5).
SIGMA0_DEFAULT = 0.3
CMA_STAGNATION_LIMIT = 10  # Stop A short window (same as GPU-pso)

# Stopping criteria — identical to GPU-pso Stop A / B / C
REL_IMPROVE_TOL = 1e-6
CONTRACTION_RATIO_TOL = 0.10
STOP_A_WINDOW = 10
STOP_B_WINDOW = 50
