"""Configuration for GPU-pso (self-contained; does not touch pso/)."""
from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
ROOT_DIR = PACKAGE_DIR.parent  # pvtR/

PVTXPERT_BIN = ROOT_DIR / "pvtXpert.bin"
TEMPLATE_SRC = ROOT_DIR / "Problem" / "LiveOil1_Template.json"
DOCKER_IMAGE = os.environ.get("PVTXPERT_DOCKER_IMAGE", "nvidia-pytorch:24.03")

# Default: all 8 A100s for parallel particle evaluations
DEFAULT_GPUS = list(range(8))

WORKSPACES_DIR = PACKAGE_DIR / "workspaces"
RESULTS_DIR = PACKAGE_DIR / "results"

ENVELOPE_NAME = "LiveOil1_3.731e+02_Envelope.dat"
BLOCK_SIZE = "32"

# Match sequential pso/pso_rPar.py adaptive coefficients (linear baseline)
W_MAX = 1.0
W_MIN = 0.1
C1_MAX = 2.0
C1_MIN = 0.2
C2_MAX = 2.0
C2_MIN = 0.2
PSO_STAGNATION_LIMIT = 10

# Feedback-driven coefficient adaptation (diversity + improvement)
DIV_EXPLORE = 0.20          # normalized_diversity above → explore
DIV_EXPLOIT = 0.10          # below → compact; combine with improvement
IMPROVE_EPS = 1e-6          # best-loss improvement threshold
STAG_FORCE_ITERS = 10       # consecutive flat iters → force exploitation
W_EXPLORE_STEP = 0.02
C2_EXPLORE_STEP = 0.02
W_EXPLOIT_STEP = 0.05
C2_EXPLOIT_STEP = 0.05

# Stopping criteria (Stop A / B / C)
REL_IMPROVE_TOL = 1e-6          # relative improvement = |ΔL|/max(1,|L_old|)
CONTRACTION_RATIO_TOL = 0.10    # current_diversity / initial_diversity
STOP_A_WINDOW = 10              # short relative-stagnation window (Stop A)
STOP_B_WINDOW = 50              # long relative-stagnation window (Stop B)

# Contraction-driven one-way swarm shrink
SWARM_SHRINK_STEP = 0.05        # −5% particles per −5% contraction_ratio band
SWARM_MIN_FRACTION = 0.10       # never below 10% of initial (then min particle floor)
MIN_PARTICLES = 10              # absolute floor on swarm size
