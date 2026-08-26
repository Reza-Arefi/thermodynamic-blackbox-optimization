# Black-Box Optimization for High-Dimensional Thermodynamic Parameter Calibration

Companion code, tables, figures, and numerical results for the paper:

> **Black-Box Optimization of High-Dimensional Thermodynamic Parameter Calibration: A Comparative Study of Sequential, Population-Based, and GPU-Parallel Optimization Methods**

This repository packages the **optimization algorithms**, **stopping criteria**, **manuscript tables and figures**, and **run summaries** used in that study. The thermodynamic flash solver that evaluates each candidate parameter vector is treated as an **external black-box objective**: it is **not** redistributed here. Optimizers call only `evaluate(x) → loss` (and, for population methods, batched `evaluate_many`).

All tabulated losses, iteration counts, evaluation counts, wall-clock times, and derived comparison tables in this repository come from **completed optimization runs** recorded in per-run `result.json` files (and the CSVs/TeX built from them). Figures in this package are the **generated** manuscript assets corresponding to those runs and analyses.

---

## Repository map (where to look)

| Path | What you find |
|------|----------------|
| [`common/`](common/) | Black-box objective interface, unified / GPU stopping helpers, Flash adapter skeleton |
| [`GD/`](GD/) [`NM/`](NM/) [`PSO/`](PSO/) [`Newton/`](Newton/) | Sequential optimizers: algorithm, settings, native stopping |
| [`GPU_PSO/`](GPU_PSO/) [`GPU_DE/`](GPU_DE/) [`GPU_CMAES/`](GPU_CMAES/) | GPU-oriented population methods (batch objective), Stop A/B/C/L |
| [`docs/`](docs/) | Stopping-criteria documentation and LaTeX summary table |
| [`tables/`](tables/) | Paper comparison tables (TeX + CSV) |
| [`results/`](results/) | Light run summaries (`result.json` only) + index CSVs |
| [`figures/`](figures/) | Manuscript figures organized by topic |
| [`manuscript_figures/`](manuscript_figures/) | Full figure tree (phase maps, loss surfaces, paradigm panels) |
| [`performance_profiles/`](performance_profiles/) | Dolan–Moré / performance-profile plots and source CSVs |
| [`Fixed-budget/`](Fixed-budget/) | Anytime / fixed-budget GPU comparison (**prefer `from_table/`**) |
| [`smoke_test.py`](smoke_test.py) | Runs all optimizers on an analytic quadratic (no flash required) |

Quick orientation for manuscript materials:

- **Method comparison tables:** [`tables/results_low.tex`](tables/results_low.tex), [`tables/results_high.tex`](tables/results_high.tex)
- **GPU multi-seed (median / IQR):** [`tables/tab_gpu_multiseed.tex`](tables/tab_gpu_multiseed.tex), [`tables/results_gpu_seeds_summary.csv`](tables/results_gpu_seeds_summary.csv)
- **Stopping criteria table:** [`tables/tab_stopping_criteria_all.tex`](tables/tab_stopping_criteria_all.tex) and [`docs/`](docs/)
- **Fixed-budget (manuscript):** see [Fixed-budget preferred paths](#fixed-budget-analysis-manuscript) below

---

## Scientific problem (what is optimized)

**Decision variables.** Binary interaction parameters (BIPs) of a multicomponent equation-of-state model, packed as a length-210 upper-triangle vector. Comparative experiments activate the **top-\(N\)** parameters ranked by a full-dimensional gradient-descent sensitivity study (\(N \in \{2,3,5,8,10,20,40,80,120,160,210\}\)).

**Objective (black box).** For a candidate BIP vector the private flash / stability stack returns a bubble-point (phase-envelope) curve. The scalar loss is the **mean squared error (MSE)** between:

1. the experimental envelope pressures, and  
2. the computed envelope pressures,

both resampled by linear interpolation onto a **fixed grid of 1000 compositions** \(Z_{\mathrm{CO}_2}\) spanning the experimental \(Z\) range. The optimizer never sees flash internals—only this scalar loss (or \(+\infty\) on failed evaluations).

**Bounds.** Free parameters are box-constrained at \(\pm 0.5\) about the baseline BIP value (same protocol for all methods).

---

## Optimizers included

Each method folder exposes `settings.py`, `stopping.py` (or shared GPU stopping), and `optimize.py` with a `run_*` entry point that accepts a black-box `Objective` / `BatchObjective`.

| Family | Folder | Algorithm (this package) |
|--------|--------|---------------------------|
| Sequential | `GD/` | Finite-difference gradients + Adagrad-style adaptive steps |
| Sequential | `NM/` | Nelder–Mead simplex |
| Sequential | `PSO/` | Particle swarm (host updates; serial objective calls) |
| Sequential | `Newton/` | Damped Newton / Levenberg–Marquardt, FD gradient & Hessian, Armijo line search |
| Population / GPU-oriented | `GPU_PSO/` | PSO with batched `evaluate_many` (parallel flash jobs in the full study) |
| Population / GPU-oriented | `GPU_DE/` | DE/rand/1/bin, batched trials |
| Population / GPU-oriented | `GPU_CMAES/` | Hansen CMA-ES, batched offspring |

**Wiring your flash (private environment).** Implement `evaluate` / `evaluate_many` using [`common/flash_adapter_skeleton.py`](common/flash_adapter_skeleton.py). Do not commit proprietary binaries into this repository.

**Smoke test (no flash):**

```bash
python3 smoke_test.py
```

---

## Stopping criteria

### Native scientific stopping (method-specific, OR logic)

Documented in [`docs/STOPPING_CRITERIA_SUMMARY.md`](docs/STOPPING_CRITERIA_SUMMARY.md), [`docs/tab_stopping_criteria_all.tex`](docs/tab_stopping_criteria_all.tex), and each method’s `stopping.py` / `Newton/STOPPING_CRITERIA.md`.

Summary:

- **GD / Newton:** \(\|\nabla L\|_\infty < 5\times10^{-5}\); \(\|\Delta\mathbf{x}\|_2 < 1\times10^{-5}\); loss stagnation (50 iterations, \(|\Delta L|<10^{-8}\)) with small gradient; Newton also uses extended no-improvement + gradient gate.
- **NM:** simplex contraction / max distance thresholds; loss stagnation; complete stagnation of loss and geometry.
- **PSO:** normalized swarm contraction and diversity thresholds; loss stagnation; extended stagnation.
- **GPU-PSO / GPU-DE / GPU-CMA-ES:** **Stop A** (short relative-improvement plateau + contraction ratio), **Stop B** (long relative-improvement plateau), **Stop C** (iteration budget), with a minimum-iteration gate before A/B.

Relative improvement for GPU methods:

\[
\mathrm{rel\_imp} = \frac{|L_{\mathrm{old}}-L_{\mathrm{new}}|}{\max(1,|L_{\mathrm{old}}|)}.
\]

### Unified table protocol (`simple_stop`)

Used for the fair comparison tables:

- **Stop L:** best loss \(< 1.0\) (configurable `--loss-target`)
- **Stop S:** relative-improvement plateau for 50 iterations (configurable window / tolerance)
- **Stop C:** `max_iter`

Shared helpers: [`common/stopping_unified.py`](common/stopping_unified.py), [`common/gpu_stopping.py`](common/gpu_stopping.py).

---

## What was calculated and what was generated

Everything below is produced from **executed optimization runs** and **deterministic post-processing** of those runs (no synthetic substitution of table primary metrics).

### 1. Per-run optimization records

For each method × dimension × seed configuration that finished, a `result.json` stores the **recorded** final MSE, iteration count, evaluation count, wall-clock time, and stop reason.

| Collection | Location | Contents |
|------------|----------|----------|
| Seed-42 method table | [`results/seed42_table/`](results/seed42_table/) | GD, NM, PSO, GPU-PSO, GPU-CMAES, GPU-DE by \(N\) |
| GPU multi-seed | [`results/gpu_seeds/`](results/gpu_seeds/) | GPU-PSO / GPU-CMAES / GPU-DE, seeds 42–44 |
| Flat index | [`results/results_index.csv`](results/results_index.csv) | One row per `result.json` |
| GPU median/IQR summary | [`results/results_gpu_seeds_summary.csv`](results/results_gpu_seeds_summary.csv) | Aggregated from completed seeds |

### 2. Comparison tables (paper)

| File | Role |
|------|------|
| [`tables/results_low.tex`](tables/results_low.tex) | Methods vs \(N=2\ldots40\): MSE, iterations, evaluations, time |
| [`tables/results_high.tex`](tables/results_high.tex) | Methods vs \(N=80\ldots210\) |
| [`tables/tab_gpu_multiseed.tex`](tables/tab_gpu_multiseed.tex) | GPU methods, three seeds: **median (IQR)** |
| [`tables/results_gpu_seeds_summary.csv`](tables/results_gpu_seeds_summary.csv) | Same multi-seed aggregation as CSV |
| [`tables/tab_stopping_criteria_all.tex`](tables/tab_stopping_criteria_all.tex) | Stopping-criteria summary for the paper |
| [`tables/results_table.csv`](tables/results_table.csv), [`tables/optimization_results_table.csv`](tables/optimization_results_table.csv) | Spreadsheet forms of comparison metrics |

Primary table metrics (loss, iterations, evaluations, time) are taken from the completed runs above. Multi-seed entries are the **median** and **interquartile range** computed over the finished seeds for that method–dimension pair.

### 3. Fixed-budget / anytime analysis

**Manuscript set (use this):** [`Fixed-budget/from_table/`](Fixed-budget/from_table/) — index in [`Fixed-budget/MANUSCRIPT_PREFERRED.md`](Fixed-budget/MANUSCRIPT_PREFERRED.md).

| Asset | Path |
|-------|------|
| Analysis write-up | [`Fixed-budget/from_table/Fixed_budget_analysis.md`](Fixed-budget/from_table/Fixed_budget_analysis.md) |
| Figure notes (Fig 1–2) | [`Fixed-budget/from_table/figures/chapter/FIGURE_NOTES.md`](Fixed-budget/from_table/figures/chapter/FIGURE_NOTES.md) |
| Chapter figures | [`Fixed-budget/from_table/figures/chapter/`](Fixed-budget/from_table/figures/chapter/) |
| Wide budget tables | `from_table/tables/fixed_budget_wide_B*.csv` |
| Winners | `from_table/tables/winners_*.csv` |
| Normalized AUC | `from_table/tables/auc_normalized_*.csv` |
| Run summaries | `from_table/tables/run_summary_*.csv` |
| Fill-kind inventory | `from_table/tables/fill_kind_counts.csv` |

**How fixed-budget values are obtained.** For each selected run (lowest final loss among available seeds for that method×\(N\)), the **anytime history of best-so-far loss versus cumulative evaluations** is read from the completed run. At each budget \(B\) on a fixed grid, \(J_{\mathrm{best}}(B)\) is taken from that history when \(B\) lies within the recorded evaluation range. When a run has already stopped before \(B\), the **final recorded best loss** is used as the fixed-budget reading (explicit hold of the last observed value). Fill-kind columns in the CSVs label which rule was applied for each cell (`observed` vs post-stop hold, etc.).

Legacy sweep write-up (optional): [`Fixed-budget/Fixed_budget_analysis.md`](Fixed-budget/Fixed_budget_analysis.md) — marked legacy in [`Fixed-budget/LEGACY_NOTE.txt`](Fixed-budget/LEGACY_NOTE.txt).

### 4. Performance profiles

| Path | Content |
|------|---------|
| [`performance_profiles/`](performance_profiles/) | Profiles vs loss, iterations, evaluations, runtime (PNG + CSV) |
| [`figures/performance_profiles/`](figures/performance_profiles/) | Same plots plus convergence comparison figures |
| `fig2_dolan_more_ratios.csv` | Dolan–Moré ratios for fixed-budget Fig 2 |

Profiles are computed from the tabulated method×problem outcomes (performance ratios relative to the best method on each instance).

### 5. Manuscript figures

| Topic | Locations |
|-------|-----------|
| Phase maps (\(T = 323.15, 373.15, 423.15\,\mathrm{K}\)) | [`figures/phase_map/`](figures/phase_map/), [`manuscript_figures/phase_map/`](manuscript_figures/phase_map/) |
| Smoothed loss surfaces | [`figures/ls_smooth/`](figures/ls_smooth/), [`manuscript_figures/ls_smooth/`](manuscript_figures/ls_smooth/) |
| Paradigm comparison | [`figures/paradigm_comparison/`](figures/paradigm_comparison/), `…_from_table/` |
| GD chapter (210P sensitivity, trajectories, cumulative mass) | [`figures/chapter7_gd/`](figures/chapter7_gd/) |
| Fixed-budget plots | [`figures/fixed_budget/`](figures/fixed_budget/), [`Fixed-budget/figures/`](Fixed-budget/figures/) |

Formats typically include **PNG, PDF, and EPS** for journal use.

---

## Fixed-budget analysis (manuscript)

Prefer this chain when citing fixed-budget / anytime GPU results:

1. [`Fixed-budget/from_table/Fixed_budget_analysis.md`](Fixed-budget/from_table/Fixed_budget_analysis.md)  
2. [`Fixed-budget/from_table/figures/chapter/FIGURE_NOTES.md`](Fixed-budget/from_table/figures/chapter/FIGURE_NOTES.md)  
3. [`Fixed-budget/from_table/tables/`](Fixed-budget/from_table/tables/) — especially `fixed_budget_wide_B*`, `winners_*`, `auc_normalized_*`, `run_summary_*`, `fill_kind_counts`

Chapter figures:

- Fig 1 — evaluation-based convergence at \(N=10,40,210\): `Fig_eval_convergence_N10_40_210.{png,pdf}`  
- Fig 2 — budgets \(B=1000\) and \(B=2000\) plus Dolan–Moré profile: `Fig_fixed_budget_B1000_B2000_profile.{png,pdf}`

---

## How to use the optimizers (black-box)

```python
import numpy as np
from GD import run_gd, GDSettings
from common.flash_adapter_skeleton import FlashCurveObjective  # implement evaluate()

# Your private flash-backed objective:
# obj = FlashCurveObjective(base_kij, param_indices=[203, 190, 150])
# x0 = np.array([base_kij[i] for i in param_indices])

# Demo without flash:
from common.objective import QuadraticDemo
obj = QuadraticDemo(dim=4, x_star=np.full(4, 0.25))
x0 = np.zeros(4)

result = run_gd(
    obj,
    x0,
    GDSettings(max_iter=2000, simple_stop=True, loss_target=1.0),
)
print(result["best_loss"], result["iterations"], result["evaluations"], result["stop_reason"])
```

Population methods (`GPU_PSO`, `GPU_DE`, `GPU_CMAES`) accept the same objective; provide `evaluate_many` for parallel evaluation in production.

---

## Reproducibility notes

- **Seeds:** Primary single-seed tables use seed **42**. GPU multi-seed tables use seeds **42, 43, 44**; medians/IQRs are computed only over completed seeds (see red / `n_seeds < 3` notes in the TeX when a cell is still incomplete).
- **Parameter sets:** Top-\(N\) indices follow the 210-parameter GD sensitivity ranking used in the paper tables.
- **Objective definition:** MSE on the 1000-point interpolated envelope curves (experimental vs computed), identical across methods.
- **This package** redistributes optimizer logic, stopping rules, and the **numerical products** of the study (tables, figures, `result.json` summaries). The flash executable and proprietary model templates remain outside the repository by design.

---

## Citation

If you use this code or the accompanying numerical results, please cite the paper:

**Black-Box Optimization of High-Dimensional Thermodynamic Parameter Calibration: A Comparative Study of Sequential, Population-Based, and GPU-Parallel Optimization Methods**

(Full bibliographic entry to be completed upon publication.)

---

## License / proprietary boundary

Optimization source, documentation, tables, figures, and run summaries in this repository are intended for open scholarly use with the paper. **Do not** expect `pvtXpert` / flash binaries, CUDA flash sources, or full workspace checkpoints here—those stay in the private calibration environment behind the black-box objective interface.
