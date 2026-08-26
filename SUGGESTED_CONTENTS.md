# What was added under Git/ (copied, not moved)

## tables/
- results_low.tex, results_high.tex
- tab_gpu_multiseed.tex, results_gpu_seeds_summary.csv
- results_table.csv, optimization_results_table.csv
- tab_stopping_criteria_all.tex

## figures/
- phase_map/          (323.15 / 373.15 / 423.15 K, png+pdf+eps)
- ls_smooth/          (ls_smooth10/11)
- paradigm_comparison/ (+ panels, CAPTION, source CSV)
- paradigm_comparison_from_table/
- chapter7_gd/        (GD chapter figs + KPI CSVs)
- fixed_budget/       (+ chapter/ notes & paper figs)
- fixed_budget_from_table/

## results/ (light only)
- results_gpu_seeds_summary.csv
- results_index.csv                 (flat index of all result.json)
- seed42_table/**/result.json       (65 runs)
- gpu_seeds/**/result.json          (86 runs)
- No workspaces, logs, or checkpoints

## Fixed-budget/
Useful package for the paper’s fixed-budget / anytime comparison:
- Fixed_budget_analysis.md          (sweep-based analysis)
- from_table/Fixed_budget_analysis.md  (table-aligned analysis; prefer for manuscript)
- tables/*.csv                      (wide/long budgets, winners, AUC, fill kinds)
- from_table/tables/*.csv
- figures/ + figures/chapter/       (anytime, bars, profiles, FIGURE_NOTES.md)
- from_table/figures/
- fixed_budget_analysis.py, make_chapter_figures.py
- missing_data_list.md, missing_runs_to_launch.csv

Still omitted (on purpose): Flash/pvtXpert, workspaces, raw logs, full checkpoints.

## Fixed-budget/
**Manuscript preferred:** see `Fixed-budget/MANUSCRIPT_PREFERRED.md`

- Prefer `from_table/Fixed_budget_analysis.md` (best seed by final loss per method×N)
- Figure notes: `figures/chapter/FIGURE_NOTES.md` (also under `from_table/figures/chapter/`)
- Preferred tables: `from_table/tables/` — `fixed_budget_wide_B*.csv`, `winners_*.csv`, `auc_normalized_*.csv`, `run_summary_*.csv`, `fill_kind_counts.csv`
- Root `Fixed_budget_analysis.md` = older sweep-history (legacy; see `LEGACY_NOTE.txt`)
- Scripts: `fixed_budget_analysis.py`, `make_chapter_figures.py`

## performance_profiles/
- Dolan–Moré / performance profiles (loss, iterations, evaluations, runtime) png+csv
- Also under `figures/performance_profiles/` (+ methods_convergence_comparison.png)
- `fig2_dolan_more_ratios.csv` (Fixed-budget Fig 2 ratios)

## manuscript_figures/
Full copy of manuscript figure tree (phase_map, ls_smooth, paradigm_comparison*).
Generator scripts included where present. Mirrors also live under `figures/`.

## Newton/
Damped Newton / LM black-box optimizer + native stopping + `STOPPING_CRITERIA.md`
