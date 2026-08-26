# Fixed-budget — manuscript preferred paths

Use the **`from_table/`** package for paper-aligned GPU anytime / fixed-budget
readouts (best seed by final loss per method×N). The root
`Fixed_budget_analysis.md` is an **older sweep-history** version; keep it for
reference only.

## Preferred analysis
- `from_table/Fixed_budget_analysis.md`

## Preferred figure notes
- `figures/chapter/FIGURE_NOTES.md`  
  - Fig 1: evaluation-based convergence (N=10 / 40 / 210)  
  - Fig 2: fixed budget B=1000 / B=2000 + Dolan–Moré profile  

Chapter figures:
- `figures/chapter/Fig_eval_convergence_N10_40_210.{png,pdf}`
- `figures/chapter/Fig_fixed_budget_B1000_B2000_profile.{png,pdf}`

Also mirrored under `../figures/fixed_budget/chapter/` for the flat figures tree.

## Preferred tables (`from_table/tables/`)
| Kind | Files |
|------|--------|
| Wide budgets | `fixed_budget_wide_B500.csv`, `…_B1000`, `…_B2000`, `…_B5000`, `…_B10000`, `…_B15000` |
| Winners | `winners_best_seed.csv`, `winners_seed42.csv` |
| AUC | `auc_normalized_best_seed.csv`, `auc_normalized_seed42.csv` |
| Run summaries | `run_summary_best_seed.csv`, `run_summary_seed42.csv` |
| Fill-kind counts | `fill_kind_counts.csv` (+ `fixed_budget_fill_wide_B*.csv`) |
| Focus / long | `fixed_budget_focus_B1000_2000_5000.csv`, `fixed_budget_long_best_seed.csv` |
| Seed choice | `selected_best_seed_runs.csv`, `seed_choice_by_method_dim.csv` |

## Legacy (optional)
- `Fixed_budget_analysis.md` — older sweep-history analysis  
- `tables/` — companion CSVs from that older pipeline  
- `missing_data_list.md`, `missing_runs_to_launch.csv`
