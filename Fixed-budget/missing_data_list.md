# Missing data inventory for fixed-budget GPU-PSO / GPU-CMAES / GPU-DE

Generated for the seed-42 + seed-2 comparison grid  
`N ∈ {2, 3, 5, 8, 10, 20, 40, 80, 120, 160, 210}`.

## A. Fully missing runs (no history at all) — **IN PROGRESS**

| # | method | seed | n | status | notes |
|---|--------|------|---|--------|-------|
| 1–11 | GPU-DE | 2 | 2…210 | **RUNNING** sequential sweep | PID parent `bash GPU-de/run_param_sweep.sh`; log below |

Launch details (live):

```text
GPUS=all (0–7)
WORKERS=16  → 128 concurrent pvtXpert slots
SEED=2  RANDOM_PARAMS=1  (matches PSO/CMAES seed2 BIPs)
CPU_POP=1  (NumPy pop ops — avoids Torch CUDA segfault on this driver)
MAX_ITER=2000
SWEEP_DIR=GPU-de/results/sweep_gpu_all_w16_seed2_random/
MASTER_LOG=GPU-de/gpu_de_sweep_gpu_all_w16_seed2_random.log
```

## Disconnect-safe & resume

Already designed for long runs:

| Mechanism | Behavior |
|-----------|----------|
| **Detach** | `nohup` + `setsid` (new session); survives SSH close |
| **SIGHUP** | Ignored in bash sweep and `GPU-de/run.py` |
| **Checkpoint** | Every iteration → `gpu_de_checkpoint.npz` + `_meta.json` |
| **Auto-resume** | If checkpoint exists, `run.py` continues from last iteration |
| **Sweep skip** | Finished/converged N skipped on re-run |
| **Double-start guard** | Won’t launch a second copy while one is active |

```bash
# status
bash Fixed-budget/run_missing_jobs.sh --status

# if process died, resume (safe anytime)
bash Fixed-budget/run_missing_jobs.sh

# monitor
tail -f GPU-de/gpu_de_sweep_gpu_all_w16_seed2_random.log
```

After the sweep finishes:

```bash
python Fixed-budget/fixed_budget_analysis.py
```


## B. Incomplete high budgets (history exists, short early stop) — **not relaunched by default**

These have full anytime curves up to Stop A/B, but fixed-budget cells with  
`B > total_evals` use `approx_hold_final`. Re-running only helps if early-stop is  
relaxed or a higher evaluation floor is forced (different CLI/regime).

See `tables/rerun_todo.csv` issues `short_vs_peers` / `high_approx_fraction`  
(≈40 medium-priority rows, seed 42). Typical short cases:

| method | seed | n | total_evals | peer_max_same_N |
|--------|------|---|-------------|-----------------|
| GPU-CMAES | 42 | 2 | 350 | 1500 |
| GPU-DE | 42 | 2 | 440 | 1500 |
| GPU-CMAES | 42 | 3 | 195 | 1155 |
| GPU-PSO | 42 | 210 | 1770 | 12400 |
| GPU-CMAES | 42 | 210 | 2950 | 12400 |
| … | … | … | … | (full list in CSV) |

## C. Already complete (do not re-run)

| method | seed | cases present |
|--------|------|----------------|
| GPU-PSO | 42 | 11/11 |
| GPU-PSO | 2 | 11/11 |
| GPU-CMAES | 42 | 11/11 |
| GPU-CMAES | 2 | 11/11 |
| GPU-DE | 42 | 11/11 |
| GPU-DE | 2 | **0/11** ← batch A |

## Launch command

```bash
cd ~/LCADAME/pvtR
source /opt/persistence/miniconda3/etc/profile.d/conda.sh
conda activate ddpm_env
bash Fixed-budget/run_missing_jobs.sh
# monitor:
tail -f GPU-de/gpu_de_sweep_gpu_all_w16_seed2_random.log
```

After finishes:

```bash
python Fixed-budget/fixed_budget_analysis.py
```
