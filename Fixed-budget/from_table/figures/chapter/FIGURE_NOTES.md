# Figure notes (subsection fixed-budget GPU optimizers)
## Figure 1 — Evaluation-based convergence
- Panels: **(a)** N=10, **(b)** N=40, **(c)** N=210.
- Axes: $J_{\mathrm{best}}$ (log) vs simulator evaluations $N_{\mathrm{eval}}$.
- Curves use **observed history only** (best seed per method×N by final loss). No dotted extrapolations after early stopping.
- Reads: N=10 CMA-ES advantage; N=40 DE can overtake late; N=210 early-budget vs long-budget contrast.

## Figure 2 — Fixed-budget comparison
- **(a)** B=1000, **(b)** B=2000: $J_{\mathrm{best}}(N,B)$ vs $N$ (log–log).
- **(c)** Dolan–Moré performance profile. Caption should state: computed over the fully observed benchmark instances at $B{=}1000$ and $B{=}2000$. Ratio $r_{p,s}=J_{p,s}/\min_s J_{p,s}$. No in-axes annotation (legend upper right).
- If a run stopped before B, $J_{\mathrm{best}}$ holds the final observed loss (post-hoc fixed-budget reading).

## Selected seeds (best final loss)
```
   method  n_params  seed  final_loss  total_evals
GPU-CMAES        10    42    0.657602         3750
   GPU-DE        10    42    1.495204         3300
  GPU-PSO        10    42    0.828694         4737
GPU-CMAES        40     2    0.429462        20550
   GPU-DE        40    42    0.304449        16100
  GPU-PSO        40    42    0.452813         8248
GPU-CMAES       210     2    1.542089         5900
   GPU-DE       210    42    0.899544        12400
  GPU-PSO       210     2    0.854962         4710
```
