# Newton Stopping Criteria (Fair Comparison)

Newton uses the **same criterion framework** as GD / NM / PSO: dimension-independent L∞ (or L2 displacement) metrics, identical stagnation windows, OR logic.

## Comparison

| Criterion | GD | NM | PSO | **Newton** |
|-----------|----|----|-----|------------|
| **1. Primary** | `max\|∇L\| < 5e-5` | `simplex_contraction < 5e-5` | `normalized_contraction < 1e-5` | `max\|∇L\| < 5e-5` |
| **2. Movement** | `\|\|Δx\|\|_2 < 1e-5` | `max_dist < 1e-5` | `swarm_diversity < 1e-5` | `\|\|Δx\|\|_2 < 1e-5` |
| **3. Stagnation** | 50 iters, `\|ΔL\|<1e-8` + primary `<1e-4` | same | same | **same** |
| **4. Extended** | (missing in GD) | 10 iters stuck | 10 iters stuck | **10 iters no improve** + `max\|∇L\| < 1e-3` |

Newton’s primary metric matches **GD** (gradient L∞). Criterion 4 matches **PSO/NM** so Newton is not favored or penalized relative to those methods.

## Method details

- **Update:** damped Newton / Levenberg–Marquardt: solve `(H + λI) s = ∇L`, then `x ← x − α s`
- **Hessian:** finite-difference (same relative step style as GD: `δ ≈ |xᵢ|/100`)
- **Globalization:** Armijo backtracking on `α`; `λ` adapted up/down
- **Bounds:** `±0.5` around initial value (same as GD/NM/PSO)
- **Max iterations:** default 2000 (same safety limit)

## Cost note

Each Newton iteration needs ≈ `1 + n + n(n+1)/2` function evaluations for FD gradient + Hessian. Prefer small `n` first; for large `n`, compare methods on **loss vs evaluations**, not only iterations.
