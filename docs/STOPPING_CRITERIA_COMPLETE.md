# Complete Stopping Criteria for All Optimization Methods

This document lists all stopping criteria for each optimization method.

---

## 1. Nelder-Mead (NM) - Standalone

**File**: `nm_rPar.py`  
**Default Max Iterations**: `2000`  
**Default Stagnation Limit**: `10` iterations

### Stopping Criteria (OR Logic - ANY criterion can stop)

| Criterion | Condition | Threshold | Description |
|-----------|-----------|-----------|-------------|
| **Criterion 1** | `simplex_contraction < 5e-5` | `5e-5` | Simplex contraction (max distance from best point to any vertex) |
| **Criterion 2** | `max_dist < 1e-5` | `1e-5` | Max distance between simplex vertices |
| **Criterion 3** | Loss unchanged for **50** iterations + `simplex_contraction < 1e-4` | `1e-8` (absolute) | Loss stagnation with small simplex |
| **Criterion 4** | Loss stuck + Simplex stuck for **10** iterations | `1e-10` (absolute) | Completely stuck (no loss/simplex change) |

**Criterion 3 Details**:
- Requires: `it > 50`
- Check: All losses in last 50 iterations unchanged (within `1e-8`)
- AND: `simplex_contraction < 1e-4`

**Criterion 4 Details**:
- Requires: `it > nm_stagnation_limit` (default: 10)
- Check: Loss unchanged (within `1e-10`) for last 10 iterations
- AND: Simplex size unchanged (rounded to 6 decimals) for last 10 iterations

**Safety Limit**:
- If `it >= max_iter` (2000) without convergence → Warning, stops

---

## 2. Particle Swarm Optimization (PSO) - Standalone

**File**: `pso_rPar.py`  
**Default Max Iterations**: `2000`  
**Default Stagnation Limit**: `10` iterations

### Stopping Criteria (OR Logic - ANY criterion can stop)

| Criterion | Condition | Threshold | Description |
|-----------|-----------|-----------|-------------|
| **Criterion 1** | `normalized_contraction < 1e-5` | `1e-5` | Normalized swarm contraction (diversity / parameter range) |
| **Criterion 2** | `position_std < 1e-5` | `1e-5` | Absolute swarm diversity (std of particle positions) |
| **Criterion 3** | Loss unchanged for **50** iterations + `normalized_contraction < 1e-4` | `1e-8` (absolute) | Loss stagnation with small swarm |
| **Criterion 4** | Loss unchanged for **10** iterations + `normalized_contraction < 0.05` | `1e-8` (absolute) | Extended loss stagnation |

**Criterion 3 Details**:
- Requires: `t > 50`
- Check: All losses in last 50 iterations unchanged (within `1e-8`)
- AND: `normalized_contraction < 1e-4`

**Criterion 4 Details**:
- Requires: `t > pso_stagnation_limit` (default: 10)
- Check: All losses in last 10 iterations unchanged (within `1e-8`)
- AND: `normalized_contraction < 0.05` (5%)

**Minimum Iterations**:
- Convergence requires: `t >= 5`

**Safety Limit**:
- If `t >= max_iter` (2000) without convergence → Warning, stops

---

## 3. Hybrid PSO-NM

**File**: `pso-nm_rPar.py`  
**Default PSO Safety Limit**: `1000` iterations  
**Default NM Safety Limit**: `1000` iterations  
**Default PSO Stagnation Limit**: `10` iterations  
**Default NM Stagnation Limit**: `10` iterations

### Phase 1: PSO Phase Stopping Criteria

The PSO phase can stop in **two ways**:

#### A. PSO Convergence (Early Exit - Skips NM Phase)

**Logic**: OR (ANY criterion can trigger)

| Criterion | Condition | Threshold | Description |
|-----------|-----------|-----------|-------------|
| **Criterion 1** | `normalized_contraction < 1e-5` | `1e-5` | Normalized swarm contraction |
| **Criterion 2** | `position_std < 1e-5` | `1e-5` | Absolute swarm diversity |
| **Criterion 3** | Loss unchanged for **50** iterations + `normalized_contraction < 1e-4` | `1e-8` (absolute) | Loss stagnation with small swarm |
| **Criterion 4** | Loss unchanged for **10** iterations + `normalized_contraction < 0.05` | `1e-8` (absolute) | Extended loss stagnation |

**Details**: Identical to standalone PSO convergence criteria.

**Minimum Iterations**: `t >= max(5, loss_stagnation_window)` (default: 10)

**Result**: If converged → Optimization complete, **skips NM phase**

---

#### B. Switching to NM Phase (Normal Transition)

**Logic**: AND (BOTH conditions must be met)

| Condition | Threshold | Default | Description |
|-----------|-----------|---------|-------------|
| **Contraction** | `normalized_contraction < contraction_tolerance` | `0.01` (1%) | Swarm contraction threshold |
| **Loss Stagnation** | Relative change < `loss_stagnation_tolerance` over `loss_stagnation_window` | `1e-4` (0.01%) over `10` iterations | Relative loss stagnation |

**Loss Stagnation Calculation**:
```
loss_stagnation = |L(t) - L(t-window)| / (|L(t-window)| + ε)
loss_stagnated = (loss_stagnation < loss_stagnation_tolerance)
```

**Minimum Iterations**: `t >= max(5, loss_stagnation_window)` (default: 10)

**Result**: If both met → **Switches to NM phase**

---

#### C. PSO Safety Limit

- If `t >= pso_safety_limit` (1000) without convergence or switching → Warning, **continues to NM phase**

---

### Phase 2: NM Phase Stopping Criteria

**Logic**: OR (ANY criterion can stop)

| Criterion | Condition | Threshold | Description |
|-----------|-----------|-----------|-------------|
| **Criterion 1** | `simplex_contraction < 5e-5` | `5e-5` | Simplex contraction |
| **Criterion 2** | `max_dist < 1e-5` | `1e-5` | Max distance between vertices |
| **Criterion 3** | Loss unchanged for **50** iterations + `simplex_contraction < 1e-4` | `1e-8` (absolute) | Loss stagnation with small simplex |
| **Criterion 4** | Loss stuck + Simplex stuck for **10** iterations | `1e-10` (absolute) | Completely stuck |

**Details**: Identical to standalone NM convergence criteria.

**Safety Limit**:
- If `it >= nm_safety_limit` (1000) without convergence → Warning, stops

---

## 4. Gradient Descent (GD) - Standalone

**File**: `gd_rPar.py`  
**Default Max Iterations**: `2000`

### Stopping Criteria (OR Logic - ANY criterion can stop)

| Criterion | Condition | Threshold | Description |
|-----------|-----------|-----------|-------------|
| **Criterion 1** | `max_gradient_magnitude < 5e-5` | `5e-5` | L∞ norm of gradient: `||∇L||_∞ = max\|∇L_i\|` |
| **Criterion 2** | `param_displacement < 1e-5` | `1e-5` | L2 norm of parameter change between iterations |
| **Criterion 3** | Loss unchanged for **50** iterations + `max_gradient_magnitude < 1e-4` | `1e-8` (absolute) | Loss stagnation with small gradient |

**Criterion 3 Details**:
- Requires: `iter_num > 50`
- Check: All losses in last 50 iterations unchanged (within `1e-8`)
- AND: `max_gradient_magnitude < 1e-4`

**Safety Limit**:
- If `iter_num >= num_iterations` (2000) without convergence → Warning, stops

---

## Summary Comparison Table

| Method | Max Iterations | Convergence Logic | Number of Criteria | Stagnation Limit |
|--------|---------------|-------------------|---------------------|------------------|
| **NM** | 2000 | OR (any) | 4 | 10 iterations |
| **PSO** | 2000 | OR (any) | 4 | 10 iterations |
| **Hybrid PSO** | 1000 (safety) | OR (any) for convergence, AND (both) for switching | 4 (convergence) or 2 (switching) | 10 iterations |
| **Hybrid NM** | 1000 (safety) | OR (any) | 4 | 10 iterations |
| **GD** | 2000 | OR (any) | 3 | N/A |

---

## Key Differences

### 1. **Hybrid PSO has TWO stopping mechanisms**:
   - **Convergence** (strict, skips NM): Same as standalone PSO
   - **Switching** (lenient, goes to NM): Much more lenient (1% contraction vs 0.001%)

### 2. **Contraction Thresholds**:
   - **PSO Convergence**: `1e-5` (0.001%)
   - **PSO Switching**: `0.01` (1%) - **1000× more lenient**
   - **NM Convergence**: `5e-5` (0.005%)

### 3. **Loss Stagnation Types**:
   - **PSO Convergence**: Absolute change (`1e-8`)
   - **PSO Switching**: Relative change (`1e-4` relative)
   - **NM Convergence**: Absolute change (`1e-8` or `1e-10`)

### 4. **Safety Limits**:
   - **Standalone methods**: 2000 iterations
   - **Hybrid PSO phase**: 1000 iterations
   - **Hybrid NM phase**: 1000 iterations

---

## Notes

1. **All methods use OR logic** for convergence (any criterion can stop)
2. **Hybrid switching uses AND logic** (both conditions must be met)
3. **Stagnation limits are consistent**: 10 iterations for all methods
4. **Loss tolerance**: `1e-8` for most criteria, `1e-10` for "completely stuck" detection
5. **Minimum iterations**: PSO requires at least 5-10 iterations before convergence can trigger



