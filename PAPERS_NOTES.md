# Research Papers: Code Analysis and Implementation Notes

> Tracks what we extracted from each paper's code, how it works, and what we implemented.
> Updated: 2026-04-08

---

## Priority Ranking

| # | Paper | Venue | Code Applied? | Where |
|---|-------|-------|--------------|-------|
| 1 | **WireMask-BBO** | NeurIPS 2023 | ✅ Yes | `_wire_pull_perturb()` in placer.py |
| 2 | **MaskRegulate** | NeurIPS 2024 | ✅ Yes | `_density_gradient_perturb()` in placer.py |
| 3 | **ChiPFormer** | ICML 2023 | 🔲 Planned | `submissions/ml_placer/` (Role B) |
| 4 | DREAMPlace | DAC 2019 | 🔲 Planned | `scripts/pb_to_bookshelf.py` (Role C) |
| 5 | MaskPlace | NeurIPS 2022 | 🔲 Future | Foundation for ChiPFormer |

---

## Paper 1: WireMask-BBO (NeurIPS 2023)

**Full title:** "Macro Placement by Wire-Mask-Guided Black-Box Optimization"
**Authors:** Gu et al. (LAMDA Group, Nanjing University)
**GitHub:** https://github.com/lamda-bbo/WireMask-BBO
**Key file:** `utils.py` (greedy evaluator), `BO.py` (Bayesian optimizer)

---

### How It Works

WireMask-BBO separates placement into two parts: a **fast greedy evaluator** (the "wire mask") and a **black-box optimizer** (Bayesian or evolutionary) that queries it.

#### The Wire Mask Evaluator (`utils.py`)

The core insight: instead of scoring a placement after placing ALL macros, score each macro's position **relative to what's already placed**:

```python
# For each macro (in order of net connectivity, heaviest first):
# 1. Compute HPWL contribution of each grid cell for this macro
# 2. Place macro at the minimum-cost cell
# 3. Repeat

def greedy_placer_with_init_coordinate(node_id_ls, placedb, grid_num, grid_size, init_coords):
    for macro_id in rank_macros(node_id_ls):  # heaviest-connected first
        best_pos = argmin over all grid cells (
            HPWL_increase_if_placed_here(macro_id, cell)
        )
        place(macro_id, best_pos)
```

The HPWL increase of placing macro `i` at cell `(r, c)` is:
```
For each net N connected to macro i:
  current_bbox = bounding box of all OTHER macros in N already placed
  new_x = max(current_max_x, cell_x) - min(current_min_x, cell_x)
  new_y = max(current_max_y, cell_y) - min(current_min_y, cell_y)
  delta_HPWL += weight_N * (new_x + new_y - current_HPWL_N)
```

This is fast because it uses simple bounding box math with no expensive routing involved.

#### The Black-Box Optimizer

The BBO (TuRBO / Bayesian / EA) optimizes the **initial coordinate hints** passed to the greedy evaluator:
```python
class placement_eval:
    def __call__(self, x):  # x = flat [x0, y0, x1, y1, ..., xN, yN] array
        return greedy_placer(init_coords=x)  # greedy placer returns HPWL
# TuRBO minimizes HPWL over x in [0, grid_num]^(2N)
```

The BBO never touches macro positions directly. It only controls where the greedy placer starts from, and the greedy placer resolves overlaps by picking the minimum-cost grid cell.

---

### What We Adapted

We can't use their exact greedy placer (different grid format, IBM vs. ISPD2005 benchmarks). But we extracted the **wire-pull concept**:

**Key insight:** For each macro, the greedy placer always moves it *toward* its net centroid (the point that minimizes total HPWL for all its nets). We compute this analytically as a vector field:

```python
def _compute_wire_pull(pos, benchmark, n):
    pull = np.zeros((n, 2))
    for net_idx, nodes in enumerate(benchmark.net_nodes):
        weight = benchmark.net_weights[net_idx]
        hard_nodes = [nd for nd in nodes if nd < n]
        centroid = pos[hard_nodes].mean(axis=0)   # net centroid
        for nd in hard_nodes:
            pull[nd] += weight * (centroid - pos[nd])  # pull toward centroid
    return pull
```

Then `_wire_pull_perturb()` moves `init_pos` in the pull direction (capped at `frac × canvas size`) before re-legalizing.

**When this helps:** When wirelength is a meaningful cost and macros are sitting too far from their connected partners. Most useful for small benchmarks (ibm01-ibm05).

**Limitation:** Our proxy cost is dominated by congestion, not wirelength (congestion is about 2.0, wirelength is about 0.06). Wire-pull can increase congestion by clustering macros together. The time budget handles this safely: if wire-pull produces a worse proxy score, the original baseline result wins.

---

## Paper 2: MaskRegulate (NeurIPS 2024)

**Full title:** "RL Policy as Macro Regulator Rather than Macro Placer"
**Authors:** Chen et al. (LAMDA Group, Nanjing University)
**GitHub:** https://github.com/lamda-bbo/macro-regulator
**Key file:** `src/place_env/place_env.py` (environment + heatmaps)

---

### How It Works

MaskRegulate's key observation: RL placement policies (like Circuit Training) are brittle because they place from scratch. A single bad decision early on causes cascading errors in every placement after it. Instead, MaskRegulate trains an RL agent that **adjusts an existing layout** (one already computed by DREAMPlace global placement).

#### The Regularity Mask (`get_regular_mask()`)

MaskRegulate introduces a **regularity metric** alongside HPWL. The regularity mask penalizes positions that would make the placement unbalanced:

```python
def get_regular_mask(macro):
    # ratio_x = how much the macro "pulls" the center of mass left/right
    ratio_x = macro.size_x / canvas_width
    ratio_y = macro.size_y / canvas_height
    ratio_sum = ratio_x + ratio_y

    regular_mask = np.zeros((grid, grid))
    for r in range(grid):
        for c in range(grid):
            # Cost = weighted distance from canvas center
            regular_mask[r, c] = (
                ratio_x / ratio_sum * abs(c - grid/2) / grid +
                ratio_y / ratio_sum * abs(r - grid/2) / grid
            )
    return regular_mask
```

The combined step reward is:
```python
reward = wire_coeff * (-wire_mask[x, y]) + (1 - wire_coeff) * (-regular_mask[x, y])
```

This means the RL agent simultaneously:
- Reduces HPWL (wire component)
- Maintains balanced/spread placement (regularity component)

The -73% congestion improvement vs MaskPlace comes from the regularity term preventing macro clustering.

#### The Wire Mask (`get_wire_mask()`)

For each macro being placed, evaluates HPWL sensitivity at each grid cell:
```python
def get_wire_mask(macro):
    wire_mask = np.zeros((grid, grid))
    for net in macro.connected_nets:
        # Bounding box of other macros in this net (already placed)
        bbox = get_bbox(net, exclude=macro)
        for r, c in grid_cells:
            # HPWL expansion if we place macro here
            wire_mask[r, c] += net.weight * hpwl_expansion(bbox, r, c)
    return wire_mask
```

#### Observation Space

3-channel image per step:
- Channel 0: Canvas occupancy (0=empty, 0.5=boundary, 1=occupied)
- Channel 1: Wire mask (HPWL sensitivity)
- Channel 2: Position mask (binary: can macro fit here?)

CNN actor takes these 3 channels and outputs a probability distribution over grid cells.

---

### What We Adapted

We extracted the **regularity concept** as a density gradient:

Instead of training an RL agent, we directly compute the macro density grid and push macros toward empty cells:

```python
def _density_gradient_perturb(init_pos, leg_pos, ...):
    # 1. Build 20×20 occupancy grid from legalized positions
    grid = _congestion_heatmap(leg_pos, n, cw, ch, G=20)

    # 2. Smooth (3× box blur ≈ Gaussian σ=1.5) to get gradients
    smooth = blur(grid)

    # 3. Negative gradient = direction toward lower density
    grad_x[r, c] = -(smooth[r, c+1] - smooth[r, c-1]) / 2   # toward lower density
    grad_y[r, c] = -(smooth[r+1, c] - smooth[r-1, c]) / 2

    # 4. Push each macro in its gradient direction
    for i in range(n):
        cell = which_cell(leg_pos[i])
        perturbed[i] = init_pos[i] + magnitude * normalize(grad[cell])
```

**When this helps:** When macros are clustered in certain zones, causing high congestion there. After re-legalizing from the pushed starting positions, macros should be more evenly spread across the canvas, which reduces congestion and lowers the proxy score.

**Why no RL:** MaskRegulate trains for hours per benchmark. Our directed perturbation captures the same geometric idea in under 0.1 seconds with no training required.

---

## Paper 3: ChiPFormer (ICML 2023)

**Full title:** "ChiPFormer: Transferable Chip Placement via Offline Decision Transformer"
**Authors:** Lai et al. (ShanghaiTech / Microsoft Research)
**GitHub:** https://github.com/laiyao1/chipformer
**Key file:** `mingpt/model_placement.py` (DT backbone), `mingpt/trainer_placement.py` (environment)

---

### How It Works

ChiPFormer's key insight: RL placement methods (like Circuit Training and MaskPlace) require thousands of training episodes per circuit. That is impractical when you have a new circuit you have never seen before. ChiPFormer trains **once** on a dataset of existing placements and then **transfers** to new circuits without any retraining (zero-shot).

#### Architecture: Offline Decision Transformer

Based on the Decision Transformer (Chen et al., 2021), adapted for chip placement:

```
Input sequence per timestep:
  [R_t, s_t, a_t]  = [return-to-go, state-image, action]

Where:
  R_t = sum of future rewards (tells the model how good a result to aim for)
  s_t = 3-channel 84x84 image (canvas + wire_mask + position_mask)
  a_t = integer grid cell index (where to place the next macro)

At inference: set R_t to a high value and the model generates high-quality actions.
```

The DT backbone is a 6-layer causal GPT (128 embedding dim) with a critical addition: **circuit graph embeddings** as a global conditioning token.

#### Circuit Graph Embeddings (VGAE)

```python
# graph_model.py: Variational Graph Autoencoder
class VGAE(nn.Module):
    def encode(self, X):
        hidden = GraphConv(X)           # Aggregate neighbor features
        z = gaussian(mean(hidden), std(hidden))  # Reparameterization trick
        return z                        # Per-node embeddings [num_macros, hidden]

# graph_eval.py: Average over all nodes to get one vector per circuit
z_emb_avg = z.mean(axis=0)  # Shape: [hidden], one summary vector per circuit
```

This circuit embedding captures the *topology* of the netlist (how macros are connected to each other) as a single compact vector. The Decision Transformer uses this vector to condition all of its placement decisions, which is what allows it to transfer to circuits it has never seen before.

#### Dataset Generation

ChiPFormer requires a dataset of (state, action, reward) trajectories:
- 500 expert placements per circuit (from SA or greedy)
- Each placement is a sequence of T=num_macros steps
- Reward at each step = HPWL delta

```python
class StateActionReturnDataset(Dataset):
    # Loads pre-collected trajectories from .pkl files
    # Each trajectory: sequence of (canvas_img, wire_img, mask_img, action, reward)
    # Return-to-go: R_t = sum(rewards[t:]), measuring how good the rest of the episode was
```

---

### What We Can Adapt (Role B: ML Lead)

We're NOT implementing the full DT (requires GPU training + dataset collection). But we can:

1. **Dataset generation**: Our 17 benchmarks × 5 restarts × proxy scores IS a small dataset of (placement, score) pairs. Start collecting these systematically.

2. **VGAE circuit embeddings**: The `graph_model.py` VGAE is ~100 lines and gives us netlist topology features. These embeddings can initialize an MLP that predicts good noise levels per benchmark (meta-learning the noise_fracs parameter).

3. **Wire heatmap state** (`trainer_placement.py`):
   ```python
   # Gradient-based wire heatmap:
   net_img[i, :] += (start_x - i) * weight    # x-axis gradient
   net_img[:, j] += (start_y - j) * weight    # y-axis gradient
   ```
   This 84×84 image is exactly our 20×20 grid, higher resolution. Can use as input to a learned policy.

**Planned implementation path (Week 2+):**
```
submissions/ml_placer/
    gnn_model.py     ← VGAE circuit embeddings (adapt from graph_model.py)
    placer.py        ← MacroPlacer using GNN-predicted perturbation directions
    train.py         ← Train on collected (benchmark, placement, proxy_score) data
    data/runs.pkl    ← Growing dataset from experiments
```

---

## What We Did NOT Use (and Why)

| Paper | Why Not Used |
|-------|-------------|
| **MaskPlace** (NeurIPS 2022) | Needs per-circuit RL training (hours); we have no GPU training setup. The pixel-canvas state representation is cool but requires a trained CNN. |
| **Chip+Diffusion** (ICML 2025) | Needs large pre-training dataset; too early-stage for integration. Most interesting architecturally for long-term Role B work. |
| **DREAMPlace** (DAC 2019) | Requires LEF/DEF or Bookshelf format; our benchmarks are `.pb.txt`. Bridge converter (`scripts/pb_to_bookshelf.py`) is Role C Week 2 task. |
| **RePlAce** (TCAD 2019) | This IS the competition baseline (avg 1.4578). We're trying to BEAT it, not use it. Understanding its density model is the research goal. |
| **GiFt** (ICCAD 2024) | Only useful as DREAMPlace warm-start; depends on DREAMPlace integration first. |

---

## Implementation Changelog

| Date | What Changed | Files Modified |
|------|-------------|----------------|
| 2026-04-08 | WireMask-BBO wire-pull + MaskRegulate density-gradient restarts | `placer.py` |
| 2026-04-08 | This notes file created | `PAPERS_NOTES.md` |
| 2026-04-07 | Adaptive time budget + macro threshold (n>350 use density fallback) | `placer.py` |
| 2026-04-07 | Multi-restart legalization with exact proxy scoring | `placer.py` |
| 2026-04-07 | Team roles + paper summary | `TEAM_GUIDE.md` |

---

## How to Test the New Directed Restarts

```bash
# Quick test on ibm01 (fast, ~30s)
python -m macro_place.evaluate submissions/sameer_v1/placer.py -b ibm01

# Expected output with directed restarts:
#   Restart 0 (baseline)...          proxy=1.2253
#   Restart 1 (density-grad frac=4%) proxy=???   ← new
#   Restart 2 (wire-pull frac=6%)    proxy=???   ← new
#   Restart 3 (random noise=2%)      proxy=???
#   Best proxy=???

# Compare against previous version:
# sameer_v1 best known: ibm01=1.1854, ibm03=1.3944, ibm08=1.5251
# If directed restarts find better → update baselines above

# Full 17-benchmark run (~30 min):
python -m macro_place.evaluate submissions/sameer_v1/placer.py --all
```

---

## Next Research Priorities

1. **Test directed restarts**: Run full 17-benchmark eval, compare to 1.49 average.
2. **Noise sweep on ibm01**: Try density-grad at frac=0.02, 0.04, 0.06, 0.08 to find the best value.
3. **ChiPFormer VGAE**: Implement circuit embeddings to predict per-benchmark optimal parameters.
4. **MaskRegulate regularity metric**: Add the size-weighted centering component (not just occupancy density).
5. **DREAMPlace bridge**: Convert `.pb.txt` → Bookshelf for GPU-accelerated global placement.
