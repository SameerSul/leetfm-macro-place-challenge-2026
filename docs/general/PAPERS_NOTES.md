# Research Papers: Code Analysis and Implementation Notes

> Historical research log. Most implementation notes describe the retired
> proxy-optimized path. The active system is hierarchy-only; use
> `ARCHITECTURE.md`, `DESIGN_FLOW.md`, and `docs/theory/` for the current flow.
> Updated: 2026-06-16

---

## Historical Priority Ranking

| # | Paper | Venue | Code Applied? | Where |
|---|-------|-------|--------------|-------|
| 1 | **WireMask-BBO** | NeurIPS 2023 | Historical | retired proxy perturbation notes |
| 2 | **MaskRegulate** | NeurIPS 2024 | Historical | retired proxy perturbation notes |
| 3 | **TILOS SA Assessment** | TCAD 2024 | Historical | retired macro-swap proxy path |
| 4 | **Hybro / WireMask Sweep** | arXiv 2024 | 🔲 Next | Per-macro greedy congestion-move sweep |
| 5 | **IncreMacro** | TCAD 2025 | 🔲 Next | Hill-climbing local search after restarts |
| 6 | **RUDY Demand Map** | ePlace-MS (TCAD 2015) | 🔲 Next | Fast in-loop congestion estimation |
| 7 | **Congestion-aware legalization** | UCSD/general | 🔲 Next | Weight displacement by congestion in legalize |
| 8 | **ChiPFormer** | ICML 2023 | 🔲 Future | `system/ml_placer/` (Role B) |
| 9 | DREAMPlace | DAC 2019 | ✅ Yes | `src/dreamplace_bridge/` grouped hierarchy placement |
| 10 | MaskPlace | NeurIPS 2022 | 🔲 Future | Foundation for ChiPFormer |

---

## NEW: 2026-05-02 Literature Survey — CPU-Actionable Papers

> Full survey conducted 2026-05-02. Focus: papers with no GPU/ML requirement, applicable to
> our Python placer within the 3300s competition budget.

---

### Survey Paper A: TILOS SA Assessment (IEEE TCAD 2024)

**Full title:** "An Updated Assessment of Reinforcement Learning for Macro Placement"
**Authors:** Cheng Zheng, Tianheng Fang, Andrew B. Kahng et al. (TILOS / UCSD)
**arXiv:** https://arxiv.org/abs/2302.11014
**Venue:** IEEE TCAD, accepted Dec 2024 (arXiv v3: 2024)

**Core algorithm:** Reproduces Google Circuit Training (CT) RL placer, then shows a carefully implemented SA baseline beats it on all benchmarks. SA uses 5 move operators:
- Swap (24%): exchange positions of two macros
- Shift (24%): move one macro by a small delta in x or y
- Move (24%): teleport one macro to a random grid cell
- Shuffle (24%): random full restart (like our current noise restart)
- Flip (4%): flip macro orientation (rotate 90°/180°)

**Key insight — GWTW metaheuristic:** Go-With-the-Winners runs N parallel SA chains. Every 10% of the run, they sync: clone the top-8 solutions across all chains, discarding the worst. This alone achieves 26% better proxy cost than naive SA with 4× fewer cores.

**Key numbers:**
- Temperature: T0=0.005, Tmin=1e-8, cooling=exp(ln(Tmin/T0)/Iters), 20×N_macros moves per T step
- SA beats CT 17/17 on proxy cost on academic benchmarks
- GWTW achieves this on CPU with multi-threading

**What to implement in our placer:**
1. **Macro swap operator**: Instead of only Gaussian noise restarts, add a swap-perturbation: pick 2 macros, exchange their positions, re-legalize, score. O(1) per move, much cheaper than full restart.
2. **Temperature-calibrated SA loop**: With 3300s budget and ~43s/score for ibm08, we can run ~77 evaluations. A proper SA loop with T0=0.005, cooling=0.97, accept bad moves with probability exp(-delta/T) is better than pure greedy best-of-N.
3. **GWTW as sequential restart**: Without parallelism, simulate GWTW by keeping top-K=3 best placements found so far and periodically re-seeding the noise perturbation from one of them (not always from the initial.plc baseline).

---

### Survey Paper B: Hybro (arXiv 2402.18311, Feb 2024)

**Full title:** "Hybro: Escaping Local Optima in Global Placement"
**Authors:** Ke Xue, Xi Lin, Yunqi Shi et al. (Nanjing University / Huawei Noah's Ark)
**arXiv:** https://arxiv.org/abs/2402.18311

**Core algorithm:** Alternates between DREAMPlace (gradient descent to convergence) and one of three perturbation operators. The key CPU-usable operator is **WireMask greedy sweep**:
- For each macro in random order, compute delta_HPWL for every legal grid cell
- Move the macro to the cell that minimizes delta_HPWL
- Repeat until no improvement (or budget)

**Key results:** On ISPD 2005, Hybro-WireMask achieves avg ranking 2.67 vs 4.17 for multi-restart DREAMPlace alone. Up to 50% HPWL reduction.

**What to implement:** A **congestion-aware greedy sweep** (our extension of WireMask):
```python
def congestion_wiremask_sweep(best_pl, plc, benchmark, n):
    """Move each macro to its best grid cell by proxy cost delta."""
    improved = True
    while improved:
        improved = False
        macro_order = np.random.permutation(n)
        for i in macro_order:
            # Try all grid cells for macro i, keep best
            best_delta, best_cell = 0, None
            for (row, col) in enumerate_legal_grid_cells(i, benchmark):
                delta = proxy_delta_approx(i, row, col, best_pl, plc, benchmark)
                if delta < best_delta:
                    best_delta, best_cell = delta, (row, col)
            if best_cell is not None:
                move_macro(i, best_cell, best_pl)
                improved = True
    return best_pl
```
The challenge: computing `proxy_delta_approx` cheaply without full evaluator calls. Options:
1. Use RUDY congestion map (see Paper F below) as congestion delta estimate
2. Use net bounding-box HPWL delta (fast) + RUDY density change (medium)
3. Accept that full evaluator calls (43s each) are too slow for per-cell scan → use a coarse grid

**Practical implementation:** Run WireMask sweep with full proxy evaluation only at the end of each sweep (after updating all macros), not per-cell. This reduces the sweep to O(N_macros) evaluations per round.

---

### Survey Paper C: WireMask-BBO (NeurIPS 2023) — full analysis already in this file

**See Paper 1 section below.** New insight from the full survey: the EA variant (WireMask-EA) with swap-only genetic operators is the best CPU method in the paper. The BBO optimizes macro ordering, not positions directly — our current approach of optimizing initial positions is equivalent.

---

### Survey Paper D: MaskRegulate (NeurIPS 2024) — full analysis already in this file

**See Paper 2 section below.** New insight: the RegularMask formula pushes macros toward chip edges, which reduces central congestion. Our current `_routing_congestion_perturb` (moves macros away from congested cells) is already doing this implicitly, but the RegularMask adds an explicit centering-avoidance term.

**New idea:** Add a soft edge-affinity term to our perturbation target:
```python
def edge_affinity_score(pos, canvas_w, canvas_h):
    """Lower is better: macros near edges = lower central congestion."""
    return sum(
        min(x/canvas_w, 1 - x/canvas_w) + min(y/canvas_h, 1 - y/canvas_h)
        for (x, y) in pos
    )
```
Use this as a tie-breaker when proxy costs are within epsilon.

---

### Survey Paper E: IncreMacro (IEEE TCAD Aug 2025)

**Full title:** "IncreMacro: Incremental Macro Placement Refinement"
**Authors:** Yuan Pu, Tinghuan Chen et al. (CUHK, HKUST)
**Paper:** https://www.cse.cuhk.edu.hk/~byu/papers/J137-TCAD2025-IncreMacro.pdf

**Core algorithm:** Post-placement incremental local search. Given an existing legal macro placement, iteratively perturb one macro at a time:
1. For each macro, try moving it to each of its K nearest grid neighbors
2. Score each candidate with proxy cost (fast, incremental update)
3. Accept if improved (greedy descent)
4. Repeat for all macros in random order (one "pass")
5. Run multiple passes until no improvement

**Why this helps us:** Our current approach scores full placements. IncreMacro shows that incremental per-macro moves converge to better local optima faster because the search space is smaller per step.

**Implementation:** After each restart finds a new best, run an IncreMacro polish:
```python
def incremental_polish(best_pl, plc, benchmark, n, t_budget):
    """Hill-climbing local search: move each macro to best neighboring cell."""
    t0 = time.time()
    improved = True
    while improved and time.time() - t0 < t_budget:
        improved = False
        for i in np.random.permutation(n):
            for delta_r, delta_c in [(-1,0),(1,0),(0,-1),(0,1),(-2,0),(2,0),(0,-2),(0,2)]:
                new_pos = shift_macro(best_pl, i, delta_r, delta_c, benchmark)
                new_leg = _will_legalize(new_pos, benchmark)
                score = compute_proxy_cost(new_leg, benchmark, plc)['proxy_cost']
                if score < current_best:
                    current_best = score
                    best_pl = new_leg
                    improved = True
                    break  # take first improvement
    return best_pl
```
At 43s/score and 8 neighbors, one full pass = 8×N_macros evaluations. For ibm08 (n=301): 2408 evaluations = 103,000s (way too slow for full evaluator).

**Practical approach:** Only run IncreMacro using FAST scoring (no full PlacementCost — use HPWL proxy or RUDY). Or run one pass of IncreMacro with the full evaluator only on the top-1 candidate found via noise restarts (use remaining budget).

---

### Survey Paper F: ePlace-MS (IEEE TCAD 2015)

**Full title:** "ePlace-MS: Electrostatics-Based Placement for Mixed-Size Circuits"
**Authors:** Jingwei Lu, Pengwen Chen, Chin-Chih Chang, Lu Sha et al. (UCSD)
**Paper:** https://cseweb.ucsd.edu/~jlu/papers/eplace-ms-tcad14/paper.pdf

**Core algorithm:** All cells (including large macros) modeled as positive charges. Density cost = electrostatic potential energy. Electric field computed via FFT (Poisson's equation). Nesterov's method minimizes WL + density jointly. The density gradient = electric field = force repelling macros from high-density regions.

**Key insight for us:** Our `_routing_congestion_perturb()` is already computing a CONGESTION gradient. The ePlace-MS approach computes a DENSITY gradient (smoothed macro occupancy → forces). Combining both signals (congestion + density) could give a better perturbation direction.

**RUDY demand map implementation:**
```python
def compute_rudy_map(pos, benchmark, n, grid_size=32):
    """
    RUDY: Rectangular Uniform Wire Density.
    For each net, spread routing demand uniformly over its bounding box.
    Returns demand[grid_size, grid_size] — cells with high demand are congested.
    """
    cw, ch = benchmark.canvas_width, benchmark.canvas_height
    gw, gh = cw / grid_size, ch / grid_size
    demand = np.zeros((grid_size, grid_size))
    
    for net_idx in range(benchmark.num_nets):
        nodes = benchmark.net_nodes[net_idx]
        weight = benchmark.net_weights[net_idx]
        # Bounding box in canvas coordinates
        xs = [pos[nd][0] for nd in nodes if nd < n]
        ys = [pos[nd][1] for nd in nodes if nd < n]
        if len(xs) < 2:
            continue
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        W, H = (x1 - x0) / gw, (y1 - y0) / gh
        if W < 1e-6 or H < 1e-6:
            continue
        # Grid cells covered by this net's bbox
        r0, r1 = int(y0/gh), min(int(y1/gh)+1, grid_size)
        c0, c1 = int(x0/gw), min(int(x1/gw)+1, grid_size)
        demand[r0:r1, c0:c1] += weight / (W * H)
    
    return demand
```

This runs in O(N_nets) ≈ O(1000-5000) per call — microseconds, not 43 seconds. Can be called inside the SA acceptance loop as a cheap congestion proxy.

**Potential use:** Replace or augment the expensive `plc.get_horizontal/vertical_routing_congestion()` calls (which require full scoring) with RUDY estimates during the perturbation phase. Only call the full evaluator to confirm improvements.

---

### Survey Paper G: B*-Tree Fast-SA (IEEE TCAD 2006)

**Full title:** "Modern Floorplanning Based on B*-Tree and Fast Simulated Annealing"
**Authors:** Tung-Chieh Chen, Yao-Wen Chang (NTU)
**Paper:** https://cc.ee.ntu.edu.tw/~ywchang/Papers/tcad06-mfloorplanning.pdf

**Core algorithm:** B*-tree encodes macro relative positions (left-to-right, bottom-to-top ordering). SA perturbations on the tree structure give legally placed results after decode. Three-stage SA cooling: high-T exploration, mid-T mixed, low-T refinement.

**Three-stage temperature schedule** (directly applicable to our noise restarts):
- Stage 1 (exploration, top 30% of iterations): high temperature, accept moves with delta×0.8 probability; emphasize diverse fracs (10-20% noise)
- Stage 2 (exploitation, middle 50%): medium temperature; emphasize winning fracs (4-8% noise)  
- Stage 3 (refinement, bottom 20%): low temperature, only accept improvements; emphasize small fracs (1-3% noise)

This maps to our existing noise_fracs list structure: the 395 entries already cycle through this pattern organically, but an explicit staged approach would be better calibrated.

---

### Survey Paper H: Routability-Driven Macro Placement (DATE 2019)

**Full title:** "Routability-Driven Macro Placement with Embedded CNN-Based Prediction"
**Authors:** Yu-Hung Huang, Zhiyao Xie et al. (UIUC)
**Paper:** https://zhiyaoxie.com/files/DATE19_Macro.pdf

**Core algorithm:** CNN predicts routing congestion from macro positions (no full router needed). Embedded in SA loop for fast accept/reject. Features: macro bounding boxes, pin density per grid cell.

**CPU-usable insight (RUDY as cheap CNN substitute):** The key is that RUDY demand map is a fast approximation to what the CNN was trained to predict. RUDY for all nets can be computed in O(N_nets) ≈ milliseconds. Full PlacementCost evaluation takes 43s. Using RUDY as a pre-filter: compute RUDY for all candidates, only call full evaluator on top-K candidates.

---

### Survey Paper I: TILOS/IEEE (2026) — arXiv:11300304

**Full title:** (Unable to fetch — IEEE paywall. Based on arxiv number, likely a 2025/2026 paper from TILOS on macro placement on modern benchmarks.)
**Note:** If this is the 2025 PartclEDA challenge baseline paper from the organizers, it would describe the exact contest setup. The IEEE DOI 11300304 maps to a 2025 publication. Need institutional access to read full text.

**From context:** The PartclEDA challenge leaderboard shows SA baseline=2.1251, RePlAce=1.4578, DREAMPlace/UT Austin=1.4076. The gap between DREAMPlace and RePlAce is 0.035 — achievable with CPU-only methods if we target the right benchmarks.

---

## Top 3 Actionable Ideas (CPU-only, 3300s budget, Python)

### Idea 1: RUDY-Guided Perturbation (Replaces expensive plc calls during search)

**What:** Compute RUDY demand map in O(N_nets) ≈ milliseconds instead of calling `plc.get_routing_congestion()` which requires full scoring (43s for ibm08).

**How it changes our algorithm:**
- Currently: 1 full eval (43s) per restart candidate
- With RUDY pre-filter: compute RUDY for 20 perturbed candidates (milliseconds), keep top-3, only full-eval those top-3 (3×43s = 129s vs 20×43s = 860s)
- Net effect: 6.6× more restarts in same budget

**Implementation steps:**
1. Implement `compute_rudy_map(pos, benchmark, grid_size=32)` (see code above)
2. In `_routing_congestion_perturb`, replace `plc.get_horizontal_routing_congestion()` with `compute_rudy_map` for gradient direction computation
3. Only call full `compute_proxy_cost` on the final legalized candidate (not during perturbation steps)

**Expected impact:** 6× more restarts = ibm08 goes from 55 to 330 restarts with same 3300s budget.

---

### Idea 2: Macro Swap + SA Loop (Replaces pure noise restart)

**What:** Add a macro-swap move operator to SA: pick 2 macros randomly, swap their positions, re-legalize, accept/reject with Metropolis criterion.

**Why better than noise restarts:** Swap is a structured move — it preserves the overall placement topology while exploring different relative orderings. Noise restart is unstructured — it randomly displaces all macros simultaneously, discarding good partial solutions.

**Temperature schedule for SA:**
```python
T0 = 0.005       # Calibrated to proxy cost scale (proxy ~1.2-1.8)
T_min = 1e-6
n_steps = budget_s / t_one_score  # e.g., 77 for ibm08
cooling = (T_min / T0) ** (1.0 / n_steps)

T = T0
best_pos, best_score = baseline_pos, baseline_score
current_pos, current_score = baseline_pos, baseline_score

for step in range(n_steps):
    # Pick move operator
    move_type = rng.choice(['swap','shift','move'], p=[0.33,0.33,0.33])
    candidate_pos = apply_move(current_pos, move_type, rng)
    candidate_leg = _will_legalize(candidate_pos, benchmark)
    candidate_score = compute_proxy_cost(candidate_leg, benchmark, plc)['proxy_cost']
    
    delta = candidate_score - current_score
    if delta < 0 or rng.random() < exp(-delta / T):
        current_pos = candidate_leg
        current_score = candidate_score
        if candidate_score < best_score:
            best_pos, best_score = candidate_leg, candidate_score
    T *= cooling
```

**Expected impact:** SA with proper accept criterion should escape local minima that our greedy best-of-N cannot. The TILOS paper showed SA beats RL on ALL benchmarks when properly implemented.

---

### Idea 3: Staged Noise Schedule (Low effort, immediate benefit)

**What:** Replace the flat cycling noise_fracs list with an explicit 3-stage schedule:
- **Stage 1** (first 33% of budget): large noise (8-15%) for global exploration
- **Stage 2** (middle 50% of budget): medium noise (4-8%) for exploitation  
- **Stage 3** (last 17% of budget): small noise (1-3%) for refinement around best found

**Why:** Our current 395-entry cycling list already has this structure implicitly, but the 0.06-dominant cycle doesn't adapt to what's been found. After 100 restarts all giving 1.18x for ibm01, Stage 3 should focus on 1-3% noise around the best 1.1854 position.

**Implementation:** Replace `noise_fracs` iteration with:
```python
elapsed_frac = (time.time() - t0) / self.time_budget_s
if elapsed_frac < 0.33:   # Stage 1: explore
    frac = rng.choice([0.08, 0.10, 0.12, 0.15, 0.20])
elif elapsed_frac < 0.83: # Stage 2: exploit
    frac = rng.choice([0.04, 0.06, 0.07, 0.08])
else:                      # Stage 3: refine
    frac = rng.choice([0.01, 0.02, 0.03, 0.04])
    # Perturb from best_pos instead of init_pos
    init_for_restart = extract_pos(best_pl)  # best found so far
```

**Risk:** The winning draws for ibm01/ibm08 (specific rng at position 2 with 6% frac) would no longer be guaranteed to appear — must test carefully to avoid regressions.

---

## Implementation Changelog

| Date | What Changed | Files Modified |
|------|-------------|----------------|
| 2026-05-02 | Literature survey: 9 new papers; RUDY, SA, WireMask sweep, staged noise ideas | `PAPERS_NOTES.md` |
| 2026-04-08 | WireMask-BBO wire-pull + MaskRegulate density-gradient restarts | `placer.py` |
| 2026-04-08 | This notes file created | `PAPERS_NOTES.md` |
| 2026-04-07 | Adaptive time budget + macro threshold (n>350 use density fallback) | `placer.py` |
| 2026-04-07 | Multi-restart legalization with exact proxy scoring | `placer.py` |
| 2026-04-07 | Team roles + paper summary | `TEAM_GUIDE.md` |

---

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
system/ml_placer/
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
| **DREAMPlace** (DAC 2019) | Now integrated through `src/dreamplace_bridge/`: TILOS/ICCAD04 `.pb.txt` + `.plc` are converted to Bookshelf, DREAMPlace runs as an async subprocess, and outputs are exact-scored as ordinary candidates. It is not the final accept objective. |
| **RePlAce** (TCAD 2019) | This IS the competition baseline (avg 1.4578). We're trying to BEAT it, not use it. Understanding its density model is the research goal. |
| **GiFt** (ICCAD 2024) | Only useful as DREAMPlace warm-start; depends on DREAMPlace integration first. |

---

## NEW: 2026-05-03 Survey — Google Circuit Training & TILOS SA Follow-up

### J: Google Circuit Training (Nature 2021) — Key Insights for v18

**Title:** "A graph placement methodology for fast chip design"
**Venue:** Nature 591, 2021
**Authors:** Mirhoseini et al. (Google Brain)

**What they do**: Train a GNN-based RL policy (PPO) to sequentially place macros on a
canvas grid. The policy learns to minimize the proxy cost = WL + 0.5×density + 0.5×congestion
— exactly our objective function. Their proxy IS our proxy cost.

**Insights for our placer:**
1. **Proxy cost formula**: Confirmed that their exact formula matches ours (weights: WL=1.0,
   density=0.5, congestion=0.5). Our compute_proxy_cost is a faithful implementation.
2. **Sequential placement matters**: They place macros one-by-one in connectivity order
   (highest-connectivity first). This is like our legalization ORDER — we use largest-area first.
   **Trying connectivity-order legalization** might be a quick win.
3. **Canvas gridding**: They use a fixed grid (typically 32×32 or 64×64). All placements snap
   to grid centers. Our legalization already does this effectively (spiral search on discrete grid).
4. **Congestion smoothing**: Their congestion uses SMOOTH_RANGE=2 (we see this in our plc data).
   The smoothing means congestion is spread over a 5×5 cell neighborhood, making the signal smoother.

**Key actionable idea**: Try legalization in **connectivity order** (place macros with most nets first).
This naturally keeps highly-connected macros near their net partners, reducing WL without SA.
Implementation: 1 line change in `_will_legalize` — change the sort key from area to net_degree.

### K: TILOS SA + "Go-With-The-Winners" (TCAD 2024 / IEEE 11300304)

**What they found**: Carefully implemented SA with 5 move operators beats Circuit Training RL.
The key differentiator: "Go-with-the-winners" (GWTW) allocates more iterations to SA chains
that are performing well, avoiding wasted effort on stuck chains.

**Insights for v18:**
- **Connectivity-order placement** (same as Google's insight): their SA uses connectivity
  order for the initial placement; we can adopt this for legalization order.
- **Move schedule**: 24% swap, 24% shift, 17% scale, 17% rotate, 18% wiggle.
  Our Phase 4 only does SWAP. SHIFT (move one macro by delta) is essentially what noise
  restarts do, but globally. A LOCAL shift (move 1 macro only) might be more targeted.
- **Per-macro greedy move**: after finding a good placement, try moving EACH macro to the
  best alternative position one at a time (like a greedy descent). This is O(n) swaps from
  best_pl, each targeting a specific macro.

**Actionable (easy, 2h)**: Add "targeted single-macro shift" to Phase 4:
For each macro i in descending congestion order:
  Try moving macro i to nearest lower-congestion cell, re-legalize, score.
  Keep if better. This is a focused greedy descent from best_pl.

### L: UCSD Min-Displacement Legalization Paper (Conferences/396)

The UCSD c396 paper appears to be on FPGA macro placement or min-cost flow assignment.
Key technique: ILP/cost-flow to optimally assign macros to regions before legalization.

**Insight for us**: Our legalization is greedy (largest-first spiral). An ILP pre-assignment
step could reduce total displacement by globally optimizing the macro-to-region assignment.
This is complex but the concept is: instead of greedy sequential placement, solve a small
bipartite matching (macros → grid regions) to minimize expected displacement, then legalize
each macro within its assigned region.

**Feasibility**: Medium-hard. Would need scipy.optimize or a custom solver. The benefit
over greedy legalization is unclear for our use case (we're already doing many restarts).

## Implementation Changelog

| Date | What Changed | Files Modified |
|------|-------------|----------------|
| 2026-05-03 | v17: parallel scoring workers (N workers, own PlacementCost each) | `placer.py` |
| 2026-05-03 | v16: Phase 4 macro-swap + PHASE4_RESERVE_S time split | `placer.py` |
| 2026-05-03 | New test scripts for all remaining TBD benchmarks (ibm02-09) | `scripts/` |
| 2026-05-02 | Literature survey 9 papers; RUDY/GWTW/swap identified as next | `PAPERS_NOTES.md` |
| 2026-05-02 | v15: 3300s budget, raised thresholds, 395 fracs, SKIP_EXACT empty | `placer.py` |
| 2026-04-08 | WireMask-BBO wire-pull + MaskRegulate density-gradient restarts | `placer.py` |
| 2026-04-08 | This notes file created | `PAPERS_NOTES.md` |
| 2026-04-07 | Adaptive time budget + macro threshold (n>350 use density fallback) | `placer.py` |
| 2026-04-07 | Multi-restart legalization with exact proxy scoring | `placer.py` |
| 2026-04-07 | Team roles + paper summary | `TEAM_GUIDE.md` |

---

## How to Test the New Directed Restarts

Historical proxy-path note: directed restarts are not part of the current
hierarchy-only production system. Use this block only when intentionally
studying the archived April/May restart experiments.

```bash
# Quick test on ibm01 (fast, ~30s)
uv run evaluate src/main.py -b ibm10

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
uv run evaluate src/main.py --all
```

---

## Next Research Priorities

1. **Test directed restarts**: Run full 17-benchmark eval, compare to 1.49 average.
2. **Noise sweep on ibm01**: Try density-grad at frac=0.02, 0.04, 0.06, 0.08 to find the best value.
3. **ChiPFormer VGAE**: Implement circuit embeddings to predict per-benchmark optimal parameters.
4. **MaskRegulate regularity metric**: Add the size-weighted centering component (not just occupancy density).
5. **DREAMPlace bridge**: Convert `.pb.txt` → Bookshelf for GPU-accelerated global placement.
