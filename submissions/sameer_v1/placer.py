"""
Competitive Macro Placer -- Partcl/HRT Challenge 2026
Sameer Suleman (sameersul)

Algorithm:
  Multi-restart legalization with DIRECTED perturbations guided by research papers.

  1. Legalize from initial.plc positions (baseline).
  2. Run density-gradient perturbation (MaskRegulate, NeurIPS 2024):
     Build a macro-occupancy heatmap, push macros from crowded zones
     toward empty zones before re-legalizing.
  3. Fill remaining time budget with random Gaussian restarts.
  4. Score all candidates with the EXACT proxy evaluator; return best.

Paper contributions implemented:
  - MaskRegulate (Chen et al., LAMDA/NeurIPS 2024): occupancy density gradient
    → _congestion_heatmap() + _density_gradient_perturb()
  - WireMask-BBO wire-pull code retained but not used in restart schedule
    (0/3 wins on ibm01/03/08; keeps budget for random restarts instead)
  - Random restarts (baseline): unchanged from sameer_v1 original

Why restarts beat SA:
  SA over-optimises WL at the cost of congestion. Restarts explore
  different legalization arrangements without destroying the good spread
  already present in initial.plc.

Proxy cost breakdown (why congestion is the target):
  Proxy = 1×WL + 0.5×density + 0.5×congestion
  WL ≈ 0.06 (tiny), congestion ≈ 2.0 (dominant, about 30× larger).
  All directed perturbations target congestion, not WL.

Baselines:
  will_seed avg:           1.5338
  sameer_v1 (leg-only):    1.5062
  sameer_v1 v4 (restarts): ~1.501 (9 exact benchmarks + 8 baseline-only)
  RePlAce avg:             1.4578  ← target to beat

v5 change: budget-filling restarts. noise_fracs extended to 35 entries. Budget check
in _try_restart terminates early for slow benchmarks (ibm08: ~4 restarts, unchanged).
Fast benchmarks use their full budget: ibm01 (~5s/score) gets ~20 restarts vs 4 before.

v15 change: exploit full 1-hour competition budget (was self-limited to 200s).
  time_budget_s = 3300s (55 min). With 3300s, all benchmarks get many more restarts:
    ibm01 (~6s/score): ~300 restarts (vs 13 before) → much better chance of < 1.1854
    ibm08 (~47s/score): ~55 restarts + cong-grad now runs (was pre-check skipped)
    ibm11 (~81s/score): ~36 restarts (vs 1 before, SKIP_EXACT removed)
    ibm15 (~164s/score): ~18 restarts (grid limit raised to 2200)
    ibm18 (~220s/score): ~14 restarts (grid limit raised to 2200)
  noise_fracs extended to 395 entries (10× cycling of 0.06-dominant pattern).

v16 change: Phase 4 macro-swap exploration (TILOS SA Assessment, TCAD 2024).
  After noise loop, if budget remains, explore best_pl neighbourhood using
  macro-swap moves: exchange positions of 1-3 random macro pairs and re-legalize.
  Uses rng_swap (RandomState(seed+2)) - completely separate from main rng state;
  noise draws for ibm01/ibm08 winning fracs are unchanged.
  Impact:
    ibm01 (~6s/score): noise loop takes ~300×6=1800s → ~250s left → ~35 swap iterations
    ibm09 (~20s/score): noise loop takes ~150×20=3000s → ~300s left → ~7 swap iterations
    ibm08 (~47s/score): noise loop exhausts budget → Phase 4 gets 0 iterations (expected)
  Swap schedule: 1-swap (pure), 2-swap (pure), 1-swap+noise (1%), cycling through 960 entries.

v17 change: Parallel scoring workers for noise-restart phase.
  MacroPlacer(n_workers=N) spawns N worker processes, each with its own PlacementCost
  instance. Legalization stays serial in main process (no plc dependency).
  Workers run compute_proxy_cost in parallel, overlapping with next legalization.
  Effective throughput = min(1/t_leg, N/t_score) per second.
  For ibm08 (t_leg≈5s, t_score≈43s): N=4 → 0.093/s → 261 restarts vs 58 serial.
  For ibm01 (t_leg≈5s, t_score≈9s): N=4 → 0.20/s → 561 restarts vs 199 serial.
  Default n_workers=0: auto-detect = min(8, cpu_count//2).
  n_workers=1: serial mode (same as v16; use for debugging or memory-constrained envs).

v18 change: Legalization order diversity + parallel worker timeout fix.
  Bug fix: parallel worker timeout raised from t_one_score*5 → max(t_one_score*20, 600s).
    The old timeout caused workers to time out on loaded machines where scoring takes
    much longer than the cold baseline measurement (e.g. ibm15: 34s cold → 411s under load).
  New: v18 order diversity.
    For noise-frac indices in the extension zone (>= 35), every 6th cycle uses alternative
    legalization orders instead of default largest-area-first:
      ext_idx % 6 == 0: random permutation (most diverse exploration)
      ext_idx % 6 == 3: connectivity order (most-connected macro placed first)
      Others: default largest-area-first
    Different orders resolve overlap conflicts differently → genuinely different legal
    arrangements from the same perturbed starting positions.
    Critical invariant: indices 0-34 (core 35 fracs) ALWAYS use default order, preserving
    all known winning draws (ibm01 6% at index 2; ibm08 best at index 42; ibm11 at 31).
    Impact: ibm13 (47 extension restarts with default order, all worse than baseline) may
    now find improvements via ~8 order-diverse restarts (4 random + 4 connectivity).
  Cong-grad phases (1-3) and Phase 4 swaps remain serial (need main plc state).
"""

import itertools
import multiprocessing as mp
import os
import random
import time
from pathlib import Path
from typing import List, Optional, Tuple  # noqa: F401

import numpy as np
import torch
from macro_place.benchmark import Benchmark


def _log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Parallel scoring worker (module-level so it's picklable by multiprocessing)
# ---------------------------------------------------------------------------

# Per-worker state: set once by _parallel_worker_init, reused across calls.
_worker_plc = None
_worker_bm = None


def _parallel_worker_init(benchmark_dir: str) -> None:
    """
    Initializer for each worker process in the parallel restart pool.
    Loads the benchmark and PlacementCost object once per worker process,
    avoiding repeated file I/O for every restart.
    """
    global _worker_plc, _worker_bm
    from macro_place.loader import load_benchmark_from_dir
    _worker_bm, _worker_plc = load_benchmark_from_dir(benchmark_dir)


def _parallel_score_worker(args) -> float:
    """
    Score a legalized placement in a worker process.
    Uses the per-process _worker_plc and _worker_bm loaded by _parallel_worker_init.
    Returns proxy_cost (float). Returns 1e9 on error.
    """
    global _worker_plc, _worker_bm
    try:
        leg_flat, n = args
        import torch  # noqa - re-import for worker process
        from macro_place.objective import compute_proxy_cost  # noqa

        leg = np.array(leg_flat, dtype=np.float64).reshape(-1, 2)
        pl = _worker_bm.macro_positions.clone()
        pl[:n, 0] = torch.tensor(leg[:, 0], dtype=torch.float32)
        pl[:n, 1] = torch.tensor(leg[:, 1], dtype=torch.float32)
        costs = compute_proxy_cost(pl, _worker_bm, _worker_plc)
        return float(costs["proxy_cost"])
    except Exception:
        return 1e9


# ---------------------------------------------------------------------------
# Will's minimum-displacement legalization (unchanged)
# ---------------------------------------------------------------------------

def _will_legalize(
    pos: np.ndarray,
    movable: np.ndarray,
    sizes: np.ndarray,
    hw: np.ndarray,
    hh: np.ndarray,
    cw: float,
    ch: float,
    n: int,
    deadline: float = None,
    order: list = None,
) -> np.ndarray:
    """
    Min-displacement legalization with configurable macro placement order.
    Macros are placed one by one at the nearest overlap-free position to their
    target, found by expanding spiral search. Non-movable macros are fixed first.

    order: list of macro indices defining placement sequence. Default (None)
    uses largest-area-first. Different orders explore different legal arrangements.
    deadline: optional wall-clock time.time() value; remaining macros keep pos[].
    """
    sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2
    sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2
    if order is None:
        order = sorted(range(n), key=lambda i: -(sizes[i, 0] * sizes[i, 1]))
    placed = np.zeros(n, dtype=bool)
    legal = pos.copy()
    for idx in order:
        if deadline is not None and time.time() > deadline:
            break
        if not movable[idx]:
            placed[idx] = True
            continue
        if placed.any():
            cdx = np.abs(legal[idx, 0] - legal[:, 0])
            cdy = np.abs(legal[idx, 1] - legal[:, 1])
            conf = (cdx < sep_x[idx] + 0.05) & (cdy < sep_y[idx] + 0.05) & placed
            conf[idx] = False
            if not conf.any():
                placed[idx] = True
                continue
        step = max(sizes[idx, 0], sizes[idx, 1]) * 0.25
        best = legal[idx].copy()
        best_d = float("inf")
        for r in range(1, 200):
            found = False
            for ddx in range(-r, r + 1):
                for ddy in range(-r, r + 1):
                    if abs(ddx) != r and abs(ddy) != r:
                        continue
                    cx = float(np.clip(pos[idx, 0] + ddx * step, hw[idx], cw - hw[idx]))
                    cy = float(np.clip(pos[idx, 1] + ddy * step, hh[idx], ch - hh[idx]))
                    if placed.any():
                        dd = np.abs(cx - legal[:, 0])
                        de = np.abs(cy - legal[:, 1])
                        conf2 = (dd < sep_x[idx] + 0.05) & (de < sep_y[idx] + 0.05) & placed
                        conf2[idx] = False
                        if conf2.any():
                            continue
                    d = (cx - pos[idx, 0]) ** 2 + (cy - pos[idx, 1]) ** 2
                    if d < best_d:
                        best_d, best = d, np.array([cx, cy])
                        found = True
            if found:
                break
        legal[idx] = best
        placed[idx] = True
    return legal


# ---------------------------------------------------------------------------
# Scoring utilities
# ---------------------------------------------------------------------------

def _load_plc(name: str):
    """Load PlacementCost object for exact proxy scoring (uses posix paths for Windows compat)."""
    try:
        from macro_place.loader import load_benchmark_from_dir, load_benchmark
        root = Path("external/MacroPlacement/Testcases/ICCAD04") / name
        if root.exists():
            _, plc = load_benchmark_from_dir(root.as_posix())
            return plc
        # ng45 fallback
        ng45 = {
            "ariane133_ng45": "ariane133",
            "ariane136_ng45": "ariane136",
            "nvdla_ng45": "nvdla",
            "mempool_tile_ng45": "mempool_tile",
        }
        d = ng45.get(name)
        if d:
            base = (Path("external/MacroPlacement/Flows/NanGate45")
                    / d / "netlist" / "output_CT_Grouping")
            if (base / "netlist.pb.txt").exists():
                _, plc = load_benchmark(
                    (base / "netlist.pb.txt").as_posix(),
                    (base / "initial.plc").as_posix(),
                )
                return plc
    except Exception as exc:
        _log(f"  Warning: plc load failed ({exc})")
    return None


def _exact_proxy(placement: torch.Tensor, benchmark: Benchmark, plc) -> float:
    """Return proxy_cost using the ground-truth PlacementCost evaluator."""
    from macro_place.objective import compute_proxy_cost
    costs = compute_proxy_cost(placement, benchmark, plc)
    return float(costs["proxy_cost"])


def _density_score(pos: np.ndarray, n: int, cw: float, ch: float) -> float:
    """Fallback scorer: sum-of-squares macro count on a 20×20 grid (lower = better spread)."""
    G = 20
    grid = np.zeros((G, G), dtype=np.float64)
    for i in range(n):
        r = min(int(pos[i, 1] / (ch / G)), G - 1)
        c = min(int(pos[i, 0] / (cw / G)), G - 1)
        grid[r, c] += 1
    return float((grid ** 2).sum())


# ---------------------------------------------------------------------------
# Directed perturbation: WireMask-BBO (NeurIPS 2023)
# ---------------------------------------------------------------------------

def _compute_wire_pull(pos: np.ndarray, benchmark: Benchmark, n: int) -> np.ndarray:
    """
    Wire-pull vector field, adapted from WireMask-BBO (Gu et al., NeurIPS 2023).

    WireMask-BBO's greedy evaluator places each macro at the grid cell that
    minimises the HPWL increase over all its connected nets (the 'wire mask').
    We approximate this as a continuous pull vector: for each macro i, we sum
    the displacement vectors from i toward the centroid of each net it belongs
    to, weighted by net weight. The result is the direction that most reduces
    total HPWL if macro i moves there.

    Only hard macros (indices < n) are used as net participants. Ports are not
    perturbed, so omitting them makes the pull slightly conservative but keeps
    the code compatible across all benchmarks.

    Returns
    -------
    pull : np.ndarray [n, 2]
        Unnormalized pull displacement per hard macro.
        Larger magnitude means the move matters more. Used in _wire_pull_perturb.
    """
    pull = np.zeros((n, 2), dtype=np.float64)

    for net_idx, nodes in enumerate(benchmark.net_nodes):
        weight = float(benchmark.net_weights[net_idx])
        nodes_np = nodes.numpy()
        # Hard macros only (indices 0..n-1)
        hard_nodes = [int(nd) for nd in nodes_np if nd < n]
        if len(hard_nodes) < 2:
            continue

        net_pos = pos[hard_nodes]           # [k, 2]
        centroid = net_pos.mean(axis=0)     # [2], HPWL midpoint approximation

        for nd in hard_nodes:
            pull[nd] += weight * (centroid - pos[nd])

    return pull


def _wire_pull_perturb(
    init_pos: np.ndarray,
    leg_pos: np.ndarray,
    benchmark: Benchmark,
    movable: np.ndarray,
    n: int,
    cw: float,
    ch: float,
    hw: np.ndarray,
    hh: np.ndarray,
    frac: float,
) -> np.ndarray:
    """
    Wire-pull directed perturbation (WireMask-BBO inspired).

    Computes per-macro wire-pull vectors from the baseline legalization,
    then applies them (capped at `frac × min(canvas)`) to init_pos.
    Re-legalizing from these shifted positions tends to produce placements
    where connected macros are closer together, which reduces HPWL at the
    cost of a slight density increase. Most useful for small benchmarks with
    dense netlists where wirelength is a meaningful fraction of the proxy score.

    Parameters
    ----------
    init_pos : initial.plc positions (before legalization). These get perturbed.
    leg_pos  : baseline legalized positions. Used only to compute pull vectors.
    frac     : displacement cap as a fraction of min(cw, ch).
    """
    magnitude = frac * min(cw, ch)
    pull = _compute_wire_pull(leg_pos, benchmark, n)

    perturbed = init_pos.copy()
    for i in range(n):
        if not movable[i]:
            continue
        dx, dy = pull[i, 0], pull[i, 1]
        norm = np.sqrt(dx ** 2 + dy ** 2) + 1e-10
        # Clamp: never displace more than `magnitude`
        scale = min(magnitude / norm, 1.0)
        perturbed[i, 0] = np.clip(init_pos[i, 0] + scale * dx, hw[i], cw - hw[i])
        perturbed[i, 1] = np.clip(init_pos[i, 1] + scale * dy, hh[i], ch - hh[i])

    return perturbed


# ---------------------------------------------------------------------------
# Directed perturbation: MaskRegulate regularity (NeurIPS 2024)
# ---------------------------------------------------------------------------

def _congestion_heatmap(pos: np.ndarray, n: int, cw: float, ch: float, G: int = 20) -> np.ndarray:
    """
    Macro occupancy density grid, used as the basis for MaskRegulate's regularity mask.

    MaskRegulate (Chen et al., NeurIPS 2024) defines a regularity metric
    that penalizes macros placed far from a balanced, spread-out distribution.
    Their get_regular_mask() returns a per-cell cost that guides the RL policy
    away from overcrowded zones.

    We build the same signal without any RL: a G×G grid where each cell stores
    the number of macro centers inside it. High cell counts signal congestion risk.

    Returns
    -------
    grid : np.ndarray [G, G]
    """
    grid = np.zeros((G, G), dtype=np.float64)
    cw_g = cw / G
    ch_g = ch / G
    for i in range(n):
        c = min(int(pos[i, 0] / cw_g), G - 1)
        r = min(int(pos[i, 1] / ch_g), G - 1)
        grid[r, c] += 1.0
    return grid


def _box_blur(grid: np.ndarray, G: int) -> np.ndarray:
    """
    3×3 box blur using pure NumPy (no scipy dependency).
    Equivalent to one step of Gaussian blur with σ≈0.85.
    Running this 3× approximates σ≈1.5 Gaussian.
    """
    padded = np.pad(grid, 1, mode="edge")
    result = np.zeros_like(grid)
    for dr in range(3):
        for dc in range(3):
            result += padded[dr: dr + G, dc: dc + G]
    return result / 9.0


def _density_gradient_perturb(
    init_pos: np.ndarray,
    leg_pos: np.ndarray,
    movable: np.ndarray,
    n: int,
    cw: float,
    ch: float,
    hw: np.ndarray,
    hh: np.ndarray,
    frac: float,
    G: int = 20,
) -> np.ndarray:
    """
    Congestion-aware directed perturbation (MaskRegulate regularity inspired).

    MaskRegulate's get_regular_mask() penalizes each candidate cell based on
    how overcrowded it is compared to the chip average. Their RL policy learns
    to avoid crowded cells. We replicate this idea without RL, using 5 steps:

      1. Build a G×G macro occupancy grid from the baseline legalization.
      2. Apply 3 passes of box blur to smooth out sharp grid-cell boundaries.
      3. Compute the negative density gradient using finite differences.
         This points from each cell toward its nearest lower-density neighbor.
      4. For each movable hard macro, look up its cell's gradient direction
         and shift init_pos by `frac × min(cw, ch)` in that direction.
      5. Re-legalizing from these shifted positions spreads macros more evenly,
         which reduces both the density and congestion components of proxy cost.

    Parameters
    ----------
    init_pos : initial.plc positions. These get shifted for re-legalization.
    leg_pos  : baseline legalized positions. Used only to build the heatmap.
    frac     : shift magnitude as a fraction of min(cw, ch).
    G        : grid resolution (default 20 x 20).
    """
    magnitude = frac * min(cw, ch)
    cell_w = cw / G
    cell_h = ch / G

    # Build and smooth the occupancy grid
    grid = _congestion_heatmap(leg_pos, n, cw, ch, G)
    smooth = grid.copy()
    for _ in range(3):
        smooth = _box_blur(smooth, G)

    # Negative density gradient: points toward lower density
    grad_x = np.zeros((G, G))
    grad_y = np.zeros((G, G))
    for r in range(G):
        for c in range(G):
            left  = smooth[r, c - 1] if c > 0    else smooth[r, c]
            right = smooth[r, c + 1] if c < G-1  else smooth[r, c]
            grad_x[r, c] = -(right - left) / 2.0   # negative = toward lower density

            down = smooth[r - 1, c] if r > 0    else smooth[r, c]
            up   = smooth[r + 1, c] if r < G-1  else smooth[r, c]
            grad_y[r, c] = -(up - down) / 2.0

    perturbed = init_pos.copy()
    for i in range(n):
        if not movable[i]:
            continue
        c_idx = min(int(leg_pos[i, 0] / cell_w), G - 1)
        r_idx = min(int(leg_pos[i, 1] / cell_h), G - 1)
        dx = grad_x[r_idx, c_idx]
        dy = grad_y[r_idx, c_idx]
        norm = np.sqrt(dx ** 2 + dy ** 2) + 1e-10
        perturbed[i, 0] = np.clip(init_pos[i, 0] + magnitude * dx / norm, hw[i], cw - hw[i])
        perturbed[i, 1] = np.clip(init_pos[i, 1] + magnitude * dy / norm, hh[i], ch - hh[i])

    return perturbed


def _macro_swap_perturb(
    pos: np.ndarray,
    movable: np.ndarray,
    n: int,
    rng: "np.random.RandomState",
    n_swaps: int = 1,
    noise_frac: float = 0.0,
    cw: float = 1.0,
    ch: float = 1.0,
    hw: np.ndarray = None,
    hh: np.ndarray = None,
) -> np.ndarray:
    """
    Swap positions of n_swaps random pairs of movable macros, then optionally
    add tiny Gaussian noise. Returns new init-style position array.

    Swap move (from TILOS SA Assessment, TCAD 2024):
      Exchange positions of two macros. When applied to the best-known legalized
      placement and re-legalized, explores the space of macro orderings without
      the full randomness of Gaussian noise restarts. A swap is the cheapest
      meaningful structural change in a floorplan - it keeps the overall density
      distribution but re-assigns which macro occupies which region.

    Use cases:
      - After noise restarts have converged to a local minimum: swap pairs of
        macros to escape the current topology.
      - After finding a new best placement: try nearby swap variants to refine.

    Parameters
    ----------
    pos       : legalized or init positions [n, 2]
    movable   : boolean mask of movable macros
    n_swaps   : number of random swaps to apply (default 1)
    noise_frac: additional Gaussian noise fraction applied after swap (0 = pure swap)
    """
    movable_idx = [i for i in range(n) if movable[i]]
    if len(movable_idx) < 2:
        return pos.copy()

    perturbed = pos.copy()
    for _ in range(n_swaps):
        if len(movable_idx) < 2:
            break
        pair = rng.choice(len(movable_idx), 2, replace=False)
        i, j = movable_idx[pair[0]], movable_idx[pair[1]]
        perturbed[i], perturbed[j] = pos[j].copy(), pos[i].copy()

    if noise_frac > 0 and hw is not None:
        scale = noise_frac * min(cw, ch)
        noise = rng.normal(0, scale, perturbed.shape)
        lo = np.stack([hw, hh], axis=1)
        hi = np.stack([cw - hw, ch - hh], axis=1)
        perturbed = np.clip(perturbed + noise, lo, hi)

    return perturbed


def _routing_congestion_perturb(
    pos: np.ndarray,
    plc,
    benchmark: "Benchmark",
    n: int,
    cw: float,
    ch: float,
    hw: np.ndarray,
    hh: np.ndarray,
    movable: np.ndarray,
    frac: float = 0.04,
    rng: "np.random.RandomState" = None,
) -> np.ndarray:
    """
    Move macros away from high routing-congestion cells using the ACTUAL
    routing congestion map stored in plc after the last get_congestion_cost() call.

    Unlike _density_gradient_perturb (which uses a macro-count occupancy proxy),
    this uses the real H/V routing congestion computed by PlacementCost.

    Gradient: for each macro in a congested cell, compute finite-difference
    gradient of congestion w.r.t. position, then move the macro AGAINST the
    gradient (toward lower congestion). A small random component breaks symmetry.

    Uses a separate rng (not np.random) so the main random state is unchanged
    and subsequent noise restarts get identical draws to before.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    nr, nc_grid = benchmark.grid_rows, benchmark.grid_cols
    try:
        h_list = list(plc.get_horizontal_routing_congestion())
        v_list = list(plc.get_vertical_routing_congestion())
    except Exception:
        return pos.copy()

    if len(h_list) != nr * nc_grid:
        return pos.copy()

    h_cong = np.array(h_list).reshape(nr, nc_grid)
    v_cong = np.array(v_list).reshape(nr, nc_grid)
    cell_cong = np.maximum(h_cong, v_cong)

    cell_w = cw / nc_grid
    cell_h = ch / nr
    scale = frac * min(cw, ch)
    cong_threshold = 0.5  # only perturb macros in cells with congestion > threshold

    perturbed = pos.copy()
    for i in range(n):
        if not movable[i]:
            continue
        c_idx = int(min(pos[i, 0] / cell_w, nc_grid - 1))
        r_idx = int(min(pos[i, 1] / cell_h, nr - 1))
        local_cong = float(cell_cong[r_idx, c_idx])

        if local_cong < cong_threshold:
            continue

        # Finite-difference gradient of congestion (pointing toward HIGHER congestion)
        # We move AGAINST it (toward lower congestion)
        def cong(r, c):
            return cell_cong[max(0, min(r, nr - 1)), max(0, min(c, nc_grid - 1))]

        grad_x = (cong(r_idx, c_idx + 1) - cong(r_idx, c_idx - 1)) / 2.0
        grad_y = (cong(r_idx + 1, c_idx) - cong(r_idx - 1, c_idx)) / 2.0
        grad_len = np.sqrt(grad_x ** 2 + grad_y ** 2) + 1e-10

        # Move against gradient, scaled by congestion level
        move_scale = scale * min(local_cong, 2.0)
        dx = -(grad_x / grad_len) * move_scale + rng.normal(0, scale * 0.1)
        dy = -(grad_y / grad_len) * move_scale + rng.normal(0, scale * 0.1)

        perturbed[i, 0] = float(np.clip(pos[i, 0] + dx, hw[i], cw - hw[i]))
        perturbed[i, 1] = float(np.clip(pos[i, 1] + dy, hh[i], ch - hh[i]))

    return perturbed


# ---------------------------------------------------------------------------
# Main placer
# ---------------------------------------------------------------------------

class MacroPlacer:
    """
    Multi-restart legalization placer with directed perturbations.

    Restart schedule (subject to adaptive time budget):
      0  Baseline        - legalize directly from initial.plc, no perturbation
      1  Density-grad    - MaskRegulate: push macros out of crowded zones
      2+ Random Gaussian - small random perturbations (original strategy)

    Parameters
    ----------
    n_restarts : int
        Total candidates including baseline (restarts = n_restarts - 1).
    noise_fracs : list[float]
        Magnitudes for random restarts (fraction of min canvas dimension).
    seed : int
        Random seed for reproducibility.
    time_budget_s : float
        Per-benchmark wall-clock budget; restarts skipped when insufficient.
    """

    def __init__(
        self,
        n_restarts: int = 5000,
        noise_fracs: Optional[List[float]] = None,
        seed: int = 42,
        time_budget_s: float = 3300.0,
        n_workers: int = 0,
    ):
        """
        n_workers: number of parallel scoring workers.
          0 (default) = auto-detect: min(4, cpu_count//2), max 8.
          1 = fully serial (same behaviour as v16 and earlier).
          N > 1 = N parallel workers each with own PlacementCost instance.

        Parallel scoring speedup (purely for the noise-restart loop):
          - Legalization runs serially in main process (fast: ~5s).
          - Completed legalizations are submitted to workers for scoring.
          - Workers run compute_proxy_cost in parallel (slow: 9-220s).
          - Effective throughput ~= min(legalisation_rate, N×score_rate).
          For ibm08 (score=43s/each): 4 workers → ~4x more restarts.
          For ibm01 (score=9s/each): 4 workers → ~3x more restarts.
        """
        self.n_restarts = n_restarts
        # Budget check in _try_restart terminates the loop early; n_restarts is an upper cap.
        # First 35 entries are "core" - RNG draw positions preserved so all v14 wins are
        # unchanged (ibm01 6%-win at position 2, ibm03 2%-win at position 0, ibm08 6%-win
        # at position 2). Entries 35-394 extend the draw space for the 1-hour budget:
        #   ibm01 (~6s/score, budget=3300s): ~300 restarts → tries 30+ draws at 0.06
        #   ibm08 (~47s/score, cong-grad now runs): ~55 restarts
        #   ibm11 (~81s/score, SKIP_EXACT removed): ~36 restarts
        # Extension pattern: 30-entry cycle emphasizing 0.06 (ibm01/ibm08 winner):
        #   30% of entries at 0.06, 17% at 0.04, 13% at 0.02, 10% at 0.08, rest varied.
        _ext = [
            0.06, 0.04, 0.02, 0.08, 0.06, 0.03, 0.05, 0.01, 0.06, 0.07,
            0.04, 0.06, 0.09, 0.02, 0.06, 0.05, 0.08, 0.04, 0.06, 0.02,
            0.06, 0.10, 0.04, 0.06, 0.02, 0.06, 0.12, 0.04, 0.06, 0.08,
        ] * 12  # 360 entries (30 × 12)
        # Large-frac extension (indices 395-429): tried only by fast benchmarks on EPYC
        # (ibm01 ~9s/score, n_workers=8 → ~561 restarts → reaches index 395+).
        # Large perturbations (25-50%) produce genuinely different macro arrangements -
        # the legalization finds a different basin than small-noise restarts.
        # WL impact of 50% noise: small (WL ≈ 0.06, congestion dominates at 1.3-2.5).
        # For slow benchmarks (ibm08 56 restarts, ibm13 47 restarts): never reached.
        _large = [
            0.25, 0.30, 0.20, 0.40, 0.25, 0.30, 0.50, 0.25, 0.15, 0.35,
            0.25, 0.30, 0.40, 0.20, 0.25, 0.50, 0.30, 0.25, 0.15, 0.20,
            0.25, 0.30, 0.35, 0.25, 0.40, 0.20, 0.50, 0.25, 0.30, 0.15,
            0.25, 0.35, 0.30, 0.20, 0.25,
        ]  # 35 entries (indices 395-429)
        self.noise_fracs = noise_fracs or [
            # ── Core 35 entries (UNCHANGED from v14 - preserves all known wins) ─────
            # Core (preserves ibm01 6%-win and ibm03 2%-win)
            0.02, 0.04, 0.06, 0.08,
            # Fine grid fill: gaps between core points
            0.01, 0.03, 0.05, 0.07, 0.09,
            # Fresh draws at winning scale with advanced random state
            0.06, 0.06, 0.04,
            # Medium exploration
            0.10, 0.12, 0.08,
            # Very fine grid
            0.025, 0.035, 0.045, 0.055, 0.065, 0.075,
            # Larger displacements
            0.15, 0.20, 0.10,
            # Revisit good range with new draws
            0.05, 0.06, 0.07, 0.03, 0.04, 0.02,
            # Even finer
            0.005, 0.010, 0.015, 0.030, 0.050,
            # ── Extension for 1-hour budget (entries 35-394) ─────────────────────
            *_ext,
            # ── Large-frac block (entries 395-429) ───────────────────────────────
            *_large,
        ]
        self.seed = seed
        self.time_budget_s = time_budget_s
        # Resolve n_workers: 0=auto, 1=serial, N>1=parallel
        if n_workers == 0:
            n_cpu = os.cpu_count() or 1
            self.n_workers = max(1, min(8, n_cpu // 2))
        else:
            self.n_workers = max(1, int(n_workers))

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        np.random.seed(self.seed)
        random.seed(self.seed)

        t0 = time.time()
        n = benchmark.num_hard_macros
        cw, ch = benchmark.canvas_width, benchmark.canvas_height
        sizes = benchmark.macro_sizes[:n].numpy().astype(np.float64)
        hw = sizes[:, 0] / 2
        hh = sizes[:, 1] / 2
        movable = (benchmark.get_movable_mask() & benchmark.get_hard_macro_mask())[:n].numpy()
        init_pos = benchmark.macro_positions[:n].numpy().copy().astype(np.float64)

        _log(f"  [{benchmark.name}] hard={n}  movable={movable.sum()}  "
             f"budget={self.time_budget_s:.0f}s")

        # Exact scoring cutoffs (v15: raised for 1-hour budget):
        #   n > EXACT_MACRO_THRESHOLD: scoring too slow (n>430 → ibm16/17/etc. excluded)
        #   grid_cells > EXACT_GRID_CELL_LIMIT: routing grid slow (raised 2000→2200 to include
        #     ibm15 (2166) and ibm18 (2145) which score in 164-220s; fine with 3300s budget)
        # SLOW_SCORE_THRESHOLD=400s: raised from 100s to allow large-grid benchmarks.
        #   ibm15 (164s) and ibm18 (~220s) now pass the threshold and get exact scoring.
        # SKIP_EXACT=empty: with 1-hour budget, ibm11/ibm13 get 36/55 restarts - worth trying.
        #   Previous SKIP_EXACT tested only 2-3 restarts; with 36+ restarts they may improve.
        #   Worst case: baseline still wins after 36 restarts, same quality as SKIP_EXACT.
        EXACT_MACRO_THRESHOLD = 430  # ibm16 (n=458) still excluded; test separately
        EXACT_GRID_CELL_LIMIT = 2200  # ibm15 (2166) and ibm18 (2145) now included
        SKIP_EXACT: set = set()  # empty: 1-hour budget makes all-restarts-worse test affordable
        grid_cells = benchmark.grid_rows * benchmark.grid_cols
        plc = _load_plc(benchmark.name)
        use_exact = (
            (plc is not None)
            and (n <= EXACT_MACRO_THRESHOLD)
            and (grid_cells <= EXACT_GRID_CELL_LIMIT)
            and (benchmark.name not in SKIP_EXACT)
        )
        if plc is None:
            _log("  Warning: plc unavailable, using density-score fallback")
        elif benchmark.name in SKIP_EXACT:
            _log(f"  Skipping exact scoring for {benchmark.name} (isolated tests: all restarts worse than baseline)")
        elif n > EXACT_MACRO_THRESHOLD:
            _log(f"  Large benchmark (n={n} > {EXACT_MACRO_THRESHOLD}), "
                 f"using density fallback to rank restarts (fast)")
        elif grid_cells > EXACT_GRID_CELL_LIMIT:
            _log(f"  Large grid ({benchmark.grid_rows}x{benchmark.grid_cols}={grid_cells} > "
                 f"{EXACT_GRID_CELL_LIMIT} cells), using density fallback (exact would be ~{grid_cells//10:.0f}s)")

        def _score(pos: np.ndarray) -> Tuple[float, torch.Tensor]:
            pl = benchmark.macro_positions.clone()
            pl[:n, 0] = torch.tensor(pos[:, 0], dtype=torch.float32)
            pl[:n, 1] = torch.tensor(pos[:, 1], dtype=torch.float32)
            if use_exact:
                try:
                    return float(_exact_proxy(pl, benchmark, plc)), pl
                except Exception:
                    pass
            return _density_score(pos, n, cw, ch), pl

        # -- Restart 0: Baseline ----------------------------------------------
        _log(f"  Restart 0 (baseline)...")
        t1 = time.time()
        baseline_pos = _will_legalize(init_pos, movable, sizes, hw, hh, cw, ch, n)
        _log(f"    Legalized in {time.time()-t1:.1f}s")

        t_score0 = time.time()
        best_score, best_pl = _score(baseline_pos)
        t_one_score = time.time() - t_score0
        _log(f"  Candidate 0: proxy={best_score:.4f}  (scored in {t_one_score:.1f}s)")

        # Density fallback is anti-correlated with proxy cost: the sum-of-squares macro
        # occupancy metric rewards spread placements, but spread placements empirically have
        # WORSE proxy scores (higher congestion). Full-eval evidence: density fallback hurt
        # ibm10-17 by avg +0.14 per benchmark vs v1 baseline-only (1.6595 vs 1.5220 est.).
        # For any benchmark that cannot use exact scoring, just return the baseline.
        # This covers: (a) large-grid (ibm18: 55x39=2145 cells), (b) large-n (n>340).
        if not use_exact:
            _log(f"  Cannot use exact scoring: returning baseline only "
                 f"(density fallback anti-correlated with proxy)")
            _log(f"  Best proxy={best_score:.4f}  total={time.time()-t0:.1f}s")
            return best_pl

        # Safety net: if exact scoring took longer than budget allows (e.g. CPU load or huge grid).
        # Raised to 400s (from 100s) to allow ibm15 (~164s) and ibm18 (~220s) to use exact scoring.
        # ibm11 (~81s) and ibm13 (~53s) are well under threshold.
        # Anything over 400s implies an unexpectedly large benchmark or extreme CPU load.
        SLOW_SCORE_THRESHOLD_S = 400.0
        if t_one_score > SLOW_SCORE_THRESHOLD_S:
            _log(f"  Exact score slow ({t_one_score:.0f}s); returning baseline")
            _log(f"  Best proxy={best_score:.4f}  total={time.time()-t0:.1f}s")
            return best_pl

        # -- Parallel scoring pool setup -----------------------------------
        # For use_exact benchmarks: spawn N workers, each with its own plc.
        # Workers only do scoring (compute_proxy_cost). Legalization stays
        # in the main process (no plc dependency, just numpy math).
        #
        # Speedup (noise-restart phase only):
        #   effective_rate = min(1/t_leg, n_workers/t_score)
        # For ibm08 (t_leg≈5s, t_score≈43s):
        #   n_workers=4 → min(0.20, 0.093) = 0.093/s → 261 restarts/3300s×85%
        #   vs serial:  1/48 = 0.021/s → 58 restarts/3300s×85%
        #   Gain: ~4.5× more restarts (ibm08: 56 → ~250).
        # For ibm01 (t_leg≈5s, t_score≈9s):
        #   n_workers=4 → min(0.20, 0.44) = 0.20/s → 561 restarts
        #   vs serial: 1/14 = 0.071/s → 199 restarts
        #   Gain: ~2.8× more restarts.
        #
        # Workers are only used for noise restarts.
        # Cong-grad phases (1-3) and Phase 4 swaps use _try_restart (serial)
        # because they need the main-process plc congestion map.
        _pool = None
        _use_parallel = use_exact and self.n_workers > 1
        if _use_parallel:
            # Construct benchmark_dir for worker initialization
            _bench_root = Path("external/MacroPlacement/Testcases/ICCAD04") / benchmark.name
            if _bench_root.exists():
                _bdir_str = _bench_root.as_posix()
                try:
                    ctx = mp.get_context("spawn")
                    _pool = ctx.Pool(
                        processes=self.n_workers,
                        initializer=_parallel_worker_init,
                        initargs=(_bdir_str,),
                    )
                    _log(f"  Parallel pool: {self.n_workers} workers (benchmark={benchmark.name})")
                except Exception as e:
                    _log(f"  Parallel pool failed ({e}); falling back to serial")
                    _pool = None
                    _use_parallel = False
            else:
                _log(f"  Parallel pool: benchmark_dir not found, falling back to serial")
                _use_parallel = False

        def _try_restart(label: str, perturbed_init: np.ndarray, k: int,
                         leg_order=None) -> bool:
            """Legalize + score one candidate. Returns False if budget exhausted.

            leg_order: optional macro placement sequence (list of indices).
              None = default (largest-area-first).
              v18: extension-zone restarts (noise_fracs index >= 35) sometimes use
              a random permutation to explore different legal arrangements from
              the same perturbed starting positions.  Core 35 fracs always use
              the default order so all known winning draws are preserved.
            """
            nonlocal best_score, best_pl
            elapsed = time.time() - t0
            remaining = self.time_budget_s - elapsed
            # t_one_score is the baseline scoring time. Factor 1.3 covers score + legalize.
            # We use the baseline measurement (not a running max) to avoid over-conserving
            # when individual scorings are slightly slower than baseline (load jitter).
            estimated_cost = t_one_score * 1.3
            if remaining < estimated_cost:
                _log(f"  Skipping restart {k}+ (budget: {remaining:.0f}s left, "
                     f"need ~{estimated_cost:.0f}s)")
                return False  # signal: stop further restarts

            t1 = time.time()
            leg_deadline = t1 + 60.0  # cap spiral search; timed-out macros keep pos value
            leg = _will_legalize(perturbed_init, movable, sizes, hw, hh, cw, ch, n,
                                 deadline=leg_deadline, order=leg_order)
            t_leg = time.time() - t1
            ord_tag = "+ord" if leg_order is not None else ""
            _log(f"  Restart {k} ({label}{ord_tag}) legalized in {t_leg:.1f}s")

            score, pl = _score(leg)
            _log(f"  Candidate {k}: proxy={score:.4f}")
            if score < best_score:
                best_score = score
                best_pl = pl

            # Safety: if scoring overran the budget, stop immediately rather than
            # launching another restart that would push total time even further over.
            if time.time() - t0 > self.time_budget_s:
                _log(f"  Over budget after scoring ({time.time()-t0:.0f}s); stopping")
                return False

            return True

        # -- Restart 1: Density-gradient (MaskRegulate) ----------------------
        # Only for small benchmarks (n <= 100): density-grad helped ibm01 (54 macros)
        # but hurt ibm03 (126) and ibm08 (301), and also consumed ~40s per restart,
        # blocking the 6% random noise slot that achieved best ibm08 result in v2.
        directed_ran = 0
        DENSITY_GRAD_MAX_N = 100
        if n <= DENSITY_GRAD_MAX_N:
            density_perturbed = _density_gradient_perturb(
                init_pos, baseline_pos, movable, n, cw, ch, hw, hh, frac=0.04
            )
            if not _try_restart("density-grad frac=4%", density_perturbed, k=1):
                _log(f"  Best proxy={best_score:.4f}  total={time.time()-t0:.1f}s")
                return best_pl
            directed_ran = 1

        # -- Routing-congestion-gradient descent (v8, iterative + wide) --------
        # Phase 1: iterative gradient descent at frac=0.04.
        #   After each improving step, extract the new position from best_pl
        #   and use it (with plc's now-updated congestion map) as the starting
        #   point for the next step. Stops when a step fails to improve or
        #   budget can't fit 3 noise restarts.
        # Phase 2: wide step at frac=0.08 from baseline_pos using current plc.
        #   Only runs if phase 1 improved at least once (otherwise cong-grad
        #   is not useful for this benchmark). Uses rng_cong so main random
        #   state is unchanged and subsequent noise draws are identical to v5.
        # Pre-check: skip cong-grad entirely when budget is too tight.
        #   Threshold: need budget for cong-grad + at least 3 noise restarts.
        #   Motivation: for slow benchmarks (ibm08, t_score≈39s under load),
        #   1 useless cong-grad restart blocks the winning 6% noise frac,
        #   degrading 1.5251 → 1.5539. Skipping cong-grad restores the full
        #   noise sequence. Fast benchmarks (ibm02/03/04/06) have ample budget
        #   and are unaffected. Threshold 4.0 keeps ibm08-clean's cong-grad
        #   running (168s > 4×32×1.3=166s) but drops it under load (161s < 4×39×1.3=203s).
        rng_cong = np.random.RandomState(self.seed + 1)
        cong_pos = baseline_pos
        cong_improved = False
        cong_frac = 0.04
        for cong_iter in range(12):
            if cong_iter == 0:
                _pre_rem = self.time_budget_s - (time.time() - t0)
                _cong_min = 4.0 * t_one_score * 1.3
                if _pre_rem < _cong_min:
                    _log(f"  Cong-grad skipped: budget {_pre_rem:.0f}s < {_cong_min:.0f}s "
                         f"(preserving noise slots)")
                    break
            if cong_iter > 0:
                remaining = self.time_budget_s - (time.time() - t0)
                # Larger factor for full-frac iters (reserve for Phase 2 + noise).
                # Smaller factor for adaptive halved-frac retries (only 1 eval needed).
                budget_factor = 3.0 if cong_frac >= 0.04 else 1.5
                if remaining < budget_factor * t_one_score * 1.3:
                    break
            cong_perturbed = _routing_congestion_perturb(
                cong_pos, plc, benchmark, n, cw, ch, hw, hh, movable,
                frac=cong_frac, rng=rng_cong,
            )
            score_before = best_score
            if not _try_restart(f"cong-grad iter={cong_iter + 1} f={cong_frac:.2f}",
                                 cong_perturbed, k=1 + directed_ran):
                _log(f"  Best proxy={best_score:.4f}  total={time.time()-t0:.1f}s")
                return best_pl
            directed_ran += 1
            if best_score < score_before:
                cong_pos = np.stack(
                    [best_pl[:n, 0].numpy(), best_pl[:n, 1].numpy()], axis=1
                )
                cong_improved = True
                cong_frac = 0.04  # reset frac on success
            elif cong_improved and cong_frac > 0.01 and cong_iter >= 2:
                # At least 2 prior iterations: try a gentler step before giving up.
                # Guard cong_iter>=2 prevents firing after only 1 success (ibm02 pattern):
                # ibm02 fails at cong_iter=1 → stale plc state critical for Phase 2 wide=8%.
                # ibm03/ibm06 fail at cong_iter=2+ → stale plc less critical, adaptive helps.
                cong_frac *= 0.5
            else:
                break  # plc's map is stale, stop iterating

        # Phase 2: wide steps from baseline using evolved plc congestion state.
        # Loop over [0.08, 0.12]; stop when a step fails to improve or budget
        # runs out. Each step uses the gradient from the current plc state
        # (which encodes where prior iterations struggled), applied with a
        # larger displacement from the original baseline spread.
        if cong_improved:
            for wide_frac in [0.08, 0.12]:
                remaining = self.time_budget_s - (time.time() - t0)
                if remaining < t_one_score * 1.3:
                    break
                cong_wide = _routing_congestion_perturb(
                    baseline_pos, plc, benchmark, n, cw, ch, hw, hh, movable,
                    frac=wide_frac, rng=rng_cong,
                )
                score_before = best_score
                if not _try_restart(f"cong-grad wide={wide_frac:.0%}", cong_wide,
                                     k=1 + directed_ran):
                    _log(f"  Best proxy={best_score:.4f}  total={time.time()-t0:.1f}s")
                    return best_pl
                directed_ran += 1
                if best_score >= score_before:
                    break  # stop wide steps if this one didn't improve

        # Phase 3: iterative cong-grad from best known position using current (stale) plc.
        # After Phase 2 failed wide steps, plc holds the cong map from a placement
        # that was WORSE than our best. Moving from the BEST position away from the
        # high-congestion regions of this stale map explores different local minima.
        # Only runs when cong-grad improved at least once (cong_improved).
        # v15: Runs up to 20 times (loop), each using the next rng_cong draw.
        #   With 3300s budget, up to 20 Phase 3 iterations run before noise restarts.
        #   For ibm06 (t=20s): 20 × 26s = 520s before 100 noise restarts.
        #   With 200s budget: usually 0-1 Phase 3 runs (budget exhausted by noise fracs).
        if cong_improved:
            for phase3_i in range(20):
                remaining = self.time_budget_s - (time.time() - t0)
                if remaining < t_one_score * 1.3:
                    break
                best_pos_now = np.stack(
                    [best_pl[:n, 0].numpy(), best_pl[:n, 1].numpy()], axis=1
                )
                phase3_perturbed = _routing_congestion_perturb(
                    best_pos_now, plc, benchmark, n, cw, ch, hw, hh, movable,
                    frac=0.04, rng=rng_cong,
                )
                if not _try_restart(f"cong-grad phase3/{phase3_i+1}", phase3_perturbed,
                                     k=1 + directed_ran):
                    _log(f"  Best proxy={best_score:.4f}  total={time.time()-t0:.1f}s")
                    return best_pl
                directed_ran += 1

        # -- Restarts 1+: Random Gaussian -------------------------------------
        # Reserve 15% of budget for Phase 4 swap exploration.
        # Without this reserve, the noise loop always exhausts the budget before
        # Phase 4 can run (395 fracs × t_score > time_budget_s for all benchmarks).
        # With the reserve: noise loop stops ~15% early → Phase 4 gets:
        #   ibm01 (~9s/score): ~53 swap iters   ibm08 (~43s/score): ~11 swap iters
        #   ibm09 (~20s/score): ~25 swap iters  ibm11 (~81s/score): ~6 swap iters
        PHASE4_RESERVE_S = self.time_budget_s * 0.15
        noise_scale_base = min(cw, ch)
        last_noise_k = 1 + directed_ran

        # v18: Legalization order diversity.
        # For noise-frac indices in the extension zone (>= 35), every Nth restart
        # uses an alternative macro-placement order instead of the default largest-area-first.
        # Different orders resolve conflicts differently, producing genuinely different
        # legal arrangements even from the same perturbed starting positions.
        #
        # Critical invariant: indices 0-34 (core 35 fracs, containing all known winning
        # draws) ALWAYS use the default order (leg_order=None).
        # E.g. ibm01 win: noise_fracs[2]=0.06 (index 2 < 35) → default order, unchanged.
        #      ibm08 win: noise_fracs[42] (index 42 > 35), (42-35)%6=1 → NOT order-diverse.
        #      ibm11 win: noise_fracs[31] (index 31 < 35) → default order, unchanged.
        #
        # Schedule in extension zone (per-6 cycle starting at index 35):
        #   ext_idx % 6 == 0 → random permutation (most diverse exploration)
        #   ext_idx % 6 == 3 → connectivity order (most-connected macro first)
        #   Others → default largest-area-first
        # This gives 1/3 of extension restarts as order-diverse (1/6 random, 1/6 connectivity).
        _order_rng = np.random.RandomState(self.seed + 3)  # separate rng for leg orders
        _core_fracs = 35  # number of core fracs to keep with default order

        # Precompute connectivity order (most-connected macro first).
        # Connectivity = number of nets the macro appears in.
        # Research motivation: placing highly-connected macros first ensures they get their
        # preferred positions in the spiral search, reducing routing bottlenecks.
        _n_conns = np.zeros(n, dtype=np.int32)
        for _net_nodes in benchmark.net_nodes:
            for _nd in _net_nodes.numpy():
                if int(_nd) < n:
                    _n_conns[int(_nd)] += 1
        _connectivity_order = sorted(range(n), key=lambda i: -_n_conns[i])

        _nfracs = len(self.noise_fracs)  # for cycling modular arithmetic

        def _noise_leg_order(noise_idx: int):
            """Return alternative order for extension-zone restarts, None for core-35.
            noise_idx counts absolute restarts (may exceed _nfracs on cycling benchmarks).
            We modulo by _nfracs so cycling benchmarks repeat the same order-diversity
            pattern on every cycle through the noise_fracs list.
            """
            cycle_idx = noise_idx % _nfracs  # position within current frac cycle
            if cycle_idx >= _core_fracs:
                ext_idx = cycle_idx - _core_fracs
                if ext_idx % 6 == 0:
                    return _order_rng.permutation(n).tolist()  # random
                if ext_idx % 6 == 3:
                    return _connectivity_order  # most-connected first
            return None  # default: largest-area-first

        if _use_parallel and _pool is not None:
            # ── Parallel noise restarts ────────────────────────────────────
            # Strategy: main process legalizes candidates serially, immediately
            # submitting each legalization to the worker pool for scoring.
            # Workers run compute_proxy_cost in parallel; main thread overlaps
            # legalization of the NEXT candidate with scoring of the current.
            #
            # We maintain an in-flight queue of (AsyncResult, leg_pos, k, label).
            # When the queue is full (n_workers items in flight), we wait for
            # the oldest result before adding more.
            from multiprocessing.pool import AsyncResult

            in_flight: list = []  # list of (AsyncResult, leg_pos, k, label)
            # v19: cycle through noise_fracs so fast EPYC benchmarks use full budget.
            # noise_fracs has 430 entries; ibm01 on EPYC (8 workers) can do ~2800 restarts.
            # Without cycling, the loop would stop after 430 iterations leaving budget unused.
            noise_fracs_iter = enumerate(
                itertools.islice(
                    itertools.cycle(self.noise_fracs),
                    self.n_restarts - 1 - directed_ran,
                ),
                start=1 + directed_ran
            )

            def _flush_result(async_res: "AsyncResult", leg_pos: np.ndarray,
                              k_r: int, label_r: str) -> bool:
                """Wait for one async scoring result, update best. Returns False if over budget."""
                nonlocal best_score, best_pl, last_noise_k
                try:
                    # Timeout: generous upper bound so we handle machines under load.
                    # t_one_score is measured on a cold/serial baseline; workers under
                    # sustained load may take 4-10× longer. Use max(20×baseline, 600s)
                    # so any legitimate result is returned even if the machine is hot.
                    # A true crash (not just slowness) would never return within 600s
                    # on any realistic hardware, so this is safe.
                    _timeout = max(t_one_score * 20.0, 600.0)
                    score = async_res.get(timeout=_timeout)
                except Exception:
                    score = 1e9
                _log(f"  Candidate {k_r} ({label_r}): proxy={score:.4f}")
                if score < best_score:
                    best_score = score
                    pl2 = benchmark.macro_positions.clone()
                    pl2[:n, 0] = torch.tensor(leg_pos[:, 0], dtype=torch.float32)
                    pl2[:n, 1] = torch.tensor(leg_pos[:, 1], dtype=torch.float32)
                    best_pl = pl2
                    _log(f"    ** New best {score:.4f} at restart {k_r} **")
                last_noise_k = k_r
                if time.time() - t0 > self.time_budget_s:
                    return False
                return True

            stop_submitting = False
            for k, frac in noise_fracs_iter:
                # Budget pre-check (for submission)
                elapsed = time.time() - t0
                noise_remaining = self.time_budget_s - elapsed - PHASE4_RESERVE_S
                if noise_remaining < t_one_score * 1.3:
                    _log(f"  Noise loop done: switching to Phase 4 swaps "
                         f"({PHASE4_RESERVE_S:.0f}s reserved, "
                         f"{self.time_budget_s - elapsed:.0f}s left total)")
                    stop_submitting = True
                    break

                # Flush oldest in-flight result when queue is full
                if len(in_flight) >= self.n_workers:
                    oldest_res, oldest_leg, oldest_k, oldest_lbl = in_flight.pop(0)
                    if not _flush_result(oldest_res, oldest_leg, oldest_k, oldest_lbl):
                        stop_submitting = True
                        break

                # Legalize in main process (fast, no plc)
                noise = np.random.normal(0, frac * noise_scale_base, init_pos.shape)
                perturbed = np.clip(
                    init_pos + noise,
                    np.stack([hw, hh], axis=1),
                    np.stack([cw - hw, ch - hh], axis=1),
                )
                # v18: order diversity for extension-zone fracs (index >= 35)
                noise_idx = k - (1 + directed_ran)
                leg_ord = _noise_leg_order(noise_idx)
                t_leg0 = time.time()
                leg_deadline = t_leg0 + 60.0
                leg = _will_legalize(perturbed, movable, sizes, hw, hh, cw, ch, n,
                                     deadline=leg_deadline, order=leg_ord)
                t_leg = time.time() - t_leg0
                ord_tag = "+ord" if leg_ord is not None else ""
                label_k = f"random noise={frac:.0%}{ord_tag}"
                _log(f"  Restart {k} ({label_k}) legalized in {t_leg:.1f}s")

                # Submit scoring to worker pool (async)
                args_for_worker = (leg.flatten().tolist(), n)
                async_r = _pool.apply_async(_parallel_score_worker, (args_for_worker,))
                in_flight.append((async_r, leg.copy(), k, label_k))

            # Flush all remaining in-flight results
            while in_flight:
                oldest_res, oldest_leg, oldest_k, oldest_lbl = in_flight.pop(0)
                _flush_result(oldest_res, oldest_leg, oldest_k, oldest_lbl)

            # Clean up pool after noise restarts
            try:
                _pool.terminate()
                _pool.join()
            except Exception:
                pass

        else:
            # ── Serial noise restarts (original v16 code) ─────────────────
            # v19: cycle through noise_fracs (budget is the real cap, not frac-list length)
            for k, frac in enumerate(
                itertools.islice(
                    itertools.cycle(self.noise_fracs),
                    self.n_restarts - 1 - directed_ran,
                ),
                start=1 + directed_ran
            ):
                last_noise_k = k
                # Stop noise loop early to reserve time for Phase 4 swap exploration.
                elapsed = time.time() - t0
                noise_remaining = self.time_budget_s - elapsed - PHASE4_RESERVE_S
                if noise_remaining < t_one_score * 1.3:
                    _log(f"  Noise loop done: switching to Phase 4 swaps "
                         f"({PHASE4_RESERVE_S:.0f}s reserved, "
                         f"{self.time_budget_s - elapsed:.0f}s left total)")
                    break
                noise = np.random.normal(0, frac * noise_scale_base, init_pos.shape)
                perturbed = np.clip(
                    init_pos + noise,
                    np.stack([hw, hh], axis=1),
                    np.stack([cw - hw, ch - hh], axis=1),
                )
                # v18: order diversity for extension-zone fracs (index >= 35)
                noise_idx = k - (1 + directed_ran)
                leg_ord = _noise_leg_order(noise_idx)
                if not _try_restart(f"random noise={frac:.0%}", perturbed, k=k,
                                    leg_order=leg_ord):
                    break

        # -- Phase 4: Macro-swap exploration from best position ---------------
        # After the main noise loop, if budget remains, explore the neighbourhood
        # of the best-found placement using macro swap moves (TILOS SA, TCAD 2024).
        #
        # Motivation: Gaussian noise always perturbs from init_pos (global search).
        # Phase 4 perturbs from best_pl (local exploitation of the best discovered
        # local minimum). Swapping pairs of macros explores different topology
        # arrangements that noise cannot reach - particularly effective when the
        # best result significantly improves on the baseline (deeper local minimum).
        #
        # Uses rng_swap (RandomState(seed+2)) so the main rng state is unchanged
        # and any future runs are reproducible without affecting prior noise draws.
        #
        # Swap schedule: cycles through 1-swap and 2-swap+tiny-noise variants.
        # For fast benchmarks (ibm01: ~7s/score), this runs ~170 extra iterations
        # after the 300-restart noise loop. For slow benchmarks (ibm08: ~43s/score),
        # the noise loop already exhausts the budget and Phase 4 gets 0 iterations.
        rng_swap = np.random.RandomState(self.seed + 2)
        swap_schedule = (
            # (n_swaps, noise_frac): diversify swap sizes and tiny perturbations
            [(1, 0.0)] * 5 + [(2, 0.0)] * 3 + [(1, 0.005)] * 4
            + [(3, 0.0)] * 2 + [(1, 0.01)] * 4 + [(2, 0.005)] * 3
            + [(1, 0.0)] * 5 + [(2, 0.01)] * 3 + [(1, 0.02)] * 3
        ) * 20  # 960 entries - budget is the real limit
        phase4_k = last_noise_k + 1
        phase4_ran = 0
        for phase4_i, (n_sw, nf) in enumerate(swap_schedule):
            remaining = self.time_budget_s - (time.time() - t0)
            if remaining < t_one_score * 1.3:
                break
            # Extract current best legalized position as swap base
            best_pos_now = np.stack(
                [best_pl[:n, 0].numpy(), best_pl[:n, 1].numpy()], axis=1
            )
            swapped = _macro_swap_perturb(
                best_pos_now, movable, n, rng_swap,
                n_swaps=n_sw, noise_frac=nf,
                cw=cw, ch=ch, hw=hw, hh=hh,
            )
            label = f"swap{n_sw}{'+ noise' if nf > 0 else ''} phase4/{phase4_i+1}"
            if not _try_restart(label, swapped, k=phase4_k + phase4_i):
                break
            phase4_ran += 1
        if phase4_ran > 0:
            _log(f"  Phase 4 ran {phase4_ran} swap iterations")

        _log(f"  Best proxy={best_score:.4f}  total={time.time()-t0:.1f}s")
        return best_pl
