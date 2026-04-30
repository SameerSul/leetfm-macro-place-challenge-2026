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
"""

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
) -> np.ndarray:
    """
    Largest-macro-first min-displacement legalization.
    Macros are placed one by one (largest area first) at the nearest
    overlap-free position to their target, found by expanding spiral search.
    Non-movable macros are fixed in place first.
    """
    sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2
    sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2
    order = sorted(range(n), key=lambda i: -(sizes[i, 0] * sizes[i, 1]))
    placed = np.zeros(n, dtype=bool)
    legal = pos.copy()
    for idx in order:
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
        n_restarts: int = 50,
        noise_fracs: Optional[List[float]] = None,
        seed: int = 42,
        time_budget_s: float = 200.0,
    ):
        self.n_restarts = n_restarts
        # Budget check in _try_restart terminates the loop early; n_restarts is an upper cap.
        # First 4 entries [0.02, 0.04, 0.06, 0.08] are the "core" fracs — their np.random
        # draw positions are preserved, so ibm01/03/08 winning restarts (6% and 2%) are
        # unchanged. Entries 5+ fill remaining budget for fast benchmarks:
        #   ibm01 (~5s/score): ~20 restarts fit in 200s → uses entries through ~20
        #   ibm08 (~36s/score): ~4 restarts fit → only core 4 used, unchanged behavior
        self.noise_fracs = noise_fracs or [
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
        ]
        self.seed = seed
        self.time_budget_s = time_budget_s

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

        # Exact scoring cutoffs:
        #   n > EXACT_MACRO_THRESHOLD: scoring too slow without exception
        #   grid_cells > EXACT_GRID_CELL_LIMIT: routing grid makes scoring slow regardless of n
        #     ibm18: 55x39=2145 cells → ~220s; ibm08: 38x34=1292 cells → ~39s
        # Threshold 400 includes ibm11 (n=373, 76s/score, gets 1 restart in 200s budget).
        # ibm13 (n=424, 101s/score) exceeds SLOW_SCORE_THRESHOLD and returns baseline anyway.
        EXACT_MACRO_THRESHOLD = 400
        EXACT_GRID_CELL_LIMIT = 2000
        grid_cells = benchmark.grid_rows * benchmark.grid_cols
        plc = _load_plc(benchmark.name)
        use_exact = (
            (plc is not None)
            and (n <= EXACT_MACRO_THRESHOLD)
            and (grid_cells <= EXACT_GRID_CELL_LIMIT)
        )
        if plc is None:
            _log("  Warning: plc unavailable, using density-score fallback")
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
        # This covers: (a) large-grid (ibm18: 55x39=2145 cells), (b) large-n (n>350).
        if not use_exact:
            _log(f"  Cannot use exact scoring: returning baseline only "
                 f"(density fallback anti-correlated with proxy)")
            _log(f"  Best proxy={best_score:.4f}  total={time.time()-t0:.1f}s")
            return best_pl

        # Safety net: if exact scoring took longer than expected (e.g. CPU load),
        # return baseline so we don't waste budget on unreliable density comparisons.
        SLOW_SCORE_THRESHOLD_S = 100.0
        if t_one_score > SLOW_SCORE_THRESHOLD_S:
            _log(f"  Exact score slow ({t_one_score:.0f}s); returning baseline")
            _log(f"  Best proxy={best_score:.4f}  total={time.time()-t0:.1f}s")
            return best_pl

        def _try_restart(label: str, perturbed_init: np.ndarray, k: int) -> bool:
            """Legalize + score one candidate. Returns True if best score updated."""
            nonlocal best_score, best_pl
            elapsed = time.time() - t0
            remaining = self.time_budget_s - elapsed
            # t_one_score is pure scoring time. Legalization adds 1-10s on top.
            # Factor 1.3 covers score + typical legalize without over-reserving budget.
            estimated_cost = t_one_score * 1.3
            if remaining < estimated_cost:
                _log(f"  Skipping restart {k}+ (budget: {remaining:.0f}s left, "
                     f"need ~{estimated_cost:.0f}s)")
                return False  # signal: stop further restarts

            t1 = time.time()
            leg = _will_legalize(perturbed_init, movable, sizes, hw, hh, cw, ch, n)
            _log(f"  Restart {k} ({label}) legalized in {time.time()-t1:.1f}s")

            score, pl = _score(leg)
            _log(f"  Candidate {k}: proxy={score:.4f}")
            if score < best_score:
                best_score = score
                best_pl = pl
                return True  # improvement found
            return True  # no improvement but keep going

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

        # -- Routing-congestion-gradient restart (v6) -------------------------
        # After scoring baseline with exact proxy, plc has the routing congestion
        # map computed. Use it to move macros in high-congestion cells toward
        # lower-congestion neighbors. Uses separate rng (rng_cong) so the main
        # np.random state is unchanged and subsequent noise draws are identical to
        # before this restart was added.
        rng_cong = np.random.RandomState(self.seed + 1)
        cong_perturbed = _routing_congestion_perturb(
            baseline_pos, plc, benchmark, n, cw, ch, hw, hh, movable, frac=0.04, rng=rng_cong
        )
        if not _try_restart("congestion-grad frac=4%", cong_perturbed, k=1 + directed_ran):
            _log(f"  Best proxy={best_score:.4f}  total={time.time()-t0:.1f}s")
            return best_pl
        directed_ran += 1

        # -- Restarts 1+: Random Gaussian -------------------------------------
        noise_scale_base = min(cw, ch)
        for k, frac in enumerate(
            self.noise_fracs[: self.n_restarts - 1 - directed_ran], start=1 + directed_ran
        ):
            noise = np.random.normal(0, frac * noise_scale_base, init_pos.shape)
            perturbed = np.clip(
                init_pos + noise,
                np.stack([hw, hh], axis=1),
                np.stack([cw - hw, ch - hh], axis=1),
            )
            if not _try_restart(f"random noise={frac:.0%}", perturbed, k=k):
                break

        _log(f"  Best proxy={best_score:.4f}  total={time.time()-t0:.1f}s")
        return best_pl
