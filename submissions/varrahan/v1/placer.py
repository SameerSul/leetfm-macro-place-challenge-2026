"""
Competitive Macro Placer -- Partcl/HRT Challenge 2026
Varrahan Uthayan (varrahan)

Algorithm:
  Multi-restart legalization with iterative routing-congestion-gradient
  perturbations, scored against the exact PlacementCost proxy.

  Pipeline per benchmark (200s soft budget, 60s overrun allowed for
  directed phases):
    0.       Baseline      legalize from initial.plc
    Phase 1  cong-grad     up to 12 iterative steps at frac=0.04 with adaptive
                           halving; each improving step updates the source
                           position for the next iter (uses live plc cong map)
    Phase 2  cong-grad     wide steps from baseline at frac=0.08, 0.12 using
                           the evolved (now-stale) plc cong map; early-exits
                           on first non-improvement
    Phase 3  cong-grad     perturb the current best at frac=0.04 using the
                           stale plc map — finds basins missed by Phase 1/2
                           (where ibm04's 1.3316 win lives)
    Noise tail             Random Gaussian restarts (1%-20%) fill remaining
                           budget; per-benchmark schedule preserves ibm01 6%
                           and ibm03 2% winners

  All candidates re-legalized and scored with PlacementCost; lowest proxy wins.

Why this pipeline:
  - Proxy = 1*WL + 0.5*density + 0.5*congestion. WL ~0.06, cong ~2.0:
    congestion dominates ~30x, so all directed moves target it (not WL).
  - SA-on-WL clusters macros, spikes congestion, regresses. Restarts explore
    legalization variants without destroying initial.plc's hand-tuned spread.

Baselines (full --all average over 17 IBM ICCAD04 benchmarks):
  will_seed             1.5338
  sameer_v1 leg-only    1.5062
  v12 (this code)       1.4854   stable, current best
  RePlAce               1.4578   <- challenge grand-prize threshold
  UT Austin DREAMPlace  1.4076   leaderboard #1 (GPU)
"""

import random
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from macro_place.benchmark import Benchmark


def _log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Will's minimum-displacement legalization (unchanged)
# ---------------------------------------------------------------------------

def _ring_offsets(r: int) -> np.ndarray:
    """Offsets (ddx, ddy) on the spiral ring at radius r, in the same lex
    order as the original nested-loop traversal: for ddx in -r..r, for ddy in
    -r..r if (|ddx|=r or |ddy|=r). Returns a [K, 2] int64 array (K = 8r for
    r>=1, K=1 for r=0).

    Lex order matters: `np.argmin` returns the first-occurrence index of the
    minimum, so on ties this matches the original `if d < best_d` strict
    less-than semantics that kept the lex-first candidate.
    """
    if r == 0:
        return np.array([[0, 0]], dtype=np.int64)
    # Left edge: ddx = -r, ddy in [-r, r]
    e1_ddx = np.full(2 * r + 1, -r, dtype=np.int64)
    e1_ddy = np.arange(-r, r + 1, dtype=np.int64)
    # Middle columns: ddx in (-r, r), ddy in {-r, +r} interleaved per ddx
    mid_range = np.arange(-r + 1, r, dtype=np.int64)  # length 2r-1
    mid_ddx = np.repeat(mid_range, 2)
    mid_ddy = np.tile(np.array([-r, r], dtype=np.int64), len(mid_range))
    # Right edge: ddx = +r, ddy in [-r, r]
    e2_ddx = np.full(2 * r + 1, r, dtype=np.int64)
    e2_ddy = np.arange(-r, r + 1, dtype=np.int64)
    return np.stack(
        [
            np.concatenate([e1_ddx, mid_ddx, e2_ddx]),
            np.concatenate([e1_ddy, mid_ddy, e2_ddy]),
        ],
        axis=1,
    )


def _will_legalize(
    pos: np.ndarray,
    movable: np.ndarray,
    sizes: np.ndarray,
    hw: np.ndarray,
    hh: np.ndarray,
    cw: float,
    ch: float,
    n: int,
    deadline: float | None = None,
    order: list | None = None,
) -> np.ndarray:
    """
    Min-displacement legalization with configurable macro placement order.
    Macros are placed one by one at the nearest overlap-free position to their
    target, found by expanding spiral search. Non-movable macros are fixed first.

    order: list of macro indices defining placement sequence. Default (None)
    uses largest-area-first. Different orders explore different legal arrangements.
    deadline: optional wall-clock time.time() value; remaining macros keep pos[].

    Spiral search is vectorized: per ring we build all K candidate positions at
    once and run a single [K, P] conflict matrix against the P already-placed
    macros (instead of K serial scalar comparisons inside Python loops). The
    lex-order ring traversal in _ring_offsets combined with np.argmin's
    first-occurrence semantics preserves the original tie-breaking, so the
    output is bit-equivalent to the prior nested-loop version.
    """
    sep_x_mat = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2  # [n, n]
    sep_y_mat = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2
    if order is None:
        order = sorted(range(n), key=lambda i: -(sizes[i, 0] * sizes[i, 1]))
    placed = np.zeros(n, dtype=bool)
    legal = pos.copy()
    MAX_R = 200
    EPS = 0.05  # separation tolerance, mirrors the original `+ 0.05` constant

    for idx in order:
        if deadline is not None and time.time() > deadline:
            break
        if not movable[idx]:
            placed[idx] = True
            continue

        sep_x_idx = sep_x_mat[idx]
        sep_y_idx = sep_y_mat[idx]

        # Current-position conflict check (only over actually-placed macros).
        # When no macros are placed yet, fall through to spiral search to match
        # the prior behavior of always moving the first movable macro by 1 step.
        if placed.any():
            cdx = np.abs(legal[idx, 0] - legal[placed, 0])
            cdy = np.abs(legal[idx, 1] - legal[placed, 1])
            if not (
                (cdx < sep_x_idx[placed] + EPS) & (cdy < sep_y_idx[placed] + EPS)
            ).any():
                placed[idx] = True
                continue

        # Spiral search
        step = max(sizes[idx, 0], sizes[idx, 1]) * 0.25
        px = float(pos[idx, 0])
        py = float(pos[idx, 1])
        hw_idx = float(hw[idx])
        hh_idx = float(hh[idx])
        placed_x = legal[placed, 0]
        placed_y = legal[placed, 1]
        sep_xp = sep_x_idx[placed]
        sep_yp = sep_y_idx[placed]
        best = legal[idx].copy()

        for r in range(1, MAX_R):
            ring = _ring_offsets(r)
            cand_x = np.clip(px + ring[:, 0] * step, hw_idx, cw - hw_idx)
            cand_y = np.clip(py + ring[:, 1] * step, hh_idx, ch - hh_idx)
            if placed_x.size > 0:
                # [K, P] overlap test in one numpy op
                dx_mat = np.abs(cand_x[:, None] - placed_x[None, :])
                dy_mat = np.abs(cand_y[:, None] - placed_y[None, :])
                bad = (
                    (dx_mat < sep_xp[None, :] + EPS)
                    & (dy_mat < sep_yp[None, :] + EPS)
                ).any(axis=1)
                valid = ~bad
            else:
                valid = np.ones(len(cand_x), dtype=bool)
            if not valid.any():
                continue
            # argmin returns first occurrence → matches original "first improvement wins".
            # CRITICAL: d² must be computed in pos.dtype precision to match the original
            # scalar code's `(cx - pos[idx, 0])` behavior. In the scalar, `cx` is a Python
            # float (weak scalar) and `pos[idx, 0]` is a numpy scalar of dtype pos.dtype;
            # numpy demotes the Python float to pos.dtype, so the subtraction (and d²)
            # happens at pos.dtype precision. When pos is float32 (the iter≥2 cong-grad
            # pipeline round-trips through best_pl as float32), this float32 precision
            # breaks ties between symmetric candidates: e.g. (cx-pos_x)² vs (cy-pos_y)²
            # round differently at small step. Without this match, argmin picks the
            # lex-first candidate among true ties; the original scalar picks whichever
            # has the (artifactually) smaller float32 d². Matching the artifact is
            # required for bit-equivalence with sameer_v1.
            diff_x = cand_x.astype(pos.dtype, copy=False) - pos[idx, 0]
            diff_y = cand_y.astype(pos.dtype, copy=False) - pos[idx, 1]
            d2 = diff_x * diff_x + diff_y * diff_y
            best_local = int(np.argmin(np.where(valid, d2, np.inf)))
            best = np.array([cand_x[best_local], cand_y[best_local]])
            break

        legal[idx] = best
        placed[idx] = True
    return legal


# ---------------------------------------------------------------------------
# Scoring utilities
# ---------------------------------------------------------------------------

def _load_plc(name: str, benchmark: Optional[Benchmark] = None):
    """Load PlacementCost for exact proxy scoring (posix paths for Windows compat).

    Caches the loaded plc on the benchmark object as `_cached_plc` so repeated
    place() calls on the same benchmark in dev iteration skip the ~1-3s load.
    """
    if benchmark is not None:
        cached = getattr(benchmark, "_cached_plc", None)
        if cached is not None:
            return cached
    try:
        from macro_place.loader import load_benchmark_from_dir, load_benchmark
        root = Path("external/MacroPlacement/Testcases/ICCAD04") / name
        plc = None
        if root.exists():
            _, plc = load_benchmark_from_dir(root.as_posix())
        else:
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
        if plc is not None and benchmark is not None:
            setattr(benchmark, "_cached_plc", plc)
        return plc
    except Exception as exc:
        _log(f"  Warning: plc load failed ({exc})")
    return None


def _exact_proxy(placement: torch.Tensor, benchmark: Benchmark, plc) -> float:
    """Return proxy_cost using the ground-truth PlacementCost evaluator."""
    from macro_place.objective import compute_proxy_cost
    costs = compute_proxy_cost(placement, benchmark, plc)
    return float(costs["proxy_cost"])


# ---------------------------------------------------------------------------
# Directed perturbation: macro occupancy spread (small-benchmark only)
# ---------------------------------------------------------------------------

def _congestion_heatmap(pos: np.ndarray, n: int, cw: float, ch: float, G: int = 20) -> np.ndarray:
    """G x G grid of macro-center counts per cell. High count => crowded cell."""
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
    Occupancy-spreading perturbation. Builds a smoothed macro-count heatmap,
    computes the negative-density gradient, and shifts each movable macro by
    `frac * min(cw, ch)` in the direction of lower local density before
    re-legalization. Only fires for n <= 100 (never on IBM benchmarks).
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
    benchmark: Benchmark,
    n: int,
    cw: float,
    ch: float,
    hw: np.ndarray,
    hh: np.ndarray,
    movable: np.ndarray,
    frac: float = 0.04,
    rng: np.random.RandomState | None = None,
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

    cell_cong = np.maximum(
        np.asarray(h_list).reshape(nr, nc_grid),
        np.asarray(v_list).reshape(nr, nc_grid),
    )

    cell_w = cw / nc_grid
    cell_h = ch / nr
    scale = frac * min(cw, ch)
    cong_threshold = 0.5  # only perturb macros in cells with congestion > threshold

    # Per-macro cell indices and local congestion (vectorized over all n macros)
    c_idx_all = np.minimum((pos[:n, 0] / cell_w).astype(np.int64), nc_grid - 1)
    r_idx_all = np.minimum((pos[:n, 1] / cell_h).astype(np.int64), nr - 1)
    local_cong_all = cell_cong[r_idx_all, c_idx_all]

    # Macros that qualify for perturbation: movable AND in a congested cell.
    # RNG draws below are sequenced in qualifying-macro order (mask is a boolean
    # over 0..n-1 so np.where preserves the original per-i traversal order that
    # the prior scalar loop used).
    mask = movable.astype(bool) & (local_cong_all >= cong_threshold)
    perturbed = pos.copy()
    if not mask.any():
        return perturbed

    r_idx = r_idx_all[mask]
    c_idx = c_idx_all[mask]
    local_cong = local_cong_all[mask]

    # Bounds-safe neighbor lookups for the finite-difference gradient
    c_left = np.maximum(c_idx - 1, 0)
    c_right = np.minimum(c_idx + 1, nc_grid - 1)
    r_down = np.maximum(r_idx - 1, 0)
    r_up = np.minimum(r_idx + 1, nr - 1)

    grad_x = (cell_cong[r_idx, c_right] - cell_cong[r_idx, c_left]) / 2.0
    grad_y = (cell_cong[r_up, c_idx] - cell_cong[r_down, c_idx]) / 2.0
    grad_len = np.sqrt(grad_x ** 2 + grad_y ** 2) + 1e-10

    # rng.normal(size=(k, 2)) draws in C order: noise[0,0], noise[0,1], noise[1,0], ...
    # This matches the scalar loop's interleaved (dx-noise, dy-noise) pair-per-macro
    # draw order exactly, so rng_cong advances identically and downstream Phase 3
    # / noise restarts see the same RNG state as the pre-vectorization version.
    move_scale = scale * np.minimum(local_cong, 2.0)
    noise = rng.normal(0.0, scale * 0.1, size=(int(mask.sum()), 2))
    dx = -(grad_x / grad_len) * move_scale + noise[:, 0]
    dy = -(grad_y / grad_len) * move_scale + noise[:, 1]

    perturbed[mask, 0] = np.clip(pos[mask, 0] + dx, hw[mask], cw - hw[mask])
    perturbed[mask, 1] = np.clip(pos[mask, 1] + dy, hh[mask], ch - hh[mask])

    return perturbed


# ---------------------------------------------------------------------------
# Main placer
# ---------------------------------------------------------------------------

class MacroPlacer:
    """
    Multi-restart legalization placer with congestion-gradient perturbations.

    Restart pipeline (subject to adaptive 200s + 60s-overrun budget):
      0        Baseline       legalize directly from initial.plc
      [n<=100] Density-grad   occupancy-spreading shift (never fires on IBM)
      Phase 1  cong-grad      up to 12 iterative steps at frac=0.04 with
                              adaptive halving on non-improvement
      Phase 2  cong-grad      wide steps from baseline at frac=0.08, 0.12
      Phase 3  cong-grad      perturb current best at frac=0.04 using stale plc
      Tail     Random noise   1%-20% gaussian, schedule preserves prior wins

    All candidates legalized then scored via PlacementCost; lowest proxy wins.
    Benchmarks with n>400, grid>2200 cells, or scoring > SLOW_SCORE_THRESHOLD_S
    return baseline only — sum-of-squares density fallback was empirically
    anti-correlated with proxy cost.

    Parameters
    ----------
    n_restarts : int
        Upper cap on total candidates (budget check is the real limit).
    noise_fracs : list[float]
        Magnitudes for random restarts (fraction of min canvas dimension).
    seed : int
        Random seed for reproducibility.
    time_budget_s : float
        Per-benchmark wall-clock soft budget.
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

        # Exact scoring cutoffs (re-measured 2026-05-08, clean CPU):
        #   ibm11 (n=373, grid=1755): 17s   — was estimated 75-263s under load
        #   ibm15 (n=393, grid=2166): 43s   — was estimated 160s (PROGRESS.md note 5)
        #   ibm18 (n=285, grid=2145): 62s   — was estimated 220s
        #   ibm13 (n=424): excluded by n threshold; ibm10/12/14/16/17 by both.
        # SLOW_SCORE_THRESHOLD_S=100s + post-scoring budget guard catch any regression
        # under load (e.g. if ibm11 jumps back to 263s, baseline is returned after one
        # scoring call instead of attempting restarts).
        EXACT_MACRO_THRESHOLD = 400  # includes ibm11 (n=373), ibm15 (n=393); excludes ibm13 (n=424)
        EXACT_GRID_CELL_LIMIT = 2200  # includes ibm15 (2166), ibm18 (2145); excludes ibm12 (2209)
        grid_cells = benchmark.grid_rows * benchmark.grid_cols
        plc = _load_plc(benchmark.name, benchmark)
        use_exact = (
            (plc is not None)
            and (n <= EXACT_MACRO_THRESHOLD)
            and (grid_cells <= EXACT_GRID_CELL_LIMIT)
        )
        if plc is None:
            _log("  Warning: plc unavailable, returning baseline only")
        elif n > EXACT_MACRO_THRESHOLD:
            _log(f"  Large benchmark (n={n} > {EXACT_MACRO_THRESHOLD}); "
                 f"restarts unrankable without exact proxy — returning baseline")
        elif grid_cells > EXACT_GRID_CELL_LIMIT:
            _log(f"  Large grid ({benchmark.grid_rows}x{benchmark.grid_cols}={grid_cells} > "
                 f"{EXACT_GRID_CELL_LIMIT}); restarts unrankable — returning baseline")

        # Shared scratch buffer for placement tensors. Filled in-place per
        # candidate by _score / the baseline build; only cloned when a candidate
        # becomes the new best_pl. Saves one clone per non-winning restart
        # (most restarts don't win).
        pl_scratch = benchmark.macro_positions.clone()

        def _score(pos: np.ndarray) -> float:
            """Update pl_scratch with hard-macro positions and return exact proxy.

            Caller must clone pl_scratch immediately if it needs to persist the
            result — the next _score call overwrites it.
            """
            pl_scratch[:n, 0] = torch.tensor(pos[:, 0], dtype=torch.float32)
            pl_scratch[:n, 1] = torch.tensor(pos[:, 1], dtype=torch.float32)
            return float(_exact_proxy(pl_scratch, benchmark, plc))

        # -- Restart 0: Baseline ----------------------------------------------
        _log(f"  Restart 0 (baseline)...")
        t1 = time.time()
        baseline_pos = _will_legalize(init_pos, movable, sizes, hw, hh, cw, ch, n)
        _log(f"    Legalized in {time.time()-t1:.1f}s")

        # Fill the scratch buffer with baseline positions; reused below either
        # as the returned baseline-only tensor or as the input to the first score.
        pl_scratch[:n, 0] = torch.tensor(baseline_pos[:, 0], dtype=torch.float32)
        pl_scratch[:n, 1] = torch.tensor(baseline_pos[:, 1], dtype=torch.float32)

        # No exact-scoring path => return baseline directly. Past experiments
        # confirmed that the sum-of-squares occupancy fallback is anti-correlated
        # with proxy cost (rewards spread, which hurts congestion), so unranked
        # baseline beats density-ranked restarts on every n>400 / large-grid case.
        #
        # Displacement-ranked multi-order tested 2026-05-19, REJECTED:
        #   - Hypothesis: lower total displacement from initial.plc → lower
        #     proxy (since initial.plc is hand-tuned).
        #   - Reality: tallest order minimized displacement on ibm10 (414 vs
        #     1051 default) but raised congestion → proxy 1.5658 vs v12's 1.4037
        #     (+0.162 regression). On dense benchmarks (ibm12), smallest-area
        #     order produced INVALID placements (27 overlaps) because big macros
        #     placed last couldn't find slots within the 60s spiral deadline.
        #   - Conclusion: displacement-sum is NOT a useful proxy ranker.
        #     Different orderings produce legitimately different placements,
        #     not strictly-better ones.
        if not use_exact:
            _log(f"  total={time.time()-t0:.1f}s")
            return pl_scratch  # safe: no more in-place writes will happen

        t_score0 = time.time()
        best_score = float(_exact_proxy(pl_scratch, benchmark, plc))
        t_one_score = time.time() - t_score0
        best_pl = pl_scratch.clone()
        _log(f"  Candidate 0: proxy={best_score:.4f}  (scored in {t_one_score:.1f}s)")

        # Safety net: if exact scoring took longer than expected (CPU load),
        # return baseline so we don't run out of budget mid-restart.
        SLOW_SCORE_THRESHOLD_S = 100.0
        if t_one_score > SLOW_SCORE_THRESHOLD_S:
            _log(f"  Exact score slow ({t_one_score:.0f}s); returning baseline")
            _log(f"  Best proxy={best_score:.4f}  total={time.time()-t0:.1f}s")
            return best_pl

        # Directed restarts (cong-grad Phase 1/2/3) can use up to BUDGET_OVERRUN_S
        # extra seconds beyond time_budget_s. Reasoning: a single transient scoring
        # spike on Phase 1 iter=0 (~200s vs typical ~7s on ibm04) was killing the
        # entire placer pipeline, blocking Phase 2/3 where the productive ibm04 win
        # lives (1.3316). With 60s overrun, ibm04 recovers Phase 3 even after a spike.
        # Noise restarts stay strict (allow_overrun=False default) — they're
        # exploratory and shouldn't push us over budget on dead-end benchmarks.
        BUDGET_OVERRUN_S = 60.0

        def _try_restart(label: str, perturbed_init: np.ndarray, k: int,
                         allow_overrun: bool = False,
                         order: Optional[List[int]] = None) -> bool:
            """Legalize + score one candidate. Returns False if budget exhausted.

            `order` (optional) is a custom macro placement order passed to
            _will_legalize. Default (None) uses largest-area first. Multi-order
            restarts vary this to explore different legal arrangements from the
            same starting positions.
            """
            nonlocal best_score, best_pl, t_one_score
            elapsed = time.time() - t0
            cap = self.time_budget_s + (BUDGET_OVERRUN_S if allow_overrun else 0.0)
            remaining = cap - elapsed
            # t_one_score is a running max over observed scoring times (initialized
            # from the baseline score). Factor 1.3 covers score + legalize.
            # Running-max (v11 design, removed in v12) is re-added because under
            # --all CPU contention, scorings can be 3-5x slower than baseline —
            # a much larger swing than "load jitter". Without adapting, the budget
            # check approves restarts that then exceed the cap, causing Phase 3
            # to be skipped on benchmarks like ibm04 (1.3316 → 1.3449 regression
            # observed in the multi-order --all run). The trade-off: brief blips
            # also tighten the budget, but blips that double t_one_score still
            # leave 60s overrun for directed phases.
            estimated_cost = t_one_score * 1.3
            if remaining < estimated_cost:
                _log(f"  Skipping restart {k}+ (budget: {remaining:.0f}s left, "
                     f"need ~{estimated_cost:.0f}s)")
                return False  # signal: stop further restarts

            t1 = time.time()
            leg_deadline = t1 + 60.0  # cap spiral search; timed-out macros keep pos value
            leg = _will_legalize(perturbed_init, movable, sizes, hw, hh, cw, ch, n,
                                 deadline=leg_deadline, order=order)
            t_leg = time.time() - t1
            _log(f"  Restart {k} ({label}) legalized in {t_leg:.1f}s")

            t_score_start = time.time()
            score = _score(leg)
            t_score_observed = time.time() - t_score_start
            if t_score_observed > t_one_score:
                t_one_score = t_score_observed
            _log(f"  Candidate {k}: proxy={score:.4f}")
            if score < best_score:
                best_score = score
                best_pl = pl_scratch.clone()  # snapshot only on improvement

            # Safety: if scoring overran the (possibly relaxed) cap, stop immediately
            # rather than launching another restart that would push time further over.
            if time.time() - t0 > cap:
                _log(f"  Over budget after scoring ({time.time()-t0:.0f}s, cap={cap:.0f}s); stopping")
                return False

            return True

        # -- Restart 1: Density-gradient (occupancy spread, small benchmarks) -
        # Only for small benchmarks (n <= 100): occupancy-spread helped ibm01 (54
        # macros) but hurt ibm03 (126) and ibm08 (301), and also consumed ~40s per
        # restart, blocking the 6% random noise slot that wins on ibm08.
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
        rng_cong = np.random.RandomState(self.seed + 1)
        cong_pos = baseline_pos
        cong_improved = False
        cong_frac = 0.04
        for cong_iter in range(12):
            if cong_iter > 0:
                # Use relaxed cap (matches _try_restart's allow_overrun=True path)
                # so a transient spike on iter=0 doesn't block the whole loop.
                remaining = (self.time_budget_s + BUDGET_OVERRUN_S) - (time.time() - t0)
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
                                 cong_perturbed, k=1 + directed_ran,
                                 allow_overrun=True):
                break  # don't kill Phase 2/3 — they have their own budget checks
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
                # Use relaxed cap so Phase 2 still fires after a Phase 1 spike.
                remaining = (self.time_budget_s + BUDGET_OVERRUN_S) - (time.time() - t0)
                if remaining < t_one_score * 1.3:
                    break
                cong_wide = _routing_congestion_perturb(
                    baseline_pos, plc, benchmark, n, cw, ch, hw, hh, movable,
                    frac=wide_frac, rng=rng_cong,
                )
                score_before = best_score
                if not _try_restart(f"cong-grad wide={wide_frac:.0%}", cong_wide,
                                     k=1 + directed_ran, allow_overrun=True):
                    break  # don't kill Phase 3 — it has its own check
                directed_ran += 1
                if best_score >= score_before:
                    break  # stop wide steps if this one didn't improve

        # Phase 3: cong-grad from best known position using current (stale) plc.
        # After Phase 2 failed wide steps, plc holds the cong map from a placement
        # that was WORSE than our best. Moving from the BEST position away from the
        # high-congestion regions of this stale map may explore a different local
        # minimum. Only runs when cong-grad improved at least once (cong_improved)
        # so we know the gradient signal is useful for this benchmark.
        if cong_improved:
            # Use relaxed cap so Phase 3 fires after a Phase 1 spike — this is
            # where ibm04's 1.3316 win lives.
            remaining = (self.time_budget_s + BUDGET_OVERRUN_S) - (time.time() - t0)
            if remaining >= t_one_score * 1.3:
                best_pos_now = np.stack(
                    [best_pl[:n, 0].numpy(), best_pl[:n, 1].numpy()], axis=1
                )
                phase3_perturbed = _routing_congestion_perturb(
                    best_pos_now, plc, benchmark, n, cw, ch, hw, hh, movable,
                    frac=0.04, rng=rng_cong,
                )
                if _try_restart("cong-grad phase3", phase3_perturbed,
                                 k=1 + directed_ran, allow_overrun=True):
                    directed_ran += 1
                # On Phase 3 failure, fall through to noise loop (which will
                # likely also skip on its own strict pre-check)

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
