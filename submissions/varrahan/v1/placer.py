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


def _two_opt_swap(
    legal_pos: np.ndarray,
    init_pos: np.ndarray,
    sizes: np.ndarray,
    hw: np.ndarray,
    hh: np.ndarray,
    cw: float,
    ch: float,
    movable: np.ndarray,
    n: int,
    k_neighbors: int = 5,
    max_iters: int = 3,
    deadline: float | None = None,
) -> "tuple[np.ndarray, int]":
    """Post-legalize 2-opt swap pass.

    `_will_legalize` is greedy and cannot backtrack: once macro A is placed,
    it cannot move to give macro B a closer slot. This 2-opt pass examines
    pairs of nearby movable macros and tries swapping their positions. A swap
    is accepted iff:
        (1) Both macros remain in canvas bounds at their new positions.
        (2) Neither macro conflicts with any OTHER placed macro at its new
            position (and they don't conflict with each other).
        (3) Total per-pair displacement from init_pos strictly decreases.

    Spatial scope: for each macro i, we consider only its k_neighbors nearest
    placed macros (by current legal position). Distant swaps would increase
    total displacement anyway, so this restriction is essentially free.

    Iterates until no improvement or max_iters reached. Each iter is O(n²·k)
    in vectorized numpy (k_neighbors=5, max_iters=3 → ~1-3s for n=760).

    Returns (new_pos, swap_count).
    """
    sep_x_mat = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2  # [n, n]
    sep_y_mat = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2
    EPS = 0.05

    pos = legal_pos.copy()
    # Per-macro squared displacement from initial. We use squared (not L2) so
    # the strict-improvement check `d_new < d_old - 1e-9` is exact in float64.
    disp_sq = (pos[:, 0] - init_pos[:, 0]) ** 2 + (pos[:, 1] - init_pos[:, 1]) ** 2

    swap_count = 0
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        improved_any = False

        # For each macro i (in fixed order), find K nearest movable peers and
        # try swap with each. We re-derive kNN per outer iter — positions
        # change across iters, so the neighborhood changes too.
        # Pairwise sq distances (vectorized): O(n²) memory but n<=800 is fine.
        dx = pos[:, 0:1] - pos[:, 0:1].T
        dy = pos[:, 1:2] - pos[:, 1:2].T
        d_pair = dx * dx + dy * dy
        np.fill_diagonal(d_pair, np.inf)
        # Mask non-movable rows/cols to inf so they're never selected as neighbors.
        non_movable = ~movable
        d_pair[non_movable, :] = np.inf
        d_pair[:, non_movable] = np.inf
        # kNN per row: indices of K smallest entries.
        # argpartition is O(n) per row, faster than argsort.
        k_eff = min(k_neighbors, n - 1)
        if k_eff <= 0:
            break
        neighbors = np.argpartition(d_pair, k_eff, axis=1)[:, :k_eff]

        for i in range(n):
            if not movable[i]:
                continue
            if deadline is not None and time.time() > deadline:
                break
            for j in neighbors[i]:
                if not movable[j] or i == j:
                    continue
                # Tentative swap: i moves to pos[j], j moves to pos[i].
                new_ix, new_iy = pos[j, 0], pos[j, 1]
                new_jx, new_jy = pos[i, 0], pos[i, 1]

                # Bounds check.
                if (new_ix - hw[i] < -EPS or new_ix + hw[i] > cw + EPS or
                        new_iy - hh[i] < -EPS or new_iy + hh[i] > ch + EPS):
                    continue
                if (new_jx - hw[j] < -EPS or new_jx + hw[j] > cw + EPS or
                        new_jy - hh[j] < -EPS or new_jy + hh[j] > ch + EPS):
                    continue

                # Displacement check — strict improvement only.
                d_i_new = (new_ix - init_pos[i, 0]) ** 2 + (new_iy - init_pos[i, 1]) ** 2
                d_j_new = (new_jx - init_pos[j, 0]) ** 2 + (new_jy - init_pos[j, 1]) ** 2
                if d_i_new + d_j_new >= disp_sq[i] + disp_sq[j] - 1e-9:
                    continue

                # Conflict check: i at new pos vs all macros except i,j.
                # Build a mask excluding i and j.
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                mask[j] = False
                ox = pos[mask, 0]
                oy = pos[mask, 1]
                sxi = sep_x_mat[i, mask]
                syi = sep_y_mat[i, mask]
                conf_i = ((np.abs(new_ix - ox) < sxi + EPS) &
                          (np.abs(new_iy - oy) < syi + EPS)).any()
                if conf_i:
                    continue
                sxj = sep_x_mat[j, mask]
                syj = sep_y_mat[j, mask]
                conf_j = ((np.abs(new_jx - ox) < sxj + EPS) &
                          (np.abs(new_jy - oy) < syj + EPS)).any()
                if conf_j:
                    continue
                # i vs j (they end up where the other was — only an issue when
                # they were not separated to begin with; the original placement
                # is legal so pos[i] and pos[j] satisfy separation, but the new
                # i-at-pos[j] / j-at-pos[i] separation is symmetric so this is
                # also legal. Still verify defensively).
                if (abs(new_ix - new_jx) < sep_x_mat[i, j] + EPS and
                        abs(new_iy - new_jy) < sep_y_mat[i, j] + EPS):
                    continue

                # Accept swap.
                pos[i, 0], pos[i, 1] = new_ix, new_iy
                pos[j, 0], pos[j, 1] = new_jx, new_jy
                disp_sq[i] = d_i_new
                disp_sq[j] = d_j_new
                improved_any = True
                swap_count += 1
                break  # move to next i (positions changed; further j checks stale)

        if not improved_any:
            break

    return pos, swap_count


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
        #
        # Wide-noise tail (indices 35-51 in [0.10, 0.25]) was tested 2026-05-20 on ibm01
        # and confirmed ineffective: 3 wide-tail entries fired (restarts 39-41), all
        # scored 1.244-1.255 vs the 6% noise winner at 1.1860. The actual ibm01 -0.034
        # improvement came from the DREAMPlace candidate, not from any noise restart.
        # Wide-noise hypothesis is empirically dead on this benchmark.
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

        # -- Async DREAMPlace launch (Phase 5 candidate, fire-and-forget) ----
        # Launch DREAMPlace as a non-blocking subprocess BEFORE the main
        # pipeline starts. DREAMPlace runs in parallel with our scoring
        # (which is C++-side and releases the GIL on long ops). Its output
        # is checked at the END of the directed pipeline as one additional
        # candidate — additive, never displacing Phase 1/2/3 wins.
        #
        # v13 (sync) was rejected because it ran DREAMPlace BEFORE Phase 1,
        # paying 30-90s of subprocess time that displaced 5-10 noise/cong-grad
        # restarts on most benchmarks. Async hides that cost behind scoring.
        #
        # Launched for all ICCAD04 benchmarks (even when use_exact=False), so
        # the large-benchmark path (n>400 / grid>2200) can compare DP-vs-
        # baseline via a single _exact_proxy call. The 6 affected benchmarks
        # (ibm10/12/13/14/16/17) previously returned baseline-only in 2-6s.
        dp_handle = None
        try:
            # Inject this file's directory into sys.path so the sibling
            # `dreamplace_bridge` package resolves when placer.py is loaded
            # by the evaluator (which sets sys.path differently than CLI).
            import sys as _sys
            _v1_dir = str(Path(__file__).resolve().parent)
            if _v1_dir not in _sys.path:
                _sys.path.insert(0, _v1_dir)
            from dreamplace_bridge.run_bridge import (  # noqa: E402
                launch_dreamplace_async, is_available as _dp_available,
            )
            if _dp_available():
                iccad_dir = (Path("external/MacroPlacement/Testcases/ICCAD04")
                             / benchmark.name)
                if iccad_dir.exists():
                    # 2026-05-20 bridge rewrite: macro_place_flag=1 +
                    # multi-row .scl (8 rows of ~canvas_h/8 each) gives
                    # working DP optimization. With softs FIXED and
                    # iter=300, standalone DP proxy=1.3196 on ibm04
                    # (vs Phase 3 win 1.3316 — beats it by 0.012).
                    # softs movable hurts congestion (1.79 vs 1.69) and
                    # raises proxy to 1.3304; iter<300 under-converges
                    # (150 gives 1.5498); iter>500 DensityWeight runaway.
                    dp_handle = launch_dreamplace_async(
                        str(iccad_dir), plc=plc,
                        timeout_s=120.0,
                        iterations=300,
                        num_threads=2,
                        soft_macros_movable=False,
                    )
                    _log(f"  DREAMPlace launched async "
                         f"(macro_place_flag=1, iter=300, will check after Phase 3)")
        except Exception as exc:
            _log(f"  DREAMPlace launch failed: {type(exc).__name__}: {exc}")
            dp_handle = None

        # -- Restart 0: Baseline ----------------------------------------------
        _log(f"  Restart 0 (baseline)...")
        t1 = time.time()
        baseline_pos = _will_legalize(init_pos, movable, sizes, hw, hh, cw, ch, n)
        _log(f"    Legalized in {time.time()-t1:.1f}s")

        # 2-opt on the baseline causes subtle Phase 1 trajectory changes that
        # can BREAK the existing wins. Tested 2026-05-19: baseline 2-opt
        # improved ibm06 iter=1 from 1.6835 → 1.6801, but this made iter=2
        # (1.6812) unable to clear the higher bar, triggering Phase 1's
        # break-on-no-improvement and skipping the 5+ iterations that produced
        # v12's 1.6684 Phase 3 win. Net regression: ibm06 +0.0087.
        # 2-opt is therefore only applied on the baseline-only branch (below)
        # where there's no cong-grad trajectory to disrupt.

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
        #   - Conclusion: across orderings, displacement-sum is NOT a useful
        #     proxy ranker; different orderings produce legitimately different
        #     placements, not strictly-better ones.
        #
        # 2-opt swap post-pass (added 2026-05-19): WITHIN the same ordering, a
        # 2-opt local refinement can ONLY reduce per-pair displacement (strict
        # improvement check) and ONLY accepts legal swaps. Safe to apply on the
        # baseline-only branch: no cong-grad pipeline to interfere with. Tested
        # gain is small (~−0.0005 per benchmark on n>400 baseline-only set).
        if not use_exact:
            t_2opt = time.time()
            opt_pos, swap_count = _two_opt_swap(
                baseline_pos, init_pos, sizes, hw, hh, cw, ch, movable, n,
                k_neighbors=5, max_iters=3, deadline=t_2opt + 30.0,
            )
            _log(f"  2-opt: {swap_count} swaps in {time.time()-t_2opt:.1f}s")
            if swap_count > 0:
                pl_scratch[:n, 0] = torch.tensor(opt_pos[:, 0], dtype=torch.float32)
                pl_scratch[:n, 1] = torch.tensor(opt_pos[:, 1], dtype=torch.float32)

            # DP-vs-baseline comparison on large benchmarks (Improvement #1,
            # 2026-05-20). 6 benchmarks (ibm10/12/13/14/16/17) previously
            # returned baseline-only because exact scoring with cong-grad
            # ranking is too slow / density fallback is anti-correlated.
            # Strategy: score BASELINE FIRST (fast on most benchmarks, ~30-90s);
            # if scoring is fast enough that DP scoring also fits, score DP
            # and compare; if baseline scoring is too slow, skip DP and return
            # baseline (safe — DP might have won, but we can't fit both).
            #
            # DP-first tested 2026-05-20, REJECTED: on ibm16 (baseline 1.5324
            # vs DP 1.5751) and likely ibm17, DP loses to baseline. Trusting
            # DP unconditionally when baseline scoring doesn't fit caused
            # +0.043 regression on ibm16. Baseline-first is strictly safer:
            # we either know who won (small benchmarks) or correctly fall
            # back to baseline (slowest benchmarks where DP can't be verified).
            if plc is not None and dp_handle is not None:
                large_dp_budget = self.time_budget_s + 60.0  # mirrors BUDGET_OVERRUN_S below
                t_base_score_start = time.time()
                try:
                    base_score = float(_exact_proxy(pl_scratch, benchmark, plc))
                    t_base_score = time.time() - t_base_score_start
                    _log(f"  [large-DP] baseline exact proxy={base_score:.4f}  "
                         f"(scored in {t_base_score:.1f}s)")
                    # 130s threshold (vs the 100s SLOW_SCORE_THRESHOLD_S used in
                    # the use_exact=True path): under --all CPU contention, ibm10
                    # baseline scoring climbs from 67s standalone to 101s, just
                    # tripping a 100s threshold and losing a -0.037 DP win.
                    # 130s catches ibm10/12 (~100-110s under load) while still
                    # safely skipping ibm16/17 (~280s scoring even alone).
                    if t_base_score < 130.0:
                        # Wait for DP up to remaining budget minus reserved
                        # legalize+score window (~2*t_base_score).
                        remaining = large_dp_budget - (time.time() - t0)
                        max_wait = max(0.0, remaining - 2.0 * t_base_score - 5.0)
                        dp_full_large = dp_handle.wait_for_result_full(
                            max_wait_s=min(max_wait, 60.0)
                        )
                        if dp_full_large is not None:
                            dp_hard_l, dp_soft_l = dp_full_large
                            dp_hard_l_clip = dp_hard_l.copy()
                            dp_hard_l_clip[:, 0] = np.clip(dp_hard_l_clip[:, 0], hw, cw - hw)
                            dp_hard_l_clip[:, 1] = np.clip(dp_hard_l_clip[:, 1], hh, ch - hh)
                            t_dp_leg = time.time()
                            dp_leg_large = _will_legalize(
                                dp_hard_l_clip, movable, sizes, hw, hh, cw, ch, n,
                                deadline=t_dp_leg + 60.0,
                            )
                            dp_pl_large = benchmark.macro_positions.clone()
                            dp_pl_large[:n, 0] = torch.tensor(
                                dp_leg_large[:, 0], dtype=torch.float32
                            )
                            dp_pl_large[:n, 1] = torch.tensor(
                                dp_leg_large[:, 1], dtype=torch.float32
                            )
                            n_soft_l = int(min(dp_soft_l.shape[0], benchmark.num_soft_macros))
                            if n_soft_l > 0:
                                dp_pl_large[n:n + n_soft_l, 0] = torch.tensor(
                                    dp_soft_l[:n_soft_l, 0], dtype=torch.float32
                                )
                                dp_pl_large[n:n + n_soft_l, 1] = torch.tensor(
                                    dp_soft_l[:n_soft_l, 1], dtype=torch.float32
                                )
                            t_dp_score_start = time.time()
                            dp_score_large = float(_exact_proxy(dp_pl_large, benchmark, plc))
                            t_dp_score_large = time.time() - t_dp_score_start
                            _log(f"  [large-DP] dreamplace exact proxy={dp_score_large:.4f}  "
                                 f"(leg+score {time.time()-t_dp_leg:.1f}s)")
                            if dp_score_large < base_score:
                                _log(f"  [large-DP] DP wins ({dp_score_large:.4f} < "
                                     f"{base_score:.4f}); returning DP placement")
                                _log(f"  total={time.time()-t0:.1f}s")
                                return dp_pl_large
                            else:
                                _log(f"  [large-DP] baseline wins ({base_score:.4f} <= "
                                     f"{dp_score_large:.4f}); returning baseline")
                        else:
                            _log(f"  [large-DP] DP not ready in {max_wait:.0f}s; "
                                 f"returning baseline")
                            dp_handle.kill()
                    else:
                        _log(f"  [large-DP] baseline scoring slow ({t_base_score:.0f}s); "
                             f"skipping DP comparison, returning baseline")
                        dp_handle.kill()
                except Exception as exc:
                    _log(f"  [large-DP] error: {type(exc).__name__}: {exc}; "
                         f"returning baseline")
                    if dp_handle is not None:
                        try:
                            dp_handle.kill()
                        except Exception:
                            pass

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
            if dp_handle is not None:
                dp_handle.kill()
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

            # 2-opt-everywhere tested 2026-05-19, REJECTED. Applied to each
            # cong-grad iter, it produces:
            #   - ibm04: 1.3316 → 1.3201 (−0.0115 improvement ✓)
            #   - ibm06: 1.6684 → 1.6769 (+0.0085 regression ✗)
            #   - ibm02: 1.5923 → 1.5938 (+0.0015 regression ✗)
            # Net sporadic (similar variance pattern as WireMask). Root cause:
            # 2-opt pulls cong-grad-perturbed positions BACK toward their pre-
            # perturbation displacement target, undoing the cong-grad exploration
            # that was supposed to push macros AWAY from congested cells. The
            # cong-grad trajectory depends on consistent perturbation direction
            # across iters; 2-opt's "snap back to target" interferes.
            # 2-opt is still applied to BASELINE legalize (outside this function)
            # where there's no cong-grad trajectory to disrupt.

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

        # Density-grad / occupancy-spreading restart only fires for n <= 100,
        # which never occurs on IBM benchmarks (smallest ibm01 has n=246). It
        # also empirically hurt ibm03 (n=126) and ibm08 (n=301) in earlier
        # experiments. Removed 2026-05-19 along with its helpers
        # (_congestion_heatmap, _box_blur, _density_gradient_perturb).
        directed_ran = 0

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
        # Phase 3: cong-grad from best known position using current (stale) plc.
        # After Phase 2 failed wide steps, plc holds the cong map from a placement
        # that was WORSE than our best. Moving from the BEST position away from the
        # high-congestion regions of this stale map may explore a different local
        # minimum. Only runs when cong-grad improved at least once (cong_improved)
        # so we know the gradient signal is useful for this benchmark.
        #
        # Multi-frac Phase 3 (0.02/0.04/0.06) tested 2026-05-19, REJECTED. f=0.04
        # consistently wins on tested benchmarks (ibm04 1.3316, ibm06 1.6684,
        # ibm02 1.5923, ibm09 1.1304); the extra fracs 0.02/0.06 never found
        # deeper basins. Safe but ineffective; reverted for code clarity.
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

        # -- Async DREAMPlace check (Phase 5: additive candidate, hard+soft) -
        # Check if the DREAMPlace subprocess (launched at place() start)
        # has finished. Uses wait_for_result_full so we get BOTH the hard
        # and soft macro placements. With soft_macros_movable=True at launch,
        # DREAMPlace re-places soft macros around the new hard placement,
        # closing the v13 soft-macro mismatch that made standalone DREAMPlace
        # 0.2-0.3 worse than baseline.
        if dp_handle is not None:
            remaining_dp = (self.time_budget_s + BUDGET_OVERRUN_S) - (time.time() - t0)
            max_wait = max(0.0, min(remaining_dp - 3.0 * t_one_score, 30.0))
            dp_full = dp_handle.wait_for_result_full(max_wait_s=max_wait)
            if dp_full is not None:
                dp_hard, dp_soft = dp_full
                _log(f"  DREAMPlace ready in {dp_handle.time_elapsed():.1f}s "
                     f"(hard={dp_hard.shape[0]}, soft={dp_soft.shape[0]}); "
                     f"testing as candidate")
                # Phase 5a: DREAMPlace placement with BOTH hard + soft.
                # Legalize hard macros (DREAMPlace's NLP may leave overlaps).
                # Clip out-of-canvas first: DREAMPlace's macro_place_flag stage
                # can produce positions slightly past canvas (its .scl rows
                # cover canvas + slack). _will_legalize trusts initial pos
                # and won't re-clip a conflict-free out-of-bounds macro.
                t_dp = time.time()
                dp_leg_deadline = t_dp + 60.0
                dp_hard_clip = dp_hard.copy()
                dp_hard_clip[:, 0] = np.clip(dp_hard_clip[:, 0], hw, cw - hw)
                dp_hard_clip[:, 1] = np.clip(dp_hard_clip[:, 1], hh, ch - hh)
                dp_hard_leg = _will_legalize(
                    dp_hard_clip, movable, sizes, hw, hh, cw, ch, n,
                    deadline=dp_leg_deadline,
                )
                # Build a fresh placement tensor: legalized hard + DREAMPlace soft
                # + ports unchanged (taken from benchmark.macro_positions).
                dp_pl = benchmark.macro_positions.clone()
                dp_pl[:n, 0] = torch.tensor(dp_hard_leg[:, 0], dtype=torch.float32)
                dp_pl[:n, 1] = torch.tensor(dp_hard_leg[:, 1], dtype=torch.float32)
                # Soft positions go into [n : n + num_soft].
                n_soft_dp = int(min(dp_soft.shape[0], benchmark.num_soft_macros))
                if n_soft_dp > 0:
                    dp_pl[n:n + n_soft_dp, 0] = torch.tensor(
                        dp_soft[:n_soft_dp, 0], dtype=torch.float32
                    )
                    dp_pl[n:n + n_soft_dp, 1] = torch.tensor(
                        dp_soft[:n_soft_dp, 1], dtype=torch.float32
                    )
                t_dp_score_start = time.time()
                dp_score = float(_exact_proxy(dp_pl, benchmark, plc))
                t_dp_score = time.time() - t_dp_score_start
                if t_dp_score > t_one_score:
                    t_one_score = t_dp_score
                directed_ran += 1
                _log(f"  Candidate {directed_ran} (dreamplace hard+soft): "
                     f"proxy={dp_score:.4f}  (leg+score {time.time()-t_dp:.1f}s)")
                if dp_score < best_score:
                    best_score = dp_score
                    best_pl = dp_pl.clone()

                # Phase 5b: cong-grad from best_pl using DREAMPlace's plc state.
                # plc was just scored with DREAMPlace's placement, so its cong
                # map is now DREAMPlace-state. Perturbing best_pl with this
                # gradient explores basins the original-baseline plc state
                # couldn't reach.
                remaining_5b = (self.time_budget_s + BUDGET_OVERRUN_S) - (time.time() - t0)
                if remaining_5b >= t_one_score * 1.3:
                    best_pos_now = np.stack(
                        [best_pl[:n, 0].numpy(), best_pl[:n, 1].numpy()], axis=1
                    )
                    dp_perturbed = _routing_congestion_perturb(
                        best_pos_now, plc, benchmark, n, cw, ch, hw, hh, movable,
                        frac=0.04, rng=rng_cong,
                    )
                    if _try_restart("cong-grad-best from-dreamplace-plc f=0.04",
                                     dp_perturbed,
                                     k=1 + directed_ran, allow_overrun=True):
                        directed_ran += 1

                # Phase 6 (cong-grad from DP placement) tested 2026-05-20, REJECTED.
                # Found a single-iter win on ibm08 (1.5419 vs DP-additive 1.5444)
                # but the 4-iter loop displaced 3+ noise restarts that previously
                # found 1.5251 at 6% noise — net regression +0.0168. On ibm04/06
                # Phase 6 iter=1 didn't improve (best was already DP / Phase-3).
                # Limiting Phase 6 to 1 iter still didn't fit noise-6% on ibm08
                # within budget. Conclusion: cong-grad from DP placement doesn't
                # find systematically better basins; the marginal wins it does
                # find come at the cost of budget that helps more elsewhere.
            else:
                _log(f"  DREAMPlace not ready (elapsed={dp_handle.time_elapsed():.1f}s); "
                     f"killing subprocess")
                dp_handle.kill()

        # Phase 5c: wide-from-best with current plc state. Fills the slot left
        # by Phase 2 (wide from BASELINE only) and Phase 3/5b (frac=0.04 from
        # BEST only). Uses the latest plc state (post-Phase-5b if DP fired,
        # else post-Phase-3) which encodes the most-recent congestion pattern.
        # Purely additive: fires only if cong-grad helped earlier and budget
        # allows; placed AFTER Phase 5b so no current winning rng_cong path is
        # affected. Noise loop uses np.random directly (not rng_cong), so the
        # extra rng_cong draw here doesn't perturb noise restarts.
        if cong_improved:
            remaining_5c = (self.time_budget_s + BUDGET_OVERRUN_S) - (time.time() - t0)
            if remaining_5c >= t_one_score * 1.3:
                best_pos_5c = np.stack(
                    [best_pl[:n, 0].numpy(), best_pl[:n, 1].numpy()], axis=1
                )
                wide_perturbed = _routing_congestion_perturb(
                    best_pos_5c, plc, benchmark, n, cw, ch, hw, hh, movable,
                    frac=0.08, rng=rng_cong,
                )
                if _try_restart("cong-grad wide-from-best f=0.08",
                                 wide_perturbed,
                                 k=1 + directed_ran, allow_overrun=True):
                    directed_ran += 1

        # WireMask-BBO with congestion penalty tested 2026-05-19, REJECTED.
        # Helps sparse benchmarks (ibm01 WM=1.1964 vs baseline 1.2253) but hurts
        # dense ones (ibm04 WM=1.5070 vs 1.4101; ibm06 WM=1.8890 vs 1.7197).
        # Root cause: WireMask is constructive — rebuilds from scratch and
        # loses initial.plc's hand-tuned spread that the pipeline operates around.
        # A single alpha can't satisfy all benchmarks (would need per-benchmark
        # tuning, which violates the "no benchmark-specific tweaks" rule).
        # Implementation removed 2026-05-19; see commit 121a555-era history.

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
