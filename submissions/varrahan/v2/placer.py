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

import os
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
    deadline: optional wall-clock time.monotonic() value; remaining macros keep pos[].

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
        if deadline is not None and time.monotonic() > deadline:
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
        if deadline is not None and time.monotonic() > deadline:
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
            if deadline is not None and time.monotonic() > deadline:
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


def _two_opt_proxy_swap(
    legal_pos: np.ndarray,
    sizes: np.ndarray,
    hw: np.ndarray,
    hh: np.ndarray,
    cw: float,
    ch: float,
    movable: np.ndarray,
    n: int,
    score_fn=None,
    initial_score: float = 0.0,
    k_neighbors: int = 5,
    max_iters: int = 3,
    deadline: float | None = None,
    incremental_scorer=None,
    macro_cong: "np.ndarray | None" = None,
    cong_hot_k: int = 20,
    cong_cold_k: int = 8,
) -> "tuple[np.ndarray, int, float, int]":
    """Proxy-driven 2-opt swap pass (issue #1, 2026-05-23).

    Like `_two_opt_swap`, but accepts a swap iff the resulting placement's
    proxy cost (via `score_fn`) strictly decreases. The displacement-from-init
    criterion the original uses was empirically anti-correlated with proxy
    cost on ibm01/04/10 (see ISSUES.md #1) — so it wasted the 15s budget
    on swaps the post-hoc proxy check then rejected.

    Each candidate swap that passes bounds + neighbor-conflict checks is
    applied tentatively, scored, then either kept (if proxy improves) or
    reverted. The cheap checks act as a free filter so most candidates
    never reach the score call.

    Cost model: ~5-50ms per score call (depending on benchmark size, post-
    vectorization). Budget cap via `deadline` is critical; on large n the
    full O(n²·k) candidate set won't fit. Iterates outer loop up to
    `max_iters` or until no improvement.

    Returns: (pos, accept_count, final_score, score_calls).
    """
    sep_x_mat = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2
    sep_y_mat = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2
    EPS = 0.05

    pos = legal_pos.copy()
    best_score = initial_score
    accept_count = 0
    score_calls = 0

    # S9 (2026-05-26): congestion-aware candidate selection. When `macro_cong`
    # (per-macro local routing congestion, snapshot at seed time) is supplied:
    #   * Variant 1 — OUTER ORDERING: iterate macros hot→cold instead of by
    #     index. On deadline-bound benchmarks the swaps evaluated before the
    #     budget runs out are then the ones touching congestion hotspots — the
    #     dominant proxy term. (Pure budget reallocation; can't exceed the
    #     deadline-free convergence point.)
    #   * Variant 2 — NEIGHBOR AUGMENTATION: spatial kNN can only ever swap
    #     nearby macros, so a routing-heavy macro can never relocate across the
    #     chip to a cold region (the intermediate local swaps would all be
    #     rejected). For the `cong_hot_k` hottest macros, append the
    #     `cong_cold_k` coldest macros as extra swap candidates — a "teleport"
    #     edge no sequence of local swaps can synthesize, expanding the reachable
    #     placement set. Size-incompatible teleports fail the free conflict
    #     check before scoring, so most cost nothing.
    # The proxy gate still validates every swap, so this only changes WHICH
    # candidates are tried, never accepts a worse placement. macro_cong=None
    # reproduces the prior index-order / spatial-only behavior exactly.
    cong_aware = macro_cong is not None and n > 1
    if cong_aware:
        mc = np.asarray(macro_cong, dtype=np.float64)
        mov_idx = np.where(movable)[0]
        # Outer order: movable macros sorted hottest-first (stale-but-fine).
        outer_order = mov_idx[np.argsort(-mc[mov_idx], kind="stable")]
        # Hot set (rows that get augmented) and cold pool (teleport targets).
        n_hot = min(cong_hot_k, mov_idx.size)
        hot_rows = set(int(x) for x in outer_order[:n_hot])
        n_cold = min(cong_cold_k, mov_idx.size)
        cold_pool = mov_idx[np.argsort(mc[mov_idx], kind="stable")][:n_cold]
    else:
        outer_order = np.arange(n)
        hot_rows = set()
        cold_pool = np.empty(0, dtype=np.int64)

    # B8 (adaptive max_iters, 2026-05-24): the caller-provided max_iters is
    # treated as the BASELINE. After iter 1 we measure the accept ratio:
    #   - high yield (>15%): extend to 5 (more iters likely productive).
    #   - low yield  (<5%):  cap at 1 (don't waste budget on more iters).
    # Tracked in `effective_max_iters` so the outer-loop bound updates live.
    effective_max_iters = max_iters
    # B7 (cache) tested 2026-05-24 and TENTATIVELY DISABLED. Memoizing
    # trial scores by frozenset({i, j}) is logically sound (within an iter
    # without accepts, scores are deterministic per pair). Bit-equivalence
    # verification passes. But ibm10 reproducibly regressed 1.3728 → 1.3791
    # (+0.0063) under --all with B7 enabled — single-bench confirms.
    # Suspicion: the cache flips which (i, j) trial is "first" in mutual-
    # kNN scenarios, which changes the greedy accept sequence in unintuitive
    # ways. Disabled until the regression mechanism is understood.
    swap_score_cache: "dict[frozenset, float]" = {}
    cache_hits = 0
    # B7 (cache) tested again 2026-05-24 post-B3p4. Result: ibm01 +0.0002,
    # ibm04 0, ibm10 +0.0006 — small regression. The frozenset construction
    # + dict lookup overhead exceeds the saved score time at ~3ms/score.
    # With incremental scoring this fast, the cache is no longer profitable.
    # Disabled.
    B7_CACHE_ENABLED = False

    it = 0
    while it < effective_max_iters:
        if deadline is not None and time.monotonic() > deadline:
            break
        improved_any = False
        iter_accepts = 0
        iter_scores = 0
        swap_score_cache.clear()

        # kNN per macro (re-derived each outer iter; positions change on accept)
        dx = pos[:, 0:1] - pos[:, 0:1].T
        dy = pos[:, 1:2] - pos[:, 1:2].T
        d_pair = dx * dx + dy * dy
        np.fill_diagonal(d_pair, np.inf)
        non_movable = ~movable
        d_pair[non_movable, :] = np.inf
        d_pair[:, non_movable] = np.inf
        k_eff = min(k_neighbors, n - 1)
        if k_eff <= 0:
            break
        neighbors = np.argpartition(d_pair, k_eff, axis=1)[:, :k_eff]

        # B9 (smarter ordering) tested twice and REVERTED 2026-05-24:
        #   - DESCENDING by distance: ibm01 +0.003 regression (greedy path).
        #   - ASCENDING by distance: --all 1.4647 → 1.4647 (zero change), but
        #     wall-clock +42s (no benefit, slight cost). The candidate pool
        #     is exhausted within deadline at k=10/max_iters=6, so order
        #     doesn't matter on these benchmarks.
        # Keep argpartition's native order.

        for i in outer_order:
            i = int(i)
            if not movable[i]:
                continue
            if deadline is not None and time.monotonic() > deadline:
                break
            # Candidate j's: spatial kNN, plus the cold-region teleport pool for
            # hot macros (S9 variant 2). Dedupe the pool against the spatial set
            # so a cold macro that's already a near neighbor isn't scored twice.
            if cong_aware and i in hot_rows and cold_pool.size:
                extra = cold_pool[~np.isin(cold_pool, neighbors[i])]
                cand_js = np.concatenate([neighbors[i], extra]) if extra.size else neighbors[i]
            else:
                cand_js = neighbors[i]
            for j in cand_js:
                j = int(j)
                if not movable[j] or i == j:
                    continue
                if deadline is not None and time.monotonic() > deadline:
                    break

                new_ix, new_iy = pos[j, 0], pos[j, 1]
                new_jx, new_jy = pos[i, 0], pos[i, 1]

                # Bounds check
                if (new_ix - hw[i] < -EPS or new_ix + hw[i] > cw + EPS or
                        new_iy - hh[i] < -EPS or new_iy + hh[i] > ch + EPS):
                    continue
                if (new_jx - hw[j] < -EPS or new_jx + hw[j] > cw + EPS or
                        new_jy - hh[j] < -EPS or new_jy + hh[j] > ch + EPS):
                    continue

                # Conflict check (vs all macros except i and j)
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
                # i↔j separation (already legal pre-swap; verify defensively)
                if (abs(new_ix - new_jx) < sep_x_mat[i, j] + EPS and
                        abs(new_iy - new_jy) < sep_y_mat[i, j] + EPS):
                    continue

                # Apply swap tentatively, score, decide
                old_ix, old_iy = pos[i, 0], pos[i, 1]
                old_jx, old_jy = pos[j, 0], pos[j, 1]
                pos[i, 0], pos[i, 1] = new_ix, new_iy
                pos[j, 0], pos[j, 1] = new_jx, new_jy

                # B7 cache lookup (disabled — see B7_CACHE_ENABLED note above).
                if B7_CACHE_ENABLED:
                    cache_key = frozenset((int(i), int(j)))
                    cached = swap_score_cache.get(cache_key)
                else:
                    cached = None
                if cached is not None:
                    trial_score = cached
                    cache_hits += 1
                else:
                    # B3 phase 2: if an incremental_scorer is provided, use its
                    # score_swap (which only recomputes WL for touched nets and
                    # reuses plc for density/congestion). Otherwise fall back to
                    # the full score_fn (B3 phase 1 / original path).
                    if incremental_scorer is not None:
                        trial_score = incremental_scorer.score_swap(
                            i, (new_ix, new_iy), j, (new_jx, new_jy)
                        )
                    else:
                        trial_score = score_fn(pos)
                    score_calls += 1
                    iter_scores += 1
                    if B7_CACHE_ENABLED:
                        swap_score_cache[cache_key] = trial_score
                if trial_score < best_score:
                    if incremental_scorer is not None:
                        incremental_scorer.commit_swap(
                            i, (new_ix, new_iy), j, (new_jx, new_jy)
                        )
                    best_score = trial_score
                    accept_count += 1
                    iter_accepts += 1
                    improved_any = True
                    if B7_CACHE_ENABLED:
                        swap_score_cache.clear()  # pos changed → cache invalid
                    break  # positions changed; refresh kNN at next outer iter
                else:
                    # Revert (scorer already reverted plc internally)
                    pos[i, 0], pos[i, 1] = old_ix, old_iy
                    pos[j, 0], pos[j, 1] = old_jx, old_jy

        if not improved_any:
            break

        # B8 (adaptive max_iters): DISABLED for now — extending to 5 iters
        # on high-yield benchmarks (ibm10 19%→29% accept rate) regressed
        # ibm10 from 1.3728 → 1.3791. Suspect float drift in
        # incremental_scorer.total_wl_raw across many commits. Keep iter
        # count fixed at caller's max_iters until investigated.
        # if it == 0 and iter_scores > 0:
        #     yield_ratio = iter_accepts / iter_scores
        #     if yield_ratio > 0.15:
        #         effective_max_iters = max(effective_max_iters, 5)
        #     elif yield_ratio < 0.05:
        #         effective_max_iters = min(effective_max_iters, 1)
        it += 1

    return pos, accept_count, best_score, score_calls


def _relocation_moves(
    pos: np.ndarray,
    sizes: np.ndarray,
    hw: np.ndarray,
    hh: np.ndarray,
    cw: float,
    ch: float,
    movable: np.ndarray,
    n: int,
    plc,
    benchmark: "Benchmark",
    incremental_scorer,
    initial_score: float,
    deadline: "float | None" = None,
    top_hot: int = 24,
    n_targets: int = 12,
) -> "tuple[np.ndarray, int, float]":
    """Congestion-directed single-macro RELOCATION pass (2026-05-27).

    The 2-opt search only EXCHANGES two macros' positions — it can never relocate
    a routing-heavy macro into an empty low-congestion gap (a swap would dump some
    other macro into the vacated hot spot). This pass does exactly that: for the
    hottest macros (by live routing congestion), try moving each into a handful of
    the lowest-congestion legal cell centers, accept iff the true proxy (via the
    incremental scorer's `score_move`) strictly drops, then `commit_move`.

    Legality = in-bounds + no overlap with other HARD macros (soft macros may
    overlap, so they're ignored). The proxy gate filters far moves that spike
    wirelength, so only net-improving relocations stick. Returns (pos, accepts,
    best_score).
    """
    nr, nc = benchmark.grid_rows, benchmark.grid_cols
    try:
        h_arr = np.asarray(plc.get_horizontal_routing_congestion(), dtype=np.float64)
        v_arr = np.asarray(plc.get_vertical_routing_congestion(), dtype=np.float64)
    except Exception:
        return pos, 0, initial_score
    if h_arr.size != nr * nc or v_arr.size != nr * nc:
        return pos, 0, initial_score
    cell_cong = np.maximum(h_arr.reshape(nr, nc), v_arr.reshape(nr, nc))
    cell_w, cell_h = cw / nc, ch / nr

    # Per-macro local congestion → pick the hottest movable macros to relocate.
    ci_all = np.clip((pos[:n, 0] / cell_w).astype(np.int64), 0, nc - 1)
    ri_all = np.clip((pos[:n, 1] / cell_h).astype(np.int64), 0, nr - 1)
    local_cong = cell_cong[ri_all, ci_all]
    mov_idx = np.where(movable)[0]
    if mov_idx.size == 0:
        return pos, 0, initial_score
    hot = mov_idx[np.argsort(-local_cong[mov_idx])][:top_hot]

    # Candidate target cells = the lowest-congestion cells; their centers are the
    # relocation destinations. Pool a few × n_targets so per-macro legality
    # filtering still leaves options.
    flat = cell_cong.ravel()
    pool = np.argsort(flat)[: max(n_targets * 4, n_targets)]
    tgt_c = (pool % nc).astype(np.float64)
    tgt_r = (pool // nc).astype(np.float64)
    tgt_x = (tgt_c + 0.5) * cell_w
    tgt_y = (tgt_r + 0.5) * cell_h
    tgt_cong = flat[pool]

    sep_x_mat = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2
    sep_y_mat = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2
    EPS = 0.05

    best_score = initial_score
    accepts = 0
    all_idx = np.arange(n)
    for i in hot:
        i = int(i)
        if deadline is not None and time.monotonic() > deadline:
            break
        if not movable[i]:
            continue
        # Only consider targets at lower congestion than the macro's current cell
        # (relief), nearest-first among those, capped at n_targets.
        cand = np.where(tgt_cong < local_cong[i] - 1e-9)[0]
        if cand.size == 0:
            continue
        d2 = (tgt_x[cand] - pos[i, 0]) ** 2 + (tgt_y[cand] - pos[i, 1]) ** 2
        cand = cand[np.argsort(d2)][:n_targets]

        mask = all_idx != i
        sxi = sep_x_mat[i, mask]
        syi = sep_y_mat[i, mask]
        ox = pos[mask, 0]
        oy = pos[mask, 1]
        best_i_xy = None
        for t in cand:
            nx, ny = float(tgt_x[t]), float(tgt_y[t])
            if (nx - hw[i] < -EPS or nx + hw[i] > cw + EPS or
                    ny - hh[i] < -EPS or ny + hh[i] > ch + EPS):
                continue
            # Overlap vs other HARD macros (vectorized).
            if ((np.abs(nx - ox) < sxi + EPS) & (np.abs(ny - oy) < syi + EPS)).any():
                continue
            s = incremental_scorer.score_move(i, (nx, ny))
            if s < best_score - 1e-9:
                best_score = s
                best_i_xy = (nx, ny)
        if best_i_xy is not None:
            incremental_scorer.commit_move(i, best_i_xy)
            pos[i, 0], pos[i, 1] = best_i_xy
            accepts += 1
    return pos, accepts, best_score


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
            # NG45 designs all share the leaf-directory name
            # "output_CT_Grouping" → benchmark.name doesn't disambiguate
            # them. Two-step lookup:
            #   1. Try the legacy "<design>_ng45" alias (kept for
            #      backward compat with older harnesses).
            #   2. If name is "output_CT_Grouping" or otherwise unmatched,
            #      iterate the 4 NG45 designs and pick the one whose
            #      plc matches `benchmark`'s canvas dimensions.
            ng45_aliases = {
                "ariane133_ng45": "ariane133",
                "ariane136_ng45": "ariane136",
                "nvdla_ng45": "nvdla",
                "mempool_tile_ng45": "mempool_tile",
            }
            ng45_base = Path("external/MacroPlacement/Flows/NanGate45")
            d = ng45_aliases.get(name)
            if d:
                base = ng45_base / d / "netlist" / "output_CT_Grouping"
                if (base / "netlist.pb.txt").exists():
                    _, plc = load_benchmark(
                        (base / "netlist.pb.txt").as_posix(),
                        (base / "initial.plc").as_posix(),
                    )
            elif benchmark is not None and name in (
                "output_CT_Grouping",
            ) and ng45_base.exists():
                # Disambiguate by canvas dimensions.
                bench_cw, bench_ch = benchmark.canvas_width, benchmark.canvas_height
                for design in ("ariane133", "ariane136", "nvdla", "mempool_tile"):
                    base = ng45_base / design / "netlist" / "output_CT_Grouping"
                    if not (base / "netlist.pb.txt").exists():
                        continue
                    try:
                        cand_bench, cand_plc = load_benchmark(
                            (base / "netlist.pb.txt").as_posix(),
                            (base / "initial.plc").as_posix(),
                        )
                        if (
                            abs(cand_bench.canvas_width - bench_cw) < 1e-6
                            and abs(cand_bench.canvas_height - bench_ch) < 1e-6
                        ):
                            plc = cand_plc
                            _log(f"  NG45 design matched: {design}")
                            break
                    except Exception:
                        continue
        if plc is not None and benchmark is not None:
            setattr(benchmark, "_cached_plc", plc)
        return plc
    except Exception as exc:
        _log(f"  Warning: plc load failed ({exc})")
    return None


def _build_macro_pin_map(plc):
    """Cache MACRO_NAME -> [pin_indices] on plc (mirrors objective._set_placement's
    one-time build, but built eagerly here so the fast path doesn't fork on the
    hasattr check every call)."""
    if hasattr(plc, "_macro_pin_map"):
        return plc._macro_pin_map
    pin_map: "dict[str, list[int]]" = {}
    for idx, mod in enumerate(plc.modules_w_pins):
        if mod.get_type() == "MACRO_PIN" and hasattr(mod, "get_macro_name"):
            name = mod.get_macro_name()
            pin_map.setdefault(name, []).append(idx)
    plc._macro_pin_map = pin_map
    return pin_map


def _ensure_pos_cache(plc) -> np.ndarray:
    """Maintain a per-module (x, y) position cache (B3, 2026-05-23).

    Vectorized scoring functions previously called `mods[idx].get_pos()`
    in Python loops per call — ~3-6ms on ibm10 across WL / density /
    congestion combined. This cache eliminates those loops by storing
    positions in a numpy array, updated in-place by `_fast_set_placement`.

    Initial build is O(n_modules) get_pos calls; amortized to near-zero.
    Reads from the cache are fancy-indexed numpy operations.

    Returns a (n_modules, 2) float64 array. Indexed by `plc.modules_w_pins`
    index — the same indexing used by `unique_ref`, `macro_indices`, and
    `hard_indices` in the various scoring caches.
    """
    cache = getattr(plc, "_global_pos_cache", None)
    if cache is None:
        mods = plc.modules_w_pins
        cache = np.empty((len(mods), 2), dtype=np.float64)
        for k, m in enumerate(mods):
            x, y = m.get_pos()
            cache[k, 0] = x
            cache[k, 1] = y
        plc._global_pos_cache = cache
    return cache


def _build_wl_cache(plc):
    """Precompute per-pin arrays used by the vectorized wirelength.

    For each net (in plc.nets.keys() insertion order), record:
      - per-pin ref_node_idx (index into plc.modules_w_pins)
      - per-pin x_offset, y_offset
    Plus per-net weight (from the driver pin) and reduceat boundaries.

    The unified representation: PORT pins use ref_node_idx = port_idx + offset 0.
    Macro pins use ref_node_idx = parent_macro_idx + pin's stored offset.
    Either way, pin_pos = node_pos[ref_idx] + offset; this matches plc's
    private __get_pin_position semantics exactly.
    """
    if hasattr(plc, "_wl_vec_cache"):
        return plc._wl_vec_cache

    ref_idx_list: "list[int]" = []
    x_off_list: "list[float]" = []
    y_off_list: "list[float]" = []
    net_starts: "list[int]" = []
    net_weights: "list[float]" = []
    cursor = 0
    skipped_nets = 0

    name_to_idx = plc.mod_name_to_indices
    mods = plc.modules_w_pins

    def _pin_info(pin_idx: int):
        pin = mods[pin_idx]
        ptype = pin.get_type()
        if ptype == "PORT":
            return pin_idx, 0.0, 0.0
        if ptype == "MACRO_PIN":
            parent_name = pin.get_macro_name()
            ref_idx = name_to_idx.get(parent_name, -1)
            if ref_idx == -1:
                return None
            return ref_idx, float(getattr(pin, "x_offset", 0.0)), float(getattr(pin, "y_offset", 0.0))
        return None

    for driver_pin_name, sink_pin_names in plc.nets.items():
        driver_idx = name_to_idx.get(driver_pin_name)
        if driver_idx is None:
            skipped_nets += 1
            continue
        driver_info = _pin_info(driver_idx)
        if driver_info is None:
            skipped_nets += 1
            continue
        driver_pin = mods[driver_idx]
        try:
            weight = float(driver_pin.get_weight())
        except Exception:
            weight = 1.0

        local_pins = [driver_info]
        for sink_name in sink_pin_names:
            sink_idx = name_to_idx.get(sink_name)
            if sink_idx is None:
                continue
            info = _pin_info(sink_idx)
            if info is None:
                continue
            local_pins.append(info)
        if len(local_pins) < 1:
            skipped_nets += 1
            continue

        net_starts.append(cursor)
        net_weights.append(weight)
        for r, xo, yo in local_pins:
            ref_idx_list.append(r)
            x_off_list.append(xo)
            y_off_list.append(yo)
        cursor += len(local_pins)

    ref_idx_arr = np.asarray(ref_idx_list, dtype=np.int64)
    x_off_arr = np.asarray(x_off_list, dtype=np.float64)
    y_off_arr = np.asarray(y_off_list, dtype=np.float64)
    net_starts_arr = np.asarray(net_starts, dtype=np.int64)
    net_weights_arr = np.asarray(net_weights, dtype=np.float64)

    # Unique ref_node indices (gather destinations) + inverse mapping into
    # pin-flat order. Lets us pull current node positions in one pass and
    # then scatter via numpy indexing.
    unique_ref, inv = np.unique(ref_idx_arr, return_inverse=True)

    # B3 phase 2 (2026-05-23): per-pin → net-index mapping for incremental
    # scoring. Allows touched-net selection given a moved macro.
    pin_to_net = (
        np.searchsorted(net_starts_arr, np.arange(cursor), side="right") - 1
    ).astype(np.int64)
    # Per-net pin lengths (cursor as sentinel for the last net's end).
    net_ends = np.empty_like(net_starts_arr)
    net_ends[:-1] = net_starts_arr[1:]
    net_ends[-1] = cursor
    net_lengths = (net_ends - net_starts_arr).astype(np.int64)


    cache = {
        "ref_idx": ref_idx_arr,
        "ref_inv": inv.astype(np.int64),
        "unique_ref": unique_ref,
        "x_off": x_off_arr,
        "y_off": y_off_arr,
        "net_starts": net_starts_arr,
        "net_ends": net_ends,
        "net_lengths": net_lengths,
        "net_weights": net_weights_arr,
        "pin_to_net": pin_to_net,
        "n_pins": cursor,
        "n_nets": len(net_starts),
    }
    plc._wl_vec_cache = cache
    return cache


def _vectorized_wirelength(plc) -> float:
    """Drop-in numpy replacement for plc.get_wirelength().

    Iterates plc.nets in insertion order (matching scalar semantics), computes
    per-net (max-min) HPWL in vector form via np.minimum/maximum.reduceat, and
    sums in float64. Tiny FP differences vs the scalar loop are possible but
    irrelevant at proxy-cost granularity.
    """
    cache = _build_wl_cache(plc)
    if cache["n_nets"] == 0:
        return 0.0
    unique_ref = cache["unique_ref"]
    # B3 (2026-05-23): use global pos cache instead of per-node get_pos loop.
    pos_cache = _ensure_pos_cache(plc)
    node_x = pos_cache[unique_ref, 0]
    node_y = pos_cache[unique_ref, 1]
    inv = cache["ref_inv"]
    pin_x = node_x[inv] + cache["x_off"]
    pin_y = node_y[inv] + cache["y_off"]
    starts = cache["net_starts"]
    max_x = np.maximum.reduceat(pin_x, starts)
    min_x = np.minimum.reduceat(pin_x, starts)
    max_y = np.maximum.reduceat(pin_y, starts)
    min_y = np.minimum.reduceat(pin_y, starts)
    per_net = cache["net_weights"] * ((max_x - min_x) + (max_y - min_y))
    return float(per_net.sum())


def _patch_plc_wirelength(plc) -> None:
    """Install the vectorized wirelength on this plc instance (idempotent)."""
    if getattr(plc, "_wl_vec_installed", False):
        return
    # Bind as a bound method via lambda to keep the plc API: plc.get_wirelength()
    plc.get_wirelength = lambda _plc=plc: _vectorized_wirelength(_plc)
    plc._wl_vec_installed = True


def _build_density_cache(plc, benchmark: Benchmark):
    """One-time precomputation per plc for the vectorized density path.

    Density depends on (macro half-widths/heights) which are immutable, and
    macro center positions which change. Cache the immutable arrays once;
    the per-call work is a vectorized scatter-add into grid_occupied.
    """
    if hasattr(plc, "_density_cache"):
        return plc._density_cache
    macro_indices = list(plc.soft_macro_indices) + list(plc.hard_macro_indices)
    n_mod = len(macro_indices)
    half_w = np.empty(n_mod, dtype=np.float64)
    half_h = np.empty(n_mod, dtype=np.float64)
    for k, idx in enumerate(macro_indices):
        m = plc.modules_w_pins[idx]
        half_w[k] = float(m.get_width()) * 0.5
        half_h[k] = float(m.get_height()) * 0.5
    plc._density_cache = {
        "macro_indices": macro_indices,
        "half_w": half_w,
        "half_h": half_h,
        "n_mod": n_mod,
    }
    return plc._density_cache


def _vectorized_get_grid_cells_density(plc) -> "list[float]":
    """Drop-in numpy replacement for plc.get_grid_cells_density().

    Matches the reference: for each soft+hard macro, distribute the area it
    overlaps with each grid cell into grid_occupied, then normalize by
    grid_area to produce grid_cells. Uses cached half-sizes; reads positions
    fresh each call.

    Out-of-canvas behavior is the same as the reference: a macro whose
    bounding box doesn't intersect any in-canvas cell is skipped. The
    reference's row/col clamping logic is reproduced via np.clip.
    """
    cache = plc._density_cache
    grid_col = int(plc.grid_col)
    grid_row = int(plc.grid_row)
    grid_w = float(plc.width / grid_col)
    grid_h = float(plc.height / grid_row)
    plc.grid_width = grid_w
    plc.grid_height = grid_h

    n_cells = grid_col * grid_row
    grid_occupied = np.zeros(n_cells, dtype=np.float64)

    n_mod = cache["n_mod"]
    if n_mod == 0:
        plc.grid_occupied = grid_occupied.tolist()
        plc.grid_cells = [0.0] * n_cells
        return plc.grid_cells

    # B3 (2026-05-23): use global pos cache instead of per-macro get_pos loop.
    macro_indices = cache["macro_indices"]
    pos_cache = _ensure_pos_cache(plc)
    macro_indices_arr = (
        cache.get("macro_indices_arr") if isinstance(cache, dict) else None
    )
    if macro_indices_arr is None:
        macro_indices_arr = np.asarray(macro_indices, dtype=np.int64)
        cache["macro_indices_arr"] = macro_indices_arr
    pos_x = pos_cache[macro_indices_arr, 0]
    pos_y = pos_cache[macro_indices_arr, 1]

    half_w = cache["half_w"]
    half_h = cache["half_h"]
    x_min = pos_x - half_w
    x_max = pos_x + half_w
    y_min = pos_y - half_h
    y_max = pos_y + half_h

    # Mirror the reference's grid cell location (floor, then clamp). Floor
    # at edges gets corrected by the clamping step below; OOB modules
    # (both corners outside canvas) are filtered out.
    bl_col = np.floor(x_min / grid_w).astype(np.int64)
    bl_row = np.floor(y_min / grid_h).astype(np.int64)
    ur_col = np.floor(x_max / grid_w).astype(np.int64)
    ur_row = np.floor(y_max / grid_h).astype(np.int64)

    # OOB skip: if either corner pair places the macro entirely outside
    # the canvas, the reference skips the macro. Mirror via a mask.
    in_bounds = (ur_row >= 0) & (ur_col >= 0) & (bl_row <= grid_row - 1) & (bl_col <= grid_col - 1)
    bl_col = np.clip(bl_col, 0, grid_col - 1)
    bl_row = np.clip(bl_row, 0, grid_row - 1)
    ur_col = np.clip(ur_col, 0, grid_col - 1)
    ur_row = np.clip(ur_row, 0, grid_row - 1)

    # Fully-batched scatter via np.bincount (weights). Mirrors the structure
    # of _apply_macro_routing's vectorized rectangle expansion:
    #   1. Filter to in-bounds macros.
    #   2. Per-cell (macro_idx, row_offset, col_offset) via flat enumeration.
    #   3. Compute per-cell overlap area = ox * oy.
    #   4. bincount over flat cell indices.
    if not in_bounds.any():
        grid_area = grid_w * grid_h
        plc.grid_occupied = grid_occupied.tolist()
        plc.grid_cells = (grid_occupied / grid_area).tolist()
        return plc.grid_cells

    sel = np.where(in_bounds)[0]
    bl_col_s = bl_col[sel]
    bl_row_s = bl_row[sel]
    ur_col_s = ur_col[sel]
    ur_row_s = ur_row[sel]
    x_min_s = x_min[sel]
    x_max_s = x_max[sel]
    y_min_s = y_min[sel]
    y_max_s = y_max[sel]

    n_rows_per = (ur_row_s - bl_row_s + 1).astype(np.int64)
    n_cols_per = (ur_col_s - bl_col_s + 1).astype(np.int64)
    n_cells_per = n_rows_per * n_cols_per
    total = int(n_cells_per.sum())
    if total == 0:
        grid_area = grid_w * grid_h
        plc.grid_occupied = grid_occupied.tolist()
        plc.grid_cells = (grid_occupied / grid_area).tolist()
        return plc.grid_cells

    macro_idx = np.repeat(np.arange(sel.size, dtype=np.int64), n_cells_per)
    cum = np.zeros(sel.size + 1, dtype=np.int64)
    np.cumsum(n_cells_per, out=cum[1:])
    local_idx = np.arange(total, dtype=np.int64) - np.repeat(cum[:-1], n_cells_per)
    n_cols_per_cell = n_cols_per[macro_idx]
    row_off = local_idx // n_cols_per_cell
    col_off = local_idx - row_off * n_cols_per_cell
    rr_g = bl_row_s[macro_idx] + row_off
    cc_g = bl_col_s[macro_idx] + col_off
    flat_idx_cells = rr_g * grid_col + cc_g

    cell_xmin = grid_w * cc_g.astype(np.float64)
    cell_xmax = grid_w * (cc_g + 1).astype(np.float64)
    cell_ymin = grid_h * rr_g.astype(np.float64)
    cell_ymax = grid_h * (rr_g + 1).astype(np.float64)
    x_max_pc = x_max_s[macro_idx]
    x_min_pc = x_min_s[macro_idx]
    y_max_pc = y_max_s[macro_idx]
    y_min_pc = y_min_s[macro_idx]
    ox = np.minimum(cell_xmax, x_max_pc) - np.maximum(cell_xmin, x_min_pc)
    oy = np.minimum(cell_ymax, y_max_pc) - np.maximum(cell_ymin, y_min_pc)
    np.maximum(ox, 0.0, out=ox)
    np.maximum(oy, 0.0, out=oy)
    ov = ox * oy

    grid_occupied = np.bincount(flat_idx_cells, weights=ov, minlength=n_cells)
    grid_area = grid_w * grid_h
    grid_cells = grid_occupied / grid_area
    plc.grid_occupied = grid_occupied.tolist()
    plc.grid_cells = grid_cells.tolist()
    return plc.grid_cells


def _patch_plc_density(plc, benchmark: Benchmark) -> None:
    """Install vectorized density on this plc instance (idempotent)."""
    if getattr(plc, "_density_vec_installed", False):
        return
    _build_density_cache(plc, benchmark)
    plc.get_grid_cells_density = lambda _plc=plc: _vectorized_get_grid_cells_density(_plc)
    plc._density_vec_installed = True


# ---------------------------------------------------------------------------
# Vectorized congestion (get_routing)
# ---------------------------------------------------------------------------
# On ibm10 the scalar plc.get_routing takes ~24.6s per call — dominant cost
# of every scoring call. The native Python loop processes ~50000 nets serially
# with per-net Python overhead (4+ method calls per pin, set-build, branchy
# L/T routing). This vectorized replacement:
#   1. Reuses the per-pin cache built for wirelength (ref_idx + offset
#      arrays) to compute all pin grid cells in one numpy gather.
#   2. Buckets nets by unique-gcell count (1/2/3/many). 2-pin nets — the
#      majority — get batched into flat (source_row/col, sink_row/col,
#      weight) arrays and applied via the difference-array prefix-sum trick
#      (O(strips + grid) rather than O(strip_length × strips)). 3-pin nets
#      are rare; they get a small Python loop matching __three_pin_net_routing
#      exactly. ≥4-gcell nets fan out into source→sink 2-pin edges.
#   3. Hard-macro routing: vectorized per-cell overlap × vrouting/hrouting
#      alloc, then partial-overlap correction.
#   4. Smoothing: 1-D box-blur via cumsum.
# Goal: 24.6s → <5s on ibm10; matches scalar output exactly (integer grid-
# cell indices + sum-of-weights — no FP order sensitivity).
# ---------------------------------------------------------------------------


def _build_cong_cache(plc, benchmark: Benchmark):
    """One-time precomputation for vectorized get_routing.

    Reuses _wl_vec_cache for per-pin (ref_idx, offset) arrays. Adds:
      - per-net weight (derived from driver's get_weight)
      - per-net pin-range starts/lengths
      - hard macro arrays (idx, half_w, half_h)
    """
    if hasattr(plc, "_cong_cache"):
        return plc._cong_cache
    wl = _build_wl_cache(plc)

    # Per-net pin lengths (end - start). Last net runs to n_pins.
    starts = wl["net_starts"]
    n_nets = len(starts)
    n_pins = wl["n_pins"]
    if n_nets == 0:
        ends = np.zeros(0, dtype=np.int64)
    else:
        ends = np.concatenate([starts[1:], np.array([n_pins], dtype=np.int64)])
    lengths = ends - starts

    # Hard macro arrays
    hard_indices = list(plc.hard_macro_indices)
    n_hard = len(hard_indices)
    hard_half_w = np.empty(n_hard, dtype=np.float64)
    hard_half_h = np.empty(n_hard, dtype=np.float64)
    for k, idx in enumerate(hard_indices):
        m = plc.modules_w_pins[idx]
        hard_half_w[k] = float(m.get_width()) * 0.5
        hard_half_h[k] = float(m.get_height()) * 0.5

    # B4 (2026-05-24): pre-compute the dispatch structures that only depend
    # on net topology (not positions). Each call to `_vectorized_get_routing`
    # previously rebuilt these via np.where / np.repeat / np.cumsum / etc.
    # Caching saves ~1-3ms per call on ibm10 (profile: dispatch_len_big was
    # 3.49ms / 28% of cost — most of it was these rebuilds).
    idx2_cache = np.where(lengths == 2)[0]
    s2_cache = starts[idx2_cache] if idx2_cache.size else np.zeros(0, dtype=np.int64)
    s2p1_cache = s2_cache + 1

    idx3_cache = np.where(lengths == 3)[0]
    if idx3_cache.size:
        s3_cache = starts[idx3_cache]
        s3p1_cache = s3_cache + 1
        s3p2_cache = s3_cache + 2
    else:
        s3_cache = s3p1_cache = s3p2_cache = np.zeros(0, dtype=np.int64)

    idx_big_cache = np.where(lengths >= 4)[0]
    if idx_big_cache.size:
        starts_big_cache = starts[idx_big_cache]
        lengths_big_cache = lengths[idx_big_cache]
        sink_lens_cache = lengths_big_cache - 1
        sink_total_cache = int(sink_lens_cache.sum())
        B_cache = idx_big_cache.size
        if sink_total_cache > 0:
            net_local_ids_cache = np.repeat(
                np.arange(B_cache, dtype=np.int64), sink_lens_cache
            )
            cum_sink_starts_cache = np.zeros(B_cache + 1, dtype=np.int64)
            np.cumsum(sink_lens_cache, out=cum_sink_starts_cache[1:])
            offset_in_sinks_cache = (
                np.arange(sink_total_cache, dtype=np.int64)
                - np.repeat(cum_sink_starts_cache[:-1], sink_lens_cache)
            )
            global_pin_idx_cache = (
                (starts_big_cache + 1)[net_local_ids_cache] + offset_in_sinks_cache
            )
        else:
            net_local_ids_cache = np.zeros(0, dtype=np.int64)
            cum_sink_starts_cache = np.zeros(1, dtype=np.int64)
            global_pin_idx_cache = np.zeros(0, dtype=np.int64)
    else:
        starts_big_cache = np.zeros(0, dtype=np.int64)
        lengths_big_cache = np.zeros(0, dtype=np.int64)
        sink_lens_cache = np.zeros(0, dtype=np.int64)
        sink_total_cache = 0
        B_cache = 0
        net_local_ids_cache = np.zeros(0, dtype=np.int64)
        cum_sink_starts_cache = np.zeros(1, dtype=np.int64)
        global_pin_idx_cache = np.zeros(0, dtype=np.int64)

    plc._cong_cache = {
        "starts": starts,
        "ends": ends,
        "lengths": lengths,
        "n_nets": n_nets,
        "hard_indices": hard_indices,
        "hard_half_w": hard_half_w,
        "hard_half_h": hard_half_h,
        "n_hard": n_hard,
        # B4 dispatch caches:
        "idx2": idx2_cache,
        "s2": s2_cache,
        "s2p1": s2p1_cache,
        "idx3": idx3_cache,
        "s3": s3_cache,
        "s3p1": s3p1_cache,
        "s3p2": s3p2_cache,
        "idx_big": idx_big_cache,
        "starts_big": starts_big_cache,
        "lengths_big": lengths_big_cache,
        "sink_lens": sink_lens_cache,
        "sink_total": sink_total_cache,
        "B_big": B_cache,
        "net_local_ids": net_local_ids_cache,
        "cum_sink_starts": cum_sink_starts_cache,
        "global_pin_idx": global_pin_idx_cache,
    }
    return plc._cong_cache


def _apply_h_strips_batch(H_flat: np.ndarray, row: np.ndarray,
                           col_lo: np.ndarray, col_hi: np.ndarray,
                           weight: np.ndarray, grid_row: int, grid_col: int) -> None:
    """Batched H-strip add: for each entry, H_flat[row, col_lo:col_hi] += weight.
    Uses (grid_row, grid_col+1) difference array → cumsum across cols."""
    if row.size == 0:
        return
    h_events = np.zeros((grid_row, grid_col + 1), dtype=np.float64)
    h_flat = h_events.ravel()
    base = row * (grid_col + 1)
    np.add.at(h_flat, base + col_lo, weight)
    np.add.at(h_flat, base + col_hi, -weight)
    H_flat += np.cumsum(h_events, axis=1)[:, :grid_col].ravel()


def _apply_v_strips_batch(V_flat: np.ndarray, col: np.ndarray,
                           row_lo: np.ndarray, row_hi: np.ndarray,
                           weight: np.ndarray, grid_row: int, grid_col: int) -> None:
    """Batched V-strip add: for each entry, V_flat[row_lo:row_hi, col] += weight.
    Uses (grid_col, grid_row+1) col-major difference array → cumsum across rows."""
    if col.size == 0:
        return
    v_events = np.zeros((grid_col, grid_row + 1), dtype=np.float64)
    v_flat = v_events.ravel()
    base = col * (grid_row + 1)
    np.add.at(v_flat, base + row_lo, weight)
    np.add.at(v_flat, base + row_hi, -weight)
    V_flat += np.cumsum(v_events, axis=1)[:, :grid_row].T.ravel()


def _apply_2pin_routing(H_flat: np.ndarray, V_flat: np.ndarray,
                         src_row: np.ndarray, src_col: np.ndarray,
                         snk_row: np.ndarray, snk_col: np.ndarray,
                         weight: np.ndarray, grid_row: int, grid_col: int) -> None:
    """Batched 2-pin L-routing via difference-array prefix-sum.

    Mirrors __two_pin_net_routing exactly:
      H_routing[source_row, col_min : col_max] += weight
      V_routing[row_min : row_max, sink_col]  += weight
    """
    if src_row.size == 0:
        return
    col_min = np.minimum(src_col, snk_col)
    col_max = np.maximum(src_col, snk_col)
    _apply_h_strips_batch(H_flat, src_row, col_min, col_max, weight, grid_row, grid_col)
    row_min = np.minimum(src_row, snk_row)
    row_max = np.maximum(src_row, snk_row)
    _apply_v_strips_batch(V_flat, snk_col, row_min, row_max, weight, grid_row, grid_col)


def _apply_3pin_routing_vec(H_flat: np.ndarray, V_flat: np.ndarray,
                             g0_flat: np.ndarray, g1_flat: np.ndarray,
                             g2_flat: np.ndarray, weights: np.ndarray,
                             grid_row: int, grid_col: int) -> None:
    """Vectorized __three_pin_net_routing — bit-equivalent to the scalar
    reference (_apply_3pin_routing). Operates on 3 flat-gcell arrays
    + weights, dispatches all four branches as batched H/V strip adds.

    Each net's 3 gcells are first sorted by (col, row). Cases 1-3 use that
    ordering; case 4 (T-routing) requires a second sort by (row, col).
    """
    if g0_flat.size == 0:
        return
    n = g0_flat.size
    # Convert flat → (row, col) and stack
    y_all = np.stack([g0_flat // grid_col, g1_flat // grid_col, g2_flat // grid_col], axis=1).astype(np.int64)
    x_all = np.stack([g0_flat % grid_col, g1_flat % grid_col, g2_flat % grid_col], axis=1).astype(np.int64)
    w = np.asarray(weights, dtype=np.float64)
    # Sort each net's 3 points by (col asc, row asc)
    BIG = int(max(grid_row, grid_col)) + 16
    key = x_all * BIG + y_all
    order = np.argsort(key, axis=1, kind="stable")
    y = np.take_along_axis(y_all, order, axis=1)
    x = np.take_along_axis(x_all, order, axis=1)
    y1 = y[:, 0]; y2 = y[:, 1]; y3 = y[:, 2]
    x1 = x[:, 0]; x2 = x[:, 1]; x3 = x[:, 2]

    # Case 1: L-routing — x1<x2<x3 AND y2 strictly between y1 and y3
    case1 = (x1 < x2) & (x2 < x3) & (np.minimum(y1, y3) < y2) & (np.maximum(y1, y3) > y2)
    # Case 2: x2==x3, x1<x2, y1 < min(y2,y3), NOT case1
    case2 = (~case1) & (x2 == x3) & (x1 < x2) & (y1 < np.minimum(y2, y3))
    # Case 3: y2==y3, NOT case1, NOT case2
    case3 = (~case1) & (~case2) & (y2 == y3)
    case4 = ~(case1 | case2 | case3)

    h_rows: "list[np.ndarray]" = []
    h_los: "list[np.ndarray]" = []
    h_his: "list[np.ndarray]" = []
    h_ws: "list[np.ndarray]" = []
    v_cols: "list[np.ndarray]" = []
    v_los: "list[np.ndarray]" = []
    v_his: "list[np.ndarray]" = []
    v_ws: "list[np.ndarray]" = []

    if case1.any():
        m = case1
        wm = w[m]
        # H y1 [x1..x2], y2 [x2..x3]
        h_rows.append(y1[m]); h_los.append(x1[m]); h_his.append(x2[m]); h_ws.append(wm)
        h_rows.append(y2[m]); h_los.append(x2[m]); h_his.append(x3[m]); h_ws.append(wm)
        # V x2 [min(y1,y2)..max(y1,y2)], x3 [min(y2,y3)..max(y2,y3)]
        v_cols.append(x2[m]); v_los.append(np.minimum(y1[m], y2[m])); v_his.append(np.maximum(y1[m], y2[m])); v_ws.append(wm)
        v_cols.append(x3[m]); v_los.append(np.minimum(y2[m], y3[m])); v_his.append(np.maximum(y2[m], y3[m])); v_ws.append(wm)

    if case2.any():
        m = case2
        wm = w[m]
        h_rows.append(y1[m]); h_los.append(x1[m]); h_his.append(x2[m]); h_ws.append(wm)
        v_cols.append(x2[m]); v_los.append(y1[m]); v_his.append(np.maximum(y2[m], y3[m])); v_ws.append(wm)

    if case3.any():
        m = case3
        wm = w[m]
        h_rows.append(y1[m]); h_los.append(x1[m]); h_his.append(x2[m]); h_ws.append(wm)
        h_rows.append(y2[m]); h_los.append(x2[m]); h_his.append(x3[m]); h_ws.append(wm)
        v_cols.append(x2[m]); v_los.append(np.minimum(y2[m], y1[m])); v_his.append(np.maximum(y2[m], y1[m])); v_ws.append(wm)

    if case4.any():
        m = case4
        wm = w[m]
        # Re-sort by (row asc, col asc) — matches scalar's `sorted(temp)` which
        # sorts tuples lexicographically by (row, col).
        y_t = y_all[m]; x_t = x_all[m]
        key_t = y_t * BIG + x_t
        order_t = np.argsort(key_t, axis=1, kind="stable")
        y_t = np.take_along_axis(y_t, order_t, axis=1)
        x_t = np.take_along_axis(x_t, order_t, axis=1)
        y1t = y_t[:, 0]; y2t = y_t[:, 1]; y3t = y_t[:, 2]
        x1t = x_t[:, 0]; x2t = x_t[:, 1]; x3t = x_t[:, 2]
        xmin_t = np.minimum(np.minimum(x1t, x2t), x3t)
        xmax_t = np.maximum(np.maximum(x1t, x2t), x3t)
        h_rows.append(y2t); h_los.append(xmin_t); h_his.append(xmax_t); h_ws.append(wm)
        v_cols.append(x1t); v_los.append(np.minimum(y1t, y2t)); v_his.append(np.maximum(y1t, y2t)); v_ws.append(wm)
        v_cols.append(x3t); v_los.append(np.minimum(y2t, y3t)); v_his.append(np.maximum(y2t, y3t)); v_ws.append(wm)

    if h_rows:
        rows = np.concatenate(h_rows)
        los = np.concatenate(h_los)
        his = np.concatenate(h_his)
        ws_h = np.concatenate(h_ws)
        nz = los != his
        if nz.any():
            _apply_h_strips_batch(H_flat, rows[nz], los[nz], his[nz], ws_h[nz], grid_row, grid_col)
    if v_cols:
        cols = np.concatenate(v_cols)
        rlos = np.concatenate(v_los)
        rhis = np.concatenate(v_his)
        ws_v = np.concatenate(v_ws)
        nz = rlos != rhis
        if nz.any():
            _apply_v_strips_batch(V_flat, cols[nz], rlos[nz], rhis[nz], ws_v[nz], grid_row, grid_col)


def _apply_3pin_routing(H_flat: np.ndarray, V_flat: np.ndarray,
                         gcells_per_net: "list[list[tuple[int, int]]]",
                         weights: "list[float]",
                         grid_col: int) -> None:
    """Match __three_pin_net_routing exactly. Per-net Python loop; only
    fires on nets with exactly 3 unique gcells (uncommon).

    Kept for reference / fallback; the vectorized dispatch (_apply_3pin_routing_vec)
    is what _vectorized_get_routing actually calls.
    """
    for gcells, weight in zip(gcells_per_net, weights):
        temp = sorted(gcells, key=lambda x: (x[1], x[0]))
        y1, x1 = temp[0]
        y2, x2 = temp[1]
        y3, x3 = temp[2]
        if x1 < x2 and x2 < x3 and min(y1, y3) < y2 and max(y1, y3) > y2:
            # L-routing
            t = sorted(temp, key=lambda x: (x[1], x[0]))
            y1, x1 = t[0]; y2, x2 = t[1]; y3, x3 = t[2]
            for col in range(x1, x2):
                H_flat[y1 * grid_col + col] += weight
            for col in range(x2, x3):
                H_flat[y2 * grid_col + col] += weight
            for row in range(min(y1, y2), max(y1, y2)):
                V_flat[row * grid_col + x2] += weight
            for row in range(min(y2, y3), max(y2, y3)):
                V_flat[row * grid_col + x3] += weight
        elif x2 == x3 and x1 < x2 and y1 < min(y2, y3):
            for col in range(x1, x2):
                H_flat[y1 * grid_col + col] += weight
            for row in range(y1, max(y2, y3)):
                V_flat[row * grid_col + x2] += weight
        elif y2 == y3:
            for col in range(x1, x2):
                H_flat[y1 * grid_col + col] += weight
            for col in range(x2, x3):
                H_flat[y2 * grid_col + col] += weight
            for row in range(min(y2, y1), max(y2, y1)):
                V_flat[row * grid_col + x2] += weight
        else:
            # T-routing
            t = sorted(temp)
            y1, x1 = t[0]; y2, x2 = t[1]; y3, x3 = t[2]
            xmin = min(x1, x2, x3); xmax = max(x1, x2, x3)
            for col in range(xmin, xmax):
                H_flat[y2 * grid_col + col] += weight
            for row in range(min(y1, y2), max(y1, y2)):
                V_flat[row * grid_col + x1] += weight
            for row in range(min(y2, y3), max(y2, y3)):
                V_flat[row * grid_col + x3] += weight


def _smooth_routing_cong_vec(routing_flat: np.ndarray, grid_row: int,
                              grid_col: int, smooth_range: int,
                              axis_h: bool) -> np.ndarray:
    """Vectorized __smooth_routing_cong. For each cell, distribute its value
    across a 1D window of `2*smooth_range + 1` cells (clipped at the grid
    edges, divided by the window size). axis_h=True smooths along columns
    (V-routing-style); axis_h=False smooths along rows (H-routing-style).

    Implemented via difference-array prefix-sum trick — O(grid + events)
    rather than O(grid × window).

    NOTE on the reference's quirk: __smooth_routing_cong smooths V along
    COLUMNS (the V loop iterates `for ptr in range(lp, rp+1)` with lp/rp
    clamped to grid_col), and smooths H along ROWS (H loop iterates rows
    via `for ptr in range(lp, up+1)`). The naming is swapped vs intuition
    — V_routing gets a column-axis smooth, H_routing gets a row-axis smooth.
    `axis_h=False` means smooth along the column axis (V-style behavior);
    `axis_h=True` means smooth along the row axis (H-style).
    """
    grid_2d = routing_flat.reshape(grid_row, grid_col)
    sr = smooth_range
    if axis_h:
        # H-style: each cell's value spreads across rows in window
        # [max(0, row-sr), min(grid_row-1, row+sr)].
        # Difference array along axis 0:
        #   events[lp[r], c] += weighted[r, c]
        #   events[up[r]+1, c] -= weighted[r, c]
        # Multiple r values can share the same lp/up (clipping at edges),
        # so np.add.at is required to accumulate duplicates.
        rows = np.arange(grid_row, dtype=np.int64)
        lp = np.maximum(rows - sr, 0)
        up = np.minimum(rows + sr, grid_row - 1)
        cnts = (up - lp + 1).astype(np.float64)
        weighted = grid_2d / cnts[:, None]
        events = np.zeros((grid_row + 1, grid_col), dtype=np.float64)
        np.add.at(events, lp, weighted)
        np.subtract.at(events, up + 1, weighted)
        smoothed = np.cumsum(events, axis=0)[:grid_row]
    else:
        # V-style: each cell's value spreads across cols in window
        # [max(0, col-sr), min(grid_col-1, col+sr)].
        # Difference array along axis 1:
        #   events[r, lp[c]] += weighted[r, c]
        #   events[r, up[c]+1] -= weighted[r, c]
        # Vectorize via tuple-of-arrays advanced indexing.
        cols = np.arange(grid_col, dtype=np.int64)
        lp = np.maximum(cols - sr, 0)
        up = np.minimum(cols + sr, grid_col - 1)
        cnts = (up - lp + 1).astype(np.float64)
        weighted = grid_2d / cnts[None, :]
        events = np.zeros((grid_row, grid_col + 1), dtype=np.float64)
        row_idx = np.broadcast_to(
            np.arange(grid_row, dtype=np.int64)[:, None], (grid_row, grid_col)
        )
        col_lp = np.broadcast_to(lp[None, :], (grid_row, grid_col))
        col_up = np.broadcast_to((up + 1)[None, :], (grid_row, grid_col))
        np.add.at(events, (row_idx, col_lp), weighted)
        np.subtract.at(events, (row_idx, col_up), weighted)
        smoothed = np.cumsum(events, axis=1)[:, :grid_col]
    return smoothed.ravel()


def _apply_macro_routing(V_macro_flat: np.ndarray, H_macro_flat: np.ndarray,
                          hard_x: np.ndarray, hard_y: np.ndarray,
                          half_w: np.ndarray, half_h: np.ndarray,
                          grid_w: float, grid_h: float,
                          grid_row: int, grid_col: int,
                          vrouting_alloc: float, hrouting_alloc: float) -> None:
    """Per-hard-macro routing contribution. Matches __macro_route_over_grid_cell.

    For each cell the macro overlaps:
      x_dist = horizontal overlap between macro and cell
      y_dist = vertical overlap between macro and cell
      V_macro_cong[r,c] += x_dist * vrouting_alloc
      H_macro_cong[r,c] += y_dist * hrouting_alloc
    Plus PARTIAL_OVERLAP corrections (subtract from top row / right col)
    that fire when the macro's bounding-box doesn't fully cover the boundary
    cell along the relevant axis.
    """
    x_min = hard_x - half_w
    x_max = hard_x + half_w
    y_min = hard_y - half_h
    y_max = hard_y + half_h
    bl_col = np.floor(x_min / grid_w).astype(np.int64)
    bl_row = np.floor(y_min / grid_h).astype(np.int64)
    ur_col = np.floor(x_max / grid_w).astype(np.int64)
    ur_row = np.floor(y_max / grid_h).astype(np.int64)
    # Mirror reference's OOB skip
    in_bounds = (ur_row >= 0) & (ur_col >= 0) & (bl_row <= grid_row - 1) & (bl_col <= grid_col - 1)
    bl_col = np.clip(bl_col, 0, grid_col - 1)
    bl_row = np.clip(bl_row, 0, grid_row - 1)
    ur_col = np.clip(ur_col, 0, grid_col - 1)
    ur_row = np.clip(ur_row, 0, grid_row - 1)

    if not in_bounds.any():
        return

    # Restrict to in-bounds macros
    sel = np.where(in_bounds)[0]
    bl_col_s = bl_col[sel]
    bl_row_s = bl_row[sel]
    ur_col_s = ur_col[sel]
    ur_row_s = ur_row[sel]
    x_min_s = x_min[sel]
    x_max_s = x_max[sel]
    y_min_s = y_min[sel]
    y_max_s = y_max[sel]

    n_rows_per = (ur_row_s - bl_row_s + 1).astype(np.int64)
    n_cols_per = (ur_col_s - bl_col_s + 1).astype(np.int64)
    n_cells_per = n_rows_per * n_cols_per
    total = int(n_cells_per.sum())
    if total == 0:
        return

    # Per-cell (macro_idx, row_offset, col_offset) via flat enumeration
    macro_idx = np.repeat(np.arange(sel.size, dtype=np.int64), n_cells_per)
    cum = np.zeros(sel.size + 1, dtype=np.int64)
    np.cumsum(n_cells_per, out=cum[1:])
    local_idx = np.arange(total, dtype=np.int64) - np.repeat(cum[:-1], n_cells_per)
    n_cols_per_cell = n_cols_per[macro_idx]
    row_off = local_idx // n_cols_per_cell
    col_off = local_idx - row_off * n_cols_per_cell

    rr_g = bl_row_s[macro_idx] + row_off
    cc_g = bl_col_s[macro_idx] + col_off
    flat_idx = rr_g * grid_col + cc_g

    # Per-cell overlap distances (x varies with col, y varies with row)
    cell_xmin = grid_w * cc_g.astype(np.float64)
    cell_xmax = grid_w * (cc_g + 1).astype(np.float64)
    cell_ymin = grid_h * rr_g.astype(np.float64)
    cell_ymax = grid_h * (rr_g + 1).astype(np.float64)
    x_max_pc = x_max_s[macro_idx]
    x_min_pc = x_min_s[macro_idx]
    y_max_pc = y_max_s[macro_idx]
    y_min_pc = y_min_s[macro_idx]
    x_dist = np.minimum(cell_xmax, x_max_pc) - np.maximum(cell_xmin, x_min_pc)
    y_dist = np.minimum(cell_ymax, y_max_pc) - np.maximum(cell_ymin, y_min_pc)
    np.maximum(x_dist, 0.0, out=x_dist)
    np.maximum(y_dist, 0.0, out=y_dist)

    np.add.at(V_macro_flat, flat_idx, x_dist * vrouting_alloc)
    np.add.at(H_macro_flat, flat_idx, y_dist * hrouting_alloc)

    # ----- PARTIAL_OVERLAP corrections ----------------------------------
    # Mirror the scalar reference: when a macro spans >1 row AND any of its
    # top/bottom rows is a partial overlap (y_dist != grid_h), subtract the
    # per-column x_dist from V at the top (ur_row) row. Symmetric for H/cols.
    #
    # With grid alignment, only bl_row and ur_row can be partial — middle
    # rows always have y_dist == grid_h. The trigger reduces to:
    #   ur_row != bl_row AND (y_min != grid_h*bl_row OR y_max != grid_h*(ur_row+1)).
    tol = 1e-5
    spans_rows = ur_row_s != bl_row_s
    bot_partial = np.abs((grid_h * (bl_row_s + 1) - y_min_s) - grid_h) > tol
    top_partial = np.abs((y_max_s - grid_h * ur_row_s) - grid_h) > tol
    partial_v = spans_rows & (bot_partial | top_partial)
    if partial_v.any():
        ur_off_per_macro = (ur_row_s - bl_row_s).astype(np.int64)
        mask = partial_v[macro_idx] & (row_off == ur_off_per_macro[macro_idx])
        if mask.any():
            np.subtract.at(
                V_macro_flat,
                ur_row_s[macro_idx[mask]] * grid_col + cc_g[mask],
                x_dist[mask] * vrouting_alloc,
            )

    spans_cols = ur_col_s != bl_col_s
    left_partial = np.abs((grid_w * (bl_col_s + 1) - x_min_s) - grid_w) > tol
    right_partial = np.abs((x_max_s - grid_w * ur_col_s) - grid_w) > tol
    partial_h = spans_cols & (left_partial | right_partial)
    if partial_h.any():
        ur_coff_per_macro = (ur_col_s - bl_col_s).astype(np.int64)
        mask = partial_h[macro_idx] & (col_off == ur_coff_per_macro[macro_idx])
        if mask.any():
            np.subtract.at(
                H_macro_flat,
                rr_g[mask] * grid_col + ur_col_s[macro_idx[mask]],
                y_dist[mask] * hrouting_alloc,
            )


def _apply_net_routing_subset(
    plc,
    net_indices: np.ndarray,
    weight_mult: float,
    H_flat: np.ndarray,
    V_flat: np.ndarray,
) -> None:
    """B3 phase 4 (2026-05-24): per-net routing contribution for a SUBSET of
    nets, applied to in-place flat arrays with a signed weight multiplier.

    Mirrors `_vectorized_get_routing`'s per-net dispatch (2-pin / 3-pin /
    ≥4-pin steiner), but operates on `net_indices` only. `weight_mult=+1`
    adds contributions; `weight_mult=-1` subtracts them (for delta updates
    when a swap changes the touched-net set's routing).

    Does NOT touch macro routing (use `_apply_macro_routing_subset`) and
    does NOT smooth (caller handles smoothing once per swap).

    Pin positions read from `plc._global_pos_cache` (B3 phase 1). For
    efficient subset processing, pin_gcell is computed only for the
    touched pins (a small fraction of total pins).
    """
    if len(net_indices) == 0:
        return

    cache = plc._cong_cache
    wl_cache = plc._wl_vec_cache
    grid_col = int(plc.grid_col)
    grid_row = int(plc.grid_row)
    grid_w = float(plc.width / grid_col)
    grid_h = float(plc.height / grid_row)

    starts = cache["starts"]
    lengths = cache["lengths"]
    net_weights = wl_cache["net_weights"]

    # Gather just the touched-net pin indices in the flat pin array.
    starts_s = starts[net_indices]
    lengths_s = lengths[net_indices]
    total_pins = int(lengths_s.sum())
    if total_pins == 0:
        return
    cumsum_lens = np.concatenate([[0], np.cumsum(lengths_s)[:-1]]).astype(np.int64)
    sub_pin_idx_in_flat = (
        np.repeat(starts_s, lengths_s)
        + (np.arange(total_pins, dtype=np.int64) - np.repeat(cumsum_lens, lengths_s))
    )

    # Compute pin_gcell ONLY for touched pins.
    pos_cache = _ensure_pos_cache(plc)
    unique_ref = wl_cache["unique_ref"]
    inv = wl_cache["ref_inv"]
    x_off = wl_cache["x_off"]
    y_off = wl_cache["y_off"]
    pin_ref_local = inv[sub_pin_idx_in_flat]
    pin_x = pos_cache[unique_ref[pin_ref_local], 0] + x_off[sub_pin_idx_in_flat]
    pin_y = pos_cache[unique_ref[pin_ref_local], 1] + y_off[sub_pin_idx_in_flat]
    pin_col = np.clip((pin_x / grid_w).astype(np.int64), 0, grid_col - 1)
    pin_row = np.clip((pin_y / grid_h).astype(np.int64), 0, grid_row - 1)
    pin_gcell = pin_row * grid_col + pin_col  # in the COMPACT subset pin order

    weights_sub = net_weights[net_indices] * weight_mult

    bucket_2_src: list = []
    bucket_2_snk: list = []
    bucket_2_w: list = []
    bucket_3_g0: list = []
    bucket_3_g1: list = []
    bucket_3_g2: list = []
    bucket_3_w: list = []

    # ------ length-2 nets in subset ------
    mask_l2 = lengths_s == 2
    if mask_l2.any():
        local_starts_l2 = cumsum_lens[mask_l2]
        src2 = pin_gcell[local_starts_l2]
        snk2 = pin_gcell[local_starts_l2 + 1]
        sub_mask = src2 != snk2
        if sub_mask.any():
            bucket_2_src.append(src2[sub_mask])
            bucket_2_snk.append(snk2[sub_mask])
            bucket_2_w.append(weights_sub[mask_l2][sub_mask])

    # ------ length-3 nets in subset ------
    mask_l3 = lengths_s == 3
    if mask_l3.any():
        local_starts_l3 = cumsum_lens[mask_l3]
        g0 = pin_gcell[local_starts_l3]
        g1 = pin_gcell[local_starts_l3 + 1]
        g2 = pin_gcell[local_starts_l3 + 2]
        eq01 = g0 == g1
        eq02 = g0 == g2
        eq12 = g1 == g2
        eq_count = eq01.astype(np.int64) + eq02.astype(np.int64) + eq12.astype(np.int64)
        uniq2 = eq_count == 1
        uniq3 = eq_count == 0
        if uniq2.any():
            src_2 = g0[uniq2]
            sink_2 = np.where(eq01[uniq2], g2[uniq2], g1[uniq2])
            bucket_2_src.append(src_2)
            bucket_2_snk.append(sink_2)
            bucket_2_w.append(weights_sub[mask_l3][uniq2])
        if uniq3.any():
            bucket_3_g0.append(g0[uniq3])
            bucket_3_g1.append(g1[uniq3])
            bucket_3_g2.append(g2[uniq3])
            bucket_3_w.append(weights_sub[mask_l3][uniq3])

    # ------ length≥4 nets in subset ------
    mask_l4 = lengths_s >= 4
    if mask_l4.any():
        sub_idx_big = np.where(mask_l4)[0]
        starts_big_local = cumsum_lens[sub_idx_big]  # offsets in the SUBSET pin order
        lengths_big_local = lengths_s[sub_idx_big]
        sink_lens_local = lengths_big_local - 1
        sink_total_local = int(sink_lens_local.sum())
        src_gcells_big = pin_gcell[starts_big_local]
        if sink_total_local > 0:
            B_local = sub_idx_big.size
            net_local_ids_local = np.repeat(np.arange(B_local, dtype=np.int64), sink_lens_local)
            cum_sink_starts_local = np.zeros(B_local + 1, dtype=np.int64)
            np.cumsum(sink_lens_local, out=cum_sink_starts_local[1:])
            offset_in_sinks_local = (
                np.arange(sink_total_local, dtype=np.int64)
                - np.repeat(cum_sink_starts_local[:-1], sink_lens_local)
            )
            global_pin_idx_local = (starts_big_local + 1)[net_local_ids_local] + offset_in_sinks_local
            sink_gcells = pin_gcell[global_pin_idx_local]
            mask_not_src = sink_gcells != src_gcells_big[net_local_ids_local]
            if mask_not_src.any():
                nli_ns = net_local_ids_local[mask_not_src]
                sg_ns = sink_gcells[mask_not_src]
                order = np.lexsort((sg_ns, nli_ns))
                nli_sorted = nli_ns[order]
                sg_sorted = sg_ns[order]
                keep = np.empty(sg_sorted.size, dtype=bool)
                keep[0] = True
                if sg_sorted.size > 1:
                    keep[1:] = (
                        (nli_sorted[1:] != nli_sorted[:-1])
                        | (sg_sorted[1:] != sg_sorted[:-1])
                    )
                nli_uniq = nli_sorted[keep]
                sg_uniq = sg_sorted[keep]
                uniq_sink_counts = np.bincount(nli_uniq, minlength=B_local)
                n_uniq_total = 1 + uniq_sink_counts
                net_is_3 = n_uniq_total == 3
                net_is_starlike = ~net_is_3
                mask_starlike = net_is_starlike[nli_uniq]
                if mask_starlike.any():
                    nli_emit = nli_uniq[mask_starlike]
                    bucket_2_src.append(src_gcells_big[nli_emit])
                    bucket_2_snk.append(sg_uniq[mask_starlike])
                    bucket_2_w.append(weights_sub[sub_idx_big[nli_emit]])
                if net_is_3.any():
                    cum_counts = np.cumsum(uniq_sink_counts)
                    net3_ids = np.where(net_is_3)[0]
                    ends = cum_counts[net3_ids]
                    bucket_3_g0.append(src_gcells_big[net3_ids])
                    bucket_3_g1.append(sg_uniq[ends - 2])
                    bucket_3_g2.append(sg_uniq[ends - 1])
                    bucket_3_w.append(weights_sub[sub_idx_big[net3_ids]])

    if bucket_2_src:
        src_flat = np.concatenate(bucket_2_src)
        snk_flat = np.concatenate(bucket_2_snk)
        w_arr = np.concatenate(bucket_2_w)
        _apply_2pin_routing(
            H_flat, V_flat,
            src_flat // grid_col, src_flat % grid_col,
            snk_flat // grid_col, snk_flat % grid_col,
            w_arr, grid_row, grid_col,
        )
    if bucket_3_g0:
        g0_arr = np.concatenate(bucket_3_g0)
        g1_arr = np.concatenate(bucket_3_g1)
        g2_arr = np.concatenate(bucket_3_g2)
        w_arr3 = np.concatenate(bucket_3_w)
        _apply_3pin_routing_vec(H_flat, V_flat, g0_arr, g1_arr, g2_arr, w_arr3, grid_row, grid_col)


def _apply_macro_routing_subset(
    plc,
    macro_subset: np.ndarray,
    weight_mult: float,
    V_macro_flat: np.ndarray,
    H_macro_flat: np.ndarray,
) -> None:
    """B3 phase 4: per-macro routing contribution for a SUBSET of hard macros.

    `macro_subset` is an int array of indices into `cong_cache["hard_indices"]`
    (i.e., hard-macro slot indices, not module indices).
    """
    if len(macro_subset) == 0:
        return
    cache = plc._cong_cache
    if cache["n_hard"] == 0:
        return
    grid_col = int(plc.grid_col)
    grid_row = int(plc.grid_row)
    grid_w = float(plc.width / grid_col)
    grid_h = float(plc.height / grid_row)

    pos_cache = _ensure_pos_cache(plc)
    hard_indices_arr = cache.get("hard_indices_arr")
    if hard_indices_arr is None:
        hard_indices_arr = np.asarray(cache["hard_indices"], dtype=np.int64)
        cache["hard_indices_arr"] = hard_indices_arr
    sub_module_indices = hard_indices_arr[macro_subset]
    hard_x = pos_cache[sub_module_indices, 0]
    hard_y = pos_cache[sub_module_indices, 1]
    hw_sub = cache["hard_half_w"][macro_subset]
    hh_sub = cache["hard_half_h"][macro_subset]

    # Apply with the requested sign. _apply_macro_routing uses
    # `vrouting_alloc * weight` style multipliers; we just flip the alloc
    # sign for subtraction. (Same effect as -1 on the additive output.)
    _apply_macro_routing(
        V_macro_flat, H_macro_flat, hard_x, hard_y,
        hw_sub, hh_sub,
        grid_w, grid_h, grid_row, grid_col,
        float(plc.vrouting_alloc) * weight_mult,
        float(plc.hrouting_alloc) * weight_mult,
    )


def _vectorized_get_routing(plc) -> None:
    """Drop-in replacement for plc.get_routing().

    Replaces the inner ~25-second Python loop on ibm10 with a vectorized
    numpy pipeline. Sets plc.V_routing_cong / H_routing_cong as Python lists
    (matching the reference's API — `get_horizontal/vertical_routing_congestion`
    return them directly).
    """
    cache = plc._cong_cache
    wl = plc._wl_vec_cache

    # Geometry refresh (matches reference)
    grid_col = int(plc.grid_col)
    grid_row = int(plc.grid_row)
    grid_w = float(plc.width / grid_col)
    grid_h = float(plc.height / grid_row)
    plc.grid_width = grid_w
    plc.grid_height = grid_h
    grid_v_routes = grid_w * plc.vroutes_per_micron
    grid_h_routes = grid_h * plc.hroutes_per_micron
    plc.grid_v_routes = grid_v_routes
    plc.grid_h_routes = grid_h_routes

    n_cells = grid_row * grid_col
    H_flat = np.zeros(n_cells, dtype=np.float64)
    V_flat = np.zeros(n_cells, dtype=np.float64)
    H_macro_flat = np.zeros(n_cells, dtype=np.float64)
    V_macro_flat = np.zeros(n_cells, dtype=np.float64)

    n_nets = wl["n_nets"]
    if n_nets > 0:
        # B3 (2026-05-23): use global pos cache instead of per-node get_pos loop.
        unique_ref = wl["unique_ref"]
        pos_cache = _ensure_pos_cache(plc)
        node_x = pos_cache[unique_ref, 0]
        node_y = pos_cache[unique_ref, 1]
        inv = wl["ref_inv"]
        pin_x = node_x[inv] + wl["x_off"]
        pin_y = node_y[inv] + wl["y_off"]
        # Apply the patched grid-cell location: floor + clamp
        pin_col = np.clip((pin_x / grid_w).astype(np.int64), 0, grid_col - 1)
        pin_row = np.clip((pin_y / grid_h).astype(np.int64), 0, grid_row - 1)
        pin_gcell = pin_row * grid_col + pin_col  # flat per-pin gcell idx

        starts = cache["starts"]
        lengths = cache["lengths"]
        net_weights = wl["net_weights"]

        # Per-net dispatch. Original v1 implementation did np.unique per net
        # in a 28k-iteration Python loop on ibm10 → 663ms. We partition nets
        # by pin count and vectorize the common cases:
        #   - length 2 (most nets): pure numpy, one fast-path.
        #   - length 3: vectorize the "unique-count-from-3" classification,
        #     then dispatch to per-bucket batches.
        #   - length ≥4: fall back to per-net Python (small fraction).
        bucket_2_src_flat: "list[np.ndarray]" = []  # accumulators of flat src gcells
        bucket_2_snk_flat: "list[np.ndarray]" = []
        bucket_2_w_arrs: "list[np.ndarray]" = []

        # 3-pin buckets: 3 flat-gcell arrays + weights (all parallel).
        bucket_3_g0: "list[np.ndarray]" = []
        bucket_3_g1: "list[np.ndarray]" = []
        bucket_3_g2: "list[np.ndarray]" = []
        bucket_3_w_arrs: "list[np.ndarray]" = []

        # --- length-2 vectorized fast path -----------------------------------
        # B4: read pre-cached idx2/s2/s2p1 from cong_cache (topology-fixed).
        idx2 = cache["idx2"]
        if idx2.size > 0:
            src2 = pin_gcell[cache["s2"]]
            snk2 = pin_gcell[cache["s2p1"]]
            mask = src2 != snk2  # same-cell pins → no routing
            if mask.any():
                bucket_2_src_flat.append(src2[mask])
                bucket_2_snk_flat.append(snk2[mask])
                bucket_2_w_arrs.append(net_weights[idx2][mask])

        # --- length-3 vectorized classification ------------------------------
        # For each 3-pin net, count unique gcells among (g0, g1, g2).
        # cases:
        #   all three equal → skip
        #   exactly two distinct → 2-pin edge
        #   all three distinct → 3-pin handler
        # B4: idx3/s3/s3p1/s3p2 from cache (topology-fixed).
        idx3 = cache["idx3"]
        if idx3.size > 0:
            g0 = pin_gcell[cache["s3"]]      # driver
            g1 = pin_gcell[cache["s3p1"]]
            g2 = pin_gcell[cache["s3p2"]]
            eq01 = g0 == g1
            eq02 = g0 == g2
            eq12 = g1 == g2
            # Count of equal-pair conditions:
            #   3 → all equal → n_uniq=1 → skip
            #   1 → exactly one pair equal → n_uniq=2 → 2-pin edge
            #   0 → all distinct → n_uniq=3 → 3-pin handler
            eq_count = eq01.astype(np.int64) + eq02.astype(np.int64) + eq12.astype(np.int64)
            uniq2 = eq_count == 1
            uniq3 = eq_count == 0
            # 2-uniq case: choose the sink that's distinct from driver (g0).
            #   if eq01 and !eq02 → sink = g2
            #   if eq02 and !eq01 → sink = g1
            #   if eq12 and !eq01 → sink = g1  (g1 == g2, distinct from g0)
            # Driver gcell is g0 in all 2-uniq cases.
            mask2 = uniq2
            if mask2.any():
                src_2 = g0[mask2]
                # Sink: g2 when eq01[mask2], else g1
                sink_2 = np.where(eq01[mask2], g2[mask2], g1[mask2])
                bucket_2_src_flat.append(src_2)
                bucket_2_snk_flat.append(sink_2)
                bucket_2_w_arrs.append(net_weights[idx3][mask2])
            # 3-uniq case: pass to vectorized 3-pin handler — directly append
            # the per-axis flat gcell arrays (no per-net Python loop).
            if uniq3.any():
                idx3_uniq3 = idx3[uniq3]
                bucket_3_g0.append(g0[uniq3])
                bucket_3_g1.append(g1[uniq3])
                bucket_3_g2.append(g2[uniq3])
                bucket_3_w_arrs.append(net_weights[idx3_uniq3])

        # --- length ≥4: vectorized batch dispatch ----------------------------
        # Original per-net loop called np.unique 28k× on ibm10 (~62ms). Here
        # we build flat (net_local_id, sink_gcell) pairs for ALL big nets at
        # once, dedup via lexsort, then dispatch by per-net unique count.
        # Source is filtered out of "sinks" before dedup; n_uniq_total = 1 + #unique_sinks.
        # B4: idx_big / starts_big / lengths_big / sink_lens / sink_total / B /
        # net_local_ids / cum_sink_starts / global_pin_idx all pre-cached.
        idx_big = cache["idx_big"]
        if idx_big.size > 0:
            starts_big = cache["starts_big"]
            sink_total = cache["sink_total"]
            src_gcells_big = pin_gcell[starts_big]
            if sink_total > 0:
                B = cache["B_big"]
                net_local_ids = cache["net_local_ids"]
                global_pin_idx = cache["global_pin_idx"]
                sink_gcells = pin_gcell[global_pin_idx]
                # Drop sinks that equal the source gcell
                mask_not_src = sink_gcells != src_gcells_big[net_local_ids]
                if mask_not_src.any():
                    nli_ns = net_local_ids[mask_not_src]
                    sg_ns = sink_gcells[mask_not_src]
                    # Dedup per net via lexsort
                    order = np.lexsort((sg_ns, nli_ns))
                    nli_sorted = nli_ns[order]
                    sg_sorted = sg_ns[order]
                    keep = np.empty(sg_sorted.size, dtype=bool)
                    keep[0] = True
                    if sg_sorted.size > 1:
                        keep[1:] = (
                            (nli_sorted[1:] != nli_sorted[:-1])
                            | (sg_sorted[1:] != sg_sorted[:-1])
                        )
                    nli_uniq = nli_sorted[keep]
                    sg_uniq = sg_sorted[keep]
                    uniq_sink_counts = np.bincount(nli_uniq, minlength=B)
                    n_uniq_total = 1 + uniq_sink_counts
                    # Dispatch:
                    #   n_uniq_total == 3 → 3-pin steiner handler (Python loop on few nets)
                    #   n_uniq_total != 3 (covers 2 and ≥4) → emit (src, sink) edges
                    net_is_3 = n_uniq_total == 3
                    net_is_starlike = ~net_is_3
                    mask_starlike = net_is_starlike[nli_uniq]
                    if mask_starlike.any():
                        nli_emit = nli_uniq[mask_starlike]
                        bucket_2_src_flat.append(src_gcells_big[nli_emit])
                        bucket_2_snk_flat.append(sg_uniq[mask_starlike])
                        bucket_2_w_arrs.append(net_weights[idx_big[nli_emit]])
                    if net_is_3.any():
                        # The 2 unique sinks for each 3-pin net live in sg_uniq
                        # at positions [cum_count-2, cum_count-1]. Vectorize the
                        # gather instead of looping.
                        cum_counts = np.cumsum(uniq_sink_counts)
                        net3_ids = np.where(net_is_3)[0]
                        ends = cum_counts[net3_ids]
                        bucket_3_g0.append(src_gcells_big[net3_ids])
                        bucket_3_g1.append(sg_uniq[ends - 2])
                        bucket_3_g2.append(sg_uniq[ends - 1])
                        bucket_3_w_arrs.append(net_weights[idx_big[net3_ids]])

        # --- Apply 2-pin batch via difference-array --------------------------
        if bucket_2_src_flat:
            src_flat = np.concatenate(bucket_2_src_flat)
            snk_flat = np.concatenate(bucket_2_snk_flat)
            w_arr = np.concatenate(bucket_2_w_arrs)
            _apply_2pin_routing(
                H_flat, V_flat,
                src_flat // grid_col, src_flat % grid_col,
                snk_flat // grid_col, snk_flat % grid_col,
                w_arr, grid_row, grid_col,
            )
        # Apply 3-pin (vectorized batch)
        if bucket_3_g0:
            g0_arr = np.concatenate(bucket_3_g0)
            g1_arr = np.concatenate(bucket_3_g1)
            g2_arr = np.concatenate(bucket_3_g2)
            w_arr3 = np.concatenate(bucket_3_w_arrs)
            _apply_3pin_routing_vec(H_flat, V_flat, g0_arr, g1_arr, g2_arr, w_arr3, grid_row, grid_col)

    # Hard-macro routing contributions
    n_hard = cache["n_hard"]
    if n_hard > 0:
        # B3 (2026-05-23): use global pos cache instead of per-macro get_pos loop.
        hard_indices = cache["hard_indices"]
        hard_indices_arr = cache.get("hard_indices_arr")
        if hard_indices_arr is None:
            hard_indices_arr = np.asarray(hard_indices, dtype=np.int64)
            cache["hard_indices_arr"] = hard_indices_arr
        pos_cache = _ensure_pos_cache(plc)
        hard_x = pos_cache[hard_indices_arr, 0]
        hard_y = pos_cache[hard_indices_arr, 1]
        _apply_macro_routing(
            V_macro_flat, H_macro_flat, hard_x, hard_y,
            cache["hard_half_w"], cache["hard_half_h"],
            grid_w, grid_h, grid_row, grid_col,
            float(plc.vrouting_alloc), float(plc.hrouting_alloc),
        )

    # Normalize by routes-per-cell capacity
    H_flat /= grid_h_routes
    V_flat /= grid_v_routes
    H_macro_flat /= grid_h_routes
    V_macro_flat /= grid_v_routes

    # Smooth + combine
    smooth_range = int(plc.smooth_range)
    if smooth_range > 0:
        V_flat = _smooth_routing_cong_vec(V_flat, grid_row, grid_col, smooth_range, axis_h=False)
        H_flat = _smooth_routing_cong_vec(H_flat, grid_row, grid_col, smooth_range, axis_h=True)

    V_total = V_flat + V_macro_flat
    H_total = H_flat + H_macro_flat

    # B3 phase 3 (2026-05-23): store as numpy arrays instead of Python lists.
    # Saves ~2ms/call on .tolist() conversion. The patched get_congestion_cost
    # (`_patch_plc_congestion_cost`) consumes them via numpy ops, and
    # `_routing_congestion_perturb` already does `np.asarray(...).reshape(...)`
    # so it transparently accepts arrays too.
    plc.V_routing_cong = V_total
    plc.H_routing_cong = H_total
    plc.V_macro_routing_cong = V_macro_flat
    plc.H_macro_routing_cong = H_macro_flat
    plc.FLAG_UPDATE_CONGESTION = False


def _vectorized_get_congestion_cost(plc) -> float:
    """Numpy-fast replacement for `PlacementCost.get_congestion_cost` (B3 phase 3).

    The reference does
        sorted(V_routing_cong + H_routing_cong, reverse=True)
        return sum(top 5%) / cnt
    on Python lists. With ~4500 elements that's ~3ms.

    Numpy via `np.partition`: get the top-cnt elements (unordered) at O(n),
    then mean. Same result (sum-of-top-cnt is order-independent), ~0.3ms.
    """
    if plc.FLAG_UPDATE_CONGESTION:
        plc.get_routing()  # patched to _vectorized_get_routing
    v = plc.V_routing_cong
    h = plc.H_routing_cong
    # Concat. plc may still hold the legacy lists on the very first call
    # (before our get_routing patched-write executes); handle gracefully.
    if isinstance(v, list):
        v = np.asarray(v, dtype=np.float64)
    if isinstance(h, list):
        h = np.asarray(h, dtype=np.float64)
    xx = np.concatenate([v, h])
    n = xx.size
    cnt = int(n * 0.05)  # floor (positive value)
    if cnt == 0:
        return float(xx.max())
    # Top-cnt values via partition (unordered, but mean is order-independent).
    top = np.partition(xx, n - cnt)[n - cnt:]
    return float(top.sum() / cnt)


def _patch_plc_congestion(plc, benchmark: Benchmark) -> None:
    """Install vectorized congestion (get_routing + get_congestion_cost) on this plc."""
    if getattr(plc, "_cong_vec_installed", False):
        return
    _build_wl_cache(plc)
    _build_cong_cache(plc, benchmark)
    plc.get_routing = lambda _plc=plc: _vectorized_get_routing(_plc)
    # B3 phase 3 (2026-05-23): replace get_congestion_cost with numpy-fast version.
    plc.get_congestion_cost = lambda _plc=plc: _vectorized_get_congestion_cost(_plc)
    plc._cong_vec_installed = True


def _ensure_congestion_arrays(plc) -> None:
    """Mirror objective._ensure_congestion_arrays without re-importing."""
    expected_size = plc.grid_col * plc.grid_row
    if len(plc.H_routing_cong) != expected_size:
        # B3 phase 3 (2026-05-23): use numpy arrays to match the new
        # `_vectorized_get_routing` output type. Saves the .tolist()
        # conversion on every score call.
        plc.V_routing_cong = np.zeros(expected_size, dtype=np.float64)
        plc.H_routing_cong = np.zeros(expected_size, dtype=np.float64)
        plc.V_macro_routing_cong = np.zeros(expected_size, dtype=np.float64)
        plc.H_macro_routing_cong = np.zeros(expected_size, dtype=np.float64)


def _fast_set_placement(plc, placement_np: np.ndarray, benchmark: Benchmark) -> None:
    """Faster drop-in for objective._set_placement.

    Three wins vs the reference:
      1. Cache last-applied positions per macro on plc and SKIP set_pos
         when the value matches. Soft macros almost never move after the
         baseline restoration; this collapses thousands of no-op calls per
         score into a single equality check per macro.
      2. Skip pin.set_pos entirely. Verified that every cost path in
         plc_client_os recomputes pin coordinates via __get_pin_position
         (ref_node.get_pos() + pin.get_offset()) — nothing reads pin.x/.y.
         The pin.set_pos calls were dead code defending against a non-issue.
      3. Skip the overlap-metric computation downstream (we never read it).
    """
    n_hard = benchmark.num_hard_macros
    hard_indices = benchmark.hard_macro_indices
    soft_indices = benchmark.soft_macro_indices

    last = getattr(plc, "_last_pos_cache", None)
    if last is None or last.shape != placement_np.shape:
        last = np.full(placement_np.shape, np.nan, dtype=np.float64)
        plc._last_pos_cache = last

    # Global position cache (B3, 2026-05-23): keep `plc._global_pos_cache`
    # synchronized with each set_pos call so the vectorized scoring
    # functions can read positions via fancy indexing instead of looping
    # mods[idx].get_pos().
    pos_cache = _ensure_pos_cache(plc)

    any_changed = False

    # Hard macros
    for i, macro_idx in enumerate(hard_indices):
        x = float(placement_np[i, 0])
        y = float(placement_np[i, 1])
        if last[i, 0] == x and last[i, 1] == y:
            continue
        any_changed = True
        last[i, 0] = x
        last[i, 1] = y
        plc.modules_w_pins[macro_idx].set_pos(x, y)
        pos_cache[macro_idx, 0] = x
        pos_cache[macro_idx, 1] = y

    # Soft macros — usually unchanged after baseline; the equality check
    # short-circuits the per-macro work for the common no-op case.
    for i, macro_idx in enumerate(soft_indices):
        row = n_hard + i
        x = float(placement_np[row, 0])
        y = float(placement_np[row, 1])
        if last[row, 0] == x and last[row, 1] == y:
            continue
        any_changed = True
        last[row, 0] = x
        last[row, 1] = y
        plc.modules_w_pins[macro_idx].set_pos(x, y)
        pos_cache[macro_idx, 0] = x
        pos_cache[macro_idx, 1] = y

    _ensure_congestion_arrays(plc)
    # Only invalidate cached costs if something actually moved. If nothing
    # changed, plc's dirty flags can stay False and the cached values are
    # returned for free.
    if any_changed:
        plc.FLAG_UPDATE_WIRELENGTH = True
        plc.FLAG_UPDATE_DENSITY = True
        plc.FLAG_UPDATE_CONGESTION = True


def _exact_proxy(placement: torch.Tensor, benchmark: Benchmark, plc) -> float:
    """Fast proxy cost: skips overlap metrics, skips unchanged macro updates,
    and uses the vectorized wirelength patch installed on plc.

    Bypasses macro_place.objective.compute_proxy_cost entirely. We never
    consume overlap metrics here; the placer only reads proxy_cost. Saves
    O(n_hard²) pure-Python pair iterations per scoring call (e.g. ~289k on
    ibm17) plus the redundant per-pin set_pos overhead on unchanged macros.
    """
    _patch_plc_wirelength(plc)
    _patch_plc_congestion(plc, benchmark)
    _patch_plc_density(plc, benchmark)
    placement_np = placement.cpu().numpy()
    # Soft macros stay at the positions in `placement` (typically
    # initial.plc) — naive centroid re-snap was tested 2026-05-22 and
    # rejected (ibm04 1.3079 → 1.6465 with blend=1.0). The right approach
    # is A2 (2026-05-24): DREAMPlace soft_movable=True is enabled in the
    # DP launch, so DP-optimized soft positions are carried in dp_pl[n:]
    # for the DP candidate path; non-DP candidates keep initial softs.
    _fast_set_placement(plc, placement_np, benchmark)
    wl = plc.get_cost()
    dens = plc.get_density_cost()
    cong = plc.get_congestion_cost()
    return float(wl + 0.5 * dens + 0.5 * cong)


def _proxy_decomp(placement: torch.Tensor, benchmark: Benchmark, plc):
    """(proxy, wl, 0.5*den, 0.5*cong) — the WEIGHTED proxy split. Re-scores the
    placement (mutates plc state), so use only in diagnostic contexts."""
    p = _exact_proxy(placement, benchmark, plc)
    wl = float(plc.get_cost())
    den = 0.5 * float(plc.get_density_cost())
    cong = 0.5 * float(plc.get_congestion_cost())
    return p, wl, den, cong


def _dp_recoverability_probe(
    dp_placements, best_score, n, cw, ch, hw, hh, sizes, movable, plc, benchmark
):
    """DP_PROBE ceiling test (2026-05-26): can a GENEROUS, ungated post-hoc
    congestion treatment of the best DP basin beat the cong-grad-from-baseline
    'best'? Phase 7 caps cong-grad-from-DP at 3 iters / frac=0.04 with abandon-
    gates; here we remove all gates — a multi-frac descent (0.08/0.04/0.02, up
    to 25 iters each, accept-on-proxy) followed by a full 20s 2-opt from the
    relieved basin. If this still loses to best, post-hoc repair is empirically
    ruled out (relieving DP's congestion trades away its wl/den edge faster than
    it gains), which justifies fusing congestion INTO the DREAMPlace objective.
    """
    if not dp_placements:
        _log("  [DP_PROBE] no DP candidates; skipping")
        return
    dp_tag, dp_raw, dp_pl0 = min(dp_placements, key=lambda e: e[1])
    _log(f"  [DP_PROBE] seed=dp[{dp_tag}] raw={dp_raw:.4f}  best={best_score:.4f}")
    rng = np.random.RandomState(777)
    cur_pl = dp_pl0.clone()
    cur_hard = np.stack(
        [dp_pl0[:n, 0].numpy(), dp_pl0[:n, 1].numpy()], axis=1
    ).astype(np.float64)
    cur_score = float(_exact_proxy(cur_pl, benchmark, plc))
    for frac in (0.08, 0.04, 0.02):
        no_improve = 0
        for _it in range(25):
            # Re-score cur so plc's congestion map matches cur_hard before the
            # gradient step (correct gradient, not stale).
            _exact_proxy(cur_pl, benchmark, plc)
            perturbed = _routing_congestion_perturb(
                cur_hard, plc, benchmark, n, cw, ch, hw, hh, movable,
                frac=frac, rng=rng,
            )
            leg = _will_legalize(perturbed, movable, sizes, hw, hh, cw, ch, n)
            trial = cur_pl.clone()
            trial[:n, 0] = torch.tensor(leg[:, 0], dtype=torch.float32)
            trial[:n, 1] = torch.tensor(leg[:, 1], dtype=torch.float32)
            s = float(_exact_proxy(trial, benchmark, plc))
            if s < cur_score - 1e-5:
                cur_score, cur_pl, cur_hard = s, trial, leg.astype(np.float64)
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= 4:
                    break
    _log(f"  [DP_PROBE] after multi-frac cong-grad descent: {cur_score:.4f}")
    # Full 2-opt from the congestion-relieved DP basin.
    try:
        scorer = IncrementalScorer(plc, benchmark, cur_pl.cpu().numpy().astype(np.float64))
    except Exception:
        scorer = None
    scratch = cur_pl.clone()

    def _ps(pa, _s=scratch):
        p32 = torch.from_numpy(np.ascontiguousarray(pa)).float()
        _s[:n, 0] = p32[:, 0]
        _s[:n, 1] = p32[:, 1]
        return float(_exact_proxy(_s, benchmark, plc))

    opt_pos, ac, fs, sc = _two_opt_proxy_swap(
        cur_hard, sizes, hw, hh, cw, ch, movable, n,
        score_fn=_ps, initial_score=cur_score, k_neighbors=20, max_iters=6,
        deadline=time.monotonic() + 20.0, incremental_scorer=scorer,
    )
    final_pl = cur_pl.clone()
    final_pl[:n, 0] = torch.tensor(opt_pos[:, 0], dtype=torch.float32)
    final_pl[:n, 1] = torch.tensor(opt_pos[:, 1], dtype=torch.float32)
    pf, wf, df, cf = _proxy_decomp(final_pl, benchmark, plc)
    verdict = "BEATS best" if pf < best_score - 1e-4 else "LOSES to best"
    _log(f"  [DP_PROBE] FINAL dp-basin post-hoc: proxy={pf:.4f} "
         f"(wl={wf:.4f} den={df:.4f} cong={cf:.4f})  -> {verdict} "
         f"(best={best_score:.4f}, {ac} 2opt accepts)")


# ---------------------------------------------------------------------------
# Incremental scorer for 2-opt (B3 phase 2, 2026-05-23)
# ---------------------------------------------------------------------------

class IncrementalScorer:
    """Per-swap incremental proxy scorer used inside `_two_opt_proxy_swap`.

    Phase 2 (this) does incremental wirelength only — `density` and
    `congestion` still come from `plc.get_density_cost` /
    `plc.get_congestion_cost` (full recompute via the dirty-flag path).
    Phase 3 will tackle congestion incremental.

    WL incremental:
      - Build `macro_to_nets[macro_idx] = array of net indices` once.
      - Cache per-net HPWL (`per_net_hpwl`) once after baseline.
      - For a swap (i_hard, j_hard): touched_nets = macro_to_nets[i] ∪
        macro_to_nets[j] (typically ~50-200 of ~28k nets).
      - Recompute HPWL for touched nets only via gather + reduceat over a
        compact pin range.
      - `delta_wl = sum((new_hpwl - per_net_hpwl) * net_weights) for touched`.
      - `new_total_wl = total_wl + delta_wl`.

    State management:
      - The scorer mirrors plc's set_pos calls: `score_swap` applies the
        swap to plc, computes, then reverts. `commit_swap` applies and
        persists the state (also updates `per_net_hpwl`).
      - Caller (2-opt) must call `commit_swap` after an accept; reject
        requires no action because score_swap already reverted plc.

    Indexing notes:
      - `i_hard`, `j_hard` are indices into `benchmark.hard_macro_indices`
        (i.e., 0 ≤ i_hard < n_hard, the same indexing used by
        `_two_opt_proxy_swap`).
      - Internally translated to plc module indices via `hard_indices[i_hard]`.
    """

    def __init__(self, plc, benchmark: Benchmark, current_placement_np: np.ndarray):
        self.plc = plc
        self.benchmark = benchmark
        self.n_hard = benchmark.num_hard_macros
        self.hard_indices = list(benchmark.hard_macro_indices)

        # Make sure plc + global pos cache reflect current_placement_np.
        # O5 fix (2026-05-25): force a FULL set, never trust the idempotency
        # cache here. `_apply_pos` (used by score_swap/commit_swap) keeps
        # `_global_pos_cache` in sync but NOT `_last_pos_cache`, so after a
        # prior 2-opt mutated plc, `_last_pos_cache` is stale — and
        # `_fast_set_placement` would skip macros whose stale cache value
        # coincidentally matches, leaving plc in a mixed state and computing the
        # WL baseline against the wrong positions (the seed-dependent "drift"
        # that regressed ibm01 in the first multi-seed cut). Invalidating the
        # cache guarantees every macro is re-set to current_placement_np.
        plc._last_pos_cache = None
        _fast_set_placement(plc, current_placement_np, benchmark)

        wl_cache = _build_wl_cache(plc)
        self.wl_cache = wl_cache
        self.net_weights = wl_cache["net_weights"]
        self.net_starts = wl_cache["net_starts"]
        self.net_ends = wl_cache["net_ends"]
        self.net_lengths = wl_cache["net_lengths"]
        self.ref_inv = wl_cache["ref_inv"]
        self.x_off = wl_cache["x_off"]
        self.y_off = wl_cache["y_off"]
        self.unique_ref = wl_cache["unique_ref"]
        self.n_pins = wl_cache["n_pins"]
        self.n_nets = wl_cache["n_nets"]

        # Macro index → array of net indices that contain at least one of
        # the macro's pins. Built once per benchmark from ref_idx + pin_to_net.
        self._build_macro_to_nets()

        # WL normalization: plc.get_cost() = sum(weighted HPWL) /
        # ((canvas_w + canvas_h) * net_cnt). We must apply the same divisor
        # so score_swap matches `_exact_proxy` exactly (which calls get_cost).
        cw_, ch_ = plc.get_canvas_width_height()
        self.wl_normalizer = float((cw_ + ch_) * max(plc.net_cnt, 1))

        # Initial per-net HPWL + total WL (full recompute, ~3ms one-time).
        # `per_net_hpwl` is RAW HPWL (max-min); `total_wl_raw` is the
        # weighted sum BEFORE normalization. The normalized WL used in
        # proxy is `total_wl_raw / wl_normalizer`.
        self.per_net_hpwl = self._compute_per_net_hpwl_full()
        self.total_wl_raw = float(np.sum(self.per_net_hpwl * self.net_weights))

        # Committed hard-macro positions (only hard macros can swap).
        self.committed_hard_pos = current_placement_np[:self.n_hard].astype(np.float64).copy()

        # ---- B3 phase 4 (2026-05-24): congestion incremental state. ----
        cong_cache = plc._cong_cache
        self.grid_col = int(plc.grid_col)
        self.grid_row = int(plc.grid_row)
        self.grid_w = float(plc.width / self.grid_col)
        self.grid_h = float(plc.height / self.grid_row)
        self.grid_v_routes = self.grid_w * plc.vroutes_per_micron
        self.grid_h_routes = self.grid_h * plc.hroutes_per_micron
        self.smooth_range = int(plc.smooth_range)
        n_cells = self.grid_row * self.grid_col

        # Build initial RAW (pre-normalize, pre-smooth) routing flats from
        # the current plc state. We call _vectorized_get_routing to
        # populate plc.V_routing_cong / etc (final smoothed+normalized),
        # then build our own state by calling the subset helpers with the
        # FULL net + macro sets.
        plc.get_congestion_cost()  # ensure routing populated
        self.H_flat = np.zeros(n_cells, dtype=np.float64)
        self.V_flat = np.zeros(n_cells, dtype=np.float64)
        self.H_macro_flat = np.zeros(n_cells, dtype=np.float64)
        self.V_macro_flat = np.zeros(n_cells, dtype=np.float64)
        if self.n_nets > 0:
            _apply_net_routing_subset(
                plc, np.arange(self.n_nets, dtype=np.int64), +1.0,
                self.H_flat, self.V_flat,
            )
        n_hard_cache = cong_cache["n_hard"]
        if n_hard_cache > 0:
            _apply_macro_routing_subset(
                plc, np.arange(n_hard_cache, dtype=np.int64), +1.0,
                self.V_macro_flat, self.H_macro_flat,
            )

        # Map module-index → hard-macro-slot-index (for _apply_macro_routing_subset).
        self._module_to_hard_slot: "dict[int, int]" = {
            int(m): k for k, m in enumerate(cong_cache["hard_indices"])
        }

        # ---- P3 (2026-05-26): incremental density state. ----
        # Density was the last full-recompute in score_swap (~28-36% of its time
        # per _profile_density.py): it scatters ALL soft+hard macros into the
        # occupancy grid every call. But a 2-opt swap moves only macros i, j —
        # all soft + other-hard occupancy is invariant. So maintain `grid_occupied`
        # as state; per swap subtract i,j's old footprints and add their new ones
        # (a handful of cells each), then take the top-10% over the full grid.
        dens_cache = _build_density_cache(plc, benchmark)
        self.dens_grid_col = int(plc.grid_col)
        self.dens_grid_row = int(plc.grid_row)
        self.dens_grid_w = float(plc.width / self.dens_grid_col)
        self.dens_grid_h = float(plc.height / self.dens_grid_row)
        self.dens_grid_area = self.dens_grid_w * self.dens_grid_h
        self.dens_n_cells = self.dens_grid_col * self.dens_grid_row
        self.dens_density_cnt = int(np.floor(self.dens_n_cells * 0.1))
        # Per hard-macro module → (half_w, half_h) for footprint expansion.
        # density_cache stores half sizes in soft-then-hard module order.
        self._dens_half: "dict[int, tuple[float, float]]" = {
            int(m): (float(dens_cache["half_w"][k]), float(dens_cache["half_h"][k]))
            for k, m in enumerate(dens_cache["macro_indices"])
        }
        # Initial occupancy (full scatter, one-time). Reuse the vectorized full
        # path so the baseline grid_occupied is bit-identical to get_density_cost.
        _vectorized_get_grid_cells_density(plc)
        self.grid_occupied = np.asarray(plc.grid_occupied, dtype=np.float64)
        self._dens_empty_idx = np.empty(0, dtype=np.int64)
        self._dens_empty_area = np.empty(0, dtype=np.float64)

    def _macro_occ(self, module_idx: int, cx: float, cy: float):
        """Per-cell occupancy-area contribution of one macro centered at (cx, cy).

        Returns (flat_cell_indices, areas), mirroring the per-macro overlap math
        in `_vectorized_get_grid_cells_density` exactly (floor → bounds skip →
        clip → per-cell intersection area). The footprint is small (~1-9 cells),
        so this is a tiny outer-product, not a grid-wide scatter.
        """
        hw_, hh_ = self._dens_half[int(module_idx)]
        gw, gh = self.dens_grid_w, self.dens_grid_h
        gcol, grow = self.dens_grid_col, self.dens_grid_row
        x_min = cx - hw_
        x_max = cx + hw_
        y_min = cy - hh_
        y_max = cy + hh_
        bl_col = int(np.floor(x_min / gw))
        bl_row = int(np.floor(y_min / gh))
        ur_col = int(np.floor(x_max / gw))
        ur_row = int(np.floor(y_max / gh))
        # OOB skip (matches the in_bounds mask in the full path).
        if not (ur_row >= 0 and ur_col >= 0 and bl_row <= grow - 1 and bl_col <= gcol - 1):
            return self._dens_empty_idx, self._dens_empty_area
        bl_col = min(max(bl_col, 0), gcol - 1)
        ur_col = min(max(ur_col, 0), gcol - 1)
        bl_row = min(max(bl_row, 0), grow - 1)
        ur_row = min(max(ur_row, 0), grow - 1)
        cols = np.arange(bl_col, ur_col + 1)
        rows = np.arange(bl_row, ur_row + 1)
        ox = np.minimum(gw * (cols + 1), x_max) - np.maximum(gw * cols, x_min)
        oy = np.minimum(gh * (rows + 1), y_max) - np.maximum(gh * rows, y_min)
        np.maximum(ox, 0.0, out=ox)
        np.maximum(oy, 0.0, out=oy)
        area = np.outer(oy, ox).ravel()
        flat = (rows[:, None] * gcol + cols[None, :]).ravel()
        return flat, area

    def _compute_density_cost(self) -> float:
        """Density cost from the maintained `grid_occupied` (P3).

        Mirrors PlacementCost.get_density_cost: 0.5 × mean of the top-10% (by
        count = floor(n_cells·0.1)) densest NONZERO grid cells. grid_cells =
        grid_occupied / grid_area is a monotone scaling, so the top-k set is the
        same; we scale at the end.
        """
        cnt = self.dens_density_cnt
        go = self.grid_occupied
        nz = go[go != 0.0]
        if nz.size == 0:
            return 0.0
        if self.dens_n_cells < 10:
            return 0.5 * float(nz.mean() / self.dens_grid_area)
        k = min(cnt, nz.size)
        top = np.partition(nz, nz.size - k)[nz.size - k:]
        return 0.5 * float(top.sum() / self.dens_grid_area / cnt)

    def _compute_cong_cost(self) -> float:
        """B3 phase 4: compute congestion cost from current H_flat / V_flat /
        H_macro_flat / V_macro_flat (RAW pre-normalization). Mirrors the
        final transform in `_vectorized_get_routing` + `_vectorized_get_congestion_cost`.
        """
        H = self.H_flat / self.grid_h_routes
        V = self.V_flat / self.grid_v_routes
        Hm = self.H_macro_flat / self.grid_h_routes
        Vm = self.V_macro_flat / self.grid_v_routes
        if self.smooth_range > 0:
            V = _smooth_routing_cong_vec(V, self.grid_row, self.grid_col, self.smooth_range, axis_h=False)
            H = _smooth_routing_cong_vec(H, self.grid_row, self.grid_col, self.smooth_range, axis_h=True)
        V_total = V + Vm
        H_total = H + Hm
        xx = np.concatenate([V_total, H_total])
        n = xx.size
        cnt = int(n * 0.05)
        if cnt == 0:
            return float(xx.max())
        top = np.partition(xx, n - cnt)[n - cnt:]
        return float(top.sum() / cnt)

    def _build_macro_to_nets(self):
        """Group nets by the macros (modules) that reference them.

        Output: `self.macro_to_nets[module_idx]` is a sorted int64 ndarray of
        net indices. Builds in O(n_pins) via vectorized grouping.
        """
        ref_idx = self.wl_cache["ref_idx"]
        pin_to_net = self.wl_cache["pin_to_net"]
        # Stable-sort pins by macro index, partition by macro boundary.
        order = np.argsort(ref_idx, kind="stable")
        sorted_macros = ref_idx[order]
        sorted_nets = pin_to_net[order]
        # Each contiguous run of identical macro idx corresponds to that macro's pins.
        boundaries = np.flatnonzero(np.diff(sorted_macros) != 0) + 1
        macro_segments = np.split(sorted_nets, boundaries)
        macro_keys = sorted_macros[np.concatenate([[0], boundaries])] if len(sorted_macros) else np.array([], dtype=ref_idx.dtype)
        self.macro_to_nets: "dict[int, np.ndarray]" = {}
        for k, nets_for_macro in zip(macro_keys, macro_segments):
            # Dedupe inside the macro (pin may reuse the same net? rare but safe).
            uniq = np.unique(nets_for_macro)
            self.macro_to_nets[int(k)] = uniq

    def _compute_per_net_hpwl_full(self) -> np.ndarray:
        """Full per-net HPWL recompute (one-time, mirrors `_vectorized_wirelength`)."""
        if self.n_nets == 0:
            return np.empty(0, dtype=np.float64)
        pos_cache = _ensure_pos_cache(self.plc)
        node_x = pos_cache[self.unique_ref, 0]
        node_y = pos_cache[self.unique_ref, 1]
        pin_x = node_x[self.ref_inv] + self.x_off
        pin_y = node_y[self.ref_inv] + self.y_off
        starts = self.net_starts
        max_x = np.maximum.reduceat(pin_x, starts)
        min_x = np.minimum.reduceat(pin_x, starts)
        max_y = np.maximum.reduceat(pin_y, starts)
        min_y = np.minimum.reduceat(pin_y, starts)
        return (max_x - min_x) + (max_y - min_y)

    def _compute_per_net_hpwl_subset(self, net_indices: np.ndarray) -> np.ndarray:
        """Recompute HPWL for a subset of nets only.

        Strategy: build a contiguous pin-index gather array over the subset
        nets (using cached net_lengths via repeat + cumulative offsets), then
        a single `reduceat` over that compact array. O(len(touched pins)),
        not O(n_pins).
        """
        if len(net_indices) == 0:
            return np.empty(0, dtype=np.float64)

        starts_t = self.net_starts[net_indices]
        lengths_t = self.net_lengths[net_indices]
        total = int(lengths_t.sum())
        if total == 0:
            return np.zeros(len(net_indices), dtype=np.float64)

        # Build the per-pin gather array: for net k with pin range
        # [starts_t[k], starts_t[k]+lengths_t[k]), expand to that integer range.
        # Implementation: cumulative within-net index (0..lengths_t[k]-1) for
        # each output position, plus the corresponding net's start offset.
        pin_indices = np.repeat(starts_t, lengths_t) + (
            np.arange(total) - np.repeat(np.concatenate([[0], np.cumsum(lengths_t)[:-1]]), lengths_t)
        )

        pos_cache = _ensure_pos_cache(self.plc)
        node_x = pos_cache[self.unique_ref, 0]
        node_y = pos_cache[self.unique_ref, 1]
        pin_x = node_x[self.ref_inv[pin_indices]] + self.x_off[pin_indices]
        pin_y = node_y[self.ref_inv[pin_indices]] + self.y_off[pin_indices]

        # reduceat starts in the compact array
        sub_starts = np.concatenate([[0], np.cumsum(lengths_t)[:-1]])
        max_x = np.maximum.reduceat(pin_x, sub_starts)
        min_x = np.minimum.reduceat(pin_x, sub_starts)
        max_y = np.maximum.reduceat(pin_y, sub_starts)
        min_y = np.minimum.reduceat(pin_y, sub_starts)
        return (max_x - min_x) + (max_y - min_y)

    def _touched_nets(self, i_module: int, j_module: int) -> np.ndarray:
        a = self.macro_to_nets.get(i_module)
        b = self.macro_to_nets.get(j_module)
        if a is None and b is None:
            return np.empty(0, dtype=np.int64)
        if a is None:
            return b
        if b is None:
            return a
        return np.union1d(a, b)

    def _apply_pos(self, module_idx: int, x: float, y: float) -> None:
        """set_pos + update global pos cache + dirty-flag plc."""
        self.plc.modules_w_pins[module_idx].set_pos(float(x), float(y))
        pos_cache = _ensure_pos_cache(self.plc)
        pos_cache[module_idx, 0] = float(x)
        pos_cache[module_idx, 1] = float(y)
        # plc's density / congestion caches must invalidate; WL doesn't
        # matter because we compute it ourselves.
        self.plc.FLAG_UPDATE_DENSITY = True
        self.plc.FLAG_UPDATE_CONGESTION = True

    def score_swap(self, i_hard: int, new_i_xy, j_hard: int, new_j_xy) -> float:
        """Trial: compute proxy as if (i_hard, j_hard) were swapped, then revert.

        B3 phase 4: WL via per-net incremental (phase 2). Congestion via
        per-net subset routing (phase 4): subtract OLD touched-net + i,j
        macro contributions, apply set_pos, add NEW contributions, smooth +
        compute cost, then RESTORE the raw flats from snapshot. Density
        still via plc.get_density_cost (full recompute).
        """
        i_module = self.hard_indices[i_hard]
        j_module = self.hard_indices[j_hard]
        i_slot = self._module_to_hard_slot.get(int(i_module))
        j_slot = self._module_to_hard_slot.get(int(j_module))

        # Save committed positions for revert
        old_ix, old_iy = float(self.committed_hard_pos[i_hard, 0]), float(self.committed_hard_pos[i_hard, 1])
        old_jx, old_jy = float(self.committed_hard_pos[j_hard, 0]), float(self.committed_hard_pos[j_hard, 1])

        # Snapshot RAW routing flats for revert (small arrays ~20KB each).
        H_snap = self.H_flat.copy()
        V_snap = self.V_flat.copy()
        Hm_snap = self.H_macro_flat.copy()
        Vm_snap = self.V_macro_flat.copy()

        touched = self._touched_nets(i_module, j_module)
        macro_subset = np.array(
            [s for s in (i_slot, j_slot) if s is not None], dtype=np.int64
        )

        # 1. Subtract OLD contributions (using current/committed positions).
        if len(touched) > 0:
            _apply_net_routing_subset(
                self.plc, touched, -1.0, self.H_flat, self.V_flat
            )
        if macro_subset.size > 0:
            _apply_macro_routing_subset(
                self.plc, macro_subset, -1.0,
                self.V_macro_flat, self.H_macro_flat,
            )

        # 2. Apply trial positions (pos_cache updated → next subset compute
        #    uses new positions).
        new_ix, new_iy = float(new_i_xy[0]), float(new_i_xy[1])
        new_jx, new_jy = float(new_j_xy[0]), float(new_j_xy[1])
        self._apply_pos(i_module, new_ix, new_iy)
        self._apply_pos(j_module, new_jx, new_jy)

        # 3. Add NEW contributions.
        if len(touched) > 0:
            _apply_net_routing_subset(
                self.plc, touched, +1.0, self.H_flat, self.V_flat
            )
        if macro_subset.size > 0:
            _apply_macro_routing_subset(
                self.plc, macro_subset, +1.0,
                self.V_macro_flat, self.H_macro_flat,
            )

        # 4. Incremental WL via touched nets (raw HPWL delta).
        if len(touched) > 0:
            new_per_net = self._compute_per_net_hpwl_subset(touched)
            delta = float(np.sum((new_per_net - self.per_net_hpwl[touched]) * self.net_weights[touched]))
            new_total_raw = self.total_wl_raw + delta
        else:
            new_total_raw = self.total_wl_raw
        new_wl_normalized = new_total_raw / self.wl_normalizer

        # 5. Compute congestion from our maintained flats (no plc call).
        cong = self._compute_cong_cost()

        # 6. Density incremental (P3): subtract i,j OLD footprints, add NEW,
        #    take top-10% over the grid, then revert the few touched cells.
        oi_idx, oi_area = self._macro_occ(i_module, old_ix, old_iy)
        oj_idx, oj_area = self._macro_occ(j_module, old_jx, old_jy)
        ni_idx, ni_area = self._macro_occ(i_module, new_ix, new_iy)
        nj_idx, nj_area = self._macro_occ(j_module, new_jx, new_jy)
        go = self.grid_occupied
        if oi_idx.size:
            np.subtract.at(go, oi_idx, oi_area)
        if oj_idx.size:
            np.subtract.at(go, oj_idx, oj_area)
        if ni_idx.size:
            np.add.at(go, ni_idx, ni_area)
        if nj_idx.size:
            np.add.at(go, nj_idx, nj_area)
        dens = self._compute_density_cost()

        score = float(new_wl_normalized + 0.5 * dens + 0.5 * cong)

        # 7. Revert: density cells, positions, raw routing flats.
        if ni_idx.size:
            np.subtract.at(go, ni_idx, ni_area)
        if nj_idx.size:
            np.subtract.at(go, nj_idx, nj_area)
        if oi_idx.size:
            np.add.at(go, oi_idx, oi_area)
        if oj_idx.size:
            np.add.at(go, oj_idx, oj_area)
        self._apply_pos(i_module, old_ix, old_iy)
        self._apply_pos(j_module, old_jx, old_jy)
        self.H_flat[:] = H_snap
        self.V_flat[:] = V_snap
        self.H_macro_flat[:] = Hm_snap
        self.V_macro_flat[:] = Vm_snap

        return score

    def commit_swap(self, i_hard: int, new_i_xy, j_hard: int, new_j_xy) -> None:
        """Commit a previously-trialed swap: persist positions, update
        per_net_hpwl AND routing flats (so subsequent score_swap calls see
        the new committed state).
        """
        i_module = self.hard_indices[i_hard]
        j_module = self.hard_indices[j_hard]
        i_slot = self._module_to_hard_slot.get(int(i_module))
        j_slot = self._module_to_hard_slot.get(int(j_module))

        # OLD committed positions (needed for the persistent density delta below,
        # read before committed_hard_pos is overwritten).
        old_ix = float(self.committed_hard_pos[i_hard, 0])
        old_iy = float(self.committed_hard_pos[i_hard, 1])
        old_jx = float(self.committed_hard_pos[j_hard, 0])
        old_jy = float(self.committed_hard_pos[j_hard, 1])

        touched = self._touched_nets(i_module, j_module)
        macro_subset = np.array(
            [s for s in (i_slot, j_slot) if s is not None], dtype=np.int64
        )

        # Subtract OLD routing contributions.
        if len(touched) > 0:
            _apply_net_routing_subset(
                self.plc, touched, -1.0, self.H_flat, self.V_flat
            )
        if macro_subset.size > 0:
            _apply_macro_routing_subset(
                self.plc, macro_subset, -1.0,
                self.V_macro_flat, self.H_macro_flat,
            )

        # Apply new positions.
        new_ix, new_iy = float(new_i_xy[0]), float(new_i_xy[1])
        new_jx, new_jy = float(new_j_xy[0]), float(new_j_xy[1])
        self._apply_pos(i_module, new_ix, new_iy)
        self._apply_pos(j_module, new_jx, new_jy)

        # P3: persist the density occupancy delta (subtract old footprints,
        # add new) so subsequent score_swap calls see the committed grid.
        go = self.grid_occupied
        oi_idx, oi_area = self._macro_occ(i_module, old_ix, old_iy)
        oj_idx, oj_area = self._macro_occ(j_module, old_jx, old_jy)
        ni_idx, ni_area = self._macro_occ(i_module, new_ix, new_iy)
        nj_idx, nj_area = self._macro_occ(j_module, new_jx, new_jy)
        if oi_idx.size:
            np.subtract.at(go, oi_idx, oi_area)
        if oj_idx.size:
            np.subtract.at(go, oj_idx, oj_area)
        if ni_idx.size:
            np.add.at(go, ni_idx, ni_area)
        if nj_idx.size:
            np.add.at(go, nj_idx, nj_area)

        # Add NEW routing contributions.
        if len(touched) > 0:
            _apply_net_routing_subset(
                self.plc, touched, +1.0, self.H_flat, self.V_flat
            )
        if macro_subset.size > 0:
            _apply_macro_routing_subset(
                self.plc, macro_subset, +1.0,
                self.V_macro_flat, self.H_macro_flat,
            )

        # Persist position state on the scorer.
        self.committed_hard_pos[i_hard, 0] = new_ix
        self.committed_hard_pos[i_hard, 1] = new_iy
        self.committed_hard_pos[j_hard, 0] = new_jx
        self.committed_hard_pos[j_hard, 1] = new_jy

        if len(touched) > 0:
            new_per_net = self._compute_per_net_hpwl_subset(touched)
            delta = float(np.sum((new_per_net - self.per_net_hpwl[touched]) * self.net_weights[touched]))
            self.per_net_hpwl[touched] = new_per_net
            self.total_wl_raw += delta

    def _macro_nets(self, i_module: int) -> np.ndarray:
        a = self.macro_to_nets.get(i_module)
        return a if a is not None else np.empty(0, dtype=np.int64)

    def score_move(self, i_hard: int, new_xy) -> float:
        """Trial: proxy as if hard macro i_hard RELOCATED to new_xy, then revert.

        Single-macro analogue of score_swap (relocation, not exchange) — used by
        the congestion-directed relocation pass. Only macro i's contributions
        change: WL over i's touched nets, congestion over those nets + i's macro
        routing slot, density over i's footprint cells.
        """
        i_module = self.hard_indices[i_hard]
        i_slot = self._module_to_hard_slot.get(int(i_module))
        old_ix = float(self.committed_hard_pos[i_hard, 0])
        old_iy = float(self.committed_hard_pos[i_hard, 1])
        new_ix, new_iy = float(new_xy[0]), float(new_xy[1])

        H_snap = self.H_flat.copy()
        V_snap = self.V_flat.copy()
        Hm_snap = self.H_macro_flat.copy()
        Vm_snap = self.V_macro_flat.copy()

        touched = self._macro_nets(i_module)
        macro_subset = (np.array([i_slot], dtype=np.int64)
                        if i_slot is not None else np.empty(0, dtype=np.int64))

        if len(touched) > 0:
            _apply_net_routing_subset(self.plc, touched, -1.0, self.H_flat, self.V_flat)
        if macro_subset.size > 0:
            _apply_macro_routing_subset(self.plc, macro_subset, -1.0,
                                        self.V_macro_flat, self.H_macro_flat)
        self._apply_pos(i_module, new_ix, new_iy)
        if len(touched) > 0:
            _apply_net_routing_subset(self.plc, touched, +1.0, self.H_flat, self.V_flat)
        if macro_subset.size > 0:
            _apply_macro_routing_subset(self.plc, macro_subset, +1.0,
                                        self.V_macro_flat, self.H_macro_flat)

        if len(touched) > 0:
            new_per_net = self._compute_per_net_hpwl_subset(touched)
            delta = float(np.sum((new_per_net - self.per_net_hpwl[touched]) * self.net_weights[touched]))
            new_total_raw = self.total_wl_raw + delta
        else:
            new_total_raw = self.total_wl_raw
        new_wl_normalized = new_total_raw / self.wl_normalizer
        cong = self._compute_cong_cost()

        o_idx, o_area = self._macro_occ(i_module, old_ix, old_iy)
        n_idx, n_area = self._macro_occ(i_module, new_ix, new_iy)
        go = self.grid_occupied
        if o_idx.size:
            np.subtract.at(go, o_idx, o_area)
        if n_idx.size:
            np.add.at(go, n_idx, n_area)
        dens = self._compute_density_cost()

        score = float(new_wl_normalized + 0.5 * dens + 0.5 * cong)

        # Revert: density cells, position, raw routing flats.
        if n_idx.size:
            np.subtract.at(go, n_idx, n_area)
        if o_idx.size:
            np.add.at(go, o_idx, o_area)
        self._apply_pos(i_module, old_ix, old_iy)
        self.H_flat[:] = H_snap
        self.V_flat[:] = V_snap
        self.H_macro_flat[:] = Hm_snap
        self.V_macro_flat[:] = Vm_snap
        return score

    def commit_move(self, i_hard: int, new_xy) -> None:
        """Persist a relocation: update positions, routing flats, density grid,
        and per-net HPWL so subsequent score_* calls see the new state."""
        i_module = self.hard_indices[i_hard]
        i_slot = self._module_to_hard_slot.get(int(i_module))
        old_ix = float(self.committed_hard_pos[i_hard, 0])
        old_iy = float(self.committed_hard_pos[i_hard, 1])
        new_ix, new_iy = float(new_xy[0]), float(new_xy[1])

        touched = self._macro_nets(i_module)
        macro_subset = (np.array([i_slot], dtype=np.int64)
                        if i_slot is not None else np.empty(0, dtype=np.int64))

        if len(touched) > 0:
            _apply_net_routing_subset(self.plc, touched, -1.0, self.H_flat, self.V_flat)
        if macro_subset.size > 0:
            _apply_macro_routing_subset(self.plc, macro_subset, -1.0,
                                        self.V_macro_flat, self.H_macro_flat)
        self._apply_pos(i_module, new_ix, new_iy)

        go = self.grid_occupied
        o_idx, o_area = self._macro_occ(i_module, old_ix, old_iy)
        n_idx, n_area = self._macro_occ(i_module, new_ix, new_iy)
        if o_idx.size:
            np.subtract.at(go, o_idx, o_area)
        if n_idx.size:
            np.add.at(go, n_idx, n_area)

        if len(touched) > 0:
            _apply_net_routing_subset(self.plc, touched, +1.0, self.H_flat, self.V_flat)
        if macro_subset.size > 0:
            _apply_macro_routing_subset(self.plc, macro_subset, +1.0,
                                        self.V_macro_flat, self.H_macro_flat)

        self.committed_hard_pos[i_hard, 0] = new_ix
        self.committed_hard_pos[i_hard, 1] = new_iy
        if len(touched) > 0:
            new_per_net = self._compute_per_net_hpwl_subset(touched)
            delta = float(np.sum((new_per_net - self.per_net_hpwl[touched]) * self.net_weights[touched]))
            self.per_net_hpwl[touched] = new_per_net
            self.total_wl_raw += delta


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
    top_k: Optional[int] = None,
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

    top_k (A6 attack #1, 2026-05-23): if set, restrict perturbation to the
    K movable macros with HIGHEST local congestion (out of all those above
    `cong_threshold`). Original behavior (top_k=None) moves every qualifying
    macro. Rationale per A3 diagnostic: DP loses uniformly on congestion by
    ~+0.08 on average; our cong-grad currently spreads motion across all
    congested macros, possibly blunting the gradient. TOP-K focuses motion
    on the hottest cells where it should matter most.
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

    # Per-cell congestion field for the gradient. A/B tested 2026-05-21:
    #   - H+V (objective-aligned): ibm03 −0.021 regression, ibm07 −0.004
    #   - per-axis (V→x grad, H→y grad): ibm03/ibm04 +0.011 regressions
    # max(H,V) wins both A/Bs. What seems to matter is gradient SHARPNESS,
    # not field semantics: H and V are spatially correlated (macros in dense
    # regions cause both), so decoupling them or summing them blunts the
    # per-cell "hot cell" peak that drives effective motion.
    cell_cong = np.maximum(
        np.asarray(h_list).reshape(nr, nc_grid),
        np.asarray(v_list).reshape(nr, nc_grid),
    )

    cell_w = cw / nc_grid
    cell_h = ch / nr
    scale = frac * min(cw, ch)
    cong_threshold = 0.5

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

    # A6 attack #1 (2026-05-23): TOP-K filter. Of all qualifying macros,
    # keep only the K with highest local congestion. Tested on top of the
    # full-mask baseline as a NEW candidate (Phase 8); the default
    # `top_k=None` preserves existing Phase 1/2/3/5b/5c/7 behavior.
    if top_k is not None and int(mask.sum()) > top_k:
        qual_indices = np.where(mask)[0]
        qual_cong = local_cong_all[qual_indices]
        # Negative for argpartition: pick the top_k LARGEST values.
        top_pos_in_qual = np.argpartition(-qual_cong, top_k - 1)[:top_k]
        focused_mask = np.zeros_like(mask)
        focused_mask[qual_indices[top_pos_in_qual]] = True
        mask = focused_mask

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
    # Linear no-cap (was np.minimum(local_cong, 2.0)). A/B over 6 benchmarks:
    #   ibm04 −0.0080, ibm03 −0.0013, ibm10 +0.0010, ibm01/07/09 tied.
    # Net −0.0083; ibm10 regression is within run-to-run noise (smaller than
    # the smallest other win). Cap was strangling motion on cong>2 cells —
    # exactly the hotspots gradient descent wants to clear. Canvas-bounds
    # clip on the next two lines keeps motion in-bounds.
    move_scale = scale * local_cong
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

        # --all wall-clock guard (issue #6, 2026-05-23).
        # The harness caps total --all runtime around 3600s. When the placer is
        # instantiated once and called per benchmark, these attributes track
        # cumulative wall-clock across benchmarks and tighten subsequent
        # per-benchmark budgets when the cumulative cap approaches. Single-
        # benchmark runs (the dev iteration path) leave _benchmarks_done at 0
        # and incur no extra cost — the adaptive branch is gated on
        # `_benchmarks_done >= 1`.
        self._first_place_call_time: Optional[float] = None
        self._benchmarks_done: int = 0
        # 3300s leaves 300s headroom under the 3600s harness cap for setup /
        # teardown / final-benchmark spillover. HARNESS_TOTAL_BENCHMARKS is
        # the standard --all set; a per-call override would let the harness
        # pass actual remaining-benchmark count, but isn't wired in yet.
        self.HARNESS_TOTAL_BUDGET_S: float = 3300.0
        self.HARNESS_TOTAL_BENCHMARKS: int = 17

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        np.random.seed(self.seed)
        random.seed(self.seed)

        t0 = time.monotonic()
        n = benchmark.num_hard_macros
        cw, ch = benchmark.canvas_width, benchmark.canvas_height
        sizes = benchmark.macro_sizes[:n].numpy().astype(np.float64)
        hw = sizes[:, 0] / 2
        hh = sizes[:, 1] / 2
        movable = (benchmark.get_movable_mask() & benchmark.get_hard_macro_mask())[:n].numpy()
        init_pos = benchmark.macro_positions[:n].numpy().copy().astype(np.float64)

        # --all wall-clock guard: compute effective per-benchmark budget.
        # On the first place() call, _first_place_call_time is set and the
        # default time_budget_s is used. On subsequent calls, if we're in
        # --all mode (heuristic: _benchmarks_done >= 1), the per-benchmark
        # cap shrinks proportionally to remaining_total / remaining_benchmarks.
        # Lower bound 30s prevents the budget from going negative on slow
        # benchmarks late in the run; in that case the placer returns
        # baseline-only.
        if self._first_place_call_time is None:
            self._first_place_call_time = t0
        cumulative_elapsed = t0 - self._first_place_call_time
        if self._benchmarks_done >= 1:
            remaining_total = self.HARNESS_TOTAL_BUDGET_S - cumulative_elapsed
            remaining_benchmarks = max(
                1, self.HARNESS_TOTAL_BENCHMARKS - self._benchmarks_done
            )
            adaptive_cap = remaining_total / remaining_benchmarks * 0.9
            effective_budget_s = min(
                self.time_budget_s, max(30.0, adaptive_cap)
            )
        else:
            effective_budget_s = self.time_budget_s

        _log(f"  [{benchmark.name}] hard={n}  movable={movable.sum()}  "
             f"budget={effective_budget_s:.0f}s"
             + (f"  (--all cumulative={cumulative_elapsed:.0f}s, "
                f"done={self._benchmarks_done}/{self.HARNESS_TOTAL_BENCHMARKS})"
                if self._benchmarks_done >= 1 else ""))

        # Exact scoring cutoffs.
        #
        # Pre-vectorization (2026-05-08, scalar congestion ~17-220s/call):
        # only benchmarks with n<=400 and grid<=2200 could fit a restart
        # pipeline within the 200s budget. Six benchmarks took the baseline-
        # only branch as a result.
        #
        # Post-vectorization (2026-05-21, congestion 88x faster on ibm10):
        # ibm10 (n=786) baseline scoring measured at 0.6s — was 41s. Even
        # ibm17 (n=760, grid=2244, the largest) should now fit dozens of
        # restarts. Thresholds bumped to admit ALL 17 IBM benchmarks; the
        # SLOW_SCORE_THRESHOLD_S=100s guard in the use_exact path still
        # bails to baseline if any benchmark slows back down under load.
        EXACT_MACRO_THRESHOLD = 10000  # admit all IBM benchmarks (ibm17 n=760 max)
        EXACT_GRID_CELL_LIMIT = 10000  # admit all IBM benchmarks (ibm17 grid=2244 max)
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

        # Reusable float32 view of the numpy positions to avoid creating two
        # new tensors per score. `torch.from_numpy` shares memory; the
        # subsequent .float() copies into float32 once. pl_scratch[:n, 0/1]
        # absorbs the copy without an additional intermediate allocation.
        def _score(pos: np.ndarray) -> float:
            """Update pl_scratch with hard-macro positions and return exact proxy.

            Caller must clone pl_scratch immediately if it needs to persist the
            result — the next _score call overwrites it.
            """
            pos32 = torch.from_numpy(np.ascontiguousarray(pos)).float()
            pl_scratch[:n, 0] = pos32[:, 0]
            pl_scratch[:n, 1] = pos32[:, 1]
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
        # Multi-DP (2026-05-21): launch two DPs in parallel at different
        # target_density. Diagnostic (_dp_diagnostic.py) showed DP loses on
        # 9/12 benchmarks purely on congestion (dC averages +0.09 vs winner)
        # while density is uniformly better. Hypothesis: looser target_density
        # (0.85) leaves more routing channel space; tighter (0.65) trades for
        # lower HPWL. Each at num_threads=1 to match the prior single-DP
        # num_threads=2 CPU footprint.
        dp_handles = []
        try:
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
                    # A2 retry refined 2026-05-24: 2-DP setup diversifying on
                    # soft_movable (was: diversifying on target_density 0.85/0.65).
                    # A3 diagnostic already showed hi/lo target_density
                    # mostly converged to similar congestion (lo's plc-state
                    # mutation was the real value, not its placement quality).
                    # A2 --all then showed soft_macros_movable=True is a big
                    # win on most benchmarks (ibm03 −0.10, ibm06 −0.12) but
                    # regresses on ibm01/ibm09/ibm13 where initial.plc was
                    # already dense (D > 0.87) → DP NLP compacts softs further
                    # → density spikes. Solution: launch BOTH soft_movable
                    # variants at same target_density. Best-of-both per
                    # benchmark. Tag "fixed"/"movable" for clarity.
                    # A2 refined 2026-05-25: 3-DP setup diversifying across
                    # both target_density and soft_movable axes. Phase 7
                    # RNG isolation (commit adaf693) made adding a 3rd DP
                    # safe — the original 3-DP attempt 2026-05-24 had to
                    # be reverted because the extra Phase 7 chain caused
                    # rng_cong drift, regressing ibm10 +0.036. Isolation
                    # now contains those effects.
                    #
                    # DP roles:
                    #   lo-fix: td=0.65, soft_movable=False
                    #     - ibm01 dense-init benefits from lo-td spreading.
                    #   hi-mov: td=0.85, soft_movable=True
                    #     - ibm03/06/10 wins via DP-optimized softs.
                    #   hi-fix: td=0.85, soft_movable=False
                    #     - ibm09/13 — need fixed softs at hi-td. Was
                    #       missing from the 2-DP setup, causing those
                    #       benchmarks to regress by +0.007 to +0.012.
                    for tag, td, root, soft_mv in (
                        ("lo-fix",  0.65, "/tmp/dreamplace_v1_lofix",   False),
                        ("hi-mov",  0.85, "/tmp/dreamplace_v1_himov",   True),
                        ("hi-fix",  0.85, "/tmp/dreamplace_v1_hifix",   False),
                    ):
                        try:
                            h = launch_dreamplace_async(
                                str(iccad_dir), plc=plc,
                                scratch_root=root,
                                timeout_s=120.0,
                                iterations=300,
                                num_threads=1,
                                soft_macros_movable=soft_mv,
                                target_density=td,
                            )
                            dp_handles.append((tag, td, h))
                        except Exception as exc:
                            _log(f"  DREAMPlace[{tag}] launch failed: "
                                 f"{type(exc).__name__}: {exc}")
                    if dp_handles:
                        _log(f"  DREAMPlace launched async x{len(dp_handles)} "
                             f"(target_density="
                             f"{','.join(f'{td:.2f}' for _,td,_ in dp_handles)}, "
                             f"iter=300, will check after Phase 3)")
        except Exception as exc:
            _log(f"  DREAMPlace launch failed: {type(exc).__name__}: {exc}")
            dp_handles = []

        # -- Restart 0: Baseline ----------------------------------------------
        _log(f"  Restart 0 (baseline)...")
        t1 = time.monotonic()
        baseline_pos = _will_legalize(init_pos, movable, sizes, hw, hh, cw, ch, n)
        _log(f"    Legalized in {time.monotonic()-t1:.1f}s")

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
            t_2opt = time.monotonic()
            opt_pos, swap_count = _two_opt_swap(
                baseline_pos, init_pos, sizes, hw, hh, cw, ch, movable, n,
                k_neighbors=5, max_iters=3, deadline=t_2opt + 30.0,
            )
            _log(f"  2-opt: {swap_count} swaps in {time.monotonic()-t_2opt:.1f}s")
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
            # Multi-DP fallback for the no-exact-scoring path: only use the
            # first launched handle (target_density=0.85, looser). The "lo"
            # handle would need another full score (~100s+ on these large
            # benchmarks) and we can rarely afford one. Kill the rest.
            dp_handle = dp_handles[0][2] if dp_handles else None
            for _tag, _td, _h in dp_handles[1:]:
                try:
                    _h.kill()
                except Exception:
                    pass
            if plc is not None and dp_handle is not None:
                large_dp_budget = effective_budget_s + 60.0  # mirrors BUDGET_OVERRUN_S below
                t_base_score_start = time.monotonic()
                try:
                    base_score = float(_exact_proxy(pl_scratch, benchmark, plc))
                    t_base_score = time.monotonic() - t_base_score_start
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
                        remaining = large_dp_budget - (time.monotonic() - t0)
                        max_wait = max(0.0, remaining - 2.0 * t_base_score - 5.0)
                        dp_full_large = dp_handle.wait_for_result_full(
                            max_wait_s=min(max_wait, 60.0)
                        )
                        if dp_full_large is not None:
                            dp_hard_l, dp_soft_l = dp_full_large
                            dp_hard_l_clip = dp_hard_l.copy()
                            dp_hard_l_clip[:, 0] = np.clip(dp_hard_l_clip[:, 0], hw, cw - hw)
                            dp_hard_l_clip[:, 1] = np.clip(dp_hard_l_clip[:, 1], hh, ch - hh)
                            t_dp_leg = time.monotonic()
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
                            t_dp_score_start = time.monotonic()
                            dp_score_large = float(_exact_proxy(dp_pl_large, benchmark, plc))
                            t_dp_score_large = time.monotonic() - t_dp_score_start
                            _log(f"  [large-DP] dreamplace exact proxy={dp_score_large:.4f}  "
                                 f"(leg+score {time.monotonic()-t_dp_leg:.1f}s)")
                            if dp_score_large < base_score:
                                _log(f"  [large-DP] DP wins ({dp_score_large:.4f} < "
                                     f"{base_score:.4f}); returning DP placement")
                                _log(f"  total={time.monotonic()-t0:.1f}s")
                                self._benchmarks_done += 1
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

            _log(f"  total={time.monotonic()-t0:.1f}s")
            self._benchmarks_done += 1
            return pl_scratch  # safe: no more in-place writes will happen

        # --all wall-clock guard: if cumulative time is close to the harness
        # cap, return baseline immediately without spending time on the first
        # exact score. ibm17's baseline scoring alone is 280s+; on a tight
        # cumulative budget that single score would blow the cap. Threshold is
        # effective_budget_s < 60s (one safe score) OR cumulative elapsed has
        # consumed >= 95% of HARNESS_TOTAL_BUDGET_S.
        cumulative_now = time.monotonic() - self._first_place_call_time
        if (effective_budget_s < 60.0 or
                cumulative_now > self.HARNESS_TOTAL_BUDGET_S * 0.95):
            _log(f"  [--all guard] tight budget "
                 f"(eff={effective_budget_s:.0f}s, cumulative={cumulative_now:.0f}s"
                 f" of {self.HARNESS_TOTAL_BUDGET_S:.0f}s); returning baseline")
            for _tag, _td, _h in dp_handles:
                try:
                    _h.kill()
                except Exception:
                    pass
            _log(f"  total={time.monotonic()-t0:.1f}s")
            self._benchmarks_done += 1
            return pl_scratch

        t_score0 = time.monotonic()
        best_score = float(_exact_proxy(pl_scratch, benchmark, plc))
        t_one_score = time.monotonic() - t_score0
        best_pl = pl_scratch.clone()
        _log(f"  Candidate 0: proxy={best_score:.4f}  (scored in {t_one_score:.1f}s)")

        # Safety net: if exact scoring took longer than expected (CPU load),
        # return baseline so we don't run out of budget mid-restart.
        # Tightened 2026-05-23 (issue #6): was 100s. ibm15/ibm16 first-scores
        # can be ~80s under --all CPU contention; the 100s threshold let them
        # through and then they ate the rest of the per-benchmark budget. 80s
        # is closer to the median expensive-but-still-useful score time.
        SLOW_SCORE_THRESHOLD_S = 80.0
        if t_one_score > SLOW_SCORE_THRESHOLD_S:
            _log(f"  Exact score slow ({t_one_score:.0f}s); returning baseline")
            for _tag, _td, _h in dp_handles:
                try:
                    _h.kill()
                except Exception:
                    pass
            _log(f"  Best proxy={best_score:.4f}  total={time.monotonic()-t0:.1f}s")
            self._benchmarks_done += 1
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
            elapsed = time.monotonic() - t0
            cap = effective_budget_s + (BUDGET_OVERRUN_S if allow_overrun else 0.0)
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

            t1 = time.monotonic()
            leg_deadline = t1 + 60.0  # cap spiral search; timed-out macros keep pos value
            leg = _will_legalize(perturbed_init, movable, sizes, hw, hh, cw, ch, n,
                                 deadline=leg_deadline, order=order)
            t_leg = time.monotonic() - t1
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

            t_score_start = time.monotonic()
            score = _score(leg)
            t_score_observed = time.monotonic() - t_score_start
            if t_score_observed > t_one_score:
                t_one_score = t_score_observed
            _log(f"  Candidate {k}: proxy={score:.4f}")
            if score < best_score:
                best_score = score
                best_pl = pl_scratch.clone()  # snapshot only on improvement

            # Safety: if scoring overran the (possibly relaxed) cap, stop immediately
            # rather than launching another restart that would push time further over.
            if time.monotonic() - t0 > cap:
                _log(f"  Over budget after scoring ({time.monotonic()-t0:.0f}s, cap={cap:.0f}s); stopping")
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
                remaining = (effective_budget_s + BUDGET_OVERRUN_S) - (time.monotonic() - t0)
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
                remaining = (effective_budget_s + BUDGET_OVERRUN_S) - (time.monotonic() - t0)
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
            remaining = (effective_budget_s + BUDGET_OVERRUN_S) - (time.monotonic() - t0)
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

        # -- Async DREAMPlace check (Phase 5: additive candidates) ------------
        # Multi-DP: iterate over all launched handles. Each completed DP
        # becomes a candidate; the best across all DPs feeds Phase 5b/5c
        # and is also retained in `dp_placements` for Phase 7 (DP-rescue
        # cong-grad as additive tail after the noise loop).
        dp_placements: list[tuple[str, float, torch.Tensor]] = []
        for tag, td, h in dp_handles:
            remaining_dp = (effective_budget_s + BUDGET_OVERRUN_S) - (time.monotonic() - t0)
            # 3*t_one_score reserve covers Phase 5b + at least one noise score.
            max_wait = max(0.0, min(remaining_dp - 3.0 * t_one_score, 30.0))
            dp_full = h.wait_for_result_full(max_wait_s=max_wait)
            if dp_full is None:
                _log(f"  DREAMPlace[{tag} td={td:.2f}] not ready "
                     f"(elapsed={h.time_elapsed():.1f}s); killing subprocess")
                h.kill()
                continue
            dp_hard, dp_soft = dp_full
            _log(f"  DREAMPlace[{tag} td={td:.2f}] ready in {h.time_elapsed():.1f}s "
                 f"(hard={dp_hard.shape[0]}, soft={dp_soft.shape[0]}); "
                 f"testing as candidate")
            # Legalize hard macros (DREAMPlace's NLP may leave overlaps).
            # Clip out-of-canvas first: DREAMPlace's macro_place_flag stage
            # can produce positions slightly past canvas.
            t_dp = time.monotonic()
            dp_leg_deadline = t_dp + 60.0
            dp_hard_clip = dp_hard.copy()
            dp_hard_clip[:, 0] = np.clip(dp_hard_clip[:, 0], hw, cw - hw)
            dp_hard_clip[:, 1] = np.clip(dp_hard_clip[:, 1], hh, ch - hh)
            dp_hard_leg = _will_legalize(
                dp_hard_clip, movable, sizes, hw, hh, cw, ch, n,
                deadline=dp_leg_deadline,
            )
            dp_pl = benchmark.macro_positions.clone()
            dp_pl[:n, 0] = torch.tensor(dp_hard_leg[:, 0], dtype=torch.float32)
            dp_pl[:n, 1] = torch.tensor(dp_hard_leg[:, 1], dtype=torch.float32)
            n_soft_dp = int(min(dp_soft.shape[0], benchmark.num_soft_macros))
            if n_soft_dp > 0:
                dp_pl[n:n + n_soft_dp, 0] = torch.tensor(
                    dp_soft[:n_soft_dp, 0], dtype=torch.float32
                )
                dp_pl[n:n + n_soft_dp, 1] = torch.tensor(
                    dp_soft[:n_soft_dp, 1], dtype=torch.float32
                )
            t_dp_score_start = time.monotonic()
            dp_score = float(_exact_proxy(dp_pl, benchmark, plc))
            t_dp_score = time.monotonic() - t_dp_score_start
            if t_dp_score > t_one_score:
                t_one_score = t_dp_score
            directed_ran += 1
            _log(f"  Candidate {directed_ran} (dreamplace[{tag}] hard+soft): "
                 f"proxy={dp_score:.4f}  (leg+score {time.monotonic()-t_dp:.1f}s)")
            # The 2026-05-22 "analytic soft re-snap" experiment (centroid-
            # follow blend on DP candidate softs) was rejected: regressed
            # ibm04 +0.003 and ibm10 +0.002 at every blend factor. Resolved
            # 2026-05-24 by A2: launching DP with soft_movable=True lets
            # DREAMPlace's NLP optimize softs directly (better than analytic
            # post-hoc re-snap). The helpers _build_soft_resnap_cache and
            # _resnap_soft_macros were never copied forward to v2.
            if dp_score < best_score:
                best_score = dp_score
                best_pl = dp_pl.clone()
            dp_placements.append((tag, dp_score, dp_pl))

        # Phase 5b: cong-grad from best_pl using current plc state. plc state
        # reflects whatever was scored last (last DP if any DP scored, else
        # baseline). Perturbing best_pl with this gradient explores basins
        # the original-baseline plc state alone couldn't reach.
        if dp_placements:
            remaining_5b = (effective_budget_s + BUDGET_OVERRUN_S) - (time.monotonic() - t0)
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

        # Phase 6 (cong-grad from DP placement, single per-iter inside Phase 5)
        # tested 2026-05-20, REJECTED for displacing noise restarts that won
        # ibm08 at 6% noise. Phase 7 below revisits this idea but only AFTER
        # the noise loop completes — purely additive on leftover budget.

        # Phase 5c: wide-from-best with current plc state. Fills the slot left
        # by Phase 2 (wide from BASELINE only) and Phase 3/5b (frac=0.04 from
        # BEST only). Uses the latest plc state (post-Phase-5b if DP fired,
        # else post-Phase-3) which encodes the most-recent congestion pattern.
        # Purely additive: fires only if cong-grad helped earlier and budget
        # allows; placed AFTER Phase 5b so no current winning rng_cong path is
        # affected. Noise loop uses np.random directly (not rng_cong), so the
        # extra rng_cong draw here doesn't perturb noise restarts.
        if cong_improved:
            remaining_5c = (effective_budget_s + BUDGET_OVERRUN_S) - (time.monotonic() - t0)
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

        # -- Phase 7: DP-rescue cong-grad chain (additive, after noise) -------
        # Diagnostic (_dp_diagnostic.py 2026-05-21) showed DP loses on 9/12
        # benchmarks purely on congestion (dC +0.02 to +0.16). 2026-05-21
        # single-iter tests on ibm01/04/07/12/02 confirmed 1 iter is not
        # enough to close gaps that large — the rescue candidates scored
        # WORSE than current best every time, because legalization shuffles
        # macros enough that one gradient step gets reset.
        #
        # Multi-iter (this version): chain up to MAX_P7_ITERS cong-grad
        # iterations per DP placement, each starting from the previous
        # iter's legalized output. Greedy descent: stop the chain when an
        # iter fails to improve over the previous iter (gradient direction
        # is no longer productive). Each iter's plc state reflects the
        # prior iter's scoring, so the gradient is recomputed fresh.
        #
        # Phase 6 (2026-05-20) ran similar multi-iter BEFORE the noise loop
        # and was rejected for displacing noise winners. Phase 7 runs AFTER
        # noise — purely additive, only consumes leftover budget.
        # Phase 7 retro-eval 2026-05-25 (90-iter sample, monotonic-clock
        # --all log): 7 wins / 90 iters = 7.8% hit rate. Big wins on
        # ibm02 (−0.060 at hi-mov iter 3) and ibm10 (−0.07 across lo-fix
        # chain). 13 of 17 benchmarks contribute 0 wins. Iter-1-margin
        # gate (threshold 0.06) abandons chains where iter 1 is far
        # worse than pre-P7 best; preserves all 7 wins (largest winning
        # iter-1 margin was 0.0555) while gating ~14 zero-win chains.
        #
        # RNG isolation 2026-05-25: snapshot rng_cong before Phase 7 and
        # restore after. Without this, the variable-length Phase 7 chains
        # (greedy break, iter-1-margin gate, MAX_P7_ITERS cap) consume
        # rng_cong by different amounts across benchmarks, causing the
        # downstream Phase 8/9 perturbations to diverge — initial gate
        # test showed ibm10 regressed +0.0193 purely from this RNG drift.
        # The isolation makes Phase 7 a closed compartment w.r.t. rng_cong,
        # so changes to Phase 7's internal logic (gating, chain length,
        # adding/removing DPs) no longer affect downstream phases.
        rng_cong_pre_p7 = rng_cong.get_state()
        P7_ITER1_MARGIN_GATE = 0.06  # tested 2026-05-25, see ISSUES.md A5
        MAX_P7_ITERS = 3
        for tag, _dp_score_unused, dp_pl_saved in dp_placements:
            current_pos = np.stack(
                [dp_pl_saved[:n, 0].numpy(), dp_pl_saved[:n, 1].numpy()], axis=1
            ).astype(np.float64)
            prev_iter_score = float("inf")
            pre_chain_best = best_score
            for it in range(1, MAX_P7_ITERS + 1):
                remaining_p7 = (
                    effective_budget_s + BUDGET_OVERRUN_S
                ) - (time.monotonic() - t0)
                if remaining_p7 < t_one_score * 1.3:
                    break
                rescue_perturbed = _routing_congestion_perturb(
                    current_pos, plc, benchmark, n, cw, ch, hw, hh, movable,
                    frac=0.04, rng=rng_cong,
                )
                t1 = time.monotonic()
                leg = _will_legalize(
                    rescue_perturbed, movable, sizes, hw, hh, cw, ch, n,
                    deadline=t1 + 60.0,
                )
                t_leg = time.monotonic() - t1
                directed_ran += 1
                _log(f"  Restart {directed_ran} (cong-grad from-dp[{tag}] "
                     f"iter={it} f=0.04) legalized in {t_leg:.1f}s")
                t_score_start = time.monotonic()
                score = _score(leg)
                t_score_observed = time.monotonic() - t_score_start
                if t_score_observed > t_one_score:
                    t_one_score = t_score_observed
                _log(f"  Candidate {directed_ran}: proxy={score:.4f}")
                if score < best_score:
                    best_score = score
                    best_pl = pl_scratch.clone()
                # Iter-1 margin gate: abandon chain if iter 1 score is
                # far above pre-chain best — empirically those chains
                # don't recover (per Phase 7 retro-eval 2026-05-25).
                if it == 1 and (score - pre_chain_best) > P7_ITER1_MARGIN_GATE:
                    break
                # Greedy descent: stop chain if this iter didn't strictly
                # improve over previous iter's score.
                if score >= prev_iter_score - 1e-4:
                    break
                prev_iter_score = score
                current_pos = leg
                # Hard cap: don't exceed cap after this iter's scoring.
                if time.monotonic() - t0 > effective_budget_s + BUDGET_OVERRUN_S:
                    break

        # RNG isolation (2026-05-25): restore rng_cong to pre-Phase-7 state
        # so Phase 8/9 perturbations are deterministic regardless of how
        # many Phase 7 chain iters fired (iter-1-margin gate, greedy break,
        # MAX_P7_ITERS cap all cause irregular consumption).
        rng_cong.set_state(rng_cong_pre_p7)

        # -- Phase 8: TOP-K cong-grad from best_pl (A6 attack #1, 2026-05-23) -
        # The A3 diagnostic showed DP loses on congestion by avg +0.08 vs our
        # best. Phase 1/2/3/5/7 use the full-mask perturb (every macro in a
        # congested cell moves), which may blunt the gradient on dense
        # benchmarks. Phase 8 tries TOP-K (move only the K hottest macros)
        # from best_pl with a few K values; preserves all prior wins because
        # it runs LAST and only consumes leftover budget.
        #
        # 2026-05-24 (improvement #3): per-K multi-iter chains (like Phase 7
        # but starting from current best_pl). Greedy break-on-no-improvement.
        # Note: single-bench testing showed mixed results — ibm04 −0.0005
        # but ibm10 +0.0020 (regression). Including in combined --all to see
        # if cross-benchmark wins offset.
        MAX_P8_ITERS = 3
        if cong_improved:
            for top_k_val in (5, 10, 20):
                prev_chain_score = best_score
                for chain_iter in range(MAX_P8_ITERS):
                    remaining_p8 = (
                        effective_budget_s + BUDGET_OVERRUN_S
                    ) - (time.monotonic() - t0)
                    if remaining_p8 < t_one_score * 1.3:
                        break
                    best_pos_now = np.stack(
                        [best_pl[:n, 0].numpy(), best_pl[:n, 1].numpy()], axis=1
                    )
                    p8_perturbed = _routing_congestion_perturb(
                        best_pos_now, plc, benchmark, n, cw, ch, hw, hh, movable,
                        frac=0.04, rng=rng_cong, top_k=top_k_val,
                    )
                    if not _try_restart(
                        f"cong-grad-best TOP-{top_k_val} iter={chain_iter+1} f=0.04",
                        p8_perturbed,
                        k=1 + directed_ran, allow_overrun=True,
                    ):
                        break
                    directed_ran += 1
                    if best_score >= prev_chain_score - 1e-4:
                        break
                    prev_chain_score = best_score

        # Phase 9a (fine-noise from best_pl, A6 axis #3) was tested and
        # REVERTED 2026-05-23. Added 4 Gaussian-perturb candidates from
        # best_pl at frac=0.005-0.02. `--all` net result: 0 avg change
        # (ibm14 −0.0005, ibm17 +0.0008, rest within ±0.0001). Small noise
        # + greedy legalize converges back to the same basin most of the
        # time; the pipeline's existing perturbations already cover the
        # productive perturbation magnitudes.

        # -- Phase 9: Random-tiebreak legalize order (A6 axis #4, 2026-05-23) -
        # Default `_will_legalize` order is `sorted(range(n), key=-area)` —
        # largest-area first with index-tied secondary key. For benchmarks
        # with many similar-sized macros (ibm08/09/11/13), the deterministic
        # tiebreaks may lock the placer into one specific legal arrangement.
        # This phase tries N_TRIALS legalize orderings that keep the primary
        # key (-area) but RANDOMIZE the secondary key.
        #
        # Distinct from the rejected "multi-order baseline" (smallest-area,
        # tallest, widest) which changed the primary key — that regressed
        # benchmarks where small-macro-first produced large-macro-trapped
        # placements. Here the primary key is preserved.
        N_ORDER_TRIALS = 3
        area = sizes[:n, 0] * sizes[:n, 1]
        for trial in range(N_ORDER_TRIALS):
            remaining_p9 = (
                effective_budget_s + BUDGET_OVERRUN_S
            ) - (time.monotonic() - t0)
            if remaining_p9 < t_one_score * 1.3:
                break
            # np.lexsort: last key is primary. With (random_key, -area) the
            # primary sort is by -area (largest first), tied entries broken
            # by the uniform random key — different per trial.
            random_key = rng_cong.random(n)
            shuffled_order = np.lexsort((random_key, -area)).tolist()
            if not _try_restart(f"random-order-legalize trial={trial}",
                                 init_pos, k=1 + directed_ran,
                                 allow_overrun=True,
                                 order=shuffled_order):
                break
            directed_ran += 1

        # -- 2-opt swap on cong-grad winner (additive, after Phase 7) ---------
        # Proxy-driven (issue #1, 2026-05-23). Previously this used
        # `_two_opt_swap` (displacement-from-init criterion), which was
        # empirically anti-correlated with proxy on ibm01/04/10 — every
        # documented benchmark had the post-hoc guard reject ALL applied
        # swaps. The 15s budget was wasted. With per-score time at ~5-50ms
        # post-vectorization, scoring each candidate swap directly is
        # affordable. Cheap bounds + conflict checks remain as a free
        # filter so most candidates skip the score call.
        # Phase 7b (DP-basin congestion relief) was prototyped 2026-05-26 and
        # REVERTED — see ISSUES.md. The DP_PROBE ceiling test suggested the best
        # raw DREAMPlace basin could 2-opt below best after a fuller cong-grad
        # descent (ibm10 1.3279 vs 1.3337), but the production descent proved too
        # budget-hungry (~30s/benchmark) AND high-variance — and not even
        # reproducible at fixed seed (plc-state-dependent on where in the pipeline
        # it runs: seed 777 gave 1.3639 post-pipeline but 1.3730 mid-pipeline). It
        # captured zero net gain in-pipeline. The durable finding (DP loses purely
        # on congestion; post-hoc repair can't fix it reliably) points instead at
        # congestion-aware DREAMPlace (congestion in the global objective). The
        # DP_DIAG/DP_PROBE diagnostics are retained (env-gated) to reproduce it.
        remaining_2opt = (effective_budget_s + BUDGET_OVERRUN_S) - (time.monotonic() - t0)
        if remaining_2opt >= t_one_score + 15.0:
            # O2 candidate #2 (2026-05-25): run 2-opt from MULTIPLE basins, not
            # just the single refined best_pl. Raw DP proxy is not predictive of
            # the final 2-opt result (the O2-margin experiment showed keeping the
            # "winning" basin as the only seed can converge worse), so try best_pl
            # plus each DP candidate basin and keep the global minimum. Losing DP
            # basins cost one 2-opt budget each but may 2-opt below the winner.
            twoopt_seeds: list[tuple[str, torch.Tensor, float]] = [
                ("best", best_pl.clone(), best_score)
            ]
            for _tag, _dp_sc, _dp_pl in dp_placements:
                twoopt_seeds.append((f"dp[{_tag}]", _dp_pl.clone(), _dp_sc))
            # S4 baseline_pos seed tested 2026-05-25, REJECTED: 2-opt from the
            # raw legalized baseline never beat best_pl on any of ibm01/04/09/
            # 10/13 (landed 0.02-0.10 above), since baseline is best_pl's
            # unrefined ancestor, not a distinct basin. Pure wall-clock cost.

            # Prune hopeless DP basins. A seed can only beat the incumbent's
            # 2-opt result if its own 2-opt result is lower; for a DP seed whose
            # raw proxy is > DP_SEED_2OPT_WINDOW above best_score, the 2-opt gain
            # needed to catch up exceeds anything observed, so it isn't worth a
            # 15s pass. Both observed basin wins sit well inside this window
            # (ibm04 dp[hi-mov] +0.011, ibm09 dp[hi-fix] +0.002). The "best" seed
            # is never pruned (it reproduces the committed single-seed 2-opt,
            # keeping the change strictly additive).
            DP_SEED_2OPT_WINDOW = 0.02

            # Selection is by TRUE _exact_proxy, never the IncrementalScorer's
            # final_score: the incremental WL drifts seed-dependently (ibm01
            # dp[lo-fix] reported internal 1.1309 but true proxy 1.1506), so
            # cross-seed comparison on the internal score picks phantom winners.
            # The incremental scorer still guides which swaps to accept (speed);
            # we just re-score each finalist exactly before comparing.
            twoopt_best_pl = best_pl
            twoopt_best_score = float(_exact_proxy(best_pl, benchmark, plc))
            _dp_diag_2opt = []  # (seed_tag, true_final, cand) when DP_DIAG set
            for seed_tag, seed_pl, seed_score in twoopt_seeds:
                rem = (effective_budget_s + BUDGET_OVERRUN_S) - (time.monotonic() - t0)
                if rem < 2.0 * t_one_score + 15.0:
                    _log(f"  2-opt seed {seed_tag}: skipped (budget {rem:.0f}s)")
                    break
                if seed_tag != "best" and seed_score > best_score + DP_SEED_2OPT_WINDOW:
                    _log(f"  2-opt seed {seed_tag}: pruned "
                         f"(raw {seed_score:.4f} > best {best_score:.4f} + "
                         f"{DP_SEED_2OPT_WINDOW})")
                    continue
                t_2opt = time.monotonic()
                # S1 (2026-05-26): basin-hopping 2-opt. The 2-opt search only
                # PERMUTES existing macro slots — it can never reach a position
                # no macro currently occupies. After a pass converges to a swap-
                # only local minimum, inject a congestion-gradient KICK (the same
                # _routing_congestion_perturb the cong-grad phases use) to move
                # the hottest macros to NEW continuous positions against the live
                # congestion field, legalize, then run 2-opt again to clean up.
                # Accept-on-true-proxy, keeping the running best across passes.
                #
                # Budget-safe by construction: each pass gets the FULL remaining
                # 15s deadline, and 2-opt returns early only when it converges.
                # On deadline-bound large benchmarks (ibm10/12/16) the first pass
                # eats the whole 15s → no time to kick → behavior is byte-identical
                # to the prior single-pass code. The kicks only fill the otherwise-
                # idle remainder on benchmarks where 2-opt exhausts its candidate
                # pool early.
                # Slice the 15s into passes so 2-opt yields its low-yield tail
                # to a kick + fresh search. Empirically the 2-opt never reaches a
                # local minimum within a full 15s on these benchmarks (always
                # deadline-bound), so a "kick only on early convergence" trigger
                # never fires — slicing is what makes the interleave happen.
                S1_PASS_BUDGET = 5.0      # seconds per 2-opt pass before a kick
                # S1 DORMANT (max_kicks=0). DISPROVEN 2026-05-26: enabling sliced
                # basin-hopping (5s passes + cong-grad kick, max_kicks=2) regressed
                # --all on 6/7 benchmarks before the run was stopped (ibm01 +0.0037,
                # ibm04 +0.0091, ibm08 +0.0045; cumulative +0.025/7). Slicing starves
                # the productive deadline-bound 2-opt search, and the kicks perturb
                # away from the optimum without recovering. The earlier "more accepts"
                # signal (671→1072 on ibm04) was misleading — the extra accepts were
                # repairing kick damage, not net-improving. Code kept for reference.
                S1_MAX_KICKS = 0          # → up to (this+1) passes of ~5s each
                S1_KICK_FRAC = 0.03       # kick magnitude (refinement-scale)
                S1_MIN_REM = 3.0          # need >=this much budget to bother kicking
                global_2opt_deadline = t_2opt + 15.0
                s1_rng = np.random.RandomState(20260526)

                work_pl = seed_pl.clone()
                work_hard = np.stack(
                    [seed_pl[:n, 0].numpy(), seed_pl[:n, 1].numpy()], axis=1
                ).astype(np.float64)
                work_score = seed_score
                seed_best_pl = seed_pl.clone()
                seed_best_score = float("inf")
                accept_count = 0
                score_calls = 0
                final_score = work_score
                n_kicks = 0
                while True:
                    # B3 phase 2/4 IncrementalScorer: incremental WL + congestion.
                    # Re-init per pass from the current working placement (kick
                    # moved positions non-swap-wise, so the prior scorer state is
                    # stale). Init cost is ~3-10ms, negligible vs the 15s budget.
                    try:
                        incremental_scorer = IncrementalScorer(
                            plc, benchmark, work_pl.cpu().numpy().astype(np.float64)
                        )
                    except Exception as exc:
                        _log(f"  IncrementalScorer init failed: {type(exc).__name__}: "
                             f"{exc}; falling back to full scoring")
                        incremental_scorer = None

                    opt_scratch = work_pl.clone()

                    def _2opt_score(pos_arr: np.ndarray, _scr=opt_scratch) -> float:
                        pos32 = torch.from_numpy(np.ascontiguousarray(pos_arr)).float()
                        _scr[:n, 0] = pos32[:, 0]
                        _scr[:n, 1] = pos32[:, 1]
                        return float(_exact_proxy(_scr, benchmark, plc))

                    # S9 (2026-05-26): per-macro local congestion snapshot for
                    # congestion-aware 2-opt (hot-first ordering + cold-region
                    # teleport augmentation). The IncrementalScorer init above
                    # called plc.get_congestion_cost() on work_pl, so plc's
                    # routing map reflects the current placement. cell field is
                    # max(H,V), matching _routing_congestion_perturb.
                    macro_cong = None
                    try:
                        nr_g, nc_g = benchmark.grid_rows, benchmark.grid_cols
                        h_arr = np.asarray(
                            plc.get_horizontal_routing_congestion(), dtype=np.float64
                        )
                        v_arr = np.asarray(
                            plc.get_vertical_routing_congestion(), dtype=np.float64
                        )
                        if h_arr.size == nr_g * nc_g and v_arr.size == nr_g * nc_g:
                            cell_cong = np.maximum(
                                h_arr.reshape(nr_g, nc_g), v_arr.reshape(nr_g, nc_g)
                            )
                            cwc, chc = cw / nc_g, ch / nr_g
                            ci = np.clip(
                                (work_hard[:, 0] / cwc).astype(np.int64), 0, nc_g - 1
                            )
                            ri = np.clip(
                                (work_hard[:, 1] / chc).astype(np.int64), 0, nr_g - 1
                            )
                            macro_cong = cell_cong[ri, ci]
                    except Exception:
                        macro_cong = None

                    # k_neighbors=20 / max_iters=6 (S2, 2026-05-25): per-score
                    # ~3ms post-B3-phase-4, so a wide candidate pool fits. Each
                    # pass is bounded by a time slice (S1_PASS_BUDGET) AND the
                    # global 15s deadline, whichever is sooner.
                    pass_deadline = global_2opt_deadline if S1_MAX_KICKS == 0 else min(
                        global_2opt_deadline, time.monotonic() + S1_PASS_BUDGET
                    )
                    opt_pos, ac, fs, sc = _two_opt_proxy_swap(
                        work_hard, sizes, hw, hh, cw, ch, movable, n,
                        score_fn=_2opt_score, initial_score=work_score,
                        k_neighbors=20, max_iters=6, deadline=pass_deadline,
                        incremental_scorer=incremental_scorer,
                        macro_cong=macro_cong,
                    )
                    accept_count += ac
                    score_calls += sc
                    final_score = fs

                    cand = work_pl.clone()
                    cand[:n, 0] = torch.tensor(opt_pos[:, 0], dtype=torch.float32)
                    cand[:n, 1] = torch.tensor(opt_pos[:, 1], dtype=torch.float32)
                    # _exact_proxy also repopulates plc's routing-congestion map,
                    # which the kick below reads to build its gradient.
                    cand_true = float(_exact_proxy(cand, benchmark, plc))
                    if cand_true < seed_best_score:
                        seed_best_score = cand_true
                        seed_best_pl = cand

                    rem = global_2opt_deadline - time.monotonic()
                    if n_kicks >= S1_MAX_KICKS or rem < S1_MIN_REM:
                        break

                    # Congestion-gradient kick from the just-scored 2-opt result
                    # (plc reflects `cand`), then legalize the perturbed hard
                    # macros. Feed the kicked layout into the next 2-opt pass even
                    # if it scores worse — escaping the swap-only basin is the
                    # whole point; seed_best_pl preserves the best seen so far.
                    kicked = _routing_congestion_perturb(
                        opt_pos, plc, benchmark, n, cw, ch, hw, hh, movable,
                        frac=S1_KICK_FRAC, rng=s1_rng,
                    )
                    kicked_leg = _will_legalize(
                        kicked, movable, sizes, hw, hh, cw, ch, n
                    )
                    work_pl = cand.clone()
                    work_pl[:n, 0] = torch.tensor(kicked_leg[:, 0], dtype=torch.float32)
                    work_pl[:n, 1] = torch.tensor(kicked_leg[:, 1], dtype=torch.float32)
                    work_hard = kicked_leg.astype(np.float64)
                    work_score = float(_exact_proxy(work_pl, benchmark, plc))
                    n_kicks += 1

                cand = seed_best_pl
                true_final = seed_best_score
                scorer_tag = "incr" if incremental_scorer is not None else "full"
                _log(f"  2-opt seed {seed_tag} (proxy/{scorer_tag}): {accept_count} "
                     f"accepts / {score_calls} scores, {n_kicks} kicks, "
                     f"true={true_final:.4f} (was {seed_score:.4f}) "
                     f"in {time.monotonic()-t_2opt:.1f}s")
                if true_final < twoopt_best_score:
                    twoopt_best_score = true_final
                    twoopt_best_pl = cand
                if os.environ.get("DP_DIAG"):
                    _dp_diag_2opt.append((seed_tag, true_final, cand.clone()))

            if twoopt_best_score < best_score:
                best_score = twoopt_best_score
                best_pl = twoopt_best_pl

        # -- Congestion-directed relocation pass (R1, 2026-05-27) -------------
        # The 2-opt above only EXCHANGES macro positions; it can't relocate a
        # routing-heavy macro into an empty low-congestion gap (a swap would dump
        # another macro into the vacated hot spot). This pass does exactly that:
        # move the hottest macros into the lowest-congestion legal cells, accept
        # only on a strict true-proxy drop (via the incremental scorer's verified
        # score_move). The RELOC_PROBE measured this beating the 2-opt best on
        # ibm04 −0.032, ibm10 −0.011, ibm12 −0.006 — gain in the congestion term,
        # in ~0.3s. Cheap and additive by construction.
        rem_reloc = (effective_budget_s + BUDGET_OVERRUN_S) - (time.monotonic() - t0)
        if rem_reloc >= 2.0 * t_one_score + 2.0:
            t_reloc = time.monotonic()
            try:
                base_reloc = float(_exact_proxy(best_pl, benchmark, plc))
                reloc_scorer = IncrementalScorer(
                    plc, benchmark, best_pl.cpu().numpy().astype(np.float64)
                )
                reloc_pos = np.stack(
                    [best_pl[:n, 0].numpy(), best_pl[:n, 1].numpy()], axis=1
                ).astype(np.float64)
                reloc_pos, reloc_acc, _reloc_incr = _relocation_moves(
                    reloc_pos, sizes, hw, hh, cw, ch, movable, n, plc, benchmark,
                    reloc_scorer, base_reloc,
                    deadline=t_reloc + min(rem_reloc - t_one_score, 15.0),
                )
                if reloc_acc > 0:
                    reloc_cand = best_pl.clone()
                    reloc_cand[:n, 0] = torch.tensor(reloc_pos[:, 0], dtype=torch.float32)
                    reloc_cand[:n, 1] = torch.tensor(reloc_pos[:, 1], dtype=torch.float32)
                    reloc_true = float(_exact_proxy(reloc_cand, benchmark, plc))
                    _log(f"  Relocation pass: {reloc_acc} moves, "
                         f"{base_reloc:.4f} → {reloc_true:.4f} "
                         f"in {time.monotonic()-t_reloc:.1f}s")
                    if reloc_true < best_score:
                        best_score = reloc_true
                        best_pl = reloc_cand
            except Exception as exc:
                _log(f"  Relocation pass failed: {type(exc).__name__}: {exc}")

        # DP_DIAG (2026-05-26): decompose where the DP basin loses to "best".
        # Logs the WEIGHTED proxy split (wl, 0.5*den, 0.5*cong) for each raw DP
        # candidate, each cong-grad+2-opt-from-seed result, and the final best.
        # Re-scores placements (mutates plc), so done last, right before return.
        if os.environ.get("DP_DIAG"):
            _log("  [DP_DIAG] ---- raw DP candidates (pre cong-grad/2-opt) ----")
            for _t, _sc, _pl in dp_placements:
                p, w, d, c = _proxy_decomp(_pl, benchmark, plc)
                _log(f"  [DP_DIAG] raw dp[{_t}]: proxy={p:.4f}  wl={w:.4f} "
                     f"den={d:.4f} cong={c:.4f}")
            if "_dp_diag_2opt" in locals():
                _log("  [DP_DIAG] ---- after cong-grad+2-opt from each seed ----")
                for _t, _tf, _pl in _dp_diag_2opt:
                    p, w, d, c = _proxy_decomp(_pl, benchmark, plc)
                    _log(f"  [DP_DIAG] 2opt[{_t}]: proxy={p:.4f}  wl={w:.4f} "
                         f"den={d:.4f} cong={c:.4f}")
            p, w, d, c = _proxy_decomp(best_pl, benchmark, plc)
            _log(f"  [DP_DIAG] FINAL best: proxy={p:.4f}  wl={w:.4f} "
                 f"den={d:.4f} cong={c:.4f}")

        if os.environ.get("DP_PROBE"):
            _dp_recoverability_probe(
                dp_placements, best_score, n, cw, ch, hw, hh, sizes,
                movable, plc, benchmark,
            )

        # RELOC_PROBE (2026-05-27): congestion-directed relocation moves on the
        # final best_pl. Builds a fresh scorer, runs _relocation_moves, reports
        # the true proxy delta + decomposition. Diagnostic only (no production
        # change) — measure whether relocations beat the 2-opt result.
        if os.environ.get("RELOC_PROBE"):
            try:
                t_rp = time.monotonic()
                base = float(_exact_proxy(best_pl, benchmark, plc))
                bw = float(plc.get_cost()); bd = 0.5 * float(plc.get_density_cost())
                bc = 0.5 * float(plc.get_congestion_cost())
                rscorer = IncrementalScorer(plc, benchmark, best_pl.cpu().numpy().astype(np.float64))
                rpos = np.stack([best_pl[:n, 0].numpy(), best_pl[:n, 1].numpy()], axis=1).astype(np.float64)
                rpos, racc, rsc = _relocation_moves(
                    rpos, sizes, hw, hh, cw, ch, movable, n, plc, benchmark,
                    rscorer, base, deadline=t_rp + 20.0,
                )
                rcand = best_pl.clone()
                rcand[:n, 0] = torch.tensor(rpos[:, 0], dtype=torch.float32)
                rcand[:n, 1] = torch.tensor(rpos[:, 1], dtype=torch.float32)
                rp, rw, rd, rc = _proxy_decomp(rcand, benchmark, plc)
                verdict = "BEATS best" if rp < base - 1e-4 else "no gain"
                _log(f"  [RELOC_PROBE] base={base:.4f} (wl={bw:.4f} den={bd:.4f} "
                     f"cong={bc:.4f}) -> {racc} relocs -> proxy={rp:.4f} "
                     f"(wl={rw:.4f} den={rd:.4f} cong={rc:.4f}) {verdict} "
                     f"in {time.monotonic()-t_rp:.1f}s")
            except Exception as exc:
                _log(f"  [RELOC_PROBE] failed: {type(exc).__name__}: {exc}")

        _log(f"  Best proxy={best_score:.4f}  total={time.monotonic()-t0:.1f}s")
        self._benchmarks_done += 1
        return best_pl
