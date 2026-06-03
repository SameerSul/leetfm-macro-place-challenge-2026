"""Local search move operators."""

import time

import numpy as np
import torch

from placer.config import _GPU_DEVICE, _USE_GPU

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
        # GPU path: O(N^2) pairwise squared-L2 via the identity
        #   |a-b|^2 = |a|^2 + |b|^2 - 2<a,b>
        # using torch.mm (matrix multiply) — works correctly on DirectML, CUDA,
        # and CPU. torch.cdist is NOT used: on DirectML it returns a wrong shape
        # ([N,1,2] instead of [N,N]).
        if _USE_GPU:
            with torch.no_grad():
                pos_t = torch.from_numpy(pos).to(_GPU_DEVICE, dtype=torch.float32)
                xn = (pos_t * pos_t).sum(-1)          # [N] squared norms
                d_pair = (xn.unsqueeze(1) + xn.unsqueeze(0)
                          - 2.0 * torch.mm(pos_t, pos_t.t())).cpu().numpy().astype(np.float64)
        else:
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


