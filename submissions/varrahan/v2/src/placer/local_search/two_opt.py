"""Local search move operators."""

import time

import numpy as np
import torch

from placer.config import _GPU_DEVICE, _USE_GPU
from placer.geometry import separation_matrices
from placer.local_search.fields import _density_field
from placer.ml.data_collection import TraceFields, get_candidate_trace, net_degree_features

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
    """Proxy-driven 2-opt swap pass.

    Accepts a swap iff the resulting placement's proxy cost (via `score_fn` or
    an incremental scorer) strictly decreases. Each candidate that passes the
    cheap bounds + neighbor-conflict checks is applied tentatively, scored, then
    kept or reverted. Restricts candidates to each macro's spatial kNN; iterates
    up to `max_iters` or until no improvement. `deadline` caps the budget (the
    full O(n²·k) set won't fit on large n).

    Returns: (pos, accept_count, final_score, score_calls).
    """
    sep_x_mat, sep_y_mat = separation_matrices(sizes)
    EPS = 0.05

    pos = legal_pos.copy()
    trace = get_candidate_trace()
    tf = None
    trace_density_ri = trace_density_ci = None
    if trace is not None and incremental_scorer is not None:
        bm = incremental_scorer.benchmark
        tf = TraceFields(dens=_density_field(incremental_scorer, bm.grid_rows, bm.grid_cols))
        if tf.dens is not None:
            trace_density_ci = np.clip(
                (pos[:, 0] / (cw / bm.grid_cols)).astype(np.int64), 0, bm.grid_cols - 1
            )
            trace_density_ri = np.clip(
                (pos[:, 1] / (ch / bm.grid_rows)).astype(np.int64), 0, bm.grid_rows - 1
            )
    best_score = initial_score
    accept_count = 0
    score_calls = 0

    # Congestion-aware candidate selection (when per-macro `macro_cong` is given):
    #   * order macros hot→cold so the budget goes to the hotspots that dominate
    #     the proxy.
    #   * for the cong_hot_k hottest, append the cong_cold_k coldest as "teleport"
    #     candidates beyond the spatial kNN (size-incompatible ones fail the
    #     conflict check before scoring).
    # The proxy gate validates every swap, so this only changes which candidates
    # are tried; macro_cong=None is the index-order / spatial-only path.
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
        mc_scale = max(float(np.max(np.abs(mc[mov_idx]))), 1e-12)
    else:
        outer_order = np.arange(n)
        hot_rows = set()
        cold_pool = np.empty(0, dtype=np.int64)
        mc_scale = 1.0

    it = 0
    while it < max_iters:
        if deadline is not None and time.monotonic() > deadline:
            break
        improved_any = False

        # kNN per macro (re-derived each outer iter; positions change on accept).
        # GPU path: pairwise squared-L2 via |a-b|² = |a|²+|b|²-2<a,b> with
        # torch.mm (cdist returns a wrong shape on DirectML, so it's avoided).
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
            state_score = best_score
            group_id = trace.next_group_id("hard_2opt") if trace is not None else None
            rejected_bounds = 0
            rejected_overlap = 0
            scored = 0
            spatial_set = set(int(x) for x in neighbors[i])
            for candidate_rank, j in enumerate(cand_js):
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
                    rejected_bounds += 1
                    continue
                if (new_jx - hw[j] < -EPS or new_jx + hw[j] > cw + EPS or
                        new_jy - hh[j] < -EPS or new_jy + hh[j] > ch + EPS):
                    rejected_bounds += 1
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
                    rejected_overlap += 1
                    continue
                sxj = sep_x_mat[j, mask]
                syj = sep_y_mat[j, mask]
                conf_j = ((np.abs(new_jx - ox) < sxj + EPS) &
                          (np.abs(new_jy - oy) < syj + EPS)).any()
                if conf_j:
                    rejected_overlap += 1
                    continue
                # i↔j separation (already legal pre-swap; verify defensively)
                if (abs(new_ix - new_jx) < sep_x_mat[i, j] + EPS and
                        abs(new_iy - new_jy) < sep_y_mat[i, j] + EPS):
                    rejected_overlap += 1
                    continue

                # Apply swap tentatively, score, decide
                old_ix, old_iy = pos[i, 0], pos[i, 1]
                old_jx, old_jy = pos[j, 0], pos[j, 1]
                pos[i, 0], pos[i, 1] = new_ix, new_iy
                pos[j, 0], pos[j, 1] = new_jx, new_jy

                # Incremental scorer recomputes only touched-net WL + reuses plc
                # for density/congestion; score_fn is the full-recompute fallback.
                if incremental_scorer is not None:
                    trial_score = incremental_scorer.score_swap(
                        i, (new_ix, new_iy), j, (new_jx, new_jy)
                    )
                else:
                    trial_score = score_fn(pos)
                score_calls += 1
                scored += 1
                if trace is not None:
                    degree_features = {}
                    if incremental_scorer is not None:
                        degree_features.update(
                            net_degree_features(
                                incremental_scorer,
                                incremental_scorer.hard_indices[i],
                                "i_",
                            )
                        )
                        degree_features.update(
                            net_degree_features(
                                incremental_scorer,
                                incremental_scorer.hard_indices[j],
                                "j_",
                            )
                        )
                    trace.record(
                        operator="hard_2opt",
                        field="congestion" if cong_aware else "spatial",
                        group_id=group_id,
                        state_score=state_score,
                        trial_score=trial_score,
                        candidate_rank=candidate_rank,
                        group_size=len(cand_js),
                        candidate_source="spatial_knn" if j in spatial_set else "cold_teleport",
                        features={
                            **degree_features,
                            "accepted_in_pass": accept_count,
                            "i_w_norm": float(sizes[i, 0] / cw),
                            "i_h_norm": float(sizes[i, 1] / ch),
                            "j_w_norm": float(sizes[j, 0] / cw),
                            "j_h_norm": float(sizes[j, 1] / ch),
                            "i_x_norm": float(old_ix / cw),
                            "i_y_norm": float(old_iy / ch),
                            "j_x_norm": float(old_jx / cw),
                            "j_y_norm": float(old_jy / ch),
                            "distance_norm": float(
                                np.hypot(old_ix - old_jx, old_iy - old_jy) / np.hypot(cw, ch)
                            ),
                            "i_congestion_norm": float(mc[i] / mc_scale) if cong_aware else 0.0,
                            "j_congestion_norm": float(mc[j] / mc_scale) if cong_aware else 0.0,
                            "i_density_norm": (
                                tf.dens_at(trace_density_ri[i], trace_density_ci[i])
                                if tf.dens is not None
                                else 0.0
                            ),
                            "j_density_norm": (
                                tf.dens_at(trace_density_ri[j], trace_density_ci[j])
                                if tf.dens is not None
                                else 0.0
                            ),
                            "source_hot_rank_norm": float(
                                np.where(outer_order == i)[0][0] / max(len(outer_order) - 1, 1)
                            ),
                        },
                    )
                if trial_score < best_score:
                    if incremental_scorer is not None:
                        incremental_scorer.commit_swap(
                            i, (new_ix, new_iy), j, (new_jx, new_jy)
                        )
                    best_score = trial_score
                    accept_count += 1
                    improved_any = True
                    break  # positions changed; refresh kNN at next outer iter
                else:
                    # Revert (scorer already reverted plc internally)
                    pos[i, 0], pos[i, 1] = old_ix, old_iy
                    pos[j, 0], pos[j, 1] = old_jx, old_jy
            if trace is not None:
                trace.event(
                    "candidate_group_summary",
                    operator="hard_2opt",
                    field="congestion" if cong_aware else "spatial",
                    group_id=group_id,
                    generated=int(len(cand_js)),
                    scored=scored,
                    rejected_bounds=rejected_bounds,
                    rejected_overlap=rejected_overlap,
                    stopped_after_accept=bool(improved_any),
                )

        if not improved_any:
            break
        it += 1

    return pos, accept_count, best_score, score_calls
