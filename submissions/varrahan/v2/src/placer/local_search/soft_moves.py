"""Local search move operators."""

import os
import time
from typing import TYPE_CHECKING

import numpy as np

from placer.local_search.fields import _congestion_field, _density_field
from placer.ml.data_collection import TraceFields, get_candidate_trace, net_degree_features

if TYPE_CHECKING:
    from macro_place.benchmark import Benchmark

def _two_opt_soft_swap(
    soft_pos: np.ndarray,
    cw: float,
    ch: float,
    n: int,
    plc,
    benchmark: "Benchmark",
    incremental_scorer,
    initial_score: float,
    deadline: "float | None" = None,
    top_hot: int = 64,
    k_neighbors: int = 12,
    soft_movable: "np.ndarray | None" = None,
    use_density: bool = True,
    n_cold_teleports: int = 0,
    net_centroid: "np.ndarray | None" = None,
    wl_blend: float = 0.0,
) -> "tuple[np.ndarray, int, float]":
    """Pair-swap two SOFT macros' positions, accept-on-true-proxy.

    The exchange single-soft relocation can't make. Softs may overlap (no legality
    check; the proxy gate selects). Picks `top_hot` softs by the hotness field and
    for each tries its `k_neighbors` nearest movable softs plus `n_cold_teleports`
    coldest (long-range exchanges kNN can't reach). A macro is skipped once swapped.

    use_density: hotness field, occupancy (True) vs max(H,V) congestion (False);
        running both per round finds moves the other can't.
    net_centroid / wl_blend: blend toward the WL anchor in the kNN sort (ordering
        only).
    """
    num_soft = incremental_scorer.num_soft
    if num_soft < 2:
        return soft_pos, 0, initial_score
    nr, nc = benchmark.grid_rows, benchmark.grid_cols
    trace = get_candidate_trace()
    trace_field = "density" if use_density else "congestion"
    cell_w, cell_h = cw / nc, ch / nr

    # Hotness field: density (occupancy) or congestion (plc routing).
    cell_field = (_density_field(incremental_scorer, nr, nc) if use_density
                  else _congestion_field(plc, nr, nc))
    if cell_field is None:
        return soft_pos, 0, initial_score
    field_max = max(float(cell_field.max()), 1e-12)
    tf = None
    if trace is not None:
        tf = TraceFields(
            cong=_congestion_field(plc, nr, nc),
            dens=_density_field(incremental_scorer, nr, nc),
        )

    ci = np.clip((soft_pos[:, 0] / cell_w).astype(np.int64), 0, nc - 1)
    ri = np.clip((soft_pos[:, 1] / cell_h).astype(np.int64), 0, nr - 1)
    local_field = cell_field[ri, ci]
    order = np.argsort(-local_field)
    if soft_movable is not None:
        sm = np.asarray(soft_movable, dtype=bool)
        order = order[sm[order]]
    hot = order[:top_hot]
    if soft_movable is not None:
        movable_idx = np.where(np.asarray(soft_movable, dtype=bool))[0]
    else:
        movable_idx = np.arange(num_soft)
    movable_pos = soft_pos[movable_idx]

    # Precompute the n_cold_teleports globally-coldest movable softs once per
    # pass; appended to each hot macro's kNN candidates.
    cold_tele = None
    if n_cold_teleports > 0:
        movable_local_field = local_field[movable_idx]
        cold_order = np.argsort(movable_local_field)[:n_cold_teleports]
        cold_tele = movable_idx[cold_order]

    # Skip full scoring when the WL delta alone is too large to recover.
    # 3e-4 calibrated on ibm13 (_calibrate_wl_prefilter.py): rejects ~23% of
    # score_swap_soft calls, drops only ~0.2% of improving swaps. Tunable via env.
    _env_wl = os.environ.get("SOFT_2OPT_WL_PREFILTER")
    WL_PREFILTER = float(_env_wl) if _env_wl not in (None, "") else 3e-4

    best_score = initial_score
    accepts = 0
    swapped = np.zeros(num_soft, dtype=bool)
    for k1 in hot:
        k1 = int(k1)
        if swapped[k1]:
            continue
        if deadline is not None and time.monotonic() > deadline:
            break
        # Spatial kNN: nearest k_neighbors movable softs (skipping k1 itself),
        # optionally blended toward k1's WL anchor (net_centroid).
        d2 = ((movable_pos[:, 0] - soft_pos[k1, 0]) ** 2 +
              (movable_pos[:, 1] - soft_pos[k1, 1]) ** 2)
        if wl_blend > 0.0 and net_centroid is not None:
            d2c = ((movable_pos[:, 0] - net_centroid[k1, 0]) ** 2 +
                   (movable_pos[:, 1] - net_centroid[k1, 1]) ** 2)
            d2 = (1.0 - wl_blend) * d2 + wl_blend * d2c
        sorted_local = np.argsort(d2)
        nbrs = movable_idx[sorted_local]
        nbrs = nbrs[nbrs != k1][:k_neighbors]
        # A1c: append the cold-teleport candidates (dedup vs nbrs and exclude k1).
        if cold_tele is not None:
            extra = cold_tele[cold_tele != k1]
            extra = extra[~np.isin(extra, nbrs)]
            if extra.size > 0:
                nbrs = np.concatenate([nbrs, extra])

        best_pair = None  # (k2, xy1, xy2)
        state_score = best_score
        group_id = trace.next_group_id("soft_2opt") if trace is not None else None
        spatial_set = set(int(x) for x in nbrs[:k_neighbors])
        rejected_already_swapped = 0
        rejected_wl_prefilter = 0
        scored = 0
        for candidate_rank, k2 in enumerate(nbrs):
            k2 = int(k2)
            if swapped[k2]:
                rejected_already_swapped += 1
                continue
            # Swap: k1 takes k2's old position, k2 takes k1's old position.
            new_xy1 = (float(soft_pos[k2, 0]), float(soft_pos[k2, 1]))
            new_xy2 = (float(soft_pos[k1, 0]), float(soft_pos[k1, 1]))
            wl_d = incremental_scorer.wl_delta_swap_soft(k1, new_xy1, k2, new_xy2)
            if wl_d > WL_PREFILTER:
                rejected_wl_prefilter += 1
                continue
            s = incremental_scorer.score_swap_soft(k1, new_xy1, k2, new_xy2)
            scored += 1
            if trace is not None:
                trace.record(
                    operator="soft_2opt",
                    field=trace_field,
                    group_id=group_id,
                    state_score=state_score,
                    trial_score=s,
                    candidate_rank=candidate_rank,
                    group_size=len(nbrs),
                    candidate_source="spatial_knn" if k2 in spatial_set else "cold_teleport",
                    features={
                        **net_degree_features(
                            incremental_scorer,
                            incremental_scorer.soft_indices[k1],
                            "k1_",
                        ),
                        **net_degree_features(
                            incremental_scorer,
                            incremental_scorer.soft_indices[k2],
                            "k2_",
                        ),
                        "accepted_in_pass": accepts,
                        "distance_norm": float(
                            np.hypot(
                                soft_pos[k1, 0] - soft_pos[k2, 0],
                                soft_pos[k1, 1] - soft_pos[k2, 1],
                            )
                            / np.hypot(cw, ch)
                        ),
                        "k1_field_norm": float(local_field[k1] / field_max),
                        "k2_field_norm": float(local_field[k2] / field_max),
                        "k1_congestion_norm": tf.cong_at(ri[k1], ci[k1]),
                        "k2_congestion_norm": tf.cong_at(ri[k2], ci[k2]),
                        "k1_density_norm": tf.dens_at(ri[k1], ci[k1]),
                        "k2_density_norm": tf.dens_at(ri[k2], ci[k2]),
                        "wl_delta": float(wl_d),
                        "source_hot_rank_norm": float(
                            np.where(hot == k1)[0][0] / max(len(hot) - 1, 1)
                        ),
                    },
                )
            if s < best_score - 1e-9:
                best_score = s
                best_pair = (k2, new_xy1, new_xy2)

        if best_pair is not None:
            k2_win, xy1, xy2 = best_pair
            incremental_scorer.commit_swap_soft(k1, xy1, k2_win, xy2)
            soft_pos[k1, 0], soft_pos[k1, 1] = xy1
            soft_pos[k2_win, 0], soft_pos[k2_win, 1] = xy2
            swapped[k1] = True
            swapped[k2_win] = True
            accepts += 1
        if trace is not None:
            trace.event(
                "candidate_group_summary",
                operator="soft_2opt",
                field=trace_field,
                group_id=group_id,
                generated=int(len(nbrs)),
                scored=scored,
                rejected_already_swapped=rejected_already_swapped,
                rejected_wl_prefilter=rejected_wl_prefilter,
            )
    return soft_pos, accepts, best_score
