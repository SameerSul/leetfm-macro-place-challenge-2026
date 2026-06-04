"""Local search move operators."""

import time
from typing import TYPE_CHECKING

import numpy as np

from placer.geometry import separation_matrices
from placer.local_search.fields import _congestion_field, _density_field
from placer.ml.data_collection import TraceFields, get_candidate_trace, net_degree_features

if TYPE_CHECKING:
    from macro_place.benchmark import Benchmark

def _two_opt_hard_soft_swap(
    hard_pos: np.ndarray,
    soft_pos: np.ndarray,
    sizes: np.ndarray,
    hw: np.ndarray,
    hh: np.ndarray,
    cw: float,
    ch: float,
    movable_hard: np.ndarray,
    n: int,
    plc,
    benchmark: "Benchmark",
    incremental_scorer,
    initial_score: float,
    deadline: "float | None" = None,
    top_hot: int = 24,
    k_neighbors: int = 12,
    soft_movable: "np.ndarray | None" = None,
    use_density: bool = False,
) -> "tuple[np.ndarray, np.ndarray, int, float]":
    """Hard-soft cross-swap: exchange a hard macro's position with a soft's -
    pairs neither hard-2opt nor soft-2opt can find (each swaps within its kind).

    Hot list = top_hot hottest hards by the chosen field (max(H,V) congestion, or
    occupancy when use_density=True); partners = each one's k_neighbors nearest
    movable softs. Legality: the hard's new slot must be in-bounds with no overlap
    vs OTHER hards (softs may overlap, no check). Accept-on-true-proxy.
    Returns (hard_pos, soft_pos, accepts, best_score).
    """
    num_soft = incremental_scorer.num_soft
    if num_soft < 1:
        return hard_pos, soft_pos, 0, initial_score
    nr, nc = benchmark.grid_rows, benchmark.grid_cols
    trace = get_candidate_trace()
    trace_field = "density" if use_density else "congestion"
    cell_w, cell_h = cw / nc, ch / nr

    cell_field = (_density_field(incremental_scorer, nr, nc) if use_density
                  else _congestion_field(plc, nr, nc))
    if cell_field is None:
        return hard_pos, soft_pos, 0, initial_score
    field_max = max(float(cell_field.max()), 1e-12)
    tf = None
    if trace is not None:
        tf = TraceFields(
            cong=_congestion_field(plc, nr, nc),
            dens=_density_field(incremental_scorer, nr, nc),
        )

    # Hot hards by chosen field.
    hci = np.clip((hard_pos[:n, 0] / cell_w).astype(np.int64), 0, nc - 1)
    hri = np.clip((hard_pos[:n, 1] / cell_h).astype(np.int64), 0, nr - 1)
    local_h = cell_field[hri, hci]
    sci = np.clip((soft_pos[:, 0] / cell_w).astype(np.int64), 0, nc - 1)
    sri = np.clip((soft_pos[:, 1] / cell_h).astype(np.int64), 0, nr - 1)
    mov_hard = np.where(movable_hard)[0]
    if mov_hard.size == 0:
        return hard_pos, soft_pos, 0, initial_score
    hot = mov_hard[np.argsort(-local_h[mov_hard])][:top_hot]

    if soft_movable is not None:
        movable_soft_idx = np.where(np.asarray(soft_movable, dtype=bool))[0]
    else:
        movable_soft_idx = np.arange(num_soft)
    if movable_soft_idx.size == 0:
        return hard_pos, soft_pos, 0, initial_score
    movable_soft_pos = soft_pos[movable_soft_idx]

    # Pairwise hard-hard separation (for legality of the hard's destination).
    sep_x_mat, sep_y_mat = separation_matrices(sizes)
    EPS = 0.05
    all_hard_idx = np.arange(n)

    best_score = initial_score
    accepts = 0
    swapped_hard = np.zeros(n, dtype=bool)
    swapped_soft = np.zeros(num_soft, dtype=bool)
    for i in hot:
        i = int(i)
        if swapped_hard[i]:
            continue
        if deadline is not None and time.monotonic() > deadline:
            break

        # kNN over movable softs.
        d2 = ((movable_soft_pos[:, 0] - hard_pos[i, 0]) ** 2 +
              (movable_soft_pos[:, 1] - hard_pos[i, 1]) ** 2)
        sorted_local = np.argsort(d2)
        nbrs = movable_soft_idx[sorted_local][:k_neighbors]

        # Pre-mask: legality check vs OTHER hards (exclude i from the
        # collision set since i is the one being moved away).
        mask = all_hard_idx != i
        sxi = sep_x_mat[i, mask]
        syi = sep_y_mat[i, mask]
        ox = hard_pos[mask, 0]
        oy = hard_pos[mask, 1]

        best_pair = None  # (k_soft, hard_new_xy, soft_new_xy)
        state_score = best_score
        group_id = trace.next_group_id("hard_soft_swap") if trace is not None else None
        rejected_bounds = 0
        rejected_overlap = 0
        rejected_already_swapped = 0
        scored = 0
        for candidate_rank, k_soft in enumerate(nbrs):
            k_soft = int(k_soft)
            if swapped_soft[k_soft]:
                rejected_already_swapped += 1
                continue
            # Hard takes the soft's position; soft takes the hard's position.
            hx, hy = float(soft_pos[k_soft, 0]), float(soft_pos[k_soft, 1])
            sx, sy = float(hard_pos[i, 0]), float(hard_pos[i, 1])
            # In-bounds for the hard at its new position.
            if (hx - hw[i] < -EPS or hx + hw[i] > cw + EPS or
                    hy - hh[i] < -EPS or hy + hh[i] > ch + EPS):
                rejected_bounds += 1
                continue
            # No overlap with other hard macros at (hx, hy).
            if ((np.abs(hx - ox) < sxi + EPS) & (np.abs(hy - oy) < syi + EPS)).any():
                rejected_overlap += 1
                continue
            s = incremental_scorer.score_swap_hard_soft(i, (hx, hy), k_soft, (sx, sy))
            scored += 1
            if trace is not None:
                trace.record(
                    operator="hard_soft_swap",
                    field=trace_field,
                    group_id=group_id,
                    state_score=state_score,
                    trial_score=s,
                    candidate_rank=candidate_rank,
                    group_size=len(nbrs),
                    candidate_source="spatial_knn",
                    features={
                        **net_degree_features(
                            incremental_scorer,
                            incremental_scorer.hard_indices[i],
                            "hard_",
                        ),
                        **net_degree_features(
                            incremental_scorer,
                            incremental_scorer.soft_indices[k_soft],
                            "soft_",
                        ),
                        "accepted_in_pass": accepts,
                        "hard_w_norm": float(sizes[i, 0] / cw),
                        "hard_h_norm": float(sizes[i, 1] / ch),
                        "distance_norm": float(np.hypot(hx - sx, hy - sy) / np.hypot(cw, ch)),
                        "hard_field_norm": float(local_h[i] / field_max),
                        "hard_congestion_norm": tf.cong_at(hri[i], hci[i]),
                        "soft_congestion_norm": tf.cong_at(sri[k_soft], sci[k_soft]),
                        "hard_density_norm": tf.dens_at(hri[i], hci[i]),
                        "soft_density_norm": tf.dens_at(sri[k_soft], sci[k_soft]),
                        "source_hot_rank_norm": float(
                            np.where(hot == i)[0][0] / max(len(hot) - 1, 1)
                        ),
                    },
                )
            if s < best_score - 1e-9:
                best_score = s
                best_pair = (k_soft, (hx, hy), (sx, sy))

        if best_pair is not None:
            k_win, h_xy, s_xy = best_pair
            incremental_scorer.commit_swap_hard_soft(i, h_xy, k_win, s_xy)
            hard_pos[i, 0], hard_pos[i, 1] = h_xy
            soft_pos[k_win, 0], soft_pos[k_win, 1] = s_xy
            swapped_hard[i] = True
            swapped_soft[k_win] = True
            accepts += 1
        if trace is not None:
            trace.event(
                "candidate_group_summary",
                operator="hard_soft_swap",
                field=trace_field,
                group_id=group_id,
                generated=int(len(nbrs)),
                scored=scored,
                rejected_bounds=rejected_bounds,
                rejected_overlap=rejected_overlap,
                rejected_already_swapped=rejected_already_swapped,
            )
    return hard_pos, soft_pos, accepts, best_score


def _three_opt_hard_soft_soft(
    hard_pos: np.ndarray,
    soft_pos: np.ndarray,
    sizes: np.ndarray,
    hw: np.ndarray,
    hh: np.ndarray,
    cw: float,
    ch: float,
    movable_hard: np.ndarray,
    n: int,
    plc,
    benchmark: "Benchmark",
    incremental_scorer,
    initial_score: float,
    deadline: "float | None" = None,
    top_hot: int = 15,
    k_inner: int = 5,
    soft_movable: "np.ndarray | None" = None,
    use_density: bool = False,
) -> "tuple[np.ndarray, np.ndarray, int, float]":
    """Hard-soft-soft 3-cycle: H takes S1's slot, S1 takes S2's, S2 takes H's.

    For each hot hard H, S1 ranges over H's k_inner nearest movable softs and S2
    over S1's. Reaches what 2-opt can't: when H wants S1's slot but H<->S1 alone
    hurts (S1 must move too), the combined cycle accepts. Legality: H's new slot
    in-bounds with no overlap vs OTHER hards (softs may overlap). Cost is
    O(top_hot * k_inner^2), deadline-gated.
    """
    num_soft = incremental_scorer.num_soft
    if num_soft < 2:
        return hard_pos, soft_pos, 0, initial_score
    nr, nc = benchmark.grid_rows, benchmark.grid_cols
    trace = get_candidate_trace()
    trace_field = "density" if use_density else "congestion"
    cell_w, cell_h = cw / nc, ch / nr

    cell_field = (_density_field(incremental_scorer, nr, nc) if use_density
                  else _congestion_field(plc, nr, nc))
    if cell_field is None:
        return hard_pos, soft_pos, 0, initial_score
    field_max = max(float(cell_field.max()), 1e-12)
    tf = None
    if trace is not None:
        tf = TraceFields(
            cong=_congestion_field(plc, nr, nc),
            dens=_density_field(incremental_scorer, nr, nc),
        )

    # Hot hards by chosen field.
    hci = np.clip((hard_pos[:n, 0] / cell_w).astype(np.int64), 0, nc - 1)
    hri = np.clip((hard_pos[:n, 1] / cell_h).astype(np.int64), 0, nr - 1)
    local_h = cell_field[hri, hci]
    sci = np.clip((soft_pos[:, 0] / cell_w).astype(np.int64), 0, nc - 1)
    sri = np.clip((soft_pos[:, 1] / cell_h).astype(np.int64), 0, nr - 1)
    mov_hard = np.where(movable_hard)[0]
    if mov_hard.size == 0:
        return hard_pos, soft_pos, 0, initial_score
    hot = mov_hard[np.argsort(-local_h[mov_hard])][:top_hot]

    if soft_movable is not None:
        movable_soft_idx = np.where(np.asarray(soft_movable, dtype=bool))[0]
    else:
        movable_soft_idx = np.arange(num_soft)
    if movable_soft_idx.size < 2:
        return hard_pos, soft_pos, 0, initial_score
    movable_soft_pos = soft_pos[movable_soft_idx]

    sep_x_mat, sep_y_mat = separation_matrices(sizes)
    EPS = 0.05
    all_hard_idx = np.arange(n)

    best_score = initial_score
    accepts = 0
    swapped_hard = np.zeros(n, dtype=bool)
    swapped_soft = np.zeros(num_soft, dtype=bool)

    for i in hot:
        i = int(i)
        if swapped_hard[i]:
            continue
        if deadline is not None and time.monotonic() > deadline:
            break

        # kNN softs around hard i - candidates for S1.
        d2_h = ((movable_soft_pos[:, 0] - hard_pos[i, 0]) ** 2 +
                (movable_soft_pos[:, 1] - hard_pos[i, 1]) ** 2)
        s1_order = np.argsort(d2_h)[:k_inner]
        s1_cands = movable_soft_idx[s1_order]

        # Hard legality pre-mask (other hards' positions).
        mask = all_hard_idx != i
        sxi = sep_x_mat[i, mask]
        syi = sep_y_mat[i, mask]
        ox = hard_pos[mask, 0]
        oy = hard_pos[mask, 1]

        best_triple = None
        state_score = best_score
        group_id = trace.next_group_id("hard_soft_soft_cycle") if trace is not None else None
        generated = 0
        rejected_bounds = 0
        rejected_overlap = 0
        rejected_already_swapped = 0
        scored = 0
        candidate_rank = 0
        for k1_rank, k1 in enumerate(s1_cands):
            k1 = int(k1)
            if swapped_soft[k1]:
                rejected_already_swapped += 1
                continue
            # Hard's new position = S1's old position. Check legality.
            hx, hy = float(soft_pos[k1, 0]), float(soft_pos[k1, 1])
            if (hx - hw[i] < -EPS or hx + hw[i] > cw + EPS or
                    hy - hh[i] < -EPS or hy + hh[i] > ch + EPS):
                rejected_bounds += 1
                continue
            if ((np.abs(hx - ox) < sxi + EPS) & (np.abs(hy - oy) < syi + EPS)).any():
                rejected_overlap += 1
                continue

            # kNN softs around k1 - candidates for S2.
            d2_k1 = ((movable_soft_pos[:, 0] - soft_pos[k1, 0]) ** 2 +
                     (movable_soft_pos[:, 1] - soft_pos[k1, 1]) ** 2)
            # +1 in case k1 is in its own neighbor list.
            s2_order = np.argsort(d2_k1)[:k_inner + 1]
            s2_cands = movable_soft_idx[s2_order]

            for k2 in s2_cands:
                k2 = int(k2)
                if k2 == k1 or swapped_soft[k2]:
                    rejected_already_swapped += 1
                    continue
                generated += 1
                # Cycle: H → S1's old, S1 → S2's old, S2 → H's old.
                s1_new_xy = (float(soft_pos[k2, 0]), float(soft_pos[k2, 1]))
                s2_new_xy = (float(hard_pos[i, 0]), float(hard_pos[i, 1]))
                s = incremental_scorer.score_cycle_hard_soft_soft(
                    i, (hx, hy), k1, s1_new_xy, k2, s2_new_xy
                )
                scored += 1
                if trace is not None:
                    trace.record(
                        operator="hard_soft_soft_cycle",
                        field=trace_field,
                        group_id=group_id,
                        state_score=state_score,
                        trial_score=s,
                        candidate_rank=candidate_rank,
                        group_size=k_inner * k_inner,
                        candidate_source="nested_spatial_knn",
                        features={
                            **net_degree_features(
                                incremental_scorer,
                                incremental_scorer.hard_indices[i],
                                "hard_",
                            ),
                            **net_degree_features(
                                incremental_scorer,
                                incremental_scorer.soft_indices[k1],
                                "s1_",
                            ),
                            **net_degree_features(
                                incremental_scorer,
                                incremental_scorer.soft_indices[k2],
                                "s2_",
                            ),
                            "accepted_in_pass": accepts,
                            "hard_w_norm": float(sizes[i, 0] / cw),
                            "hard_h_norm": float(sizes[i, 1] / ch),
                            "hard_s1_distance_norm": float(
                                np.hypot(hx - hard_pos[i, 0], hy - hard_pos[i, 1])
                                / np.hypot(cw, ch)
                            ),
                            "s1_s2_distance_norm": float(
                                np.hypot(
                                    soft_pos[k2, 0] - soft_pos[k1, 0],
                                    soft_pos[k2, 1] - soft_pos[k1, 1],
                                )
                                / np.hypot(cw, ch)
                            ),
                            "hard_field_norm": float(local_h[i] / field_max),
                            "hard_congestion_norm": tf.cong_at(hri[i], hci[i]),
                            "s1_congestion_norm": tf.cong_at(sri[k1], sci[k1]),
                            "s2_congestion_norm": tf.cong_at(sri[k2], sci[k2]),
                            "hard_density_norm": tf.dens_at(hri[i], hci[i]),
                            "s1_density_norm": tf.dens_at(sri[k1], sci[k1]),
                            "s2_density_norm": tf.dens_at(sri[k2], sci[k2]),
                            "source_hot_rank_norm": float(
                                np.where(hot == i)[0][0] / max(len(hot) - 1, 1)
                            ),
                            "s1_rank_norm": float(k1_rank / max(len(s1_cands) - 1, 1)),
                        },
                    )
                candidate_rank += 1
                if s < best_score - 1e-9:
                    best_score = s
                    best_triple = (k1, (hx, hy), s1_new_xy, k2, s2_new_xy)

        if best_triple is not None:
            k1_win, h_xy, s1_xy, k2_win, s2_xy = best_triple
            incremental_scorer.commit_cycle_hard_soft_soft(
                i, h_xy, k1_win, s1_xy, k2_win, s2_xy
            )
            hard_pos[i, 0], hard_pos[i, 1] = h_xy
            soft_pos[k1_win, 0], soft_pos[k1_win, 1] = s1_xy
            soft_pos[k2_win, 0], soft_pos[k2_win, 1] = s2_xy
            swapped_hard[i] = True
            swapped_soft[k1_win] = True
            swapped_soft[k2_win] = True
            accepts += 1
        if trace is not None:
            trace.event(
                "candidate_group_summary",
                operator="hard_soft_soft_cycle",
                field=trace_field,
                group_id=group_id,
                generated=generated,
                scored=scored,
                rejected_bounds=rejected_bounds,
                rejected_overlap=rejected_overlap,
                rejected_already_swapped=rejected_already_swapped,
            )
    return hard_pos, soft_pos, accepts, best_score
