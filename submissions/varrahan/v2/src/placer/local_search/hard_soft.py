"""Local search move operators."""

import time

import numpy as np

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
    """HXS (2026-05-30): hard ⇄ soft cross-swap. Exchanges one hard macro's
    position with one soft macro's position. The hard-2opt swaps only hards,
    and the soft-2opt swaps only softs - neither can find a hard/soft pair
    whose configurations would be improved by trading places (e.g., a routing-
    heavy hard sitting in a quiet area + a high-density soft in a congested
    corridor).

    Hot list: top_hot hardest hards by the chosen field (max(H,V) congestion
    when use_density=False, occupancy when True). For each hot hard, candidate
    swap partners = its k_neighbors nearest movable softs.

    Legality: the hard's new position (the soft's old position) must satisfy
    in-bounds + no overlap with OTHER hard macros (overlap with the hard's own
    old footprint is fine - that's where it came from). The soft going to the
    hard's old slot has no legality check (softs may overlap). Accept-on-true-
    proxy via score_swap_hard_soft / commit_swap_hard_soft.

    Returns (hard_pos, soft_pos, accepts, best_score)."""
    num_soft = incremental_scorer.num_soft
    if num_soft < 1:
        return hard_pos, soft_pos, 0, initial_score
    nr, nc = benchmark.grid_rows, benchmark.grid_cols
    cell_w, cell_h = cw / nc, ch / nr

    if use_density:
        go = incremental_scorer.grid_occupied
        if go is None or go.size != nr * nc:
            return hard_pos, soft_pos, 0, initial_score
        cell_field = (go / incremental_scorer.dens_grid_area).reshape(nr, nc)
    else:
        try:
            h_arr = np.asarray(plc.get_horizontal_routing_congestion(), dtype=np.float64)
            v_arr = np.asarray(plc.get_vertical_routing_congestion(), dtype=np.float64)
        except Exception:
            return hard_pos, soft_pos, 0, initial_score
        if h_arr.size != nr * nc or v_arr.size != nr * nc:
            return hard_pos, soft_pos, 0, initial_score
        cell_field = np.maximum(h_arr.reshape(nr, nc), v_arr.reshape(nr, nc))

    # Hot hards by chosen field.
    hci = np.clip((hard_pos[:n, 0] / cell_w).astype(np.int64), 0, nc - 1)
    hri = np.clip((hard_pos[:n, 1] / cell_h).astype(np.int64), 0, nr - 1)
    local_h = cell_field[hri, hci]
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
    sep_x_mat = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2
    sep_y_mat = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2
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
        for k_soft in nbrs:
            k_soft = int(k_soft)
            if swapped_soft[k_soft]:
                continue
            # Hard takes the soft's position; soft takes the hard's position.
            hx, hy = float(soft_pos[k_soft, 0]), float(soft_pos[k_soft, 1])
            sx, sy = float(hard_pos[i, 0]), float(hard_pos[i, 1])
            # In-bounds for the hard at its new position.
            if (hx - hw[i] < -EPS or hx + hw[i] > cw + EPS or
                    hy - hh[i] < -EPS or hy + hh[i] > ch + EPS):
                continue
            # No overlap with other hard macros at (hx, hy).
            if ((np.abs(hx - ox) < sxi + EPS) & (np.abs(hy - oy) < syi + EPS)).any():
                continue
            s = incremental_scorer.score_swap_hard_soft(i, (hx, hy), k_soft, (sx, sy))
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
    """HS3 (2026-05-31): hard-soft-soft 3-cycle rotation.

    For each hot hard H, considers the cycle H → S1 → S2 → H where S1 is
    one of H's k_inner nearest movable softs, and S2 is one of S1's
    k_inner nearest movable softs. The cycle is: H takes S1's old pos,
    S1 takes S2's old pos, S2 takes H's old pos.

    Captures patterns no 2-opt can: when a hard wants S1's slot but
    swapping H↔S1 hurts because S1's current connections need to go
    elsewhere, the 3-cycle finds a single combined move that 2-opt
    (which would have to accept S1↔S2 separately) cannot reach.

    Legality: hard's new position (S1's old position) must satisfy
    in-bounds + no overlap with OTHER hard macros (overlap with hard's
    own old footprint is OK). Softs may overlap, no legality check for
    them.

    Cost: O(top_hot × k_inner²) trials. With top_hot=15, k_inner=5 →
    ~375 trials × ~10ms = ~3.8s/pass - gated by a tight deadline."""
    num_soft = incremental_scorer.num_soft
    if num_soft < 2:
        return hard_pos, soft_pos, 0, initial_score
    nr, nc = benchmark.grid_rows, benchmark.grid_cols
    cell_w, cell_h = cw / nc, ch / nr

    if use_density:
        go = incremental_scorer.grid_occupied
        if go is None or go.size != nr * nc:
            return hard_pos, soft_pos, 0, initial_score
        cell_field = (go / incremental_scorer.dens_grid_area).reshape(nr, nc)
    else:
        try:
            h_arr = np.asarray(plc.get_horizontal_routing_congestion(), dtype=np.float64)
            v_arr = np.asarray(plc.get_vertical_routing_congestion(), dtype=np.float64)
        except Exception:
            return hard_pos, soft_pos, 0, initial_score
        if h_arr.size != nr * nc or v_arr.size != nr * nc:
            return hard_pos, soft_pos, 0, initial_score
        cell_field = np.maximum(h_arr.reshape(nr, nc), v_arr.reshape(nr, nc))

    # Hot hards by chosen field.
    hci = np.clip((hard_pos[:n, 0] / cell_w).astype(np.int64), 0, nc - 1)
    hri = np.clip((hard_pos[:n, 1] / cell_h).astype(np.int64), 0, nr - 1)
    local_h = cell_field[hri, hci]
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

    sep_x_mat = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2
    sep_y_mat = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2
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
        for k1 in s1_cands:
            k1 = int(k1)
            if swapped_soft[k1]:
                continue
            # Hard's new position = S1's old position. Check legality.
            hx, hy = float(soft_pos[k1, 0]), float(soft_pos[k1, 1])
            if (hx - hw[i] < -EPS or hx + hw[i] > cw + EPS or
                    hy - hh[i] < -EPS or hy + hh[i] > ch + EPS):
                continue
            if ((np.abs(hx - ox) < sxi + EPS) & (np.abs(hy - oy) < syi + EPS)).any():
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
                    continue
                # Cycle: H → S1's old, S1 → S2's old, S2 → H's old.
                s1_new_xy = (float(soft_pos[k2, 0]), float(soft_pos[k2, 1]))
                s2_new_xy = (float(hard_pos[i, 0]), float(hard_pos[i, 1]))
                s = incremental_scorer.score_cycle_hard_soft_soft(
                    i, (hx, hy), k1, s1_new_xy, k2, s2_new_xy
                )
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
    return hard_pos, soft_pos, accepts, best_score


