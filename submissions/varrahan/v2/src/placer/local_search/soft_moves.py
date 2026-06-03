"""Local search move operators."""

import time

import numpy as np

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
    """A1 (2026-05-29): pair-swap two SOFT macros' positions, accept-on-true-
    proxy. The single-soft relocation pass can't find moves where two softs
    need to EXCHANGE places — this adds that move type. Softs may overlap so
    there's no legality check on the destinations; the proxy gate handles
    selection.

    A4 (2026-05-30): `net_centroid` + `wl_blend` — analog of A3 candidate
    ordering for the swap. The kNN sort can blend distance-to-current with
    distance-to-net-centroid so swap partners aligned with the hot soft's WL
    anchor are tried first. Pure ordering change; the proxy gate still
    validates every swap → strictly non-regressing.

    A1b: `use_density` selects the hotness field — True (default) uses grid
    occupancy (the original A1 lever); False uses `max(H,V)` routing congestion
    (analog of R3 for soft-2opt). Running BOTH fields per round (A1×2)
    finds moves the other can't — same dual-field symmetry that gave R3 + R5
    its compound gain.

    A1c: `n_cold_teleports` appends the N globally-coldest movable softs as
    extra swap candidates per hot. Spatial kNN alone can never find a long-
    range exchange between a hot and a far cold soft — analogous to S9's
    cold-teleport augmentation for hard 2-opt.

    Candidate selection: pick the `top_hot` softs by the chosen field, and for
    each such k1, try swapping with its `k_neighbors` nearest movable softs
    plus `n_cold_teleports` globally coldest movable softs. Skip pairs once
    either macro has been swapped this pass."""
    num_soft = incremental_scorer.num_soft
    if num_soft < 2:
        return soft_pos, 0, initial_score
    nr, nc = benchmark.grid_rows, benchmark.grid_cols
    cell_w, cell_h = cw / nc, ch / nr

    # Hotness field: density (occupancy) or congestion (plc routing).
    if use_density:
        go = incremental_scorer.grid_occupied
        if go is None or go.size != nr * nc:
            return soft_pos, 0, initial_score
        cell_field = (go / incremental_scorer.dens_grid_area).reshape(nr, nc)
    else:
        try:
            h_arr = np.asarray(plc.get_horizontal_routing_congestion(), dtype=np.float64)
            v_arr = np.asarray(plc.get_vertical_routing_congestion(), dtype=np.float64)
        except Exception:
            return soft_pos, 0, initial_score
        if h_arr.size != nr * nc or v_arr.size != nr * nc:
            return soft_pos, 0, initial_score
        cell_field = np.maximum(h_arr.reshape(nr, nc), v_arr.reshape(nr, nc))

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

    # A1c: precompute the n_cold_teleports globally-coldest movable softs once
    # per pass (cheap — O(num_soft)). Appended to each hot's kNN candidates.
    cold_tele = None
    if n_cold_teleports > 0:
        movable_local_field = local_field[movable_idx]
        cold_order = np.argsort(movable_local_field)[:n_cold_teleports]
        cold_tele = movable_idx[cold_order]

    best_score = initial_score
    accepts = 0
    swapped = np.zeros(num_soft, dtype=bool)
    for k1 in hot:
        k1 = int(k1)
        if swapped[k1]:
            continue
        if deadline is not None and time.monotonic() > deadline:
            break
        # Spatial kNN: nearest k_neighbors movable softs (skipping k1 itself).
        # A4: optionally blend distance-to-k1-pos with distance-to-k1-centroid
        # so candidates aligned with k1's WL anchor are ranked higher.
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
        # Speedup #30 (2026-05-30): WL-delta prefilter. The full
        # score_swap_soft costs ~5-10 ms per trial (routing apply + smoothed
        # bbox + cong sum); the WL delta predictor costs ~50 µs. Since proxy
        # = WL + 0.5·den + 0.5·cong, a swap with WL delta above the prefilter
        # threshold can't be a net win unless cong + density drop by ~2×
        # that, which is rare. Threshold loose enough (0.01) to keep every
        # historically-accepted swap (typical accepted WL delta < 0.002).
        WL_PREFILTER = 0.01
        for k2 in nbrs:
            k2 = int(k2)
            if swapped[k2]:
                continue
            # Swap: k1 takes k2's old position, k2 takes k1's old position.
            new_xy1 = (float(soft_pos[k2, 0]), float(soft_pos[k2, 1]))
            new_xy2 = (float(soft_pos[k1, 0]), float(soft_pos[k1, 1]))
            wl_d = incremental_scorer.wl_delta_swap_soft(k1, new_xy1, k2, new_xy2)
            if wl_d > WL_PREFILTER:
                continue
            s = incremental_scorer.score_swap_soft(k1, new_xy1, k2, new_xy2)
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
    return soft_pos, accepts, best_score


