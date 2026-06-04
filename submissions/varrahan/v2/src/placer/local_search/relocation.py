"""Local search move operators."""

import time
from typing import TYPE_CHECKING

import numpy as np

from placer.local_search.fields import _congestion_field, _density_field
from placer.ml.data_collection import get_candidate_trace, net_degree_features

if TYPE_CHECKING:
    from macro_place.benchmark import Benchmark

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
    net_centroid: "np.ndarray | None" = None,
    wl_blend: float = 0.0,
    use_density: bool = False,
    use_combined: bool = False,
) -> "tuple[np.ndarray, int, float]":
    """Congestion-directed single-macro RELOCATION of hard macros.

    The move 2-opt can't make: 2-opt only exchanges two macros, so it can never
    move a routing-heavy macro into an empty low-congestion gap. For the hottest
    macros (by the chosen field), this tries moving each into a handful of the
    lowest-field legal cell centers and accepts iff the true incremental proxy
    strictly drops. Legality = in-bounds + no overlap with other HARD macros
    (softs may overlap and are ignored).

    use_density: hot/cold field - False = max(H,V) routing congestion, True =
        grid occupancy. use_combined: geometric mean of normalized cong × density,
        which favours macros moderately hot on both over pure-field extremes.
    net_centroid / wl_blend: blend distance-to-current with distance-to-WL-anchor
        in target ordering (wl_blend=0 = nearest-to-current; ordering only).
    Returns (pos, accepts, best_score).
    """
    nr, nc = benchmark.grid_rows, benchmark.grid_cols
    trace = get_candidate_trace()
    trace_field = "combined" if use_combined else ("density" if use_density else "congestion")
    if use_combined:
        # Combined field: each of cong/density normalized to [0,1] by its own
        # max, then geometric-meaned so a cell ranks hot iff both terms are high.
        cong_field = _congestion_field(plc, nr, nc)
        dens_field = _density_field(incremental_scorer, nr, nc)
        if cong_field is None or dens_field is None:
            return pos, 0, initial_score
        cong_max = max(float(cong_field.max()), 1e-12)
        dens_max = max(float(dens_field.max()), 1e-12)
        cell_cong = np.sqrt((cong_field / cong_max) * (dens_field / dens_max))
    else:
        cell_cong = (_density_field(incremental_scorer, nr, nc) if use_density
                     else _congestion_field(plc, nr, nc))
        if cell_cong is None:
            return pos, 0, initial_score
    trace_cong = trace_dens = None
    trace_cong_max = trace_dens_max = 1.0
    if trace is not None:
        trace_cong = _congestion_field(plc, nr, nc)
        trace_dens = _density_field(incremental_scorer, nr, nc)
        if trace_cong is not None:
            trace_cong_max = max(float(trace_cong.max()), 1e-12)
        if trace_dens is not None:
            trace_dens_max = max(float(trace_dens.max()), 1e-12)
    cell_w, cell_h = cw / nc, ch / nr
    field_max = max(float(cell_cong.max()), 1e-12)

    # Per-macro local congestion → pick the hottest movable macros to relocate.
    ci_all = np.clip((pos[:n, 0] / cell_w).astype(np.int64), 0, nc - 1)
    ri_all = np.clip((pos[:n, 1] / cell_h).astype(np.int64), 0, nr - 1)
    local_cong = cell_cong[ri_all, ci_all]
    mov_idx = np.where(movable)[0]
    if mov_idx.size == 0:
        return pos, 0, initial_score
    hot = mov_idx[np.argsort(-local_cong[mov_idx])][:top_hot]

    # Target pool = low-field cell centers. A percentile threshold (not the
    # globally-coldest N) keeps medium-cold cells near each hot macro in play.
    flat = cell_cong.ravel()
    _thr = np.percentile(flat, 55)
    pool = np.where(flat < _thr)[0]
    if pool.size < max(n_targets, 64):
        pool = np.argsort(flat)[: max(n_targets, 64)]
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
        if wl_blend > 0.0 and net_centroid is not None:
            d2c = (tgt_x[cand] - net_centroid[i, 0]) ** 2 + (tgt_y[cand] - net_centroid[i, 1]) ** 2
            d2 = (1.0 - wl_blend) * d2 + wl_blend * d2c
        cand = cand[np.argsort(d2)][:n_targets]

        mask = all_idx != i
        sxi = sep_x_mat[i, mask]
        syi = sep_y_mat[i, mask]
        ox = pos[mask, 0]
        oy = pos[mask, 1]
        # Prep i once (subtract old routing + density), trial each candidate via
        # add-new + snapshot/restore, then commit the winner or revert. Saves one
        # routing-apply per trial (~30% per-move on this hot path).
        prep = incremental_scorer._prepare_move(i)
        best_i_xy = None
        state_score = best_score
        group_id = trace.next_group_id("hard_relocation") if trace is not None else None
        rejected_bounds = 0
        rejected_overlap = 0
        scored = 0
        try:
            for candidate_rank, t in enumerate(cand):
                nx, ny = float(tgt_x[t]), float(tgt_y[t])
                if (nx - hw[i] < -EPS or nx + hw[i] > cw + EPS or
                        ny - hh[i] < -EPS or ny + hh[i] > ch + EPS):
                    rejected_bounds += 1
                    continue
                # Overlap vs other HARD macros (vectorized).
                if ((np.abs(nx - ox) < sxi + EPS) & (np.abs(ny - oy) < syi + EPS)).any():
                    rejected_overlap += 1
                    continue
                s = incremental_scorer._trial_at(prep, (nx, ny))
                scored += 1
                if trace is not None:
                    target_flat = int(pool[t])
                    source_cong = (
                        float(trace_cong[ri_all[i], ci_all[i]] / trace_cong_max)
                        if trace_cong is not None
                        else 0.0
                    )
                    target_cong = (
                        float(trace_cong.ravel()[target_flat] / trace_cong_max)
                        if trace_cong is not None
                        else 0.0
                    )
                    source_dens = (
                        float(trace_dens[ri_all[i], ci_all[i]] / trace_dens_max)
                        if trace_dens is not None
                        else 0.0
                    )
                    target_dens = (
                        float(trace_dens.ravel()[target_flat] / trace_dens_max)
                        if trace_dens is not None
                        else 0.0
                    )
                    trace.record(
                        operator="hard_relocation",
                        field=trace_field,
                        group_id=group_id,
                        state_score=state_score,
                        trial_score=s,
                        candidate_rank=candidate_rank,
                        group_size=len(cand),
                        candidate_source="cold_cell",
                        features={
                            **net_degree_features(
                                incremental_scorer, incremental_scorer.hard_indices[i]
                            ),
                            "accepted_in_pass": accepts,
                            "macro_w_norm": float(sizes[i, 0] / cw),
                            "macro_h_norm": float(sizes[i, 1] / ch),
                            "x_norm": float(pos[i, 0] / cw),
                            "y_norm": float(pos[i, 1] / ch),
                            "target_x_norm": float(nx / cw),
                            "target_y_norm": float(ny / ch),
                            "dx_norm": float((nx - pos[i, 0]) / cw),
                            "dy_norm": float((ny - pos[i, 1]) / ch),
                            "source_field_norm": float(local_cong[i] / field_max),
                            "target_field_norm": float(tgt_cong[t] / field_max),
                            "source_congestion_norm": source_cong,
                            "target_congestion_norm": target_cong,
                            "source_density_norm": source_dens,
                            "target_density_norm": target_dens,
                            "source_hot_rank_norm": float(
                                np.where(hot == i)[0][0] / max(len(hot) - 1, 1)
                            ),
                            "target_cold_rank_norm": float(candidate_rank / max(len(cand) - 1, 1)),
                        },
                    )
                if s < best_score - 1e-9:
                    best_score = s
                    best_i_xy = (nx, ny)
            if best_i_xy is not None:
                incremental_scorer._commit_after_prep(prep, best_i_xy)
                pos[i, 0], pos[i, 1] = best_i_xy
                accepts += 1
            else:
                incremental_scorer._revert_prep(prep)
            if trace is not None:
                trace.event(
                    "candidate_group_summary",
                    operator="hard_relocation",
                    field=trace_field,
                    group_id=group_id,
                    generated=int(len(cand)),
                    scored=scored,
                    rejected_bounds=rejected_bounds,
                    rejected_overlap=rejected_overlap,
                )
        except Exception:
            # Defensive: restore committed state if anything blew up mid-trial.
            incremental_scorer._revert_prep(prep)
            raise
    return pos, accepts, best_score


def _soft_relocation_moves(
    soft_pos: np.ndarray,
    soft_hw: np.ndarray,
    soft_hh: np.ndarray,
    cw: float,
    ch: float,
    n: int,
    plc,
    benchmark: "Benchmark",
    incremental_scorer,
    initial_score: float,
    deadline: "float | None" = None,
    top_hot: int = 48,
    n_targets: int = 16,
    soft_movable: "np.ndarray | None" = None,
    use_density: bool = False,
    net_centroid: "np.ndarray | None" = None,
    wl_blend: float = 0.0,
) -> "tuple[np.ndarray, int, float]":
    """Congestion-directed SOFT-macro relocation.

    Relocates the hottest movable soft clusters into low-field cells, accept-on-
    true-proxy via the soft prep/trial path. Softs may overlap, so there's no
    conflict check - only a half-size clip to keep them in canvas bounds.

    use_density: hot/cold field - False = max(H,V) routing congestion, True =
        grid occupancy. Softs are the bulk of the density term and may overlap,
        so a density-targeted pass finds moves the cong pass can't.
    net_centroid / wl_blend: blend distance-to-current with distance-to-WL-anchor
        in target ordering (wl_blend=0 = nearest-to-current; ordering only).
    `soft_pos` is [num_soft, 2] (canvas coords). Returns (soft_pos, accepts,
    best_score).
    """
    num_soft = incremental_scorer.num_soft
    if num_soft == 0:
        return soft_pos, 0, initial_score
    nr, nc = benchmark.grid_rows, benchmark.grid_cols
    trace = get_candidate_trace()
    trace_field = "density" if use_density else "congestion"
    cell_field = (_density_field(incremental_scorer, nr, nc) if use_density
                  else _congestion_field(plc, nr, nc))
    if cell_field is None:
        return soft_pos, 0, initial_score
    trace_cong = trace_dens = None
    trace_cong_max = trace_dens_max = 1.0
    if trace is not None:
        trace_cong = _congestion_field(plc, nr, nc)
        trace_dens = _density_field(incremental_scorer, nr, nc)
        if trace_cong is not None:
            trace_cong_max = max(float(trace_cong.max()), 1e-12)
        if trace_dens is not None:
            trace_dens_max = max(float(trace_dens.max()), 1e-12)
    cell_w, cell_h = cw / nc, ch / nr
    field_max = max(float(cell_field.max()), 1e-12)

    ci = np.clip((soft_pos[:, 0] / cell_w).astype(np.int64), 0, nc - 1)
    ri = np.clip((soft_pos[:, 1] / cell_h).astype(np.int64), 0, nr - 1)
    local_cong = cell_field[ri, ci]
    # Only relocate MOVABLE softs - fixed macros must stay put (contract). The
    # IBM benchmarks have 0 fixed softs (no-op here), but NG45/other inputs may.
    order = np.argsort(-local_cong)
    if soft_movable is not None:
        sm = np.asarray(soft_movable, dtype=bool)
        order = order[sm[order]]
    hot = order[:top_hot]

    flat = cell_field.ravel()
    # Target pool = low-field cell centers (percentile threshold, not globally-
    # coldest N, so medium-cold cells near each hot soft stay in play).
    _thr = np.percentile(flat, 55)
    pool = np.where(flat < _thr)[0]
    if pool.size < max(n_targets, 64):
        pool = np.argsort(flat)[: max(n_targets, 64)]
    tgt_x = ((pool % nc).astype(np.float64) + 0.5) * cell_w
    tgt_y = ((pool // nc).astype(np.float64) + 0.5) * cell_h
    tgt_cong = flat[pool]

    best_score = initial_score
    accepts = 0
    for k in hot:
        k = int(k)
        if deadline is not None and time.monotonic() > deadline:
            break
        cand = np.where(tgt_cong < local_cong[k] - 1e-9)[0]
        if cand.size == 0:
            continue
        # Order targets by distance, optionally blended toward the WL anchor.
        d2 = (tgt_x[cand] - soft_pos[k, 0]) ** 2 + (tgt_y[cand] - soft_pos[k, 1]) ** 2
        if wl_blend > 0.0 and net_centroid is not None:
            d2c = (tgt_x[cand] - net_centroid[k, 0]) ** 2 + (tgt_y[cand] - net_centroid[k, 1]) ** 2
            d2 = (1.0 - wl_blend) * d2 + wl_blend * d2c
        cand = cand[np.argsort(d2)][:n_targets]
        # Prep k once, trial each candidate, commit-or-revert (see _relocation_moves).
        prep = incremental_scorer._prepare_move_soft(k)
        best_k_xy = None
        state_score = best_score
        group_id = trace.next_group_id("soft_relocation") if trace is not None else None
        try:
            for candidate_rank, t in enumerate(cand):
                nx = float(np.clip(tgt_x[t], soft_hw[k], cw - soft_hw[k]))
                ny = float(np.clip(tgt_y[t], soft_hh[k], ch - soft_hh[k]))
                s = incremental_scorer._trial_at_soft(prep, (nx, ny))
                if trace is not None:
                    target_flat = int(pool[t])
                    trace.record(
                        operator="soft_relocation",
                        field=trace_field,
                        group_id=group_id,
                        state_score=state_score,
                        trial_score=s,
                        candidate_rank=candidate_rank,
                        group_size=len(cand),
                        candidate_source="cold_cell",
                        features={
                            **net_degree_features(
                                incremental_scorer, incremental_scorer.soft_indices[k]
                            ),
                            "accepted_in_pass": accepts,
                            "macro_w_norm": float(2.0 * soft_hw[k] / cw),
                            "macro_h_norm": float(2.0 * soft_hh[k] / ch),
                            "x_norm": float(soft_pos[k, 0] / cw),
                            "y_norm": float(soft_pos[k, 1] / ch),
                            "target_x_norm": float(nx / cw),
                            "target_y_norm": float(ny / ch),
                            "dx_norm": float((nx - soft_pos[k, 0]) / cw),
                            "dy_norm": float((ny - soft_pos[k, 1]) / ch),
                            "source_field_norm": float(local_cong[k] / field_max),
                            "target_field_norm": float(tgt_cong[t] / field_max),
                            "source_congestion_norm": (
                                float(trace_cong[ri[k], ci[k]] / trace_cong_max)
                                if trace_cong is not None
                                else 0.0
                            ),
                            "target_congestion_norm": (
                                float(trace_cong.ravel()[target_flat] / trace_cong_max)
                                if trace_cong is not None
                                else 0.0
                            ),
                            "source_density_norm": (
                                float(trace_dens[ri[k], ci[k]] / trace_dens_max)
                                if trace_dens is not None
                                else 0.0
                            ),
                            "target_density_norm": (
                                float(trace_dens.ravel()[target_flat] / trace_dens_max)
                                if trace_dens is not None
                                else 0.0
                            ),
                            "source_hot_rank_norm": float(
                                np.where(hot == k)[0][0] / max(len(hot) - 1, 1)
                            ),
                            "target_cold_rank_norm": float(candidate_rank / max(len(cand) - 1, 1)),
                        },
                    )
                if s < best_score - 1e-9:
                    best_score = s
                    best_k_xy = (nx, ny)
            if best_k_xy is not None:
                incremental_scorer._commit_after_prep_soft(prep, best_k_xy)
                soft_pos[k, 0], soft_pos[k, 1] = best_k_xy
                accepts += 1
            else:
                incremental_scorer._revert_prep_soft(prep)
        except Exception:
            incremental_scorer._revert_prep_soft(prep)
            raise
    return soft_pos, accepts, best_score
