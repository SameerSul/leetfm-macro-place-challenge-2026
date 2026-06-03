"""Local search move operators."""

import time

import numpy as np

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
    """Congestion-directed single-macro RELOCATION pass (2026-05-27).

    H5 (2026-05-29): when `use_density=True`, the hot/cold field switches from
    routing congestion (`max(H,V)`) to grid occupancy
    (`grid_occupied / dens_grid_area`) — the same dual-field pattern that made
    R5 the dominant soft lever. Hot hard macros sitting in the densest cells
    are moved to lowest-density cold cells, accept-on-true-proxy. Same proxy
    gate, same overlap check vs other hard macros, just a different hot/cold
    selection field.

    R6 (2026-05-30): when `use_combined=True`, hotness = geometric mean of
    normalized cong AND density (each field divided by its own max). Macros
    moderately hot on both fields rank above pure-field extremes — caught
    candidates that neither pure pass prioritized.

    WL-aware target selection (R4, 2026-05-28): when `net_centroid` ([n,2], a
    macro's WL anchor) is given with `wl_blend`>0, candidate cold cells are ranked
    by a blend of distance-to-current and distance-to-WL-anchor:
      key = (1-wl_blend)·||cell − cur||² + wl_blend·||cell − centroid||²
    `wl_blend`=0 reproduces the original nearest-to-current behavior exactly.
    Higher blend prefers cold cells toward the macro's connections, so the cong-
    relieving move costs less (or saves) wirelength → more moves pass the gate.

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
    if use_combined:
        # R6 (2026-05-30): macros hot on BOTH cong AND density. Each field is
        # normalized to [0, 1] (divided by its own max), then geometric-meaned
        # so a cell is "combined hot" iff both terms are high. This catches
        # macros sitting in low-grade hot spots on both fields that lose
        # priority in either pure pass (each ranking favours pure-field
        # extremes). Strictly non-regressing — the proxy gate still validates.
        try:
            h_arr = np.asarray(plc.get_horizontal_routing_congestion(), dtype=np.float64)
            v_arr = np.asarray(plc.get_vertical_routing_congestion(), dtype=np.float64)
        except Exception:
            return pos, 0, initial_score
        if h_arr.size != nr * nc or v_arr.size != nr * nc:
            return pos, 0, initial_score
        cong_field = np.maximum(h_arr.reshape(nr, nc), v_arr.reshape(nr, nc))
        go = getattr(incremental_scorer, "grid_occupied", None)
        if go is None or go.size != nr * nc:
            return pos, 0, initial_score
        dens_field = (go / incremental_scorer.dens_grid_area).reshape(nr, nc)
        cong_max = max(float(cong_field.max()), 1e-12)
        dens_max = max(float(dens_field.max()), 1e-12)
        cell_cong = np.sqrt((cong_field / cong_max) * (dens_field / dens_max))
    elif use_density:
        go = getattr(incremental_scorer, "grid_occupied", None)
        if go is None or go.size != nr * nc:
            return pos, 0, initial_score
        cell_cong = (go / incremental_scorer.dens_grid_area).reshape(nr, nc)
    else:
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
    # relocation destinations. Use a percentile-based pool so that medium-cold
    # cells geographically close to each hot macro are included — not just the
    # globally coldest N cells which may all cluster in one corner.
    flat = cell_cong.ravel()
    _thr = np.percentile(flat, 55)  # bottom 55% by congestion ≈ 55% of grid cells
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
        # S1 (2026-05-29): prep i once (subtract old routing + density), trial
        # each candidate via add-new + snapshot/restore (no per-trial subtract-
        # old), commit on the winning target or revert if none. Saves one full
        # routing-apply per trial — ~30% per-move on the relocation hot path.
        prep = incremental_scorer._prepare_move(i)
        best_i_xy = None
        try:
            for t in cand:
                nx, ny = float(tgt_x[t]), float(tgt_y[t])
                if (nx - hw[i] < -EPS or nx + hw[i] > cw + EPS or
                        ny - hh[i] < -EPS or ny + hh[i] > ch + EPS):
                    continue
                # Overlap vs other HARD macros (vectorized).
                if ((np.abs(nx - ox) < sxi + EPS) & (np.abs(ny - oy) < syi + EPS)).any():
                    continue
                s = incremental_scorer._trial_at(prep, (nx, ny))
                if s < best_score - 1e-9:
                    best_score = s
                    best_i_xy = (nx, ny)
            if best_i_xy is not None:
                incremental_scorer._commit_after_prep(prep, best_i_xy)
                pos[i, 0], pos[i, 1] = best_i_xy
                accepts += 1
            else:
                incremental_scorer._revert_prep(prep)
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
    """Congestion-directed SOFT-macro relocation (probe of the R1 idea on softs).

    `use_density=True` (R5 probe, 2026-05-28): select hot softs and cold targets
    by the DENSITY field (occupancy grid) instead of routing congestion. Softs are
    the bulk of the density term (~30% of proxy) and may overlap, so the cong pass
    can pile them into low-cong cells without relieving density; a density-targeted
    pass attacks the top-10% occupancy cells the density cost actually measures.
    Same accept-on-true-proxy machinery; only the hot/cold field changes.

    The leverage analysis showed hard relocation can't help the soft/net-dominated
    benchmarks (ibm17/18). This relocates the hottest SOFT clusters into low-
    congestion cells, accept-on-true-proxy via `score_move_soft`. Softs may
    overlap, so there's NO legality/conflict check — only a half-size clip to keep
    them in canvas bounds. Softs contribute to WL + net-routing congestion +
    density (not macro blockage). `soft_pos` is [num_soft, 2] (canvas coords).
    Returns (soft_pos, accepts, best_score).
    """
    num_soft = incremental_scorer.num_soft
    if num_soft == 0:
        return soft_pos, 0, initial_score
    nr, nc = benchmark.grid_rows, benchmark.grid_cols
    if use_density:
        # Density field = occupancy grid maintained by the scorer (same nr×nc grid).
        go = getattr(incremental_scorer, "grid_occupied", None)
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
    cell_w, cell_h = cw / nc, ch / nr

    ci = np.clip((soft_pos[:, 0] / cell_w).astype(np.int64), 0, nc - 1)
    ri = np.clip((soft_pos[:, 1] / cell_h).astype(np.int64), 0, nr - 1)
    local_cong = cell_field[ri, ci]
    # Only relocate MOVABLE softs — fixed macros must stay put (contract). The
    # IBM benchmarks have 0 fixed softs (no-op here), but NG45/other inputs may.
    order = np.argsort(-local_cong)
    if soft_movable is not None:
        sm = np.asarray(soft_movable, dtype=bool)
        order = order[sm[order]]
    hot = order[:top_hot]

    flat = cell_field.ravel()
    # Pool: use a percentile threshold so medium-cold cells near each hot soft
    # are included — not just globally coldest N cells that may all be in one
    # corner. Each hot soft still tries only n_targets nearest candidates from
    # the pool, so per-soft cost is the same; diversity improves acceptance rate.
    _thr = np.percentile(flat, 55)  # bottom 55% by field value
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
        # Target selection: nearest-first for BOTH cong and density passes.
        # Coldness-first was TRIED 2026-05-30 but regressed density severely
        # (R1: 28 accepts -0.014 vs 75 accepts -0.045 with nearest-first). The
        # reason: coldest cells are scattered across the chip; moving there
        # spikes WL which outweighs density savings → proxy gate rejects most
        # moves. Nearest-first keeps WL stable so the density reduction passes.
        d2 = (tgt_x[cand] - soft_pos[k, 0]) ** 2 + (tgt_y[cand] - soft_pos[k, 1]) ** 2
        # A3 (2026-05-29): blend distance-to-current with distance-to-net-centroid
        # so candidate ordering biases toward k's connections. The proxy gate
        # still validates every move, so this only changes WHICH candidates
        # are tried — strictly non-regressing. wl_blend=0 reproduces the
        # original nearest-to-current ordering exactly.
        if wl_blend > 0.0 and net_centroid is not None:
            d2c = (tgt_x[cand] - net_centroid[k, 0]) ** 2 + (tgt_y[cand] - net_centroid[k, 1]) ** 2
            d2 = (1.0 - wl_blend) * d2 + wl_blend * d2c
        cand = cand[np.argsort(d2)][:n_targets]
        # S1: prep k once, trial each candidate, commit-or-revert. See the hard
        # _relocation_moves comment for the rationale.
        prep = incremental_scorer._prepare_move_soft(k)
        best_k_xy = None
        try:
            for t in cand:
                nx = float(np.clip(tgt_x[t], soft_hw[k], cw - soft_hw[k]))
                ny = float(np.clip(tgt_y[t], soft_hh[k], ch - soft_hh[k]))
                s = incremental_scorer._trial_at_soft(prep, (nx, ny))
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


