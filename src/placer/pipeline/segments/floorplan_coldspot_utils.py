"""Coldspot geometry and local-region helper utilities."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


def coldspot_window_stats(
    field: np.ndarray,
    win_cells: int,
    nr: int,
    nc: int,
    cw: float,
    ch: float,
    cache: dict[tuple[int, int], tuple[float, float, float]],
) -> tuple[float, float, float]:
    """Compute the minimum-window congestion summary for a given cell-window size."""
    field_key = id(field)
    w = int(max(1, min(win_cells, nr, nc)))
    key = (field_key, w)
    cached = cache.get(key)
    if cached is not None:
        return cached
    rows, cols = nr - w + 1, nc - w + 1
    if rows <= 0 or cols <= 0:
        r, c = divmod(int(np.argmin(field)), nc)
        value = float(field[r, c])
        out = (value, (c + 0.5) * (cw / nc), (r + 0.5) * (ch / nr))
        cache[key] = out
        return out
    integ = np.zeros((nr + 1, nc + 1), dtype=np.float64)
    integ[1:, 1:] = np.cumsum(np.cumsum(field, axis=0), axis=1)
    win_sum = (
        integ[w : w + rows, w : w + cols]
        - integ[0:rows, w : w + cols]
        - integ[w : w + rows, 0:cols]
        + integ[0:rows, 0:cols]
    )
    flat_min = int(np.argmin(win_sum))
    rr, cc = divmod(flat_min, cols)
    avg = float(win_sum[rr, cc] / float(w * w))
    out = (avg, (cc + 0.5 * w) * (cw / nc), (rr + 0.5 * w) * (ch / nr))
    cache[key] = out
    return out


def coldspot_min_window_avg(
    field: np.ndarray,
    win_cells: int,
    nr: int,
    nc: int,
    cw: float,
    ch: float,
    cache: dict[tuple[int, int], tuple[float, float, float]],
) -> float:
    """Return minimum congestion window average from cached integral stats."""
    return float(coldspot_window_stats(field, win_cells, nr, nc, cw, ch, cache)[0])


def coldspot_field_gap(
    field: np.ndarray,
    hard_xy: np.ndarray,
    sizes: np.ndarray,
    movable: np.ndarray,
    clusters: dict,
    n: int,
    nr: int,
    nc: int,
    cw: float,
    ch: float,
    min_window_avg: Callable[[np.ndarray, int], float],
) -> float:
    """Estimate the best macro-cluster field relief gap for the current layout."""
    cell_w, cell_h = cw / nc, ch / nr
    mcol = np.clip((hard_xy[:, 0] / cell_w).astype(np.int64), 0, nc - 1)
    mrow = np.clip((hard_xy[:, 1] / cell_h).astype(np.int64), 0, nr - 1)
    macro_cong = field[mrow, mcol]
    best_gap = -np.inf
    for members in clusters.values():
        members = members[movable[:n][members]]
        if members.size < 2 or members.size > 64:
            continue
        member_area = float(np.sum(sizes[members, 0] * sizes[members, 1]))
        win_microns = float(np.sqrt(member_area / 0.65))
        win_cells = max(1, int(np.ceil(win_microns / min(cell_w, cell_h))))
        gap = float(np.mean(macro_cong[members])) - min_window_avg(field, win_cells)
        if gap > best_gap:
            best_gap = gap
    return best_gap


def coldspot_opportunity(
    field: np.ndarray,
    hard_xy: np.ndarray,
    soft_xy: np.ndarray,
    clusters: dict,
    movable: np.ndarray,
    n: int,
    sizes: np.ndarray,
    nr: int,
    nc: int,
    cw: float,
    ch: float,
    const: Any,
    occupied_cells: Callable[[np.ndarray, np.ndarray], np.ndarray],
    window_stats: Callable[[np.ndarray, int], tuple[float, float, float]],
    ck_opportunity_min_cold_cells: int,
    ck_min_field_gap: float,
    ck_opportunity_min_score: float,
    ck_opportunity_top_clusters: int,
) -> dict[str, float | int | bool]:
    """Rank cold-cluster relocation opportunities from a congestion field."""
    cell_w, cell_h = cw / nc, ch / nr
    mcol = np.clip((hard_xy[:, 0] / cell_w).astype(np.int64), 0, nc - 1)
    mrow = np.clip((hard_xy[:, 1] / cell_h).astype(np.int64), 0, nr - 1)
    macro_cong = field[mrow, mcol]
    cold_pct = float(const.HIER_COLDSPOT_MEMORY_COLD_PCT)
    cold_mask = field <= float(np.percentile(field, np.clip(cold_pct, 0.0, 100.0)))
    open_cold = cold_mask & ~occupied_cells(hard_xy, soft_xy)

    best: dict[str, float | int | bool] = {
        "run": False,
        "eligible_clusters": 0,
        "score": -1.0e30,
        "field_gap": -1.0e30,
        "open_cold_cells": int(np.count_nonzero(open_cold)),
        "cluster": -1,
        "source_field": 0.0,
        "target_field": 0.0,
        "displacement_windows": 0.0,
        "cluster_ids": [],
    }
    rows = []
    for cid, raw_members in clusters.items():
        members = np.asarray(raw_members, dtype=np.int64)
        members = members[(members >= 0) & (members < n)]
        members = members[movable[:n][members]]
        if members.size < 2 or members.size > 64:
            continue
        best["eligible_clusters"] = int(best["eligible_clusters"]) + 1
        member_area = float(np.sum(sizes[members, 0] * sizes[members, 1]))
        win_microns = float(np.sqrt(member_area / 0.65))
        win_cells = max(1, int(np.ceil(win_microns / min(cell_w, cell_h))))
        source = float(np.mean(macro_cong[members]))
        target, ax, ay = window_stats(field, win_cells)
        gap = source - target
        cm = hard_xy[members].mean(axis=0)
        disp = float(np.hypot(cm[0] - ax, cm[1] - ay) / max(win_microns, 1.0))

        ar = int(np.clip(ay / cell_h, 0, nr - 1))
        ac = int(np.clip(ax / cell_w, 0, nc - 1))
        radius = max(1, min(3, win_cells))
        r0, r1 = max(0, ar - radius), min(nr, ar + radius + 1)
        c0, c1 = max(0, ac - radius), min(nc, ac + radius + 1)
        local_open = int(np.count_nonzero(open_cold[r0:r1, c0:c1]))

        score = float(gap + 0.003 * np.log1p(local_open) - 0.002 * disp)
        row = {
            "score": float(score),
            "field_gap": float(gap),
            "cluster": int(cid),
            "source_field": float(source),
            "target_field": float(target),
            "open_cold_cells": int(local_open),
            "displacement_windows": float(disp),
        }
        rows.append(row)
        if score > float(best["score"]):
            best.update(row)

    rows.sort(
        key=lambda row: (
            -float(row["score"]),
            -float(row["field_gap"]),
            -int(row["open_cold_cells"]),
            int(row["cluster"]),
        )
    )
    usable = [
        row
        for row in rows
        if float(row["field_gap"]) >= ck_min_field_gap
        and int(row["open_cold_cells"]) >= ck_opportunity_min_cold_cells
        and float(row["score"]) >= ck_opportunity_min_score
    ]
    best["cluster_ids"] = [int(row["cluster"]) for row in usable[:ck_opportunity_top_clusters]]
    best["run"] = bool(
        int(best["eligible_clusters"]) > 0
        and float(best["field_gap"]) >= ck_min_field_gap
        and int(best["open_cold_cells"]) >= ck_opportunity_min_cold_cells
        and float(best["score"]) >= ck_opportunity_min_score
        and len(best["cluster_ids"]) > 0
    )
    return best


def remember_cold_cells(field: np.ndarray, const: Any) -> np.ndarray:
    """Build cold-cell mask from a field percentile gate."""
    cold_pct = float(const.HIER_COLDSPOT_MEMORY_COLD_PCT)
    thresh = float(np.percentile(field, np.clip(cold_pct, 0.0, 100.0)))
    return np.asarray(field <= thresh, dtype=bool)


def occupied_cells(
    hard_xy: np.ndarray,
    soft_xy: np.ndarray,
    hw: np.ndarray,
    hh: np.ndarray,
    soft_hw: np.ndarray,
    soft_hh: np.ndarray,
    nr: int,
    nc: int,
    cw: float,
    ch: float,
) -> np.ndarray:
    """Mark congestion cells occupied by current macro + soft placements."""
    occupied = np.zeros((nr, nc), dtype=bool)
    cell_w, cell_h = cw / nc, ch / nr

    def _mark(pos, half_w, half_h):
        for x, y, mx, my in zip(pos[:, 0], pos[:, 1], half_w, half_h):
            c0 = int(np.floor((float(x) - float(mx)) / cell_w))
            c1 = int(np.floor((float(x) + float(mx)) / cell_w))
            r0 = int(np.floor((float(y) - float(my)) / cell_h))
            r1 = int(np.floor((float(y) + float(my)) / cell_h))
            c0 = max(0, min(nc - 1, c0))
            c1 = max(0, min(nc - 1, c1))
            r0 = max(0, min(nr - 1, r0))
            r1 = max(0, min(nr - 1, r1))
            occupied[r0 : r1 + 1, c0 : c1 + 1] = True

    if hard_xy.size:
        _mark(hard_xy, hw, hh)
    if soft_xy.size:
        _mark(soft_xy, soft_hw, soft_hh)
    return occupied


def bbox_cell_mask(xlo: float, ylo: float, xhi: float, yhi: float, nr: int, nc: int, cw: float, ch: float) -> np.ndarray:
    mask = np.zeros((nr, nc), dtype=bool)
    cell_w, cell_h = cw / nc, ch / nr
    c0 = int(np.floor(xlo / cell_w))
    c1 = int(np.floor(xhi / cell_w))
    r0 = int(np.floor(ylo / cell_h))
    r1 = int(np.floor(yhi / cell_h))
    c0 = max(0, min(nc - 1, c0))
    c1 = max(0, min(nc - 1, c1))
    r0 = max(0, min(nr - 1, r0))
    r1 = max(0, min(nr - 1, r1))
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def dilate_cell_mask(mask: np.ndarray, radius: int, nr: int, nc: int) -> np.ndarray:
    radius = max(0, int(radius))
    out = np.asarray(mask, dtype=bool).copy()
    if radius == 0 or not out.any():
        return out
    rows, cols = np.where(out)
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            rr = rows + dr
            cc = cols + dc
            valid = (rr >= 0) & (rr < nr) & (cc >= 0) & (cc < nc)
            out[rr[valid], cc[valid]] = True
    return out


def expand_bbox_to_adjacent_cold(
    xlo: float,
    ylo: float,
    xhi: float,
    yhi: float,
    hard_xy: np.ndarray,
    soft_xy: np.ndarray,
    cold_memory: np.ndarray,
    hw: np.ndarray,
    hh: np.ndarray,
    soft_hw: np.ndarray,
    soft_hh: np.ndarray,
    nr: int,
    nc: int,
    cw: float,
    ch: float,
    adaptive_max_cells: int,
) -> "tuple[float, float, float, float, int, np.ndarray, np.ndarray]":
    """Grow an initial bbox by traversing adjacent cold cells around its edge."""
    seed_mask = bbox_cell_mask(xlo, ylo, xhi, yhi, nr, nc, cw, ch)
    empty = np.zeros((nr, nc), dtype=bool)
    if not cold_memory.any():
        return xlo, ylo, xhi, yhi, 0, seed_mask, empty

    occupied = occupied_cells(hard_xy, soft_xy, hw, hh, soft_hw, soft_hh, nr, nc, cw, ch)
    open_cold = cold_memory & ~occupied
    if not open_cold.any():
        return xlo, ylo, xhi, yhi, 0, seed_mask, empty

    cell_w, cell_h = cw / nc, ch / nr
    c0 = int(np.floor(xlo / cell_w))
    c1 = int(np.floor(xhi / cell_w))
    r0 = int(np.floor(ylo / cell_h))
    r1 = int(np.floor(yhi / cell_h))
    c0 = max(0, min(nc - 1, c0))
    c1 = max(0, min(nc - 1, c1))
    r0 = max(0, min(nr - 1, r0))
    r1 = max(0, min(nr - 1, r1))

    seen = np.zeros((nr, nc), dtype=bool)
    queue: list[tuple[int, int, int]] = []
    for rr in range(max(0, r0 - 1), min(nr, r1 + 2)):
        for cc in range(max(0, c0 - 1), min(nc, c1 + 2)):
            adjacent = rr < r0 or rr > r1 or cc < c0 or cc > c1
            if adjacent and open_cold[rr, cc]:
                seen[rr, cc] = True
                queue.append((rr, cc, 0))

    reached: list[tuple[int, int]] = []
    head = 0
    while head < len(queue):
        rr, cc, dist = queue[head]
        head += 1
        reached.append((rr, cc))
        if dist >= adaptive_max_cells:
            continue
        for nr2, nc2 in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
            if nr2 < 0 or nr2 >= nr or nc2 < 0 or nc2 >= nc or seen[nr2, nc2]:
                continue
            if not open_cold[nr2, nc2]:
                continue
            seen[nr2, nc2] = True
            queue.append((nr2, nc2, dist + 1))

    if not reached:
        return xlo, ylo, xhi, yhi, 0, seed_mask, empty
    rows = np.array([p[0] for p in reached], dtype=np.int64)
    cols = np.array([p[1] for p in reached], dtype=np.int64)
    reached_mask = np.zeros((nr, nc), dtype=bool)
    reached_mask[rows, cols] = True
    xlo = min(xlo, float(cols.min()) * cell_w)
    ylo = min(ylo, float(rows.min()) * cell_h)
    xhi = max(xhi, float(cols.max() + 1) * cell_w)
    yhi = max(yhi, float(rows.max() + 1) * cell_h)
    graph_mask = seed_mask | reached_mask
    return (
        max(0.0, xlo),
        max(0.0, ylo),
        min(cw, xhi),
        min(ch, yhi),
        len(reached),
        graph_mask,
        open_cold,
    )


def coldspot_local_regions(
    hard_xy: np.ndarray,
    soft_xy: np.ndarray,
    cid: int,
    clusters: dict,
    csofts: dict,
    bridge_softs: dict,
    movable: np.ndarray,
    n: int,
    n_soft: int,
    soft_mov: np.ndarray,
    hw: np.ndarray,
    hh: np.ndarray,
    soft_hw: np.ndarray,
    soft_hh: np.ndarray,
    cw: float,
    ch: float,
    nr: int,
    nc: int,
    cold_memory: np.ndarray,
    const: Any,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict, np.ndarray, np.ndarray] | None":
    """Build local refinement regions and candidate membership masks for a cluster."""
    members = np.asarray(clusters.get(int(cid), []), dtype=np.int64)
    members = members[(members >= 0) & (members < n)]
    members = members[movable[:n][members]]
    if members.size == 0:
        return None

    owned_soft = np.asarray(csofts.get(int(cid), []), dtype=np.int64) - n
    owned_soft = owned_soft[(owned_soft >= 0) & (owned_soft < n_soft)]
    bridge_local = [
        int(k)
        for k, cids_for_soft in bridge_softs.items()
        if int(cid) in {int(v) for v in np.asarray(cids_for_soft, dtype=np.int64)}
    ]
    soft_seed = np.unique(
        np.concatenate([owned_soft, np.asarray(bridge_local, dtype=np.int64)])
        if bridge_local
        else owned_soft
    )
    if soft_seed.size:
        soft_seed = soft_seed[soft_mov[soft_seed]]

    hard_xlo = float(np.min(hard_xy[members, 0] - hw[members]))
    hard_ylo = float(np.min(hard_xy[members, 1] - hh[members]))
    hard_xhi = float(np.max(hard_xy[members, 0] + hw[members]))
    hard_yhi = float(np.max(hard_xy[members, 1] + hh[members]))
    xlo, ylo, xhi, yhi = hard_xlo, hard_ylo, hard_xhi, hard_yhi
    if soft_seed.size:
        xlo = min(xlo, float(np.min(soft_xy[soft_seed, 0] - soft_hw[soft_seed])))
        ylo = min(ylo, float(np.min(soft_xy[soft_seed, 1] - soft_hh[soft_seed])))
        xhi = max(xhi, float(np.max(soft_xy[soft_seed, 0] + soft_hw[soft_seed])))
        yhi = max(yhi, float(np.max(soft_xy[soft_seed, 1] + soft_hh[soft_seed])))

    (
        xlo,
        ylo,
        xhi,
        yhi,
        adaptive_cold_cells,
        graph_mask,
        open_cold_mask,
    ) = expand_bbox_to_adjacent_cold(
        xlo,
        ylo,
        xhi,
        yhi,
        hard_xy,
        soft_xy,
        cold_memory,
        hw,
        hh,
        soft_hw,
        soft_hh,
        nr,
        nc,
        cw,
        ch,
        int(const.HIER_COLDSPOT_ADAPTIVE_MAX_CELLS),
    )

    cell_w, cell_h = cw / nc, ch / nr
    hard_core_span = max(hard_xhi - hard_xlo, hard_yhi - hard_ylo)
    min_pad = max(cell_w, cell_h) * max(0.0, float(const.HIER_COLDSPOT_LOCAL_MIN_PAD_CELLS))
    max_pad = max(cw, ch) * max(0.0, float(const.HIER_COLDSPOT_LOCAL_MAX_PAD_FRAC))
    pad = max(
        min_pad,
        hard_core_span * max(0.0, float(const.HIER_COLDSPOT_LOCAL_HARD_PAD_FRAC)),
    )
    if max_pad > 0.0:
        pad = min(pad, max_pad)
    pad_cells = int(np.ceil(pad / max(min(cell_w, cell_h), 1e-9)))
    region_mask = dilate_cell_mask(graph_mask, pad_cells, nr, nc)
    xlo = max(0.0, xlo - pad)
    ylo = max(0.0, ylo - pad)
    xhi = min(cw, xhi + pad)
    yhi = min(ch, yhi + pad)
    target_mask = region_mask & ~occupied_cells(
        hard_xy,
        soft_xy,
        hw,
        hh,
        soft_hw,
        soft_hh,
        nr,
        nc,
        cw,
        ch,
    )
    if open_cold_mask.any():
        target_mask = target_mask & (open_cold_mask | graph_mask)
    target_pool = np.flatnonzero(target_mask.ravel()).astype(np.int64)

    if n_soft:
        inside_soft = np.where(
            soft_mov
            & (soft_xy[:, 0] >= xlo)
            & (soft_xy[:, 0] <= xhi)
            & (soft_xy[:, 1] >= ylo)
            & (soft_xy[:, 1] <= yhi)
        )[0]
        local_soft = np.unique(np.concatenate([soft_seed, inside_soft]))
    else:
        local_soft = np.zeros(0, dtype=np.int64)

    hard_region = np.column_stack([hw, hh, cw - hw, ch - hh]).astype(np.float64)
    for i in members:
        hard_region[i] = (
            max(hw[i], xlo + hw[i]),
            max(hh[i], ylo + hh[i]),
            min(cw - hw[i], xhi - hw[i]),
            min(ch - hh[i], yhi - hh[i]),
        )
        if hard_region[i, 0] > hard_region[i, 2]:
            hard_region[i, 0] = hard_region[i, 2] = float(hard_xy[i, 0])
        if hard_region[i, 1] > hard_region[i, 3]:
            hard_region[i, 1] = hard_region[i, 3] = float(hard_xy[i, 1])

    soft_region = np.column_stack([soft_hw, soft_hh, cw - soft_hw, ch - soft_hh]).astype(np.float64)
    for k in local_soft:
        soft_region[k] = (
            max(soft_hw[k], xlo + soft_hw[k]),
            max(soft_hh[k], ylo + soft_hh[k]),
            min(cw - soft_hw[k], xhi - soft_hw[k]),
            min(ch - soft_hh[k], yhi - soft_hh[k]),
        )
        if soft_region[k, 0] > soft_region[k, 2]:
            soft_region[k, 0] = soft_region[k, 2] = float(soft_xy[k, 0])
        if soft_region[k, 1] > soft_region[k, 3]:
            soft_region[k, 1] = soft_region[k, 3] = float(soft_xy[k, 1])

    hard_mask = np.zeros(n, dtype=bool)
    hard_mask[members] = True
    soft_mask = np.zeros(n_soft, dtype=bool)
    if local_soft.size:
        soft_mask[local_soft] = True
    stats = {
        "local_region_pad": float(pad),
        "local_region_pad_cells": int(pad_cells),
        "local_region_hard_core_span": float(hard_core_span),
        "adaptive_cold_cells": int(adaptive_cold_cells),
        "graph_region_cells": int(np.count_nonzero(region_mask)),
        "graph_target_cells": int(target_pool.size),
        "adaptive_region_xlo": float(xlo),
        "adaptive_region_ylo": float(ylo),
        "adaptive_region_xhi": float(xhi),
        "adaptive_region_yhi": float(yhi),
    }
    return (
        hard_region,
        soft_region,
        hard_mask,
        soft_mask,
        stats,
        target_pool,
        region_mask,
    )
