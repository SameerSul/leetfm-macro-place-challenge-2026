"""Congestion-aware expansion of hierarchy region boxes."""

from __future__ import annotations

import numpy as np


def _avg_side(field, r0, r1, c0, c1, side: str, band: int) -> float:
    nr, nc = field.shape
    if side == "left":
        lo, hi = max(0, c0 - band), max(0, c0)
        vals = field[max(0, r0) : min(nr, r1 + 1), lo:hi]
    elif side == "right":
        lo, hi = min(nc, c1 + 1), min(nc, c1 + 1 + band)
        vals = field[max(0, r0) : min(nr, r1 + 1), lo:hi]
    elif side == "down":
        lo, hi = max(0, r0 - band), max(0, r0)
        vals = field[lo:hi, max(0, c0) : min(nc, c1 + 1)]
    else:
        lo, hi = min(nr, r1 + 1), min(nr, r1 + 1 + band)
        vals = field[lo:hi, max(0, c0) : min(nc, c1 + 1)]
    if vals.size == 0:
        return float("inf")
    return float(vals.mean())


def _expand_one_region(region, idx, left, right, down, up, hw, hh, cw, ch):
    region[idx, 0] = np.maximum(hw[idx], region[idx, 0] - left)
    region[idx, 2] = np.minimum(cw - hw[idx], region[idx, 2] + right)
    region[idx, 1] = np.maximum(hh[idx], region[idx, 1] - down)
    region[idx, 3] = np.minimum(ch - hh[idx], region[idx, 3] + up)


def expand_regions_by_congestion(
    hard_region,
    soft_region,
    hard_xy,
    soft_xy,
    clusters,
    cluster_softs,
    bridge_softs,
    hw,
    hh,
    soft_hw,
    soft_hh,
    cw,
    ch,
    field,
    hot_percentile: float = 60.0,
    max_expand_frac: float = 0.08,
    side_band: int = 3,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Expand hot cluster regions toward colder neighboring grid bands."""
    if field is None or not clusters:
        return hard_region, soft_region, 0
    hard_region = hard_region.copy()
    soft_region = soft_region.copy()
    nr, nc = field.shape
    cell_w, cell_h = float(cw) / nc, float(ch) / nr
    heat = []
    for cid, mem in clusters.items():
        if len(mem) == 0:
            continue
        ci = np.clip((hard_xy[mem, 0] / cell_w).astype(np.int64), 0, nc - 1)
        ri = np.clip((hard_xy[mem, 1] / cell_h).astype(np.int64), 0, nr - 1)
        heat.append((int(cid), float(field[ri, ci].mean())))
    if not heat:
        return hard_region, soft_region, 0
    threshold = float(np.percentile([h for _, h in heat], hot_percentile))
    max_dx = max_expand_frac * float(cw)
    max_dy = max_expand_frac * float(ch)
    expanded = 0

    bridge_by_cluster: "dict[int, list[int]]" = {}
    for s, cids in (bridge_softs or {}).items():
        for cid in cids:
            bridge_by_cluster.setdefault(int(cid), []).append(int(s))

    for cid, h in heat:
        if h < threshold:
            continue
        mem = np.asarray(clusters[cid], dtype=np.int64)
        x0, x1 = float(hard_xy[mem, 0].min()), float(hard_xy[mem, 0].max())
        y0, y1 = float(hard_xy[mem, 1].min()), float(hard_xy[mem, 1].max())
        c0 = int(np.clip(np.floor(x0 / cell_w), 0, nc - 1))
        c1 = int(np.clip(np.floor(x1 / cell_w), 0, nc - 1))
        r0 = int(np.clip(np.floor(y0 / cell_h), 0, nr - 1))
        r1 = int(np.clip(np.floor(y1 / cell_h), 0, nr - 1))
        side_vals = {
            "left": _avg_side(field, r0, r1, c0, c1, "left", side_band),
            "right": _avg_side(field, r0, r1, c0, c1, "right", side_band),
            "down": _avg_side(field, r0, r1, c0, c1, "down", side_band),
            "up": _avg_side(field, r0, r1, c0, c1, "up", side_band),
        }
        finite = [v for v in side_vals.values() if np.isfinite(v)]
        if not finite:
            continue
        cold = min(finite)
        left = max_dx if side_vals["left"] <= cold + 1e-12 else 0.35 * max_dx
        right = max_dx if side_vals["right"] <= cold + 1e-12 else 0.35 * max_dx
        down = max_dy if side_vals["down"] <= cold + 1e-12 else 0.35 * max_dy
        up = max_dy if side_vals["up"] <= cold + 1e-12 else 0.35 * max_dy

        _expand_one_region(hard_region, mem, left, right, down, up, hw, hh, cw, ch)
        soft_pidx = list(np.asarray(cluster_softs.get(cid, []), dtype=np.int64))
        soft_pidx += [int(s) for s in bridge_by_cluster.get(cid, [])]
        if soft_pidx:
            sidx = np.array(
                [
                    int(p) - hard_xy.shape[0] if int(p) >= hard_xy.shape[0] else int(p)
                    for p in soft_pidx
                ]
            )
            sidx = sidx[(sidx >= 0) & (sidx < soft_xy.shape[0])]
            if sidx.size:
                _expand_one_region(
                    soft_region, sidx, left, right, down, up, soft_hw, soft_hh, cw, ch
                )
        expanded += 1
    return hard_region, soft_region, expanded
