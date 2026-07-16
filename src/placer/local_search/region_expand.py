"""Congestion-aware expansion of hierarchy region boxes."""

from __future__ import annotations

import numpy as np

from placer.local_search.fields import cold_connected_components


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


def _component_expansion(
    components,
    r0: int,
    r1: int,
    c0: int,
    c1: int,
    *,
    cell_w: float,
    cell_h: float,
    max_dx: float,
    max_dy: float,
    side_floor: float,
    side_band: int,
    max_distance_cells: int,
    graph_corridors=None,
    graph_component_weight: float = 0.0,
):
    """Return side expansion toward the best nearby cold component."""
    best = None
    max_dist = max(0, int(max_distance_cells))
    band = max(0, int(side_band))
    graph_component_weight = max(0.0, float(graph_component_weight))

    def _corridor_penalty(comp) -> float:
        if graph_component_weight <= 0.0 or not graph_corridors:
            return 0.0
        point = np.array([float(comp["centroid_c"]), float(comp["centroid_r"])], dtype=np.float64)
        best_penalty = None
        for a, b, weight in graph_corridors:
            a = np.asarray(a, dtype=np.float64)
            b = np.asarray(b, dtype=np.float64)
            vec = b - a
            denom = float(np.dot(vec, vec))
            if denom <= 1e-9:
                dist = float(np.linalg.norm(point - a))
            else:
                t = float(np.clip(np.dot(point - a, vec) / denom, 0.0, 1.0))
                dist = float(np.linalg.norm(point - (a + t * vec)))
            penalty = dist / max(1e-6, float(weight))
            if best_penalty is None or penalty < best_penalty:
                best_penalty = penalty
        return 0.0 if best_penalty is None else float(best_penalty)

    for comp in components:
        cr0 = int(comp["r0"])
        cr1 = int(comp["r1"])
        cc0 = int(comp["c0"])
        cc1 = int(comp["c1"])
        row_overlap = cr1 >= r0 - band and cr0 <= r1 + band
        col_overlap = cc1 >= c0 - band and cc0 <= c1 + band
        sides = []
        if row_overlap and cc1 < c0:
            dist = c0 - cc1 - 1
            if dist <= max_dist:
                left = min(float(max_dx), max(0.0, (c0 - cc0 + 1) * float(cell_w)))
                sides.append((dist, "left", left))
        if row_overlap and cc0 > c1:
            dist = cc0 - c1 - 1
            if dist <= max_dist:
                right = min(float(max_dx), max(0.0, (cc1 - c1 + 1) * float(cell_w)))
                sides.append((dist, "right", right))
        if col_overlap and cr1 < r0:
            dist = r0 - cr1 - 1
            if dist <= max_dist:
                down = min(float(max_dy), max(0.0, (r0 - cr0 + 1) * float(cell_h)))
                sides.append((dist, "down", down))
        if col_overlap and cr0 > r1:
            dist = cr0 - r1 - 1
            if dist <= max_dist:
                up = min(float(max_dy), max(0.0, (cr1 - r1 + 1) * float(cell_h)))
                sides.append((dist, "up", up))
        if not sides:
            continue
        dist = min(row[0] for row in sides)
        graph_penalty = _corridor_penalty(comp)
        row = (
            dist,
            float(comp["avg"]) + graph_component_weight * graph_penalty,
            float(comp["avg"]),
            -int(comp["size"]),
            int(comp["r0"]),
            int(comp["c0"]),
            float(graph_penalty),
            sides,
        )
        if best is None or row < best:
            best = row
    if best is None:
        return None

    left = float(side_floor) * float(max_dx)
    right = float(side_floor) * float(max_dx)
    down = float(side_floor) * float(max_dy)
    up = float(side_floor) * float(max_dy)
    for _dist, side, amount in best[7]:
        if side == "left":
            left = max(left, float(amount))
        elif side == "right":
            right = max(right, float(amount))
        elif side == "down":
            down = max(down, float(amount))
        elif side == "up":
            up = max(up, float(amount))
    return left, right, down, up, float(best[6])


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
    cluster_confidence=None,
    weak_confidence_max: float = 0.0,
    weak_hot_extra_frac: float = 0.0,
    weak_hot_max_clusters: int = 0,
    weak_hot_side_floor: float = 0.35,
    weak_candidate_clusters=None,
    component_cold_percentile: float = 45.0,
    component_min_cells: int = 4,
    component_max_distance_cells: int = 4,
    graph_edges=None,
    graph_component_weight: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Expand hot cluster regions toward colder neighboring grid bands."""
    expand_regions_by_congestion.last_stats = {
        "expanded": 0,
        "component_expanded": 0,
        "graph_component_expanded": 0,
        "weak_hot_reshaped": 0,
        "weak_hot_clusters": [],
        "weak_hot_candidate_clusters": [],
    }
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
    component_expanded = 0
    graph_component_expanded = 0
    cold_components = cold_connected_components(
        field,
        cold_percentile=float(component_cold_percentile),
        min_cells=max(1, int(component_min_cells)),
    )
    weak_hot_clusters: set[int] = set()
    weak_candidates = None
    if weak_candidate_clusters is not None:
        weak_candidates = {int(cid) for cid in weak_candidate_clusters}
    if cluster_confidence and weak_hot_extra_frac > 0.0 and weak_hot_max_clusters > 0:
        weak_rows = []
        for cid, h in heat:
            if weak_candidates is not None and int(cid) not in weak_candidates:
                continue
            conf = float(cluster_confidence.get(int(cid), 1.0))
            if h < threshold or conf > float(weak_confidence_max):
                continue
            weak_rows.append((float(h), -conf, int(cid)))
        weak_rows.sort(reverse=True)
        weak_hot_clusters = {int(cid) for _h, _neg_conf, cid in weak_rows[:weak_hot_max_clusters]}

    bridge_by_cluster: "dict[int, list[int]]" = {}
    for s, cids in (bridge_softs or {}).items():
        for cid in cids:
            bridge_by_cluster.setdefault(int(cid), []).append(int(s))

    graph_corridors_by_cluster: dict[int, list[tuple[np.ndarray, np.ndarray, float]]] = {}
    graph_component_weight = max(0.0, float(graph_component_weight))
    if graph_component_weight > 0.0 and graph_edges:
        centroids = {}
        for cid, raw_mem in clusters.items():
            mem = np.asarray(raw_mem, dtype=np.int64)
            if mem.size:
                centroids[int(cid)] = np.array(
                    [
                        float(np.mean(hard_xy[mem, 0] / cell_w)),
                        float(np.mean(hard_xy[mem, 1] / cell_h)),
                    ],
                    dtype=np.float64,
                )
        for edge in graph_edges:
            a = int(getattr(edge, "src", -1))
            b = int(getattr(edge, "dst", -1))
            if a not in centroids or b not in centroids:
                continue
            weight = max(0.0, float(getattr(edge, "weight", 1.0)))
            if weight <= 0.0:
                continue
            graph_corridors_by_cluster.setdefault(a, []).append(
                (centroids[a], centroids[b], weight)
            )
            graph_corridors_by_cluster.setdefault(b, []).append(
                (centroids[b], centroids[a], weight)
            )

    for cid, h in heat:
        if h < threshold:
            continue
        weak_hot = int(cid) in weak_hot_clusters
        local_max_dx = max_dx + (float(weak_hot_extra_frac) * float(cw) if weak_hot else 0.0)
        local_max_dy = max_dy + (float(weak_hot_extra_frac) * float(ch) if weak_hot else 0.0)
        side_floor = float(weak_hot_side_floor) if weak_hot else 0.35
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
        component_sides = None
        if cold_components:
            component_sides = _component_expansion(
                cold_components,
                r0,
                r1,
                c0,
                c1,
                cell_w=cell_w,
                cell_h=cell_h,
                max_dx=local_max_dx,
                max_dy=local_max_dy,
                side_floor=side_floor,
                side_band=side_band,
                max_distance_cells=component_max_distance_cells,
                graph_corridors=graph_corridors_by_cluster.get(int(cid)),
                graph_component_weight=graph_component_weight,
            )
        if component_sides is not None:
            left, right, down, up, graph_penalty = component_sides
            component_expanded += 1
            if graph_component_weight > 0.0 and graph_penalty > 0.0:
                graph_component_expanded += 1
        else:
            finite = [v for v in side_vals.values() if np.isfinite(v)]
            if not finite:
                continue
            cold = min(finite)
            left = local_max_dx if side_vals["left"] <= cold + 1e-12 else side_floor * local_max_dx
            right = (
                local_max_dx if side_vals["right"] <= cold + 1e-12 else side_floor * local_max_dx
            )
            down = local_max_dy if side_vals["down"] <= cold + 1e-12 else side_floor * local_max_dy
            up = local_max_dy if side_vals["up"] <= cold + 1e-12 else side_floor * local_max_dy

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
    expand_regions_by_congestion.last_stats = {
        "expanded": int(expanded),
        "component_expanded": int(component_expanded),
        "graph_component_expanded": int(graph_component_expanded),
        "weak_hot_reshaped": int(len(weak_hot_clusters)),
        "weak_hot_clusters": sorted(int(cid) for cid in weak_hot_clusters),
        "weak_hot_candidate_clusters": sorted(int(cid) for cid in weak_candidates or []),
    }
    return hard_region, soft_region, expanded
