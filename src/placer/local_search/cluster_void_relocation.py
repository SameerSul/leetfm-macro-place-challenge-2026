"""Capacity-aware cluster relocation into whitespace around large macros."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence

import numpy as np

from placer.local_search.fields import weighted_congestion_field
from placer.local_search.subcluster_relocation import _hard_group_is_legal
from placer.scoring.wirelength import _build_wl_cache


def _graph_taper_profile(
    members: np.ndarray,
    boundary_members: np.ndarray,
    *,
    max_hops: int,
    decay: float,
    location_graph,
) -> tuple[np.ndarray, np.ndarray]:
    """Return graph-reachable hard macros and a decaying outward shift profile."""
    members = np.asarray(members, dtype=np.int64).reshape(-1)
    boundary_members = np.asarray(boundary_members, dtype=np.int64).reshape(-1)
    if members.size == 0 or boundary_members.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)
    if location_graph is None:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)
    selected, scales = location_graph.directional_graph_profile(
        boundary_members,
        members,
        max_hops=max_hops,
        decay=decay,
    )
    if selected.size <= boundary_members.size:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)
    return selected, scales


def _subtract_rectangles(
    rect: np.ndarray,
    hard_pos: np.ndarray,
    hw: np.ndarray,
    hh: np.ndarray,
    ignored: set[int],
    *,
    min_width: float,
    min_height: float,
) -> list[np.ndarray]:
    """Subtract hard blockages from one rectangle without overlapping fragments."""
    fragments = [np.asarray(rect, dtype=np.float64)]
    for index in range(hard_pos.shape[0]):
        if index in ignored:
            continue
        bx0 = float(hard_pos[index, 0] - hw[index])
        bx1 = float(hard_pos[index, 0] + hw[index])
        by0 = float(hard_pos[index, 1] - hh[index])
        by1 = float(hard_pos[index, 1] + hh[index])
        next_fragments = []
        for fragment in fragments:
            x0, y0, x1, y1 = map(float, fragment)
            ix0, ix1 = max(x0, bx0), min(x1, bx1)
            iy0, iy1 = max(y0, by0), min(y1, by1)
            if ix1 <= ix0 or iy1 <= iy0:
                next_fragments.append(fragment)
                continue
            if ix0 - x0 >= min_width:
                next_fragments.append(np.asarray([x0, y0, ix0, y1], dtype=np.float64))
            if x1 - ix1 >= min_width:
                next_fragments.append(np.asarray([ix1, y0, x1, y1], dtype=np.float64))
            if iy0 - y0 >= min_height and ix1 - ix0 >= min_width:
                next_fragments.append(np.asarray([ix0, y0, ix1, iy0], dtype=np.float64))
            if y1 - iy1 >= min_height and ix1 - ix0 >= min_width:
                next_fragments.append(np.asarray([ix0, iy1, ix1, y1], dtype=np.float64))
        fragments = next_fragments
        if not fragments:
            break
    return [
        fragment
        for fragment in fragments
        if float(fragment[2] - fragment[0]) >= min_width
        and float(fragment[3] - fragment[1]) >= min_height
    ]


def find_large_macro_voids(
    hard_pos: np.ndarray,
    hw: np.ndarray,
    hh: np.ndarray,
    *,
    canvas_width: float | None = None,
    canvas_height: float | None = None,
    large_area_percentile: float = 70.0,
    min_width: float = 0.0,
    min_height: float = 0.0,
    max_voids: int = 128,
    subtract_blockages: bool = True,
) -> list[dict[str, object]]:
    """Return hard-clear interior gaps and large-macro-to-canvas edge pockets."""
    pos = np.asarray(hard_pos, dtype=np.float64)
    half_w = np.asarray(hw, dtype=np.float64)
    half_h = np.asarray(hh, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[0] < 1:
        return []
    areas = 4.0 * half_w * half_h
    positive = areas[areas > 0.0]
    if positive.size == 0:
        return []
    threshold = float(np.percentile(positive, np.clip(large_area_percentile, 0.0, 100.0)))
    large = np.flatnonzero(areas >= threshold - 1.0e-12)
    bases: list[dict[str, object]] = []

    for left_pos, left_raw in enumerate(large):
        for right_raw in large[left_pos + 1 :]:
            left, right = int(left_raw), int(right_raw)
            if pos[left, 0] > pos[right, 0]:
                left, right = right, left
            x0 = pos[left, 0] + half_w[left]
            x1 = pos[right, 0] - half_w[right]
            y0 = max(pos[left, 1] - half_h[left], pos[right, 1] - half_h[right])
            y1 = min(pos[left, 1] + half_h[left], pos[right, 1] + half_h[right])
            if x1 > x0 and y1 > y0:
                bases.append(
                    {
                        "rect": np.asarray([x0, y0, x1, y1], dtype=np.float64),
                        "orientation": "horizontal",
                        "boundary": (left, right),
                        "kind": "interior",
                    }
                )

            bottom, top = left, right
            if pos[bottom, 1] > pos[top, 1]:
                bottom, top = top, bottom
            y0 = pos[bottom, 1] + half_h[bottom]
            y1 = pos[top, 1] - half_h[top]
            x0 = max(pos[bottom, 0] - half_w[bottom], pos[top, 0] - half_w[top])
            x1 = min(pos[bottom, 0] + half_w[bottom], pos[top, 0] + half_w[top])
            if x1 > x0 and y1 > y0:
                bases.append(
                    {
                        "rect": np.asarray([x0, y0, x1, y1], dtype=np.float64),
                        "orientation": "vertical",
                        "boundary": (bottom, top),
                        "kind": "interior",
                    }
                )

    if canvas_width is not None and canvas_height is not None:
        cw, ch = float(canvas_width), float(canvas_height)
        for raw_index in large:
            index = int(raw_index)
            left = float(pos[index, 0] - half_w[index])
            right = float(pos[index, 0] + half_w[index])
            bottom = float(pos[index, 1] - half_h[index])
            top = float(pos[index, 1] + half_h[index])
            for rect, orientation in (
                ([0.0, bottom, left, top], "horizontal"),
                ([right, bottom, cw, top], "horizontal"),
                ([left, 0.0, right, bottom], "vertical"),
                ([left, top, right, ch], "vertical"),
            ):
                if rect[2] > rect[0] and rect[3] > rect[1]:
                    bases.append(
                        {
                            "rect": np.asarray(rect, dtype=np.float64),
                            "orientation": orientation,
                            "boundary": (index, -1),
                            "kind": "edge",
                        }
                    )

    bases.sort(
        key=lambda row: (
            -float((row["rect"][2] - row["rect"][0]) * (row["rect"][3] - row["rect"][1])),
            str(row["kind"]),
            tuple(row["boundary"]),
        )
    )
    bases = bases[: max(1, 4 * int(max_voids))]
    rows: list[dict[str, object]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for base in bases:
        boundary = tuple(int(value) for value in base["boundary"])
        ignored = {value for value in boundary if value >= 0}
        fragments = (
            _subtract_rectangles(
                np.asarray(base["rect"], dtype=np.float64),
                pos,
                half_w,
                half_h,
                ignored,
                min_width=float(min_width),
                min_height=float(min_height),
            )
            if subtract_blockages
            else [np.asarray(base["rect"], dtype=np.float64)]
        )
        for rect in fragments:
            x0, y0, x1, y1 = map(float, rect)
            width, height = x1 - x0, y1 - y0
            if width < float(min_width) or height < float(min_height):
                continue
            key = tuple(int(round(value * 1.0e6)) for value in rect)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "rect": rect,
                    "orientation": str(base["orientation"]),
                    "boundary": boundary,
                    "kind": str(base["kind"]),
                    "area": width * height,
                }
            )
    rows.sort(
        key=lambda row: (
            -float(row["area"]),
            str(row["kind"]),
            str(row["orientation"]),
            tuple(row["boundary"]),
        )
    )
    return rows[: max(0, int(max_voids))]


def _rect_cells(rect, nr: int, nc: int, cw: float, ch: float) -> np.ndarray:
    cell_w, cell_h = float(cw) / nc, float(ch) / nr
    x0, y0, x1, y1 = map(float, rect)
    c0 = max(0, int(np.floor(x0 / cell_w)))
    c1 = min(nc - 1, int(np.floor(np.nextafter(x1, -np.inf) / cell_w)))
    r0 = max(0, int(np.floor(y0 / cell_h)))
    r1 = min(nr - 1, int(np.floor(np.nextafter(y1, -np.inf) / cell_h)))
    if c1 < c0 or r1 < r0:
        return np.zeros(0, dtype=np.int64)
    rr, cc = np.meshgrid(
        np.arange(r0, r1 + 1, dtype=np.int64),
        np.arange(c0, c1 + 1, dtype=np.int64),
        indexing="ij",
    )
    return (rr * nc + cc).ravel()


def _cluster_graph_targets(hard_pos, clusters, edges) -> dict[int, np.ndarray]:
    centroids = {
        int(cid): np.mean(hard_pos[np.asarray(members, dtype=np.int64)], axis=0)
        for cid, members in clusters.items()
        if len(members)
    }
    weighted: dict[int, list[tuple[np.ndarray, float]]] = {}
    for edge in edges or ():
        src, dst = int(edge.src), int(edge.dst)
        weight = max(0.0, float(edge.weight))
        if weight <= 0.0 or src not in centroids or dst not in centroids:
            continue
        weighted.setdefault(src, []).append((centroids[dst], weight))
        weighted.setdefault(dst, []).append((centroids[src], weight))
    out = {}
    for cid, values in weighted.items():
        total = sum(weight for _point, weight in values)
        out[cid] = sum(point * weight for point, weight in values) / max(total, 1.0e-12)
    return out


def _rect_overlap_area(pos, half_w, half_h, rect) -> float:
    if len(pos) == 0:
        return 0.0
    points = np.asarray(pos, dtype=np.float64)
    x0, y0, x1, y1 = map(float, rect)
    ox = np.maximum(
        0.0,
        np.minimum(points[:, 0] + half_w, x1) - np.maximum(points[:, 0] - half_w, x0),
    )
    oy = np.maximum(
        0.0,
        np.minimum(points[:, 1] + half_h, y1) - np.maximum(points[:, 1] - half_h, y0),
    )
    return float(np.sum(ox * oy))


def _projected_void_utilization(
    rect,
    soft_pos,
    soft_hw,
    soft_hh,
    *,
    moved_soft=(),
    new_soft=None,
    new_soft_hw=None,
    new_soft_hh=None,
    new_hard=None,
    new_hard_hw=None,
    new_hard_hh=None,
) -> float:
    """Return exact rectangle occupancy after applying one proposed move."""
    x0, y0, x1, y1 = map(float, rect)
    rect_area = max((x1 - x0) * (y1 - y0), 1.0e-12)
    occupied = _rect_overlap_area(soft_pos, soft_hw, soft_hh, rect)
    moved = np.asarray(moved_soft, dtype=np.int64).reshape(-1)
    if moved.size:
        occupied -= _rect_overlap_area(soft_pos[moved], soft_hw[moved], soft_hh[moved], rect)
    if new_soft is not None and len(new_soft):
        occupied += _rect_overlap_area(new_soft, new_soft_hw, new_soft_hh, rect)
    if new_hard is not None and len(new_hard):
        occupied += _rect_overlap_area(new_hard, new_hard_hw, new_hard_hh, rect)
    return float(max(0.0, occupied) / rect_area)


def _boxes_overlap(pos: np.ndarray, half_w: np.ndarray, half_h: np.ndarray) -> bool:
    for left in range(pos.shape[0]):
        for right in range(left + 1, pos.shape[0]):
            if abs(float(pos[left, 0] - pos[right, 0])) < float(
                half_w[left] + half_w[right]
            ) and abs(float(pos[left, 1] - pos[right, 1])) < float(half_h[left] + half_h[right]):
                return True
    return False


def _soft_avoids_hard(new_soft, indices, soft_hw, soft_hh, hard_pos, hw, hh) -> bool:
    for local, soft_index in enumerate(indices):
        dx = np.abs(hard_pos[:, 0] - new_soft[local, 0])
        dy = np.abs(hard_pos[:, 1] - new_soft[local, 1])
        overlap = (dx < hw + soft_hw[int(soft_index)]) & (dy < hh + soft_hh[int(soft_index)])
        if bool(np.any(overlap)):
            return False
    return True


def _translate_soft_group(old_xy, half_w, half_h, rect, desired) -> np.ndarray | None:
    center = np.mean(old_xy, axis=0)
    relative = old_xy - center
    x0, y0, x1, y1 = map(float, rect)
    ax0 = x0 - float(np.min(relative[:, 0] - half_w))
    ax1 = x1 - float(np.max(relative[:, 0] + half_w))
    ay0 = y0 - float(np.min(relative[:, 1] - half_h))
    ay1 = y1 - float(np.max(relative[:, 1] + half_h))
    if ax0 > ax1 or ay0 > ay1:
        return None
    anchor = np.clip(np.asarray(desired, dtype=np.float64), [ax0, ay0], [ax1, ay1])
    return old_xy + (anchor - center)


def _shelf_pack_soft(
    indices, soft_hw, soft_hh, rect, desired, *, reverse: bool
) -> np.ndarray | None:
    """Pack a transient routing cohort without internal overlaps."""
    indices = np.asarray(indices, dtype=np.int64)
    x0, y0, x1, y1 = map(float, rect)
    width = x1 - x0
    gap = 0.02 * min(width, y1 - y0) / max(np.sqrt(indices.size), 1.0)
    order = sorted(
        range(indices.size),
        key=lambda local: (
            -float(soft_hh[indices[local]]),
            -float(soft_hw[indices[local]] * soft_hh[indices[local]]),
            int(indices[local]),
        ),
        reverse=bool(reverse),
    )
    packed = np.empty((indices.size, 2), dtype=np.float64)
    cursor_x, cursor_y, row_height = 0.0, 0.0, 0.0
    for local in order:
        macro_w = 2.0 * float(soft_hw[indices[local]])
        macro_h = 2.0 * float(soft_hh[indices[local]])
        if cursor_x > 0.0 and cursor_x + macro_w > width + 1.0e-9:
            cursor_x = 0.0
            cursor_y += row_height + gap
            row_height = 0.0
        if macro_w > width + 1.0e-9 or cursor_y + macro_h > y1 - y0 + 1.0e-9:
            return None
        packed[local] = [cursor_x + 0.5 * macro_w, cursor_y + 0.5 * macro_h]
        cursor_x += macro_w + gap
        row_height = max(row_height, macro_h)
    lo = np.min(packed - np.column_stack([soft_hw[indices], soft_hh[indices]]), axis=0)
    hi = np.max(packed + np.column_stack([soft_hw[indices], soft_hh[indices]]), axis=0)
    size = hi - lo
    anchor = np.clip(
        np.asarray(desired, dtype=np.float64),
        [x0 + 0.5 * size[0], y0 + 0.5 * size[1]],
        [x1 - 0.5 * size[0], y1 - 0.5 * size[1]],
    )
    return packed + anchor - 0.5 * (lo + hi)


def _routing_target(plc, hard_pos, soft_pos, indices) -> np.ndarray:
    """Return the weighted external-pin centroid of one soft routing unit."""
    cache = _build_wl_cache(plc)
    hard_ref = {int(module): index for index, module in enumerate(plc.hard_macro_indices)}
    soft_ref = {int(module): index for index, module in enumerate(plc.soft_macro_indices)}
    member_refs = {int(plc.soft_macro_indices[int(index)]) for index in indices}
    incident = set()
    for net_index, start_raw in enumerate(cache["net_starts"]):
        start = int(start_raw)
        length = int(cache["net_lengths"][net_index])
        if any(int(ref) in member_refs for ref in cache["ref_idx"][start : start + length]):
            incident.add(net_index)
    total = 0.0
    weighted = np.zeros(2, dtype=np.float64)
    for net_index in sorted(incident):
        start = int(cache["net_starts"][net_index])
        length = int(cache["net_lengths"][net_index])
        weight = max(float(cache["net_weights"][net_index]), 0.0)
        for pin in range(start, start + length):
            ref = int(cache["ref_idx"][pin])
            if ref in member_refs:
                continue
            if ref in hard_ref:
                point = hard_pos[hard_ref[ref]]
            elif ref in soft_ref:
                point = soft_pos[soft_ref[ref]]
            else:
                try:
                    point = np.asarray(plc.modules_w_pins[ref].get_pos(), dtype=np.float64)
                except Exception:
                    continue
            point = point + np.asarray([cache["x_off"][pin], cache["y_off"][pin]])
            weighted += weight * point
            total += weight
    return weighted / total if total > 0.0 else np.mean(soft_pos[indices], axis=0)


def _soft_routing_units(
    plc,
    soft_pos,
    cluster_softs,
    bridge_softs,
    soft_only_bundles,
    movable_soft,
    n,
    local_heat,
    *,
    max_fanout: int,
    max_cohort: int,
    top_cohorts: int,
    top_singletons: int,
) -> list[dict[str, object]]:
    """Build stable bundles, transient routing cohorts, and hot singleton units."""
    count = soft_pos.shape[0]
    excluded = {int(index) for index in (bridge_softs or {})}
    for members in cluster_softs.values():
        excluded.update(int(index) - int(n) for index in np.asarray(members, dtype=np.int64))
    eligible = {index for index in range(count) if movable_soft[index] and index not in excluded}
    units = []
    used: set[int] = set()
    for bundle in soft_only_bundles or ():
        members = np.asarray(getattr(bundle, "members", ()), dtype=np.int64)
        members = np.asarray(
            sorted({int(index) for index in members if int(index) in eligible}), dtype=np.int64
        )
        if members.size < 2 or any(int(index) in used for index in members):
            continue
        used.update(int(index) for index in members)
        units.append(
            {
                "kind": "stable_bundle",
                "indices": members,
                "heat": float(np.mean(local_heat[members])),
            }
        )

    if not (eligible - used):
        return units

    cache = _build_wl_cache(plc)
    soft_ref = {int(module): index for index, module in enumerate(plc.soft_macro_indices)}
    adjacency: dict[int, dict[int, float]] = {}
    for net_index, start_raw in enumerate(cache["net_starts"]):
        length = int(cache["net_lengths"][net_index])
        if length < 2 or length > max(2, int(max_fanout)):
            continue
        start = int(start_raw)
        members = sorted(
            {
                soft_ref[int(ref)]
                for ref in cache["ref_idx"][start : start + length]
                if int(ref) in soft_ref
                and soft_ref[int(ref)] in eligible
                and soft_ref[int(ref)] not in used
            }
        )
        if len(members) < 2:
            continue
        weight = max(float(cache["net_weights"][net_index]), 0.0) / max(len(members) - 1, 1)
        for left_pos, left in enumerate(members):
            for right in members[left_pos + 1 :]:
                adjacency.setdefault(left, {})[right] = (
                    adjacency.setdefault(left, {}).get(right, 0.0) + weight
                )
                adjacency.setdefault(right, {})[left] = (
                    adjacency.setdefault(right, {}).get(left, 0.0) + weight
                )

    cohorts = []
    for seed in sorted(eligible - used, key=lambda index: (-float(local_heat[index]), index)):
        if seed in used or seed not in adjacency:
            continue
        members = [seed]
        while len(members) < max(2, int(max_cohort)):
            candidates = eligible - used - set(members)
            ranked = []
            for candidate in candidates:
                support = sum(adjacency.get(member, {}).get(candidate, 0.0) for member in members)
                if support > 0.0:
                    ranked.append((-support, -float(local_heat[candidate]), int(candidate)))
            if not ranked:
                break
            members.append(min(ranked)[2])
        if len(members) < 2:
            continue
        array = np.asarray(sorted(members), dtype=np.int64)
        used.update(int(index) for index in array)
        cohorts.append(
            {
                "kind": "routing_cohort",
                "indices": array,
                "heat": float(np.mean(local_heat[array])),
            }
        )
    cohorts.sort(key=lambda row: (-float(row["heat"]), tuple(row["indices"])))
    units.extend(cohorts[: max(0, int(top_cohorts))])

    singletons = sorted(
        eligible - used,
        key=lambda index: (-float(local_heat[index]), int(index)),
    )[: max(0, int(top_singletons))]
    units.extend(
        {"kind": "soft_singleton", "indices": np.asarray([index]), "heat": float(local_heat[index])}
        for index in singletons
    )
    return units


def _interleave_lanes(lanes: Mapping[str, list[dict[str, object]]]) -> list[dict[str, object]]:
    order = ("hard_expand", "hard_leaf", "stable_bundle", "routing_cohort", "soft_singleton")
    result = []
    cursor = 0
    while True:
        added = False
        for lane in order:
            rows = lanes.get(lane, [])
            if cursor < len(rows):
                result.append(rows[cursor])
                added = True
        if not added:
            return result
        cursor += 1


def _void_cluster_relocation(
    hard_pos: np.ndarray,
    soft_pos: np.ndarray,
    hw: np.ndarray,
    hh: np.ndarray,
    soft_hw: np.ndarray,
    soft_hh: np.ndarray,
    cw: float,
    ch: float,
    n: int,
    benchmark,
    incremental_scorer,
    initial_score: float,
    *,
    clusters: Mapping[int, Sequence[int]],
    cluster_softs: Mapping[int, Sequence[int]],
    edges,
    location_graph=None,
    movable_h: np.ndarray,
    movable_soft: np.ndarray,
    bridge_softs: Mapping[int, Sequence[int]] | None = None,
    soft_only_bundles: Sequence[object] | None = None,
    candidate_allowed: Callable[[np.ndarray, np.ndarray], bool] | None = None,
    deadline: float | None = None,
    large_area_percentile: float = 70.0,
    min_gap_cells: int = 2,
    max_voids: int = 96,
    max_cluster_hard: int = 12,
    max_cluster_soft: int = 32,
    top_clusters: int = 10,
    max_expand_hard: int = 48,
    max_expand_soft: int = 96,
    top_expand_clusters: int = 12,
    graph_taper_max_hops: int = 3,
    graph_taper_decay: float = 0.55,
    max_utilization: float = 0.78,
    soft_compact_scale: float = 0.65,
    min_field_drop: float = 0.01,
    min_gain: float = 0.0001,
    soft_min_gain: float = 0.00005,
    routing_max_fanout: int = 16,
    routing_max_cohort: int = 8,
    routing_top_cohorts: int = 12,
    routing_top_singletons: int = 16,
    max_accepts: int = 4,
    max_scored: int = 64,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Relocate hierarchy leaves and routing-only soft units into true voids."""
    stats = {
        "voids": 0,
        "interior_voids": 0,
        "edge_voids": 0,
        "eligible_clusters": 0,
        "eligible_expand_clusters": 0,
        "expansion_candidates": 0,
        "expansion_accepts": 0,
        "expansion_legal": 0,
        "expansion_hierarchy_rejects": 0,
        "expansion_scored": 0,
        "expansion_best_proxy_gain": 0.0,
        "graph_taper_candidates": 0,
        "graph_taper_accepts": 0,
        "graph_taper_scored": 0,
        "graph_taper_best_proxy_gain": 0.0,
        "soft_units": 0,
        "capacity_rejects": 0,
        "overlap_rejects": 0,
        "hard_legality_rejects": 0,
        "field_rejects": 0,
        "candidates": 0,
        "legal": 0,
        "hierarchy_rejects": 0,
        "scored": 0,
        "accepts": 0,
        "accepted_cluster": -1,
        "accepted_kind": "none",
        "accepted_kinds": [],
        "best_proxy_gain": 0.0,
    }
    _void_cluster_relocation.last_stats = stats
    nr, nc = int(benchmark.grid_rows), int(benchmark.grid_cols)
    cell_w, cell_h = float(cw) / nc, float(ch) / nr
    movable_h = np.asarray(movable_h, dtype=bool)
    movable_soft = np.asarray(movable_soft, dtype=bool)
    graph_targets = _cluster_graph_targets(hard_pos, clusters, edges)
    moved_hard: set[int] = set()
    moved_soft: set[int] = set()
    current_score = float(initial_score)
    if location_graph is not None:
        location_graph.synchronize(hard_pos, soft_pos)

    for accept_round in range(max(0, int(max_accepts))):
        if int(stats["scored"]) >= max(0, int(max_scored)):
            break
        if deadline is not None and time.monotonic() >= deadline:
            break
        voids = find_large_macro_voids(
            hard_pos,
            hw,
            hh,
            canvas_width=cw,
            canvas_height=ch,
            large_area_percentile=large_area_percentile,
            min_width=max(1, int(min_gap_cells)) * cell_w,
            min_height=max(1, int(min_gap_cells)) * cell_h,
            max_voids=max_voids,
        )
        legacy_voids = find_large_macro_voids(
            hard_pos,
            hw,
            hh,
            large_area_percentile=large_area_percentile,
            min_width=max(1, int(min_gap_cells)) * cell_w,
            min_height=max(1, int(min_gap_cells)) * cell_h,
            max_voids=max_voids,
            subtract_blockages=False,
        )
        for void in voids:
            void["legacy"] = False
        for void in legacy_voids:
            void["legacy"] = True
        stats["voids"] = max(int(stats["voids"]), int(len(voids) + len(legacy_voids)))
        stats["interior_voids"] = max(
            int(stats["interior_voids"]), sum(row["kind"] == "interior" for row in voids)
        )
        stats["edge_voids"] = max(
            int(stats["edge_voids"]), sum(row["kind"] == "edge" for row in voids)
        )
        if not voids:
            break
        field = weighted_congestion_field(incremental_scorer, nr, nc)
        if field is None:
            break
        field = np.asarray(field, dtype=np.float64)
        density = np.asarray(incremental_scorer.grid_occupied, dtype=np.float64)
        density_field = density / max(float(incremental_scorer.dens_grid_area), 1.0e-12)
        void_rows = []
        for void in [*legacy_voids, *voids]:
            cells = _rect_cells(void["rect"], nr, nc, cw, ch)
            if cells.size == 0:
                continue
            row = dict(void)
            row["cells"] = cells
            row["field"] = float(np.mean(field.ravel()[cells]))
            row["density"] = float(np.mean(density_field[cells]))
            void_rows.append(row)
        if not void_rows:
            break

        lanes: dict[str, list[dict[str, object]]] = {
            "hard_expand": [],
            "hard_leaf": [],
            "stable_bundle": [],
            "routing_cohort": [],
            "soft_singleton": [],
        }
        cluster_rows = []
        for cid, raw_members in sorted(clusters.items()):
            members = np.asarray(raw_members, dtype=np.int64)
            if (
                members.size < 2
                or members.size > max(2, int(max_cluster_hard))
                or not bool(np.all(movable_h[members]))
                or any(int(member) in moved_hard for member in members)
            ):
                continue
            soft_indices = np.asarray(cluster_softs.get(int(cid), ()), dtype=np.int64) - int(n)
            soft_indices = soft_indices[
                (soft_indices >= 0)
                & (soft_indices < soft_pos.shape[0])
                & movable_soft[soft_indices]
            ]
            soft_indices = np.asarray(
                [index for index in soft_indices if int(index) not in moved_soft], dtype=np.int64
            )
            if soft_indices.size > max(0, int(max_cluster_soft)):
                continue
            points = hard_pos[members]
            source_cells = np.clip((points[:, 1] / cell_h).astype(np.int64), 0, nr - 1) * nc
            source_cells += np.clip((points[:, 0] / cell_w).astype(np.int64), 0, nc - 1)
            cluster_rows.append(
                {
                    "cid": int(cid),
                    "members": members,
                    "soft_indices": soft_indices,
                    "source_field": float(np.mean(field.ravel()[source_cells])),
                    "graph_target": np.asarray(
                        graph_targets.get(int(cid), np.mean(points, axis=0)), dtype=np.float64
                    ),
                }
            )
        cluster_rows.sort(key=lambda row: (-float(row["source_field"]), int(row["cid"])))
        cluster_rows = cluster_rows[: max(0, int(top_clusters))]
        stats["eligible_clusters"] = max(int(stats["eligible_clusters"]), int(len(cluster_rows)))

        expand_rows = []
        if len(moved_hard) < hard_pos.shape[0]:
            expand_boundary_indices = {
                int(index)
                for void in void_rows
                if not bool(void.get("legacy", False))
                for index in void["boundary"]
                if int(index) >= 0
            }
            for cid, raw_members in sorted(clusters.items()):
                members = np.asarray(raw_members, dtype=np.int64)
                if (
                    members.size < 2
                    or members.size > max(2, int(max_expand_hard))
                    or not bool(np.all(movable_h[members]))
                    or any(int(member) in moved_hard for member in members)
                    or not expand_boundary_indices.intersection(int(member) for member in members)
                ):
                    continue
                soft_indices = np.asarray(cluster_softs.get(int(cid), ()), dtype=np.int64) - int(n)
                soft_indices = soft_indices[
                    (soft_indices >= 0)
                    & (soft_indices < soft_pos.shape[0])
                    & movable_soft[soft_indices]
                ]
                if soft_indices.size > max(0, int(max_expand_soft)):
                    continue
                points = hard_pos[members]
                source_cells = np.clip((points[:, 1] / cell_h).astype(np.int64), 0, nr - 1) * nc
                source_cells += np.clip((points[:, 0] / cell_w).astype(np.int64), 0, nc - 1)
                expand_rows.append(
                    {
                        "cid": int(cid),
                        "members": members,
                        "soft_indices": soft_indices,
                        "source_field": float(np.mean(field.ravel()[source_cells])),
                    }
                )
            expand_rows.sort(key=lambda row: (-float(row["source_field"]), int(row["cid"])))
            expand_rows = expand_rows[: max(0, int(top_expand_clusters))]
        stats["eligible_expand_clusters"] = max(
            int(stats["eligible_expand_clusters"]), int(len(expand_rows))
        )
        for row in expand_rows:
            members = np.asarray(row["members"], dtype=np.int64)
            soft_indices = np.asarray(row["soft_indices"], dtype=np.int64)
            boundary_members = set(int(member) for member in members)
            for void in void_rows:
                if bool(void.get("legacy", False)) or not boundary_members.intersection(
                    int(index) for index in void["boundary"] if int(index) >= 0
                ):
                    continue
                relief = float(row["source_field"]) - float(void["field"])
                if relief < float(min_field_drop):
                    stats["field_rejects"] += 1
                    continue
                boundary = {
                    int(index) for index in void["boundary"] if int(index) in boundary_members
                }
                boundary_local = np.asarray(
                    [local for local, index in enumerate(members) if int(index) in boundary],
                    dtype=np.int64,
                )
                boundary_center = np.mean(hard_pos[members[boundary_local]], axis=0)
                void_center = 0.5 * (
                    np.asarray(void["rect"][:2], dtype=np.float64)
                    + np.asarray(void["rect"][2:], dtype=np.float64)
                )
                x0, y0, x1, y1 = map(float, void["rect"])
                if str(void["orientation"]) == "horizontal":
                    axis = 0
                    positive = bool(void_center[0] >= boundary_center[0])
                    direction = "right" if positive else "left"
                    aligned = (
                        hard_pos[members[boundary_local], 1] - hh[members[boundary_local]]
                        >= y0 - 1.0e-9
                    ) & (
                        hard_pos[members[boundary_local], 1] + hh[members[boundary_local]]
                        <= y1 + 1.0e-9
                    )
                else:
                    axis = 1
                    positive = bool(void_center[1] >= boundary_center[1])
                    direction = "top" if positive else "bottom"
                    aligned = (
                        hard_pos[members[boundary_local], 0] - hw[members[boundary_local]]
                        >= x0 - 1.0e-9
                    ) & (
                        hard_pos[members[boundary_local], 0] + hw[members[boundary_local]]
                        <= x1 + 1.0e-9
                    )
                move_members = members[boundary_local[aligned]]
                if move_members.size == 0:
                    continue
                half_axis = hw[move_members] if axis == 0 else hh[move_members]
                if positive:
                    available = float(
                        np.min((x1 if axis == 0 else y1) - hard_pos[move_members, axis] - half_axis)
                    )
                else:
                    available = float(
                        np.min(hard_pos[move_members, axis] - half_axis - (x0 if axis == 0 else y0))
                    )
                if available <= 1.0e-9:
                    continue
                cluster_center = np.mean(hard_pos[members], axis=0)
                if soft_indices.size:
                    if axis == 0:
                        soft_aligned = (soft_pos[soft_indices, 1] >= y0) & (
                            soft_pos[soft_indices, 1] <= y1
                        )
                    else:
                        soft_aligned = (soft_pos[soft_indices, 0] >= x0) & (
                            soft_pos[soft_indices, 0] <= x1
                        )
                    soft_facing = (
                        soft_pos[soft_indices, axis] >= cluster_center[axis]
                        if positive
                        else soft_pos[soft_indices, axis] <= cluster_center[axis]
                    )
                    move_soft_indices = soft_indices[soft_aligned & soft_facing]
                else:
                    move_soft_indices = soft_indices
                for fraction in (0.25, 0.50, 0.75):
                    desired_shift = float(fraction) * available * (1.0 if positive else -1.0)
                    shift = 0.0
                    new_hard = None
                    for shrink in (1.0, 0.5, 0.25, 0.125, 0.0625):
                        trial_shift = desired_shift * shrink
                        trial_hard = hard_pos[move_members].copy()
                        trial_hard[:, axis] += trial_shift
                        if not _hard_group_is_legal(hard_pos, move_members, trial_hard, hw, hh):
                            continue
                        returned_hard = hard_pos.astype(np.float32).astype(np.float64)
                        returned_targets = trial_hard.astype(np.float32).astype(np.float64)
                        if not _hard_group_is_legal(
                            returned_hard,
                            move_members,
                            returned_targets,
                            hw,
                            hh,
                            tolerance=0.0,
                        ):
                            continue
                        shift = float(trial_shift)
                        new_hard = trial_hard
                        break
                    if new_hard is None:
                        continue
                    new_soft = soft_pos[move_soft_indices].copy()
                    new_soft[:, axis] += shift
                    new_soft[:, 0] = np.clip(
                        new_soft[:, 0],
                        soft_hw[move_soft_indices],
                        cw - soft_hw[move_soft_indices],
                    )
                    new_soft[:, 1] = np.clip(
                        new_soft[:, 1],
                        soft_hh[move_soft_indices],
                        ch - soft_hh[move_soft_indices],
                    )
                    utilization = _projected_void_utilization(
                        void["rect"],
                        soft_pos,
                        soft_hw,
                        soft_hh,
                        moved_soft=move_soft_indices,
                        new_soft=new_soft,
                        new_soft_hw=soft_hw[move_soft_indices],
                        new_soft_hh=soft_hh[move_soft_indices],
                        new_hard=new_hard,
                        new_hard_hw=hw[move_members],
                        new_hard_hh=hh[move_members],
                    )
                    if utilization > float(max_utilization):
                        stats["capacity_rejects"] += 1
                        continue
                    lanes["hard_expand"].append(
                        {
                            "lane": "hard_expand",
                            "rank": (
                                -relief,
                                -abs(shift),
                                utilization,
                                int(row["cid"]),
                                direction,
                            ),
                            "cid": int(row["cid"]),
                            "members": move_members,
                            "new_hard": new_hard,
                            "soft_indices": move_soft_indices,
                            "new_soft": new_soft,
                            "outward_shift": abs(shift),
                            "kind": f"hard_soft_expand_{direction}_{void['kind']}_void",
                        }
                    )
                    taper_members, taper_scales = _graph_taper_profile(
                        members,
                        move_members,
                        max_hops=graph_taper_max_hops,
                        decay=graph_taper_decay,
                        location_graph=location_graph,
                    )
                    if taper_members.size:
                        taper_hard = hard_pos[taper_members].copy()
                        taper_hard[:, axis] += shift * taper_scales
                        if _hard_group_is_legal(hard_pos, taper_members, taper_hard, hw, hh):
                            returned_hard = hard_pos.astype(np.float32).astype(np.float64)
                            returned_targets = taper_hard.astype(np.float32).astype(np.float64)
                            if _hard_group_is_legal(
                                returned_hard,
                                taper_members,
                                returned_targets,
                                hw,
                                hh,
                                tolerance=0.0,
                            ):
                                taper_utilization = _projected_void_utilization(
                                    void["rect"],
                                    soft_pos,
                                    soft_hw,
                                    soft_hh,
                                    moved_soft=move_soft_indices,
                                    new_soft=new_soft,
                                    new_soft_hw=soft_hw[move_soft_indices],
                                    new_soft_hh=soft_hh[move_soft_indices],
                                    new_hard=taper_hard,
                                    new_hard_hw=hw[taper_members],
                                    new_hard_hh=hh[taper_members],
                                )
                                if taper_utilization > float(max_utilization):
                                    stats["capacity_rejects"] += 1
                                    continue
                                lanes["hard_expand"].append(
                                    {
                                        "lane": "hard_expand",
                                        "rank": (
                                            -relief,
                                            -abs(shift),
                                            taper_utilization,
                                            int(row["cid"]),
                                            direction,
                                        ),
                                        "cid": int(row["cid"]),
                                        "members": taper_members,
                                        "new_hard": taper_hard,
                                        "soft_indices": move_soft_indices,
                                        "new_soft": new_soft,
                                        "outward_shift": abs(shift),
                                        "kind": (
                                            f"hard_soft_expand_graph_taper_{direction}_"
                                            f"{void['kind']}_void"
                                        ),
                                    }
                                )
        uniform_expansions = [
            row for row in lanes["hard_expand"] if "graph_taper" not in str(row["kind"])
        ]
        graph_expansions = [
            row for row in lanes["hard_expand"] if "graph_taper" in str(row["kind"])
        ]
        uniform_expansions.sort(key=lambda row: row["rank"])
        graph_expansions.sort(key=lambda row: row["rank"])
        lanes["hard_expand"] = uniform_expansions[:24] + graph_expansions[:24]
        stats["graph_taper_candidates"] = max(
            int(stats["graph_taper_candidates"]), min(24, len(graph_expansions))
        )
        stats["expansion_candidates"] = max(
            int(stats["expansion_candidates"]), int(len(lanes["hard_expand"]))
        )
        for row in cluster_rows:
            if deadline is not None and time.monotonic() >= deadline:
                break
            members = row["members"]
            soft_indices = row["soft_indices"]
            old_hard = hard_pos[members]
            old_center = np.mean(old_hard, axis=0)
            rel = old_hard - old_center
            rel_x0 = float(np.min(rel[:, 0] - hw[members]))
            rel_x1 = float(np.max(rel[:, 0] + hw[members]))
            rel_y0 = float(np.min(rel[:, 1] - hh[members]))
            rel_y1 = float(np.max(rel[:, 1] + hh[members]))
            for void in void_rows:
                if deadline is not None and time.monotonic() >= deadline:
                    break
                if set(void["boundary"]).intersection(int(member) for member in members):
                    continue
                x0, y0, x1, y1 = map(float, void["rect"])
                ax0, ax1 = x0 - rel_x0, x1 - rel_x1
                ay0, ay1 = y0 - rel_y0, y1 - rel_y1
                if ax0 > ax1 or ay0 > ay1:
                    continue
                relief = float(row["source_field"]) - float(void["field"])
                if relief < float(min_field_drop):
                    stats["field_rejects"] += 1
                    continue
                target = np.clip(row["graph_target"], [ax0, ay0], [ax1, ay1])
                center = np.asarray([(ax0 + ax1) * 0.5, (ay0 + ay1) * 0.5])
                for anchor_kind, anchor in (("graph", target), ("center", center)):
                    new_hard = old_hard + (anchor - old_center)
                    old_soft = soft_pos[soft_indices]
                    new_soft = np.empty_like(old_soft)
                    if soft_indices.size:
                        if bool(
                            np.any(2.0 * soft_hw[soft_indices] > x1 - x0)
                            or np.any(2.0 * soft_hh[soft_indices] > y1 - y0)
                        ):
                            continue
                        scale = float(np.clip(soft_compact_scale, 0.0, 1.0))
                        new_soft[:] = anchor + scale * (old_soft - old_center)
                        new_soft[:, 0] = np.clip(
                            new_soft[:, 0], x0 + soft_hw[soft_indices], x1 - soft_hw[soft_indices]
                        )
                        new_soft[:, 1] = np.clip(
                            new_soft[:, 1], y0 + soft_hh[soft_indices], y1 - soft_hh[soft_indices]
                        )
                    if bool(void.get("legacy", False)):
                        move_area = float(np.sum(4.0 * hw[members] * hh[members]))
                        if soft_indices.size:
                            move_area += float(
                                np.sum(4.0 * soft_hw[soft_indices] * soft_hh[soft_indices])
                            )
                        utilization = float(void["density"]) + move_area / max(
                            float(void["area"]), 1.0e-12
                        )
                    else:
                        utilization = _projected_void_utilization(
                            void["rect"],
                            soft_pos,
                            soft_hw,
                            soft_hh,
                            moved_soft=soft_indices,
                            new_soft=new_soft,
                            new_soft_hw=soft_hw[soft_indices],
                            new_soft_hh=soft_hh[soft_indices],
                            new_hard=new_hard,
                            new_hard_hw=hw[members],
                            new_hard_hh=hh[members],
                        )
                    if utilization > float(max_utilization):
                        stats["capacity_rejects"] += 1
                        continue
                    graph_distance = float(np.linalg.norm(anchor - row["graph_target"]))
                    legacy = bool(void.get("legacy", False))
                    lanes["hard_leaf"].append(
                        {
                            "lane": "hard_leaf",
                            "rank": (
                                -relief,
                                graph_distance / max(float(np.hypot(cw, ch)), 1.0e-12),
                                utilization,
                                int(row["cid"]),
                                anchor_kind,
                            ),
                            "legacy": legacy,
                            "cid": int(row["cid"]),
                            "members": members,
                            "new_hard": new_hard,
                            "soft_indices": soft_indices,
                            "new_soft": new_soft,
                            "kind": (
                                f"hard_soft_{anchor_kind}_void"
                                if legacy
                                else f"hard_soft_{anchor_kind}_{void['kind']}_void"
                            ),
                        }
                    )

        soft_ci = np.clip((soft_pos[:, 0] / cell_w).astype(np.int64), 0, nc - 1)
        soft_ri = np.clip((soft_pos[:, 1] / cell_h).astype(np.int64), 0, nr - 1)
        local_heat = field[soft_ri, soft_ci] + density_field[soft_ri * nc + soft_ci]
        scorer_plc = getattr(incremental_scorer, "plc", None)
        units = _soft_routing_units(
            scorer_plc,
            soft_pos,
            cluster_softs,
            bridge_softs,
            soft_only_bundles,
            movable_soft,
            n,
            local_heat,
            max_fanout=routing_max_fanout,
            max_cohort=routing_max_cohort,
            top_cohorts=routing_top_cohorts,
            top_singletons=routing_top_singletons,
        )
        units = [
            unit for unit in units if not any(int(index) in moved_soft for index in unit["indices"])
        ]
        stats["soft_units"] = max(int(stats["soft_units"]), int(len(units)))
        for unit in units:
            if deadline is not None and time.monotonic() >= deadline:
                break
            indices = np.asarray(unit["indices"], dtype=np.int64)
            old_xy = soft_pos[indices]
            target = _routing_target(scorer_plc, hard_pos, soft_pos, indices)
            source_heat = float(np.mean(local_heat[indices]))
            canvas_diag = max(float(np.hypot(cw, ch)), 1.0e-12)
            ranked_voids = sorted(
                [void for void in void_rows if not bool(void.get("legacy", False))],
                key=lambda void: (
                    float(void["field"]) + float(void["density"]),
                    float(
                        np.linalg.norm(
                            0.5 * (np.asarray(void["rect"][:2]) + np.asarray(void["rect"][2:]))
                            - target
                        )
                        / canvas_diag
                    ),
                    -float(void["area"]),
                ),
            )[:8]
            for void in ranked_voids:
                if deadline is not None and time.monotonic() >= deadline:
                    break
                anchors = (
                    ("net", target),
                    ("center", 0.5 * (np.asarray(void["rect"][:2]) + np.asarray(void["rect"][2:]))),
                )
                for anchor_kind, anchor in anchors:
                    layouts = [
                        (
                            "rigid",
                            _translate_soft_group(
                                old_xy, soft_hw[indices], soft_hh[indices], void["rect"], anchor
                            ),
                        )
                    ]
                    if str(unit["kind"]) != "stable_bundle" and indices.size >= 2:
                        layouts.extend(
                            (
                                (
                                    f"shelf_{direction}",
                                    _shelf_pack_soft(
                                        indices,
                                        soft_hw,
                                        soft_hh,
                                        void["rect"],
                                        anchor,
                                        reverse=bool(direction),
                                    ),
                                )
                                for direction in (0, 1)
                            )
                        )
                    for layout_kind, new_xy in layouts:
                        if new_xy is None or not bool(np.all(np.isfinite(new_xy))):
                            continue
                        if _boxes_overlap(
                            new_xy, soft_hw[indices], soft_hh[indices]
                        ) or not _soft_avoids_hard(
                            new_xy, indices, soft_hw, soft_hh, hard_pos, hw, hh
                        ):
                            stats["overlap_rejects"] += 1
                            continue
                        new_ci = np.clip((new_xy[:, 0] / cell_w).astype(np.int64), 0, nc - 1)
                        new_ri = np.clip((new_xy[:, 1] / cell_h).astype(np.int64), 0, nr - 1)
                        target_heat = float(
                            np.mean(field[new_ri, new_ci] + density_field[new_ri * nc + new_ci])
                        )
                        relief = source_heat - target_heat
                        if relief < float(min_field_drop):
                            stats["field_rejects"] += 1
                            continue
                        utilization = _projected_void_utilization(
                            void["rect"],
                            soft_pos,
                            soft_hw,
                            soft_hh,
                            moved_soft=indices,
                            new_soft=new_xy,
                            new_soft_hw=soft_hw[indices],
                            new_soft_hh=soft_hh[indices],
                        )
                        if utilization > float(max_utilization):
                            stats["capacity_rejects"] += 1
                            continue
                        kind = str(unit["kind"])
                        lanes[kind].append(
                            {
                                "lane": kind,
                                "rank": (
                                    -relief,
                                    float(np.linalg.norm(np.mean(new_xy, axis=0) - target)),
                                    utilization,
                                    tuple(int(index) for index in indices),
                                    anchor_kind,
                                    layout_kind,
                                ),
                                "members": np.zeros(0, dtype=np.int64),
                                "new_hard": np.zeros((0, 2), dtype=np.float64),
                                "soft_indices": indices,
                                "new_soft": new_xy,
                                "kind": f"{kind}_{layout_kind}_{anchor_kind}_{void['kind']}_void",
                            }
                        )

        for rows in lanes.values():
            rows.sort(key=lambda row: row["rank"])
        legacy_hard = [row for row in lanes["hard_leaf"] if bool(row.get("legacy", False))]
        lanes["hard_leaf"] = [
            row for row in lanes["hard_leaf"] if not bool(row.get("legacy", False))
        ]
        expansion_rows = list(lanes["hard_expand"])
        lanes["hard_expand"] = []
        if accept_round == 0:
            proposals = expansion_rows + legacy_hard + _interleave_lanes(lanes)
        else:
            proposals = legacy_hard + expansion_rows + _interleave_lanes(lanes)
        if not proposals:
            break
        remaining = max(0, int(max_scored) - int(stats["scored"]))
        first_cap = min(remaining, 32 if accept_round == 0 else 16)
        best = None
        best_score = current_score
        best_expansion = None
        best_expansion_score = current_score
        best_expansion_shift = -1.0
        cursor = 0
        while cursor < len(proposals) and remaining > 0:
            batch_cap = min(remaining, first_cap if cursor == 0 else 16)
            for proposal in proposals[cursor : cursor + batch_cap]:
                if deadline is not None and time.monotonic() >= deadline:
                    break
                stats["candidates"] += 1
                members = np.asarray(proposal["members"], dtype=np.int64)
                new_hard = np.asarray(proposal["new_hard"], dtype=np.float64)
                soft_indices = np.asarray(proposal["soft_indices"], dtype=np.int64)
                new_soft = np.asarray(proposal["new_soft"], dtype=np.float64)
                is_expansion = str(proposal["kind"]).startswith("hard_soft_expand_")
                if members.size and not _hard_group_is_legal(hard_pos, members, new_hard, hw, hh):
                    stats["hard_legality_rejects"] += 1
                    continue
                if members.size:
                    returned_hard = hard_pos.astype(np.float32).astype(np.float64)
                    returned_targets = new_hard.astype(np.float32).astype(np.float64)
                    if not _hard_group_is_legal(
                        returned_hard,
                        members,
                        returned_targets,
                        hw,
                        hh,
                        tolerance=0.0,
                    ):
                        stats["overlap_rejects"] += 1
                        stats["hard_legality_rejects"] += 1
                        continue
                trial_hard, trial_soft = hard_pos.copy(), soft_pos.copy()
                trial_hard[members] = new_hard
                trial_soft[soft_indices] = new_soft
                stats["legal"] += 1
                if is_expansion:
                    stats["expansion_legal"] += 1
                if candidate_allowed is not None and not bool(
                    candidate_allowed(trial_hard, trial_soft)
                ):
                    stats["hierarchy_rejects"] += 1
                    if is_expansion:
                        stats["expansion_hierarchy_rejects"] += 1
                    continue
                if members.size:
                    score = float(
                        incremental_scorer.score_move_group(
                            members, new_hard, soft_indices, new_soft
                        )
                    )
                    required_gain = float(min_gain)
                else:
                    score = float(incremental_scorer.score_move_soft_group(soft_indices, new_soft))
                    required_gain = float(soft_min_gain)
                stats["scored"] += 1
                if is_expansion:
                    stats["expansion_scored"] += 1
                    stats["expansion_best_proxy_gain"] = max(
                        float(stats["expansion_best_proxy_gain"]), current_score - score
                    )
                if "graph_taper" in str(proposal["kind"]):
                    stats["graph_taper_scored"] += 1
                    stats["graph_taper_best_proxy_gain"] = max(
                        float(stats["graph_taper_best_proxy_gain"]), current_score - score
                    )
                remaining -= 1
                stats["best_proxy_gain"] = max(
                    float(stats["best_proxy_gain"]), current_score - score
                )
                if score < best_score - max(1.0e-9, required_gain):
                    best_score, best = score, proposal
                if is_expansion and score < current_score - max(1.0e-9, required_gain):
                    outward_shift = float(proposal.get("outward_shift", 0.0))
                    if outward_shift > best_expansion_shift + 1.0e-12 or (
                        abs(outward_shift - best_expansion_shift) <= 1.0e-12
                        and score < best_expansion_score
                    ):
                        best_expansion = proposal
                        best_expansion_score = score
                        best_expansion_shift = outward_shift
            cursor += batch_cap
            hard_prefix_pending = cursor < len(expansion_rows) + len(legacy_hard)
            if (best is not None and not hard_prefix_pending) or (
                deadline is not None and time.monotonic() >= deadline
            ):
                break
        if accept_round == 0 and best_expansion is not None:
            best, best_score = best_expansion, best_expansion_score
        if best is None:
            break

        members = np.asarray(best["members"], dtype=np.int64)
        new_hard = np.asarray(best["new_hard"], dtype=np.float64)
        soft_indices = np.asarray(best["soft_indices"], dtype=np.int64)
        new_soft = np.asarray(best["new_soft"], dtype=np.float64)
        if members.size:
            incremental_scorer.commit_move_group(members, new_hard, soft_indices, new_soft)
        else:
            incremental_scorer.commit_move_soft_group(soft_indices, new_soft)
        hard_pos[members] = new_hard
        soft_pos[soft_indices] = new_soft
        if location_graph is not None:
            location_graph.synchronize(hard_pos, soft_pos)
        moved_hard.update(int(index) for index in members)
        moved_soft.update(int(index) for index in soft_indices)
        current_score = float(best_score)
        stats["accepts"] += 1
        stats["accepted_cluster"] = int(best.get("cid", -1))
        stats["accepted_kind"] = str(best["kind"])
        stats["accepted_kinds"].append(str(best["kind"]))
        if str(best["kind"]).startswith("hard_soft_expand_"):
            stats["expansion_accepts"] += 1
            stats["accepted_expansion_shift"] = float(best.get("outward_shift", 0.0))
        if "graph_taper" in str(best["kind"]):
            stats["graph_taper_accepts"] += 1

    _void_cluster_relocation.last_stats = stats
    return hard_pos, soft_pos, int(stats["accepts"]), float(current_score)


_void_cluster_relocation.last_stats = {}
