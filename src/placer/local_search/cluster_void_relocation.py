"""Capacity-aware whole-cluster relocation into gaps between large macros."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence

import numpy as np

from placer.local_search.fields import weighted_congestion_field
from placer.local_search.subcluster_relocation import _hard_group_is_legal


def find_large_macro_voids(
    hard_pos: np.ndarray,
    hw: np.ndarray,
    hh: np.ndarray,
    *,
    large_area_percentile: float = 70.0,
    min_width: float = 0.0,
    min_height: float = 0.0,
    max_voids: int = 128,
) -> list[dict[str, object]]:
    """Return axis-aligned gaps bounded by opposing large hard-macro edges."""
    pos = np.asarray(hard_pos, dtype=np.float64)
    half_w = np.asarray(hw, dtype=np.float64)
    half_h = np.asarray(hh, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[0] < 2:
        return []
    areas = 4.0 * half_w * half_h
    positive = areas[areas > 0.0]
    if positive.size < 2:
        return []
    threshold = float(np.percentile(positive, np.clip(large_area_percentile, 0.0, 100.0)))
    large = np.flatnonzero(areas >= threshold - 1.0e-12)
    rows: list[dict[str, object]] = []
    seen: set[tuple[int, int, int, int]] = set()

    def _append(x0, y0, x1, y1, orientation: str, left: int, right: int) -> None:
        width, height = float(x1 - x0), float(y1 - y0)
        if width < float(min_width) or height < float(min_height):
            return
        key = tuple(int(round(value * 1.0e6)) for value in (x0, y0, x1, y1))
        if key in seen:
            return
        seen.add(key)
        rows.append(
            {
                "rect": np.asarray([x0, y0, x1, y1], dtype=np.float64),
                "orientation": orientation,
                "boundary": (int(left), int(right)),
                "area": width * height,
            }
        )

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
                _append(x0, y0, x1, y1, "horizontal", left, right)

            bottom, top = (left, right)
            if pos[bottom, 1] > pos[top, 1]:
                bottom, top = top, bottom
            y0 = pos[bottom, 1] + half_h[bottom]
            y1 = pos[top, 1] - half_h[top]
            x0 = max(pos[bottom, 0] - half_w[bottom], pos[top, 0] - half_w[top])
            x1 = min(pos[bottom, 0] + half_w[bottom], pos[top, 0] + half_w[top])
            if x1 > x0 and y1 > y0:
                _append(x0, y0, x1, y1, "vertical", bottom, top)

    rows.sort(
        key=lambda row: (
            -float(row["area"]),
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
    movable_h: np.ndarray,
    movable_soft: np.ndarray,
    candidate_allowed: Callable[[np.ndarray, np.ndarray], bool] | None = None,
    deadline: float | None = None,
    large_area_percentile: float = 70.0,
    min_gap_cells: int = 2,
    max_voids: int = 96,
    max_cluster_hard: int = 12,
    max_cluster_soft: int = 32,
    top_clusters: int = 10,
    max_utilization: float = 0.78,
    soft_compact_scale: float = 0.65,
    min_field_drop: float = 0.01,
    min_gain: float = 0.0001,
    max_scored: int = 48,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Relocate one hierarchy leaf into a capacity-safe inter-macro void."""
    stats = {
        "voids": 0,
        "eligible_clusters": 0,
        "capacity_rejects": 0,
        "field_rejects": 0,
        "candidates": 0,
        "legal": 0,
        "hierarchy_rejects": 0,
        "scored": 0,
        "accepts": 0,
        "accepted_cluster": -1,
        "accepted_kind": "none",
        "best_proxy_gain": 0.0,
    }
    _void_cluster_relocation.last_stats = stats
    nr, nc = int(benchmark.grid_rows), int(benchmark.grid_cols)
    cell_w, cell_h = float(cw) / nc, float(ch) / nr
    voids = find_large_macro_voids(
        hard_pos,
        hw,
        hh,
        large_area_percentile=large_area_percentile,
        min_width=max(1, int(min_gap_cells)) * cell_w,
        min_height=max(1, int(min_gap_cells)) * cell_h,
        max_voids=max_voids,
    )
    stats["voids"] = int(len(voids))
    if not voids:
        return hard_pos, soft_pos, 0, float(initial_score)
    field = weighted_congestion_field(incremental_scorer, nr, nc)
    if field is None:
        return hard_pos, soft_pos, 0, float(initial_score)
    field = np.asarray(field, dtype=np.float64)
    density = getattr(incremental_scorer, "grid_occupied", None)
    grid_area = getattr(incremental_scorer, "dens_grid_area", None)
    density_field = (
        np.asarray(density, dtype=np.float64) / float(grid_area)
        if density is not None and grid_area is not None and float(grid_area) > 0.0
        else np.zeros(nr * nc, dtype=np.float64)
    )
    movable_h = np.asarray(movable_h, dtype=bool)
    movable_soft = np.asarray(movable_soft, dtype=bool)
    graph_targets = _cluster_graph_targets(hard_pos, clusters, edges)

    void_rows = []
    for void in voids:
        cells = _rect_cells(void["rect"], nr, nc, cw, ch)
        if cells.size == 0:
            continue
        row = dict(void)
        row["cells"] = cells
        row["field"] = float(np.mean(field.ravel()[cells]))
        row["density"] = float(np.mean(density_field[cells]))
        void_rows.append(row)
    if not void_rows:
        return hard_pos, soft_pos, 0, float(initial_score)

    cluster_rows = []
    for cid, raw_members in sorted(clusters.items()):
        members = np.asarray(raw_members, dtype=np.int64)
        if (
            members.size < 2
            or members.size > max(2, int(max_cluster_hard))
            or not bool(np.all(movable_h[members]))
        ):
            continue
        soft_indices = np.asarray(cluster_softs.get(int(cid), ()), dtype=np.int64) - int(n)
        soft_indices = soft_indices[
            (soft_indices >= 0) & (soft_indices < soft_pos.shape[0]) & movable_soft[soft_indices]
        ]
        if soft_indices.size > max(0, int(max_cluster_soft)):
            continue
        points = hard_pos[members]
        source_cells = np.clip((points[:, 1] / cell_h).astype(np.int64), 0, nr - 1) * nc
        source_cells += np.clip((points[:, 0] / cell_w).astype(np.int64), 0, nc - 1)
        source_field = float(np.mean(field.ravel()[source_cells]))
        graph_target = graph_targets.get(int(cid), np.mean(points, axis=0))
        cluster_rows.append(
            {
                "cid": int(cid),
                "members": members,
                "soft_indices": soft_indices,
                "source_field": source_field,
                "graph_target": np.asarray(graph_target, dtype=np.float64),
            }
        )
    cluster_rows.sort(key=lambda row: (-float(row["source_field"]), int(row["cid"])))
    cluster_rows = cluster_rows[: max(0, int(top_clusters))]
    stats["eligible_clusters"] = int(len(cluster_rows))

    proposals = []
    for row in cluster_rows:
        members = row["members"]
        soft_indices = row["soft_indices"]
        old_hard = hard_pos[members]
        old_center = np.mean(old_hard, axis=0)
        rel = old_hard - old_center
        rel_x0 = float(np.min(rel[:, 0] - hw[members]))
        rel_x1 = float(np.max(rel[:, 0] + hw[members]))
        rel_y0 = float(np.min(rel[:, 1] - hh[members]))
        rel_y1 = float(np.max(rel[:, 1] + hh[members]))
        move_area = float(np.sum(4.0 * hw[members] * hh[members]))
        if soft_indices.size:
            move_area += float(np.sum(4.0 * soft_hw[soft_indices] * soft_hh[soft_indices]))
        for void in void_rows:
            if set(void["boundary"]).intersection(int(member) for member in members):
                continue
            x0, y0, x1, y1 = map(float, void["rect"])
            ax0, ax1 = x0 - rel_x0, x1 - rel_x1
            ay0, ay1 = y0 - rel_y0, y1 - rel_y1
            if ax0 > ax1 or ay0 > ay1:
                continue
            rect_area = max((x1 - x0) * (y1 - y0), 1.0e-12)
            projected_utilization = float(void["density"]) + move_area / rect_area
            if projected_utilization > float(max_utilization):
                stats["capacity_rejects"] += 1
                continue
            target = np.clip(np.asarray(row["graph_target"]), [ax0, ay0], [ax1, ay1])
            center = np.asarray([(ax0 + ax1) * 0.5, (ay0 + ay1) * 0.5])
            for anchor_kind, anchor in (("graph", target), ("center", center)):
                relief = float(row["source_field"]) - float(void["field"])
                if relief < float(min_field_drop):
                    stats["field_rejects"] += 1
                    continue
                graph_distance = float(np.linalg.norm(anchor - row["graph_target"]))
                proposals.append(
                    (
                        -relief,
                        graph_distance / max(float(np.hypot(cw, ch)), 1.0e-12),
                        projected_utilization,
                        int(row["cid"]),
                        anchor_kind,
                        row,
                        void,
                        anchor,
                    )
                )
    proposals.sort(key=lambda proposal: proposal[:5])

    best_score = float(initial_score)
    best_move = None
    for _relief, _graph_dist, _util, cid, anchor_kind, row, void, anchor in proposals:
        if int(stats["scored"]) >= max(0, int(max_scored)):
            break
        if deadline is not None and time.monotonic() >= deadline:
            break
        stats["candidates"] += 1
        members = np.asarray(row["members"], dtype=np.int64)
        soft_indices = np.asarray(row["soft_indices"], dtype=np.int64)
        old_center = np.mean(hard_pos[members], axis=0)
        new_hard = hard_pos[members] + (np.asarray(anchor) - old_center)
        if not _hard_group_is_legal(hard_pos, members, new_hard, hw, hh):
            continue
        x0, y0, x1, y1 = map(float, void["rect"])
        old_soft = soft_pos[soft_indices]
        new_soft = np.empty_like(old_soft)
        if soft_indices.size:
            if bool(
                np.any(2.0 * soft_hw[soft_indices] > x1 - x0)
                or np.any(2.0 * soft_hh[soft_indices] > y1 - y0)
            ):
                continue
            scale = float(np.clip(soft_compact_scale, 0.0, 1.0))
            new_soft[:] = np.asarray(anchor) + scale * (old_soft - old_center)
            new_soft[:, 0] = np.clip(
                new_soft[:, 0],
                x0 + soft_hw[soft_indices],
                x1 - soft_hw[soft_indices],
            )
            new_soft[:, 1] = np.clip(
                new_soft[:, 1],
                y0 + soft_hh[soft_indices],
                y1 - soft_hh[soft_indices],
            )
            if not bool(np.all(np.isfinite(new_soft))):
                continue
        trial_hard, trial_soft = hard_pos.copy(), soft_pos.copy()
        trial_hard[members] = new_hard
        trial_soft[soft_indices] = new_soft
        stats["legal"] += 1
        if candidate_allowed is not None and not bool(candidate_allowed(trial_hard, trial_soft)):
            stats["hierarchy_rejects"] += 1
            continue
        score = float(
            incremental_scorer.score_move_group(members, new_hard, soft_indices, new_soft)
        )
        stats["scored"] += 1
        stats["best_proxy_gain"] = max(
            float(stats["best_proxy_gain"]), float(initial_score) - score
        )
        if score < best_score - max(1.0e-9, float(min_gain)):
            best_score = score
            best_move = (cid, anchor_kind, members, new_hard, soft_indices, new_soft)

    if best_move is not None:
        cid, anchor_kind, members, new_hard, soft_indices, new_soft = best_move
        incremental_scorer.commit_move_group(members, new_hard, soft_indices, new_soft)
        hard_pos[members] = new_hard
        soft_pos[soft_indices] = new_soft
        stats["accepts"] = 1
        stats["accepted_cluster"] = int(cid)
        stats["accepted_kind"] = f"hard_soft_{anchor_kind}_void"

    _void_cluster_relocation.last_stats = stats
    return hard_pos, soft_pos, int(stats["accepts"]), float(best_score)


_void_cluster_relocation.last_stats = {}
