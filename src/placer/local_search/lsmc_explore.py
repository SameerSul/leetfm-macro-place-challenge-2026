"""Coldspot cluster kick used by the hierarchy path."""

from __future__ import annotations

import numpy as np

from placer.legalize.spiral import _will_legalize
from placer.local_search.gnn_trace import log_gnn_event
from placer.scoring.wirelength import _build_wl_cache
from utils import constants as const


def _bbox(xy: np.ndarray, members: np.ndarray, hw: np.ndarray, hh: np.ndarray) -> list[float]:
    if members.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        float(np.min(xy[members, 0] - hw[members])),
        float(np.min(xy[members, 1] - hh[members])),
        float(np.max(xy[members, 0] + hw[members])),
        float(np.max(xy[members, 1] + hh[members])),
    ]


def _cold_window_anchors(
    field: np.ndarray,
    nr: int,
    nc: int,
    cw: float,
    ch: float,
    win_cells: int,
    count: int,
) -> list[tuple[float, float, float]]:
    """Return diverse low-field window centers as (x, y, average field)."""
    count = max(1, int(count))
    w = int(max(1, min(win_cells, nr, nc)))
    rows, cols = nr - w + 1, nc - w + 1
    if rows <= 0 or cols <= 0:
        return [(0.5 * cw, 0.5 * ch, float(np.mean(field)))]

    integ = np.zeros((nr + 1, nc + 1), dtype=np.float64)
    integ[1:, 1:] = np.cumsum(np.cumsum(field, axis=0), axis=1)
    win_sum = (
        integ[w : w + rows, w : w + cols]
        - integ[0:rows, w : w + cols]
        - integ[w : w + rows, 0:cols]
        + integ[0:rows, 0:cols]
    )
    avg = win_sum / float(w * w)
    flat_order = np.argsort(avg.ravel())
    cell_w, cell_h = cw / nc, ch / nr
    min_sep_cells = max(1.0, 0.5 * float(w))
    anchors: list[tuple[float, float, float, float, float]] = []
    for flat in flat_order[: max(count * 12, count)]:
        rr = int(flat // cols)
        cc = int(flat % cols)
        cx_cell = cc + 0.5 * w
        cy_cell = rr + 0.5 * w
        if any(np.hypot(cx_cell - a[3], cy_cell - a[4]) < min_sep_cells for a in anchors):
            continue
        anchors.append(
            (
                float(np.clip(cx_cell * cell_w, 0.0, cw)),
                float(np.clip(cy_cell * cell_h, 0.0, ch)),
                float(avg[rr, cc]),
                float(cx_cell),
                float(cy_cell),
            )
        )
        if len(anchors) >= count:
            break
    if not anchors:
        rr, cc = np.unravel_index(int(np.argmin(avg)), avg.shape)
        anchors.append(
            (
                float((cc + 0.5 * w) * cell_w),
                float((rr + 0.5 * w) * cell_h),
                float(avg[rr, cc]),
                float(cc + 0.5 * w),
                float(rr + 0.5 * w),
            )
        )
    return [(a[0], a[1], a[2]) for a in anchors]


def _transform_offsets(offsets: np.ndarray, orientation: str) -> np.ndarray:
    """Apply a simple orientation transform to cluster-relative coordinates."""
    if orientation == "rot90":
        return np.column_stack([-offsets[:, 1], offsets[:, 0]])
    if orientation == "rot270":
        return np.column_stack([offsets[:, 1], -offsets[:, 0]])
    if orientation == "flip_x":
        return np.column_stack([-offsets[:, 0], offsets[:, 1]])
    if orientation == "flip_y":
        return np.column_stack([offsets[:, 0], -offsets[:, 1]])
    return offsets.copy()


def _fit_shape_scale(
    offsets: np.ndarray,
    half_w: np.ndarray,
    half_h: np.ndarray,
    window_microns: float,
    spread: float,
) -> float:
    """Scale relative offsets so the transformed cluster stays compact."""
    span_x = float(np.ptp(offsets[:, 0]) + 2.0 * np.max(half_w))
    span_y = float(np.ptp(offsets[:, 1]) + 2.0 * np.max(half_h))
    target = max(window_microns * max(0.05, float(spread)), 1.0)
    scale = min(1.0, target / max(span_x, span_y, 1.0))
    return float(max(0.05, scale))


def _shape_preserving_points(
    xy: np.ndarray,
    members: np.ndarray,
    hw: np.ndarray,
    hh: np.ndarray,
    cw: float,
    ch: float,
    ax: float,
    ay: float,
    win_microns: float,
    variant: str,
    rng,
) -> tuple[np.ndarray, dict]:
    """Create a compact whole-cluster placement around an anchor."""
    center = xy[members].mean(axis=0)
    offsets = xy[members] - center
    if variant == "border_compact":
        orientation = "orig"
    else:
        orientation = {
            "compact": "orig",
            "rot90": "rot90",
            "flip_x": "flip_x",
            "low_displacement": "orig",
        }.get(variant, "orig")
    offsets_t = _transform_offsets(offsets, orientation)
    spread = float(getattr(const, "HIER_COLDSPOT_COMPACT_SPREAD", 0.72))
    if variant == "low_displacement":
        spread = min(1.0, 0.90 * spread)
    scale = _fit_shape_scale(offsets_t, hw[members], hh[members], win_microns, spread)

    anchor = np.asarray([ax, ay], dtype=np.float64)
    if variant == "low_displacement":
        blend = float(getattr(const, "HIER_COLDSPOT_LOW_DISP_BLEND", 0.45))
        blend = float(np.clip(blend, 0.0, 0.95))
        anchor = (1.0 - blend) * anchor + blend * center
    elif variant == "border_compact":
        source_dir = center - anchor
        norm = float(np.hypot(source_dir[0], source_dir[1]))
        if norm <= 1e-9:
            source_dir = np.asarray([1.0, 0.0], dtype=np.float64)
        else:
            source_dir = source_dir / norm
        anchor = anchor + source_dir * (0.32 * win_microns)

    jitter = rng.normal(0.0, max(0.1, 0.01 * win_microns), offsets_t.shape)
    pts = anchor + scale * offsets_t + jitter
    pts[:, 0] = np.clip(pts[:, 0], hw[members], cw - hw[members])
    pts[:, 1] = np.clip(pts[:, 1], hh[members], ch - hh[members])
    stats = {
        "whole_variant": variant,
        "whole_orientation": orientation,
        "whole_shape_scale": float(scale),
        "whole_anchor_x": float(anchor[0]),
        "whole_anchor_y": float(anchor[1]),
    }
    return pts, stats


def _soft_shape_points(
    hard_xy: np.ndarray,
    soft_xy: np.ndarray,
    members: np.ndarray,
    s_local: np.ndarray,
    soft_hw: np.ndarray,
    soft_hh: np.ndarray,
    cw: float,
    ch: float,
    hard_after: np.ndarray,
    rng,
) -> np.ndarray:
    """Co-move soft macros by the hard-cluster affine shift approximation."""
    soft_new = soft_xy.copy()
    before_center = hard_xy[members].mean(axis=0)
    after_center = hard_after[members].mean(axis=0)
    delta = after_center - before_center
    span = max(
        float(np.ptp(hard_after[members, 0])),
        float(np.ptp(hard_after[members, 1])),
        1.0,
    )
    noise = rng.normal(0.0, max(0.1, 0.015 * span), (s_local.size, 2))
    pts = soft_xy[s_local] + delta + noise
    soft_new[s_local, 0] = np.clip(pts[:, 0], soft_hw[s_local], cw - soft_hw[s_local])
    soft_new[s_local, 1] = np.clip(pts[:, 1], soft_hh[s_local], ch - soft_hh[s_local])
    return soft_new


def _connected_cold_capacity(
    field: np.ndarray,
    nr: int,
    nc: int,
    cw: float,
    ch: float,
    ax: float,
    ay: float,
    target_density: float,
) -> tuple[float, int]:
    """Estimate usable coldspot area around the selected anchor."""
    cell_w, cell_h = cw / nc, ch / nr
    ar = int(np.clip(ay / cell_h, 0, nr - 1))
    ac = int(np.clip(ax / cell_w, 0, nc - 1))
    cold_pct = float(getattr(const, "HIER_COLDSPOT_MEMORY_COLD_PCT", 35.0))
    thresh = float(np.percentile(field, np.clip(cold_pct, 0.0, 100.0)))
    mask = np.asarray(field <= thresh, dtype=bool)
    if not mask[ar, ac]:
        # The best average cold window center may not be a percentile-cold cell.
        r0 = max(0, ar - 1)
        r1 = min(nr, ar + 2)
        c0 = max(0, ac - 1)
        c1 = min(nc, ac + 2)
        local = mask[r0:r1, c0:c1]
        if local.any():
            rr, cc = np.argwhere(local)[0]
            ar, ac = r0 + int(rr), c0 + int(cc)
        else:
            mask[ar, ac] = True

    max_dist = max(1, int(getattr(const, "HIER_COLDSPOT_ADAPTIVE_MAX_CELLS", 5)))
    seen = np.zeros((nr, nc), dtype=bool)
    seen[ar, ac] = True
    queue = [(ar, ac, 0)]
    head = 0
    while head < len(queue):
        rr, cc, dist = queue[head]
        head += 1
        if dist >= max_dist:
            continue
        for nr2, nc2 in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
            if nr2 < 0 or nr2 >= nr or nc2 < 0 or nc2 >= nc:
                continue
            if seen[nr2, nc2] or not mask[nr2, nc2]:
                continue
            seen[nr2, nc2] = True
            queue.append((nr2, nc2, dist + 1))

    cells = int(np.count_nonzero(seen))
    fill = max(0.05, float(getattr(const, "HIER_COLDSPOT_PARTIAL_FILL_FRAC", 0.75)))
    capacity = cells * cell_w * cell_h * max(target_density, 1e-3) * fill
    return float(capacity), cells


def _local_connectivity(
    plc,
    n: int,
    n_soft: int,
    hard_members: np.ndarray,
    soft_members: np.ndarray,
) -> tuple[dict[tuple[int, int], float], dict[int, float], dict[int, dict[int, float]]]:
    """Build local hard-hard and hard-soft weights in placement index space."""
    if plc is None:
        return {}, {}, {}
    try:
        cache = _build_wl_cache(plc)
    except Exception:
        return {}, {}, {}
    ref_idx = cache["ref_idx"]
    net_starts = cache["net_starts"]
    net_lengths = cache["net_lengths"]
    net_weights = cache["net_weights"]
    hard_b_to_a = {int(b): a for a, b in enumerate(plc.hard_macro_indices[:n])}
    soft_b_to_k = (
        {int(b): k for k, b in enumerate(plc.soft_macro_indices[:n_soft])} if n_soft > 0 else {}
    )
    hard_set = {int(v) for v in hard_members}
    soft_set = {int(v) for v in soft_members}
    hard_edges: dict[tuple[int, int], float] = {}
    hard_strength: dict[int, float] = {int(v): 0.0 for v in hard_members}
    soft_to_hard: dict[int, dict[int, float]] = {int(v): {} for v in soft_members}
    max_fanout = max(2, int(getattr(const, "CLUSTER_MAX_FANOUT", 8)))

    for net_i in range(len(net_starts)):
        length = int(net_lengths[net_i])
        if length < 2 or length > max_fanout:
            continue
        start = int(net_starts[net_i])
        refs = ref_idx[start : start + length]
        hard_local = sorted(
            {
                hard_b_to_a[int(r)]
                for r in refs
                if int(r) in hard_b_to_a and hard_b_to_a[int(r)] in hard_set
            }
        )
        soft_local = sorted(
            {
                soft_b_to_k[int(r)]
                for r in refs
                if int(r) in soft_b_to_k and soft_b_to_k[int(r)] in soft_set
            }
        )
        if not hard_local:
            continue
        weight = float(net_weights[net_i])
        for i, a in enumerate(hard_local):
            hard_strength[a] = hard_strength.get(a, 0.0) + weight
            for b in hard_local[i + 1 :]:
                key = (min(a, b), max(a, b))
                hard_edges[key] = hard_edges.get(key, 0.0) + weight
        for s in soft_local:
            bucket = soft_to_hard.setdefault(s, {})
            for h in hard_local:
                bucket[h] = bucket.get(h, 0.0) + weight
    return hard_edges, hard_strength, soft_to_hard


def _selected_subgraph_connected(
    selected: np.ndarray,
    hard_edges: dict[tuple[int, int], float],
) -> bool:
    """Return whether selected nodes are connected by local hard-hard edges."""
    if selected.size <= 1:
        return True
    selected_set = {int(v) for v in selected}
    adjacency = {int(v): [] for v in selected}
    edge_count = 0
    for (a, b), weight in hard_edges.items():
        if weight <= 0.0 or a not in selected_set or b not in selected_set:
            continue
        adjacency[a].append(b)
        adjacency[b].append(a)
        edge_count += 1
    if edge_count == 0:
        return False
    seen = {int(selected[0])}
    stack = [int(selected[0])]
    while stack:
        cur = stack.pop()
        for nxt in adjacency[cur]:
            if nxt in seen:
                continue
            seen.add(nxt)
            stack.append(nxt)
    return len(seen) == len(selected_set)


def _log_partial_reject(
    benchmark_name,
    cid: int,
    reason: str,
    *,
    trace: dict | None = None,
) -> None:
    """Emit a low-cost trace row for a generated-but-rejected partial split."""
    payload = {
        "benchmark": benchmark_name,
        "operator": "coldspot_tightening",
        "kind": "partial_frontier_reject",
        "cluster": int(cid),
        "partial_frontier": True,
        "accepted": False,
        "committed": False,
        "rejection_reason": str(reason),
    }
    if trace:
        payload.update(trace)
    log_gnn_event("hier_coldspot_partial_reject", **payload)


def _select_partial_frontier(
    members: np.ndarray,
    sizes: np.ndarray,
    hard_xy: np.ndarray,
    ax: float,
    ay: float,
    capacity: float,
    hard_edges: dict[tuple[int, int], float],
) -> tuple[np.ndarray, dict]:
    """Pick a capacity-bounded connected-ish frontier subset nearest the coldspot."""
    min_cluster = max(3, int(getattr(const, "HIER_COLDSPOT_PARTIAL_MIN_CLUSTER_HARD", 6)))
    if members.size < min_cluster or capacity <= 0.0:
        return np.zeros(0, dtype=np.int64), {
            "partial_reject_reason": "min_cluster_or_capacity",
            "partial_member_count": int(members.size),
            "partial_capacity": float(capacity),
        }
    area = sizes[members, 0] * sizes[members, 1]
    min_hard = max(2, int(getattr(const, "HIER_COLDSPOT_PARTIAL_MIN_HARD", 2)))
    if area.size < min_hard:
        return np.zeros(0, dtype=np.int64), {
            "partial_reject_reason": "min_hard",
            "partial_member_count": int(members.size),
        }
    dist = np.hypot(hard_xy[members, 0] - ax, hard_xy[members, 1] - ay)
    max_dist = max(float(dist.max()), 1e-9)
    max_area = max(float(area.max()), 1e-9)
    order = [int(members[i]) for i in np.argsort(dist)]
    selected: list[int] = [order[0]]
    selected_area = float(sizes[order[0], 0] * sizes[order[0], 1])
    overshoot = 1.15
    min_remaining = max(1, int(getattr(const, "HIER_COLDSPOT_PARTIAL_MIN_REMAINING_HARD", 3)))
    max_member_frac = float(getattr(const, "HIER_COLDSPOT_PARTIAL_MAX_MEMBER_FRAC", 0.50))
    max_selected = max(min_hard, int(np.floor(max_member_frac * members.size)))
    max_selected = min(max_selected, max(1, int(members.size) - min_remaining))

    while len(selected) < len(order):
        if len(selected) >= max_selected:
            break
        if len(selected) >= min_hard and selected_area >= capacity:
            break
        best = None
        selected_set = set(selected)
        for h in order:
            if h in selected_set:
                continue
            h_area = float(sizes[h, 0] * sizes[h, 1])
            if len(selected) >= min_hard and selected_area + h_area > capacity * overshoot:
                continue
            conn = sum(hard_edges.get((min(h, s), max(h, s)), 0.0) for s in selected)
            idx = int(np.where(members == h)[0][0])
            score = (
                float(dist[idx] / max_dist)
                + 0.25 * float(h_area / max_area)
                - 0.20 * np.log1p(conn)
            )
            row = (score, float(dist[idx]), h)
            if best is None or row < best:
                best = row
        if best is None:
            break
        h = int(best[2])
        selected.append(h)
        selected_area += float(sizes[h, 0] * sizes[h, 1])

    if len(selected) < min_hard or len(selected) >= members.size:
        return np.zeros(0, dtype=np.int64), {
            "partial_reject_reason": "selected_size",
            "partial_selected_hard": int(len(selected)),
            "partial_member_count": int(members.size),
        }

    selected_arr = np.asarray(sorted(selected), dtype=np.int64)
    selected_set = {int(v) for v in selected_arr}
    remaining = [int(v) for v in members if int(v) not in selected_set]
    if len(remaining) < min_remaining:
        return np.zeros(0, dtype=np.int64), {
            "partial_reject_reason": "min_remaining",
            "partial_selected_hard": int(selected_arr.size),
            "partial_remaining_hard": int(len(remaining)),
            "partial_member_count": int(members.size),
        }
    member_frac = float(selected_arr.size / max(1, members.size))
    if member_frac > max_member_frac:
        return np.zeros(0, dtype=np.int64), {
            "partial_reject_reason": "member_frac",
            "partial_selected_hard": int(selected_arr.size),
            "partial_member_count": int(members.size),
            "partial_member_frac": float(member_frac),
        }
    if hard_edges and not _selected_subgraph_connected(selected_arr, hard_edges):
        return np.zeros(0, dtype=np.int64), {
            "partial_reject_reason": "selected_disconnected",
            "partial_selected_hard": int(selected_arr.size),
            "partial_member_count": int(members.size),
        }
    internal = 0.0
    cut = 0.0
    for (a, b), w in hard_edges.items():
        a_sel = a in selected_set
        b_sel = b in selected_set
        if a_sel and b_sel:
            internal += float(w)
        elif a_sel != b_sel and (a in selected_set or b in selected_set):
            cut += float(w)
    cut_ratio = cut / max(cut + internal, 1e-12)
    max_cut = float(getattr(const, "HIER_COLDSPOT_PARTIAL_MAX_CUT_RATIO", 0.85))
    if internal + cut > 0.0 and cut_ratio > max_cut:
        return np.zeros(0, dtype=np.int64), {
            "partial_reject_reason": "cut_ratio",
            "partial_cut_ratio": float(cut_ratio),
        }
    return selected_arr, {
        "partial_selected_area": float(selected_area),
        "partial_capacity": float(capacity),
        "partial_cut_weight": float(cut),
        "partial_internal_weight": float(internal),
        "partial_cut_ratio": float(cut_ratio),
        "partial_remaining_hard": int(len(remaining)),
    }


def _place_frontier_subset(
    hard_xy: np.ndarray,
    members: np.ndarray,
    selected: np.ndarray,
    sizes: np.ndarray,
    hw: np.ndarray,
    hh: np.ndarray,
    cw: float,
    ch: float,
    ax: float,
    ay: float,
    win_microns: float,
    hard_edges: dict[tuple[int, int], float],
    rng,
) -> np.ndarray:
    """Place cross-cut-heavy moved macros near the source-facing coldspot border."""
    out = hard_xy.copy()
    if selected.size == 0:
        return out
    cm_before = hard_xy[members].mean(axis=0)
    toward_source = cm_before - np.asarray([ax, ay], dtype=np.float64)
    norm = float(np.hypot(toward_source[0], toward_source[1]))
    if norm <= 1e-9:
        toward_source = np.asarray([1.0, 0.0], dtype=np.float64)
    else:
        toward_source = toward_source / norm
    tangent = np.asarray([-toward_source[1], toward_source[0]], dtype=np.float64)
    selected_set = {int(v) for v in selected}
    remaining = [int(v) for v in members if int(v) not in selected_set]

    def _cross_weight(h: int) -> float:
        return sum(hard_edges.get((min(h, r), max(h, r)), 0.0) for r in remaining)

    ordered = sorted((int(v) for v in selected), key=lambda h: (-_cross_weight(h), h))
    span = max(win_microns, float(np.max(sizes[selected, 0])), float(np.max(sizes[selected, 1])))
    denom = max(1, len(ordered) - 1)
    for rank, h in enumerate(ordered):
        frac = rank / denom if denom else 0.0
        depth = (0.35 - 0.70 * frac) * span
        side = ((rank % 3) - 1) * 0.22 * span
        jitter = rng.normal(0.0, max(1.0, 0.05 * span), 2)
        pt = (
            np.asarray([ax, ay], dtype=np.float64) + toward_source * depth + tangent * side + jitter
        )
        out[h, 0] = np.clip(pt[0], hw[h], cw - hw[h])
        out[h, 1] = np.clip(pt[1], hh[h], ch - hh[h])
    return out


def _partial_split_quality_prediction(
    before_xy: np.ndarray,
    after_xy: np.ndarray,
    members: np.ndarray,
    selected: np.ndarray,
) -> tuple[bool, dict]:
    """Cheaply reject partial kicks that visibly stretch a hierarchy cluster."""
    selected_set = {int(v) for v in selected}
    remaining = np.asarray([int(v) for v in members if int(v) not in selected_set], dtype=np.int64)
    if selected.size == 0 or remaining.size == 0:
        return False, {"partial_pred_reject_reason": "empty_split"}

    before = before_xy[members]
    after = after_xy[members]
    before_centroid = before.mean(axis=0)
    after_centroid = after.mean(axis=0)
    before_radius = float(
        np.mean(np.hypot(before[:, 0] - before_centroid[0], before[:, 1] - before_centroid[1]))
    )
    after_radius = float(
        np.mean(np.hypot(after[:, 0] - after_centroid[0], after[:, 1] - after_centroid[1]))
    )
    before_bbox = 0.5 * float(np.hypot(np.ptp(before[:, 0]), np.ptp(before[:, 1])))
    after_bbox = 0.5 * float(np.hypot(np.ptp(after[:, 0]), np.ptp(after[:, 1])))
    base_span = max(before_radius, before_bbox, 1.0)
    radius_ratio = after_radius / max(before_radius, 1.0)
    bbox_ratio = after_bbox / max(before_bbox, 1.0)
    sep = float(
        np.hypot(
            after_xy[selected, 0].mean() - after_xy[remaining, 0].mean(),
            after_xy[selected, 1].mean() - after_xy[remaining, 1].mean(),
        )
    )
    sep_ratio = sep / base_span
    stats = {
        "partial_pred_radius_before": float(before_radius),
        "partial_pred_radius_after": float(after_radius),
        "partial_pred_radius_ratio": float(radius_ratio),
        "partial_pred_bbox_before": float(before_bbox),
        "partial_pred_bbox_after": float(after_bbox),
        "partial_pred_bbox_ratio": float(bbox_ratio),
        "partial_pred_separation": float(sep),
        "partial_pred_separation_ratio": float(sep_ratio),
    }
    max_radius = float(getattr(const, "HIER_COLDSPOT_PARTIAL_MAX_RADIUS_RATIO", 1.15))
    max_bbox = float(getattr(const, "HIER_COLDSPOT_PARTIAL_MAX_BBOX_RATIO", 1.20))
    max_sep = float(getattr(const, "HIER_COLDSPOT_PARTIAL_MAX_SEPARATION_RATIO", 1.50))
    if radius_ratio > max_radius:
        stats["partial_pred_reject_reason"] = "radius_ratio"
        return False, stats
    if bbox_ratio > max_bbox:
        stats["partial_pred_reject_reason"] = "bbox_ratio"
        return False, stats
    if sep_ratio > max_sep:
        stats["partial_pred_reject_reason"] = "separation_ratio"
        return False, stats
    return True, stats


def _partial_frontier_candidate(
    hard_xy,
    sizes,
    hw,
    hh,
    cw,
    ch,
    movable,
    n,
    members,
    s_local,
    soft_xy,
    soft_hw,
    soft_hh,
    cong_field,
    nr,
    nc,
    ax,
    ay,
    target_density,
    rng,
    deadline,
    plc,
    cid: int,
    benchmark_name=None,
) -> "tuple[np.ndarray, np.ndarray | None, dict] | None":
    """Generate one capacity-aware partial coldspot frontier kick."""
    n_soft = 0 if soft_xy is None else int(soft_xy.shape[0])
    soft_members = (
        np.zeros(0, dtype=np.int64) if s_local is None else np.asarray(s_local, dtype=np.int64)
    )
    capacity, cold_cells = _connected_cold_capacity(
        cong_field, nr, nc, cw, ch, ax, ay, target_density
    )
    cluster_area = float(np.sum(sizes[members, 0] * sizes[members, 1]))
    max_frac = float(getattr(const, "HIER_COLDSPOT_PARTIAL_MAX_AREA_FRAC", 0.55))
    capacity = min(capacity, cluster_area * max(0.05, min(max_frac, 0.95)))
    hard_edges, _hard_strength, soft_to_hard = _local_connectivity(
        plc, n, n_soft, members, soft_members
    )
    selected, stats = _select_partial_frontier(
        members, sizes, hard_xy, ax, ay, capacity, hard_edges
    )
    if selected.size == 0:
        _log_partial_reject(
            benchmark_name,
            int(cid),
            str(stats.get("partial_reject_reason", "selector")),
            trace={
                "partial_capacity": float(capacity),
                "partial_cold_cells": int(cold_cells),
                "member_count": int(members.size),
                **stats,
            },
        )
        return None
    moved_area = float(np.sum(sizes[selected, 0] * sizes[selected, 1]))
    win_microns = float(np.sqrt(max(moved_area, 1.0) / max(target_density, 1e-3)))
    kicked = _place_frontier_subset(
        hard_xy,
        members,
        selected,
        sizes,
        hw,
        hh,
        cw,
        ch,
        ax,
        ay,
        win_microns,
        hard_edges,
        rng,
    )
    if not bool(np.any(kicked[selected] != hard_xy[selected])):
        _log_partial_reject(
            benchmark_name,
            int(cid),
            "no_movement",
            trace={
                "partial_moved_hard": int(selected.size),
                "partial_selected_area": float(moved_area),
                "partial_capacity": float(capacity),
                "partial_cold_cells": int(cold_cells),
                "member_count": int(members.size),
                **stats,
            },
        )
        return None
    legal_hard = _will_legalize(
        kicked,
        movable,
        sizes,
        hw,
        hh,
        cw,
        ch,
        n,
        deadline=deadline,
    )
    split_ok, split_stats = _partial_split_quality_prediction(
        hard_xy,
        legal_hard,
        members,
        selected,
    )
    if not split_ok:
        _log_partial_reject(
            benchmark_name,
            int(cid),
            str(split_stats.get("partial_pred_reject_reason", "split_quality")),
            trace={
                "partial_moved_hard": int(selected.size),
                "partial_selected_area": float(moved_area),
                "partial_capacity": float(capacity),
                "partial_cold_cells": int(cold_cells),
                "member_count": int(members.size),
                **stats,
                **split_stats,
            },
        )
        return None

    soft_new = None
    soft_moved = 0
    soft_disp = np.zeros(0, dtype=np.float64)
    selected_set = {int(v) for v in selected}
    selected_soft = np.zeros(0, dtype=np.int64)
    if soft_members.size and soft_xy is not None:
        soft_scores = []
        remaining_capacity = max(0.0, capacity - moved_area)
        for s in soft_members:
            links = soft_to_hard.get(int(s), {})
            strength = sum(float(w) for h, w in links.items() if int(h) in selected_set)
            if strength <= 0.0:
                continue
            area = float((2.0 * soft_hw[int(s)]) * (2.0 * soft_hh[int(s)]))
            soft_scores.append((-strength, area, int(s)))
        soft_scores.sort()
        chosen = []
        used = 0.0
        for _neg_strength, area, s in soft_scores:
            if chosen and used + area > remaining_capacity * 1.15:
                continue
            if not chosen or used + area <= max(remaining_capacity * 1.15, area):
                chosen.append(int(s))
                used += float(area)
        if chosen:
            selected_soft = np.asarray(chosen, dtype=np.int64)
            soft_new = soft_xy.copy()
            cm_before = hard_xy[members].mean(axis=0)
            toward_source = cm_before - np.asarray([ax, ay], dtype=np.float64)
            norm = float(np.hypot(toward_source[0], toward_source[1]))
            if norm <= 1e-9:
                toward_source = np.asarray([1.0, 0.0], dtype=np.float64)
            else:
                toward_source = toward_source / norm
            tangent = np.asarray([-toward_source[1], toward_source[0]], dtype=np.float64)
            span = max(win_microns, float(np.max(soft_hw[selected_soft] + soft_hh[selected_soft])))
            for rank, s in enumerate(selected_soft):
                frac = (rank + 0.5) / max(1, selected_soft.size)
                pt = (
                    np.asarray([ax, ay], dtype=np.float64)
                    + toward_source * ((0.18 - 0.36 * frac) * span)
                    + tangent * (((rank % 2) * 2 - 1) * 0.16 * span)
                    + rng.normal(0.0, max(1.0, 0.04 * span), 2)
                )
                soft_new[s, 0] = np.clip(pt[0], soft_hw[s], cw - soft_hw[s])
                soft_new[s, 1] = np.clip(pt[1], soft_hh[s], ch - soft_hh[s])
            soft_delta = soft_new[selected_soft] - soft_xy[selected_soft]
            soft_disp = np.hypot(soft_delta[:, 0], soft_delta[:, 1])
            soft_moved = int(np.count_nonzero(np.any(soft_delta != 0.0, axis=1)))

    hard_delta = legal_hard[selected] - hard_xy[selected]
    hard_disp = np.hypot(hard_delta[:, 0], hard_delta[:, 1])
    cell_w, cell_h = cw / nc, ch / nr
    mcol_before = np.clip((hard_xy[selected, 0] / cell_w).astype(np.int64), 0, nc - 1)
    mrow_before = np.clip((hard_xy[selected, 1] / cell_h).astype(np.int64), 0, nr - 1)
    mcol_after = np.clip((legal_hard[selected, 0] / cell_w).astype(np.int64), 0, nc - 1)
    mrow_after = np.clip((legal_hard[selected, 1] / cell_h).astype(np.int64), 0, nr - 1)
    cm_before = hard_xy[selected].mean(axis=0)
    cm_after = legal_hard[selected].mean(axis=0)
    trace = {
        "partial_frontier": True,
        "partial_moved_hard": int(selected.size),
        "partial_moved_soft": int(selected_soft.size),
        "partial_cold_cells": int(cold_cells),
        "member_count": int(members.size),
        "movable_count": int(selected.size),
        "member_area": float(moved_area),
        "source_field": float(cong_field[mrow_before, mcol_before].mean()),
        "target_field": float(cong_field[mrow_after, mcol_after].mean()),
        "score": float(cong_field[mrow_after, mcol_after].mean())
        - float(cong_field[mrow_before, mcol_before].mean()),
        "anchor_x": float(ax),
        "anchor_y": float(ay),
        "x": float(ax),
        "y": float(ay),
        "window_microns": float(win_microns),
        "window_cells": int(max(1, int(np.ceil(win_microns / min(cell_w, cell_h))))),
        "target_density": float(target_density),
        "soft_count": int(soft_members.size),
        "soft_moved": int(soft_moved),
        "hard_disp_mean": float(hard_disp.mean()) if hard_disp.size else 0.0,
        "hard_disp_max": float(hard_disp.max()) if hard_disp.size else 0.0,
        "hard_dx_mean": float(hard_delta[:, 0].mean()) if hard_delta.size else 0.0,
        "hard_dy_mean": float(hard_delta[:, 1].mean()) if hard_delta.size else 0.0,
        "soft_disp_mean": float(soft_disp.mean()) if soft_disp.size else 0.0,
        "soft_disp_max": float(soft_disp.max()) if soft_disp.size else 0.0,
        "cluster_cx_before": float(cm_before[0]),
        "cluster_cy_before": float(cm_before[1]),
        "cluster_cx_after": float(cm_after[0]),
        "cluster_cy_after": float(cm_after[1]),
        "cluster_bbox_before": _bbox(hard_xy, selected, hw, hh),
        "cluster_bbox_after": _bbox(legal_hard, selected, hw, hh),
        **stats,
        **split_stats,
    }
    return legal_hard, soft_new, trace


def _coldspot_cluster_kick(
    hard_xy,
    sizes,
    hw,
    hh,
    cw,
    ch,
    movable,
    n,
    clusters,
    cluster_softs,
    soft_xy,
    soft_hw,
    soft_hh,
    soft_movable,
    cong_field,
    nr,
    nc,
    rng,
    deadline,
    target_density=0.65,
    pick="hot",
    max_size=64,
    return_trace: bool = False,
) -> "tuple[np.ndarray, np.ndarray | None] | tuple[np.ndarray, np.ndarray | None, dict] | None":
    """Gather one cluster into a low-congestion window, then legalize hard macros."""
    candidates = _coldspot_cluster_kick_candidates(
        hard_xy,
        sizes,
        hw,
        hh,
        cw,
        ch,
        movable,
        n,
        clusters,
        cluster_softs,
        soft_xy,
        soft_hw,
        soft_hh,
        soft_movable,
        cong_field,
        nr,
        nc,
        rng,
        deadline,
        target_density=target_density,
        pick=pick,
        max_size=max_size,
        kick_count=1,
    )
    if not candidates:
        return None
    hard, soft, trace = candidates[0]
    if return_trace:
        return hard, soft, trace
    return hard, soft


def _coldspot_cluster_kick_candidates(
    hard_xy,
    sizes,
    hw,
    hh,
    cw,
    ch,
    movable,
    n,
    clusters,
    cluster_softs,
    soft_xy,
    soft_hw,
    soft_hh,
    soft_movable,
    cong_field,
    nr,
    nc,
    rng,
    deadline,
    target_density=0.65,
    pick="hot",
    max_size=64,
    kick_count: int = 8,
    plc=None,
    benchmark_name=None,
    preferred_cluster_ids=None,
    max_clusters: int = 1,
) -> list[tuple[np.ndarray, np.ndarray | None, dict]]:
    """Generate legalized coldspot kick candidates for selected clusters/windows."""
    if not clusters or cong_field is None:
        return []
    cell_w, cell_h = cw / nc, ch / nr
    mcol = np.clip((hard_xy[:, 0] / cell_w).astype(np.int64), 0, nc - 1)
    mrow = np.clip((hard_xy[:, 1] / cell_h).astype(np.int64), 0, nr - 1)
    macro_cong = cong_field[mrow, mcol]

    cids: list[int] = []
    cluster_members: dict[int, np.ndarray] = {}
    cluster_members_movable: dict[int, np.ndarray] = {}
    for cid, mem in clusters.items():
        mem_all = np.asarray(mem, dtype=np.int64)
        if mem_all.size == 0:
            continue
        cid_i = int(cid)
        cids.append(cid_i)
        cluster_members[cid_i] = mem_all
        cluster_members_movable[cid_i] = mem_all[movable[mem_all]]

    if not cids:
        return []

    heat = np.array([float(macro_cong[cluster_members[c]].mean()) for c in cids])
    preferred_ids: list[int] = []
    if preferred_cluster_ids is not None:
        preferred_ids = [
            int(v)
            for v in np.asarray(preferred_cluster_ids, dtype=np.int64).ravel()
            if int(v) in cluster_members
        ]
    if preferred_ids:
        preferred_set = set(preferred_ids)
        tail = [cids[i] for i in np.argsort(-heat) if cids[i] not in preferred_set]
        order = preferred_ids + tail
    elif pick == "hot":
        order = [cids[i] for i in np.argsort(-heat)]
    else:
        med = float(np.median(heat))
        hot = [cids[i] for i in range(len(cids)) if heat[i] >= med]
        rest = [cids[i] for i in range(len(cids)) if heat[i] < med]
        rng.shuffle(hot)
        rng.shuffle(rest)
        order = hot + rest

    cluster_soft_local: dict[int, np.ndarray] = {}
    if cluster_softs is not None and soft_xy is not None:
        soft_mask = None if soft_movable is None else np.asarray(soft_movable, dtype=np.bool_)
        for cid, s_arr in cluster_softs.items():
            cid_i = int(cid)
            if cid_i not in cluster_members:
                continue
            s_local = np.asarray(s_arr, dtype=np.int64) - n
            s_local = s_local[(s_local >= 0) & (s_local < soft_xy.shape[0])]
            if soft_mask is not None and s_local.size:
                s_local = s_local[soft_mask[s_local]]
            if s_local.size:
                cluster_soft_local[cid_i] = s_local

    all_out: list[tuple[np.ndarray, np.ndarray | None, dict]] = []
    clusters_used = 0
    anchor_cache: dict[int, list[tuple[float, float, float]]] = {}
    for cid in order:
        if clusters_used >= max(1, int(max_clusters)):
            break
        members = cluster_members_movable[cid]
        if members.size < 2 or members.size > max_size:
            continue
        member_all = cluster_members[cid]
        member_area = float(np.sum(sizes[members, 0] * sizes[members, 1]))
        win_microns = float(np.sqrt(member_area / max(target_density, 1e-3)))
        win_cells = max(1, int(np.ceil(win_microns / min(cell_w, cell_h))))
        anchor_count = max(1, int(getattr(const, "HIER_COLDSPOT_ANCHOR_VARIANTS", 3)))
        cached = anchor_cache.get(win_cells)
        if cached is None or len(cached) < anchor_count:
            cached = _cold_window_anchors(cong_field, nr, nc, cw, ch, win_cells, anchor_count)
            anchor_cache[win_cells] = cached
        anchors = cached[:anchor_count]
        ax, ay = float(anchors[0][0]), float(anchors[0][1])
        jx = max(float(hw[members].max()), win_microns / 4.0)
        jy = max(float(hh[members].max()), win_microns / 4.0)

        s_local = cluster_soft_local.get(cid)
        before_bbox = _bbox(hard_xy, members, hw, hh)
        out: list[tuple[np.ndarray, np.ndarray | None, dict]] = []
        attempts = max(1, int(kick_count))
        for candidate_rank in range(attempts):
            anchor_rank = int(candidate_rank % max(1, len(anchors)))
            cand_ax, cand_ay, anchor_field = anchors[anchor_rank]
            variant_cycle = (
                "gaussian_gather",
                "compact",
                "border_compact",
                "low_displacement",
                "rot90",
                "flip_x",
            )
            variant = variant_cycle[candidate_rank % len(variant_cycle)]
            kicked = hard_xy.copy()
            variant_stats = {
                "whole_variant": variant,
                "whole_anchor_rank": int(anchor_rank),
                "whole_anchor_field": float(anchor_field),
            }
            if variant == "gaussian_gather":
                kicked[members, 0] = np.clip(
                    cand_ax + rng.normal(0.0, jx, members.size),
                    hw[members],
                    cw - hw[members],
                )
                kicked[members, 1] = np.clip(
                    cand_ay + rng.normal(0.0, jy, members.size),
                    hh[members],
                    ch - hh[members],
                )
                variant_stats.update(
                    {
                        "whole_orientation": "random",
                        "whole_shape_scale": 0.0,
                        "whole_anchor_x": float(cand_ax),
                        "whole_anchor_y": float(cand_ay),
                    }
                )
            else:
                pts, shape_stats = _shape_preserving_points(
                    hard_xy,
                    members,
                    hw,
                    hh,
                    cw,
                    ch,
                    cand_ax,
                    cand_ay,
                    win_microns,
                    variant,
                    rng,
                )
                kicked[members] = pts
                variant_stats.update(shape_stats)
            moved_hard = bool(np.any(kicked[members] != hard_xy[members]))

            soft_new = None
            soft_moved = 0
            soft_disp = np.zeros(0, dtype=np.float64)
            if s_local is not None and soft_xy is not None and s_local.size:
                if variant == "gaussian_gather":
                    soft_new = soft_xy.copy()
                    sx = cand_ax + rng.normal(0.0, jx, s_local.size)
                    sy = cand_ay + rng.normal(0.0, jy, s_local.size)
                    soft_new[s_local, 0] = np.clip(sx, soft_hw[s_local], cw - soft_hw[s_local])
                    soft_new[s_local, 1] = np.clip(sy, soft_hh[s_local], ch - soft_hh[s_local])
                else:
                    soft_new = _soft_shape_points(
                        hard_xy,
                        soft_xy,
                        members,
                        s_local,
                        soft_hw,
                        soft_hh,
                        cw,
                        ch,
                        kicked,
                        rng,
                    )
                soft_delta = soft_new[s_local] - soft_xy[s_local]
                soft_disp = np.hypot(soft_delta[:, 0], soft_delta[:, 1])
                soft_moved = int(np.count_nonzero(np.any(soft_delta != 0.0, axis=1)))

            if not moved_hard and soft_moved == 0:
                continue

            legal_hard = hard_xy.copy()
            if moved_hard:
                legal_hard = _will_legalize(
                    kicked,
                    movable,
                    sizes,
                    hw,
                    hh,
                    cw,
                    ch,
                    n,
                    deadline=deadline,
                )
            hard_delta = legal_hard[members] - hard_xy[members]
            hard_disp = np.hypot(hard_delta[:, 0], hard_delta[:, 1])
            cm_before = hard_xy[members].mean(axis=0)
            cm_after = legal_hard[members].mean(axis=0)
            mcol_after = np.clip((legal_hard[members, 0] / cell_w).astype(np.int64), 0, nc - 1)
            mrow_after = np.clip((legal_hard[members, 1] / cell_h).astype(np.int64), 0, nr - 1)
            source_field = float(macro_cong[members].mean())
            target_field = float(cong_field[mrow_after, mcol_after].mean())
            trace = {
                "cluster": int(cid),
                "candidate_rank": int(candidate_rank),
                "member_count": int(member_all.size),
                "movable_count": int(members.size),
                "member_area": float(member_area),
                "cluster_heat": source_field,
                "source_field": source_field,
                "target_field": target_field,
                "score": target_field - source_field,
                "anchor_x": float(cand_ax),
                "anchor_y": float(cand_ay),
                "x": float(cand_ax),
                "y": float(cand_ay),
                "window_microns": float(win_microns),
                "window_cells": int(win_cells),
                "target_density": float(target_density),
                "pick": str(pick),
                "soft_count": int(0 if s_local is None else s_local.size),
                "soft_moved": int(soft_moved if soft_new is not None else 0),
                "hard_disp_mean": float(hard_disp.mean()) if hard_disp.size else 0.0,
                "hard_disp_max": float(hard_disp.max()) if hard_disp.size else 0.0,
                "hard_dx_mean": float(hard_delta[:, 0].mean()) if hard_delta.size else 0.0,
                "hard_dy_mean": float(hard_delta[:, 1].mean()) if hard_delta.size else 0.0,
                "soft_disp_mean": float(soft_disp.mean()) if soft_disp.size else 0.0,
                "soft_disp_max": float(soft_disp.max()) if soft_disp.size else 0.0,
                "cluster_cx_before": float(cm_before[0]),
                "cluster_cy_before": float(cm_before[1]),
                "cluster_cx_after": float(cm_after[0]),
                "cluster_cy_after": float(cm_after[1]),
                "cluster_bbox_before": before_bbox,
                "cluster_bbox_after": _bbox(legal_hard, members, hw, hh),
                **variant_stats,
            }
            out.append((legal_hard, soft_new, trace))
        if (
            bool(getattr(const, "HIER_COLDSPOT_PARTIAL_FRONTIER", False))
            and plc is not None
            and out
        ):
            partial_count = max(0, int(getattr(const, "HIER_COLDSPOT_PARTIAL_CANDIDATES", 1)))
            for partial_rank in range(partial_count):
                partial = _partial_frontier_candidate(
                    hard_xy,
                    sizes,
                    hw,
                    hh,
                    cw,
                    ch,
                    movable,
                    n,
                    members,
                    s_local,
                    soft_xy,
                    soft_hw,
                    soft_hh,
                    cong_field,
                    nr,
                    nc,
                    ax,
                    ay,
                    target_density,
                    rng,
                    deadline,
                    plc,
                    int(cid),
                    benchmark_name,
                )
                if partial is None:
                    continue
                ph, ps, ptrace = partial
                ptrace["cluster"] = int(cid)
                ptrace["candidate_rank"] = int(attempts + partial_rank)
                ptrace["cluster_heat"] = float(ptrace.get("source_field", 0.0))
                ptrace["pick"] = str(pick)
                out.append((ph, ps, ptrace))
        if out:
            all_out.extend(out)
            clusters_used += 1
    return all_out
