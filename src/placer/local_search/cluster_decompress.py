"""Exact-gated cluster decompression for congestion relief."""

from __future__ import annotations

import os
import time

import numpy as np
import torch

from utils import constants as const
from placer.legalize.spiral import _will_legalize
from placer.local_search.fields import _congestion_field, cold_connected_components
from placer.local_search.graph_tension import candidate_graph_edge_delta
from placer.scoring.exact import _exact_proxy


def hierarchy_quality_metric(hard_xy, clusters) -> float:
    """Cluster separation quality: lower means better-contained hierarchy."""
    return float(hierarchy_quality_breakdown(hard_xy, clusters)["quality"])


def hierarchy_quality_breakdown(hard_xy, clusters) -> dict[str, float]:
    """Composite hierarchy quality for local accept gates.

    The original score was mean cluster radius divided by nearest-cluster
    centroid distance. The composite keeps that term as the dominant signal and
    adds bbox spread plus a small crowding penalty so elongated or nearly
    touching hierarchy groups are not treated as equally good.
    """
    if not clusters:
        return {"quality": 0.0, "radius": 0.0, "bbox": 0.0, "crowd": 0.0}
    cids = list(clusters.keys())
    centroids = []
    radii = []
    bbox_radii = []
    for cid in cids:
        p = hard_xy[np.asarray(clusters[cid], dtype=np.int64)]
        centroids.append(p.mean(axis=0))
        radii.append(float(np.mean(np.hypot(p[:, 0] - p[:, 0].mean(), p[:, 1] - p[:, 1].mean()))))
        bbox_radii.append(0.5 * float(np.hypot(np.ptp(p[:, 0]), np.ptp(p[:, 1]))))
    centroids = np.asarray(centroids, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    bbox_radii = np.asarray(bbox_radii, dtype=np.float64)
    rw = max(0.0, float(const.HIER_QUALITY_RADIUS_WEIGHT))
    bw = max(0.0, float(const.HIER_QUALITY_BBOX_WEIGHT))
    cw = max(0.0, float(const.HIER_QUALITY_CROWD_WEIGHT))
    wsum = max(rw + bw + cw, 1e-12)
    if len(cids) == 1:
        denom = max(float(np.hypot(np.ptp(hard_xy[:, 0]), np.ptp(hard_xy[:, 1]))), 1.0)
        radius_score = float(radii[0] / denom)
        bbox_score = float(bbox_radii[0] / denom)
        crowd_score = 0.0
        quality = (rw * radius_score + bw * bbox_score + cw * crowd_score) / wsum
        return {
            "quality": float(quality),
            "radius": radius_score,
            "bbox": bbox_score,
            "crowd": crowd_score,
        }
    d = np.hypot(
        centroids[:, None, 0] - centroids[None, :, 0],
        centroids[:, None, 1] - centroids[None, :, 1],
    )
    np.fill_diagonal(d, np.inf)
    nearest = np.maximum(np.min(d, axis=1), 1.0)
    radius_score = float(np.mean(radii / nearest))
    bbox_score = float(np.mean(bbox_radii / nearest))
    nearest_idx = np.argmin(d, axis=1)
    combined = radii + radii[nearest_idx]
    crowd_score = float(np.mean(np.maximum(0.0, combined / nearest - 1.0)))
    quality = (rw * radius_score + bw * bbox_score + cw * crowd_score) / wsum
    return {
        "quality": float(quality),
        "radius": radius_score,
        "bbox": bbox_score,
        "crowd": crowd_score,
    }


def _cell_values(pos: np.ndarray, field: np.ndarray, cw: float, ch: float) -> np.ndarray:
    nr, nc = field.shape
    cell_w, cell_h = cw / nc, ch / nr
    ci = np.clip((pos[:, 0] / cell_w).astype(np.int64), 0, nc - 1)
    ri = np.clip((pos[:, 1] / cell_h).astype(np.int64), 0, nr - 1)
    return field[ri, ci]


def _axis_room_scale(
    hard_xy: np.ndarray,
    members: np.ndarray,
    field: np.ndarray,
    cw: float,
    ch: float,
    band: int = 3,
) -> tuple[float, float]:
    """Return x/y expansion weights based on colder neighboring grid bands."""
    nr, nc = field.shape
    cell_w, cell_h = cw / nc, ch / nr
    x0, x1 = float(hard_xy[members, 0].min()), float(hard_xy[members, 0].max())
    y0, y1 = float(hard_xy[members, 1].min()), float(hard_xy[members, 1].max())
    c0 = int(np.clip(np.floor(x0 / cell_w), 0, nc - 1))
    c1 = int(np.clip(np.floor(x1 / cell_w), 0, nc - 1))
    r0 = int(np.clip(np.floor(y0 / cell_h), 0, nr - 1))
    r1 = int(np.clip(np.floor(y1 / cell_h), 0, nr - 1))

    def _mean(vals):
        return float(vals.mean()) if vals.size else float("inf")

    left = _mean(field[max(0, r0) : min(nr, r1 + 1), max(0, c0 - band) : max(0, c0)])
    right = _mean(
        field[
            max(0, r0) : min(nr, r1 + 1),
            min(nc, c1 + 1) : min(nc, c1 + 1 + band),
        ]
    )
    down = _mean(field[max(0, r0 - band) : max(0, r0), max(0, c0) : min(nc, c1 + 1)])
    up = _mean(
        field[
            min(nr, r1 + 1) : min(nr, r1 + 1 + band),
            max(0, c0) : min(nc, c1 + 1),
        ]
    )
    x_room = min(left, right)
    y_room = min(down, up)
    if not np.isfinite(x_room) and not np.isfinite(y_room):
        return 1.0, 1.0
    if x_room <= y_room:
        return 1.0, 0.25
    return 0.25, 1.0


def _local_component_bias(
    hard_xy: np.ndarray,
    members: np.ndarray,
    components: list[dict[str, object]],
    field: np.ndarray,
    cw: float,
    ch: float,
    *,
    max_distance_cells: int,
) -> tuple[np.ndarray, tuple[float, float]] | None:
    """Return a nearby cold-component anchor and preferred expansion axis."""
    if not components:
        return None
    nr, nc = field.shape
    cell_w, cell_h = cw / nc, ch / nr
    x0, x1 = float(hard_xy[members, 0].min()), float(hard_xy[members, 0].max())
    y0, y1 = float(hard_xy[members, 1].min()), float(hard_xy[members, 1].max())
    c0 = int(np.clip(np.floor(x0 / cell_w), 0, nc - 1))
    c1 = int(np.clip(np.floor(x1 / cell_w), 0, nc - 1))
    r0 = int(np.clip(np.floor(y0 / cell_h), 0, nr - 1))
    r1 = int(np.clip(np.floor(y1 / cell_h), 0, nr - 1))
    max_dist = max(0, int(max_distance_cells))
    best = None
    for comp in components:
        cr0 = int(comp["r0"])
        cr1 = int(comp["r1"])
        cc0 = int(comp["c0"])
        cc1 = int(comp["c1"])
        gap_c = max(0, max(c0 - cc1 - 1, cc0 - c1 - 1))
        gap_r = max(0, max(r0 - cr1 - 1, cr0 - r1 - 1))
        if gap_c > max_dist or gap_r > max_dist:
            continue
        dist = gap_c + gap_r
        row = (
            dist,
            float(comp["avg"]),
            -int(comp["size"]),
            int(comp["r0"]),
            int(comp["c0"]),
            comp,
        )
        if best is None or row < best:
            best = row
    if best is None:
        return None
    comp = best[5]
    anchor = np.array(
        [
            (float(comp["centroid_c"]) + 0.5) * cell_w,
            (float(comp["centroid_r"]) + 0.5) * cell_h,
        ],
        dtype=np.float64,
    )
    center = hard_xy[members].mean(axis=0)
    delta = anchor - center
    if abs(float(delta[0])) >= abs(float(delta[1])):
        axis = (1.0, float(const.HIER_DECOMPRESS_ANISO_SECONDARY))
    else:
        axis = (float(const.HIER_DECOMPRESS_ANISO_SECONDARY), 1.0)
    return anchor, axis


def _clip_to_region(xy, region, idx, hw, hh, cw, ch):
    if region is None:
        xy[:, 0] = np.clip(xy[:, 0], hw[idx], cw - hw[idx])
        xy[:, 1] = np.clip(xy[:, 1], hh[idx], ch - hh[idx])
    else:
        xy[:, 0] = np.clip(xy[:, 0], region[idx, 0], region[idx, 2])
        xy[:, 1] = np.clip(xy[:, 1], region[idx, 1], region[idx, 3])
    return xy


def _prepare_cluster_metadata(clusters, sizes, movable_h):
    metadata: dict[int, dict[str, np.ndarray | list[int]]] = {}
    for cid, mem in clusters.items():
        mem_all = np.asarray(mem, dtype=np.int64)
        if mem_all.size == 0:
            continue
        movable_members = (
            mem_all[movable_h[mem_all]] if mem_all.size else np.empty(0, dtype=np.int64)
        )
        order_by_area = list(mem_all[np.argsort(-sizes[mem_all, 0] * sizes[mem_all, 1])])
        metadata[int(cid)] = {
            "mem_all": mem_all,
            "movable_members": movable_members,
            "order_by_area": order_by_area,
        }
    return metadata


def _prepare_cluster_soft_members(
    cluster_softs,
    soft_hw,
    n,
    soft_movable=None,
):
    out: dict[int, np.ndarray] = {}
    if not cluster_softs:
        return out
    if soft_movable is not None:
        soft_movable = np.asarray(soft_movable, dtype=np.bool_)
    for cid, sidx in cluster_softs.items():
        s_local = np.asarray(sidx, dtype=np.int64) - int(n)
        s_local = s_local[(s_local >= 0) & (s_local < soft_hw.shape[0])]
        if soft_movable is not None and s_local.size:
            s_local = s_local[soft_movable[s_local]]
        if s_local.size:
            out[int(cid)] = s_local
    return out


def _prepare_bridge_soft_mapping(bridge_softs, soft_movable=None):
    bridge_to_soft = {}
    soft_to_bridge = {}
    if not bridge_softs:
        return bridge_to_soft, soft_to_bridge
    if soft_movable is not None:
        soft_movable = np.asarray(soft_movable, dtype=np.bool_)
    for sk, cids in bridge_softs.items():
        sk_i = int(sk)
        if soft_movable is not None and not bool(soft_movable[sk_i]):
            continue
        cids_arr = np.unique(np.asarray(cids, dtype=np.int64))
        if cids_arr.size == 0:
            continue
        soft_to_bridge[int(sk_i)] = cids_arr
        for cid in cids_arr:
            bridge_to_soft.setdefault(int(cid), []).append(sk_i)
    return bridge_to_soft, soft_to_bridge


def _full_tensor(hard_xy, soft_xy):
    return torch.tensor(np.vstack([hard_xy, soft_xy]).astype(np.float32), dtype=torch.float32)


def _hard_rejection_reason(hard_xy, sizes, hw, hh, cw, ch) -> str | None:
    if (
        np.any(hard_xy[:, 0] < hw - 1e-6)
        or np.any(hard_xy[:, 0] > cw - hw + 1e-6)
        or np.any(hard_xy[:, 1] < hh - 1e-6)
        or np.any(hard_xy[:, 1] > ch - hh + 1e-6)
    ):
        return "out_of_bounds"
    left = hard_xy[:, 0] - hw
    right = hard_xy[:, 0] + hw
    bottom = hard_xy[:, 1] - hh
    top = hard_xy[:, 1] + hh
    n = hard_xy.shape[0]
    for i in range(n):
        overlap = (
            (left[i] < right[i + 1 :] - 1e-6)
            & (right[i] > left[i + 1 :] + 1e-6)
            & (bottom[i] < top[i + 1 :] - 1e-6)
            & (top[i] > bottom[i + 1 :] + 1e-6)
        )
        if bool(np.any(overlap)):
            return "illegal_overlap"
    return None


def _rect_overlap_area(a, b) -> float:
    x0 = max(float(a[0]), float(b[0]))
    y0 = max(float(a[1]), float(b[1]))
    x1 = min(float(a[2]), float(b[2]))
    y1 = min(float(a[3]), float(b[3]))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float((x1 - x0) * (y1 - y0))


def _decompression_feasibility(
    cand_h: np.ndarray,
    members: np.ndarray,
    sizes: np.ndarray,
    hw: np.ndarray,
    hh: np.ndarray,
    movable_h: np.ndarray,
    *,
    target_density: float,
    min_free_ratio: float,
    max_blockage: float,
) -> tuple[bool, dict[str, float]]:
    """Estimate whether a decompression bbox has enough local unblocked area."""
    members = np.asarray(members, dtype=np.int64)
    if members.size == 0:
        return True, {
            "feasible_free_ratio": 1.0,
            "feasible_blockage_ratio": 0.0,
            "feasible_required_area": 0.0,
            "feasible_available_area": 0.0,
        }
    bbox = (
        float(np.min(cand_h[members, 0] - hw[members])),
        float(np.min(cand_h[members, 1] - hh[members])),
        float(np.max(cand_h[members, 0] + hw[members])),
        float(np.max(cand_h[members, 1] + hh[members])),
    )
    bbox_area = max(1e-9, float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])))
    member_area = float(np.sum(sizes[members, 0] * sizes[members, 1]))
    required_area = member_area / max(0.05, float(target_density))
    member_set = set(int(v) for v in members)
    blocked = 0.0
    for idx in range(cand_h.shape[0]):
        if int(idx) in member_set:
            continue
        # Movable neighbors are still real blockage for the local legalization attempt;
        # fixed and movable both matter in this cheap capacity estimate.
        other = (
            float(cand_h[idx, 0] - hw[idx]),
            float(cand_h[idx, 1] - hh[idx]),
            float(cand_h[idx, 0] + hw[idx]),
            float(cand_h[idx, 1] + hh[idx]),
        )
        blocked += _rect_overlap_area(bbox, other)
    blocked = min(float(blocked), bbox_area)
    available = max(0.0, bbox_area - blocked)
    free_ratio = float(available / max(required_area, 1e-9))
    blockage_ratio = float(blocked / bbox_area)
    feasible = bool(free_ratio >= float(min_free_ratio) and blockage_ratio <= float(max_blockage))
    return feasible, {
        "feasible_free_ratio": float(free_ratio),
        "feasible_blockage_ratio": float(blockage_ratio),
        "feasible_required_area": float(required_area),
        "feasible_available_area": float(available),
    }


def _cluster_decompression_relief(
    hard_xy,
    soft_xy,
    sizes,
    hw,
    hh,
    soft_hw,
    soft_hh,
    cw,
    ch,
    movable_h,
    soft_movable,
    n,
    clusters,
    cluster_softs,
    bridge_softs,
    hard_region,
    soft_region,
    plc,
    benchmark,
    initial_score: float,
    deadline=None,
    rounds: int = 1,
    hot_percentile: float = 65.0,
    quality_budget: float = 0.03,
    min_proxy_gain: float = 1e-4,
    use_density: bool = False,
    anisotropic: bool = False,
    anisotropic_band: int = 3,
    anisotropic_secondary: float = 0.25,
    local_component_cold_percentile: float = 45.0,
    local_component_min_cells: int = 4,
    local_component_max_distance_cells: int = 4,
    local_component_shift_frac: float = 0.0,
    cluster_priority: dict[int, float] | None = None,
    cluster_priority_weight: float = 0.0,
    graph_edges=None,
    seed_hard_xy: np.ndarray | None = None,
    graph_confidence: dict[int, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, int, float, float]:
    """Try group decompression candidates and accept exact proxy improvements."""
    if not clusters:
        return (
            hard_xy,
            soft_xy,
            0,
            float(initial_score),
            hierarchy_quality_metric(hard_xy, clusters),
        )
    nr, nc = int(benchmark.grid_rows), int(benchmark.grid_cols)
    best_score = float(initial_score)
    cur_h = hard_xy.copy()
    cur_s = soft_xy.copy()
    cur_quality = hierarchy_quality_metric(cur_h, clusters)
    accepts = 0
    factors = const.HIER_DECOMPRESS_FACTORS
    prefilter_enabled = os.environ.get(
        "HIER_GRAPH_PREFILTER",
        "1" if bool(getattr(const, "HIER_GRAPH_PREFILTER", False)) else "0",
    ).strip() not in {"0", "false", "False", "no", "NO", "off", ""}
    prefilter_low_tension = max(
        0.0, float(getattr(const, "HIER_GRAPH_PREFILTER_LOW_TENSION", 0.05))
    )
    prefilter_min_relief = max(0.0, float(getattr(const, "HIER_GRAPH_PREFILTER_MIN_RELIEF", 0.0)))
    feasibility_min_free = max(
        0.0,
        float(getattr(const, "HIER_DECOMPRESS_FEASIBILITY_MIN_FREE_RATIO", 0.70)),
    )
    feasibility_max_blockage = min(
        1.0,
        max(0.0, float(getattr(const, "HIER_DECOMPRESS_FEASIBILITY_MAX_BLOCKAGE", 0.75))),
    )
    graph_rescue_enabled = os.environ.get(
        "HIER_DECOMPRESS_GRAPH_RESCUE",
        "1" if bool(getattr(const, "HIER_DECOMPRESS_GRAPH_RESCUE", False)) else "0",
    ).strip() not in {"0", "false", "False", "no", "NO", "off", ""}
    graph_rescue_max_delta = float(getattr(const, "HIER_DECOMPRESS_GRAPH_RESCUE_MAX_DELTA", 0.0))
    graph_rescue_shrinks = tuple(
        float(v) for v in getattr(const, "HIER_DECOMPRESS_GRAPH_RESCUE_SHRINKS", (0.75, 0.55))
    )
    graph_rescue_shift_mults = tuple(
        float(v) for v in getattr(const, "HIER_DECOMPRESS_GRAPH_RESCUE_SHIFT_MULTS", (1.25,))
    )
    graph_rescue_max_variants = max(
        0, int(getattr(const, "HIER_DECOMPRESS_GRAPH_RESCUE_MAX_VARIANTS", 4))
    )
    graph_survivor_max_delta = float(
        getattr(const, "HIER_DECOMPRESS_GRAPH_SURVIVOR_MAX_DELTA", -0.01)
    )
    graph_survivor_proxy_miss = max(
        0.0, float(getattr(const, "HIER_DECOMPRESS_GRAPH_SURVIVOR_PROXY_MISS", 0.0015))
    )
    graph_survivor_top_hard = max(
        0, int(getattr(const, "HIER_DECOMPRESS_GRAPH_SURVIVOR_TOP_HARD", 8))
    )
    graph_survivor_top_soft = max(
        0, int(getattr(const, "HIER_DECOMPRESS_GRAPH_SURVIVOR_TOP_SOFT", 8))
    )
    graph_survivor_radius_cells = max(
        1, int(getattr(const, "HIER_DECOMPRESS_GRAPH_SURVIVOR_RADIUS_CELLS", 1))
    )
    graph_survivor_max_trials = max(
        0, int(getattr(const, "HIER_DECOMPRESS_GRAPH_SURVIVOR_MAX_TRIALS", 48))
    )
    cluster_meta = _prepare_cluster_metadata(clusters, sizes, movable_h)
    if not cluster_meta:
        return (
            hard_xy,
            soft_xy,
            0,
            float(initial_score),
            hierarchy_quality_metric(hard_xy, clusters),
        )
    cluster_soft_members = _prepare_cluster_soft_members(
        cluster_softs,
        soft_hw,
        n,
        soft_movable=soft_movable,
    )
    bridge_to_soft, soft_to_bridge = _prepare_bridge_soft_mapping(
        bridge_softs, soft_movable=soft_movable
    )

    for _ in range(max(1, int(rounds))):
        if deadline is not None and time.monotonic() > deadline:
            break
        # Keep decompression congestion-driven; exact proxy still gates density side effects.
        field = _congestion_field(plc, nr, nc)
        if field is None:
            break
        local_components = cold_connected_components(
            field,
            cold_percentile=float(local_component_cold_percentile),
            min_cells=max(1, int(local_component_min_cells)),
        )
        local = _cell_values(cur_h, field, cw, ch)
        heat = []
        for cid, meta in cluster_meta.items():
            mov = np.asarray(meta["movable_members"], dtype=np.int64)
            if mov.size >= 2:
                heat.append((int(cid), float(local[mov].mean())))
        if not heat:
            break
        threshold = float(np.percentile([h for _, h in heat], hot_percentile))
        priority = cluster_priority or {}
        priority_weight = max(0.0, float(cluster_priority_weight))
        span = max(float(field.max()) - float(field.min()), 1e-12)
        ordered = [
            cid
            for cid, h in sorted(
                heat,
                key=lambda x: -(
                    float(x[1]) + priority_weight * span * float(priority.get(int(x[0]), 0.0))
                ),
            )
            if h >= threshold
        ]
        centroids = {cid: cur_h[meta["mem_all"]].mean(axis=0) for cid, meta in cluster_meta.items()}
        accepted_round = False
        for cid in ordered:
            if deadline is not None and time.monotonic() > deadline:
                break
            meta = cluster_meta[cid]
            mem_all = np.asarray(meta["mem_all"], dtype=np.int64)
            mem = np.asarray(meta["movable_members"], dtype=np.int64)
            if mem.size < 2:
                continue
            center = centroids[cid]
            order = list(meta["order_by_area"])
            axis_x, axis_y = (1.0, 1.0)
            local_anchor = None
            if anisotropic:
                axis_x, axis_y = _axis_room_scale(
                    cur_h,
                    mem_all,
                    field,
                    cw,
                    ch,
                    band=max(1, int(anisotropic_band)),
                )
                axis_x = max(float(anisotropic_secondary), float(axis_x))
                axis_y = max(float(anisotropic_secondary), float(axis_y))
            if local_components:
                bias = _local_component_bias(
                    cur_h,
                    mem_all,
                    local_components,
                    field,
                    cw,
                    ch,
                    max_distance_cells=max(0, int(local_component_max_distance_cells)),
                )
                if bias is not None:
                    local_anchor, local_axis = bias
                    axis_x = max(float(anisotropic_secondary), float(local_axis[0]))
                    axis_y = max(float(anisotropic_secondary), float(local_axis[1]))
            for factor in factors:
                if deadline is not None and time.monotonic() > deadline:
                    break
                score = None
                q = None
                reason = "exact_proxy_failed"
                accepted = False
                old_score = float(best_score)
                old_quality = float(cur_quality)
                vec = cur_h[mem] - center

                def _build_candidate(
                    effective_factor: float,
                    shift_mult: float,
                ) -> tuple[np.ndarray, np.ndarray, list[int], np.ndarray]:
                    trial_h = cur_h.copy()
                    trial_s = cur_s.copy()
                    trial_scale = np.array(
                        [
                            1.0 + (effective_factor - 1.0) * axis_x,
                            1.0 + (effective_factor - 1.0) * axis_y,
                        ],
                        dtype=np.float64,
                    )
                    trial_shift = np.zeros(2, dtype=np.float64)
                    if local_anchor is not None:
                        trial_shift = (
                            (
                                np.asarray(local_anchor, dtype=np.float64)
                                - np.asarray(center, dtype=np.float64)
                            )
                            * max(0.0, float(local_component_shift_frac))
                            * max(0.0, float(effective_factor) - 1.0)
                            * max(0.0, float(shift_mult))
                        )
                    trial_h[mem] = center + vec * trial_scale + trial_shift
                    trial_h[mem] = _clip_to_region(trial_h[mem], hard_region, mem, hw, hh, cw, ch)
                    touched: list[int] = []
                    if cluster_soft_members.get(cid) is not None:
                        sidx = cluster_soft_members[cid]
                        if sidx.size:
                            touched.extend(sidx.tolist())
                            svec = trial_s[sidx] - center
                            trial_s[sidx] = center + svec * trial_scale + trial_shift
                            trial_s[sidx] = _clip_to_region(
                                trial_s[sidx], soft_region, sidx, soft_hw, soft_hh, cw, ch
                            )

                    for sk in bridge_to_soft.get(cid, ()):
                        cids = soft_to_bridge.get(sk, np.empty(0, dtype=np.int64))
                        if cids.size == 0:
                            continue
                        pts = [centroids[int(c)] for c in cids if int(c) in centroids]
                        if not pts:
                            continue
                        target = np.asarray(pts, dtype=np.float64).mean(axis=0)
                        touched.append(sk)
                        trial_s[sk : sk + 1] = 0.55 * trial_s[sk : sk + 1] + 0.45 * target
                        trial_s[sk : sk + 1] = _clip_to_region(
                            trial_s[sk : sk + 1],
                            soft_region,
                            np.array([sk], dtype=np.int64),
                            soft_hw,
                            soft_hh,
                            cw,
                            ch,
                        )
                    return trial_h, trial_s, touched, trial_scale

                def _candidate_feasibility(trial_h: np.ndarray) -> tuple[bool, dict[str, float]]:
                    default_stats = {
                        "feasible_free_ratio": 1.0,
                        "feasible_blockage_ratio": 0.0,
                        "feasible_required_area": 0.0,
                        "feasible_available_area": 0.0,
                    }
                    return _decompression_feasibility(
                        trial_h,
                        mem,
                        sizes,
                        hw,
                        hh,
                        movable_h,
                        target_density=float(getattr(const, "HIER_REGION_DENSITY", 0.65)),
                        min_free_ratio=feasibility_min_free,
                        max_blockage=feasibility_max_blockage,
                    )

                def _finalize_candidate(
                    trial_h: np.ndarray,
                    trial_s: np.ndarray,
                    touched: list[int],
                    trial_scale: np.ndarray,
                    trial_feasible: bool,
                ) -> tuple[np.ndarray, np.ndarray, bool, bool, str | None, np.ndarray]:
                    trial_changed_h = False
                    trial_changed_s = False
                    trial_reason = None
                    if not trial_feasible:
                        trial_changed_h = np.any(trial_h[mem] != cur_h[mem])
                        trial_reason = "feasibility_blocked"
                    elif np.any(trial_h[mem] != cur_h[mem]):
                        trial_h = _will_legalize(
                            trial_h,
                            movable_h,
                            sizes,
                            hw,
                            hh,
                            cw,
                            ch,
                            n,
                            deadline=deadline,
                            order=order,
                        )
                        trial_changed_h = np.any(trial_h[mem] != cur_h[mem])
                    if not trial_changed_h and touched:
                        moved_soft = np.unique(np.asarray(touched, dtype=np.int64))
                        if moved_soft.size:
                            trial_changed_s = np.any(trial_s[moved_soft] != cur_s[moved_soft])
                    if trial_reason is None:
                        trial_reason = _hard_rejection_reason(trial_h, sizes, hw, hh, cw, ch)
                    return (
                        trial_h,
                        trial_s,
                        bool(trial_changed_h),
                        bool(trial_changed_s),
                        trial_reason,
                        trial_scale,
                    )

                def _try_graph_survivor(
                    trial_h: np.ndarray,
                    trial_s: np.ndarray,
                    base_score: float,
                ) -> tuple[np.ndarray, np.ndarray, float | None, int]:
                    if (
                        graph_survivor_top_hard <= 0 and graph_survivor_top_soft <= 0
                    ) or graph_survivor_max_trials <= 0:
                        return trial_h, trial_s, None, 0
                    cell_w = float(cw) / max(1, nc)
                    cell_h = float(ch) / max(1, nr)
                    local_vals = (
                        _cell_values(trial_h[mem], field, cw, ch) if mem.size else np.empty(0)
                    )
                    order_idx = np.argsort(-local_vals)[: min(graph_survivor_top_hard, mem.size)]
                    hard_order = [int(mem[int(k)]) for k in order_idx]
                    offsets: list[tuple[float, float, int]] = []
                    for dr in range(-graph_survivor_radius_cells, graph_survivor_radius_cells + 1):
                        for dc in range(
                            -graph_survivor_radius_cells, graph_survivor_radius_cells + 1
                        ):
                            if dc == 0 and dr == 0:
                                continue
                            dist = max(abs(dc), abs(dr))
                            if dist > graph_survivor_radius_cells:
                                continue
                            offsets.append((dc * cell_w, dr * cell_h, abs(dc) + abs(dr)))
                    offsets.sort(key=lambda row: (row[2], abs(row[0]) + abs(row[1])))
                    best_trial_h = trial_h
                    best_trial_s = trial_s
                    best_trial_score = float(base_score)
                    attempts = 0
                    all_h = np.arange(n)
                    for hard_i in hard_order:
                        if deadline is not None and time.monotonic() > deadline:
                            break
                        mask = all_h != int(hard_i)
                        for dx, dy, _dist in offsets:
                            if attempts >= graph_survivor_max_trials:
                                break
                            nx = float(trial_h[hard_i, 0] + dx)
                            ny = float(trial_h[hard_i, 1] + dy)
                            if nx < hw[hard_i] or nx > cw - hw[hard_i]:
                                continue
                            if ny < hh[hard_i] or ny > ch - hh[hard_i]:
                                continue
                            if hard_region is not None:
                                if nx < hard_region[hard_i, 0] or nx > hard_region[hard_i, 2]:
                                    continue
                                if ny < hard_region[hard_i, 1] or ny > hard_region[hard_i, 3]:
                                    continue
                            overlaps = (
                                (nx - hw[hard_i] < trial_h[mask, 0] + hw[mask] - 1e-6)
                                & (nx + hw[hard_i] > trial_h[mask, 0] - hw[mask] + 1e-6)
                                & (ny - hh[hard_i] < trial_h[mask, 1] + hh[mask] - 1e-6)
                                & (ny + hh[hard_i] > trial_h[mask, 1] - hh[mask] + 1e-6)
                            )
                            if bool(np.any(overlaps)):
                                continue
                            attempts += 1
                            trial2_h = trial_h.copy()
                            trial2_h[hard_i, 0] = nx
                            trial2_h[hard_i, 1] = ny
                            q2 = hierarchy_quality_metric(trial2_h, clusters)
                            if q2 > cur_quality + quality_budget:
                                continue
                            trial_score = float(
                                _exact_proxy(_full_tensor(trial2_h, trial_s), benchmark, plc)
                            )
                            if trial_score < best_trial_score:
                                best_trial_score = trial_score
                                best_trial_h = trial2_h
                            if trial_score < old_score - min_proxy_gain:
                                return trial2_h, trial_s, trial_score, attempts
                        if attempts >= graph_survivor_max_trials:
                            break
                    if attempts < graph_survivor_max_trials and graph_survivor_top_soft > 0:
                        soft_parts = []
                        owned_soft = cluster_soft_members.get(cid)
                        if owned_soft is not None and owned_soft.size:
                            soft_parts.append(np.asarray(owned_soft, dtype=np.int64))
                        bridge_soft = np.asarray(bridge_to_soft.get(cid, ()), dtype=np.int64)
                        if bridge_soft.size:
                            soft_parts.append(bridge_soft)
                        if soft_parts:
                            soft_ids = np.unique(np.concatenate(soft_parts)).astype(np.int64)
                            soft_ids = soft_ids[(soft_ids >= 0) & (soft_ids < trial_s.shape[0])]
                            if soft_movable is not None and soft_ids.size:
                                sm = np.asarray(soft_movable, dtype=np.bool_)
                                soft_ids = soft_ids[sm[soft_ids]]
                            if soft_ids.size:
                                soft_vals = _cell_values(trial_s[soft_ids], field, cw, ch)
                                soft_order = soft_ids[
                                    np.argsort(-soft_vals)[
                                        : min(graph_survivor_top_soft, soft_ids.size)
                                    ]
                                ]
                                for soft_i_raw in soft_order:
                                    if deadline is not None and time.monotonic() > deadline:
                                        break
                                    soft_i = int(soft_i_raw)
                                    for dx, dy, _dist in offsets:
                                        if attempts >= graph_survivor_max_trials:
                                            break
                                        nx = float(
                                            np.clip(
                                                trial_s[soft_i, 0] + dx,
                                                soft_hw[soft_i],
                                                cw - soft_hw[soft_i],
                                            )
                                        )
                                        ny = float(
                                            np.clip(
                                                trial_s[soft_i, 1] + dy,
                                                soft_hh[soft_i],
                                                ch - soft_hh[soft_i],
                                            )
                                        )
                                        if soft_region is not None:
                                            if (
                                                nx < soft_region[soft_i, 0]
                                                or nx > soft_region[soft_i, 2]
                                                or ny < soft_region[soft_i, 1]
                                                or ny > soft_region[soft_i, 3]
                                            ):
                                                continue
                                        if (
                                            abs(nx - float(trial_s[soft_i, 0])) < 1e-9
                                            and abs(ny - float(trial_s[soft_i, 1])) < 1e-9
                                        ):
                                            continue
                                        attempts += 1
                                        trial2_s = trial_s.copy()
                                        trial2_s[soft_i, 0] = nx
                                        trial2_s[soft_i, 1] = ny
                                        trial_score = float(
                                            _exact_proxy(
                                                _full_tensor(trial_h, trial2_s),
                                                benchmark,
                                                plc,
                                            )
                                        )
                                        if trial_score < best_trial_score:
                                            best_trial_score = trial_score
                                            best_trial_s = trial2_s
                                        if trial_score < old_score - min_proxy_gain:
                                            return trial_h, trial2_s, trial_score, attempts
                                    if attempts >= graph_survivor_max_trials:
                                        break
                    if best_trial_score < old_score - min_proxy_gain:
                        return best_trial_h, best_trial_s, best_trial_score, attempts
                    return trial_h, trial_s, None, attempts

                cand_h, cand_s, soft_touched, scale = _build_candidate(float(factor), 1.0)
                feasible, feasible_stats = _candidate_feasibility(cand_h)
                cand_h, cand_s, changed_h, changed_s, reason, scale = _finalize_candidate(
                    cand_h, cand_s, soft_touched, scale, feasible
                )

                feasibility_rejected = False
                if reason == "feasibility_blocked":
                    feasibility_rejected = True
                if not changed_h and not changed_s:
                    continue

                graph_tension = float(priority.get(int(cid), 0.0))
                source_field = float(local[mem].mean()) if mem.size else 0.0
                target_field = (
                    float(_cell_values(cand_h[mem], field, cw, ch).mean())
                    if mem.size
                    else source_field
                )
                local_relief = float(source_field - target_field)
                graph_delta_stats = candidate_graph_edge_delta(
                    cur_h,
                    cand_h,
                    clusters,
                    graph_edges,
                    cw=cw,
                    ch=ch,
                    field=field,
                    seed_hard_xy=seed_hard_xy,
                    confidence=graph_confidence,
                    affected_clusters=[int(cid)],
                    samples=max(2, int(getattr(const, "HIER_GRAPH_TENSION_CORRIDOR_SAMPLES", 9))),
                )
                graph_rescue_attempted = False
                graph_rescue_used = False
                graph_rescue_attempts = 0
                graph_rescue_trigger = str(reason or "")
                trigger_delta = float(graph_delta_stats.get("graph_candidate_delta", 0.0))
                can_rescue = (
                    graph_rescue_enabled
                    and graph_rescue_max_variants > 0
                    and trigger_delta <= graph_rescue_max_delta
                    and reason in {"feasibility_blocked", "illegal_overlap"}
                )
                if can_rescue:
                    graph_rescue_attempted = True
                    rescue_specs: list[tuple[float, float]] = []
                    for shrink in graph_rescue_shrinks:
                        if len(rescue_specs) >= graph_rescue_max_variants:
                            break
                        if not (0.0 < float(shrink) < 1.0):
                            continue
                        rescue_specs.append((1.0 + (float(factor) - 1.0) * float(shrink), 1.0))
                    if local_anchor is not None:
                        for shift_mult in graph_rescue_shift_mults:
                            if len(rescue_specs) >= graph_rescue_max_variants:
                                break
                            rescue_specs.append((float(factor), float(shift_mult)))
                    for rescue_factor, rescue_shift_mult in rescue_specs:
                        if deadline is not None and time.monotonic() > deadline:
                            break
                        graph_rescue_attempts += 1
                        trial_h, trial_s, trial_softs, trial_scale = _build_candidate(
                            rescue_factor, rescue_shift_mult
                        )
                        trial_feasible, trial_feasible_stats = _candidate_feasibility(trial_h)
                        (
                            trial_h,
                            trial_s,
                            trial_changed_h,
                            trial_changed_s,
                            trial_reason,
                            trial_scale,
                        ) = _finalize_candidate(
                            trial_h,
                            trial_s,
                            trial_softs,
                            trial_scale,
                            trial_feasible,
                        )
                        if trial_reason is not None:
                            continue
                        if not trial_changed_h and not trial_changed_s:
                            continue
                        cand_h = trial_h
                        cand_s = trial_s
                        changed_h = trial_changed_h
                        changed_s = trial_changed_s
                        reason = None
                        feasibility_rejected = False
                        feasible_stats = trial_feasible_stats
                        scale = trial_scale
                        target_field = (
                            float(_cell_values(cand_h[mem], field, cw, ch).mean())
                            if mem.size
                            else source_field
                        )
                        local_relief = float(source_field - target_field)
                        graph_delta_stats = candidate_graph_edge_delta(
                            cur_h,
                            cand_h,
                            clusters,
                            graph_edges,
                            cw=cw,
                            ch=ch,
                            field=field,
                            seed_hard_xy=seed_hard_xy,
                            confidence=graph_confidence,
                            affected_clusters=[int(cid)],
                            samples=max(
                                2,
                                int(getattr(const, "HIER_GRAPH_TENSION_CORRIDOR_SAMPLES", 9)),
                            ),
                        )
                        graph_rescue_used = True
                        break
                graph_survivor_attempted = False
                graph_survivor_used = False
                graph_survivor_trials = 0
                graph_survivor_pre_score = None
                prefiltered = False
                q = None if feasibility_rejected else hierarchy_quality_metric(cand_h, clusters)
                if reason is None and q > cur_quality + quality_budget:
                    reason = "hierarchy_quality_failed"
                if (
                    reason is None
                    and prefilter_enabled
                    and graph_tension <= prefilter_low_tension
                    and local_relief <= prefilter_min_relief
                ):
                    reason = "prefilter_no_local_relief"
                    prefiltered = True
                if reason is None:
                    score = float(_exact_proxy(_full_tensor(cand_h, cand_s), benchmark, plc))
                    if score < old_score - min_proxy_gain:
                        cur_h, cur_s = cand_h, cand_s
                        best_score = score
                        cur_quality = q
                        accepts += 1
                        accepted_round = True
                        accepted = True
                        reason = "accepted"
                    else:
                        proxy_miss = float(score) - (old_score - min_proxy_gain)
                        strong_graph_delta = (
                            float(graph_delta_stats.get("graph_candidate_delta", 0.0))
                            <= graph_survivor_max_delta
                        )
                        if (
                            strong_graph_delta
                            and proxy_miss <= graph_survivor_proxy_miss
                            and deadline is not None
                            and time.monotonic() < deadline
                        ):
                            graph_survivor_attempted = True
                            graph_survivor_pre_score = float(score)
                            (
                                survivor_h,
                                survivor_s,
                                survivor_score,
                                survivor_trials,
                            ) = _try_graph_survivor(
                                cand_h,
                                cand_s,
                                float(score),
                            )
                            graph_survivor_trials = int(survivor_trials)
                            if survivor_score is not None:
                                cand_h = survivor_h
                                cand_s = survivor_s
                                score = float(survivor_score)
                                target_field = (
                                    float(_cell_values(cand_h[mem], field, cw, ch).mean())
                                    if mem.size
                                    else source_field
                                )
                                local_relief = float(source_field - target_field)
                                graph_delta_stats = candidate_graph_edge_delta(
                                    cur_h,
                                    cand_h,
                                    clusters,
                                    graph_edges,
                                    cw=cw,
                                    ch=ch,
                                    field=field,
                                    seed_hard_xy=seed_hard_xy,
                                    confidence=graph_confidence,
                                    affected_clusters=[int(cid)],
                                    samples=max(
                                        2,
                                        int(
                                            getattr(
                                                const,
                                                "HIER_GRAPH_TENSION_CORRIDOR_SAMPLES",
                                                9,
                                            )
                                        ),
                                    ),
                                )
                                q = hierarchy_quality_metric(cand_h, clusters)
                                if q <= cur_quality + quality_budget:
                                    cur_h, cur_s = cand_h, cand_s
                                    best_score = score
                                    cur_quality = q
                                    accepts += 1
                                    accepted_round = True
                                    accepted = True
                                    reason = "accepted"
                                    graph_survivor_used = True
                                else:
                                    reason = "hierarchy_quality_failed"
                            else:
                                reason = "exact_proxy_failed"
                        else:
                            reason = "exact_proxy_failed"
                if accepted:
                    break
            if accepted_round:
                break
        if not accepted_round:
            break
    best_score = float(_exact_proxy(_full_tensor(cur_h, cur_s), benchmark, plc))
    return cur_h, cur_s, accepts, best_score, cur_quality
