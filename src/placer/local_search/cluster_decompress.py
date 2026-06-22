"""Exact-gated cluster decompression for congestion relief."""

from __future__ import annotations

import time

import numpy as np
import torch

from utils import constants as const
from placer.legalize.spiral import _will_legalize
from placer.local_search.fields import _congestion_field
from placer.local_search.gnn_trace import gnn_trace_limit, log_gnn_event
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


def _clip_to_region(xy, region, idx, hw, hh, cw, ch):
    if region is None:
        xy[:, 0] = np.clip(xy[:, 0], hw[idx], cw - hw[idx])
        xy[:, 1] = np.clip(xy[:, 1], hh[idx], ch - hh[idx])
    else:
        xy[:, 0] = np.clip(xy[:, 0], region[idx, 0], region[idx, 2])
        xy[:, 1] = np.clip(xy[:, 1], region[idx, 1], region[idx, 3])
    return xy


def _cluster_centroids(hard_xy, clusters):
    out = {}
    for cid, mem in clusters.items():
        p = hard_xy[np.asarray(mem, dtype=np.int64)]
        out[int(cid)] = p.mean(axis=0)
    return out


def _prepare_cluster_metadata(clusters, sizes, movable_h):
    metadata: dict[int, dict[str, np.ndarray | list[int]]] = {}
    for cid, mem in clusters.items():
        mem_all = np.asarray(mem, dtype=np.int64)
        if mem_all.size == 0:
            continue
        movable_members = mem_all[movable_h[mem_all]] if mem_all.size else np.empty(0, dtype=np.int64)
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
    trace_limit = gnn_trace_limit()
    trace_count = 0
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
        local = _cell_values(cur_h, field, cw, ch)
        heat = []
        for cid, meta in cluster_meta.items():
            mov = np.asarray(meta["movable_members"], dtype=np.int64)
            if mov.size >= 2:
                heat.append((int(cid), float(local[mov].mean())))
        if not heat:
            break
        threshold = float(np.percentile([h for _, h in heat], hot_percentile))
        ordered = [cid for cid, h in sorted(heat, key=lambda x: -x[1]) if h >= threshold]
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
            for factor in factors:
                if deadline is not None and time.monotonic() > deadline:
                    break
                score = None
                q = None
                reason = "exact_proxy_failed"
                accepted = False
                old_score = float(best_score)
                old_quality = float(cur_quality)
                cand_h = cur_h.copy()
                cand_s = cur_s.copy()
                changed_h = False
                changed_s = False
                vec = cand_h[mem] - center
                scale = np.array(
                    [1.0 + (factor - 1.0) * axis_x, 1.0 + (factor - 1.0) * axis_y],
                    dtype=np.float64,
                )
                cand_h[mem] = center + vec * scale
                cand_h[mem] = _clip_to_region(cand_h[mem], hard_region, mem, hw, hh, cw, ch)
                soft_touched: list[int] = []
                if cluster_soft_members.get(cid) is not None:
                    sidx = cluster_soft_members[cid]
                    if sidx.size:
                        soft_touched.extend(sidx.tolist())
                        svec = cand_s[sidx] - center
                        cand_s[sidx] = center + svec * scale
                        cand_s[sidx] = _clip_to_region(
                            cand_s[sidx], soft_region, sidx, soft_hw, soft_hh, cw, ch
                        )

                for sk in bridge_to_soft.get(cid, ()):
                    cids = soft_to_bridge.get(sk, np.empty(0, dtype=np.int64))
                    if cids.size == 0:
                        continue
                    pts = [centroids[int(c)] for c in cids if int(c) in centroids]
                    if not pts:
                        continue
                    target = np.asarray(pts, dtype=np.float64).mean(axis=0)
                    soft_touched.append(sk)
                    cand_s[sk : sk + 1] = 0.55 * cand_s[sk : sk + 1] + 0.45 * target
                    cand_s[sk : sk + 1] = _clip_to_region(
                        cand_s[sk : sk + 1],
                        soft_region,
                        np.array([sk], dtype=np.int64),
                        soft_hw,
                        soft_hh,
                        cw,
                        ch,
                    )

                if np.any(cand_h[mem] != cur_h[mem]):
                    changed_h = True
                    cand_h = _will_legalize(
                        cand_h, movable_h, sizes, hw, hh, cw, ch, n, deadline=deadline, order=order
                    )
                    changed_h = np.any(cand_h[mem] != cur_h[mem])
                if not changed_h and soft_touched:
                    moved_soft = np.unique(np.asarray(soft_touched, dtype=np.int64))
                    if moved_soft.size:
                        changed_s = np.any(cand_s[moved_soft] != cur_s[moved_soft])
                if not changed_h and not changed_s:
                    continue

                reason = _hard_rejection_reason(cand_h, sizes, hw, hh, cw, ch)
                q = hierarchy_quality_metric(cand_h, clusters)
                if reason is None and q > cur_quality + quality_budget:
                    reason = "hierarchy_quality_failed"
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
                        reason = "exact_proxy_failed"
                if trace_count < trace_limit:
                    log_gnn_event(
                        "hier_decompression_candidate",
                        benchmark=getattr(benchmark, "name", ""),
                        operator="cluster_decompression",
                        candidate_id=trace_count,
                        cluster=int(cid),
                        movable_count=int(mem.size),
                        member_count=int(mem_all.size),
                        soft_count=int(
                            0 if cluster_softs.get(cid) is None else len(cluster_softs.get(cid))
                        ),
                        expansion_factor=float(factor),
                        axis_scale=[float(scale[0]), float(scale[1])],
                        hierarchy_quality_before=old_quality,
                        hierarchy_quality_after=None if q is None else float(q),
                        hierarchy_quality_delta=None if q is None else float(q) - old_quality,
                        old_proxy=old_score,
                        candidate_proxy=None if score is None else float(score),
                        proxy_delta=None if score is None else float(score) - old_score,
                        accepted=bool(accepted),
                        rejection_reason=None if accepted else reason,
                    )
                    trace_count += 1
                if accepted:
                    break
            if accepted_round:
                break
        if not accepted_round:
            break
    best_score = float(_exact_proxy(_full_tensor(cur_h, cur_s), benchmark, plc))
    return cur_h, cur_s, accepts, best_score, cur_quality
