"""Coldspot cluster kick used by the hierarchy path."""

from __future__ import annotations

import numpy as np

from placer.legalize.spiral import _will_legalize
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
    cell_w, cell_h = cw / nc, ch / nr
    flat_avg = avg.ravel()
    flat_order = np.argsort(flat_avg)
    min_sep_cells = max(1.0, 0.5 * float(w))
    anchors: list[tuple[float, float, float, float, float]] = []
    candidate_mult = 32
    for flat in flat_order[: max(count * candidate_mult, count)]:
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
        blend = float(const.HIER_COLDSPOT_LOW_DISP_BLEND)
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
    anchor_cache: dict[tuple[int], list[tuple[float, float, float]]] = {}
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
        cache_key = (int(win_cells),)
        cached = anchor_cache.get(cache_key)
        if cached is None or len(cached) < anchor_count:
            cached = _cold_window_anchors(
                cong_field,
                nr,
                nc,
                cw,
                ch,
                win_cells,
                anchor_count,
            )
            anchor_cache[cache_key] = cached
        anchors = cached[:anchor_count]
        jx = max(float(hw[members].max()), win_microns / 4.0)
        jy = max(float(hh[members].max()), win_microns / 4.0)

        s_local = cluster_soft_local.get(cid)
        before_bbox = _bbox(hard_xy, members, hw, hh)
        out: list[tuple[np.ndarray, np.ndarray | None, dict]] = []
        attempts = max(1, int(kick_count))
        variant_cycle = (
            "gaussian_gather", "compact", "border_compact",
            "low_displacement", "rot90", "flip_x",
        )
        for candidate_rank in range(attempts):
            anchor_rank = int(candidate_rank % max(1, len(anchors)))
            cand_ax, cand_ay, anchor_field = anchors[anchor_rank]
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
        if out:
            all_out.extend(out)
            clusters_used += 1
    return all_out
