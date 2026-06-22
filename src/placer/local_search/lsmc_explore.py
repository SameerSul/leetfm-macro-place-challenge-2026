"""Coldspot cluster kick used by the hierarchy path."""

from __future__ import annotations

import numpy as np

from placer.legalize.spiral import _will_legalize
from placer.local_search.fields import coldest_window_anchor


def _bbox(xy: np.ndarray, members: np.ndarray, hw: np.ndarray, hh: np.ndarray) -> list[float]:
    if members.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        float(np.min(xy[members, 0] - hw[members])),
        float(np.min(xy[members, 1] - hh[members])),
        float(np.max(xy[members, 0] + hw[members])),
        float(np.max(xy[members, 1] + hh[members])),
    ]


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
) -> list[tuple[np.ndarray, np.ndarray | None, dict]]:
    """Generate legalized coldspot kick candidates for one selected cluster/window."""
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
    if pick == "hot":
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

    for cid in order:
        members = cluster_members_movable[cid]
        if members.size < 2 or members.size > max_size:
            continue
        member_all = cluster_members[cid]
        member_area = float(np.sum(sizes[members, 0] * sizes[members, 1]))
        win_microns = float(np.sqrt(member_area / max(target_density, 1e-3)))
        win_cells = max(1, int(np.ceil(win_microns / min(cell_w, cell_h))))
        ax, ay = coldest_window_anchor(cong_field, nr, nc, cw, ch, win_cells)
        jx = max(float(hw[members].max()), win_microns / 4.0)
        jy = max(float(hh[members].max()), win_microns / 4.0)

        s_local = cluster_soft_local.get(cid)
        before_bbox = _bbox(hard_xy, members, hw, hh)
        out: list[tuple[np.ndarray, np.ndarray | None, dict]] = []
        attempts = max(1, int(kick_count))
        for candidate_rank in range(attempts):
            kicked = hard_xy.copy()
            kicked[members, 0] = np.clip(
                ax + rng.normal(0.0, jx, members.size), hw[members], cw - hw[members]
            )
            kicked[members, 1] = np.clip(
                ay + rng.normal(0.0, jy, members.size), hh[members], ch - hh[members]
            )
            moved_hard = bool(np.any(kicked[members] != hard_xy[members]))

            soft_new = None
            soft_moved = 0
            soft_disp = np.zeros(0, dtype=np.float64)
            if s_local is not None and soft_xy is not None and s_local.size:
                soft_new = soft_xy.copy()
                sx = ax + rng.normal(0.0, jx, s_local.size)
                sy = ay + rng.normal(0.0, jy, s_local.size)
                soft_new[s_local, 0] = np.clip(sx, soft_hw[s_local], cw - soft_hw[s_local])
                soft_new[s_local, 1] = np.clip(sy, soft_hh[s_local], ch - soft_hh[s_local])
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
                "anchor_x": float(ax),
                "anchor_y": float(ay),
                "x": float(ax),
                "y": float(ay),
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
            }
            out.append((legal_hard, soft_new, trace))
        if out:
            return out
    return []
