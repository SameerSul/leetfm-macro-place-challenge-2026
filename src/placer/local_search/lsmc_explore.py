"""Coldspot cluster kick used by the hierarchy path."""

from __future__ import annotations

import numpy as np

from placer.legalize.spiral import _will_legalize
from placer.local_search.fields import coldest_window_anchor


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
    if not clusters or cong_field is None:
        return None
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
        return None

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
        member_area = float(np.sum(sizes[members, 0] * sizes[members, 1]))
        win_microns = float(np.sqrt(member_area / max(target_density, 1e-3)))
        win_cells = max(1, int(np.ceil(win_microns / min(cell_w, cell_h))))
        ax, ay = coldest_window_anchor(cong_field, nr, nc, cw, ch, win_cells)
        jx = max(float(hw[members].max()), win_microns / 4.0)
        jy = max(float(hh[members].max()), win_microns / 4.0)

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
        s_local = cluster_soft_local.get(cid)
        if s_local is not None and soft_xy is not None and s_local.size:
            soft_new = soft_xy.copy()
            sx = ax + rng.normal(0.0, jx, s_local.size)
            sy = ay + rng.normal(0.0, jy, s_local.size)
            soft_new[s_local, 0] = np.clip(sx, soft_hw[s_local], cw - soft_hw[s_local])
            soft_new[s_local, 1] = np.clip(sy, soft_hh[s_local], ch - soft_hh[s_local])
            soft_moved = int(np.count_nonzero(np.any(soft_new[s_local] != soft_xy[s_local], axis=1)))

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
        trace = {
            "cluster": int(cid),
            "movable_count": int(members.size),
            "member_area": float(member_area),
            "cluster_heat": float(macro_cong[members].mean()),
            "anchor_x": float(ax),
            "anchor_y": float(ay),
            "window_microns": float(win_microns),
            "window_cells": int(win_cells),
            "target_density": float(target_density),
            "pick": str(pick),
            "soft_moved": int(soft_moved if soft_new is not None else 0),
        }
        if return_trace:
            return legal_hard, soft_new, trace
        return legal_hard, soft_new
    return None
