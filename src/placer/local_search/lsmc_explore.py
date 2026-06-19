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

    cids = list(clusters.keys())
    heat = np.array([float(macro_cong[clusters[c]].mean()) for c in cids])
    if pick == "hot":
        order = [cids[i] for i in np.argsort(-heat)]
    else:
        med = float(np.median(heat))
        hot = [cids[i] for i in range(len(cids)) if heat[i] >= med]
        rest = [cids[i] for i in range(len(cids)) if heat[i] < med]
        rng.shuffle(hot)
        rng.shuffle(rest)
        order = hot + rest

    for cid in order:
        members = clusters[cid]
        members = members[movable[members]]
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

        soft_new = None
        if cluster_softs is not None and soft_xy is not None:
            s_arr = cluster_softs.get(cid)
            if s_arr is not None and s_arr.size:
                s_local = s_arr - n
                if soft_movable is not None:
                    s_local = s_local[soft_movable[s_local]]
                if s_local.size:
                    soft_new = soft_xy.copy()
                    soft_new[s_local, 0] = np.clip(
                        ax + rng.normal(0.0, jx, s_local.size),
                        soft_hw[s_local],
                        cw - soft_hw[s_local],
                    )
                    soft_new[s_local, 1] = np.clip(
                        ay + rng.normal(0.0, jy, s_local.size),
                        soft_hh[s_local],
                        ch - soft_hh[s_local],
                    )
        legal_hard = _will_legalize(kicked, movable, sizes, hw, hh, cw, ch, n, deadline=deadline)
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
            "soft_moved": int(
                0 if soft_new is None else np.count_nonzero(np.any(soft_new != soft_xy, axis=1))
            ),
        }
        if return_trace:
            return legal_hard, soft_new, trace
        return legal_hard, soft_new
    return None
