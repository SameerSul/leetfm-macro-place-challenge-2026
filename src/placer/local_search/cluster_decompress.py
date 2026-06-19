"""Exact-gated cluster decompression for congestion relief."""

from __future__ import annotations

import time

import numpy as np
import torch

from placer import constants as const
from placer.legalize.spiral import _will_legalize
from placer.local_search.fields import _congestion_field
from placer.scoring.exact import _exact_proxy


def hierarchy_quality_metric(hard_xy, clusters) -> float:
    """Cluster separation quality: lower means better-contained hierarchy."""
    if not clusters:
        return 0.0
    cids = list(clusters.keys())
    centroids = []
    radii = []
    for cid in cids:
        p = hard_xy[np.asarray(clusters[cid], dtype=np.int64)]
        centroids.append(p.mean(axis=0))
        radii.append(float(np.mean(np.hypot(p[:, 0] - p[:, 0].mean(), p[:, 1] - p[:, 1].mean()))))
    centroids = np.asarray(centroids, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    if len(cids) == 1:
        denom = max(float(np.hypot(np.ptp(hard_xy[:, 0]), np.ptp(hard_xy[:, 1]))), 1.0)
        return float(radii[0] / denom)
    d = np.hypot(
        centroids[:, None, 0] - centroids[None, :, 0],
        centroids[:, None, 1] - centroids[None, :, 1],
    )
    np.fill_diagonal(d, np.inf)
    nearest = np.maximum(np.min(d, axis=1), 1.0)
    return float(np.mean(radii / nearest))


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


def _full_tensor(hard_xy, soft_xy):
    return torch.tensor(np.vstack([hard_xy, soft_xy]).astype(np.float32), dtype=torch.float32)


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

    for _ in range(max(1, int(rounds))):
        if deadline is not None and time.monotonic() > deadline:
            break
        # Keep decompression congestion-driven; exact proxy still gates density side effects.
        field = _congestion_field(plc, nr, nc)
        if field is None:
            break
        local = _cell_values(cur_h, field, cw, ch)
        heat = []
        for cid, mem in clusters.items():
            mem = np.asarray(mem, dtype=np.int64)
            mov = mem[movable_h[mem]]
            if mov.size >= 2:
                heat.append((int(cid), float(local[mov].mean())))
        if not heat:
            break
        threshold = float(np.percentile([h for _, h in heat], hot_percentile))
        ordered = [cid for cid, h in sorted(heat, key=lambda x: -x[1]) if h >= threshold]
        centroids = _cluster_centroids(cur_h, clusters)
        accepted_round = False
        for cid in ordered:
            if deadline is not None and time.monotonic() > deadline:
                break
            mem_all = np.asarray(clusters[cid], dtype=np.int64)
            mem = mem_all[movable_h[mem_all]]
            if mem.size < 2:
                continue
            center = centroids[cid]
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
                cand_h = cur_h.copy()
                cand_s = cur_s.copy()
                vec = cand_h[mem] - center
                scale = np.array(
                    [1.0 + (factor - 1.0) * axis_x, 1.0 + (factor - 1.0) * axis_y],
                    dtype=np.float64,
                )
                cand_h[mem] = center + vec * scale
                cand_h[mem] = _clip_to_region(cand_h[mem], hard_region, mem, hw, hh, cw, ch)
                order = list(mem_all[np.argsort(-sizes[mem_all, 0] * sizes[mem_all, 1])])
                cand_h = _will_legalize(
                    cand_h, movable_h, sizes, hw, hh, cw, ch, n, deadline=deadline, order=order
                )

                soft_pidx = cluster_softs.get(cid)
                if soft_pidx is not None and len(soft_pidx):
                    sidx = np.asarray(soft_pidx, dtype=np.int64) - n
                    if soft_movable is not None:
                        sidx = sidx[soft_movable[sidx]]
                    if sidx.size:
                        svec = cand_s[sidx] - center
                        cand_s[sidx] = center + svec * scale
                        cand_s[sidx] = _clip_to_region(
                            cand_s[sidx], soft_region, sidx, soft_hw, soft_hh, cw, ch
                        )

                for sk, cids in (bridge_softs or {}).items():
                    sk = int(sk)
                    if soft_movable is not None and not soft_movable[sk]:
                        continue
                    if cid not in set(int(c) for c in cids):
                        continue
                    pts = [centroids[int(c)] for c in cids if int(c) in centroids]
                    if not pts:
                        continue
                    target = np.asarray(pts, dtype=np.float64).mean(axis=0)
                    cand_s[sk] = 0.55 * cand_s[sk] + 0.45 * target
                    cand_s[sk : sk + 1] = _clip_to_region(
                        cand_s[sk : sk + 1],
                        soft_region,
                        np.array([sk], dtype=np.int64),
                        soft_hw,
                        soft_hh,
                        cw,
                        ch,
                    )

                q = hierarchy_quality_metric(cand_h, clusters)
                if q > cur_quality + quality_budget:
                    continue
                score = float(_exact_proxy(_full_tensor(cand_h, cand_s), benchmark, plc))
                if score < best_score - min_proxy_gain:
                    cur_h, cur_s = cand_h, cand_s
                    best_score = score
                    cur_quality = q
                    accepts += 1
                    accepted_round = True
                    break
            if accepted_round:
                break
        if not accepted_round:
            break
    best_score = float(_exact_proxy(_full_tensor(cur_h, cur_s), benchmark, plc))
    return cur_h, cur_s, accepts, best_score, cur_quality
