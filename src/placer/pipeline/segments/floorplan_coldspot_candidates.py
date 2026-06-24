"""Coldspot candidate ranking and fallback helpers for hierarchy coldspot tightening."""

from __future__ import annotations

import numpy as np


def graph_candidate_score(cand: dict) -> float:
    """Score a coldspot candidate using graph-region geometry and movement fields."""
    trace = cand.get("trace", {})
    if cand.get("is_noop", False):
        return -1.0e30

    source = float(trace.get("source_field", trace.get("cluster_heat", 0.0)) or 0.0)
    target = float(trace.get("target_field", source) or source)
    relief = source - target
    target_cells = float(trace.get("graph_target_cells", 0) or 0)
    region_cells = float(trace.get("graph_region_cells", 0) or 0)
    adaptive_cells = float(trace.get("adaptive_cold_cells", 0) or 0)
    hard_disp = float(trace.get("hard_disp_mean", 0.0) or 0.0)

    score = relief
    score += 0.020 * np.log1p(target_cells)
    score += 0.010 * np.log1p(adaptive_cells)
    score += 0.002 * np.log1p(region_cells)
    score -= 0.001 * hard_disp

    cand["graph_score"] = float(score)
    trace["graph_score"] = float(score)
    return float(score)


def rank_graph_coldspot_candidates(candidates: list[dict]) -> list[dict]:
    """Rank candidates by graph score descending."""
    scored = []
    for idx, cand in enumerate(candidates):
        scored.append((graph_candidate_score(cand), idx, cand))
    scored.sort(key=lambda row: (-row[0], row[1]))
    return [cand for _score, _idx, cand in scored]


def rank_exact_coldspot_candidates(
    candidates: list[dict],
    cur_proxy: float,
) -> list[dict]:
    """Rank candidates by exact pre-evaluated proxy plus graph score tie-break."""
    scored = []
    for idx, cand in enumerate(candidates):
        proxy = float(cand.get("candidate_proxy_precomputed", cur_proxy))
        graph_score = graph_candidate_score(cand)
        scored.append((proxy, -graph_score, idx, cand))
    scored.sort(key=lambda row: (row[0], row[1], row[2]))
    return [cand for _proxy, _graph, _idx, cand in scored]


def hot_cluster_fallback_candidates(
    field: np.ndarray | None,
    hard_xy: np.ndarray,
    clusters: dict,
    movable: np.ndarray,
    n: int,
    sizes: np.ndarray,
    hw: np.ndarray,
    hh: np.ndarray,
    cw: float,
    ch: float,
    nr: int,
    nc: int,
    top_k: int,
) -> list[dict]:
    """Build fallback candidates from hottest clusters when preferred candidates fail."""
    if field is not None:
        cell_w, cell_h = cw / nc, ch / nr
        mcol = np.clip((hard_xy[:, 0] / cell_w).astype(np.int64), 0, nc - 1)
        mrow = np.clip((hard_xy[:, 1] / cell_h).astype(np.int64), 0, nr - 1)
        macro_field = field[mrow, mcol]
    else:
        macro_field = np.zeros(n, dtype=np.float64)

    records = []
    for cid, raw_members in clusters.items():
        members = np.asarray(raw_members, dtype=np.int64)
        members = members[(members >= 0) & (members < n)]
        members = members[movable[:n][members]]
        if members.size < 2 or members.size > 64:
            continue

        area = float(np.sum(sizes[members, 0] * sizes[members, 1]))
        heat = float(np.mean(macro_field[members])) if field is not None else 0.0
        cx = float(np.mean(hard_xy[members, 0]))
        cy = float(np.mean(hard_xy[members, 1]))
        xlo = float(np.min(hard_xy[members, 0] - hw[members]))
        ylo = float(np.min(hard_xy[members, 1] - hh[members]))
        xhi = float(np.max(hard_xy[members, 0] + hw[members]))
        yhi = float(np.max(hard_xy[members, 1] + hh[members]))

        records.append(
            {
                "cluster": int(cid),
                "members": int(members.size),
                "cluster_area": float(area),
                "cluster_heat": float(heat),
                "source_field": float(heat),
                "target_field": float(heat),
                "cluster_cx_before": float(cx),
                "cluster_cy_before": float(cy),
                "cluster_cx_after": float(cx),
                "cluster_cy_after": float(cy),
                "cluster_bbox_before": (xlo, ylo, xhi, yhi),
                "cluster_bbox_after": (xlo, ylo, xhi, yhi),
                "hard_disp_mean": 0.0,
                "hard_disp_max": 0.0,
                "hard_dx_mean": 0.0,
                "hard_dy_mean": 0.0,
                "soft_moved": 0,
                "soft_disp_mean": 0.0,
                "soft_disp_max": 0.0,
            }
        )
    records.sort(
        key=lambda row: (
            -float(row["cluster_heat"]),
            -float(row["cluster_area"]),
            int(row["cluster"]),
        )
    )
    return records[: max(0, int(top_k))]
