"""Post-coldspot survivor stage and final reporting segment."""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
import torch

from placer.local_search.cluster_decompress import hierarchy_quality_breakdown
from placer.local_search.gnn_trace import flush_plateau_events, log_gnn_event
from placer.local_search.survivor_search import _parallel_survivor_search
from placer.pipeline.hierarchy_context import PlacementState
from placer.scoring.exact import _exact_proxy


def run_post_coldspot_finalize(
    *,
    benchmark,
    plc,
    clusters: dict,
    csofts: dict,
    bridge_softs: dict,
    hierarchy,
    hierarchy_quality_metric_fn: Callable[[np.ndarray, dict], float],
    selected_seed_name: str,
    seed_rows: list[dict[str, object]],
    pre_relief: float,
    legal: np.ndarray,
    s_pos: np.ndarray,
    cur_proxy: float,
    best_h: np.ndarray,
    best_s: np.ndarray,
    best_score: float,
    movable: np.ndarray,
    n: int,
    n_soft: int,
    sizes: np.ndarray,
    hw: np.ndarray,
    hh: np.ndarray,
    soft_hw: np.ndarray,
    soft_hh: np.ndarray,
    cw: float,
    ch: float,
    region,
    soft_region,
    const: Any,
    group_weight: int,
    region_deadline: float | None,
    log_fn: Callable[[str], None],
    trace_pass_fn: Callable[..., None],
    record_plateau_fn: Callable[..., None],
    hard_valid_fn: Callable[[np.ndarray], bool],
    deadline_fn: Callable[[float, float | None], float | None],
) -> torch.Tensor:
    """Run survivor search and emit the final hierarchy floorplan output."""

    _log = log_fn
    _trace_pass = trace_pass_fn
    _record_plateau = record_plateau_fn
    _is_hard_valid = hard_valid_fn
    _deadline = deadline_fn

    def _hard_legality_margin(hard_xy: np.ndarray, eps: float) -> dict[str, float]:
        if hard_xy.shape[0] == 0:
            return {
                "pair_margin": float("inf"),
                "bounds_margin": float("inf"),
                "min_margin": float("inf"),
            }
        bounds_x = np.minimum(hard_xy[:, 0] - hw, cw - hw - hard_xy[:, 0])
        bounds_y = np.minimum(hard_xy[:, 1] - hh, ch - hh - hard_xy[:, 1])
        bounds_margin = float(min(np.min(bounds_x), np.min(bounds_y)))
        if hard_xy.shape[0] < 2:
            pair_margin = float("inf")
        else:
            dx = np.abs(hard_xy[:, None, 0] - hard_xy[None, :, 0])
            dy = np.abs(hard_xy[:, None, 1] - hard_xy[None, :, 1])
            sep_x = hw[:, None] + hw[None, :] + float(eps)
            sep_y = hh[:, None] + hh[None, :] + float(eps)
            clear = np.maximum(dx - sep_x, dy - sep_y)
            np.fill_diagonal(clear, np.inf)
            pair_margin = float(np.min(clear))
        return {
            "pair_margin": pair_margin,
            "bounds_margin": bounds_margin,
            "min_margin": min(pair_margin, bounds_margin),
        }

    soft_mov = movable[n : n + n_soft]

    pre_survivor_score = float(cur_proxy)
    survivor_t0 = time.monotonic()
    survivor_deadline = _deadline(float(const.HIER_SURVIVOR_BUDGET_S), region_deadline)
    legal, s_pos, survivor_acc, cur_proxy = _parallel_survivor_search(
        legal,
        s_pos,
        sizes[:n],
        hw,
        hh,
        soft_hw,
        soft_hh,
        float(cw),
        float(ch),
        movable[:n],
        soft_mov,
        int(n),
        plc,
        benchmark,
        float(cur_proxy),
        clusters,
        cluster_softs=csofts,
        bridge_softs=bridge_softs,
        hard_region=region,
        soft_region=soft_region,
        deadline=survivor_deadline,
    )
    if not _is_hard_valid(legal):
        legal, s_pos, cur_proxy = best_h.copy(), best_s.copy(), best_score
        survivor_acc = 0
    survivor_stats = getattr(_parallel_survivor_search, "last_stats", {})

    _log(
        f"  [hier] survivor search: {survivor_acc} accepts, "
        f"proxy {pre_survivor_score:.4f}->{cur_proxy:.4f}"
    )
    _trace_pass(
        "survivor_search",
        pre_survivor_score,
        float(cur_proxy),
        int(survivor_acc),
        quality=hierarchy_quality_metric_fn(legal, clusters),
        gpu_rank=bool(survivor_stats.get("gpu_rank", False)),
    )
    _record_plateau(
        "survivor_search",
        pre_survivor_score,
        float(cur_proxy),
        int(survivor_acc),
        time.monotonic() - survivor_t0,
        candidates=int(survivor_stats.get("candidates", 0)),
        legal=int(survivor_stats.get("legal", 0)),
        scored=int(survivor_stats.get("scored", 0)),
        quality=hierarchy_quality_metric_fn(legal, clusters),
        gpu_rank=bool(survivor_stats.get("gpu_rank", False)),
    )

    full = np.vstack([legal, s_pos]).astype(np.float32)
    full_proxy = float(_exact_proxy(torch.tensor(full, dtype=torch.float32), benchmark, plc))
    state = PlacementState(
        legal.copy(),
        s_pos.copy(),
        full_proxy,
    )
    out = torch.tensor(state.full().astype(np.float32), dtype=torch.float32)
    proxy = float(state.proxy)
    hq_breakdown = hierarchy_quality_breakdown(legal, clusters)
    margin_eps = float(const.HIER_LEGALITY_MARGIN_EPS)
    legality_margin = _hard_legality_margin(legal, margin_eps)

    log_gnn_event(
        "hier_final",
        benchmark=benchmark.name,
        proxy=float(proxy),
        pre_relief_proxy=float(pre_relief),
        hierarchy_quality=float(hq_breakdown["quality"]),
        hierarchy_quality_radius=float(hq_breakdown["radius"]),
        hierarchy_quality_bbox=float(hq_breakdown["bbox"]),
        hierarchy_quality_crowd=float(hq_breakdown["crowd"]),
        clusters=int(len(clusters)),
        hierarchy_edges=int(len(hierarchy.edges)),
        hierarchy_oversize_split=True,
        hierarchy_split_parents=int(len(hierarchy.split_parents)),
        seed_portfolio=True,
        selected_seed=selected_seed_name,
        seed_candidates=int(len(seed_rows)),
        additive_candidate_pools=True,
        legality_margin_audit=True,
        legality_margin_eps=float(margin_eps),
        legality_pair_margin=float(legality_margin["pair_margin"]),
        legality_bounds_margin=float(legality_margin["bounds_margin"]),
        legality_min_margin=float(legality_margin["min_margin"]),
        hierarchy_confidence_mean=float(
            np.mean(list(hierarchy.cluster_confidence.values()))
            if getattr(hierarchy, "cluster_confidence", None)
            else 0.0
        ),
        group_weight=int(group_weight),
    )
    _log(
        f"  [hier] {len(clusters)} clusters, {len(hierarchy.edges)} edges, "
        f"oversize=1, "
        f"seed={selected_seed_name}, "
        f"additive=1, "
        f"margin={float(legality_margin['min_margin']):.3f}, "
        f"weight={group_weight}: proxy={proxy:.4f} "
        f"(pre-relief {pre_relief:.4f}; hierarchy-preserving NON-proxy mode)"
    )
    flush_plateau_events()
    return out
