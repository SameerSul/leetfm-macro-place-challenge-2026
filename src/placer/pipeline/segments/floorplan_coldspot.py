"""Coldspot tightening segment for hierarchy floorplan."""

from __future__ import annotations

import os
import time
from typing import Any, Callable

import numpy as np
import torch

from placer.local_search.fields import _congestion_field
from placer.local_search.graph_tension import candidate_graph_edge_delta
from placer.local_search.lsmc_explore import _coldspot_cluster_kick_candidates
from placer.local_search.relocation import (
    _soft_relocation_moves,
    _micro_shift_polish,
)
from placer.pipeline.segments.floorplan_coldspot_candidates import (
    hot_cluster_fallback_candidates,
    rank_exact_coldspot_candidates,
)
from placer.pipeline.segments.floorplan_coldspot_refine import refine_coldspot_candidate
from placer.pipeline.segments.floorplan_coldspot_utils import (
    coldspot_field_gap,
    coldspot_local_regions,
    coldspot_min_window_avg,
    coldspot_opportunity,
    coldspot_window_stats,
    occupied_cells,
    remember_cold_cells,
)
from placer.scoring.exact import _exact_proxy
from placer.scoring.incremental import IncrementalScorer


def run_coldspot_tightening(
    *,
    benchmark,
    plc,
    clusters,
    csofts,
    bridge_softs,
    movable,
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
    legal: np.ndarray,
    s_pos: np.ndarray,
    const: Any,
    log_fn: Callable[[str], None],
    hard_valid_fn: Callable[[np.ndarray], bool],
    deadline_fn: Callable[[float, float | None], float | None],
    hierarchy_quality_metric_fn: Callable[[np.ndarray, dict], float],
    hier_soft_barrier_gain: float,
    hier_micro_shift_radius: int,
    hier_micro_shift_top: int,
    hier_micro_shift_min_gain: float,
    graph_tension_fn: Callable[[np.ndarray, np.ndarray | None], dict[int, float]] | None = None,
    graph_tension_weight: float = 0.0,
    graph_edges=None,
    seed_hard_xy: np.ndarray | None = None,
    graph_confidence: dict[int, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Run coldspot tightening and return the post-coldspot placement."""
    _additive_spare = (
        lambda deadline: deadline is None
        or time.monotonic() + float(const.HIER_ADDITIVE_MIN_SPARE_S) < deadline
    )
    adaptive_min_gain = float(const.HIER_PLATEAU_PROXY_GAIN)

    ck_budget = float(const.HIER_COLDSPOT_BUDGET)
    ck_total = float(const.HIER_COLDSPOT_TOTAL)
    ck_min_gain = float(const.HIER_COLDSPOT_MIN_GAIN)
    ck_min_gain = max(ck_min_gain, adaptive_min_gain)
    ck_quality_budget = float(const.HIER_COLDSPOT_QUALITY_BUDGET)
    ck_rounds = max(1, int(const.HIER_COLDSPOT_ROUNDS))
    ck_min_field_gap = max(
        float(const.HIER_COLDSPOT_MIN_FIELD_GAP),
        float(const.HIER_COLDSPOT_STRONG_MIN_FIELD_GAP),
    )
    ck_opportunity_min_score = float(const.HIER_COLDSPOT_OPPORTUNITY_MIN_SCORE)
    ck_opportunity_min_cold_cells = max(
        0,
        int(const.HIER_COLDSPOT_OPPORTUNITY_MIN_COLD_CELLS),
    )
    ck_max_dry_rounds = max(1, int(const.HIER_COLDSPOT_MAX_DRY_ROUNDS))
    ck_opportunity_top_clusters = max(
        1,
        int(const.HIER_COLDSPOT_OPPORTUNITY_TOP_CLUSTERS),
    )
    ck_deadline = deadline_fn(float(const.HIER_COLDSPOT_BUDGET_S))

    ck_whole_variants = max(1, int(const.HIER_COLDSPOT_WHOLE_VARIANTS))

    def _env_bool(name: str, default: bool) -> bool:
        raw = os.environ.get(name)
        if raw is None:
            return bool(default)
        return raw.strip() not in {"0", "false", "False", "no", "NO", "off", ""}

    def _env_int(name: str, default: int) -> int:
        raw = os.environ.get(name)
        if raw is None or not raw.strip():
            return int(default)
        return int(raw)

    def _env_float(name: str, default: float) -> float:
        raw = os.environ.get(name)
        if raw is None or not raw.strip():
            return float(default)
        return float(raw)

    ck_graph_anchor_weight = max(
        0.0,
        _env_float(
            "HIER_COLDSPOT_GRAPH_ANCHOR_WEIGHT",
            float(getattr(const, "HIER_COLDSPOT_GRAPH_ANCHOR_WEIGHT", 0.0)),
        ),
    )
    ck_prefilter_enabled = _env_bool(
        "HIER_GRAPH_PREFILTER",
        bool(getattr(const, "HIER_GRAPH_PREFILTER", False)),
    )
    ck_prefilter_low_tension = max(
        0.0,
        _env_float(
            "HIER_GRAPH_PREFILTER_LOW_TENSION",
            float(getattr(const, "HIER_GRAPH_PREFILTER_LOW_TENSION", 0.05)),
        ),
    )
    ck_prefilter_min_relief = max(
        0.0,
        _env_float(
            "HIER_GRAPH_PREFILTER_MIN_RELIEF",
            float(getattr(const, "HIER_GRAPH_PREFILTER_MIN_RELIEF", 0.0)),
        ),
    )
    ck_graph_delta_rank = _env_bool(
        "HIER_COLDSPOT_GRAPH_DELTA_RANK",
        bool(getattr(const, "HIER_COLDSPOT_GRAPH_DELTA_RANK", False)),
    )
    ck_graph_delta_rank_weight = (
        max(
            0.0,
            _env_float(
                "HIER_COLDSPOT_GRAPH_DELTA_WEIGHT",
                float(getattr(const, "HIER_COLDSPOT_GRAPH_DELTA_WEIGHT", 0.0)),
            ),
        )
        if ck_graph_delta_rank
        else 0.0
    )

    ck_egonet_enabled = _env_bool(
        "HIER_COLDSPOT_EGONET",
        bool(getattr(const, "HIER_COLDSPOT_EGONET", False)),
    )
    ck_egonet_max_neighbors = max(
        0,
        _env_int(
            "HIER_COLDSPOT_EGONET_MAX_NEIGHBORS",
            int(getattr(const, "HIER_COLDSPOT_EGONET_MAX_NEIGHBORS", 1)),
        ),
    )
    ck_egonet_max_hard = max(
        2,
        _env_int(
            "HIER_COLDSPOT_EGONET_MAX_HARD",
            int(getattr(const, "HIER_COLDSPOT_EGONET_MAX_HARD", 96)),
        ),
    )
    ck_egonet_max_neighbor_hard = max(
        1,
        _env_int(
            "HIER_COLDSPOT_EGONET_MAX_NEIGHBOR_HARD",
            int(getattr(const, "HIER_COLDSPOT_EGONET_MAX_NEIGHBOR_HARD", 32)),
        ),
    )
    ck_egonet_min_edge_weight = max(
        0.0,
        float(
            os.environ.get(
                "HIER_COLDSPOT_EGONET_MIN_EDGE_WEIGHT",
                str(getattr(const, "HIER_COLDSPOT_EGONET_MIN_EDGE_WEIGHT", 0.0)),
            )
        ),
    )
    ck_egonet_soft_mode = (
        os.environ.get(
            "HIER_COLDSPOT_EGONET_SOFT_MODE",
            str(getattr(const, "HIER_COLDSPOT_EGONET_SOFT_MODE", "none")),
        )
        .strip()
        .lower()
    )
    if ck_egonet_soft_mode not in {"anchor", "all", "none"}:
        ck_egonet_soft_mode = "none"
    ck_egonet_min_gain = max(
        0.0,
        float(
            os.environ.get(
                "HIER_COLDSPOT_EGONET_MIN_GAIN",
                str(getattr(const, "HIER_COLDSPOT_EGONET_MIN_GAIN", 0.001)),
            )
        ),
    )

    def _adaptive_gain(
        before: float,
        after: float,
    ) -> bool:
        return float(before) - float(after) > adaptive_min_gain

    ck_soft_only_min_gain = max(
        float(const.HIER_COLDSPOT_SOFT_ONLY_MIN_GAIN),
        adaptive_min_gain,
    )

    nr, nc = int(benchmark.grid_rows), int(benchmark.grid_cols)
    soft_mov = movable[n : n + n_soft]
    bias = float(const.REGION_BIAS)

    window_stats_cache: dict[tuple[int, int], tuple[float, float, float]] = {}

    def _window_stats(field: np.ndarray, win_cells: int) -> tuple[float, float, float]:
        return coldspot_window_stats(
            field=field,
            win_cells=win_cells,
            nr=nr,
            nc=nc,
            cw=cw,
            ch=ch,
            cache=window_stats_cache,
        )

    def _min_window_avg(field: np.ndarray, win_cells: int) -> float:
        return coldspot_min_window_avg(
            field=field,
            win_cells=win_cells,
            nr=nr,
            nc=nc,
            cw=cw,
            ch=ch,
            cache=window_stats_cache,
        )

    def _coldspot_field_gap(field: np.ndarray, hard_xy: np.ndarray) -> float:
        return coldspot_field_gap(
            field=field,
            hard_xy=hard_xy,
            sizes=sizes,
            movable=movable,
            clusters=clusters,
            n=n,
            nr=nr,
            nc=nc,
            cw=cw,
            ch=ch,
            min_window_avg=_min_window_avg,
        )

    def _coldspot_opportunity(
        field: np.ndarray,
        hard_xy: np.ndarray,
        soft_xy: np.ndarray,
    ) -> dict:
        cluster_priority = (
            graph_tension_fn(hard_xy, field) if graph_tension_fn is not None else None
        )
        return coldspot_opportunity(
            field=field,
            hard_xy=hard_xy,
            soft_xy=soft_xy,
            clusters=clusters,
            movable=movable,
            n=n,
            sizes=sizes,
            nr=nr,
            nc=nc,
            cw=cw,
            ch=ch,
            const=const,
            occupied_cells=_occupied_cells,
            window_stats=_window_stats,
            ck_opportunity_min_cold_cells=ck_opportunity_min_cold_cells,
            ck_min_field_gap=ck_min_field_gap,
            ck_opportunity_min_score=ck_opportunity_min_score,
            ck_opportunity_top_clusters=ck_opportunity_top_clusters,
            cluster_priority=cluster_priority,
            cluster_priority_weight=graph_tension_weight,
        )

    def _full(h: np.ndarray, soft: np.ndarray) -> torch.Tensor:
        return torch.tensor(np.vstack([h, soft]).astype(np.float32), dtype=torch.float32)

    def _remember_cold_cells(field: np.ndarray) -> np.ndarray:
        return remember_cold_cells(field, const)

    def _occupied_cells(hard_xy: np.ndarray, soft_xy: np.ndarray) -> np.ndarray:
        return occupied_cells(
            hard_xy=hard_xy,
            soft_xy=soft_xy,
            hw=hw,
            hh=hh,
            soft_hw=soft_hw,
            soft_hh=soft_hh,
            nr=nr,
            nc=nc,
            cw=cw,
            ch=ch,
        )

    def _coldspot_local_regions(
        hard_xy: np.ndarray,
        soft_xy: np.ndarray,
        cid: int,
    ) -> (
        "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict, np.ndarray, np.ndarray] | None"
    ):
        return coldspot_local_regions(
            hard_xy=hard_xy,
            soft_xy=soft_xy,
            cid=cid,
            clusters=clusters,
            csofts=csofts,
            bridge_softs=bridge_softs,
            movable=movable,
            n=n,
            n_soft=n_soft,
            soft_mov=soft_mov,
            hw=hw,
            hh=hh,
            soft_hw=soft_hw,
            soft_hh=soft_hh,
            cw=cw,
            ch=ch,
            nr=nr,
            nc=nc,
            cold_memory=cold_memory,
            const=const,
        )

    def _rank_exact_coldspot_candidates(candidates: list[dict], current_proxy: float) -> list[dict]:
        return rank_exact_coldspot_candidates(
            candidates,
            current_proxy,
            graph_delta_weight=ck_graph_delta_rank_weight,
        )

    def _hot_cluster_fallback_candidates(
        field: np.ndarray | None,
        hard_xy: np.ndarray,
        top_k: int,
    ) -> list[dict]:
        return hot_cluster_fallback_candidates(
            field=field,
            hard_xy=hard_xy,
            clusters=clusters,
            movable=movable,
            n=n,
            sizes=sizes,
            hw=hw,
            hh=hh,
            cw=cw,
            ch=ch,
            nr=nr,
            nc=nc,
            top_k=top_k,
        )

    def _refine_coldspot_candidate(
        hard_xy: np.ndarray,
        soft_xy: np.ndarray,
        trace: dict,
    ) -> "tuple[np.ndarray, np.ndarray, float, dict]":
        return refine_coldspot_candidate(
            hard_xy=hard_xy,
            soft_xy=soft_xy,
            trace=trace,
            benchmark=benchmark,
            plc=plc,
            n=n,
            sizes=sizes,
            hw=hw,
            hh=hh,
            soft_hw=soft_hw,
            soft_hh=soft_hh,
            cw=cw,
            ch=ch,
            const=const,
            region_bias=bias,
            deadline=ck_deadline,
            local_regions_fn=_coldspot_local_regions,
            additive_spare_fn=_additive_spare,
            hier_soft_barrier_gain=hier_soft_barrier_gain,
        )

    coldspot_candidate_softs = csofts
    bridge_by_cluster: dict[int, list[int]] = {}
    if bridge_softs:
        for soft_id, soft_cids in bridge_softs.items():
            sid = int(soft_id)
            if sid < 0 or sid >= n_soft or not bool(soft_mov[sid]):
                continue
            for cid_for_soft in np.asarray(soft_cids, dtype=np.int64):
                bridge_by_cluster.setdefault(int(cid_for_soft), []).append(n + sid)
    if bridge_by_cluster:
        merged_softs: dict[int, np.ndarray] = {}
        for cid in clusters.keys():
            parts = []
            owned = np.asarray(csofts.get(int(cid), []), dtype=np.int64)
            if owned.size:
                parts.append(owned)
            bridge = np.asarray(bridge_by_cluster.get(int(cid), []), dtype=np.int64)
            if bridge.size:
                parts.append(bridge)
            if parts:
                merged_softs[int(cid)] = np.unique(np.concatenate(parts)).astype(np.int64)
        if merged_softs:
            coldspot_candidate_softs = merged_softs

    graph_edge_neighbors: dict[int, list[tuple[float, int]]] = {}
    if graph_edges is not None:
        for edge in graph_edges:
            a = int(getattr(edge, "src", -1))
            b = int(getattr(edge, "dst", -1))
            if a not in clusters or b not in clusters:
                continue
            weight = max(0.0, float(getattr(edge, "weight", 0.0)))
            graph_edge_neighbors.setdefault(a, []).append((weight, b))
            graph_edge_neighbors.setdefault(b, []).append((weight, a))
        for cid in list(graph_edge_neighbors):
            graph_edge_neighbors[cid].sort(key=lambda row: (-row[0], row[1]))

    def _graph_anchor_targets(
        hard_xy: np.ndarray,
        tension_by_id: dict[int, float],
    ) -> tuple[dict[int, tuple[float, float]], dict[int, float]]:
        if ck_graph_anchor_weight <= 0.0 or not graph_edge_neighbors:
            return {}, {}
        centroids: dict[int, np.ndarray] = {}
        for cid, raw_members in clusters.items():
            members = np.asarray(raw_members, dtype=np.int64)
            members = members[(members >= 0) & (members < hard_xy.shape[0])]
            if members.size:
                centroids[int(cid)] = np.asarray(hard_xy[members].mean(axis=0), dtype=np.float64)
        targets: dict[int, tuple[float, float]] = {}
        strengths: dict[int, float] = {}
        for cid, neighbors in graph_edge_neighbors.items():
            if cid not in centroids:
                continue
            weighted = np.zeros(2, dtype=np.float64)
            total = 0.0
            for weight, nbr in neighbors:
                if nbr not in centroids:
                    continue
                w = max(0.0, float(weight))
                if w <= 0.0:
                    continue
                weighted += w * centroids[nbr]
                total += w
            strength = max(0.0, float(tension_by_id.get(int(cid), 0.0)))
            if total > 0.0 and strength > 0.0:
                target = weighted / total
                targets[int(cid)] = (float(target[0]), float(target[1]))
                strengths[int(cid)] = float(strength)
        return targets, strengths

    def _prefilter_coldspot_trace(trace: dict) -> str | None:
        if not ck_prefilter_enabled:
            return None
        tension = float(trace.get("graph_tension", 0.0) or 0.0)
        if tension > ck_prefilter_low_tension:
            return None
        source = float(trace.get("source_field", trace.get("cluster_heat", 0.0)) or 0.0)
        target = float(trace.get("target_field", source) or source)
        relief = source - target
        trace["local_relief"] = float(relief)
        if relief <= ck_prefilter_min_relief:
            return "prefilter_no_local_relief"
        return None

    def _annotate_graph_delta(before_h: np.ndarray, after_h: np.ndarray, trace: dict) -> None:
        if graph_edges is None:
            return
        raw_cluster = int(trace.get("egonet_anchor_cluster", trace.get("cluster", -1)))
        affected = trace.get("egonet_clusters")
        if affected is None:
            affected = [raw_cluster] if raw_cluster >= 0 else None
        field = _congestion_field(ck_scorer, nr, nc)
        stats = candidate_graph_edge_delta(
            before_h,
            after_h,
            clusters,
            graph_edges,
            cw=cw,
            ch=ch,
            field=field,
            seed_hard_xy=seed_hard_xy,
            confidence=graph_confidence,
            affected_clusters=affected,
            samples=max(2, int(getattr(const, "HIER_GRAPH_TENSION_CORRIDOR_SAMPLES", 9))),
        )
        trace.update(stats)

    def _egonet_candidate_view(
        preferred_ids,
    ) -> tuple[dict, dict, list[int], dict[int, dict]]:
        if not ck_egonet_enabled:
            return clusters, coldspot_candidate_softs, list(preferred_ids), {}
        if not graph_edge_neighbors:
            return clusters, coldspot_candidate_softs, list(preferred_ids), {}
        if ck_egonet_max_neighbors <= 0:
            return clusters, coldspot_candidate_softs, list(preferred_ids), {}
        view_clusters = {int(cid): np.asarray(mem, dtype=np.int64) for cid, mem in clusters.items()}
        view_softs = {
            int(cid): np.asarray(sidx, dtype=np.int64)
            for cid, sidx in coldspot_candidate_softs.items()
        }
        trace_by_cluster: dict[int, dict] = {}
        preferred_out: list[int] = []
        for raw_cid in preferred_ids:
            cid = int(raw_cid)
            if cid not in clusters:
                continue
            chosen = [cid]
            neighbor_rows = []
            for weight, nbr in graph_edge_neighbors.get(cid, []):
                if weight < ck_egonet_min_edge_weight:
                    continue
                nbr_size = int(np.asarray(clusters[int(nbr)], dtype=np.int64).size)
                if nbr_size > ck_egonet_max_neighbor_hard:
                    continue
                size_penalty = max(1.0, float(nbr_size) ** 0.5)
                neighbor_rows.append(
                    (-(float(weight) / size_penalty), -float(weight), nbr_size, int(nbr))
                )
            for _, _, nbr_size, nbr in sorted(neighbor_rows):
                candidate = chosen + [int(nbr)]
                total_hard = sum(
                    int(np.asarray(clusters[c], dtype=np.int64).size) for c in candidate
                )
                if total_hard > ck_egonet_max_hard:
                    continue
                chosen.append(int(nbr))
                if len(chosen) - 1 >= ck_egonet_max_neighbors:
                    break
            if len(chosen) <= 1:
                preferred_out.append(cid)
                continue
            synth_id = -100000 - cid
            hard_parts = [np.asarray(clusters[c], dtype=np.int64) for c in chosen]
            if ck_egonet_soft_mode == "all":
                soft_cluster_ids = chosen
            elif ck_egonet_soft_mode == "none":
                soft_cluster_ids = []
            else:
                soft_cluster_ids = [cid]
            soft_parts = []
            for soft_cid in soft_cluster_ids:
                soft_arr = np.asarray(coldspot_candidate_softs.get(soft_cid, []), dtype=np.int64)
                if soft_arr.size:
                    soft_parts.append(soft_arr)
            view_clusters[synth_id] = np.unique(np.concatenate(hard_parts)).astype(np.int64)
            if soft_parts:
                view_softs[synth_id] = np.unique(np.concatenate(soft_parts)).astype(np.int64)
            trace_by_cluster[synth_id] = {
                "egonet_candidate": True,
                "egonet_anchor_cluster": int(cid),
                "egonet_clusters": [int(c) for c in chosen],
                "egonet_neighbor_count": int(len(chosen) - 1),
                "egonet_member_count": int(view_clusters[synth_id].size),
                "egonet_neighbor_hard_count": int(
                    view_clusters[synth_id].size - hard_parts[0].size
                ),
                "egonet_soft_mode": ck_egonet_soft_mode,
            }
            preferred_out.append(synth_id)
            preferred_out.append(cid)
        for cid in preferred_ids:
            cid_i = int(cid)
            if cid_i not in preferred_out:
                preferred_out.append(cid_i)
        return view_clusters, view_softs, preferred_out, trace_by_cluster

    def _coldspot_target_cluster(
        point: tuple[float, float] | None,
        cluster_members: dict[int, np.ndarray],
        hard_xy: np.ndarray,
    ) -> int:
        if point is None:
            return -1
        px, py = float(point[0]), float(point[1])
        best_cluster = -1
        best_dist = np.inf
        for cid, members in cluster_members.items():
            cid_i = int(cid)
            if cid_i < 0:
                continue
            mem = np.asarray(members, dtype=np.int64)
            if mem.size == 0:
                continue
            pos = np.asarray(hard_xy[mem], dtype=np.float64)
            if pos.size == 0:
                continue
            cx, cy = float(pos[:, 0].mean()), float(pos[:, 1].mean())
            dist = (cx - px) ** 2 + (cy - py) ** 2
            if dist < best_dist:
                best_dist = dist
                best_cluster = cid_i
        return best_cluster

    def _coldspot_target_cluster_from_trace(
        trace: dict,
        hard_xy: np.ndarray,
        cluster_members: dict[int, np.ndarray],
    ) -> int:
        anchor_x = trace.get("anchor_x")
        anchor_y = trace.get("anchor_y")
        cx = float(trace.get("cluster_cx_after", float(trace.get("cluster_cx_before", 0.0))))
        cy = float(trace.get("cluster_cy_after", float(trace.get("cluster_cy_before", 0.0))))
        if anchor_x is not None and anchor_y is not None:
            try:
                if np.isfinite(float(anchor_x)) and np.isfinite(float(anchor_y)):
                    return _coldspot_target_cluster(
                        (float(anchor_x), float(anchor_y)),
                        cluster_members,
                        hard_xy,
                    )
            except (TypeError, ValueError):
                pass
        if np.isfinite(cx) and np.isfinite(cy):
            return _coldspot_target_cluster((cx, cy), cluster_members, hard_xy)
        return -1

    cur_h, cur_s = legal.copy(), s_pos.copy()
    base_proxy = float(_exact_proxy(_full(cur_h, cur_s), benchmark, plc))
    cur_proxy = base_proxy
    cur_quality = hierarchy_quality_metric_fn(cur_h, clusters)
    ck_scorer = IncrementalScorer(
        plc,
        benchmark,
        np.vstack([cur_h, cur_s]).astype(np.float64),
    )
    ck_rng = np.random.default_rng(0)

    ck_acc = 0
    cold_memory = np.zeros((nr, nc), dtype=bool)
    ck_dry_rounds = 0
    ck_run_fallbacks = True

    for _ in range(ck_rounds):
        if ck_deadline is not None and time.monotonic() >= ck_deadline:
            break

        ck_round_start = float(cur_proxy)
        field = _congestion_field(ck_scorer, nr, nc)
        if field is None:
            break

        cold_memory = _remember_cold_cells(field)
        opportunity = _coldspot_opportunity(field, cur_h, cur_s)
        field_gap = float(opportunity.get("field_gap", _coldspot_field_gap(field, cur_h)))
        if not bool(opportunity["run"]):
            if field_gap < ck_min_field_gap:
                reason = "field_gap_below_threshold"
            elif int(opportunity["open_cold_cells"]) < ck_opportunity_min_cold_cells:
                reason = "cold_capacity_below_threshold"
            else:
                reason = "opportunity_score_below_threshold"
            log_fn(
                f"  [hier] coldspot tightening: skipped, "
                f"reason={reason}, field_gap={field_gap:.4f}, "
                f"opp={float(opportunity['score']):.4f}, "
                f"open_cold={int(opportunity['open_cold_cells'])}"
            )
            ck_run_fallbacks = False
            break

        if ck_dry_rounds >= ck_max_dry_rounds:
            log_fn(f"  [hier] coldspot tightening: stopped after {ck_dry_rounds} dry rounds")
            ck_run_fallbacks = False
            break

        candidate_clusters, candidate_softs, preferred_ids, egonet_trace = _egonet_candidate_view(
            opportunity["cluster_ids"]
        )
        tension_by_id = opportunity.get("cluster_tension_by_id", {}) or {}
        graph_anchor_targets, graph_anchor_strength = _graph_anchor_targets(cur_h, tension_by_id)
        generated = _coldspot_cluster_kick_candidates(
            cur_h,
            sizes[:n],
            hw,
            hh,
            cw,
            ch,
            movable[:n],
            n,
            candidate_clusters,
            candidate_softs,
            cur_s,
            soft_hw,
            soft_hh,
            soft_mov,
            field,
            nr,
            nc,
            ck_rng,
            deadline=ck_deadline,
            pick="random",
            kick_count=ck_whole_variants,
            plc=plc,
            benchmark_name=benchmark.name,
            max_size=(max(64, int(ck_egonet_max_hard)) if ck_egonet_enabled else 64),
            preferred_cluster_ids=preferred_ids,
            max_clusters=min(ck_opportunity_top_clusters, len(opportunity["cluster_ids"])),
            egonet_trace_by_cluster=egonet_trace,
            graph_anchor_targets_by_cluster=graph_anchor_targets,
            graph_anchor_strength_by_cluster=graph_anchor_strength,
            graph_anchor_weight=ck_graph_anchor_weight,
        )
        if not generated:
            ck_dry_rounds += 1
            continue

        candidate_records: list[dict] = []
        for rank, (cand_h, cand_s, cand_trace) in enumerate(generated):
            cand_trace = dict(cand_trace)
            cand_trace["candidate_rank"] = int(rank)
            anchor_x = float(cand_trace.get("anchor_x", cur_h[0, 0]))
            anchor_y = float(cand_trace.get("anchor_y", cur_h[0, 1]))
            cand_trace["target_cluster"] = int(
                _coldspot_target_cluster((anchor_x, anchor_y), clusters, cur_h)
            )
            cand_trace["coldspot_permutation_id"] = int(rank)
            tension_by_id = opportunity.get("cluster_tension_by_id", {}) or {}
            tension_cluster = int(
                cand_trace.get("egonet_anchor_cluster", cand_trace.get("cluster", -1))
            )
            cand_trace.setdefault("graph_tension", float(tension_by_id.get(tension_cluster, 0.0)))
            _annotate_graph_delta(cur_h, cand_h, cand_trace)
            cand_soft = cand_s if cand_s is not None else cur_s
            prefilter_reason = _prefilter_coldspot_trace(cand_trace)
            if prefilter_reason is not None:
                continue
            candidate_records.append(
                {
                    "candidate_rank": int(rank),
                    "hard": cand_h,
                    "soft": cand_soft,
                    "trace": cand_trace,
                }
            )

        for cand in candidate_records:
            refined_h, refined_s, refined_proxy, refine_stats = _refine_coldspot_candidate(
                cand["hard"],
                cand["soft"],
                cand["trace"],
            )
            cand["trace"].update(refine_stats)
            cand["hard"] = refined_h
            cand["soft"] = refined_s
            cand["candidate_proxy_precomputed"] = float(refined_proxy)
            cand["trace"]["target_cluster"] = int(
                _coldspot_target_cluster_from_trace(cand["trace"], cand["hard"], clusters)
            )

        ranked_records = _rank_exact_coldspot_candidates(candidate_records, cur_proxy)

        accepted_record = None
        for cand in ranked_records:
            cand_h = cand["hard"]
            cand_s = cand["soft"]
            cand_proxy = float(cand.get("candidate_proxy_precomputed", cur_proxy))
            cand_quality = hierarchy_quality_metric_fn(cand_h, clusters)
            cand_min_gain = float(ck_min_gain)
            if bool(cand.get("trace", {}).get("egonet_candidate", False)):
                cand_min_gain = max(cand_min_gain, float(ck_egonet_min_gain))
            accepted = (
                cand_quality <= cur_quality + ck_quality_budget
                and cand_proxy <= cur_proxy + ck_budget
                and cand_proxy <= base_proxy + ck_total
                and cand_proxy < cur_proxy - cand_min_gain
            )
            cand["candidate_proxy"] = float(cand_proxy)
            cand["hierarchy_quality_after"] = float(cand_quality)
            if accepted:
                accepted_record = cand
                break

        if accepted_record is not None:
            cur_h = accepted_record["hard"]
            cur_s = accepted_record["soft"]
            cur_proxy = float(accepted_record["candidate_proxy"])
            cur_quality = float(accepted_record["hierarchy_quality_after"])
            ck_scorer = IncrementalScorer(
                plc,
                benchmark,
                np.vstack([cur_h, cur_s]).astype(np.float64),
            )
            refreshed_field = _congestion_field(ck_scorer, nr, nc)
            if refreshed_field is not None:
                cold_memory = _remember_cold_cells(refreshed_field)
            ck_acc += 1
            ck_dry_rounds = 0
        else:
            ck_dry_rounds += 1
        if not _adaptive_gain(ck_round_start, float(cur_proxy)):
            ck_run_fallbacks = False
            break

    if ck_run_fallbacks and ck_acc == 0 and (ck_deadline is None or time.monotonic() < ck_deadline):
        fallback_field = _congestion_field(ck_scorer, nr, nc)
        if fallback_field is not None:
            cold_memory = _remember_cold_cells(fallback_field)

        fallback_top_k = max(1, int(const.HIER_COLDSPOT_GRAPH_FALLBACK_TOP_K))
        fallback_records = _hot_cluster_fallback_candidates(
            fallback_field,
            cur_h,
            fallback_top_k,
        )

        accepted_fallback = None
        for rank, fallback_trace in enumerate(fallback_records):
            if ck_deadline is not None and time.monotonic() >= ck_deadline:
                break
            fallback_trace = dict(fallback_trace)
            fallback_trace["candidate_rank"] = int(rank)
            fallback_trace["graph_fallback"] = True
            fallback_h, fallback_s, fallback_proxy, refine_stats = _refine_coldspot_candidate(
                cur_h,
                cur_s,
                fallback_trace,
            )
            fallback_trace.update(refine_stats)
            fallback_quality = hierarchy_quality_metric_fn(fallback_h, clusters)
            fallback_proxy = float(fallback_proxy)
            fallback_valid = hard_valid_fn(fallback_h)
            accepted = (
                fallback_valid
                and fallback_quality <= cur_quality + ck_quality_budget
                and fallback_proxy <= cur_proxy + ck_budget
                and fallback_proxy <= base_proxy + ck_total
                and fallback_proxy < cur_proxy - ck_min_gain
            )
            candidate = {
                "hard": fallback_h,
                "soft": fallback_s,
                "candidate_proxy": float(fallback_proxy),
                "hierarchy_quality_after": float(fallback_quality),
                "accepted": bool(accepted),
            }
            if accepted and (
                accepted_fallback is None
                or fallback_proxy < float(accepted_fallback["candidate_proxy"])
            ):
                accepted_fallback = candidate
        if accepted_fallback is not None:
            cur_h = accepted_fallback["hard"]
            cur_s = accepted_fallback["soft"]
            cur_proxy = float(accepted_fallback["candidate_proxy"])
            cur_quality = float(accepted_fallback["hierarchy_quality_after"])
            ck_scorer = IncrementalScorer(
                plc,
                benchmark,
                np.vstack([cur_h, cur_s]).astype(np.float64),
            )
            refreshed_field = _congestion_field(ck_scorer, nr, nc)
            if refreshed_field is not None:
                cold_memory = _remember_cold_cells(refreshed_field)
            ck_acc += 1
            log_fn(
                f"  [hier] graph-local coldspot fallback: 1 accept, "
                f"proxy {base_proxy:.4f}->{cur_proxy:.4f}"
            )

    soft_only_acc = 0
    if (
        bool(const.HIER_COLDSPOT_SOFT_ONLY)
        and ck_run_fallbacks
        and ck_acc == 0
        and n_soft
        and bool(np.any(soft_mov))
        and soft_region is not None
        and (ck_deadline is None or time.monotonic() < ck_deadline)
    ):
        soft_only_before = float(cur_proxy)
        soft_only_field = _congestion_field(ck_scorer, nr, nc)
        soft_only_target_cells = 0
        if soft_only_field is not None:
            cold_memory = _remember_cold_cells(soft_only_field)
            target_mask = cold_memory & ~_occupied_cells(cur_h, cur_s)
            target_pool = np.flatnonzero(target_mask.ravel()).astype(np.int64)
            soft_only_target_cells = int(target_pool.size)
            if target_pool.size:
                cur_s, soft_only_acc, cur_proxy = _soft_relocation_moves(
                    cur_s,
                    soft_hw,
                    soft_hh,
                    cw,
                    ch,
                    n,
                    plc,
                    benchmark,
                    ck_scorer,
                    cur_proxy,
                    deadline=ck_deadline,
                    top_hot=max(1, int(const.HIER_COLDSPOT_SOFT_ONLY_TOP_K)),
                    n_targets=max(1, int(const.HIER_COLDSPOT_SOFT_ONLY_TARGETS)),
                    soft_movable=soft_mov,
                    use_density=False,
                    region_bbox=soft_region,
                    region_bias=bias,
                    region_escape_min=float(const.HIER_COLDSPOT_LOCAL_SOFT_ESCAPE_MIN),
                    accept_min_gain=max(
                        hier_soft_barrier_gain,
                        float(ck_soft_only_min_gain),
                    ),
                    target_pool=target_pool,
                    region_mask=target_mask,
                )
        log_fn(
            f"  [hier] coldspot soft-only fallback: {soft_only_acc} accepts, "
            f"targets={soft_only_target_cells}, "
            f"proxy {soft_only_before:.4f}->{cur_proxy:.4f}"
        )

    legal, s_pos = cur_h, cur_s
    log_fn(
        f"  [hier] coldspot tightening: {ck_acc} accepts, "
        f"quality={cur_quality:.4f}, proxy {base_proxy:.4f}->{cur_proxy:.4f}"
    )
    if region is not None and soft_region is not None:
        post_ck_micro_deadline = deadline_fn(
            float(const.HIER_POST_COLDSPOT_MICRO_SHIFT_BUDGET_S),
            ck_deadline,
        )
        post_ck_micro_acc = 0
        pre_post_ck_micro_score = cur_proxy
        full = np.vstack([legal, s_pos]).astype(np.float64)
        ck_scorer = IncrementalScorer(plc, benchmark, full.copy())
        for use_density in (False, True):
            pre_ck_micro = float(cur_proxy)
            legal, s_pos, got, cur_proxy = _micro_shift_polish(
                legal,
                s_pos,
                sizes[:n],
                hw,
                hh,
                soft_hw,
                soft_hh,
                cw,
                ch,
                movable[:n],
                soft_mov,
                n,
                plc,
                benchmark,
                ck_scorer,
                cur_proxy,
                hard_region=region,
                soft_region=soft_region,
                deadline=post_ck_micro_deadline,
                radius_cells=hier_micro_shift_radius,
                top_hot=hier_micro_shift_top,
                min_gain=hier_micro_shift_min_gain,
                use_density=use_density,
            )
            post_ck_micro_acc += got
            if pre_ck_micro - float(cur_proxy) <= adaptive_min_gain:
                break
        log_fn(
            f"  [hier] post-coldspot micro-shift replay: {post_ck_micro_acc} accepts, "
            f"proxy {pre_post_ck_micro_score:.4f}->{cur_proxy:.4f}"
        )

    return legal, s_pos, cur_proxy, cur_quality
