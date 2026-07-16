"""Post-coldspot survivor stage and final reporting segment."""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
import torch

from placer.local_search.cluster_decompress import hierarchy_quality_breakdown
from placer.local_search.fields import (
    cold_connected_component_target_pool,
    weighted_congestion_field,
)
from placer.local_search.gnn_trace import flush_plateau_events, log_gnn_event
from placer.local_search.hierarchy_quality import (
    hierarchy_quality_vector,
    hierarchy_vector_contract,
    hierarchy_vector_limits,
)
from placer.local_search.hierarchy_swaps import _region_bounded_swap_relief
from placer.local_search.relocation import (
    _micro_shift_polish,
    _relocation_moves,
    _soft_relocation_moves,
)
from placer.pipeline.hierarchy_context import PlacementState
from placer.scoring.exact import _exact_proxy
from placer.scoring.incremental import IncrementalScorer


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
    seed_hierarchy_quality: float,
    seed_hierarchy_vector: dict[str, float],
    legal: np.ndarray,
    s_pos: np.ndarray,
    cur_proxy: float,
    best_h: np.ndarray,
    best_s: np.ndarray,
    best_score: float,
    audit_h: np.ndarray,
    audit_s: np.ndarray,
    audit_score: float,
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

    def _small_design_polish_eligible() -> bool:
        total = int(n) + int(n_soft)
        if int(n) < int(const.HIER_SMALL_DESIGN_HARD_MIN):
            return False
        if int(n) > int(const.HIER_SMALL_DESIGN_HARD_MAX):
            return False
        if total > int(const.HIER_SMALL_DESIGN_MACRO_MAX):
            return False
        return bool(np.all(np.asarray(movable[:n], dtype=bool)))

    def _full_hard_region(i: int) -> tuple[float, float, float, float]:
        return (float(hw[i]), float(hh[i]), float(cw - hw[i]), float(ch - hh[i]))

    def _full_soft_region(k: int) -> tuple[float, float, float, float]:
        return (
            float(soft_hw[k]),
            float(soft_hh[k]),
            float(cw - soft_hw[k]),
            float(ch - soft_hh[k]),
        )

    def _cluster_heat(field: np.ndarray | None, hard_xy: np.ndarray) -> dict[int, float]:
        if field is None or not clusters:
            return {}
        nr, nc = field.shape
        cell_w, cell_h = float(cw) / float(nc), float(ch) / float(nr)
        out: dict[int, float] = {}
        for cid, mem_raw in clusters.items():
            mem = np.asarray(mem_raw, dtype=np.int64)
            mem = mem[(mem >= 0) & (mem < int(n))]
            if mem.size == 0:
                continue
            ci = np.clip((hard_xy[mem, 0] / cell_w).astype(np.int64), 0, nc - 1)
            ri = np.clip((hard_xy[mem, 1] / cell_h).astype(np.int64), 0, nr - 1)
            out[int(cid)] = float(np.mean(field[ri, ci]))
        return out

    def _small_design_released_regions(
        scorer: IncrementalScorer,
        hard_xy: np.ndarray,
    ) -> tuple[np.ndarray | None, np.ndarray | None, list[int]]:
        confidence = getattr(hierarchy, "cluster_confidence", None) or {}
        if not confidence or region is None:
            return region, soft_region, []
        threshold = float(const.HIER_SMALL_DESIGN_RELEASE_CONFIDENCE_MAX)
        max_clusters = max(1, int(const.HIER_SMALL_DESIGN_RELEASE_MAX_CLUSTERS))
        field = weighted_congestion_field(
            scorer, int(benchmark.grid_rows), int(benchmark.grid_cols)
        )
        heat_by_cluster = _cluster_heat(field, hard_xy)
        heat_values = np.asarray(list(heat_by_cluster.values()), dtype=np.float64)
        heat_min = float(np.min(heat_values)) if heat_values.size else 0.0
        heat_span = max(float(np.max(heat_values)) - heat_min, 1e-12) if heat_values.size else 1.0
        weakest_k = max(0, int(const.HIER_SMALL_DESIGN_RELEASE_WEAKEST_K))
        if weakest_k <= 0:
            return region, soft_region, []
        weakest_rows = sorted(
            ((float(conf), int(cid)) for cid, conf in confidence.items() if int(cid) in clusters),
            key=lambda item: (item[0], item[1]),
        )[:weakest_k]
        release_rows = []
        for conf, cid in weakest_rows:
            if conf > threshold:
                continue
            heat = float(heat_by_cluster.get(cid, heat_min))
            heat_norm = (heat - heat_min) / heat_span
            weak = max(0.0, threshold - conf) / max(threshold, 1e-12)
            release_rows.append((heat_norm, weak, -conf, cid))
        release_limit = min(max_clusters, len(release_rows), weakest_k)
        released = [
            int(cid) for _heat, _weak, _neg_conf, cid in sorted(release_rows, reverse=True)
        ][:release_limit]
        if not released:
            return region, soft_region, []

        hard_out = np.array(region, dtype=np.float64, copy=True)
        for cid in released:
            for i_raw in np.asarray(clusters.get(int(cid), []), dtype=np.int64):
                i = int(i_raw)
                if 0 <= i < int(n):
                    hard_out[i, :] = _full_hard_region(i)

        soft_out = None
        if soft_region is not None and int(n_soft) > 0:
            soft_out = np.array(soft_region, dtype=np.float64, copy=True)
            soft_ids: set[int] = set()
            for cid in released:
                for p_raw in np.asarray(csofts.get(int(cid), []), dtype=np.int64):
                    k = int(p_raw) - int(n)
                    if 0 <= k < int(n_soft):
                        soft_ids.add(k)
            for k_raw, cids_raw in (bridge_softs or {}).items():
                cids = {int(x) for x in np.asarray(cids_raw, dtype=np.int64)}
                if cids.intersection(released):
                    k = int(k_raw)
                    if 0 <= k < int(n_soft):
                        soft_ids.add(k)
            for k in sorted(soft_ids):
                soft_out[k, :] = _full_soft_region(k)
        return hard_out, soft_out, released

    def _small_design_target_pool(scorer: IncrementalScorer) -> dict[str, np.ndarray] | None:
        field = weighted_congestion_field(
            scorer,
            int(benchmark.grid_rows),
            int(benchmark.grid_cols),
        )
        if field is None:
            return None
        pool = cold_connected_component_target_pool(
            field,
            cold_percentile=float(const.HIER_COLD_COMPONENT_PCT),
            max_components=max(1, int(const.HIER_COLD_COMPONENT_MAX_COMPONENTS)),
            min_cells=max(1, int(const.HIER_COLD_COMPONENT_MIN_CELLS)),
            size_weight=float(const.HIER_COLD_COMPONENT_SIZE_WEIGHT),
        )
        if np.asarray(pool.get("indices", []), dtype=np.int64).size == 0:
            return None
        return pool

    soft_mov = movable[n : n + n_soft]
    audit_budget = max(0.0, float(getattr(const, "HIER_FINAL_HIER_AUDIT_MAX_DEGRADATION", 0.0)))
    audit_limit = float(seed_hierarchy_quality) + audit_budget
    hierarchy_contract_limits = hierarchy_vector_limits(
        seed_hierarchy_vector,
        const.HIER_VECTOR_CONTRACT_ABS_SLACK,
        float(const.HIER_VECTOR_CONTRACT_REL_SLACK),
    )

    def _placement_hierarchy_vector(hard_xy: np.ndarray, soft_xy: np.ndarray):
        return hierarchy_quality_vector(
            hard_xy,
            soft_xy,
            clusters,
            csofts,
            bridge_softs,
            hierarchy.edges,
            cw,
            ch,
        )

    def _vector_contract(hard_xy: np.ndarray, soft_xy: np.ndarray):
        vector = _placement_hierarchy_vector(hard_xy, soft_xy)
        passed, violations = hierarchy_vector_contract(vector, hierarchy_contract_limits)
        return passed, vector, violations

    audit_checkpoint_h = np.array(audit_h, dtype=np.float64, copy=True)
    audit_checkpoint_s = np.array(audit_s, dtype=np.float64, copy=True)
    audit_checkpoint_score = float(audit_score)
    audit_checkpoint_quality = hierarchy_quality_metric_fn(audit_checkpoint_h, clusters)
    audit_checkpoint_vector_passed, audit_checkpoint_vector, _checkpoint_violations = (
        _vector_contract(audit_checkpoint_h, audit_checkpoint_s)
    )

    if (
        not _is_hard_valid(audit_checkpoint_h)
        or audit_checkpoint_quality > audit_limit
        or not audit_checkpoint_vector_passed
    ):
        for cand_h, cand_s, cand_score in (
            (legal, s_pos, cur_proxy),
            (best_h, best_s, best_score),
        ):
            cand_quality = hierarchy_quality_metric_fn(cand_h, clusters)
            cand_vector_passed, cand_vector, _violations = _vector_contract(cand_h, cand_s)
            if _is_hard_valid(cand_h) and cand_quality <= audit_limit and cand_vector_passed:
                audit_checkpoint_h = np.array(cand_h, dtype=np.float64, copy=True)
                audit_checkpoint_s = np.array(cand_s, dtype=np.float64, copy=True)
                audit_checkpoint_score = float(cand_score)
                audit_checkpoint_quality = float(cand_quality)
                audit_checkpoint_vector = dict(cand_vector)
                audit_checkpoint_vector_passed = True
                break

    def _update_audit_checkpoint(
        hard_xy: np.ndarray,
        soft_xy: np.ndarray,
        score: float,
    ) -> tuple[bool, float]:
        nonlocal audit_checkpoint_h, audit_checkpoint_s
        nonlocal audit_checkpoint_score, audit_checkpoint_quality, audit_checkpoint_vector
        if not _is_hard_valid(hard_xy):
            return False, float("inf")
        quality = hierarchy_quality_metric_fn(hard_xy, clusters)
        if quality > audit_limit:
            return False, float(quality)
        vector_passed, vector, _violations = _vector_contract(hard_xy, soft_xy)
        if not vector_passed:
            return False, float(quality)
        if float(score) < audit_checkpoint_score - 1e-9 or audit_checkpoint_quality > audit_limit:
            audit_checkpoint_h = np.array(hard_xy, dtype=np.float64, copy=True)
            audit_checkpoint_s = np.array(soft_xy, dtype=np.float64, copy=True)
            audit_checkpoint_score = float(score)
            audit_checkpoint_quality = float(quality)
            audit_checkpoint_vector = dict(vector)
            return True, float(quality)
        return False, float(quality)

    if _small_design_polish_eligible():
        small_t0 = time.monotonic()
        small_deadline = _deadline(float(const.HIER_SMALL_DESIGN_BUDGET_S), None)
        small_before = float(cur_proxy)
        small_acc = 0
        small_stats = {"candidates": 0, "legal": 0, "scored": 0, "accepts": 0}
        swap_stats = {
            "hh_scores": 0,
            "hh_accepts": 0,
            "hh_escape_accepts": 0,
            "hs_scores": 0,
            "hs_accepts": 0,
            "hs_escape_accepts": 0,
            "ss_scores": 0,
            "ss_accepts": 0,
            "ss_escape_accepts": 0,
            "proxy_gain": 0.0,
        }
        full_current = np.vstack([legal, s_pos]).astype(np.float64)
        small_scorer = IncrementalScorer(plc, benchmark, full_current.copy())
        best_small_score = float(cur_proxy)
        best_small_legal = np.array(legal, dtype=np.float64, copy=True)
        best_small_s_pos = np.array(s_pos, dtype=np.float64, copy=True)
        best_small_quality = hierarchy_quality_metric_fn(legal, clusters)
        small_audit_rollback = False

        def _remember_small_state() -> bool:
            nonlocal best_small_score, best_small_legal, best_small_s_pos
            nonlocal best_small_quality
            cur_quality_local = hierarchy_quality_metric_fn(legal, clusters)
            if cur_quality_local > audit_limit:
                return False
            vector_passed, _vector, _violations = _vector_contract(legal, s_pos)
            if not vector_passed:
                return False
            if float(cur_proxy) < best_small_score - 1e-9 or best_small_quality > audit_limit:
                best_small_score = float(cur_proxy)
                best_small_legal = np.array(legal, dtype=np.float64, copy=True)
                best_small_s_pos = np.array(s_pos, dtype=np.float64, copy=True)
                best_small_quality = float(cur_quality_local)
                return True
            return False

        def _restore_small_if_needed() -> bool:
            nonlocal legal, s_pos, cur_proxy, small_scorer, small_audit_rollback
            cur_quality_local = hierarchy_quality_metric_fn(legal, clusters)
            vector_passed, _vector, _violations = _vector_contract(legal, s_pos)
            if cur_quality_local <= audit_limit and vector_passed:
                _remember_small_state()
                return False
            legal = best_small_legal.copy()
            s_pos = best_small_s_pos.copy()
            cur_proxy = float(best_small_score)
            small_scorer = IncrementalScorer(
                plc,
                benchmark,
                np.vstack([legal, s_pos]).astype(np.float64),
            )
            small_audit_rollback = True
            return True

        released_region, released_soft_region, released_cids = _small_design_released_regions(
            small_scorer,
            legal,
        )
        total_macros = max(1, int(n) + int(n_soft))
        nets_per_macro = float(getattr(benchmark, "num_nets", 0)) / float(total_macros)
        high_net_lane = nets_per_macro >= float(const.HIER_SMALL_DESIGN_HIGH_NETS_PER_MACRO)
        no_release_low_net_lane = bool((not high_net_lane) and not released_cids)
        hard_top = int(
            const.HIER_SMALL_DESIGN_HIGH_HARD_TOP_K
            if high_net_lane
            else const.HIER_SMALL_DESIGN_LOW_HARD_TOP_K
        )
        hard_targets = int(
            const.HIER_SMALL_DESIGN_HIGH_HARD_TARGETS
            if high_net_lane
            else const.HIER_SMALL_DESIGN_LOW_HARD_TARGETS
        )
        hard_propose_top_m = int(
            const.HIER_SMALL_DESIGN_HIGH_HARD_PROPOSE_TOP_M
            if high_net_lane
            else const.HIER_SMALL_DESIGN_LOW_HARD_PROPOSE_TOP_M
        )
        soft_top = int(
            const.HIER_SMALL_DESIGN_HIGH_SOFT_TOP_K
            if high_net_lane
            else const.HIER_SMALL_DESIGN_LOW_SOFT_TOP_K
        )
        soft_targets = int(
            const.HIER_SMALL_DESIGN_HIGH_SOFT_TARGETS
            if high_net_lane
            else const.HIER_SMALL_DESIGN_LOW_SOFT_TARGETS
        )
        small_swap_soft_k = int(const.HIER_SMALL_DESIGN_SWAP_SOFT_K)
        if no_release_low_net_lane:
            hard_top = int(const.HIER_SMALL_DESIGN_NO_RELEASE_LOW_NET_HARD_TOP_K)
            hard_targets = int(const.HIER_SMALL_DESIGN_NO_RELEASE_LOW_NET_HARD_TARGETS)
            hard_propose_top_m = int(const.HIER_SMALL_DESIGN_NO_RELEASE_LOW_NET_HARD_PROPOSE_TOP_M)
            soft_top = int(const.HIER_SMALL_DESIGN_NO_RELEASE_LOW_NET_SOFT_TOP_K)
            soft_targets = int(const.HIER_SMALL_DESIGN_NO_RELEASE_LOW_NET_SOFT_TARGETS)
            small_swap_soft_k = int(const.HIER_SMALL_DESIGN_NO_RELEASE_LOW_NET_SWAP_SOFT_K)
        min_gain = float(const.HIER_SMALL_DESIGN_MIN_GAIN)
        escape_min = float(const.HIER_SMALL_DESIGN_RELEASE_ESCAPE_MIN)
        rounds = max(1, int(const.HIER_SMALL_DESIGN_ROUNDS))
        for _round in range(rounds):
            round_before = float(cur_proxy)
            component_target_pool = _small_design_target_pool(small_scorer)
            hard_reloc_before = float(cur_proxy)
            for use_density in (False, True):
                legal, got, cur_proxy = _relocation_moves(
                    legal,
                    sizes[:n],
                    hw,
                    hh,
                    float(cw),
                    float(ch),
                    movable[:n],
                    int(n),
                    plc,
                    benchmark,
                    small_scorer,
                    float(cur_proxy),
                    deadline=small_deadline,
                    top_hot=max(1, hard_top),
                    n_targets=max(1, hard_targets),
                    use_density=use_density,
                    propose_all=True,
                    propose_top_m=max(1, hard_propose_top_m),
                    region_bbox=released_region,
                    region_bias=float(const.REGION_BIAS),
                    region_escape_min=escape_min,
                    propose_accept_min_gain=min_gain,
                    target_pool=component_target_pool,
                )
                small_acc += int(got)
                for key in small_stats:
                    small_stats[key] += int(
                        getattr(_relocation_moves, "last_stats", {}).get(key, 0)
                    )
                if small_deadline is not None and time.monotonic() >= small_deadline:
                    break
            _restore_small_if_needed()
            hard_reloc_gain = max(0.0, float(hard_reloc_before) - float(cur_proxy))
            if (
                int(n_soft) > 0
                and bool(np.any(soft_mov))
                and (small_deadline is None or time.monotonic() < small_deadline)
            ):
                for use_density in (False, True):
                    s_pos, got, cur_proxy = _soft_relocation_moves(
                        s_pos,
                        soft_hw,
                        soft_hh,
                        float(cw),
                        float(ch),
                        int(n),
                        plc,
                        benchmark,
                        small_scorer,
                        float(cur_proxy),
                        deadline=small_deadline,
                        top_hot=max(1, soft_top),
                        n_targets=max(1, soft_targets),
                        soft_movable=soft_mov,
                        use_density=use_density,
                        region_bbox=released_soft_region,
                        region_bias=float(const.REGION_BIAS),
                        region_escape_min=escape_min,
                        accept_min_gain=min_gain,
                        target_pool=component_target_pool,
                        wl_prefilter=float(const.HIER_STRONG_SOFT_REPAIR_WL_PREFILTER),
                        gpu_batch_rank=True,
                    )
                    small_acc += int(got)
                    for key in small_stats:
                        small_stats[key] += int(
                            getattr(_soft_relocation_moves, "last_stats", {}).get(key, 0)
                        )
                    if small_deadline is not None and time.monotonic() >= small_deadline:
                        break
                _restore_small_if_needed()
            if small_deadline is not None and time.monotonic() >= small_deadline:
                break
            if released_cids and hard_reloc_gain >= min_gain:
                for use_density in (False, True):
                    hard_swap_before = float(cur_proxy)
                    legal, s_pos, got, cur_proxy, stats = _region_bounded_swap_relief(
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
                        benchmark,
                        small_scorer,
                        float(cur_proxy),
                        released_region,
                        released_soft_region,
                        deadline=small_deadline,
                        rounds=1,
                        hard_k=max(1, int(const.HIER_SMALL_DESIGN_HARD_SWAP_K)),
                        soft_k=max(1, int(small_swap_soft_k)),
                        region_bias=float(const.REGION_BIAS),
                        escape_min=escape_min,
                        min_gain=float(const.HIER_SMALL_DESIGN_SWAP_MIN_GAIN),
                        soft_barrier_gain=0.0,
                        min_field_relief=0.0,
                        enable_hh=True,
                        enable_hs=False,
                        enable_ss=False,
                        use_density=use_density,
                        hierarchy_quality_fn=lambda cand_h: hierarchy_quality_metric_fn(
                            cand_h,
                            clusters,
                        ),
                        hierarchy_quality_limit=audit_limit,
                    )
                    small_acc += int(got)
                    for key, value in stats.items():
                        if key in swap_stats:
                            swap_stats[key] += value
                    _restore_small_if_needed()
                    if float(hard_swap_before) - float(cur_proxy) <= min_gain:
                        break
                    if small_deadline is not None and time.monotonic() >= small_deadline:
                        break
            if small_deadline is not None and time.monotonic() >= small_deadline:
                break
            for use_density in (False, True):
                swap_before = float(cur_proxy)
                legal, s_pos, got, cur_proxy, stats = _region_bounded_swap_relief(
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
                    benchmark,
                    small_scorer,
                    float(cur_proxy),
                    released_region,
                    released_soft_region,
                    deadline=small_deadline,
                    rounds=1,
                    hard_k=max(1, int(const.HIER_SMALL_DESIGN_SWAP_HARD_K)),
                    soft_k=max(1, int(small_swap_soft_k)),
                    region_bias=float(const.REGION_BIAS),
                    escape_min=escape_min,
                    min_gain=float(const.HIER_SMALL_DESIGN_SWAP_MIN_GAIN),
                    soft_barrier_gain=0.0,
                    min_field_relief=0.0,
                    enable_hh=False,
                    enable_hs=True,
                    enable_ss=True,
                    use_density=use_density,
                    hierarchy_quality_fn=lambda cand_h: hierarchy_quality_metric_fn(
                        cand_h,
                        clusters,
                    ),
                    hierarchy_quality_limit=audit_limit,
                )
                small_acc += int(got)
                for key, value in stats.items():
                    if key in swap_stats:
                        swap_stats[key] += value
                _restore_small_if_needed()
                if float(swap_before) - float(cur_proxy) <= min_gain:
                    break
                if small_deadline is not None and time.monotonic() >= small_deadline:
                    break
            if small_deadline is not None and time.monotonic() >= small_deadline:
                break
            legal, s_pos, got, cur_proxy = _micro_shift_polish(
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
                small_scorer,
                float(cur_proxy),
                hard_region=released_region,
                soft_region=released_soft_region,
                deadline=small_deadline,
                radius_cells=max(1, int(const.HIER_MICRO_SHIFT_RADIUS)),
                top_hot=max(1, int(const.HIER_MICRO_SHIFT_TOP)),
                min_gain=min_gain,
                use_density=bool(high_net_lane),
            )
            small_acc += int(got)
            _restore_small_if_needed()
            if float(round_before) - float(cur_proxy) <= min_gain:
                break
        cur_quality = hierarchy_quality_metric_fn(legal, clusters)
        cur_vector_passed, _cur_vector, _violations = _vector_contract(legal, s_pos)
        if float(cur_proxy) < best_small_score and cur_quality <= audit_limit and cur_vector_passed:
            best_small_score = float(cur_proxy)
            best_small_legal = np.array(legal, dtype=np.float64, copy=True)
            best_small_s_pos = np.array(s_pos, dtype=np.float64, copy=True)
            best_small_quality = float(cur_quality)
        elif (
            cur_quality > audit_limit
            or not cur_vector_passed
            or best_small_score < float(cur_proxy)
        ):
            legal = best_small_legal
            s_pos = best_small_s_pos
            cur_proxy = float(best_small_score)
            small_audit_rollback = cur_quality > audit_limit or not cur_vector_passed
            small_scorer = IncrementalScorer(
                plc,
                benchmark,
                np.vstack([legal, s_pos]).astype(np.float64),
            )
        if not _is_hard_valid(legal):
            legal, s_pos, cur_proxy = best_h.copy(), best_s.copy(), float(best_score)
            small_acc = 0
        elif float(cur_proxy) < float(best_score) - 1e-9:
            best_h, best_s, best_score = legal.copy(), s_pos.copy(), float(cur_proxy)
        _update_audit_checkpoint(legal, s_pos, float(cur_proxy))
        _log(
            f"  [hier] small-design released polish: {small_acc} accepts, "
            f"proxy {small_before:.4f}->{float(cur_proxy):.4f}, "
            f"lane={'high_net' if high_net_lane else 'low_net'}, "
            f"released={len(released_cids)}, "
            f"swap_gain={float(swap_stats['proxy_gain']):.4f}, "
            f"audit_rollback={int(small_audit_rollback)}"
        )
        trace_extra = {
            "quality": hierarchy_quality_metric_fn(legal, clusters),
            "best_small_quality": float(best_small_quality),
            "audit_rollback": bool(small_audit_rollback),
            "high_net_lane": bool(high_net_lane),
            "no_release_low_net_lane": bool(no_release_low_net_lane),
            "nets_per_macro": float(nets_per_macro),
            "released_clusters": [int(cid) for cid in released_cids],
            "released_cluster_count": int(len(released_cids)),
            "swap_stats": swap_stats,
        }
        _trace_pass(
            "small_design_released_polish",
            small_before,
            float(cur_proxy),
            int(small_acc),
            **trace_extra,
        )
        _record_plateau(
            "small_design_released_polish",
            small_before,
            float(cur_proxy),
            int(small_acc),
            time.monotonic() - small_t0,
            candidates=int(small_stats["candidates"]),
            legal=int(small_stats["legal"]),
            scored=int(small_stats["scored"]),
            **trace_extra,
        )
    else:
        log_gnn_event(
            "hier_budget_schedule",
            benchmark=benchmark.name,
            pass_name="small_design_released_polish",
            run=False,
            num_hard=int(n),
            num_soft=int(n_soft),
            num_macros=int(n) + int(n_soft),
        )

    full = np.vstack([legal, s_pos]).astype(np.float32)
    full_proxy = float(_exact_proxy(torch.tensor(full, dtype=torch.float32), benchmark, plc))
    final_quality = hierarchy_quality_metric_fn(legal, clusters)
    vector_audit_passed, hq_vector, vector_violations = _vector_contract(legal, s_pos)
    audit_passed = final_quality <= audit_limit and vector_audit_passed
    audit_rollback = False
    rollback_quality = None
    rollback_proxy = None
    if not audit_passed and _is_hard_valid(audit_checkpoint_h):
        if audit_checkpoint_quality <= audit_limit and audit_checkpoint_vector_passed:
            rollback_quality = float(audit_checkpoint_quality)
            rollback_proxy = float(
                _exact_proxy(
                    torch.tensor(
                        np.vstack([audit_checkpoint_h, audit_checkpoint_s]).astype(np.float32),
                        dtype=torch.float32,
                    ),
                    benchmark,
                    plc,
                )
            )
            legal = audit_checkpoint_h.copy()
            s_pos = audit_checkpoint_s.copy()
            full = np.vstack([legal, s_pos]).astype(np.float32)
            full_proxy = float(rollback_proxy)
            final_quality = float(audit_checkpoint_quality)
            hq_vector = dict(audit_checkpoint_vector)
            vector_audit_passed = True
            vector_violations = {}
            audit_passed = True
            audit_rollback = True
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
        "hier_final_audit",
        benchmark=benchmark.name,
        seed_hierarchy_quality=float(seed_hierarchy_quality),
        final_hierarchy_quality=float(final_quality),
        max_degradation=float(audit_budget),
        audit_limit=float(audit_limit),
        passed=bool(audit_passed),
        rollback=bool(audit_rollback),
        rollback_quality=rollback_quality,
        rollback_proxy=rollback_proxy,
        audit_checkpoint_quality=float(audit_checkpoint_quality),
        audit_checkpoint_proxy=float(audit_checkpoint_score),
        hierarchy_vector_passed=bool(vector_audit_passed),
        hierarchy_vector_violations={key: float(value) for key, value in vector_violations.items()},
        hierarchy_vector_limits={
            key: float(value) for key, value in hierarchy_contract_limits.items()
        },
    )

    log_gnn_event(
        "hier_final",
        benchmark=benchmark.name,
        proxy=float(proxy),
        pre_relief_proxy=float(pre_relief),
        seed_hierarchy_quality=float(seed_hierarchy_quality),
        hierarchy_audit_limit=float(audit_limit),
        hierarchy_audit_passed=bool(audit_passed),
        hierarchy_audit_rollback=bool(audit_rollback),
        hierarchy_quality=float(hq_breakdown["quality"]),
        hierarchy_quality_radius=float(hq_breakdown["radius"]),
        hierarchy_quality_bbox=float(hq_breakdown["bbox"]),
        hierarchy_quality_crowd=float(hq_breakdown["crowd"]),
        hierarchy_vector_composite=float(hq_vector["composite"]),
        hierarchy_vector_compactness=float(hq_vector["cluster_compactness"]),
        hierarchy_vector_worst_spread=float(hq_vector["worst_cluster_spread"]),
        hierarchy_vector_impurity=float(hq_vector["neighbor_impurity"]),
        hierarchy_vector_edge_stretch=float(hq_vector["edge_stretch"]),
        hierarchy_vector_owned_soft=float(hq_vector["owned_soft_distance"]),
        hierarchy_vector_bridge_soft=float(hq_vector["bridge_soft_distance"]),
        hierarchy_vector_audit_passed=bool(vector_audit_passed),
        hierarchy_vector_limits={
            key: float(value) for key, value in hierarchy_contract_limits.items()
        },
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
        f"audit={'rollback' if audit_rollback else ('pass' if audit_passed else 'fail')}, "
        f"vector_audit={'pass' if vector_audit_passed else 'fail'}, "
        f"weight={group_weight}: proxy={proxy:.4f} "
        f"(pre-relief {pre_relief:.4f}; hierarchy-preserving NON-proxy mode)"
    )
    flush_plateau_events()
    return out
