"""Post-coldspot survivor stage and final reporting segment."""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
import torch

from placer.local_search.fields import (
    cold_connected_component_target_pool,
    weighted_congestion_field,
)
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
from placer.local_search.plateau_telemetry import flush_plateau_events, log_plateau_event
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
    log_fn: Callable[[str], None],
    record_plateau_fn: Callable[..., None],
    hard_valid_fn: Callable[[np.ndarray], bool],
    deadline_fn: Callable[[float, float | None], float | None],
) -> torch.Tensor:
    """Run survivor search and emit the final hierarchy floorplan output."""

    _log = log_fn
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

    def _vector_contract_with_violations(
        hard_xy: np.ndarray,
        soft_xy: np.ndarray,
    ) -> tuple[bool, tuple[str, ...]]:
        vector = _placement_hierarchy_vector(hard_xy, soft_xy)
        passed, violations = hierarchy_vector_contract(vector, hierarchy_contract_limits)
        return bool(passed), tuple(str(v) for v in violations)

    def _vector_contract(hard_xy: np.ndarray, soft_xy: np.ndarray) -> bool:
        return bool(_vector_contract_with_violations(hard_xy, soft_xy)[0])

    audit_checkpoint_h = np.array(audit_h, dtype=np.float64, copy=True)
    audit_checkpoint_s = np.array(audit_s, dtype=np.float64, copy=True)
    audit_checkpoint_score = float(audit_score)
    audit_checkpoint_quality = hierarchy_quality_metric_fn(audit_checkpoint_h, clusters)
    audit_checkpoint_vector_passed = _vector_contract(audit_checkpoint_h, audit_checkpoint_s)

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
            cand_vector_passed = _vector_contract(cand_h, cand_s)
            if _is_hard_valid(cand_h) and cand_quality <= audit_limit and cand_vector_passed:
                audit_checkpoint_h = np.array(cand_h, dtype=np.float64, copy=True)
                audit_checkpoint_s = np.array(cand_s, dtype=np.float64, copy=True)
                audit_checkpoint_score = float(cand_score)
                audit_checkpoint_quality = float(cand_quality)
                audit_checkpoint_vector_passed = True
                break

    def _update_audit_checkpoint(
        hard_xy: np.ndarray,
        soft_xy: np.ndarray,
        score: float,
    ) -> tuple[bool, float]:
        nonlocal audit_checkpoint_h, audit_checkpoint_s
        nonlocal audit_checkpoint_score, audit_checkpoint_quality
        if not _is_hard_valid(hard_xy):
            return False, float("inf")
        quality = hierarchy_quality_metric_fn(hard_xy, clusters)
        if quality > audit_limit:
            return False, float(quality)
        vector_passed = _vector_contract(hard_xy, soft_xy)
        if not vector_passed:
            return False, float(quality)
        if float(score) < audit_checkpoint_score - 1e-9 or audit_checkpoint_quality > audit_limit:
            audit_checkpoint_h = np.array(hard_xy, dtype=np.float64, copy=True)
            audit_checkpoint_s = np.array(soft_xy, dtype=np.float64, copy=True)
            audit_checkpoint_score = float(score)
            audit_checkpoint_quality = float(quality)
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
        small_last_report: dict[str, object] = {
            "restored": False,
            "reason": "checkpoint_unavailable",
            "violations": (),
            "rebuild_elapsed_s": 0.0,
            "quality_before": float("nan"),
            "quality_after": float("nan"),
            "proxy_before": float("nan"),
            "proxy_after": float("nan"),
        }

        def _remember_small_state() -> bool:
            nonlocal best_small_score, best_small_legal, best_small_s_pos
            nonlocal best_small_quality
            cur_quality_local = hierarchy_quality_metric_fn(legal, clusters)
            if cur_quality_local > audit_limit:
                return False
            vector_passed = _vector_contract(legal, s_pos)
            if not vector_passed:
                return False
            if float(cur_proxy) < best_small_score - 1e-9 or best_small_quality > audit_limit:
                best_small_score = float(cur_proxy)
                best_small_legal = np.array(legal, dtype=np.float64, copy=True)
                best_small_s_pos = np.array(s_pos, dtype=np.float64, copy=True)
                best_small_quality = float(cur_quality_local)
                return True
            return False

        def _restore_small_if_needed() -> dict[str, object]:
            nonlocal legal, s_pos, cur_proxy, small_scorer, small_audit_rollback
            start = time.perf_counter()
            cur_quality_local = hierarchy_quality_metric_fn(legal, clusters)
            vector_passed, violations = _vector_contract_with_violations(legal, s_pos)
            if cur_quality_local <= audit_limit and vector_passed:
                _remember_small_state()
                small_audit_rollback = False
                return {
                    "restored": False,
                    "reason": "ok",
                    "violations": (),
                    "rebuild_elapsed_s": 0.0,
                    "quality_before": float(cur_quality_local),
                    "quality_after": float(cur_quality_local),
                    "proxy_before": float(cur_proxy),
                    "proxy_after": float(cur_proxy),
                }
            if (
                not _is_hard_valid(best_small_legal)
                or best_small_quality > audit_limit
                or not _vector_contract(best_small_legal, best_small_s_pos)
            ):
                small_audit_rollback = bool(small_audit_rollback)
                return {
                    "restored": False,
                    "reason": "checkpoint_unavailable",
                    "violations": (),
                    "rebuild_elapsed_s": 0.0,
                    "quality_before": float(cur_quality_local),
                    "quality_after": float("nan"),
                    "proxy_before": float(cur_proxy),
                    "proxy_after": float(cur_proxy),
                }
            reason = "hierarchy_vector"
            if cur_quality_local > audit_limit:
                reason = "quality"
            proxy_before = float(cur_proxy)
            legal = best_small_legal.copy()
            s_pos = best_small_s_pos.copy()
            cur_proxy = float(best_small_score)
            small_scorer = IncrementalScorer(
                plc,
                benchmark,
                np.vstack([legal, s_pos]).astype(np.float64),
            )
            small_audit_rollback = True
            return {
                "restored": True,
                "reason": reason,
                "violations": list(violations),
                "rebuild_elapsed_s": float(time.perf_counter() - start),
                "quality_before": float(cur_quality_local),
                "quality_after": float(best_small_quality),
                "proxy_before": float(proxy_before),
                "proxy_after": float(best_small_score),
            }

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
            round_abort = False
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
            small_last_report = _restore_small_if_needed()
            if small_last_report.get("restored"):
                round_abort = True
                continue
            hard_reloc_gain = max(0.0, float(hard_reloc_before) - float(cur_proxy))
            if (
                int(n_soft) > 0
                and bool(np.any(soft_mov))
                and (small_deadline is None or time.monotonic() < small_deadline)
                and not round_abort
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
                    )
                    small_acc += int(got)
                    for key in small_stats:
                        small_stats[key] += int(
                            getattr(_soft_relocation_moves, "last_stats", {}).get(key, 0)
                        )
                    if small_deadline is not None and time.monotonic() >= small_deadline:
                        break
                small_last_report = _restore_small_if_needed()
                if small_last_report.get("restored"):
                    round_abort = True
                    continue
            if small_deadline is not None and time.monotonic() >= small_deadline:
                break
            if released_cids and hard_reloc_gain >= min_gain:
                if round_abort:
                    continue
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
                    small_last_report = _restore_small_if_needed()
                    if small_last_report.get("restored"):
                        round_abort = True
                        break
                    if float(hard_swap_before) - float(cur_proxy) <= min_gain:
                        break
                    if small_deadline is not None and time.monotonic() >= small_deadline:
                        break
                if round_abort:
                    continue
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
                small_last_report = _restore_small_if_needed()
                if small_last_report.get("restored"):
                    round_abort = True
                    break
                if float(swap_before) - float(cur_proxy) <= min_gain:
                    break
                if small_deadline is not None and time.monotonic() >= small_deadline:
                    break

            if small_deadline is not None and time.monotonic() >= small_deadline:
                break
            if round_abort:
                continue
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
            small_last_report = _restore_small_if_needed()
            if small_last_report.get("restored"):
                round_abort = True
                continue
            if float(round_before) - float(cur_proxy) <= min_gain:
                break
        cur_quality = hierarchy_quality_metric_fn(legal, clusters)
        cur_vector_passed = _vector_contract(legal, s_pos)
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
        small_last_proposed_after = float(cur_proxy)
        small_last_report = _restore_small_if_needed()
        small_audit_rollback = bool(small_last_report.get("restored", bool(small_audit_rollback)))
        if not _is_hard_valid(legal):
            legal, s_pos, cur_proxy = best_h.copy(), best_s.copy(), float(best_score)
            small_acc = 0
            small_last_report = {
                "restored": False,
                "reason": "checkpoint_unavailable",
                "violations": (),
                "rebuild_elapsed_s": 0.0,
                "quality_before": float("nan"),
                "quality_after": float("nan"),
                "proxy_before": float(cur_proxy),
                "proxy_after": float(cur_proxy),
            }
            small_audit_rollback = True
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
            "audit_rollback_reason": str(small_last_report.get("reason", "")),
            "audit_rollback_violations": list(small_last_report.get("violations", ())),
            "audit_rebuild_s": float(small_last_report.get("rebuild_elapsed_s", 0.0)),
            "audit_quality_before": float(small_last_report.get("quality_before", float("nan"))),
            "audit_quality_after": float(small_last_report.get("quality_after", float("nan"))),
            "audit_proxy_before": float(small_last_report.get("proxy_before", float("nan"))),
            "audit_proxy_after": float(small_last_report.get("proxy_after", float("nan"))),
        }
        _record_plateau(
            "small_design_released_polish",
            small_before,
            float(cur_proxy),
            int(small_acc),
            time.monotonic() - small_t0,
            candidates=int(small_stats["candidates"]),
            legal=int(small_stats["legal"]),
            scored=int(small_stats["scored"]),
            proposed_after=small_last_proposed_after,
            rollback_report=small_last_report,
            **trace_extra,
        )
    else:
        log_plateau_event(
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
    vector_audit_passed = _vector_contract(legal, s_pos)
    audit_passed = final_quality <= audit_limit and vector_audit_passed
    audit_rollback = False
    if not audit_passed and _is_hard_valid(audit_checkpoint_h):
        if audit_checkpoint_quality <= audit_limit and audit_checkpoint_vector_passed:
            full_proxy = float(
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
            final_quality = float(audit_checkpoint_quality)
            vector_audit_passed = True
            audit_passed = True
            audit_rollback = True
    state = PlacementState(
        legal.copy(),
        s_pos.copy(),
        full_proxy,
    )
    out = torch.tensor(state.full().astype(np.float32), dtype=torch.float32)
    proxy = float(state.proxy)
    margin_eps = float(const.HIER_LEGALITY_MARGIN_EPS)
    legality_margin = _hard_legality_margin(legal, margin_eps)

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
