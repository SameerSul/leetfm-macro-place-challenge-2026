"""Seed-portfolio helpers for hierarchy floorplan execution."""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from typing import Any, Callable

import numpy as np
import torch

from placer.local_search.hierarchy_quality import (
    HIERARCHY_VECTOR_METRICS,
    hierarchy_quality_vector,
    hierarchy_vector_contract,
    hierarchy_vector_limits,
    hierarchy_vector_margins,
)
from placer.local_search.plateau_telemetry import log_plateau_event


def _hard_placement_is_legal(
    hard_xy: np.ndarray,
    half_widths: np.ndarray,
    half_heights: np.ndarray,
    canvas_width: float,
    canvas_height: float,
) -> bool:
    """Return whether hard-macro centers are in bounds and non-overlapping."""
    hard = np.asarray(hard_xy, dtype=np.float64)
    hw = np.asarray(half_widths, dtype=np.float64)
    hh = np.asarray(half_heights, dtype=np.float64)
    if hard.shape != (hw.size, 2) or hw.shape != hh.shape:
        return False
    if hard.shape[0] == 0:
        return True
    tolerance = 1.0e-6
    if np.any(hard[:, 0] < hw - tolerance) or np.any(
        hard[:, 0] > float(canvas_width) - hw + tolerance
    ):
        return False
    if np.any(hard[:, 1] < hh - tolerance) or np.any(
        hard[:, 1] > float(canvas_height) - hh + tolerance
    ):
        return False
    dx = np.abs(hard[:, None, 0] - hard[None, :, 0])
    dy = np.abs(hard[:, None, 1] - hard[None, :, 1])
    separated = (dx + tolerance >= hw[:, None] + hw[None, :]) | (
        dy + tolerance >= hh[:, None] + hh[None, :]
    )
    np.fill_diagonal(separated, True)
    return bool(separated.all())


def select_recursive_prototype_leaves(
    clusters,
    cluster_confidence,
    movable_hard: np.ndarray,
    hard_xy: np.ndarray,
    hw: np.ndarray,
    hh: np.ndarray,
    *,
    max_leaves: int,
    max_members: int = 16,
    excluded: tuple[int, ...] = (),
) -> list[int]:
    """Select compact, high-confidence leaves that are safe to freeze."""
    excluded_set = {int(cid) for cid in excluded}
    ranked = []
    for cid, members in clusters.items():
        cid = int(cid)
        mem = np.asarray(members, dtype=np.int64)
        if (
            cid in excluded_set
            or mem.size < 2
            or mem.size > max(2, int(max_members))
            or not bool(np.all(movable_hard[mem]))
        ):
            continue
        confidence = float((cluster_confidence or {}).get(cid, 0.0))
        if confidence <= 0.0:
            continue
        left = float(np.min(hard_xy[mem, 0] - hw[mem]))
        right = float(np.max(hard_xy[mem, 0] + hw[mem]))
        bottom = float(np.min(hard_xy[mem, 1] - hh[mem]))
        top = float(np.max(hard_xy[mem, 1] + hh[mem]))
        footprint = max((right - left) * (top - bottom), 1.0e-12)
        utilization = float(np.sum(4.0 * hw[mem] * hh[mem]) / footprint)
        ranked.append((-round(confidence, 12), -round(utilization, 12), -int(mem.size), cid))
    ranked.sort()
    return [int(row[-1]) for row in ranked[: max(0, int(max_leaves))]]


def select_initial_recurrent_leaves(
    clusters,
    cluster_confidence,
    movable_hard: np.ndarray,
    hard_xy: np.ndarray,
    hw: np.ndarray,
    hh: np.ndarray,
    *,
    canvas_width: float,
    canvas_height: float,
    max_leaves: int,
    max_members: int = 16,
) -> list[int]:
    """Select strong initial-placement anchors with broad canvas coverage."""
    limit = max(0, int(max_leaves))
    if limit == 0:
        return []
    ranked = []
    for cid, members in clusters.items():
        cid = int(cid)
        mem = np.asarray(members, dtype=np.int64)
        if (
            mem.size < 2
            or mem.size > max(2, int(max_members))
            or not bool(np.all(movable_hard[mem]))
        ):
            continue
        confidence = float((cluster_confidence or {}).get(cid, 0.0))
        if confidence <= 0.0:
            continue
        left = float(np.min(hard_xy[mem, 0] - hw[mem]))
        right = float(np.max(hard_xy[mem, 0] + hw[mem]))
        bottom = float(np.min(hard_xy[mem, 1] - hh[mem]))
        top = float(np.max(hard_xy[mem, 1] + hh[mem]))
        footprint = max((right - left) * (top - bottom), 1.0e-12)
        utilization = float(np.sum(4.0 * hw[mem] * hh[mem]) / footprint)
        ranked.append(
            {
                "cid": cid,
                "confidence": round(confidence, 12),
                "utilization": round(utilization, 12),
                "members": int(mem.size),
                "center": np.mean(hard_xy[mem], axis=0),
            }
        )
    ranked.sort(
        key=lambda row: (
            -row["confidence"],
            -row["utilization"],
            -row["members"],
            row["cid"],
        )
    )
    pool = ranked[: max(limit, 4 * limit)]
    if not pool:
        return []
    selected = [pool.pop(0)]
    width = max(float(canvas_width), 1.0e-12)
    height = max(float(canvas_height), 1.0e-12)
    while pool and len(selected) < limit:

        def _coverage_key(row):
            center = row["center"]
            min_distance = min(
                ((float(center[0]) - float(anchor["center"][0])) / width) ** 2
                + ((float(center[1]) - float(anchor["center"][1])) / height) ** 2
                for anchor in selected
            )
            return (
                -round(min_distance, 12),
                -row["confidence"],
                -row["utilization"],
                -row["members"],
                row["cid"],
            )

        pool.sort(key=_coverage_key)
        selected.append(pool.pop(0))
    return [int(row["cid"]) for row in selected]


def should_run_initial_recurrent(
    dreamplace_score: float,
    initial_score: float,
    *,
    dreamplace_contract_passed: bool,
    minimum_proxy_advantage: float,
) -> bool:
    """Return whether a contract-repairing recurrent solve is worth its cost."""
    if dreamplace_contract_passed:
        return False
    return float(dreamplace_score) <= float(initial_score) * (1.0 - float(minimum_proxy_advantage))


def select_seed_candidate(
    rows: list[dict[str, object]],
    *,
    hierarchy_first: bool,
    absolute_slack: float,
    relative_slack: float,
    component_absolute_slack: Mapping[str, float] | None = None,
    component_relative_slack: float = 0.0,
    component_reference_name: str = "initial",
    component_reference_vector: Mapping[str, float] | None = None,
    headroom_aware: bool = False,
    proxy_band_absolute: float = 0.0,
    proxy_band_relative: float = 0.0,
) -> dict[str, object]:
    """Select a proxy-competitive seed with optional hierarchy headroom."""
    if not rows:
        raise ValueError("seed portfolio is empty")
    eligible = rows
    if component_absolute_slack is not None:
        reference = next(
            (row for row in rows if str(row["name"]) == str(component_reference_name)),
            min(rows, key=lambda row: (float(row["score"]), str(row["name"]))),
        )
        reference_vector = (
            component_reference_vector
            if component_reference_vector is not None
            else reference.get("hierarchy_vector")
        )
        if not isinstance(reference_vector, Mapping):
            raise ValueError("component hierarchy contract requires complete seed vectors")
        limits = hierarchy_vector_limits(
            reference_vector,
            component_absolute_slack,
            component_relative_slack,
        )
        eligible = []
        for row in rows:
            vector = row.get("hierarchy_vector")
            if not isinstance(vector, Mapping):
                raise ValueError("component hierarchy contract requires complete seed vectors")
            passed, violations = hierarchy_vector_contract(vector, limits)
            row["hierarchy_contract_eligible"] = bool(passed)
            row["hierarchy_contract_violations"] = violations
            row["hierarchy_contract_reference"] = str(reference["name"])
            row["hierarchy_contract_reference_vector"] = dict(reference_vector)
            row["hierarchy_contract_limits"] = limits
            if passed:
                eligible.append(row)
        if not eligible:
            reference_passed, reference_violations = hierarchy_vector_contract(
                reference_vector,
                limits,
            )
            if not reference_passed:
                raise ValueError(
                    "no seed candidate satisfies the component hierarchy contract; "
                    f"reference violations: {reference_violations}"
                )
            eligible = [reference]
    if not hierarchy_first and headroom_aware:
        best_proxy = min(float(row["score"]) for row in eligible)
        proxy_slack = max(
            float(proxy_band_absolute),
            abs(best_proxy) * float(proxy_band_relative),
        )
        proxy_band = [row for row in eligible if float(row["score"]) <= best_proxy + proxy_slack]

        def _headroom_key(row: Mapping[str, object]):
            vector = row.get("hierarchy_vector")
            limits = row.get("hierarchy_contract_limits")
            if not isinstance(vector, Mapping) or not isinstance(limits, Mapping):
                return (
                    0.0,
                    float(row["hierarchy_composite"]),
                    float(row["score"]),
                    str(row["name"]),
                )
            normalized = [
                (float(limits[key]) - float(vector.get(key, 0.0)))
                / max(abs(float(limits[key])), 1.0e-12)
                for key in HIERARCHY_VECTOR_METRICS
                if key in limits
            ]
            minimum = min(normalized) if normalized else 0.0
            return (
                -float(minimum),
                float(row["hierarchy_composite"]),
                float(row["score"]),
                str(row["name"]),
            )

        return min(proxy_band, key=_headroom_key)
    if not hierarchy_first:
        return min(eligible, key=lambda row: (float(row["score"]), str(row["name"])))
    best_quality = min(float(row["hierarchy_composite"]) for row in eligible)
    slack = max(float(absolute_slack), abs(best_quality) * float(relative_slack))
    hierarchy_band = [
        row for row in eligible if float(row["hierarchy_composite"]) <= best_quality + slack
    ]
    return min(hierarchy_band, key=lambda row: (float(row["score"]), str(row["name"])))


def repair_seed_to_contract(
    source_hard: np.ndarray,
    source_soft: np.ndarray,
    reference_hard: np.ndarray,
    reference_soft: np.ndarray,
    *,
    legalize_fn: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    vector_fn: Callable[[np.ndarray, np.ndarray], Mapping[str, float]],
    limits: Mapping[str, float],
    refine_rounds: int = 5,
) -> tuple[np.ndarray, np.ndarray, dict[str, float], float, int] | None:
    """Project a lower-proxy seed toward a passing reference without relaxing limits."""
    source_hard = np.asarray(source_hard, dtype=np.float64)
    source_soft = np.asarray(source_soft, dtype=np.float64)
    reference_hard = np.asarray(reference_hard, dtype=np.float64)
    reference_soft = np.asarray(reference_soft, dtype=np.float64)
    if source_hard.shape != reference_hard.shape or source_soft.shape != reference_soft.shape:
        raise ValueError("seed repair source/reference shapes must match")

    attempts = 0
    best: tuple[np.ndarray, np.ndarray, dict[str, float], float] | None = None
    failed_fraction = 1.0
    passing_fraction = 0.0
    for fraction in (0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125):
        hard = reference_hard + float(fraction) * (source_hard - reference_hard)
        soft = reference_soft + float(fraction) * (source_soft - reference_soft)
        legal_hard, legal_soft = legalize_fn(hard, soft)
        vector = dict(vector_fn(legal_hard, legal_soft))
        attempts += 1
        passed, _ = hierarchy_vector_contract(vector, limits)
        if passed:
            passing_fraction = float(fraction)
            best = (legal_hard, legal_soft, vector, float(fraction))
            break
        failed_fraction = float(fraction)

    if best is None:
        reference_vector = dict(vector_fn(reference_hard, reference_soft))
        reference_passed, _ = hierarchy_vector_contract(reference_vector, limits)
        if not reference_passed:
            raise ValueError("seed repair reference must satisfy the hierarchy contract")
        best = (
            reference_hard.copy(),
            reference_soft.copy(),
            reference_vector,
            0.0,
        )

    for _ in range(max(0, int(refine_rounds))):
        fraction = 0.5 * (passing_fraction + failed_fraction)
        hard = reference_hard + fraction * (source_hard - reference_hard)
        soft = reference_soft + fraction * (source_soft - reference_soft)
        legal_hard, legal_soft = legalize_fn(hard, soft)
        vector = dict(vector_fn(legal_hard, legal_soft))
        attempts += 1
        passed, _ = hierarchy_vector_contract(vector, limits)
        if passed:
            passing_fraction = fraction
            best = (legal_hard, legal_soft, vector, fraction)
        else:
            failed_fraction = fraction

    hard, soft, vector, fraction = best
    if fraction <= 0.0:
        return None
    return hard, soft, vector, float(fraction), attempts


def run_seed_portfolio(
    *,
    benchmark,
    plc,
    benchmark_dir,
    n: int,
    n_soft: int,
    clusters,
    order,
    sizes: np.ndarray,
    hw: np.ndarray,
    hh: np.ndarray,
    soft_hw: np.ndarray,
    soft_hh: np.ndarray,
    movable: np.ndarray,
    groups: dict | list | tuple | None,
    csofts,
    bridge_softs,
    hierarchy_edges,
    cluster_confidence=None,
    cluster_source: str = "hierarchy",
    cw: float,
    ch: float,
    const: Any,
    logger: Callable[[str], None],
    run_dreamplace: Callable[..., tuple[np.ndarray, np.ndarray]],
    will_legalize: Callable[..., np.ndarray],
    exact_proxy_fn: Callable[[torch.Tensor, Any, Any], float],
    soft_relocation_fn: Callable[..., tuple[np.ndarray, float]],
    incremental_scorer_cls: type,
    group_weight: int,
    random_seed: int = 1000,
    scratch_root: str = "/tmp/dreamplace_v1_hier",
    event_sink=None,
) -> tuple[np.ndarray, np.ndarray, float, list[dict[str, object]]]:
    """Create and score the seed portfolio used by hierarchy floorplanning.

    The seed portfolio starts from the DREAMPlace seed and several small
    perturbations that preserve hierarchy intent while improving overlap
    survivability before downstream region cleanup.
    """
    benchmark_name = str(getattr(benchmark, "_hierarchy_trace_name", benchmark.name))

    def _emit_seed_status(name: str, status: str) -> None:
        if event_sink is None:
            return
        from visualizer.events import emit_event

        emit_event(
            event_sink,
            "seed_status",
            seed_name=str(name),
            status=str(status),
            metrics_stale=True,
        )

    def _log_stage_timing(stage: str, elapsed_s: float, **extra) -> None:
        payload = {
            "benchmark": benchmark_name,
            "stage": str(stage),
            "elapsed_s": float(elapsed_s),
        }
        payload.update(extra)
        log_plateau_event("hier_stage_timing", **payload)

    def _first_legalize(
        hard_xy: np.ndarray,
        seed_deadline: float,
        name: str,
    ) -> np.ndarray:
        return will_legalize(
            hard_xy,
            movable[:n],
            sizes[:n],
            hw,
            hh,
            cw,
            ch,
            n,
            deadline=seed_deadline,
            order=order,
        )

    def _prepare_dreamplace_candidate(
        *,
        group_weight: int,
        random_seed: int,
        scratch_root: str,
    ):
        dreamplace_t0 = time.perf_counter()
        raw_hard, raw_soft = run_dreamplace(
            str(benchmark_dir),
            plc=plc,
            scratch_root=scratch_root,
            iterations=300,
            num_threads=2,
            random_seed=random_seed,
            soft_macros_movable=True,
            cluster_groups=(groups or None),
            group_weight=group_weight,
            return_full=True,
            seed_name="dreamplace",
        )
        _log_stage_timing(
            "seed_dreamplace_cache_lookup",
            float(time.perf_counter() - dreamplace_t0),
            candidate="dreamplace",
        )
        seed_creation_t0 = time.perf_counter()
        legal_hard = _first_legalize(
            raw_hard.copy(),
            time.monotonic() + 120,
            "dreamplace",
        )
        legal_hard = will_legalize(
            legal_hard,
            movable[:n],
            sizes[:n],
            hw,
            hh,
            cw,
            ch,
            n,
            deadline=time.monotonic() + 120,
            order=None,
        )
        _log_stage_timing(
            "seed_creation",
            float(time.perf_counter() - seed_creation_t0),
            candidate="dreamplace",
        )

        return legal_hard, raw_soft

    def _prepare_recursive_candidate(
        name: str,
        fixed_leaf_ids: list[int],
        fixed_hard: np.ndarray,
    ):
        hard_indices = sorted(
            {
                int(index)
                for cid in fixed_leaf_ids
                for index in np.asarray(clusters[int(cid)], dtype=np.int64)
            }
        )
        fixed_positions = {}
        for hard_index in hard_indices:
            module_index = plc.hard_macro_indices[hard_index]
            module_name = plc.modules_w_pins[module_index].get_name()
            fixed_positions[module_name] = (
                float(fixed_hard[hard_index, 0]),
                float(fixed_hard[hard_index, 1]),
            )
        if not fixed_positions:
            raise ValueError("recursive prototype requires at least one fixed hard macro")

        dreamplace_t0 = time.perf_counter()
        raw_hard, raw_soft = run_dreamplace(
            str(benchmark_dir),
            plc=plc,
            scratch_root=scratch_root,
            iterations=300,
            num_threads=2,
            random_seed=random_seed,
            soft_macros_movable=True,
            cluster_groups=(groups or None),
            group_weight=group_weight,
            return_full=True,
            seed_name=name,
            temporary_fixed_positions=fixed_positions,
        )
        _log_stage_timing(
            "seed_recursive_dreamplace",
            float(time.perf_counter() - dreamplace_t0),
            candidate=name,
            fixed_leaves=[int(cid) for cid in fixed_leaf_ids],
            fixed_hard_count=len(hard_indices),
        )
        recursive_movable = movable[:n].copy()
        recursive_movable[np.asarray(hard_indices, dtype=np.int64)] = False
        legal_hard = will_legalize(
            raw_hard.copy(),
            recursive_movable,
            sizes[:n],
            hw,
            hh,
            cw,
            ch,
            n,
            deadline=time.monotonic() + 120,
            order=order,
        )
        legal_hard = will_legalize(
            legal_hard,
            recursive_movable,
            sizes[:n],
            hw,
            hh,
            cw,
            ch,
            n,
            deadline=time.monotonic() + 120,
            order=None,
        )
        return legal_hard, raw_soft, hard_indices

    def _clip_seed(hard_xy: np.ndarray, soft_xy: np.ndarray):
        hard_xy = hard_xy.copy()
        soft_xy = soft_xy.copy()
        hard_mov = movable[:n]
        hard_xy[hard_mov, 0] = np.clip(hard_xy[hard_mov, 0], hw[hard_mov], cw - hw[hard_mov])
        hard_xy[hard_mov, 1] = np.clip(hard_xy[hard_mov, 1], hh[hard_mov], ch - hh[hard_mov])
        if n_soft:
            soft_mov_local = movable[n : n + n_soft]
            soft_xy[soft_mov_local, 0] = np.clip(
                soft_xy[soft_mov_local, 0],
                soft_hw[soft_mov_local],
                cw - soft_hw[soft_mov_local],
            )
            soft_xy[soft_mov_local, 1] = np.clip(
                soft_xy[soft_mov_local, 1],
                soft_hh[soft_mov_local],
                ch - soft_hh[soft_mov_local],
            )
        return hard_xy, soft_xy

    def _legalize_seed(
        name: str,
        hard_xy,
        soft_xy,
        *,
        budget_s: float = 60.0,
        record_timing: bool = True,
    ):
        legalize_t0 = time.perf_counter()
        hard_xy, soft_xy = _clip_seed(hard_xy, soft_xy)
        seed_deadline = time.monotonic() + float(budget_s)
        legal_hard = _first_legalize(
            hard_xy,
            seed_deadline,
            name,
        )
        legal_hard = will_legalize(
            legal_hard,
            movable[:n],
            sizes[:n],
            hw,
            hh,
            cw,
            ch,
            n,
            deadline=seed_deadline,
            order=None,
        )
        if record_timing:
            _log_stage_timing(
                "seed_creation",
                float(time.perf_counter() - legalize_t0),
                candidate=str(name),
                budget_s=float(budget_s),
            )
        return legal_hard, soft_xy

    def _score_seed(
        name: str,
        hard_xy,
        soft_xy,
        *,
        do_soft_cleanup: bool = False,
        cleanup_budget_s: float = 30.0,
    ):
        prescore_t0 = time.perf_counter()
        hard_xy = np.asarray(hard_xy, dtype=np.float64).copy()
        soft_xy = np.asarray(soft_xy, dtype=np.float64).copy()
        soft_mov_local = movable[n : n + n_soft]
        full = np.vstack([hard_xy, soft_xy]).astype(np.float64)
        score = float(exact_proxy_fn(torch.tensor(full, dtype=torch.float32), benchmark, plc))
        if do_soft_cleanup and n_soft:
            cand_scorer = incremental_scorer_cls(plc, benchmark, full.copy())
            for use_density in (False, True):
                soft_xy, _, score = soft_relocation_fn(
                    soft_xy,
                    soft_hw,
                    soft_hh,
                    cw,
                    ch,
                    n,
                    plc,
                    benchmark,
                    cand_scorer,
                    score,
                    deadline=time.monotonic() + float(cleanup_budget_s),
                    top_hot=1024,
                    n_targets=6,
                    soft_movable=soft_mov_local,
                    use_density=use_density,
                )
        _log_stage_timing(
            "seed_prescore",
            float(time.perf_counter() - prescore_t0),
            candidate=str(name),
            do_soft_cleanup=bool(do_soft_cleanup),
            score=float(score),
        )
        return {
            "name": name,
            "hard": hard_xy,
            "soft": soft_xy,
            "score": float(score),
        }

    def _hierarchy_coverage(row_vector: Mapping[str, float]) -> dict[str, float]:
        return {
            "clustered_hard_count": float(row_vector.get("clustered_hard_count", 0.0)),
            "clustered_hard_fraction": float(row_vector.get("clustered_hard_fraction", 0.0)),
            "unclustered_hard_count": float(row_vector.get("unclustered_hard_count", 0.0)),
            "owned_soft_count": float(row_vector.get("owned_soft_count", 0.0)),
            "owned_soft_coverage": float(row_vector.get("owned_soft_coverage", 0.0)),
            "bridge_soft_count": float(row_vector.get("bridge_soft_count", 0.0)),
            "bridge_soft_coverage": float(row_vector.get("bridge_soft_coverage", 0.0)),
            "soft_coverage": float(row_vector.get("soft_coverage", 0.0)),
            "soft_total": float(row_vector.get("soft_total", 0.0)),
        }

    def _has_explicit_path_tags() -> bool:
        try:
            hard_b = list(plc.hard_macro_indices[:n])
            tagged = sum(1 for idx in hard_b if "/" in str(plc.modules_w_pins[int(idx)].get_name()))
        except Exception:
            return False
        min_group = max(2, int(const.HIER_TAG_PREFIX_MIN_GROUP))
        return tagged >= max(min_group, int(0.5 * n))

    def _route_channel_seed(base_hard, base_soft):
        hard = base_hard.copy()
        soft = base_soft.copy()
        hard_mov = movable[:n].astype(bool)
        soft_mov = movable[n : n + n_soft].astype(bool)
        min_cluster = max(2, int(const.HIER_SEED_ROUTE_CHANNEL_MIN_CLUSTER))
        lane_frac = max(0.0, float(const.HIER_SEED_ROUTE_CHANNEL_LANE_FRAC))
        push_frac = max(0.0, float(const.HIER_SEED_ROUTE_CHANNEL_PUSH_FRAC))
        max_shift_frac = max(0.0, float(const.HIER_SEED_ROUTE_CHANNEL_MAX_SHIFT_FRAC))
        if lane_frac <= 0.0 or push_frac <= 0.0 or max_shift_frac <= 0.0:
            return hard, soft

        def _channel_delta(xy, local_hw, local_hh, center, span_x, span_y, index_bias):
            dx = xy[:, 0] - center[0]
            dy = xy[:, 1] - center[1]
            sx = np.where(dx >= 0.0, 1.0, -1.0)
            sy = np.where(dy >= 0.0, 1.0, -1.0)
            sx = np.where(np.abs(dx) > 1.0e-9, sx, np.where(index_bias % 2 == 0, 1.0, -1.0))
            sy = np.where(np.abs(dy) > 1.0e-9, sy, np.where(index_bias % 3 == 0, 1.0, -1.0))
            lane_x = max(float(np.median(local_hw)) * 0.75, float(span_x) * lane_frac)
            lane_y = max(float(np.median(local_hh)) * 0.75, float(span_y) * lane_frac)
            max_x = float(span_x) * max_shift_frac
            max_y = float(span_y) * max_shift_frac
            push_x = np.maximum(0.0, lane_x - np.abs(dx)) * push_frac
            push_y = np.maximum(0.0, lane_y - np.abs(dy)) * push_frac
            out = np.zeros_like(xy)
            out[:, 0] = sx * np.minimum(push_x, max_x)
            out[:, 1] = sy * np.minimum(push_y, max_y)
            return out

        for cid, mem in clusters.items():
            mem = np.asarray(mem, dtype=np.int64)
            if mem.size < min_cluster:
                continue
            active = mem[hard_mov[mem]]
            if active.size == 0:
                continue
            left = float(np.min(hard[mem, 0] - hw[mem]))
            right = float(np.max(hard[mem, 0] + hw[mem]))
            bottom = float(np.min(hard[mem, 1] - hh[mem]))
            top = float(np.max(hard[mem, 1] + hh[mem]))
            span_x = max(1.0, right - left)
            span_y = max(1.0, top - bottom)
            center = np.array([(left + right) * 0.5, (bottom + top) * 0.5], dtype=np.float64)
            hard[active] += _channel_delta(
                hard[active],
                hw[active],
                hh[active],
                center,
                span_x,
                span_y,
                active,
            )
            if n_soft:
                owned = np.asarray(csofts.get(int(cid), []), dtype=np.int64) - n
                owned = owned[(owned >= 0) & (owned < n_soft)]
                owned = owned[soft_mov[owned]]
                if owned.size:
                    soft[owned] += _channel_delta(
                        soft[owned],
                        soft_hw[owned],
                        soft_hh[owned],
                        center,
                        span_x,
                        span_y,
                        owned + mem.size,
                    )
        return _clip_seed(hard, soft)

    def _select_seed_portfolio(dp_hard, dp_soft):
        initial = benchmark.macro_positions.detach().cpu().numpy().astype(np.float64)
        init_hard = initial[:n].copy()
        init_soft = initial[n : n + n_soft].copy()
        _emit_seed_status("initial", "building")
        try:
            initial_legal_hard, initial_legal_soft = _legalize_seed(
                "initial",
                init_hard,
                init_soft,
                budget_s=45.0,
            )
        except Exception as exc:
            raise RuntimeError(
                "legalized initial.plc is required as the hierarchy-contract reference"
            ) from exc
        immutable_contract_keys = (
            "cluster_compactness",
            "worst_cluster_spread",
            "neighbor_impurity",
            "edge_stretch",
        )
        contract_absolute_slack = const.HIER_VECTOR_CONTRACT_ABS_SLACK
        contract_relative_slack = float(const.HIER_VECTOR_CONTRACT_REL_SLACK)
        legalized_reference_vector = hierarchy_quality_vector(
            initial_legal_hard,
            initial_legal_soft,
            clusters,
            csofts,
            bridge_softs,
            hierarchy_edges,
            cw,
            ch,
        )
        seed_reference_vector = legalized_reference_vector
        seed_reference_kind = "legalized_initial"
        seed_reference_name = "initial"
        if str(cluster_source) == "hierarchy_single_component_soft_affinity":
            if _hard_placement_is_legal(init_hard, hw, hh, cw, ch):
                raw_reference_vector = hierarchy_quality_vector(
                    init_hard,
                    init_soft,
                    clusters,
                    csofts,
                    bridge_softs,
                    hierarchy_edges,
                    cw,
                    ch,
                )
                raw_reference_limits = hierarchy_vector_limits(
                    raw_reference_vector,
                    contract_absolute_slack,
                    contract_relative_slack,
                )
                raw_reference_feasible, _ = hierarchy_vector_contract(
                    legalized_reference_vector,
                    raw_reference_limits,
                )
                if raw_reference_feasible:
                    seed_reference_vector = raw_reference_vector
                    seed_reference_kind = "raw_initial"
            else:
                seed_reference_vector = hierarchy_quality_vector(
                    dp_hard,
                    dp_soft,
                    clusters,
                    csofts,
                    bridge_softs,
                    hierarchy_edges,
                    cw,
                    ch,
                )
                seed_reference_kind = "dreamplace"
                seed_reference_name = "dreamplace"
        reference_contract_limits = hierarchy_vector_limits(
            seed_reference_vector,
            contract_absolute_slack,
            contract_relative_slack,
        )
        immutable_contract_limits = {
            key: float(reference_contract_limits[key])
            for key in immutable_contract_keys
            if key in reference_contract_limits
        }
        mandatory = {"dreamplace"}

        def _immutable_contract_pass(hard_xy, soft_xy):
            vector = hierarchy_quality_vector(
                np.asarray(hard_xy, dtype=np.float64),
                np.asarray(soft_xy, dtype=np.float64),
                clusters,
                csofts,
                bridge_softs,
                hierarchy_edges,
                cw,
                ch,
            )
            if not immutable_contract_limits:
                return True, vector
            for key, limit in immutable_contract_limits.items():
                if float(vector.get(key, 0.0)) > float(limit) + 1.0e-12:
                    return False, vector
            return True, vector

        rows: list[dict[str, object]] = []
        _emit_seed_status("dreamplace", "scoring")
        try:
            dp_passed, _ = _immutable_contract_pass(dp_hard, dp_soft)
            if not dp_passed:
                logger(
                    "  [hier] seed dreamplace failed immutable-contract prefilter; keeping for stability"
                )
            rows.append(_score_seed("dreamplace", dp_hard, dp_soft, do_soft_cleanup=True))
        except Exception as exc:
            logger("  [hier] seed dreamplace scoring failed: " f"{type(exc).__name__}: {exc}")
        _emit_seed_status("initial", "scoring")
        initial_row = _score_seed("initial", initial_legal_hard, initial_legal_soft)
        rows.append(initial_row)
        dreamplace_row = next(
            (row for row in rows if str(row["name"]) == "dreamplace"),
            None,
        )
        dreamplace_contract_passed = True
        if dreamplace_row is not None:
            dreamplace_vector = hierarchy_quality_vector(
                dreamplace_row["hard"],
                dreamplace_row["soft"],
                clusters,
                csofts,
                bridge_softs,
                hierarchy_edges,
                cw,
                ch,
            )
            dreamplace_contract_passed, _ = hierarchy_vector_contract(
                dreamplace_vector,
                reference_contract_limits,
            )
        minimum_advantage = float(
            getattr(const, "HIER_INITIAL_RECURRENT_MIN_PROXY_ADVANTAGE", 0.15)
        )
        recurrent_worthwhile = bool(
            dreamplace_row is not None
            and should_run_initial_recurrent(
                float(dreamplace_row["score"]),
                float(initial_row["score"]),
                dreamplace_contract_passed=dreamplace_contract_passed,
                minimum_proxy_advantage=minimum_advantage,
            )
        )
        initial_recurrent_ids = []
        if recurrent_worthwhile:
            initial_recurrent_ids = select_initial_recurrent_leaves(
                clusters,
                cluster_confidence,
                movable[:n],
                initial_legal_hard,
                hw,
                hh,
                canvas_width=cw,
                canvas_height=ch,
                max_leaves=int(getattr(const, "HIER_INITIAL_RECURRENT_MAX_LEAVES", 4)),
                max_members=int(getattr(const, "HIER_RE2MAP_MAX_LEAF_HARD", 16)),
            )
        if recurrent_worthwhile and len(initial_recurrent_ids) >= int(
            getattr(const, "HIER_INITIAL_RECURRENT_MIN_LEAVES", 2)
        ):
            _emit_seed_status("initial_recurrent", "building")
            try:
                initial_recurrent_hard, initial_recurrent_soft, fixed_hard_indices = (
                    _prepare_recursive_candidate(
                        "initial_recurrent",
                        initial_recurrent_ids,
                        initial_legal_hard,
                    )
                )
                passed, _ = _immutable_contract_pass(
                    initial_recurrent_hard,
                    initial_recurrent_soft,
                )
                if passed:
                    _emit_seed_status("initial_recurrent", "scoring")
                    row = _score_seed(
                        "initial_recurrent",
                        initial_recurrent_hard,
                        initial_recurrent_soft,
                        do_soft_cleanup=True,
                    )
                    row["recursive_fixed_leaves"] = tuple(int(cid) for cid in initial_recurrent_ids)
                    row["recursive_fixed_hard"] = tuple(int(i) for i in fixed_hard_indices)
                    row["initial_recurrent"] = True
                    rows.append(row)
                else:
                    logger("  [hier] seed initial_recurrent failed immutable contract prefilter")
            except Exception as exc:
                logger(
                    "  [hier] seed initial_recurrent failed recursive prototype: "
                    f"{type(exc).__name__}: {exc}"
                )
        recursive_seed_count = max(
            0,
            min(2, int(getattr(const, "HIER_RE2MAP_RECURSIVE_SEEDS", 2))),
        )
        fixed_leaf_ids: list[int] = []
        recursive_hard = dp_hard
        for round_index in range(recursive_seed_count):
            next_ids = select_recursive_prototype_leaves(
                clusters,
                cluster_confidence,
                movable[:n],
                recursive_hard,
                hw,
                hh,
                max_leaves=1,
                max_members=int(getattr(const, "HIER_RE2MAP_MAX_LEAF_HARD", 16)),
                excluded=tuple(fixed_leaf_ids),
            )
            if not next_ids:
                break
            fixed_leaf_ids.extend(next_ids)
            name = f"re2map_recursive_{round_index + 1}"
            _emit_seed_status(name, "building")
            try:
                recursive_hard, recursive_soft, fixed_hard_indices = _prepare_recursive_candidate(
                    name, fixed_leaf_ids, recursive_hard
                )
            except Exception as exc:
                logger(
                    f"  [hier] seed {name} failed recursive prototype: "
                    f"{type(exc).__name__}: {exc}"
                )
                break
            passed, _ = _immutable_contract_pass(recursive_hard, recursive_soft)
            if not passed:
                logger(f"  [hier] seed {name} failed immutable contract prefilter")
                continue
            try:
                _emit_seed_status(name, "scoring")
                row = _score_seed(name, recursive_hard, recursive_soft, do_soft_cleanup=True)
                row["recursive_fixed_leaves"] = tuple(int(cid) for cid in fixed_leaf_ids)
                row["recursive_fixed_hard"] = tuple(int(i) for i in fixed_hard_indices)
                rows.append(row)
            except Exception as exc:
                logger(f"  [hier] seed {name} failed scoring: {type(exc).__name__}: {exc}")
        raw_candidates = []
        if _has_explicit_path_tags():
            raw_candidates.append(("route_channel", *_route_channel_seed(dp_hard, dp_soft)))
        for name, cand_h, cand_s in raw_candidates:
            _emit_seed_status(name, "building")
            try:
                legal_hard, legal_soft = _legalize_seed(name, cand_h, cand_s, budget_s=45.0)
            except Exception as exc:
                logger(f"  [hier] seed {name} failed prescore: {type(exc).__name__}: {exc}")
                continue
            passed, _ = _immutable_contract_pass(legal_hard, legal_soft)
            if not passed and name not in mandatory:
                logger(f"  [hier] seed {name} failed immutable contract prefilter")
                continue
            try:
                _emit_seed_status(name, "scoring")
                rows.append(_score_seed(name, legal_hard, legal_soft))
            except Exception as exc:
                logger(f"  [hier] seed {name} failed scoring: {type(exc).__name__}: {exc}")
        for row in rows:
            vector = hierarchy_quality_vector(
                np.asarray(row["hard"], dtype=np.float64),
                np.asarray(row["soft"], dtype=np.float64),
                clusters,
                csofts,
                bridge_softs,
                hierarchy_edges,
                cw,
                ch,
            )
            row["hierarchy_vector"] = vector
            row["hierarchy_composite"] = float(vector["composite"])
            row["hierarchy_coverage"] = _hierarchy_coverage(vector)
            row["hierarchy_provenance"] = {
                "source": str(cluster_source),
                "immutable_contract_limits": dict(
                    (str(k), float(v)) for k, v in immutable_contract_limits.items()
                ),
                "recursive_fixed_leaves": list(row.get("recursive_fixed_leaves", ())),
                "recursive_fixed_hard": list(row.get("recursive_fixed_hard", ())),
            }
        if seed_reference_kind == "dreamplace":
            seed_reference_vector = dict(
                next(row for row in rows if str(row["name"]) == "dreamplace")["hierarchy_vector"]
            )
        hierarchy_first = os.environ.get(
            "HIER_SEED_HIERARCHY_SELECT",
            "1" if bool(const.HIER_SEED_HIERARCHY_SELECT) else "0",
        ).strip().lower() in {"1", "true", "yes", "on"}
        selected = select_seed_candidate(
            rows,
            hierarchy_first=hierarchy_first,
            absolute_slack=float(const.HIER_SEED_HIERARCHY_ABS_SLACK),
            relative_slack=float(const.HIER_SEED_HIERARCHY_REL_SLACK),
            component_absolute_slack=const.HIER_VECTOR_CONTRACT_ABS_SLACK,
            component_relative_slack=float(const.HIER_VECTOR_CONTRACT_REL_SLACK),
            component_reference_name=seed_reference_name,
            component_reference_vector=seed_reference_vector,
            headroom_aware=bool(const.HIER_SEED_HEADROOM_SELECT),
            proxy_band_absolute=float(const.HIER_SEED_PROXY_BAND_ABS),
            proxy_band_relative=float(const.HIER_SEED_PROXY_BAND_REL),
        )
        repair_reference_hard = initial_legal_hard
        repair_reference_soft = initial_legal_soft
        if seed_reference_kind == "raw_initial":
            repair_reference_hard = init_hard
            repair_reference_soft = init_soft
        elif seed_reference_kind == "dreamplace":
            repair_reference_hard = dp_hard
            repair_reference_soft = dp_soft
        repair_limits = hierarchy_vector_limits(
            seed_reference_vector,
            const.HIER_VECTOR_CONTRACT_ABS_SLACK,
            float(const.HIER_VECTOR_CONTRACT_REL_SLACK),
        )
        lower_failed = [
            row
            for row in rows
            if str(row["name"]) in mandatory
            and not bool(row.get("hierarchy_contract_eligible", True))
            and len(dict(row.get("hierarchy_contract_violations", {}))) == 1
            and float(row["score"]) < float(selected["score"]) - 1.0e-12
        ]
        for source in sorted(lower_failed, key=lambda row: (float(row["score"]), str(row["name"]))):
            source_name = str(source["name"])
            repair_t0 = time.perf_counter()

            def _repair_legalize(hard_xy, soft_xy):
                return _legalize_seed(
                    f"repair_{source_name}",
                    hard_xy,
                    soft_xy,
                    budget_s=45.0,
                    record_timing=False,
                )

            def _repair_vector(hard_xy, soft_xy):
                return hierarchy_quality_vector(
                    hard_xy,
                    soft_xy,
                    clusters,
                    csofts,
                    bridge_softs,
                    hierarchy_edges,
                    cw,
                    ch,
                )

            repaired = repair_seed_to_contract(
                np.asarray(source["hard"], dtype=np.float64),
                np.asarray(source["soft"], dtype=np.float64),
                repair_reference_hard,
                repair_reference_soft,
                legalize_fn=_repair_legalize,
                vector_fn=_repair_vector,
                limits=repair_limits,
            )
            if repaired is None:
                continue
            repaired_hard, repaired_soft, repaired_vector, fraction, attempts = repaired
            if fraction < float(const.HIER_SEED_CONTRACT_REPAIR_MIN_FRACTION):
                _log_stage_timing(
                    "seed_contract_repair",
                    float(time.perf_counter() - repair_t0),
                    candidate=f"repair_{source_name}",
                    source=source_name,
                    fraction=float(fraction),
                    attempts=int(attempts),
                    source_score=float(source["score"]),
                    retained=False,
                    reason="insufficient_source_fraction",
                )
                continue
            repaired_row = _score_seed(
                f"repair_{source_name}",
                repaired_hard,
                repaired_soft,
            )
            repaired_row["hierarchy_vector"] = repaired_vector
            repaired_row["hierarchy_composite"] = float(repaired_vector["composite"])
            repaired_row["hierarchy_coverage"] = _hierarchy_coverage(repaired_vector)
            repaired_row["hierarchy_provenance"] = {
                "source": str(cluster_source),
                "immutable_contract_limits": dict(
                    (str(k), float(v)) for k, v in immutable_contract_limits.items()
                ),
                "repair_source": source_name,
                "repair_fraction": float(fraction),
            }
            repaired_row["repair_source"] = source_name
            repaired_row["repair_fraction"] = float(fraction)
            repaired_row["repair_attempts"] = int(attempts)
            rows.append(repaired_row)
            _log_stage_timing(
                "seed_contract_repair",
                float(time.perf_counter() - repair_t0),
                candidate=str(repaired_row["name"]),
                source=source_name,
                fraction=float(fraction),
                attempts=int(attempts),
                source_score=float(source["score"]),
                repaired_score=float(repaired_row["score"]),
                retained=True,
            )
        if lower_failed:
            selected = select_seed_candidate(
                rows,
                hierarchy_first=hierarchy_first,
                absolute_slack=float(const.HIER_SEED_HIERARCHY_ABS_SLACK),
                relative_slack=float(const.HIER_SEED_HIERARCHY_REL_SLACK),
                component_absolute_slack=const.HIER_VECTOR_CONTRACT_ABS_SLACK,
                component_relative_slack=float(const.HIER_VECTOR_CONTRACT_REL_SLACK),
                component_reference_name=seed_reference_name,
                component_reference_vector=seed_reference_vector,
                headroom_aware=bool(const.HIER_SEED_HEADROOM_SELECT),
                proxy_band_absolute=float(const.HIER_SEED_PROXY_BAND_ABS),
                proxy_band_relative=float(const.HIER_SEED_PROXY_BAND_REL),
            )
        rows.sort(
            key=lambda row: (
                row is not selected,
                float(row["score"]),
                str(row["name"]),
            )
        )
        for row in rows:
            row["selected"] = row is selected
            row["hierarchy_contract_reference_kind"] = seed_reference_kind
        reference_name = str(selected.get("hierarchy_contract_reference", "initial"))
        reference_row = next(row for row in rows if str(row["name"]) == reference_name)
        reference_vector = dict(
            selected.get(
                "hierarchy_contract_reference_vector",
                reference_row["hierarchy_vector"],
            )
        )
        for row in rows:
            vector = dict(row["hierarchy_vector"])
            limits = dict(row.get("hierarchy_contract_limits", {}))
            margins = hierarchy_vector_margins(vector, limits) if limits else {}
            log_plateau_event(
                "hierarchy_contract_audit",
                benchmark=benchmark_name,
                stage="seed_candidate",
                candidate=str(row["name"]),
                reference=reference_name,
                reference_kind=str(row.get("hierarchy_contract_reference_kind", "unknown")),
                selected=bool(row["selected"]),
                passed=bool(row.get("hierarchy_contract_eligible", True)),
                score=float(row["score"]),
                hierarchy_first=bool(hierarchy_first),
                hierarchy_source=str(cluster_source),
                vector=vector,
                reference_vector=reference_vector,
                limits=limits,
                margins=margins,
                violations=dict(row.get("hierarchy_contract_violations", {})),
                coverage=dict(row.get("hierarchy_coverage", {})),
                provenance=dict(row.get("hierarchy_provenance", {})),
            )
        summary = ", ".join(
            f"{r['name']}={float(r['score']):.4f}/hq={float(r['hierarchy_composite']):.5f}"
            f"/contract={int(bool(r.get('hierarchy_contract_eligible', True)))}"
            f"/cov_h={float((r.get('hierarchy_coverage') or {}).get('clustered_hard_fraction', 0.0)):.3f}"
            f"/cov_s={float((r.get('hierarchy_coverage') or {}).get('soft_coverage', 0.0)):.3f}"
            f"/src={str(r.get('hierarchy_provenance', {}).get('source', cluster_source))}"
            for r in rows
        )
        logger(
            f"  [hier] seed portfolio prescore: {summary}; selected={selected['name']}; "
            f"hierarchy_first={int(hierarchy_first)}; contract_reference={reference_name}"
            f"/{seed_reference_kind}"
        )
        _emit_seed_status(str(selected["name"]), "selected")
        return selected["hard"], selected["soft"], float(selected["score"]), rows

    _emit_seed_status("dreamplace", "building")
    hard, soft = _prepare_dreamplace_candidate(
        group_weight=group_weight,
        random_seed=random_seed,
        scratch_root=scratch_root,
    )
    return _select_seed_portfolio(hard, soft)
