"""Seed-portfolio helpers for hierarchy floorplan execution."""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from typing import Any, Callable

import numpy as np
import torch

from placer.local_search.hierarchy_quality import (
    hierarchy_quality_vector,
    hierarchy_vector_contract,
    hierarchy_vector_limits,
)
from placer.local_search.plateau_telemetry import log_plateau_event
from utils.config import HAS_NUMBA, _numba_njit

if HAS_NUMBA:

    @_numba_njit(cache=True, fastmath=False)
    def _synthetic_clearance_delta_jit(hard, eligible, temp_hw, temp_hh, delta):
        """Accumulate one synthetic-clearance push iteration."""
        delta.fill(0.0)
        n = hard.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                move_i = eligible[i]
                move_j = eligible[j]
                if not (move_i or move_j):
                    continue
                dx = hard[i, 0] - hard[j, 0]
                dy = hard[i, 1] - hard[j, 1]
                overlap_x = temp_hw[i] + temp_hw[j] - abs(dx)
                overlap_y = temp_hh[i] + temp_hh[j] - abs(dy)
                if overlap_x <= 0.0 or overlap_y <= 0.0:
                    continue
                if overlap_x <= overlap_y:
                    push_x = 0.5 * overlap_x * (1.0 if dx >= 0.0 else -1.0)
                    if move_i and move_j:
                        delta[i, 0] += push_x
                        delta[j, 0] -= push_x
                    elif move_i:
                        delta[i, 0] += 2.0 * push_x
                    else:
                        delta[j, 0] -= 2.0 * push_x
                else:
                    push_y = 0.5 * overlap_y * (1.0 if dy >= 0.0 else -1.0)
                    if move_i and move_j:
                        delta[i, 1] += push_y
                        delta[j, 1] -= push_y
                    elif move_i:
                        delta[i, 1] += 2.0 * push_y
                    else:
                        delta[j, 1] -= 2.0 * push_y


def select_seed_candidate(
    rows: list[dict[str, object]],
    *,
    hierarchy_first: bool,
    absolute_slack: float,
    relative_slack: float,
    component_absolute_slack: Mapping[str, float] | None = None,
    component_relative_slack: float = 0.0,
    component_reference_name: str = "initial",
) -> dict[str, object]:
    """Select the lowest proxy seed inside the active hierarchy contract."""
    if not rows:
        raise ValueError("seed portfolio is empty")
    eligible = rows
    if component_absolute_slack is not None:
        reference = next(
            (row for row in rows if str(row["name"]) == str(component_reference_name)),
            min(rows, key=lambda row: (float(row["score"]), str(row["name"]))),
        )
        reference_vector = reference.get("hierarchy_vector")
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
    if not hierarchy_first:
        return min(eligible, key=lambda row: (float(row["score"]), str(row["name"])))
    best_quality = min(float(row["hierarchy_composite"]) for row in eligible)
    slack = max(float(absolute_slack), abs(best_quality) * float(relative_slack))
    hierarchy_band = [
        row for row in eligible if float(row["hierarchy_composite"]) <= best_quality + slack
    ]
    return min(hierarchy_band, key=lambda row: (float(row["score"]), str(row["name"])))


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
) -> tuple[np.ndarray, np.ndarray, float, list[dict[str, object]]]:
    """Create and score the seed portfolio used by hierarchy floorplanning.

    The seed portfolio starts from the DREAMPlace seed and several small
    perturbations that preserve hierarchy intent while improving overlap
    survivability before downstream region cleanup.
    """
    benchmark_name = str(benchmark.name)

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
        *,
        constraint_graph: bool = False,
    ) -> np.ndarray:
        if not constraint_graph:
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
        from placer.legalize.constraint_graph import _will_legalize_constraint_graph

        projected, stats = _will_legalize_constraint_graph(
            hard_xy,
            movable[:n],
            sizes[:n],
            hw,
            hh,
            cw,
            ch,
            n,
            deadline=seed_deadline,
            max_rounds=max(1, int(getattr(const, "HIER_CONSTRAINT_GRAPH_MAX_ROUNDS", 6))),
        )
        logger(
            "  [hier] constraint-graph legalize "
            f"{name}: overlaps {int(stats['initial_overlaps'])}->"
            f"{int(stats['final_overlaps'])}, constraints="
            f"{int(stats['x_constraints'])}+{int(stats['y_constraints'])}, "
            f"rounds={int(stats['rounds'])}, infeasible={int(bool(stats['infeasible']))}, "
            f"elapsed={float(stats['elapsed_s']):.3f}s"
        )
        return projected

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
        constraint_graph: bool = False,
    ):
        legalize_t0 = time.perf_counter()
        hard_xy, soft_xy = _clip_seed(hard_xy, soft_xy)
        seed_deadline = time.monotonic() + float(budget_s)
        legal_hard = _first_legalize(
            hard_xy,
            seed_deadline,
            name,
            constraint_graph=constraint_graph,
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
        _log_stage_timing(
            "seed_creation",
            float(time.perf_counter() - legalize_t0),
            candidate=str(name),
            constraint_graph=bool(constraint_graph),
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

    def _expanded_seed(base_hard, base_soft):
        hard = base_hard.copy()
        soft = base_soft.copy()
        frac = float(const.HIER_SEED_EXPANSION_FRAC)
        hard_mov = movable[:n]
        if np.any(hard_mov):
            center = np.mean(hard[hard_mov], axis=0)
            hard[hard_mov] = center + (1.0 + frac) * (hard[hard_mov] - center)
        if n_soft and np.any(movable[n : n + n_soft]):
            soft_mov_local = movable[n : n + n_soft]
            center = np.mean(soft[soft_mov_local], axis=0)
            soft[soft_mov_local] = center + (1.0 + frac) * (soft[soft_mov_local] - center)
        return hard, soft

    def _synthetic_clearance_seed(base_hard, base_soft):
        hard = base_hard.copy()
        hard_mov = movable[:n]
        area = sizes[:n, 0] * sizes[:n, 1]
        area_limit = float(np.percentile(area, float(const.HIER_SEED_CLEARANCE_AREA_PCT)))
        eligible = hard_mov & (area <= area_limit)
        if not np.any(eligible):
            return hard, base_soft.copy()
        temp_hw = hw.copy()
        temp_hh = hh.copy()
        temp_hw[eligible] *= 1.0 + float(const.HIER_SEED_CLEARANCE_FRAC)
        temp_hh[eligible] *= 1.0 + float(const.HIER_SEED_CLEARANCE_FRAC)
        iters = max(1, int(const.HIER_SEED_CLEARANCE_ITERS))
        delta = np.zeros_like(hard)
        for _ in range(iters):
            if HAS_NUMBA:
                _synthetic_clearance_delta_jit(hard, eligible, temp_hw, temp_hh, delta)
            else:
                delta.fill(0.0)
                for i in range(n):
                    for j in range(i + 1, n):
                        move_i = bool(eligible[i])
                        move_j = bool(eligible[j])
                        if not (move_i or move_j):
                            continue
                        dx = float(hard[i, 0] - hard[j, 0])
                        dy = float(hard[i, 1] - hard[j, 1])
                        ox = float(temp_hw[i] + temp_hw[j] - abs(dx))
                        oy = float(temp_hh[i] + temp_hh[j] - abs(dy))
                        if ox <= 0.0 or oy <= 0.0:
                            continue
                        if ox <= oy:
                            sx = 1.0 if dx >= 0.0 else -1.0
                            push = np.array([0.5 * ox * sx, 0.0], dtype=np.float64)
                        else:
                            sy = 1.0 if dy >= 0.0 else -1.0
                            push = np.array([0.0, 0.5 * oy * sy], dtype=np.float64)
                        if move_i and move_j:
                            delta[i] += push
                            delta[j] -= push
                        elif move_i:
                            delta[i] += 2.0 * push
                        elif move_j:
                            delta[j] -= 2.0 * push
            hard[eligible] += 0.5 * delta[eligible]
            hard, _soft = _clip_seed(hard, base_soft)
        return hard, base_soft.copy()

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
        immutable_contract_keys = (
            "cluster_compactness",
            "worst_cluster_spread",
            "neighbor_impurity",
            "edge_stretch",
        )
        contract_absolute_slack = const.HIER_VECTOR_CONTRACT_ABS_SLACK
        contract_relative_slack = float(const.HIER_VECTOR_CONTRACT_REL_SLACK)
        seed_reference_vector = hierarchy_quality_vector(
            init_hard,
            init_soft,
            clusters,
            csofts,
            bridge_softs,
            hierarchy_edges,
            cw,
            ch,
        )
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
        mandatory = {"dreamplace", "constraint_graph_initial"}

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
        try:
            dp_passed, _ = _immutable_contract_pass(dp_hard, dp_soft)
            if not dp_passed:
                logger("  [hier] seed dreamplace failed immutable-contract prefilter; keeping for stability")
            rows.append(_score_seed("dreamplace", dp_hard, dp_soft, do_soft_cleanup=True))
        except Exception as exc:
            logger(
                "  [hier] seed dreamplace scoring failed: "
                f"{type(exc).__name__}: {exc}"
            )
        raw_candidates = []
        raw_candidates.append(("initial", init_hard, init_soft))
        for alpha in tuple(float(a) for a in const.HIER_SEED_BLEND_ALPHAS):
            hard = (1.0 - alpha) * dp_hard + alpha * init_hard
            soft = (1.0 - alpha) * dp_soft + alpha * init_soft
            raw_candidates.append((f"blend_{alpha:.2f}", hard, soft))
        raw_candidates.append(("expand", *_expanded_seed(dp_hard, dp_soft)))
        raw_candidates.append(("synthetic_clearance", *_synthetic_clearance_seed(dp_hard, dp_soft)))
        if _has_explicit_path_tags():
            raw_candidates.append(("route_channel", *_route_channel_seed(dp_hard, dp_soft)))
        for name, cand_h, cand_s in raw_candidates:
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
                rows.append(_score_seed(name, legal_hard, legal_soft))
            except Exception as exc:
                logger(f"  [hier] seed {name} failed scoring: {type(exc).__name__}: {exc}")
        try:
            cg_hard, cg_soft = _legalize_seed(
                "constraint_graph_initial",
                init_hard,
                init_soft,
                budget_s=45.0,
                constraint_graph=True,
            )
            cg_passed, _ = _immutable_contract_pass(cg_hard, cg_soft)
            if not cg_passed:
                logger(
                    "  [hier] seed constraint_graph_initial failed immutable contract prefilter "
                    "(kept due mandatory candidate path)"
                )
            rows.append(_score_seed("constraint_graph_initial", cg_hard, cg_soft))
        except Exception as exc:
            logger(
                "  [hier] seed constraint_graph_initial failed prescore: "
                f"{type(exc).__name__}: {exc}"
            )
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
                    (str(k), float(v))
                    for k, v in immutable_contract_limits.items()
                ),
            }
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
        summary = ", ".join(
            f"{r['name']}={float(r['score']):.4f}/hq={float(r['hierarchy_composite']):.5f}"
            f"/contract={int(bool(r.get('hierarchy_contract_eligible', True)))}"
            f"/cov_h={float((r.get('hierarchy_coverage') or {}).get('clustered_hard_fraction', 0.0)):.3f}"
            f"/cov_s={float((r.get('hierarchy_coverage') or {}).get('soft_coverage', 0.0)):.3f}"
            f"/src={str(r.get('hierarchy_provenance', {}).get('source', cluster_source))}"
            for r in rows
        )
        reference_name = str(selected.get("hierarchy_contract_reference", "initial"))
        logger(
            f"  [hier] seed portfolio prescore: {summary}; selected={selected['name']}; "
            f"hierarchy_first={int(hierarchy_first)}; contract_reference={reference_name}"
        )
        return selected["hard"], selected["soft"], float(selected["score"]), rows

    hard, soft = _prepare_dreamplace_candidate(
        group_weight=group_weight,
        random_seed=random_seed,
        scratch_root=scratch_root,
    )
    return _select_seed_portfolio(hard, soft)
