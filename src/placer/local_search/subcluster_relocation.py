"""One-level parent-bounded relocation of hierarchy child clusters."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence

import numpy as np

from placer.legalize.spiral import _will_legalize
from placer.local_search.fields import (
    _congestion_field,
    _density_field,
    cold_connected_component_target_pool,
    weighted_congestion_field,
)
from placer.local_search.hierarchy_swaps import _region_bounded_swap_relief
from placer.local_search.relocation import _relocation_moves, _soft_relocation_moves


def _hard_group_is_legal(
    hard_pos: np.ndarray,
    members: np.ndarray,
    targets: np.ndarray,
    hw: np.ndarray,
    hh: np.ndarray,
    *,
    tolerance: float = 1.0e-9,
) -> bool:
    """Check a rigid child against hard macros outside that child."""
    outside = np.ones(hard_pos.shape[0], dtype=bool)
    outside[members] = False
    if not bool(np.any(outside)):
        return True
    other = hard_pos[outside]
    other_hw = hw[outside]
    other_hh = hh[outside]
    dx = np.abs(targets[:, None, 0] - other[None, :, 0])
    dy = np.abs(targets[:, None, 1] - other[None, :, 1])
    separate = (dx + tolerance >= hw[members, None] + other_hw[None, :]) | (
        dy + tolerance >= hh[members, None] + other_hh[None, :]
    )
    return bool(np.all(separate))


def _hard_state_is_legal(
    hard_pos: np.ndarray,
    hw: np.ndarray,
    hh: np.ndarray,
    *,
    tolerance: float = 1.0e-9,
) -> bool:
    """Check hard-hard legality for a complete candidate state."""
    dx = np.abs(hard_pos[:, None, 0] - hard_pos[None, :, 0])
    dy = np.abs(hard_pos[:, None, 1] - hard_pos[None, :, 1])
    separate = (dx + tolerance >= hw[:, None] + hw[None, :]) | (
        dy + tolerance >= hh[:, None] + hh[None, :]
    )
    np.fill_diagonal(separate, True)
    return bool(np.all(separate))


def _field_values(
    xy: np.ndarray,
    field: np.ndarray,
    cell_w: float,
    cell_h: float,
) -> np.ndarray:
    """Sample a routing field at macro centers."""
    nr, nc = field.shape
    cols = np.clip((xy[:, 0] / cell_w).astype(np.int64), 0, nc - 1)
    rows = np.clip((xy[:, 1] / cell_h).astype(np.int64), 0, nr - 1)
    return field[rows, cols]


def _group_shift_bounds(
    hard_pos: np.ndarray,
    soft_pos: np.ndarray,
    members: np.ndarray,
    soft_indices: np.ndarray,
    hard_parent_region: np.ndarray,
    soft_parent_region: np.ndarray | None,
    soft_hw: np.ndarray,
    soft_hh: np.ndarray,
    cw: float,
    ch: float,
) -> tuple[float, float, float, float]:
    """Return the rigid shift interval keeping every member in its parent."""
    old_hard = hard_pos[members]
    old_soft = soft_pos[soft_indices]
    dx_lo = float(np.max(hard_parent_region[members, 0] - old_hard[:, 0]))
    dx_hi = float(np.min(hard_parent_region[members, 2] - old_hard[:, 0]))
    dy_lo = float(np.max(hard_parent_region[members, 1] - old_hard[:, 1]))
    dy_hi = float(np.min(hard_parent_region[members, 3] - old_hard[:, 1]))
    if soft_indices.size:
        if soft_parent_region is None:
            dx_lo = max(dx_lo, float(np.max(soft_hw[soft_indices] - old_soft[:, 0])))
            dx_hi = min(dx_hi, float(np.min(cw - soft_hw[soft_indices] - old_soft[:, 0])))
            dy_lo = max(dy_lo, float(np.max(soft_hh[soft_indices] - old_soft[:, 1])))
            dy_hi = min(dy_hi, float(np.min(ch - soft_hh[soft_indices] - old_soft[:, 1])))
        else:
            dx_lo = max(
                dx_lo,
                float(np.max(soft_parent_region[soft_indices, 0] - old_soft[:, 0])),
            )
            dx_hi = min(
                dx_hi,
                float(np.min(soft_parent_region[soft_indices, 2] - old_soft[:, 0])),
            )
            dy_lo = max(
                dy_lo,
                float(np.max(soft_parent_region[soft_indices, 1] - old_soft[:, 1])),
            )
            dy_hi = min(
                dy_hi,
                float(np.min(soft_parent_region[soft_indices, 3] - old_soft[:, 1])),
            )
    return dx_lo, dx_hi, dy_lo, dy_hi


def _legalize_group_candidate(
    hard_pos: np.ndarray,
    members: np.ndarray,
    targets: np.ndarray,
    hard_parent_region: np.ndarray,
    hw: np.ndarray,
    hh: np.ndarray,
    cw: float,
    ch: float,
    deadline: float | None,
) -> "np.ndarray | None":
    """Legalize only an affected child set against fixed outside macros."""
    trial = hard_pos.copy()
    trial[members] = targets
    movable = np.zeros(hard_pos.shape[0], dtype=bool)
    movable[members] = True
    outside = np.flatnonzero(~movable)
    area = 4.0 * hw[members] * hh[members]
    member_order = members[np.argsort(-area, kind="stable")]
    order = [int(index) for index in outside]
    order.extend(int(index) for index in member_order)
    sizes = np.column_stack([2.0 * hw, 2.0 * hh])
    legal = _will_legalize(
        trial,
        movable,
        sizes,
        hw,
        hh,
        cw,
        ch,
        hard_pos.shape[0],
        deadline=deadline,
        order=order,
    )
    placed = legal[members]
    inside = (
        (placed[:, 0] >= hard_parent_region[members, 0] - 1.0e-9)
        & (placed[:, 0] <= hard_parent_region[members, 2] + 1.0e-9)
        & (placed[:, 1] >= hard_parent_region[members, 1] - 1.0e-9)
        & (placed[:, 1] <= hard_parent_region[members, 3] + 1.0e-9)
    )
    if not bool(np.all(inside)) or not _hard_state_is_legal(legal, hw, hh):
        return None
    return placed


def _subcluster_relocation(
    hard_pos: np.ndarray,
    soft_pos: np.ndarray,
    hw: np.ndarray,
    hh: np.ndarray,
    soft_hw: np.ndarray,
    soft_hh: np.ndarray,
    cw: float,
    ch: float,
    n: int,
    benchmark,
    incremental_scorer,
    initial_score: float,
    *,
    child_clusters: Mapping[int, Sequence[int]],
    parent_clusters: Mapping[int, Sequence[int]],
    parent_children: Mapping[int, Sequence[int]],
    cluster_softs: Mapping[int, Sequence[int]],
    movable_h: np.ndarray,
    soft_movable: np.ndarray,
    hard_parent_region: np.ndarray | None,
    soft_parent_region: np.ndarray | None,
    child_graph_tension: Mapping[int, float] | None = None,
    graph_priority_weight: float = 0.0,
    candidate_allowed: Callable[[np.ndarray, np.ndarray], bool] | None = None,
    deadline: float | None = None,
    top_children: int = 4,
    top_swaps: int = 4,
    max_child_hard: int = 64,
    max_child_soft: int = 32,
    compact_scale: float = 0.90,
    cold_percentile: float = 35.0,
    max_components: int = 4,
    min_component_cells: int = 4,
    n_anchors: int = 2,
    shift_fractions: Sequence[float] = (0.5, 1.0),
    min_field_drop: float = 0.02,
    min_gain: float = 0.0001,
    max_scored: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Rigidly move one child cluster inside its retained parent region."""
    score_limit = None if max_scored is None else max(0, int(max_scored))
    stats = {
        "parents": int(len(parent_clusters)),
        "eligible_children": 0,
        "candidates": 0,
        "legal": 0,
        "hierarchy_rejects": 0,
        "field_rejects": 0,
        "swap_candidates": 0,
        "swap_legal": 0,
        "swap_scored": 0,
        "legalized_candidates": 0,
        "scored": 0,
        "accepts": 0,
        "accepted_kind": "none",
        "best_candidate_gain": 0.0,
        "graph_prioritized_children": 0,
        "graph_tension_mean": 0.0,
        "score_limit": score_limit,
        "quota_exhausted": bool(score_limit is not None and score_limit <= 0),
    }
    _subcluster_relocation.last_stats = stats
    if not parent_clusters or not parent_children or hard_parent_region is None:
        return hard_pos, soft_pos, 0, float(initial_score)

    movable_h = np.asarray(movable_h, dtype=bool)
    soft_movable = np.asarray(soft_movable, dtype=bool)
    nr, nc = int(benchmark.grid_rows), int(benchmark.grid_cols)
    field = weighted_congestion_field(incremental_scorer, nr, nc)
    if field is None or np.asarray(field).size == 0:
        return hard_pos, soft_pos, 0, float(initial_score)
    field = np.asarray(field, dtype=np.float64)
    field_span = max(float(np.max(field) - np.min(field)), 1.0e-12)
    cell_w, cell_h = float(cw) / nc, float(ch) / nr
    child_graph_tension = child_graph_tension or {}

    rows: list[dict[str, object]] = []
    for parent_id, child_ids in sorted(parent_children.items()):
        parent_members = set(
            int(index) for index in np.asarray(parent_clusters.get(int(parent_id), ())).reshape(-1)
        )
        if len(child_ids) < 2 or not parent_members:
            continue
        for child_id_raw in child_ids:
            child_id = int(child_id_raw)
            members = np.asarray(child_clusters.get(child_id, ()), dtype=np.int64).reshape(-1)
            members = members[(members >= 0) & (members < hard_pos.shape[0])]
            if (
                members.size < 2
                or members.size > max(2, int(max_child_hard))
                or not all(int(index) in parent_members for index in members)
                or not bool(np.all(movable_h[members]))
            ):
                continue
            soft_indices = np.asarray(cluster_softs.get(child_id, ()), dtype=np.int64) - int(n)
            soft_indices = soft_indices[(soft_indices >= 0) & (soft_indices < soft_pos.shape[0])]
            if soft_indices.size and not bool(np.all(soft_movable[soft_indices])):
                continue
            if soft_indices.size > max(0, int(max_child_soft)):
                continue
            points = hard_pos[members]
            if soft_indices.size:
                points = np.vstack([points, soft_pos[soft_indices]])
            heat = float(np.mean(_field_values(points, field, cell_w, cell_h)))
            graph_tension = max(0.0, float(child_graph_tension.get(child_id, 0.0)))
            rows.append(
                {
                    "parent": int(parent_id),
                    "child": child_id,
                    "members": members,
                    "soft_indices": soft_indices,
                    "heat": heat,
                    "graph_tension": graph_tension,
                    "priority": heat
                    + max(0.0, float(graph_priority_weight)) * field_span * graph_tension,
                }
            )
    rows.sort(key=lambda row: (-float(row["priority"]), int(row["parent"]), int(row["child"])))
    all_rows = rows
    rows = all_rows[: max(1, int(top_children))]
    stats["eligible_children"] = int(len(all_rows))
    stats["graph_prioritized_children"] = int(
        sum(float(row["graph_tension"]) > 0.0 for row in rows)
    )
    stats["graph_tension_mean"] = (
        float(np.mean([float(row["graph_tension"]) for row in rows])) if rows else 0.0
    )
    if not rows:
        return hard_pos, soft_pos, 0, float(initial_score)

    pool = cold_connected_component_target_pool(
        field,
        cold_percentile=float(cold_percentile),
        max_components=max(1, int(max_components)),
        min_cells=max(1, int(min_component_cells)),
        size_weight=0.35,
    )
    pool_indices = np.asarray(pool.get("indices", ()), dtype=np.int64)
    if pool_indices.size == 0:
        return hard_pos, soft_pos, 0, float(initial_score)
    pool_x = ((pool_indices % nc).astype(np.float64) + 0.5) * cell_w
    pool_y = ((pool_indices // nc).astype(np.float64) + 0.5) * cell_h
    pool_field = field.ravel()[pool_indices]
    fractions = sorted({float(value) for value in shift_fractions if 0.0 < float(value) <= 1.0})
    if not fractions:
        return hard_pos, soft_pos, 0, float(initial_score)

    best_score = float(initial_score)
    best_move = None
    best_kind = "none"
    canvas2 = max(float(cw) ** 2 + float(ch) ** 2, 1.0)
    seen_shifts: set[tuple[int, int, int]] = set()
    for row in rows:
        if score_limit is not None and int(stats["scored"]) >= score_limit:
            break
        if deadline is not None and time.monotonic() >= deadline:
            break
        child_id = int(row["child"])
        members = np.asarray(row["members"], dtype=np.int64)
        soft_indices = np.asarray(row["soft_indices"], dtype=np.int64)
        old_hard = hard_pos[members]
        old_soft = soft_pos[soft_indices]
        centroid = np.mean(old_hard, axis=0)

        dx_lo, dx_hi, dy_lo, dy_hi = _group_shift_bounds(
            hard_pos,
            soft_pos,
            members,
            soft_indices,
            hard_parent_region,
            soft_parent_region,
            soft_hw,
            soft_hh,
            cw,
            ch,
        )
        if dx_lo > dx_hi or dy_lo > dy_hi:
            continue

        feasible = (
            (pool_x >= float(np.min(hard_parent_region[members, 0])) - 1.0e-9)
            & (pool_x <= float(np.max(hard_parent_region[members, 2])) + 1.0e-9)
            & (pool_y >= float(np.min(hard_parent_region[members, 1])) - 1.0e-9)
            & (pool_y <= float(np.max(hard_parent_region[members, 3])) + 1.0e-9)
        )
        possible = np.flatnonzero(feasible)
        full_shifts: list[np.ndarray] = []
        if possible.size:
            distance2 = (pool_x[possible] - centroid[0]) ** 2 + (
                pool_y[possible] - centroid[1]
            ) ** 2
            rank = pool_field[possible] + 0.05 * distance2 / canvas2
            anchors = possible[np.argsort(rank, kind="stable")[: max(1, int(n_anchors))]]
            full_shifts.extend(
                np.asarray(
                    [pool_x[int(anchor)] - centroid[0], pool_y[int(anchor)] - centroid[1]],
                    dtype=np.float64,
                )
                for anchor in anchors
            )
        # Packed parents often have no cell center inside the exact rigid-shift
        # interval. Boundary shifts still expose the available local room.
        full_shifts.extend(
            np.asarray(shift, dtype=np.float64)
            for shift in (
                (dx_lo, 0.0),
                (dx_hi, 0.0),
                (0.0, dy_lo),
                (0.0, dy_hi),
                (dx_lo, dy_lo),
                (dx_lo, dy_hi),
                (dx_hi, dy_lo),
                (dx_hi, dy_hi),
            )
        )

        for full_shift in full_shifts:
            for fraction in fractions:
                if score_limit is not None and int(stats["scored"]) >= score_limit:
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    break
                shift = float(fraction) * full_shift
                if abs(float(shift[0])) < 0.05 * cell_w and abs(float(shift[1])) < 0.05 * cell_h:
                    continue
                shift_key = (
                    child_id,
                    int(round(float(shift[0]) * 1.0e6)),
                    int(round(float(shift[1]) * 1.0e6)),
                )
                if shift_key in seen_shifts:
                    continue
                seen_shifts.add(shift_key)
                stats["candidates"] += 1

                new_hard = old_hard + shift
                used_legalizer = False
                hard_inside = (
                    (new_hard[:, 0] >= hard_parent_region[members, 0] - 1.0e-9)
                    & (new_hard[:, 0] <= hard_parent_region[members, 2] + 1.0e-9)
                    & (new_hard[:, 1] >= hard_parent_region[members, 1] - 1.0e-9)
                    & (new_hard[:, 1] <= hard_parent_region[members, 3] + 1.0e-9)
                )
                if not bool(np.all(hard_inside)) or not _hard_group_is_legal(
                    hard_pos,
                    members,
                    new_hard,
                    hw,
                    hh,
                ):
                    compact_targets = (
                        centroid
                        + float(np.clip(compact_scale, 0.5, 1.0)) * (old_hard - centroid)
                        + shift
                    )
                    legalized = _legalize_group_candidate(
                        hard_pos,
                        members,
                        compact_targets,
                        hard_parent_region,
                        hw,
                        hh,
                        cw,
                        ch,
                        deadline,
                    )
                    if legalized is None:
                        continue
                    new_hard = legalized
                    used_legalizer = True
                    stats["legalized_candidates"] += 1
                actual_shift = np.mean(new_hard, axis=0) - centroid
                new_soft = old_soft + actual_shift
                if soft_indices.size:
                    if soft_parent_region is None:
                        soft_inside = (
                            (new_soft[:, 0] >= soft_hw[soft_indices] - 1.0e-9)
                            & (new_soft[:, 0] <= cw - soft_hw[soft_indices] + 1.0e-9)
                            & (new_soft[:, 1] >= soft_hh[soft_indices] - 1.0e-9)
                            & (new_soft[:, 1] <= ch - soft_hh[soft_indices] + 1.0e-9)
                        )
                    else:
                        soft_inside = (
                            (new_soft[:, 0] >= soft_parent_region[soft_indices, 0] - 1.0e-9)
                            & (new_soft[:, 0] <= soft_parent_region[soft_indices, 2] + 1.0e-9)
                            & (new_soft[:, 1] >= soft_parent_region[soft_indices, 1] - 1.0e-9)
                            & (new_soft[:, 1] <= soft_parent_region[soft_indices, 3] + 1.0e-9)
                        )
                    if not bool(np.all(soft_inside)):
                        continue
                stats["legal"] += 1
                moved_points = (
                    new_hard if not soft_indices.size else np.vstack([new_hard, new_soft])
                )
                new_heat = float(np.mean(_field_values(moved_points, field, cell_w, cell_h)))
                if new_heat >= float(row["heat"]) - float(min_field_drop):
                    stats["field_rejects"] += 1
                    continue

                trial_hard = hard_pos.copy()
                trial_soft = soft_pos.copy()
                trial_hard[members] = new_hard
                trial_soft[soft_indices] = new_soft
                if candidate_allowed is not None and not bool(
                    candidate_allowed(trial_hard, trial_soft)
                ):
                    stats["hierarchy_rejects"] += 1
                    continue

                score = float(
                    incremental_scorer.score_move_group(
                        members,
                        new_hard,
                        soft_indices,
                        new_soft,
                    )
                )
                stats["scored"] += 1
                stats["best_candidate_gain"] = max(
                    float(stats["best_candidate_gain"]),
                    float(initial_score) - score,
                )
                if score < best_score - max(1.0e-9, float(min_gain)):
                    best_score = score
                    best_kind = (
                        "localized_child_relocation" if used_legalizer else "rigid_child_relocation"
                    )
                    best_move = (
                        members.copy(),
                        new_hard.copy(),
                        soft_indices.copy(),
                        new_soft.copy(),
                    )

    by_parent: dict[int, list[dict[str, object]]] = {}
    for row in all_rows:
        by_parent.setdefault(int(row["parent"]), []).append(row)
    swap_pairs = []
    for parent_id, siblings in by_parent.items():
        for left_index, left in enumerate(siblings):
            for right in siblings[left_index + 1 :]:
                contrast = abs(float(left["priority"]) - float(right["priority"]))
                swap_pairs.append(
                    (
                        -contrast,
                        -max(float(left["priority"]), float(right["priority"])),
                        int(parent_id),
                        int(left["child"]),
                        int(right["child"]),
                        left,
                        right,
                    )
                )
    swap_pairs.sort(key=lambda row: row[:5])
    for _contrast, _heat, _parent, _left_id, _right_id, left, right in swap_pairs[
        : max(0, int(top_swaps))
    ]:
        if score_limit is not None and int(stats["scored"]) >= score_limit:
            break
        if deadline is not None and time.monotonic() >= deadline:
            break
        left_members = np.asarray(left["members"], dtype=np.int64)
        right_members = np.asarray(right["members"], dtype=np.int64)
        left_soft = np.asarray(left["soft_indices"], dtype=np.int64)
        right_soft = np.asarray(right["soft_indices"], dtype=np.int64)
        left_center = np.mean(hard_pos[left_members], axis=0)
        right_center = np.mean(hard_pos[right_members], axis=0)
        left_shift = right_center - left_center
        right_shift = left_center - right_center
        stats["candidates"] += 1
        stats["swap_candidates"] += 1
        members = np.concatenate([left_members, right_members]).astype(np.int64)
        soft_indices = np.concatenate([left_soft, right_soft]).astype(np.int64)
        new_hard = np.vstack(
            [hard_pos[left_members] + left_shift, hard_pos[right_members] + right_shift]
        )
        trial_hard = hard_pos.copy()
        trial_hard[members] = new_hard
        hard_inside = (
            (new_hard[:, 0] >= hard_parent_region[members, 0] - 1.0e-9)
            & (new_hard[:, 0] <= hard_parent_region[members, 2] + 1.0e-9)
            & (new_hard[:, 1] >= hard_parent_region[members, 1] - 1.0e-9)
            & (new_hard[:, 1] <= hard_parent_region[members, 3] + 1.0e-9)
        )
        if not bool(np.all(hard_inside)) or not _hard_state_is_legal(trial_hard, hw, hh):
            scale = float(np.clip(compact_scale, 0.5, 1.0))
            compact_targets = np.vstack(
                [
                    right_center + scale * (hard_pos[left_members] - left_center),
                    left_center + scale * (hard_pos[right_members] - right_center),
                ]
            )
            legalized = _legalize_group_candidate(
                hard_pos,
                members,
                compact_targets,
                hard_parent_region,
                hw,
                hh,
                cw,
                ch,
                deadline,
            )
            if legalized is None:
                continue
            new_hard = legalized
            trial_hard[members] = new_hard
            stats["legalized_candidates"] += 1
            swap_kind = "localized_sibling_swap"
        else:
            swap_kind = "rigid_sibling_swap"
        left_actual_shift = np.mean(new_hard[: left_members.size], axis=0) - left_center
        right_actual_shift = np.mean(new_hard[left_members.size :], axis=0) - right_center
        new_soft = np.vstack(
            [
                soft_pos[left_soft] + left_actual_shift,
                soft_pos[right_soft] + right_actual_shift,
            ]
        )
        if soft_indices.size:
            if soft_parent_region is None:
                soft_inside = (
                    (new_soft[:, 0] >= soft_hw[soft_indices] - 1.0e-9)
                    & (new_soft[:, 0] <= cw - soft_hw[soft_indices] + 1.0e-9)
                    & (new_soft[:, 1] >= soft_hh[soft_indices] - 1.0e-9)
                    & (new_soft[:, 1] <= ch - soft_hh[soft_indices] + 1.0e-9)
                )
            else:
                soft_inside = (
                    (new_soft[:, 0] >= soft_parent_region[soft_indices, 0] - 1.0e-9)
                    & (new_soft[:, 0] <= soft_parent_region[soft_indices, 2] + 1.0e-9)
                    & (new_soft[:, 1] >= soft_parent_region[soft_indices, 1] - 1.0e-9)
                    & (new_soft[:, 1] <= soft_parent_region[soft_indices, 3] + 1.0e-9)
                )
            if not bool(np.all(soft_inside)):
                continue
        trial_soft = soft_pos.copy()
        trial_soft[soft_indices] = new_soft
        stats["legal"] += 1
        stats["swap_legal"] += 1
        if candidate_allowed is not None and not bool(candidate_allowed(trial_hard, trial_soft)):
            stats["hierarchy_rejects"] += 1
            continue
        score = float(
            incremental_scorer.score_move_group(
                members,
                new_hard,
                soft_indices,
                new_soft,
            )
        )
        stats["scored"] += 1
        stats["swap_scored"] += 1
        stats["best_candidate_gain"] = max(
            float(stats["best_candidate_gain"]),
            float(initial_score) - score,
        )
        if score < best_score - max(1.0e-9, float(min_gain)):
            best_score = score
            best_kind = swap_kind
            best_move = (
                members.copy(),
                new_hard.copy(),
                soft_indices.copy(),
                new_soft.copy(),
            )

    if best_move is not None:
        members, new_hard, soft_indices, new_soft = best_move
        incremental_scorer.commit_move_group(members, new_hard, soft_indices, new_soft)
        hard_pos[members] = new_hard
        soft_pos[soft_indices] = new_soft
        stats["accepts"] = 1
        stats["accepted_kind"] = best_kind

    stats["quota_exhausted"] = bool(score_limit is not None and int(stats["scored"]) >= score_limit)
    _subcluster_relocation.last_stats = stats
    return hard_pos, soft_pos, int(stats["accepts"]), float(best_score)


def _deep_cluster_field_heat(
    hard_pos: np.ndarray,
    soft_pos: np.ndarray,
    child_clusters: Mapping[int, Sequence[int]],
    cluster_softs: Mapping[int, Sequence[int]],
    field: np.ndarray | None,
    *,
    n: int,
    cw: float,
    ch: float,
) -> dict[int, float]:
    """Sample one placement field over every deepest child and its owned softs."""
    if field is None or np.asarray(field).size == 0:
        return {}
    field = np.asarray(field, dtype=np.float64)
    nr, nc = field.shape
    cell_w, cell_h = float(cw) / nc, float(ch) / nr
    result: dict[int, float] = {}
    for child_id, raw_members in child_clusters.items():
        members = np.asarray(raw_members, dtype=np.int64).reshape(-1)
        members = members[(members >= 0) & (members < hard_pos.shape[0])]
        points = hard_pos[members]
        soft_indices = np.asarray(cluster_softs.get(int(child_id), ()), dtype=np.int64) - int(n)
        soft_indices = soft_indices[(soft_indices >= 0) & (soft_indices < soft_pos.shape[0])]
        if soft_indices.size:
            points = (
                np.vstack([points, soft_pos[soft_indices]])
                if points.size
                else soft_pos[soft_indices]
            )
        if points.size:
            result[int(child_id)] = float(np.mean(_field_values(points, field, cell_w, cell_h)))
    return result


def _deep_cluster_margin_fractions(
    child_clusters: Mapping[int, Sequence[int]],
    congestion_heat: Mapping[int, float] | None,
    density_heat: Mapping[int, float] | None,
    graph_tension: Mapping[int, float] | None,
    *,
    base_margin: float,
    extra_margin: float,
    congestion_weight: float = 0.45,
    density_weight: float = 0.35,
    graph_weight: float = 0.20,
) -> dict[int, float]:
    """Blend normalized field heat and graph pressure into child-box margins."""
    congestion_heat = congestion_heat or {}
    density_heat = density_heat or {}
    graph_tension = graph_tension or {}
    weights = np.asarray(
        [
            max(0.0, float(congestion_weight)),
            max(0.0, float(density_weight)),
            max(0.0, float(graph_weight)),
        ],
        dtype=np.float64,
    )
    if float(np.sum(weights)) <= 1.0e-12:
        weights[:] = (1.0, 0.0, 0.0)

    def _scale(values: Mapping[int, float]) -> dict[int, float]:
        top = max((max(0.0, float(value)) for value in values.values()), default=0.0)
        if top <= 1.0e-12:
            return {}
        return {int(cid): max(0.0, float(value)) / top for cid, value in values.items()}

    cong = _scale(congestion_heat)
    dens = _scale(density_heat)
    graph = _scale(graph_tension)
    margins: dict[int, float] = {}
    for child_id in child_clusters:
        signals = np.asarray(
            [
                cong.get(int(child_id), 0.0),
                dens.get(int(child_id), 0.0),
                graph.get(int(child_id), 0.0),
            ],
            dtype=np.float64,
        )
        present = np.asarray(
            [bool(cong), bool(dens), bool(graph)],
            dtype=np.float64,
        )
        active_weights = weights * present
        denom = float(np.sum(active_weights))
        pressure = float(np.dot(active_weights, signals) / denom) if denom > 1.0e-12 else 0.0
        margins[int(child_id)] = max(0.0, float(base_margin)) + max(
            0.0, float(extra_margin)
        ) * float(np.clip(pressure, 0.0, 1.0))
    return margins


def _graph_anchor_arrays(
    hard_pos: np.ndarray,
    soft_pos: np.ndarray,
    child_clusters: Mapping[int, Sequence[int]],
    cluster_softs: Mapping[int, Sequence[int]],
    graph_edges,
    *,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Map weighted neighboring-child centroids onto member target anchors."""
    hard_anchor = np.asarray(hard_pos, dtype=np.float64).copy()
    soft_anchor = np.asarray(soft_pos, dtype=np.float64).copy()
    centroids: dict[int, np.ndarray] = {}
    for child_id, raw_members in child_clusters.items():
        members = np.asarray(raw_members, dtype=np.int64).reshape(-1)
        members = members[(members >= 0) & (members < hard_pos.shape[0])]
        if members.size:
            centroids[int(child_id)] = np.asarray(np.mean(hard_pos[members], axis=0))
    neighbor_sum = {cid: np.zeros(2, dtype=np.float64) for cid in centroids}
    neighbor_weight = {cid: 0.0 for cid in centroids}
    for edge in graph_edges or ():
        left = int(getattr(edge, "src", -1))
        right = int(getattr(edge, "dst", -1))
        weight = max(0.0, float(getattr(edge, "weight", 1.0)))
        if weight <= 0.0 or left not in centroids or right not in centroids:
            continue
        neighbor_sum[left] += weight * centroids[right]
        neighbor_sum[right] += weight * centroids[left]
        neighbor_weight[left] += weight
        neighbor_weight[right] += weight
    for child_id, centroid in centroids.items():
        if neighbor_weight[child_id] <= 1.0e-12:
            anchor = centroid
        else:
            anchor = neighbor_sum[child_id] / neighbor_weight[child_id]
        members = np.asarray(child_clusters[child_id], dtype=np.int64)
        members = members[(members >= 0) & (members < hard_anchor.shape[0])]
        hard_anchor[members] = anchor
        soft_indices = np.asarray(cluster_softs.get(child_id, ()), dtype=np.int64) - int(n)
        soft_indices = soft_indices[(soft_indices >= 0) & (soft_indices < soft_anchor.shape[0])]
        soft_anchor[soft_indices] = anchor
    return hard_anchor, soft_anchor


def _region_target_pool(
    region: np.ndarray,
    indices: np.ndarray,
    *,
    rows: int,
    cols: int,
    cw: float,
    ch: float,
) -> np.ndarray:
    """Return grid-cell centers intersecting the union of selected member boxes."""
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    indices = indices[(indices >= 0) & (indices < region.shape[0])]
    if indices.size == 0:
        return np.zeros(0, dtype=np.int64)
    boxes = region[indices]
    xlo, ylo = float(np.min(boxes[:, 0])), float(np.min(boxes[:, 1]))
    xhi, yhi = float(np.max(boxes[:, 2])), float(np.max(boxes[:, 3]))
    cell_w, cell_h = float(cw) / int(cols), float(ch) / int(rows)
    xs = (np.arange(int(cols), dtype=np.float64) + 0.5) * cell_w
    ys = (np.arange(int(rows), dtype=np.float64) + 0.5) * cell_h
    valid_cols = np.flatnonzero((xs >= xlo - 1.0e-9) & (xs <= xhi + 1.0e-9))
    valid_rows = np.flatnonzero((ys >= ylo - 1.0e-9) & (ys <= yhi + 1.0e-9))
    if not valid_cols.size or not valid_rows.size:
        return np.zeros(0, dtype=np.int64)
    return (valid_rows[:, None] * int(cols) + valid_cols[None, :]).reshape(-1)


def _deep_cluster_internal_relief(
    hard_pos: np.ndarray,
    soft_pos: np.ndarray,
    sizes: np.ndarray,
    hw: np.ndarray,
    hh: np.ndarray,
    soft_hw: np.ndarray,
    soft_hh: np.ndarray,
    cw: float,
    ch: float,
    n: int,
    plc,
    benchmark,
    incremental_scorer,
    initial_score: float,
    *,
    child_clusters: Mapping[int, Sequence[int]],
    cluster_softs: Mapping[int, Sequence[int]],
    subcluster_labels: np.ndarray,
    graph_edges,
    graph_confidence: Mapping[int, float] | None,
    graph_tension: Mapping[int, float] | None,
    seed_hard_xy: np.ndarray | None,
    movable_h: np.ndarray,
    soft_movable: np.ndarray,
    hard_region: np.ndarray | None,
    soft_region: np.ndarray | None,
    candidate_allowed: Callable[[np.ndarray, np.ndarray], bool] | None = None,
    deadline: float | None = None,
    top_children: int = 4,
    hard_targets: int = 4,
    soft_targets: int = 4,
    relocation_targets: int = 3,
    swap_k: int = 4,
    graph_priority_weight: float = 0.20,
    graph_anchor_blend: float = 0.15,
    graph_delta_weight: float = 0.10,
    min_gain: float = 0.0001,
    max_scored_per_call: int = 8,
    max_scored: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Relocate and swap members inside immutable deepest-child boxes."""
    score_limit = None if max_scored is None else max(0, int(max_scored))
    stats = {
        "eligible_children": 0,
        "selected_children": 0,
        "graph_prioritized_children": 0,
        "candidates": 0,
        "legal": 0,
        "scored": 0,
        "hierarchy_rejects": 0,
        "hard_accepts": 0,
        "soft_accepts": 0,
        "swap_accepts": 0,
        "swap_scored": 0,
        "accepts": 0,
        "score_limit": score_limit,
        "quota_exhausted": bool(score_limit is not None and score_limit <= 0),
    }
    _deep_cluster_internal_relief.last_stats = stats
    if not child_clusters or hard_region is None or soft_region is None:
        return hard_pos, soft_pos, 0, float(initial_score)

    nr, nc = int(benchmark.grid_rows), int(benchmark.grid_cols)
    congestion = _congestion_field(incremental_scorer, nr, nc)
    density = _density_field(incremental_scorer, nr, nc)
    if congestion is None and density is None:
        return hard_pos, soft_pos, 0, float(initial_score)
    congestion_heat = _deep_cluster_field_heat(
        hard_pos, soft_pos, child_clusters, cluster_softs, congestion, n=n, cw=cw, ch=ch
    )
    density_heat = _deep_cluster_field_heat(
        hard_pos, soft_pos, child_clusters, cluster_softs, density, n=n, cw=cw, ch=ch
    )
    graph_tension = graph_tension or {}

    def _normalized(values: Mapping[int, float]) -> dict[int, float]:
        maximum = max((max(0.0, float(value)) for value in values.values()), default=0.0)
        if maximum <= 1.0e-12:
            return {}
        return {int(cid): max(0.0, float(value)) / maximum for cid, value in values.items()}

    cong_norm = _normalized(congestion_heat)
    density_norm = _normalized(density_heat)
    graph_norm = _normalized(graph_tension)
    rows = []
    movable_h = np.asarray(movable_h, dtype=bool)
    soft_movable = np.asarray(soft_movable, dtype=bool)
    for child_id, raw_members in child_clusters.items():
        members = np.asarray(raw_members, dtype=np.int64).reshape(-1)
        members = members[(members >= 0) & (members < hard_pos.shape[0])]
        child_movable = members[movable_h[members]]
        soft_indices = np.asarray(cluster_softs.get(int(child_id), ()), dtype=np.int64) - int(n)
        soft_indices = soft_indices[(soft_indices >= 0) & (soft_indices < soft_pos.shape[0])]
        child_soft_movable = soft_indices[soft_movable[soft_indices]]
        if not child_movable.size and not child_soft_movable.size:
            continue
        graph_value = graph_norm.get(int(child_id), 0.0)
        priority = (
            0.45 * cong_norm.get(int(child_id), 0.0)
            + 0.35 * density_norm.get(int(child_id), 0.0)
            + max(0.0, float(graph_priority_weight)) * graph_value
        )
        rows.append(
            (
                -float(priority),
                int(child_id),
                members,
                child_movable,
                soft_indices,
                child_soft_movable,
                float(graph_value),
            )
        )
    rows.sort(key=lambda row: (row[0], row[1]))
    stats["eligible_children"] = int(len(rows))
    rows = rows[: max(1, int(top_children))]
    stats["selected_children"] = int(len(rows))
    stats["graph_prioritized_children"] = int(sum(row[6] > 0.0 for row in rows))
    if not rows:
        return hard_pos, soft_pos, 0, float(initial_score)

    hard_anchor, soft_anchor = _graph_anchor_arrays(
        hard_pos,
        soft_pos,
        child_clusters,
        cluster_softs,
        graph_edges,
        n=n,
    )
    assigned_hard = np.flatnonzero(np.asarray(subcluster_labels, dtype=np.int64) >= 0)
    best_score = float(initial_score)
    accepts = 0

    def _remaining() -> int | None:
        if score_limit is None:
            return None
        return max(0, score_limit - int(stats["scored"]))

    def _call_quota() -> int | None:
        remaining = _remaining()
        cap = max(1, int(max_scored_per_call))
        return cap if remaining is None else min(cap, remaining)

    def _hard_allowed(index: int, x: float, y: float) -> bool:
        box = hard_region[int(index)]
        if x < box[0] - 1.0e-9 or x > box[2] + 1.0e-9:
            return False
        if y < box[1] - 1.0e-9 or y > box[3] + 1.0e-9:
            return False
        old = hard_pos[int(index)].copy()
        hard_pos[int(index)] = (float(x), float(y))
        try:
            return candidate_allowed is None or bool(candidate_allowed(hard_pos, soft_pos))
        finally:
            hard_pos[int(index)] = old

    def _soft_allowed(index: int, x: float, y: float) -> bool:
        box = soft_region[int(index)]
        if x < box[0] - 1.0e-9 or x > box[2] + 1.0e-9:
            return False
        if y < box[1] - 1.0e-9 or y > box[3] + 1.0e-9:
            return False
        old = soft_pos[int(index)].copy()
        soft_pos[int(index)] = (float(x), float(y))
        try:
            return candidate_allowed is None or bool(candidate_allowed(hard_pos, soft_pos))
        finally:
            soft_pos[int(index)] = old

    def _accumulate(source: dict) -> None:
        stats["candidates"] += int(source.get("candidates", 0))
        stats["legal"] += int(source.get("legal", 0))
        stats["scored"] += int(source.get("scored", 0))
        stats["hierarchy_rejects"] += int(source.get("hierarchy_rejects", 0))

    for _priority, child_id, members, child_movable, soft_indices, child_soft, _graph in rows:
        remaining = _remaining()
        if remaining is not None and remaining <= 0:
            break
        if deadline is not None and time.monotonic() >= deadline:
            break
        hard_mask = np.zeros(hard_pos.shape[0], dtype=bool)
        hard_mask[child_movable] = True
        soft_mask = np.zeros(soft_pos.shape[0], dtype=bool)
        soft_mask[child_soft] = True
        hard_pool = _region_target_pool(
            hard_region,
            members,
            rows=nr,
            cols=nc,
            cw=cw,
            ch=ch,
        )
        soft_pool = _region_target_pool(
            soft_region,
            soft_indices,
            rows=nr,
            cols=nc,
            cw=cw,
            ch=ch,
        )

        if child_movable.size and hard_pool.size:
            for use_density in (False, True):
                quota = _call_quota()
                if quota is not None and quota <= 0:
                    break
                hard_pos, got, best_score = _relocation_moves(
                    hard_pos,
                    sizes,
                    hw,
                    hh,
                    cw,
                    ch,
                    hard_mask,
                    n,
                    plc,
                    benchmark,
                    incremental_scorer,
                    best_score,
                    deadline=deadline,
                    top_hot=max(1, min(int(hard_targets), child_movable.size)),
                    n_targets=max(1, int(relocation_targets)),
                    net_centroid=hard_anchor,
                    wl_blend=max(0.0, min(1.0, float(graph_anchor_blend))),
                    use_density=bool(use_density),
                    propose_all=True,
                    propose_top_m=max(1, int(relocation_targets)),
                    region_bbox=hard_region,
                    region_bias=1.0,
                    region_escape_min=float("inf"),
                    propose_accept_min_gain=max(1.0e-9, float(min_gain)),
                    target_pool=hard_pool,
                    candidate_allowed=_hard_allowed,
                    max_scored=quota,
                )
                call_stats = dict(getattr(_relocation_moves, "last_stats", {}))
                _accumulate(call_stats)
                stats["hard_accepts"] += int(got)
                accepts += int(got)

        if child_soft.size and soft_pool.size:
            for use_density in (False, True):
                quota = _call_quota()
                if quota is not None and quota <= 0:
                    break
                soft_pos, got, best_score = _soft_relocation_moves(
                    soft_pos,
                    soft_hw,
                    soft_hh,
                    cw,
                    ch,
                    n,
                    plc,
                    benchmark,
                    incremental_scorer,
                    best_score,
                    deadline=deadline,
                    top_hot=max(1, min(int(soft_targets), child_soft.size)),
                    n_targets=max(1, int(relocation_targets)),
                    soft_movable=soft_mask,
                    use_density=bool(use_density),
                    net_centroid=soft_anchor,
                    wl_blend=max(0.0, min(1.0, float(graph_anchor_blend))),
                    wl_prefilter=0.0,
                    region_bbox=soft_region,
                    region_bias=1.0,
                    region_escape_min=float("inf"),
                    accept_min_gain=max(1.0e-9, float(min_gain)),
                    target_pool=soft_pool,
                    candidate_allowed=_soft_allowed,
                    max_scored=quota,
                )
                call_stats = dict(getattr(_soft_relocation_moves, "last_stats", {}))
                _accumulate(call_stats)
                stats["soft_accepts"] += int(got)
                accepts += int(got)

        quota = _call_quota()
        if child_movable.size >= 2 and (quota is None or quota > 0):

            def _swap_quality(trial_hard: np.ndarray) -> float:
                inside = (
                    (trial_hard[assigned_hard, 0] >= hard_region[assigned_hard, 0] - 1.0e-9)
                    & (trial_hard[assigned_hard, 0] <= hard_region[assigned_hard, 2] + 1.0e-9)
                    & (trial_hard[assigned_hard, 1] >= hard_region[assigned_hard, 1] - 1.0e-9)
                    & (trial_hard[assigned_hard, 1] <= hard_region[assigned_hard, 3] + 1.0e-9)
                )
                if not bool(np.all(inside)):
                    return 1.0
                if candidate_allowed is not None and not bool(
                    candidate_allowed(trial_hard, soft_pos)
                ):
                    return 1.0
                return 0.0

            hard_priority = np.zeros(hard_pos.shape[0], dtype=np.float64)
            hard_priority[members] = float(graph_norm.get(int(child_id), 0.0))
            hard_pos, soft_pos, got, best_score, swap_stats = _region_bounded_swap_relief(
                hard_pos,
                soft_pos,
                sizes,
                hw,
                hh,
                soft_hw,
                soft_hh,
                cw,
                ch,
                hard_mask,
                soft_mask,
                benchmark,
                incremental_scorer,
                best_score,
                hard_region,
                soft_region,
                deadline=deadline,
                rounds=1,
                hard_k=max(1, int(swap_k)),
                soft_k=1,
                region_bias=1.0,
                escape_min=float("inf"),
                min_gain=max(1.0e-9, float(min_gain)),
                enable_hh=True,
                enable_hs=False,
                enable_ss=False,
                hierarchy_quality_fn=_swap_quality,
                hierarchy_quality_limit=0.0,
                hard_priority=hard_priority,
                priority_weight=max(0.0, float(graph_priority_weight)),
                graph_clusters=dict(child_clusters),
                graph_labels=np.asarray(subcluster_labels, dtype=np.int64),
                graph_edges=graph_edges,
                graph_confidence=graph_confidence,
                seed_hard_xy=seed_hard_xy,
                graph_delta_weight=max(0.0, float(graph_delta_weight)),
                max_scored=quota,
            )
            swap_scored = int(swap_stats.get("hh_scores", 0))
            stats["scored"] += swap_scored
            stats["swap_scored"] += swap_scored
            stats["swap_accepts"] += int(got)
            accepts += int(got)

    stats["accepts"] = int(accepts)
    stats["quota_exhausted"] = bool(score_limit is not None and int(stats["scored"]) >= score_limit)
    _deep_cluster_internal_relief.last_stats = stats
    return hard_pos, soft_pos, int(accepts), float(best_score)


_subcluster_relocation.last_stats = {}
_deep_cluster_internal_relief.last_stats = {}
