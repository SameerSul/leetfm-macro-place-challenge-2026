"""One-level parent-bounded relocation of hierarchy child clusters."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence

import numpy as np

from placer.legalize.spiral import _will_legalize
from placer.local_search.fields import (
    cold_connected_component_target_pool,
    weighted_congestion_field,
)


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
    cell_w, cell_h = float(cw) / nc, float(ch) / nr

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
            rows.append(
                {
                    "parent": int(parent_id),
                    "child": child_id,
                    "members": members,
                    "soft_indices": soft_indices,
                    "heat": heat,
                }
            )
    rows.sort(key=lambda row: (-float(row["heat"]), int(row["parent"]), int(row["child"])))
    all_rows = rows
    rows = all_rows[: max(1, int(top_children))]
    stats["eligible_children"] = int(len(all_rows))
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
                contrast = abs(float(left["heat"]) - float(right["heat"]))
                swap_pairs.append(
                    (
                        -contrast,
                        -max(float(left["heat"]), float(right["heat"])),
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


_subcluster_relocation.last_stats = {}
