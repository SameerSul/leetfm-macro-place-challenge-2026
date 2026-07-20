"""Compound hierarchy-bounded relocation for related soft macros."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence

import numpy as np

from placer.local_search.fields import (
    cold_connected_component_target_pool,
    weighted_congestion_field,
)


def _related_soft_groups(
    soft_pos: np.ndarray,
    local_heat: np.ndarray,
    cluster_softs: Mapping[int, np.ndarray],
    bridge_softs: Mapping[int, np.ndarray] | None,
    soft_movable: np.ndarray,
    n: int,
    group_size: int,
    soft_bundles: Sequence[object] | None = None,
    soft_role_evidence: Mapping[int, Mapping[str, object]] | None = None,
) -> list[dict[str, object]]:
    """Build one compact hot soft group per owner or bridge signature."""
    raw_groups: list[tuple[str, object, np.ndarray]] = []
    explicit_members: set[int] = set()
    for bundle in soft_bundles or ():
        members = np.asarray(getattr(bundle, "members", ()), dtype=np.int64)
        if members.size < 2:
            continue
        source = str(getattr(bundle, "source", "bundle"))
        key = str(getattr(bundle, "key", ""))
        raw_groups.append((source, key, members))
        explicit_members.update(int(member) for member in members)

    for cid, members in sorted(cluster_softs.items()):
        indices = np.asarray(members, dtype=np.int64) - int(n)
        if soft_role_evidence is not None:
            indices = np.asarray(
                [
                    index
                    for index in indices
                    if str(soft_role_evidence.get(int(index), {}).get("confidence", "low"))
                    == "high"
                ],
                dtype=np.int64,
            )
        if explicit_members:
            indices = indices[~np.isin(indices, np.fromiter(explicit_members, dtype=np.int64))]
        raw_groups.append(("owned", int(cid), indices))

    bridge_by_signature: dict[tuple[int, ...], list[int]] = {}
    for soft_k, cids in (bridge_softs or {}).items():
        signature = tuple(sorted(int(cid) for cid in np.asarray(cids).reshape(-1)))
        if signature:
            bridge_by_signature.setdefault(signature, []).append(int(soft_k))
    for signature, members in sorted(bridge_by_signature.items()):
        indices = np.asarray(members, dtype=np.int64)
        if soft_role_evidence is not None:
            indices = np.asarray(
                [
                    index
                    for index in indices
                    if str(soft_role_evidence.get(int(index), {}).get("confidence", "low"))
                    == "high"
                ],
                dtype=np.int64,
            )
        if explicit_members:
            indices = indices[~np.isin(indices, np.fromiter(explicit_members, dtype=np.int64))]
        raw_groups.append(("bridge", signature, indices))

    groups = []
    max_group = max(2, int(group_size))
    for role, relation, indices in raw_groups:
        indices = np.asarray(indices, dtype=np.int64)
        valid = (indices >= 0) & (indices < soft_pos.shape[0])
        indices = indices[valid]
        indices = indices[soft_movable[indices]]
        if indices.size < 2:
            continue

        hottest = int(indices[np.argmax(local_heat[indices])])
        distance2 = np.sum((soft_pos[indices] - soft_pos[hottest]) ** 2, axis=1)
        order = np.lexsort((indices, distance2))
        selected = np.asarray(indices[order[:max_group]], dtype=np.int64)
        groups.append(
            {
                "role": role,
                "relation": relation,
                "indices": selected,
                "heat": float(np.mean(local_heat[selected])),
            }
        )
    groups.sort(key=lambda row: (-float(row["heat"]), str(row["role"]), str(row["relation"])))
    return groups


def _compound_soft_relocation(
    soft_pos: np.ndarray,
    soft_hw: np.ndarray,
    soft_hh: np.ndarray,
    cw: float,
    ch: float,
    n: int,
    benchmark,
    incremental_scorer,
    initial_score: float,
    *,
    cluster_softs: Mapping[int, np.ndarray],
    bridge_softs: Mapping[int, np.ndarray] | None = None,
    soft_bundles: Sequence[object] | None = None,
    soft_role_evidence: Mapping[int, Mapping[str, object]] | None = None,
    soft_movable: np.ndarray | None = None,
    region_bbox: np.ndarray | None = None,
    candidate_allowed: Callable[[np.ndarray], bool] | None = None,
    deadline: float | None = None,
    top_groups: int = 4,
    group_size: int = 6,
    cold_percentile: float = 35.0,
    max_components: int = 4,
    min_component_cells: int = 4,
    n_anchors: int = 2,
    shift_fractions: Sequence[float] = (0.5, 1.0),
    min_field_drop: float = 0.02,
    min_gain: float = 0.00005,
    max_scored: int | None = None,
) -> tuple[np.ndarray, int, float]:
    """Co-move a related soft group and exact-score only the completed move."""
    stats = {
        "groups": 0,
        "candidates": 0,
        "hierarchy_rejects": 0,
        "field_rejects": 0,
        "scored": 0,
        "accepts": 0,
        "best_candidate_gain": 0.0,
        "score_limit": None if max_scored is None else max(0, int(max_scored)),
        "quota_exhausted": bool(max_scored is not None and int(max_scored) <= 0),
    }
    _compound_soft_relocation.last_stats = stats
    num_soft = int(soft_pos.shape[0])
    if num_soft < 2 or region_bbox is None:
        return soft_pos, 0, float(initial_score)

    movable = (
        np.ones(num_soft, dtype=bool)
        if soft_movable is None
        else np.asarray(soft_movable, dtype=bool)
    )
    if movable.size != num_soft or np.count_nonzero(movable) < 2:
        return soft_pos, 0, float(initial_score)

    nr, nc = int(benchmark.grid_rows), int(benchmark.grid_cols)
    field = weighted_congestion_field(incremental_scorer, nr, nc)
    if field is None or field.size == 0:
        return soft_pos, 0, float(initial_score)
    field = np.asarray(field, dtype=np.float64)
    cell_w, cell_h = float(cw) / nc, float(ch) / nr
    ci = np.clip((soft_pos[:, 0] / cell_w).astype(np.int64), 0, nc - 1)
    ri = np.clip((soft_pos[:, 1] / cell_h).astype(np.int64), 0, nr - 1)
    local_heat = field[ri, ci]

    groups = _related_soft_groups(
        soft_pos,
        local_heat,
        cluster_softs,
        bridge_softs,
        movable,
        n,
        group_size,
        soft_bundles=soft_bundles,
        soft_role_evidence=soft_role_evidence,
    )[: max(1, int(top_groups))]
    stats["groups"] = int(len(groups))
    if not groups:
        return soft_pos, 0, float(initial_score)

    target_pool = cold_connected_component_target_pool(
        field,
        cold_percentile=float(cold_percentile),
        max_components=max(1, int(max_components)),
        min_cells=max(1, int(min_component_cells)),
        size_weight=0.35,
    )
    pool = np.asarray(target_pool["indices"], dtype=np.int64)
    if pool.size == 0:
        return soft_pos, 0, float(initial_score)
    pool_x = ((pool % nc).astype(np.float64) + 0.5) * cell_w
    pool_y = ((pool // nc).astype(np.float64) + 0.5) * cell_h
    pool_field = field.ravel()[pool]

    best_score = float(initial_score)
    best_indices = None
    best_targets = None
    canvas2 = max(float(cw) ** 2 + float(ch) ** 2, 1.0)
    fractions = sorted({float(value) for value in shift_fractions if 0.0 < float(value) <= 1.0})
    score_limit = None if max_scored is None else max(0, int(max_scored))

    for group in groups:
        if score_limit is not None and int(stats["scored"]) >= score_limit:
            break
        if deadline is not None and time.monotonic() >= deadline:
            break
        relation_indices = np.asarray(group["indices"], dtype=np.int64)
        group_sizes = sorted({2, min(4, relation_indices.size), relation_indices.size})
        for active_size in group_sizes:
            indices = relation_indices[:active_size]
            old_xy = soft_pos[indices]
            source_heat = float(np.mean(local_heat[indices]))
            centroid = np.mean(old_xy, axis=0)
            dx_lo = float(np.max(region_bbox[indices, 0] - old_xy[:, 0]))
            dx_hi = float(np.min(region_bbox[indices, 2] - old_xy[:, 0]))
            dy_lo = float(np.max(region_bbox[indices, 1] - old_xy[:, 1]))
            dy_hi = float(np.min(region_bbox[indices, 3] - old_xy[:, 1]))
            feasible = (
                (pool_x >= centroid[0] + dx_lo - 1e-9)
                & (pool_x <= centroid[0] + dx_hi + 1e-9)
                & (pool_y >= centroid[1] + dy_lo - 1e-9)
                & (pool_y <= centroid[1] + dy_hi + 1e-9)
            )
            possible = np.flatnonzero(feasible)
            if possible.size == 0:
                continue
            distance2 = (pool_x[possible] - centroid[0]) ** 2 + (
                pool_y[possible] - centroid[1]
            ) ** 2
            rank = pool_field[possible] + 0.05 * distance2 / canvas2
            anchors = possible[np.argsort(rank, kind="stable")[: max(1, int(n_anchors))]]

            for anchor in anchors:
                delta = np.array(
                    [pool_x[int(anchor)] - centroid[0], pool_y[int(anchor)] - centroid[1]],
                    dtype=np.float64,
                )
                for fraction in fractions:
                    if score_limit is not None and int(stats["scored"]) >= score_limit:
                        break
                    if deadline is not None and time.monotonic() >= deadline:
                        break
                    shift = fraction * delta
                    if (
                        abs(float(shift[0])) < 0.25 * cell_w
                        and abs(float(shift[1])) < 0.25 * cell_h
                    ):
                        continue
                    new_xy = old_xy + shift
                    new_xy[:, 0] = np.clip(new_xy[:, 0], soft_hw[indices], cw - soft_hw[indices])
                    new_xy[:, 1] = np.clip(new_xy[:, 1], soft_hh[indices], ch - soft_hh[indices])
                    inside = (
                        (new_xy[:, 0] >= region_bbox[indices, 0] - 1e-9)
                        & (new_xy[:, 0] <= region_bbox[indices, 2] + 1e-9)
                        & (new_xy[:, 1] >= region_bbox[indices, 1] - 1e-9)
                        & (new_xy[:, 1] <= region_bbox[indices, 3] + 1e-9)
                    )
                    if not bool(np.all(inside)):
                        continue
                    stats["candidates"] += 1

                    new_ci = np.clip((new_xy[:, 0] / cell_w).astype(np.int64), 0, nc - 1)
                    new_ri = np.clip((new_xy[:, 1] / cell_h).astype(np.int64), 0, nr - 1)
                    new_heat = float(np.mean(field[new_ri, new_ci]))
                    if new_heat >= source_heat - float(min_field_drop):
                        stats["field_rejects"] += 1
                        continue

                    trial_soft = soft_pos.copy()
                    trial_soft[indices] = new_xy
                    if candidate_allowed is not None and not bool(candidate_allowed(trial_soft)):
                        stats["hierarchy_rejects"] += 1
                        continue

                    score = float(incremental_scorer.score_move_soft_group(indices, new_xy))
                    stats["scored"] += 1
                    stats["best_candidate_gain"] = max(
                        float(stats["best_candidate_gain"]),
                        float(initial_score) - float(score),
                    )
                    if score < best_score - max(1e-9, float(min_gain)):
                        best_score = float(score)
                        best_indices = indices.copy()
                        best_targets = new_xy.copy()

    if best_indices is not None and best_targets is not None:
        incremental_scorer.commit_move_soft_group(best_indices, best_targets)
        soft_pos[best_indices] = best_targets
        stats["accepts"] = 1

    stats["quota_exhausted"] = bool(score_limit is not None and int(stats["scored"]) >= score_limit)

    _compound_soft_relocation.last_stats = stats
    return soft_pos, int(stats["accepts"]), float(best_score)


_compound_soft_relocation.last_stats = {}
