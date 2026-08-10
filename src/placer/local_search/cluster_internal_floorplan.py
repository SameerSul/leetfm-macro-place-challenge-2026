"""Topology-aware floorplanning inside immutable hierarchy leaves."""

from __future__ import annotations

import time
from dataclasses import dataclass
from collections.abc import Callable, Mapping, Sequence

import numpy as np

from placer.local_search.subcluster_relocation import (
    _hard_state_is_legal,
    _legalize_group_candidate,
)
from placer.scoring.wirelength import _build_wl_cache


@dataclass
class ClusterTopology:
    """Weighted internal and boundary connectivity for one hierarchy leaf."""

    members: np.ndarray
    soft_indices: np.ndarray
    internal: np.ndarray
    external_sum: np.ndarray
    external_weight: np.ndarray
    soft_hard: np.ndarray

    @property
    def internal_degree(self) -> np.ndarray:
        return np.sum(self.internal, axis=1) + np.sum(self.soft_hard, axis=0)

    @property
    def boundary_ratio(self) -> np.ndarray:
        internal = self.internal_degree
        total = internal + self.external_weight
        return np.divide(
            self.external_weight,
            total,
            out=np.zeros_like(total),
            where=total > 0.0,
        )


def _cluster_topologies(
    plc,
    hard_pos: np.ndarray,
    soft_pos: np.ndarray,
    labels: np.ndarray,
    clusters: Mapping[int, Sequence[int]],
    cluster_softs: Mapping[int, Sequence[int]],
    n: int,
    *,
    max_fanout: int = 32,
) -> dict[int, ClusterTopology]:
    """Build leaf-local graphs directly from the evaluator net topology."""
    hard_pos = np.asarray(hard_pos, dtype=np.float64)
    soft_pos = np.asarray(soft_pos, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    hard_ref = {int(module): index for index, module in enumerate(plc.hard_macro_indices[:n])}
    soft_ref = {
        int(module): index
        for index, module in enumerate(plc.soft_macro_indices[: soft_pos.shape[0]])
    }
    soft_owner: dict[int, int] = {}
    for cid, full_indices in cluster_softs.items():
        for full_index in np.asarray(full_indices, dtype=np.int64):
            soft_index = int(full_index) - int(n)
            if 0 <= soft_index < soft_pos.shape[0]:
                soft_owner[soft_index] = int(cid)

    result: dict[int, ClusterTopology] = {}
    hard_local: dict[int, tuple[int, int]] = {}
    soft_local: dict[int, tuple[int, int]] = {}
    for cid_raw, raw_members in clusters.items():
        cid = int(cid_raw)
        members = np.asarray(raw_members, dtype=np.int64).reshape(-1)
        members = members[(members >= 0) & (members < hard_pos.shape[0])]
        soft_indices = np.asarray(cluster_softs.get(cid, ()), dtype=np.int64) - int(n)
        soft_indices = soft_indices[(soft_indices >= 0) & (soft_indices < soft_pos.shape[0])]
        result[cid] = ClusterTopology(
            members=members,
            soft_indices=soft_indices,
            internal=np.zeros((members.size, members.size), dtype=np.float64),
            external_sum=np.zeros((members.size, 2), dtype=np.float64),
            external_weight=np.zeros(members.size, dtype=np.float64),
            soft_hard=np.zeros((soft_indices.size, members.size), dtype=np.float64),
        )
        for local, hard_index in enumerate(members):
            hard_local[int(hard_index)] = (cid, local)
        for local, soft_index in enumerate(soft_indices):
            soft_local[int(soft_index)] = (cid, local)

    cache = _build_wl_cache(plc)
    refs = cache["ref_idx"]
    for net_index, start_raw in enumerate(cache["net_starts"]):
        length = int(cache["net_lengths"][net_index])
        if length < 2 or length > max(2, int(max_fanout)):
            continue
        start = int(start_raw)
        endpoints = []
        seen = set()
        for ref in refs[start : start + length]:
            ref = int(ref)
            if ref in hard_ref:
                endpoint = (0, int(hard_ref[ref]))
            elif ref in soft_ref:
                endpoint = (1, int(soft_ref[ref]))
            else:
                continue
            if endpoint not in seen:
                endpoints.append(endpoint)
                seen.add(endpoint)
        if len(endpoints) < 2:
            continue
        pair_weight = float(cache["net_weights"][net_index]) / max(len(endpoints) - 1, 1)
        for left_index, left in enumerate(endpoints):
            for right in endpoints[left_index + 1 :]:
                for source, target in ((left, right), (right, left)):
                    source_kind, source_index = source
                    target_kind, target_index = target
                    if source_kind != 0 or source_index not in hard_local:
                        continue
                    cid, source_local = hard_local[source_index]
                    topology = result[cid]
                    if target_kind == 0 and target_index in hard_local:
                        target_cid, target_local = hard_local[target_index]
                        if target_cid == cid:
                            topology.internal[source_local, target_local] += pair_weight
                            continue
                        target_xy = hard_pos[target_index]
                    elif target_kind == 1:
                        if soft_owner.get(target_index) == cid and target_index in soft_local:
                            _target_cid, soft_index = soft_local[target_index]
                            topology.soft_hard[soft_index, source_local] += pair_weight
                            continue
                        target_xy = soft_pos[target_index]
                    else:
                        continue
                    topology.external_sum[source_local] += pair_weight * target_xy
                    topology.external_weight[source_local] += pair_weight
    return result


def _normalize_axis(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    lo, hi = float(np.min(values)), float(np.max(values))
    if hi - lo <= 1.0e-12:
        return np.zeros_like(values)
    return 2.0 * (values - lo) / (hi - lo) - 1.0


def _spectral_coordinates(internal: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Return deterministic two-dimensional coordinates for an induced graph."""
    count = int(internal.shape[0])
    if count <= 1 or not np.any(internal > 0.0):
        centered = current - np.mean(current, axis=0)
        return np.column_stack([_normalize_axis(centered[:, 0]), _normalize_axis(centered[:, 1])])
    weights = 0.5 * (internal + internal.T)
    laplacian = np.diag(np.sum(weights, axis=1)) - weights
    _values, vectors = np.linalg.eigh(laplacian)
    x = vectors[:, 1] if count >= 2 else np.zeros(count)
    y = vectors[:, 2] if count >= 3 else current[:, 1] - np.mean(current[:, 1])
    # Eigenvector sign is arbitrary. Anchor it to the current placement so the
    # same topology always produces the same physical orientation.
    for axis, current_axis in ((x, current[:, 0]), (y, current[:, 1])):
        if float(np.dot(axis, current_axis - np.mean(current_axis))) < 0.0:
            axis *= -1.0
    return np.column_stack([_normalize_axis(x), _normalize_axis(y)])


def _topology_targets(
    topology: ClusterTopology,
    hard_pos: np.ndarray,
    hw: np.ndarray,
    hh: np.ndarray,
    hard_region: np.ndarray,
    *,
    utilization: float,
    boundary_threshold: float,
    boundary_channel_frac: float,
    transform: str,
) -> np.ndarray:
    """Place core graph nodes internally and external-facing nodes on ports."""
    members = topology.members
    current = hard_pos[members]
    coordinates = _spectral_coordinates(topology.internal, current)
    if transform == "swap":
        coordinates = coordinates[:, ::-1]
    elif transform == "flip_x":
        coordinates[:, 0] *= -1.0
    elif transform == "flip_y":
        coordinates[:, 1] *= -1.0

    outer_lo = np.array(
        [
            float(np.min(hard_region[members, 0] - hw[members])),
            float(np.min(hard_region[members, 1] - hh[members])),
        ]
    )
    outer_hi = np.array(
        [
            float(np.max(hard_region[members, 2] + hw[members])),
            float(np.max(hard_region[members, 3] + hh[members])),
        ]
    )
    available = np.maximum(outer_hi - outer_lo, 1.0e-6)
    area = float(np.sum(4.0 * hw[members] * hh[members]))
    target_area = area / max(float(utilization), 0.20)
    current_span = np.ptp(current, axis=0) + 2.0 * np.array(
        [float(np.max(hw[members])), float(np.max(hh[members]))]
    )
    aspect = max(float(current_span[0] / max(current_span[1], 1.0e-9)), 0.25)
    width = min(float(available[0]), max(float(np.sqrt(target_area * aspect)), 1.0e-6))
    height = min(float(available[1]), max(float(target_area / max(width, 1.0e-9)), 1.0e-6))
    if height > available[1]:
        height = float(available[1])
        width = min(float(available[0]), float(target_area / max(height, 1.0e-9)))
    box_size = np.asarray([width, height], dtype=np.float64)
    center = np.clip(np.mean(current, axis=0), outer_lo + 0.5 * box_size, outer_hi - 0.5 * box_size)
    box_lo = center - 0.5 * box_size
    box_hi = center + 0.5 * box_size
    usable = np.maximum(box_hi - box_lo, 1.0e-6)
    targets = center + 0.40 * coordinates * usable

    ratios = topology.boundary_ratio
    boundary = np.flatnonzero(ratios >= float(boundary_threshold))
    for local in boundary:
        weight = float(topology.external_weight[local])
        if weight <= 0.0:
            continue
        destination = topology.external_sum[local] / weight
        direction = destination - center
        channel_x = float(boundary_channel_frac) * width * min(1.0, ratios[local])
        channel_y = float(boundary_channel_frac) * height * min(1.0, ratios[local])
        if abs(float(direction[0])) >= abs(float(direction[1])):
            targets[local, 0] = (
                box_hi[0] - hw[members[local]] - channel_x
                if direction[0] >= 0.0
                else box_lo[0] + hw[members[local]] + channel_x
            )
            targets[local, 1] = np.clip(
                destination[1],
                box_lo[1] + hh[members[local]],
                box_hi[1] - hh[members[local]],
            )
        else:
            targets[local, 1] = (
                box_hi[1] - hh[members[local]] - channel_y
                if direction[1] >= 0.0
                else box_lo[1] + hh[members[local]] + channel_y
            )
            targets[local, 0] = np.clip(
                destination[0],
                box_lo[0] + hw[members[local]],
                box_hi[0] - hw[members[local]],
            )

    targets[:, 0] = np.clip(targets[:, 0], hard_region[members, 0], hard_region[members, 2])
    targets[:, 1] = np.clip(targets[:, 1], hard_region[members, 1], hard_region[members, 3])
    return targets


def _owned_soft_targets(
    topology: ClusterTopology,
    old_hard: np.ndarray,
    new_hard: np.ndarray,
    soft_pos: np.ndarray,
    soft_region: np.ndarray,
) -> np.ndarray:
    """Place owned soft macros at hard-affinity barycentres inside the island."""
    if topology.soft_indices.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    old_center = np.mean(old_hard, axis=0)
    new_center = np.mean(new_hard, axis=0)
    targets = soft_pos[topology.soft_indices] + (new_center - old_center)
    for soft_local, weights in enumerate(topology.soft_hard):
        total = float(np.sum(weights))
        if total > 0.0:
            barycenter = np.sum(weights[:, None] * new_hard, axis=0) / total
            targets[soft_local] = 0.80 * barycenter + 0.20 * targets[soft_local]
    targets[:, 0] = np.clip(
        targets[:, 0],
        soft_region[topology.soft_indices, 0],
        soft_region[topology.soft_indices, 2],
    )
    targets[:, 1] = np.clip(
        targets[:, 1],
        soft_region[topology.soft_indices, 1],
        soft_region[topology.soft_indices, 3],
    )
    return targets


def _internal_topology_cost(topology: ClusterTopology, hard_targets: np.ndarray) -> float:
    """Return weighted internal Manhattan distance for deterministic ranking."""
    delta = np.abs(hard_targets[:, None, :] - hard_targets[None, :, :])
    return 0.5 * float(np.sum(topology.internal * np.sum(delta, axis=2)))


def _topology_aware_cluster_floorplan(
    plc,
    hard_pos: np.ndarray,
    soft_pos: np.ndarray,
    hw: np.ndarray,
    hh: np.ndarray,
    cw: float,
    ch: float,
    n: int,
    incremental_scorer,
    initial_score: float,
    *,
    labels: np.ndarray,
    clusters: Mapping[int, Sequence[int]],
    cluster_softs: Mapping[int, Sequence[int]],
    movable_h: np.ndarray,
    movable_soft: np.ndarray,
    hard_region: np.ndarray,
    soft_region: np.ndarray,
    candidate_allowed: Callable[[np.ndarray, np.ndarray], bool] | None = None,
    deadline: float | None = None,
    min_hard: int = 2,
    max_hard: int = 16,
    max_soft: int = 32,
    top_clusters: int = 8,
    max_fanout: int = 32,
    utilization_variants: Sequence[float] = (0.65, 0.55, 0.45),
    transforms: Sequence[str] = ("identity", "swap", "flip_x", "flip_y"),
    boundary_threshold: float = 0.35,
    boundary_channel_frac: float = 0.04,
    min_proxy_gain: float = 1.0e-5,
    max_scored: int | None = 96,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Exact-gate topology-aware arrangements of selected hierarchy leaves."""
    score_limit = None if max_scored is None else max(0, int(max_scored))
    stats = {
        "eligible_clusters": 0,
        "selected_clusters": 0,
        "candidates": 0,
        "legal": 0,
        "hierarchy_rejects": 0,
        "scored": 0,
        "accepts": 0,
        "boundary_macros": 0,
        "legalized_candidates": 0,
        "best_proxy_gain": 0.0,
        "score_limit": score_limit,
        "quota_exhausted": bool(score_limit is not None and score_limit <= 0),
    }
    _topology_aware_cluster_floorplan.last_stats = stats
    if not clusters or score_limit == 0:
        return hard_pos, soft_pos, 0, float(initial_score)

    topologies = _cluster_topologies(
        plc,
        hard_pos,
        soft_pos,
        labels,
        clusters,
        cluster_softs,
        n,
        max_fanout=max_fanout,
    )
    movable_h = np.asarray(movable_h, dtype=bool)
    movable_soft = np.asarray(movable_soft, dtype=bool)
    rows = []
    for cid, topology in topologies.items():
        members = topology.members
        soft_indices = topology.soft_indices
        if (
            members.size < int(min_hard)
            or members.size > int(max_hard)
            or soft_indices.size > int(max_soft)
            or not bool(np.all(movable_h[members]))
            or (soft_indices.size and not bool(np.all(movable_soft[soft_indices])))
        ):
            continue
        demand = float(np.sum(topology.internal) * 0.5 + np.sum(topology.external_weight))
        if demand <= 0.0:
            continue
        current_cost = _internal_topology_cost(topology, hard_pos[members])
        boundary_count = int(np.count_nonzero(topology.boundary_ratio >= boundary_threshold))
        priority = demand + current_cost + boundary_count * demand / max(members.size, 1)
        rows.append((-priority, int(cid), topology))
    rows.sort(key=lambda row: (row[0], row[1]))
    stats["eligible_clusters"] = len(rows)
    rows = rows[: max(0, int(top_clusters))]
    stats["selected_clusters"] = len(rows)

    current_score = float(initial_score)
    accepts = 0
    for _priority, _cid, topology in rows:
        if score_limit is not None and int(stats["scored"]) >= score_limit:
            break
        if deadline is not None and time.monotonic() >= deadline:
            break
        members = topology.members
        soft_indices = topology.soft_indices
        stats["boundary_macros"] += int(
            np.count_nonzero(topology.boundary_ratio >= boundary_threshold)
        )
        best = None
        best_key = None
        for utilization in utilization_variants:
            for transform in transforms:
                if score_limit is not None and int(stats["scored"]) >= score_limit:
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    break
                stats["candidates"] += 1
                targets = _topology_targets(
                    topology,
                    hard_pos,
                    hw,
                    hh,
                    hard_region,
                    utilization=float(utilization),
                    boundary_threshold=float(boundary_threshold),
                    boundary_channel_frac=float(boundary_channel_frac),
                    transform=str(transform),
                )
                trial_hard = hard_pos.copy()
                trial_hard[members] = targets
                if not _hard_state_is_legal(trial_hard, hw, hh):
                    legalized = _legalize_group_candidate(
                        hard_pos,
                        members,
                        targets,
                        hard_region,
                        hw,
                        hh,
                        cw,
                        ch,
                        deadline,
                    )
                    if legalized is None:
                        continue
                    targets = legalized
                    trial_hard[members] = targets
                    stats["legalized_candidates"] += 1
                soft_targets = _owned_soft_targets(
                    topology,
                    hard_pos[members],
                    targets,
                    soft_pos,
                    soft_region,
                )
                trial_soft = soft_pos.copy()
                trial_soft[soft_indices] = soft_targets
                if candidate_allowed is not None and not bool(
                    candidate_allowed(trial_hard, trial_soft)
                ):
                    stats["hierarchy_rejects"] += 1
                    continue
                stats["legal"] += 1
                score = float(
                    incremental_scorer.score_move_group(
                        members,
                        targets,
                        soft_indices,
                        soft_targets,
                    )
                )
                stats["scored"] += 1
                gain = current_score - score
                stats["best_proxy_gain"] = max(float(stats["best_proxy_gain"]), gain)
                if gain < float(min_proxy_gain):
                    continue
                key = (
                    score,
                    _internal_topology_cost(topology, targets),
                    -float(utilization),
                    str(transform),
                )
                if best_key is None or key < best_key:
                    best_key = key
                    best = (targets.copy(), soft_targets.copy(), score)
        if best is None:
            continue
        targets, soft_targets, current_score = best
        incremental_scorer.commit_move_group(members, targets, soft_indices, soft_targets)
        hard_pos[members] = targets
        soft_pos[soft_indices] = soft_targets
        accepts += 1

    stats["accepts"] = int(accepts)
    stats["quota_exhausted"] = bool(score_limit is not None and int(stats["scored"]) >= score_limit)
    _topology_aware_cluster_floorplan.last_stats = stats
    return hard_pos, soft_pos, int(accepts), float(current_score)


_topology_aware_cluster_floorplan.last_stats = {}
