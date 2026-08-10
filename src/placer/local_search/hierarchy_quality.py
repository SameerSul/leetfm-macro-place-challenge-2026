"""Deterministic hierarchy-quality diagnostics for complete placements."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from utils import constants as const
from utils.config import HAS_NUMBA, _numba_njit
from placer.local_search.cluster_decompress import hierarchy_quality_breakdown

HIERARCHY_VECTOR_METRICS = (
    "cluster_compactness",
    "worst_cluster_spread",
    "neighbor_impurity",
    "edge_stretch",
    "owned_soft_distance",
    "bridge_soft_distance",
)

HIERARCHY_ISLAND_METRICS = (
    "spread",
    "bbox_span",
    "neighbor_impurity",
)


def hierarchy_island_metrics(
    hard_xy: np.ndarray,
    soft_xy: np.ndarray,
    clusters: Mapping[int, Sequence[int]],
    cluster_softs: Mapping[int, Sequence[int]] | None,
    hard_sizes: np.ndarray,
    canvas_width: float,
    canvas_height: float,
    cluster_ids: Sequence[int] | None = None,
) -> dict[int, dict[str, float]]:
    """Return independent spatial-cohesion metrics for every hierarchy leaf.

    Unlike :func:`hierarchy_quality_vector`, these values are never averaged
    across colours.  That prevents compact leaves from hiding one scattered or
    interleaved leaf in the complete-placement contract.
    """
    hard = np.asarray(hard_xy, dtype=np.float64)
    soft = np.asarray(soft_xy, dtype=np.float64)
    sizes = np.asarray(hard_sizes, dtype=np.float64)
    diag = max(float(np.hypot(canvas_width, canvas_height)), 1.0e-12)
    labels = np.full(hard.shape[0], -1, dtype=np.int64)
    valid: dict[int, np.ndarray] = {}
    for cid_raw, members_raw in clusters.items():
        members = np.asarray(members_raw, dtype=np.int64).reshape(-1)
        members = members[(members >= 0) & (members < hard.shape[0])]
        if members.size:
            cid = int(cid_raw)
            valid[cid] = members
            labels[members] = cid

    clustered = np.flatnonzero(labels >= 0)
    result: dict[int, dict[str, float]] = {}
    n_hard = hard.shape[0]
    selected = None if cluster_ids is None else {int(cid) for cid in cluster_ids}
    for cid, members in valid.items():
        if selected is not None and cid not in selected:
            continue
        points = hard[members]
        center = np.mean(points, axis=0)
        distances = np.linalg.norm(points - center, axis=1)
        spread = float(np.max(distances)) / diag if distances.size else 0.0

        lo = np.min(points - 0.5 * sizes[members], axis=0)
        hi = np.max(points + 0.5 * sizes[members], axis=0)
        bbox_span = float(np.linalg.norm(hi - lo)) / diag

        impurity = 0.0
        if members.size > 1 and clustered.size > 1:
            mismatch = []
            for macro in members:
                delta = hard[clustered] - hard[int(macro)]
                distance2 = np.sum(delta * delta, axis=1)
                distance2[clustered == int(macro)] = np.inf
                k = min(4, int(members.size) - 1, int(clustered.size) - 1)
                nearest = np.argsort(distance2, kind="stable")[:k]
                mismatch.append(float(np.mean(labels[clustered[nearest]] != cid)))
            impurity = float(np.max(mismatch)) if mismatch else 0.0

        owned_distances = []
        for full_index in np.asarray((cluster_softs or {}).get(cid, ()), dtype=np.int64):
            soft_index = int(full_index) - n_hard
            if 0 <= soft_index < soft.shape[0]:
                owned_distances.append(float(np.linalg.norm(soft[soft_index] - center)) / diag)
        owned_soft_p90 = (
            float(np.percentile(np.asarray(owned_distances), 90.0))
            if owned_distances
            else 0.0
        )

        # Same-colour rectangles form one component when their edge-to-edge
        # gaps are within a small routing-channel allowance.  This is a spatial
        # island test, not a netlist connectivity test.
        components = int(members.size)
        if members.size > 1:
            parent = np.arange(members.size, dtype=np.int64)

            def find(index: int) -> int:
                while int(parent[index]) != index:
                    parent[index] = parent[int(parent[index])]
                    index = int(parent[index])
                return index

            gap_allowance = 0.01 * diag
            for left in range(members.size):
                for right in range(left + 1, members.size):
                    a, b = int(members[left]), int(members[right])
                    gap = np.maximum(
                        np.abs(hard[a] - hard[b]) - 0.5 * (sizes[a] + sizes[b]),
                        0.0,
                    )
                    if float(np.linalg.norm(gap)) <= gap_allowance:
                        ra, rb = find(left), find(right)
                        if ra != rb:
                            parent[rb] = ra
            components = len({find(index) for index in range(members.size)})
        fragmentation = float(max(0, components - 1))

        foreign = np.flatnonzero((labels >= 0) & (labels != cid))
        if foreign.size:
            inside = np.all((hard[foreign] >= lo) & (hard[foreign] <= hi), axis=1)
            foreign_intrusion = float(np.count_nonzero(inside))
        else:
            foreign_intrusion = 0.0

        result[cid] = {
            "spread": spread,
            "bbox_span": bbox_span,
            "neighbor_impurity": impurity,
            "owned_soft_p90": owned_soft_p90,
            "fragmentation": fragmentation,
            "foreign_intrusion": foreign_intrusion,
        }
    return result


def hierarchy_island_limits(
    reference: Mapping[int, Mapping[str, float]],
    cluster_confidence: Mapping[int, float] | None,
    cluster_source: str,
    *,
    strict_confidence: float = 0.65,
    strict_relative_slack: float = 0.05,
    medium_relative_slack: float = 0.10,
    distance_absolute_slack: float = 0.001,
    impurity_absolute_slack: float = 0.25,
) -> dict[int, dict[str, float]]:
    """Build confidence-calibrated, per-colour upper limits.

    Explicit hierarchy tags and high-confidence inferred leaves are immutable
    islands.  Medium-confidence leaves retain modest geometric slack, while
    low-confidence evidence remains advisory and receives no hard limit.
    """
    explicit = str(cluster_source) == "hierarchy_path_tags"
    confidence = cluster_confidence or {}
    limits: dict[int, dict[str, float]] = {}
    for cid_raw, metrics in reference.items():
        cid = int(cid_raw)
        score = float(confidence.get(cid, 1.0 if explicit else 0.0))
        strict = bool(explicit or score >= float(strict_confidence))
        medium = bool(strict or score >= 0.5 * float(strict_confidence))
        if not medium:
            continue
        rel = (
            max(0.0, float(strict_relative_slack))
            if strict
            else max(0.0, float(medium_relative_slack))
        )
        row = {"tier": 2.0 if strict else 1.0}
        for key in HIERARCHY_ISLAND_METRICS:
            value = float(metrics.get(key, 0.0))
            if key in {"fragmentation", "foreign_intrusion"}:
                row[key] = value
            else:
                absolute = (
                    float(impurity_absolute_slack)
                    if key == "neighbor_impurity"
                    else float(distance_absolute_slack)
                )
                row[key] = value + max(absolute, abs(value) * rel)
        limits[cid] = row
    return limits


def hierarchy_island_contract(
    candidate: Mapping[int, Mapping[str, float]],
    limits: Mapping[int, Mapping[str, float]],
    *,
    tolerance: float = 1.0e-12,
) -> tuple[bool, dict[str, float]]:
    """Check every protected colour independently against its island limits."""
    violations: dict[str, float] = {}
    for cid_raw, row_limits in limits.items():
        cid = int(cid_raw)
        row = candidate.get(cid, {})
        for key in HIERARCHY_ISLAND_METRICS:
            excess = float(row.get(key, np.inf)) - float(row_limits[key])
            if excess > float(tolerance):
                violations[f"island_{cid}_{key}"] = excess
    return not violations, violations


def hierarchy_coverage_scope(vector: Mapping[str, float]) -> str:
    """Classify hierarchy evidence coverage without changing acceptance gates."""
    hard_fraction = float(vector.get("clustered_hard_fraction", 0.0))
    soft_fraction = float(vector.get("soft_coverage", 0.0))
    if hard_fraction >= 0.75 and soft_fraction >= 0.25:
        return "high"
    if hard_fraction >= 0.25 and soft_fraction >= 0.10:
        return "partial"
    return "low"


def hierarchy_vector_limits(
    reference: Mapping[str, float],
    absolute_slack: Mapping[str, float],
    relative_slack: float,
) -> dict[str, float]:
    """Build per-component upper limits from a reference hierarchy vector."""
    rel = max(0.0, float(relative_slack))
    limits = {}
    for key in HIERARCHY_VECTOR_METRICS:
        value = float(reference.get(key, 0.0))
        slack = max(0.0, float(absolute_slack.get(key, 0.0)), abs(value) * rel)
        limits[key] = value + slack
    return limits


def hierarchy_vector_contract(
    candidate: Mapping[str, float],
    limits: Mapping[str, float],
    *,
    tolerance: float = 1.0e-12,
) -> tuple[bool, dict[str, float]]:
    """Check every hierarchy component against its independent upper limit."""
    violations = {}
    for key in HIERARCHY_VECTOR_METRICS:
        excess = float(candidate.get(key, 0.0)) - float(limits[key])
        if excess > float(tolerance):
            violations[key] = excess
    return not violations, violations


def hierarchy_vector_margins(
    candidate: Mapping[str, float],
    limits: Mapping[str, float],
) -> dict[str, float]:
    """Return signed per-component headroom; negative values are violations."""
    return {
        key: float(limits[key]) - float(candidate.get(key, 0.0)) for key in HIERARCHY_VECTOR_METRICS
    }


def _edge_values(edge) -> tuple[int, int, float]:
    if isinstance(edge, Mapping):
        return int(edge["src"]), int(edge["dst"]), float(edge.get("weight", 1.0))
    if hasattr(edge, "src"):
        return int(edge.src), int(edge.dst), float(edge.weight)
    src, dst, *rest = edge
    return int(src), int(dst), float(rest[0] if rest else 1.0)


def _point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    delta = end - start
    denom = float(np.dot(delta, delta))
    if denom <= 1.0e-18:
        return float(np.linalg.norm(point - start))
    t = float(np.clip(np.dot(point - start, delta) / denom, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + t * delta)))


def _neighbor_impurity_reference(
    hard: np.ndarray,
    clustered: np.ndarray,
    labels: np.ndarray,
    own_sizes: np.ndarray,
) -> float:
    """Return the original stable-sort neighbor impurity calculation."""
    if clustered.size <= 1:
        return 0.0
    points = hard[clustered]
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    impurity_total = 0.0
    for row, macro in enumerate(clustered):
        k = min(4, max(1, int(own_sizes[macro]) - 1), clustered.size - 1)
        nearest = np.argsort(distances[row], kind="stable")[:k]
        impurity_total += float(np.mean(labels[clustered[nearest]] != labels[macro]))
    return impurity_total / clustered.size


if HAS_NUMBA:

    @_numba_njit(cache=True, fastmath=False)
    def _neighbor_impurity_jit(hard, clustered, labels, own_sizes):
        """Compute stable nearest-four impurity without an N-by-N sort."""
        n_clustered = clustered.size
        if n_clustered <= 1:
            return 0.0

        best_distances = np.empty(4, dtype=np.float64)
        best_rows = np.empty(4, dtype=np.int64)
        impurity_total = 0.0
        for row in range(n_clustered):
            macro = clustered[row]
            own_size = own_sizes[macro]
            k = min(4, max(1, own_size - 1), n_clustered - 1)
            for rank in range(k):
                best_distances[rank] = np.inf
                best_rows[rank] = n_clustered

            x = hard[macro, 0]
            y = hard[macro, 1]
            for candidate_row in range(n_clustered):
                if candidate_row == row:
                    continue
                candidate = clustered[candidate_row]
                dx = x - hard[candidate, 0]
                dy = y - hard[candidate, 1]
                # Squared distance has the same ordering as the prior Euclidean norm.
                distance_squared = dx * dx + dy * dy
                for rank in range(k):
                    if distance_squared < best_distances[rank] or (
                        distance_squared == best_distances[rank] and candidate_row < best_rows[rank]
                    ):
                        for shift in range(k - 1, rank, -1):
                            best_distances[shift] = best_distances[shift - 1]
                            best_rows[shift] = best_rows[shift - 1]
                        best_distances[rank] = distance_squared
                        best_rows[rank] = candidate_row
                        break

            mismatch_count = 0
            for rank in range(k):
                if labels[clustered[best_rows[rank]]] != labels[macro]:
                    mismatch_count += 1
            impurity_total += mismatch_count / k
        return impurity_total / n_clustered


def _neighbor_impurity(
    hard: np.ndarray,
    clustered: np.ndarray,
    labels: np.ndarray,
    own_sizes: np.ndarray,
) -> float:
    """Return stable nearest-four cluster impurity with a cached JIT kernel."""
    hard = np.ascontiguousarray(hard, dtype=np.float64)
    clustered = np.ascontiguousarray(clustered, dtype=np.int64)
    labels = np.ascontiguousarray(labels, dtype=np.int64)
    own_sizes = np.ascontiguousarray(own_sizes, dtype=np.int64)
    if HAS_NUMBA:
        return float(_neighbor_impurity_jit(hard, clustered, labels, own_sizes))
    return _neighbor_impurity_reference(hard, clustered, labels, own_sizes)


def hierarchy_quality_vector(
    hard_xy: np.ndarray,
    soft_xy: np.ndarray,
    clusters: Mapping[int, Sequence[int]],
    cluster_softs: Mapping[int, Sequence[int]] | None,
    bridge_softs: Mapping[int, Sequence[int]] | None,
    edges: Sequence | None,
    canvas_width: float,
    canvas_height: float,
) -> dict[str, float]:
    """Measure hard compactness, purity, graph stretch, and soft-role fidelity.

    ``cluster_softs`` stores full-placement indices while ``bridge_softs`` stores
    indices local to the soft array, matching :class:`HierarchyModel`.
    All new distance terms are normalized by the canvas diagonal. Lower is
    better for every penalty and for the weighted composite.
    """
    hard = np.asarray(hard_xy, dtype=np.float64)
    soft = np.asarray(soft_xy, dtype=np.float64)
    diag = max(float(np.hypot(canvas_width, canvas_height)), 1.0e-12)
    valid: dict[int, np.ndarray] = {}
    centroids: dict[int, np.ndarray] = {}
    spreads: list[float] = []
    labels = np.full(hard.shape[0], -1, dtype=np.int64)
    own_sizes = np.zeros(hard.shape[0], dtype=np.int64)
    for cid_raw, members_raw in clusters.items():
        cid = int(cid_raw)
        members = np.asarray(members_raw, dtype=np.int64)
        members = members[(members >= 0) & (members < hard.shape[0])]
        if members.size == 0:
            continue
        valid[cid] = members
        labels[members] = cid
        own_sizes[members] = members.size
        center = np.mean(hard[members], axis=0)
        centroids[cid] = center
        spreads.append(float(np.mean(np.linalg.norm(hard[members] - center, axis=1))) / diag)

    compactness = float(np.mean(spreads)) if spreads else 0.0
    worst_spread = float(np.max(spreads)) if spreads else 0.0

    clustered = np.flatnonzero(labels >= 0)
    neighbor_impurity = _neighbor_impurity(hard, clustered, labels, own_sizes)

    edge_total = 0.0
    edge_weight = 0.0
    for edge in edges or ():
        src, dst, weight = _edge_values(edge)
        if src not in centroids or dst not in centroids or weight <= 0.0:
            continue
        edge_total += weight * float(np.linalg.norm(centroids[src] - centroids[dst])) / diag
        edge_weight += weight
    edge_stretch = edge_total / edge_weight if edge_weight > 0.0 else 0.0

    owned_terms: list[float] = []
    owned_soft_indices: list[int] = []
    n_hard = hard.shape[0]
    for cid_raw, full_indices in (cluster_softs or {}).items():
        cid = int(cid_raw)
        if cid not in centroids:
            continue
        for full_index in np.asarray(full_indices, dtype=np.int64).reshape(-1):
            soft_index = int(full_index) - n_hard
            if 0 <= soft_index < soft.shape[0]:
                owned_terms.append(float(np.linalg.norm(soft[soft_index] - centroids[cid])) / diag)
                owned_soft_indices.append(soft_index)
    owned_soft_distance = float(np.mean(owned_terms)) if owned_terms else 0.0
    owned_soft_unique = int(len(np.unique(np.asarray(owned_soft_indices, dtype=np.int64))))

    bridge_terms: list[float] = []
    bridge_soft_indices: list[int] = []
    for soft_index_raw, cids_raw in (bridge_softs or {}).items():
        soft_index = int(soft_index_raw)
        if not (0 <= soft_index < soft.shape[0]):
            continue
        cids = [int(cid) for cid in np.asarray(cids_raw).reshape(-1) if int(cid) in centroids]
        if len(cids) == 1:
            distance = float(np.linalg.norm(soft[soft_index] - centroids[cids[0]]))
        elif len(cids) >= 2:
            distance = min(
                _point_segment_distance(soft[soft_index], centroids[a], centroids[b])
                for pos, a in enumerate(cids)
                for b in cids[pos + 1 :]
            )
        else:
            continue
        bridge_terms.append(distance / diag)
        bridge_soft_indices.append(soft_index)
    bridge_soft_distance = float(np.mean(bridge_terms)) if bridge_terms else 0.0
    bridge_soft_unique = int(len(np.unique(np.asarray(bridge_soft_indices, dtype=np.int64))))
    soft_count = max(int(soft.shape[0]), 1)
    soft_union = np.union1d(
        np.asarray(owned_soft_indices, dtype=np.int64),
        np.asarray(bridge_soft_indices, dtype=np.int64),
    )
    soft_assigned = int(soft_union.size) if soft.size else 0

    hard_quality = float(hierarchy_quality_breakdown(hard, valid)["quality"])
    values = {
        "cluster_compactness": compactness,
        "worst_cluster_spread": worst_spread,
        "neighbor_impurity": neighbor_impurity,
        "edge_stretch": float(edge_stretch),
        "owned_soft_distance": owned_soft_distance,
        "bridge_soft_distance": bridge_soft_distance,
        "clustered_hard_fraction": float(clustered.size / max(hard.shape[0], 1)),
        "clustered_hard_count": float(clustered.size),
        "unclustered_hard_count": float(hard.shape[0] - clustered.size),
        "owned_soft_count": float(owned_soft_unique),
        "owned_soft_coverage": float(owned_soft_unique / soft_count),
        "bridge_soft_count": float(bridge_soft_unique),
        "bridge_soft_coverage": float(bridge_soft_unique / soft_count),
        "covered_soft_count": float(soft_assigned),
        "soft_coverage": float(soft_assigned / soft_count),
        "soft_total": float(soft_count),
        "edge_count": float(sum(1 for edge in edges or ())),
    }
    weights = {
        "cluster_compactness": float(const.HIER_VECTOR_COMPACTNESS_WEIGHT),
        "worst_cluster_spread": float(const.HIER_VECTOR_WORST_SPREAD_WEIGHT),
        "neighbor_impurity": float(const.HIER_VECTOR_IMPURITY_WEIGHT),
        "edge_stretch": float(const.HIER_VECTOR_EDGE_STRETCH_WEIGHT),
        "owned_soft_distance": float(const.HIER_VECTOR_OWNED_SOFT_WEIGHT),
        "bridge_soft_distance": float(const.HIER_VECTOR_BRIDGE_SOFT_WEIGHT),
    }
    weight_sum = max(sum(max(value, 0.0) for value in weights.values()), 1.0e-12)
    composite = sum(max(weights[key], 0.0) * values[key] for key in weights) / weight_sum
    return {
        "composite": float(composite),
        "hard_containment": hard_quality,
        **values,
    }
