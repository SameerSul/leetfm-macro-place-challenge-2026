"""Deterministic hierarchy-quality diagnostics for complete placements."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from utils import constants as const
from placer.local_search.cluster_decompress import hierarchy_quality_breakdown

HIERARCHY_VECTOR_METRICS = (
    "cluster_compactness",
    "worst_cluster_spread",
    "neighbor_impurity",
    "edge_stretch",
    "owned_soft_distance",
    "bridge_soft_distance",
)


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
    for cid_raw, members_raw in clusters.items():
        cid = int(cid_raw)
        members = np.asarray(members_raw, dtype=np.int64)
        members = members[(members >= 0) & (members < hard.shape[0])]
        if members.size == 0:
            continue
        valid[cid] = members
        labels[members] = cid
        center = np.mean(hard[members], axis=0)
        centroids[cid] = center
        spreads.append(float(np.mean(np.linalg.norm(hard[members] - center, axis=1))) / diag)

    compactness = float(np.mean(spreads)) if spreads else 0.0
    worst_spread = float(np.max(spreads)) if spreads else 0.0

    clustered = np.flatnonzero(labels >= 0)
    impurity_terms: list[float] = []
    if clustered.size > 1:
        points = hard[clustered]
        distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        for row, macro in enumerate(clustered):
            own_size = int(valid[int(labels[macro])].size)
            k = min(4, max(1, own_size - 1), clustered.size - 1)
            nearest = np.argsort(distances[row], kind="stable")[:k]
            impurity_terms.append(float(np.mean(labels[clustered[nearest]] != labels[macro])))
    neighbor_impurity = float(np.mean(impurity_terms)) if impurity_terms else 0.0

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
    n_hard = hard.shape[0]
    for cid_raw, full_indices in (cluster_softs or {}).items():
        cid = int(cid_raw)
        if cid not in centroids:
            continue
        for full_index in np.asarray(full_indices, dtype=np.int64).reshape(-1):
            soft_index = int(full_index) - n_hard
            if 0 <= soft_index < soft.shape[0]:
                owned_terms.append(float(np.linalg.norm(soft[soft_index] - centroids[cid])) / diag)
    owned_soft_distance = float(np.mean(owned_terms)) if owned_terms else 0.0

    bridge_terms: list[float] = []
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
    bridge_soft_distance = float(np.mean(bridge_terms)) if bridge_terms else 0.0

    hard_quality = float(hierarchy_quality_breakdown(hard, valid)["quality"])
    values = {
        "cluster_compactness": compactness,
        "worst_cluster_spread": worst_spread,
        "neighbor_impurity": neighbor_impurity,
        "edge_stretch": float(edge_stretch),
        "owned_soft_distance": owned_soft_distance,
        "bridge_soft_distance": bridge_soft_distance,
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
    composite = sum(max(weights[key], 0.0) * values[key] for key in values) / weight_sum
    return {
        "composite": float(composite),
        "hard_containment": hard_quality,
        **values,
        "clustered_hard_fraction": float(clustered.size / max(hard.shape[0], 1)),
        "owned_soft_count": float(len(owned_terms)),
        "bridge_soft_count": float(len(bridge_terms)),
        "edge_count": float(sum(1 for edge in edges or ())),
    }
