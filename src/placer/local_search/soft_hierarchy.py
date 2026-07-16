"""Deterministic hierarchy evidence for groups of related soft macros."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from placer.scoring.wirelength import _build_wl_cache


@dataclass(frozen=True)
class SoftBundle:
    """A group of soft macros inferred to belong to one logical subsystem."""

    members: np.ndarray
    source: str
    key: str
    score: float
    confidence: str = "low"


def soft_bundle_confidence(
    score: float,
    *,
    high_threshold: float = 0.90,
    medium_threshold: float = 0.75,
) -> str:
    """Return the deterministic confidence label for one bundle score."""
    if float(score) >= float(high_threshold):
        return "high"
    if float(score) >= float(medium_threshold):
        return "medium"
    return "low"


def label_soft_bundles(
    bundles: Sequence[SoftBundle],
    *,
    high_threshold: float = 0.90,
    medium_threshold: float = 0.75,
) -> tuple[SoftBundle, ...]:
    """Attach deterministic high, medium, or low labels to bundle evidence."""
    return tuple(
        SoftBundle(
            members=bundle.members.copy(),
            source=bundle.source,
            key=bundle.key,
            score=float(bundle.score),
            confidence=soft_bundle_confidence(
                bundle.score,
                high_threshold=high_threshold,
                medium_threshold=medium_threshold,
            ),
        )
        for bundle in bundles
    )


def select_high_confidence_soft_bundles(
    bundles: Sequence[SoftBundle],
    *,
    high_threshold: float = 0.90,
    medium_threshold: float = 0.75,
) -> tuple[SoftBundle, ...]:
    """Return only non-ambiguous high-confidence bundle evidence."""
    labeled = label_soft_bundles(
        bundles,
        high_threshold=high_threshold,
        medium_threshold=medium_threshold,
    )
    return tuple(bundle for bundle in labeled if bundle.confidence == "high")


def infer_path_soft_bundles(
    names: Sequence[str],
    *,
    max_depth: int = 5,
    min_group: int = 2,
    min_coverage: float = 0.25,
) -> tuple[SoftBundle, ...]:
    """Group soft macros by the strongest useful instance-path prefix depth."""
    values = [str(name) for name in names]
    total = len(values)
    min_group = max(2, int(min_group))
    max_depth = max(1, int(max_depth))
    min_coverage = float(np.clip(min_coverage, 0.0, 1.0))
    if total < min_group:
        return ()

    path_count = sum("/" in name for name in values)
    if path_count < max(min_group, int(np.ceil(min_coverage * total))):
        return ()

    def _prefix(name: str, depth: int) -> str | None:
        parts = [part for part in name.split("/") if part]
        if len(parts) <= depth:
            return None
        return "/".join(parts[:depth])

    best: tuple[tuple[int, int, int], int, dict[str, list[int]]] | None = None
    for depth in range(1, max_depth + 1):
        buckets: dict[str, list[int]] = {}
        for index, name in enumerate(values):
            prefix = _prefix(name, depth)
            if prefix is not None:
                buckets.setdefault(prefix, []).append(index)
        groups = {
            prefix: members
            for prefix, members in buckets.items()
            if min_group <= len(members) < total
        }
        covered = sum(len(members) for members in groups.values())
        if covered / max(total, 1) < min_coverage:
            continue
        score = (len(groups), covered, depth)
        if best is None or score > best[0]:
            best = (score, depth, groups)

    if best is None:
        return ()

    _score, depth, groups = best
    coverage = sum(len(members) for members in groups.values()) / max(total, 1)
    confidence = float(min(1.0, 0.90 + 0.05 * coverage + 0.01 * min(depth, 5)))
    return tuple(
        SoftBundle(
            members=np.asarray(sorted(members), dtype=np.int64),
            source="path",
            key=prefix,
            score=confidence,
            confidence="high",
        )
        for prefix, members in sorted(groups.items())
    )


def derive_path_soft_bundles(
    plc,
    n_soft: int,
    *,
    max_depth: int = 5,
    min_group: int = 2,
    min_coverage: float = 0.25,
) -> tuple[SoftBundle, ...]:
    """Read soft names from the placement object and infer explicit bundles."""
    try:
        soft_indices = list(plc.soft_macro_indices[: int(n_soft)])
        names = [plc.modules_w_pins[int(index)].get_name() for index in soft_indices]
    except Exception:
        return ()
    return infer_path_soft_bundles(
        names,
        max_depth=max_depth,
        min_group=min_group,
        min_coverage=min_coverage,
    )


def infer_connectivity_soft_bundles(
    n_soft: int,
    soft_nets: Sequence[Sequence[int]],
    *,
    min_shared_nets: int = 2,
    edge_ratio: float = 0.6,
    max_size: int = 16,
) -> tuple[SoftBundle, ...]:
    """Build conservative soft communities from mutually strong shared nets."""
    n_soft = max(0, int(n_soft))
    min_shared_nets = max(1, int(min_shared_nets))
    edge_ratio = float(np.clip(edge_ratio, 0.0, 1.0))
    max_size = max(2, int(max_size))
    if n_soft < 2:
        return ()

    edge_counts: dict[tuple[int, int], int] = {}
    for raw_members in soft_nets:
        members = sorted({int(member) for member in raw_members if 0 <= int(member) < n_soft})
        for pos, left in enumerate(members):
            for right in members[pos + 1 :]:
                edge = (left, right)
                edge_counts[edge] = edge_counts.get(edge, 0) + 1
    if not edge_counts:
        return ()

    strongest = np.zeros(n_soft, dtype=np.int64)
    for (left, right), count in edge_counts.items():
        strongest[left] = max(strongest[left], int(count))
        strongest[right] = max(strongest[right], int(count))

    parents = np.arange(n_soft, dtype=np.int64)

    def _find(index: int) -> int:
        root = int(index)
        while int(parents[root]) != root:
            root = int(parents[root])
        while int(parents[index]) != root:
            parent = int(parents[index])
            parents[index] = root
            index = parent
        return root

    accepted_edges: dict[tuple[int, int], int] = {}
    for (left, right), count in sorted(edge_counts.items()):
        if count < min_shared_nets:
            continue
        if count < edge_ratio * strongest[left] or count < edge_ratio * strongest[right]:
            continue
        left_root, right_root = _find(left), _find(right)
        if left_root != right_root:
            parents[right_root] = left_root
        accepted_edges[(left, right)] = int(count)

    components: dict[int, list[int]] = {}
    for index in range(n_soft):
        if strongest[index] < min_shared_nets:
            continue
        components.setdefault(_find(index), []).append(index)

    bundles = []
    for members in components.values():
        if len(members) < 2 or len(members) > max_size:
            continue
        member_set = set(members)
        support = [
            count
            for (left, right), count in accepted_edges.items()
            if left in member_set and right in member_set
        ]
        if not support:
            continue
        normalized = []
        for (left, right), count in accepted_edges.items():
            if left not in member_set or right not in member_set:
                continue
            denominator = max(int(strongest[left]), int(strongest[right]), 1)
            normalized.append(float(count) / denominator)
        mean_strength = float(np.mean(normalized)) if normalized else 0.0
        mean_support = float(np.mean(support))
        score = float(min(0.84, 0.42 + 0.22 * mean_strength + 0.04 * min(mean_support, 4.0)))
        ordered = np.asarray(sorted(members), dtype=np.int64)
        bundles.append(
            SoftBundle(
                members=ordered,
                source="connectivity",
                key="softnet:" + ",".join(str(member) for member in ordered),
                score=score,
            )
        )
    bundles.sort(key=lambda bundle: (int(bundle.members[0]), len(bundle.members)))
    return tuple(bundles)


def derive_connectivity_soft_bundles(
    plc,
    n_soft: int,
    *,
    max_fanout: int = 8,
    min_shared_nets: int = 2,
    edge_ratio: float = 0.6,
    max_size: int = 16,
) -> tuple[SoftBundle, ...]:
    """Extract qualifying soft-only net memberships from a placement object."""
    try:
        cache = _build_wl_cache(plc)
        ref_idx = cache["ref_idx"]
        net_starts = cache["net_starts"]
        net_lengths = cache["net_lengths"]
        soft_lookup = {
            int(module_index): soft_index
            for soft_index, module_index in enumerate(plc.soft_macro_indices[: int(n_soft)])
        }
    except Exception:
        return ()

    soft_nets = []
    for net_index in range(len(net_starts)):
        length = int(net_lengths[net_index])
        if length < 2 or length > max(2, int(max_fanout)):
            continue
        start = int(net_starts[net_index])
        members = sorted(
            {
                soft_lookup[int(reference)]
                for reference in ref_idx[start : start + length]
                if int(reference) in soft_lookup
            }
        )
        if len(members) >= 2:
            soft_nets.append(members)
    return infer_connectivity_soft_bundles(
        n_soft,
        soft_nets,
        min_shared_nets=min_shared_nets,
        edge_ratio=edge_ratio,
        max_size=max_size,
    )


def combine_soft_bundle_evidence(
    path_bundles: Sequence[SoftBundle],
    connectivity_bundles: Sequence[SoftBundle],
    cluster_softs: dict[int, np.ndarray],
    bridge_softs: dict[int, np.ndarray],
    *,
    n_hard: int,
) -> tuple[SoftBundle, ...]:
    """Score explicit and connectivity bundle candidates against hard affinity."""
    path_bundles = tuple(path_bundles)
    owned_by: dict[int, int] = {}
    for cluster_id, full_indices in cluster_softs.items():
        for full_index in np.asarray(full_indices, dtype=np.int64).reshape(-1):
            soft_index = int(full_index) - int(n_hard)
            if soft_index >= 0:
                owned_by[soft_index] = int(cluster_id)
    bridge_by = {
        int(soft_index): tuple(sorted(int(cluster_id) for cluster_id in np.asarray(cluster_ids)))
        for soft_index, cluster_ids in bridge_softs.items()
    }
    path_member_sets = [set(map(int, bundle.members)) for bundle in path_bundles]

    combined = [
        SoftBundle(
            members=bundle.members.copy(),
            source="path",
            key=bundle.key,
            score=max(float(bundle.score), 0.95),
            confidence="high",
        )
        for bundle in path_bundles
    ]
    for bundle in connectivity_bundles:
        members = [int(member) for member in np.asarray(bundle.members, dtype=np.int64)]
        if len(members) < 2:
            continue
        member_set = set(members)
        if any(member_set <= path_set for path_set in path_member_sets):
            continue

        owner_values = [owned_by[member] for member in members if member in owned_by]
        bridge_values = [bridge_by[member] for member in members if member in bridge_by]
        owner_agreement = 0.0
        bridge_agreement = 0.0
        if owner_values:
            owner_agreement = max(owner_values.count(value) for value in set(owner_values)) / len(
                members
            )
        if bridge_values:
            bridge_agreement = max(
                bridge_values.count(value) for value in set(bridge_values)
            ) / len(members)

        score = float(bundle.score)
        sources = ["connectivity"]
        if owner_agreement >= 0.5:
            score += 0.10 * owner_agreement
            sources.append("owner")
        if bridge_agreement >= 0.5:
            score += 0.06 * bridge_agreement
            sources.append("bridge")
        combined.append(
            SoftBundle(
                members=np.asarray(sorted(members), dtype=np.int64),
                source="+".join(sources),
                key=bundle.key,
                score=float(min(score, 0.89)),
            )
        )

    combined.sort(key=lambda bundle: (bundle.source != "path", int(bundle.members[0]), bundle.key))
    return tuple(combined)
