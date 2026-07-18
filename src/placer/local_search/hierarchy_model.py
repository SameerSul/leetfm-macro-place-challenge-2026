"""First-class hierarchy model for the hierarchy placement path."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from utils import constants as const
from placer.local_search.clusters import (
    cluster_max_fanout,
    cluster_min_edge,
    compute_region_bbox,
    compute_soft_region_bbox,
    derive_hard_clusters,
    derive_oversized_hard_clusters,
    derive_path_tag_hard_clusters,
    derive_soft_cluster_roles,
    hier_region_density,
    hier_region_margin,
    hier_region_singleton,
)
from placer.local_search.soft_hierarchy import (
    SoftBundle,
    combine_soft_bundle_evidence,
    derive_connectivity_soft_bundles,
    derive_path_soft_bundles,
    label_soft_bundles,
    select_high_confidence_soft_bundles,
)
from placer.scoring.wirelength import _build_wl_cache


@dataclass(frozen=True)
class HierarchyEdge:
    """Weighted relation between two hard clusters."""

    src: int
    dst: int
    weight: float


@dataclass
class HierarchyModel:
    """Hierarchy state shared by placement passes.

    The flat benchmark netlists do not ship module hierarchy. This object keeps
    the inferred hard clusters, soft ownership/bridge roles, inter-cluster
    connectivity, and reusable region construction in one place.
    """

    labels: np.ndarray
    clusters: dict[int, np.ndarray]
    cluster_softs: dict[int, np.ndarray]
    bridge_softs: dict[int, np.ndarray]
    soft_bundles: tuple[SoftBundle, ...]
    soft_connectivity_bundles: tuple[SoftBundle, ...]
    soft_bundle_evidence: tuple[SoftBundle, ...]
    active_soft_bundles: tuple[SoftBundle, ...]
    edges: list[HierarchyEdge]
    cluster_confidence: dict[int, float]
    split_parents: dict[int, list[int]]
    cluster_source: str
    max_fanout: int
    min_edge: int

    @classmethod
    def build(cls, plc, n: int, n_soft: int, hard_sizes=None) -> "HierarchyModel":
        max_fanout = cluster_max_fanout()
        min_edge = cluster_min_edge()
        tagged = derive_path_tag_hard_clusters(plc, n)
        if tagged is not None:
            labels, clusters = tagged
            cluster_source = "hierarchy_path_tags"
        else:
            labels, clusters = derive_oversized_hard_clusters(
                plc,
                n,
                n_soft=n_soft,
                max_fanout=max_fanout,
                min_edge=min_edge,
                hard_sizes=hard_sizes,
            )
            cluster_source = "hierarchy_oversized_connectivity"
        split_parents = _derive_split_parents(
            plc,
            n,
            n_soft,
            labels,
            max_fanout,
            min_edge,
        )
        cluster_softs, bridge_softs = derive_soft_cluster_roles(
            plc,
            n,
            n_soft,
            labels,
            max_fanout=max_fanout,
            bridge_ratio=float(const.HIER_BRIDGE_SOFT_RATIO),
        )
        soft_bundles = derive_path_soft_bundles(
            plc,
            n_soft,
            max_depth=int(const.HIER_SOFT_TAG_PREFIX_MAX_DEPTH),
            min_group=int(const.HIER_SOFT_TAG_PREFIX_MIN_GROUP),
            min_coverage=float(const.HIER_SOFT_TAG_PREFIX_MIN_COVERAGE),
        )
        soft_connectivity_bundles = derive_connectivity_soft_bundles(
            plc,
            n_soft,
            max_fanout=int(const.HIER_SOFT_BUNDLE_MAX_FANOUT),
            min_shared_nets=int(const.HIER_SOFT_BUNDLE_MIN_SHARED_NETS),
            edge_ratio=float(const.HIER_SOFT_BUNDLE_EDGE_RATIO),
            max_size=int(const.HIER_SOFT_BUNDLE_MAX_SIZE),
        )
        soft_bundle_evidence = label_soft_bundles(
            combine_soft_bundle_evidence(
                soft_bundles,
                soft_connectivity_bundles,
                cluster_softs,
                bridge_softs,
                n_hard=n,
            ),
            high_threshold=float(const.HIER_SOFT_BUNDLE_HIGH_CONFIDENCE),
            medium_threshold=float(const.HIER_SOFT_BUNDLE_MEDIUM_CONFIDENCE),
        )
        active_soft_bundles = select_high_confidence_soft_bundles(
            soft_bundle_evidence,
            high_threshold=float(const.HIER_SOFT_BUNDLE_HIGH_CONFIDENCE),
            medium_threshold=float(const.HIER_SOFT_BUNDLE_MEDIUM_CONFIDENCE),
        )
        edges, confidence = _cluster_graph(
            plc,
            labels,
            clusters,
            max_fanout,
            hard_sizes=hard_sizes,
        )
        return cls(
            labels=labels,
            clusters=clusters,
            cluster_softs=cluster_softs,
            bridge_softs=bridge_softs,
            soft_bundles=soft_bundles,
            soft_connectivity_bundles=soft_connectivity_bundles,
            soft_bundle_evidence=soft_bundle_evidence,
            active_soft_bundles=active_soft_bundles,
            edges=edges,
            cluster_confidence=confidence,
            split_parents=split_parents,
            cluster_source=cluster_source,
            max_fanout=max_fanout,
            min_edge=min_edge,
        )

    def dreamplace_groups(self, plc, n: int) -> list[list[str]]:
        """Return module-name groups for grouped DREAMPlace."""
        hmi, smi = plc.hard_macro_indices, plc.soft_macro_indices
        groups: list[list[str]] = []
        for cid, mem in self.clusters.items():
            names = [plc.modules_w_pins[hmi[int(a)]].get_name() for a in mem]
            for p in self.cluster_softs.get(cid, []):
                names.append(plc.modules_w_pins[smi[int(p) - n]].get_name())
            groups.append(names)
        return groups

    def hard_regions(
        self,
        hard_xy,
        sizes,
        hw,
        hh,
        cw,
        ch,
        n,
        *,
        cluster_heat=None,
        heat_expand_frac: float = 0.0,
        heat_hot_percentile: float = 70.0,
        heat_escape_min: float = 0.25,
    ) -> np.ndarray:
        """Build per-hard center-feasible hierarchy regions."""
        return compute_region_bbox(
            hard_xy,
            sizes,
            hw,
            hh,
            cw,
            ch,
            n,
            self.labels,
            self.clusters,
            target_density=hier_region_density(),
            margin=hier_region_margin(),
            singleton_window=hier_region_singleton(),
            cluster_heat=cluster_heat,
            heat_expand_frac=heat_expand_frac,
            heat_hot_percentile=heat_hot_percentile,
            heat_escape_min=heat_escape_min,
        )

    def soft_regions(
        self,
        hard_xy,
        soft_xy,
        hard_sizes,
        hard_hw,
        hard_hh,
        soft_hw,
        soft_hh,
        cw,
        ch,
        n,
        *,
        cluster_heat=None,
        heat_expand_frac: float = 0.0,
        heat_hot_percentile: float = 70.0,
        heat_escape_min: float = 0.25,
    ) -> np.ndarray:
        """Build per-soft center-feasible hierarchy regions."""
        return compute_soft_region_bbox(
            hard_xy,
            soft_xy,
            hard_sizes,
            hard_hw,
            hard_hh,
            soft_hw,
            soft_hh,
            cw,
            ch,
            n,
            self.clusters,
            self.cluster_softs,
            bridge_softs=self.bridge_softs,
            target_density=hier_region_density(),
            margin=hier_region_margin(),
            singleton_window=hier_region_singleton(),
            cluster_heat=cluster_heat,
            heat_expand_frac=heat_expand_frac,
            heat_hot_percentile=heat_hot_percentile,
            heat_escape_min=heat_escape_min,
        )


def _cluster_graph(
    plc,
    labels: np.ndarray,
    clusters: dict[int, np.ndarray],
    max_fanout: int,
    hard_sizes=None,
):
    """Build weighted inter-cluster graph and calibrated per-cluster confidence."""
    cache = _build_wl_cache(plc)
    ref_idx = cache["ref_idx"]
    net_starts = cache["net_starts"]
    net_lengths = cache["net_lengths"]
    net_weights = cache["net_weights"]
    hb2a = {int(b): a for a, b in enumerate(plc.hard_macro_indices)}

    internal = {int(cid): 0.0 for cid in clusters}
    external = {int(cid): 0.0 for cid in clusters}
    pair_weight: dict[tuple[int, int], float] = {}
    cluster_sizes = {int(cid): int(np.asarray(members).size) for cid, members in clusters.items()}
    hard_area = None
    if hard_sizes is not None:
        hard_area = np.asarray(hard_sizes[: int(len(labels))], dtype=np.float64)
        if hard_area.ndim == 2 and hard_area.shape[1] >= 2:
            hard_area = hard_area[:, 0] * hard_area[:, 1]
        else:
            hard_area = np.ones(int(len(labels)), dtype=np.float64)
    else:
        hard_area = np.ones(int(len(labels)), dtype=np.float64)
    hard_area = np.asarray(hard_area, dtype=np.float64)
    hard_area[np.asarray(hard_area) <= 0.0] = 1.0
    cluster_area = {
        int(cid): float(np.sum(hard_area[np.asarray(members, dtype=np.int64)]))
        for cid, members in clusters.items()
        if len(np.asarray(members)) > 0
    }
    n = max(int(len(labels)), 1)
    total_area = float(np.sum(hard_area[:n])) if n > 0 else 1.0

    for net_i in range(len(net_starts)):
        length = int(net_lengths[net_i])
        if length < 2 or length > max_fanout:
            continue
        start = int(net_starts[net_i])
        hard_a = [hb2a[int(r)] for r in ref_idx[start : start + length] if int(r) in hb2a]
        cids = sorted({int(labels[a]) for a in hard_a if labels[a] >= 0})
        if not cids:
            continue
        weight = float(net_weights[net_i])
        if len(cids) == 1:
            internal[cids[0]] = internal.get(cids[0], 0.0) + weight
            continue
        for pos, a in enumerate(cids):
            external[a] = external.get(a, 0.0) + weight
            for b in cids[pos + 1 :]:
                key = (a, b)
                pair_weight[key] = pair_weight.get(key, 0.0) + weight

    edges = [HierarchyEdge(src=a, dst=b, weight=w) for (a, b), w in sorted(pair_weight.items())]
    confidence = {}
    for cid in clusters:
        inside = internal.get(int(cid), 0.0)
        outside = external.get(int(cid), 0.0)
        total = float(inside + outside)
        if total <= 0.0:
            confidence[int(cid)] = 0.0
            continue
        base_conf = float(inside / total)
        cut_ratio = float(outside / total)
        conductance = float(outside / total)
        base_term = max(0.0, min(1.0, base_conf))

        size = float(cluster_sizes.get(int(cid), 0))
        size_ratio = float(size / n)
        size_term = 1.0 / (1.0 + max(size_ratio - 0.5, 0.0))
        size_term = np.clip(size_term, 0.0, 1.0)

        area = float(cluster_area.get(int(cid), 0.0))
        area_ratio = float(area / max(total_area, 1.0))
        area_term = 1.0 / (1.0 + max(area_ratio - 0.5, 0.0))
        area_term = np.clip(area_term, 0.0, 1.0)

        conductance_term = 1.0 / (1.0 + 3.0 * conductance)
        cut_term = 1.0 / (1.0 + 4.0 * cut_ratio)
        expected_internal = float(max(size_ratio, 1.0e-12) ** 2)
        synthetic_truth = (base_term - expected_internal) / max(1.0 - expected_internal, 1.0e-12)
        synthetic_term = float(np.clip(synthetic_truth, 0.0, 1.0))

        expected = float(max(1, int(size)) * max(1, int(max_fanout)))
        evidence_term = min(1.0, float(total) / expected)
        evidence_term = 0.6 + 0.4 * evidence_term

        calibrated = base_term * evidence_term * size_term * area_term * conductance_term * cut_term * synthetic_term
        confidence[int(cid)] = float(np.clip(calibrated, 0.0, 1.0))
    return edges, confidence


def _derive_split_parents(
    plc,
    n: int,
    n_soft: int,
    labels: np.ndarray,
    max_fanout: int,
    min_edge: int,
) -> dict[int, list[int]]:
    """Map original flat clusters to child clusters created by oversized splitting."""
    _flat_labels, flat_clusters = derive_hard_clusters(
        plc,
        n,
        n_soft=n_soft,
        max_fanout=max_fanout,
        min_edge=min_edge,
    )
    split_parents: dict[int, list[int]] = {}
    for parent_id, members in flat_clusters.items():
        members = np.asarray(members, dtype=np.int64)
        child_ids = sorted({int(labels[int(i)]) for i in members if int(labels[int(i)]) >= 0})
        if len(child_ids) > 1:
            split_parents[int(parent_id)] = child_ids
    return split_parents
