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
    derive_one_level_hard_subclusters,
    derive_oversized_hard_clusters,
    derive_path_tag_hard_clusters,
    derive_soft_cluster_role_evidence,
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
    the active hard clusters, one non-recursive parent/child level, soft
    ownership/bridge roles, inter-cluster connectivity, and reusable region
    construction in one place.
    """

    labels: np.ndarray
    clusters: dict[int, np.ndarray]
    cluster_softs: dict[int, np.ndarray]
    bridge_softs: dict[int, np.ndarray]
    soft_role_evidence: dict[int, dict[str, object]]
    soft_bundles: tuple[SoftBundle, ...]
    soft_connectivity_bundles: tuple[SoftBundle, ...]
    soft_bundle_evidence: tuple[SoftBundle, ...]
    active_soft_bundles: tuple[SoftBundle, ...]
    edges: list[HierarchyEdge]
    cluster_confidence: dict[int, float]
    split_parents: dict[int, list[int]]
    subcluster_labels: np.ndarray
    subclusters: dict[int, np.ndarray]
    subcluster_softs: dict[int, np.ndarray]
    subcluster_bridge_softs: dict[int, np.ndarray]
    subcluster_edges: list[HierarchyEdge]
    subcluster_confidence: dict[int, float]
    parent_labels: np.ndarray
    parent_clusters: dict[int, np.ndarray]
    parent_children: dict[int, tuple[int, ...]]
    child_parent: dict[int, int]
    parent_cluster_softs: dict[int, np.ndarray]
    parent_bridge_softs: dict[int, np.ndarray]
    parent_edges: list[HierarchyEdge]
    parent_confidence: dict[int, float]
    subcluster_evidence: dict[int, dict[str, float | str]]
    subhierarchy_source: str
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
            path_hierarchy = getattr(plc, "_hard_clusters_path_tag_hierarchy", None)
            if path_hierarchy is None:
                retained_parent_clusters = {}
                retained_parent_children = {}
            else:
                retained_parent_clusters = dict(path_hierarchy[1])
                retained_parent_children = dict(path_hierarchy[2])
            split_parents = {}
            retained_source = "hierarchy_path_parent"
        else:
            labels, clusters = derive_oversized_hard_clusters(
                plc,
                n,
                n_soft=n_soft,
                max_fanout=max_fanout,
                min_edge=min_edge,
                hard_sizes=hard_sizes,
            )
            cluster_source = str(
                getattr(
                    plc,
                    "_hard_clusters_oversized_source",
                    "hierarchy_oversized_connectivity",
                )
            )
            retained_parent_clusters, retained_parent_children, _retained_child_parent = (
                _derive_split_hierarchy(
                    plc,
                    n,
                    n_soft,
                    labels,
                    max_fanout,
                    min_edge,
                )
            )
            split_parents = {
                int(parent_id): [int(child_id) for child_id in child_ids]
                for parent_id, child_ids in retained_parent_children.items()
            }
            retained_source = "hierarchy_connectivity_parent"

        subclusters: dict[int, np.ndarray] = {}
        parent_children: dict[int, tuple[int, ...]] = {}
        child_parent: dict[int, int] = {}
        subcluster_evidence: dict[int, dict[str, float | str]] = {}
        if retained_parent_clusters:
            parent_clusters = retained_parent_clusters
            next_child = 0
            for parent_id, active_child_ids in sorted(retained_parent_children.items()):
                child_ids = []
                for active_child_id in active_child_ids:
                    if int(active_child_id) not in clusters:
                        continue
                    child_id = int(next_child)
                    next_child += 1
                    subclusters[child_id] = np.asarray(
                        clusters[int(active_child_id)],
                        dtype=np.int64,
                    )
                    child_parent[child_id] = int(parent_id)
                    child_ids.append(child_id)
                if len(child_ids) >= 2:
                    parent_children[int(parent_id)] = tuple(child_ids)
                    subcluster_evidence[int(parent_id)] = {
                        "source": retained_source,
                        "confidence": 1.0,
                        "cut_ratio": 0.0,
                        "compactness_gain": 0.0,
                        "density_gain": 0.0,
                        "wire_gain": 0.0,
                        "pressure_support": 0.0,
                    }
            subhierarchy_source = retained_source
        else:
            (
                parent_clusters,
                subclusters,
                parent_children,
                child_parent,
                subcluster_evidence,
            ) = derive_one_level_hard_subclusters(
                plc,
                n,
                clusters,
                max_fanout=max_fanout,
                n_soft=n_soft,
                hard_sizes=hard_sizes,
                min_parent_size=int(const.HIER_SUBCLUSTER_MIN_PARENT_HARD),
                min_child_size=int(const.HIER_SUBCLUSTER_MIN_CHILD_HARD),
                max_cut_ratio=float(const.HIER_SUBCLUSTER_MAX_CUT_RATIO),
                shared_soft_weight=float(const.HIER_SUBCLUSTER_SHARED_SOFT_WEIGHT),
                proximity_weight=float(const.HIER_SUBCLUSTER_SPATIAL_PROXIMITY_WEIGHT),
                pressure_weight=float(const.HIER_SUBCLUSTER_SPATIAL_PRESSURE_WEIGHT),
                spatial_neighbors=int(const.HIER_SUBCLUSTER_SPATIAL_NEIGHBORS),
                max_soft_degree=int(const.HIER_SUBCLUSTER_SPATIAL_MAX_SOFT_DEGREE),
                min_compactness_gain=float(const.HIER_SUBCLUSTER_SPATIAL_MIN_COMPACTNESS_GAIN),
                min_confidence=float(const.HIER_SUBCLUSTER_SPATIAL_MIN_CONFIDENCE),
            )
            evidence_sources = {
                str(evidence.get("source", "")) for evidence in subcluster_evidence.values()
            }
            subhierarchy_source = (
                "hierarchy_spatial_structural_bisection"
                if "placement_spatial_structural" in evidence_sources
                else "hierarchy_one_level_bisection"
            )
        cluster_softs, bridge_softs = derive_soft_cluster_roles(
            plc,
            n,
            n_soft,
            labels,
            max_fanout=max_fanout,
            bridge_ratio=float(const.HIER_BRIDGE_SOFT_RATIO),
        )
        soft_role_evidence = derive_soft_cluster_role_evidence(
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
        subcluster_labels = np.full(n, -1, dtype=np.int64)
        for child_id, members in subclusters.items():
            subcluster_labels[np.asarray(members, dtype=np.int64)] = int(child_id)
        if subclusters:
            subcluster_softs, subcluster_bridge_softs = derive_soft_cluster_roles(
                plc,
                n,
                n_soft,
                subcluster_labels,
                max_fanout=max_fanout,
                bridge_ratio=float(const.HIER_BRIDGE_SOFT_RATIO),
            )
            subcluster_edges, subcluster_confidence = _cluster_graph(
                plc,
                subcluster_labels,
                subclusters,
                max_fanout,
                hard_sizes=hard_sizes,
            )
        else:
            subcluster_softs = {}
            subcluster_bridge_softs = {}
            subcluster_edges = []
            subcluster_confidence = {}
        parent_labels = np.full(n, -1, dtype=np.int64)
        for parent_id, members in parent_clusters.items():
            parent_labels[np.asarray(members, dtype=np.int64)] = int(parent_id)
        if parent_clusters:
            parent_cluster_softs, parent_bridge_softs = derive_soft_cluster_roles(
                plc,
                n,
                n_soft,
                parent_labels,
                max_fanout=max_fanout,
                bridge_ratio=float(const.HIER_BRIDGE_SOFT_RATIO),
            )
            parent_edges, parent_confidence = _cluster_graph(
                plc,
                parent_labels,
                parent_clusters,
                max_fanout,
                hard_sizes=hard_sizes,
            )
        else:
            parent_cluster_softs = {}
            parent_bridge_softs = {}
            parent_edges = []
            parent_confidence = {}
        return cls(
            labels=labels,
            clusters=clusters,
            cluster_softs=cluster_softs,
            bridge_softs=bridge_softs,
            soft_role_evidence=soft_role_evidence,
            soft_bundles=soft_bundles,
            soft_connectivity_bundles=soft_connectivity_bundles,
            soft_bundle_evidence=soft_bundle_evidence,
            active_soft_bundles=active_soft_bundles,
            edges=edges,
            cluster_confidence=confidence,
            split_parents=split_parents,
            subcluster_labels=subcluster_labels,
            subclusters=subclusters,
            subcluster_softs=subcluster_softs,
            subcluster_bridge_softs=subcluster_bridge_softs,
            subcluster_edges=subcluster_edges,
            subcluster_confidence=subcluster_confidence,
            parent_labels=parent_labels,
            parent_clusters=parent_clusters,
            parent_children=parent_children,
            child_parent=child_parent,
            parent_cluster_softs=parent_cluster_softs,
            parent_bridge_softs=parent_bridge_softs,
            parent_edges=parent_edges,
            parent_confidence=parent_confidence,
            subcluster_evidence=subcluster_evidence,
            subhierarchy_source=(subhierarchy_source if parent_clusters else "none"),
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

    def parent_hard_regions(
        self,
        hard_xy,
        sizes,
        hw,
        hh,
        cw,
        ch,
        n,
    ) -> "np.ndarray | None":
        """Build center-feasible regions for the retained parent layer."""
        if not self.parent_clusters:
            return None
        return compute_region_bbox(
            hard_xy,
            sizes,
            hw,
            hh,
            cw,
            ch,
            n,
            self.parent_labels,
            self.parent_clusters,
            target_density=hier_region_density(),
            margin=hier_region_margin(),
            singleton_window=hier_region_singleton(),
        )

    def parent_soft_regions(
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
    ) -> "np.ndarray | None":
        """Build soft center regions for the retained parent layer."""
        if not self.parent_clusters:
            return None
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
            self.parent_clusters,
            self.parent_cluster_softs,
            bridge_softs=self.parent_bridge_softs,
            target_density=hier_region_density(),
            margin=hier_region_margin(),
            singleton_window=hier_region_singleton(),
        )

    def subcluster_hard_regions(
        self,
        hard_xy,
        sizes,
        hw,
        hh,
        cw,
        ch,
        n,
        *,
        cluster_margins,
    ) -> "np.ndarray | None":
        """Build deepest-child boxes from each footprint plus its margin."""
        if not self.subclusters:
            return None
        return compute_region_bbox(
            hard_xy,
            sizes,
            hw,
            hh,
            cw,
            ch,
            n,
            self.subcluster_labels,
            self.subclusters,
            target_density=hier_region_density(),
            margin=0.0,
            singleton_window=0.0,
            cluster_margins=cluster_margins,
        )

    def subcluster_soft_regions(
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
        cluster_margins,
    ) -> "np.ndarray | None":
        """Build owned-soft boxes for the deepest retained child layer."""
        if not self.subclusters:
            return None
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
            self.subclusters,
            self.subcluster_softs,
            bridge_softs=None,
            target_density=hier_region_density(),
            margin=0.0,
            singleton_window=0.0,
            cluster_margins=cluster_margins,
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
        base_term = base_conf
        base_term = max(0.0, min(1.0, base_term))

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

        calibrated = (
            base_term
            * evidence_term
            * size_term
            * area_term
            * conductance_term
            * cut_term
            * synthetic_term
        )
        confidence[int(cid)] = float(np.clip(calibrated, 0.0, 1.0))
    return edges, confidence


def _derive_split_hierarchy(
    plc,
    n: int,
    n_soft: int,
    labels: np.ndarray,
    max_fanout: int,
    min_edge: int,
) -> tuple[dict[int, np.ndarray], dict[int, tuple[int, ...]], dict[int, int]]:
    """Retain one flat-component parent above oversized leaf clusters."""
    _flat_labels, flat_clusters = derive_hard_clusters(
        plc,
        n,
        n_soft=n_soft,
        max_fanout=max_fanout,
        min_edge=min_edge,
    )
    parent_clusters: dict[int, np.ndarray] = {}
    parent_children: dict[int, tuple[int, ...]] = {}
    child_parent: dict[int, int] = {}
    for parent_id, members in flat_clusters.items():
        members = np.asarray(members, dtype=np.int64)
        child_ids = sorted({int(labels[int(i)]) for i in members if int(labels[int(i)]) >= 0})
        if len(child_ids) > 1:
            parent_clusters[int(parent_id)] = members.copy()
            parent_children[int(parent_id)] = tuple(child_ids)
            for child_id in child_ids:
                child_parent[int(child_id)] = int(parent_id)
    return parent_clusters, parent_children, child_parent


def _derive_split_parents(
    plc,
    n: int,
    n_soft: int,
    labels: np.ndarray,
    max_fanout: int,
    min_edge: int,
) -> dict[int, list[int]]:
    """Backward-compatible parent-to-child map for verification utilities."""
    _parents, children, _child_parent = _derive_split_hierarchy(
        plc,
        n,
        n_soft,
        labels,
        max_fanout,
        min_edge,
    )
    return {
        int(parent_id): [int(child_id) for child_id in child_ids]
        for parent_id, child_ids in children.items()
    }
