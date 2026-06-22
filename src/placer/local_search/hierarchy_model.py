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
    derive_cluster_softs,
    derive_hard_clusters,
    derive_oversized_hard_clusters,
    derive_path_tag_hard_clusters,
    derive_soft_cluster_roles,
    hier_region_density,
    hier_region_margin,
    hier_region_singleton,
)
from placer.scoring.wirelength import _build_wl_cache


@dataclass(frozen=True)
class HierarchyEdge:
    """Weighted relation between two hard clusters."""

    src: int
    dst: int
    weight: float
    net_count: int


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
    edges: list[HierarchyEdge]
    cluster_confidence: dict[int, float]
    split_parents: dict[int, list[int]]
    max_fanout: int
    min_edge: int

    @classmethod
    def build(cls, plc, n: int, n_soft: int, hard_sizes=None) -> "HierarchyModel":
        max_fanout = cluster_max_fanout()
        min_edge = cluster_min_edge()
        tagged = derive_path_tag_hard_clusters(plc, n)
        if tagged is not None:
            labels, clusters = tagged
        elif const.HIER_OVERSIZE_CLUSTER_SPLIT:
            labels, clusters = derive_oversized_hard_clusters(
                plc,
                n,
                n_soft=n_soft,
                max_fanout=max_fanout,
                min_edge=min_edge,
                hard_sizes=hard_sizes,
            )
        else:
            labels, clusters = derive_hard_clusters(
                plc,
                n,
                n_soft=n_soft,
                max_fanout=max_fanout,
                min_edge=min_edge,
            )
        split_parents = _derive_split_parents(
            plc,
            n,
            n_soft,
            labels,
            max_fanout,
            min_edge,
        )
        if const.HIER_BRIDGE_SOFTS:
            cluster_softs, bridge_softs = derive_soft_cluster_roles(
                plc,
                n,
                n_soft,
                labels,
                max_fanout=max_fanout,
                bridge_ratio=float(const.HIER_BRIDGE_SOFT_RATIO),
            )
        else:
            cluster_softs = derive_cluster_softs(plc, n, n_soft, labels, max_fanout=max_fanout)
            bridge_softs = {}
        edges, confidence = _cluster_graph(plc, labels, clusters, max_fanout)
        return cls(
            labels=labels,
            clusters=clusters,
            cluster_softs=cluster_softs,
            bridge_softs=bridge_softs,
            edges=edges,
            cluster_confidence=confidence,
            split_parents=split_parents,
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


def _cluster_graph(plc, labels: np.ndarray, clusters: dict[int, np.ndarray], max_fanout: int):
    """Build weighted inter-cluster graph and simple per-cluster confidence."""
    cache = _build_wl_cache(plc)
    ref_idx = cache["ref_idx"]
    net_starts = cache["net_starts"]
    net_lengths = cache["net_lengths"]
    net_weights = cache["net_weights"]
    hb2a = {int(b): a for a, b in enumerate(plc.hard_macro_indices)}

    internal = {int(cid): 0.0 for cid in clusters}
    external = {int(cid): 0.0 for cid in clusters}
    pair_weight: dict[tuple[int, int], float] = {}
    pair_count: dict[tuple[int, int], int] = {}

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
                pair_count[key] = pair_count.get(key, 0) + 1

    edges = [
        HierarchyEdge(src=a, dst=b, weight=w, net_count=pair_count[(a, b)])
        for (a, b), w in sorted(pair_weight.items())
    ]
    confidence = {}
    for cid in clusters:
        inside = internal.get(int(cid), 0.0)
        outside = external.get(int(cid), 0.0)
        confidence[int(cid)] = float(inside / max(inside + outside, 1e-12))
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
    if not const.HIER_OVERSIZE_CLUSTER_SPLIT:
        return {}
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
