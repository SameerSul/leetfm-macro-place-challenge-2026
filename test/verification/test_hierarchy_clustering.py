import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import placer.local_search.clusters as cluster_module
from placer.local_search.clusters import (
    _cosine_affinity_components,
    _recursive_bisect_component,
    derive_one_level_hard_subclusters,
    derive_path_tag_hard_clusters,
)


def _two_community_graph():
    edge_weight = {}
    for group in (range(4), range(4, 8)):
        for left in group:
            for right in group:
                if left < right:
                    edge_weight[(left, right)] = 5.0
    edge_weight[(3, 4)] = 0.25
    return edge_weight


def test_strict_partial_bisection_recovers_clear_single_component_boundary():
    leaves = _recursive_bisect_component(
        np.arange(8, dtype=np.int64),
        _two_community_graph(),
        np.ones(8, dtype=np.float64),
        max_size=2,
        min_size=3,
        max_cut_ratio=0.15,
    )

    assert len(leaves) == 2
    assert sorted(tuple(int(value) for value in leaf) for leaf in leaves) == [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
    ]


def test_strict_partial_bisection_rejects_weak_component_boundary():
    edge_weight = _two_community_graph()
    for left in range(4):
        for right in range(4, 8):
            edge_weight[(left, right)] = 2.0

    leaves = _recursive_bisect_component(
        np.arange(8, dtype=np.int64),
        edge_weight,
        np.ones(8, dtype=np.float64),
        max_size=2,
        min_size=3,
        max_cut_ratio=0.15,
    )

    assert len(leaves) == 1
    assert np.array_equal(leaves[0], np.arange(8, dtype=np.int64))


def test_soft_affinity_components_merge_tiny_fragment_to_strongest_group():
    affinity = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.1, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.1],
            [0.2, 0.2, 0.0, 0.0],
        ]
    )

    groups = _cosine_affinity_components(
        affinity,
        cosine_threshold=0.90,
        min_size=2,
    )

    assert groups is not None
    assert [group.tolist() for group in groups] == [[0, 1, 4], [2, 3]]


class _NamedMacro:
    def __init__(self, name):
        self.name = str(name)

    def get_name(self):
        return self.name


def test_path_tags_retain_exactly_one_useful_parent_level():
    names = [
        f"top/{side}/{unit}/{bank}/m{macro}"
        for side, unit in (("a", "u0"), ("b", "v0"))
        for bank in ("b0", "b1")
        for macro in (0, 1)
    ]
    plc = SimpleNamespace(
        hard_macro_indices=np.arange(len(names), dtype=np.int64),
        modules_w_pins=[_NamedMacro(name) for name in names],
    )

    labels, clusters = derive_path_tag_hard_clusters(plc, len(names))
    hierarchy = plc._hard_clusters_path_tag_hierarchy
    parent_clusters = hierarchy[1]
    parent_children = hierarchy[2]
    child_parent = hierarchy[3]
    parent_depth = hierarchy[4]

    assert len(clusters) == 4
    assert np.array_equal(labels, np.repeat(np.arange(4, dtype=np.int64), 2))
    assert parent_depth == 3
    assert {parent: children for parent, children in parent_children.items()} == {
        0: (0, 1),
        1: (2, 3),
    }
    assert {child: parent for child, parent in child_parent.items()} == {
        0: 0,
        1: 0,
        2: 1,
        3: 1,
    }
    assert [members.tolist() for members in parent_clusters.values()] == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
    ]


def test_active_cluster_gets_only_one_non_recursive_bisection(monkeypatch):
    monkeypatch.setattr(
        cluster_module,
        "_hard_edge_maps",
        lambda _plc, _n, _fanout: ({}, _two_community_graph()),
    )

    parents, children, parent_children, child_parent = derive_one_level_hard_subclusters(
        SimpleNamespace(),
        8,
        {7: np.arange(8, dtype=np.int64)},
        max_fanout=8,
        hard_sizes=np.ones((8, 2), dtype=np.float64),
        min_parent_size=8,
        min_child_size=2,
        max_cut_ratio=0.15,
    )

    assert parents[7].tolist() == list(range(8))
    assert len(children) == 2
    assert sorted(tuple(members.tolist()) for members in children.values()) == [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
    ]
    assert parent_children == {7: (0, 1)}
    assert child_parent == {0: 7, 1: 7}
