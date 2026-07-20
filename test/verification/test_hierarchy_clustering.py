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


class _PlacedMacro(_NamedMacro):
    def __init__(self, name, pos, width=1.0, height=1.0):
        super().__init__(name)
        self.pos = tuple(float(value) for value in pos)
        self.width = float(width)
        self.height = float(height)

    def get_pos(self):
        return self.pos

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height


def _cache_from_ref_nets(ref_nets):
    ref_nets = [np.asarray(net, dtype=np.int64) for net in ref_nets]
    lengths = np.asarray([len(net) for net in ref_nets], dtype=np.int64)
    starts = (
        np.cumsum(np.concatenate([np.asarray([0]), lengths[:-1]])).astype(np.int64)
        if ref_nets
        else np.zeros(0, dtype=np.int64)
    )
    return {
        "ref_idx": (np.concatenate(ref_nets) if ref_nets else np.zeros(0, dtype=np.int64)),
        "net_starts": starts,
        "net_lengths": lengths,
        "net_weights": np.ones(len(ref_nets), dtype=np.float64),
    }


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
    positions = [
        (2.0, 2.0),
        (3.0, 2.0),
        (2.0, 3.0),
        (3.0, 3.0),
        (27.0, 17.0),
        (28.0, 17.0),
        (27.0, 18.0),
        (28.0, 18.0),
    ]
    plc = SimpleNamespace(
        hard_macro_indices=np.arange(8, dtype=np.int64),
        soft_macro_indices=np.zeros(0, dtype=np.int64),
        modules_w_pins=[_PlacedMacro(f"m{index}", pos) for index, pos in enumerate(positions)],
        get_canvas_width_height=lambda: (32.0, 22.0),
    )
    ref_nets = [list(edge) for edge in _two_community_graph()]
    monkeypatch.setattr(
        cluster_module,
        "_build_wl_cache",
        lambda _plc: _cache_from_ref_nets(ref_nets),
    )

    parents, children, parent_children, child_parent, evidence = derive_one_level_hard_subclusters(
        plc,
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
    assert evidence[7]["source"] == "placement_spatial_structural"


def test_connectivity_without_physical_evidence_does_not_create_fallback(monkeypatch):
    monkeypatch.setattr(
        cluster_module,
        "_hard_edge_maps",
        lambda _plc, _n, _fanout: ({}, _two_community_graph()),
    )

    result = derive_one_level_hard_subclusters(
        SimpleNamespace(),
        8,
        {0: np.arange(8, dtype=np.int64)},
        max_fanout=8,
        hard_sizes=np.ones((8, 2), dtype=np.float64),
        min_parent_size=8,
        min_child_size=2,
        max_cut_ratio=0.15,
    )

    assert result == ({}, {}, {}, {}, {})


def test_one_level_split_uses_close_macros_connected_through_shared_softs(monkeypatch):
    positions = [
        (2.0, 2.0),
        (3.0, 2.0),
        (2.0, 3.0),
        (3.0, 3.0),
        (27.0, 17.0),
        (28.0, 17.0),
        (27.0, 18.0),
        (28.0, 18.0),
        (2.5, 2.5),
        (27.5, 17.5),
    ]
    plc = SimpleNamespace(
        hard_macro_indices=np.arange(8, dtype=np.int64),
        soft_macro_indices=np.asarray([8, 9], dtype=np.int64),
        modules_w_pins=[_PlacedMacro(f"m{index}", pos) for index, pos in enumerate(positions)],
        get_canvas_width_height=lambda: (32.0, 22.0),
    )
    ref_nets = [[hard, 8] for hard in range(4)] + [[hard, 9] for hard in range(4, 8)]
    monkeypatch.setattr(
        cluster_module, "_build_wl_cache", lambda _plc: _cache_from_ref_nets(ref_nets)
    )

    parents, children, parent_children, child_parent, evidence = derive_one_level_hard_subclusters(
        plc,
        8,
        {4: np.arange(8, dtype=np.int64)},
        max_fanout=8,
        n_soft=2,
        hard_sizes=np.ones((8, 2), dtype=np.float64),
        min_parent_size=8,
        min_child_size=2,
        max_cut_ratio=0.15,
    )

    assert parents[4].tolist() == list(range(8))
    assert sorted(tuple(members.tolist()) for members in children.values()) == [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
    ]
    assert parent_children == {4: (0, 1)}
    assert child_parent == {0: 4, 1: 4}
    assert evidence[4]["source"] == "placement_spatial_structural"
    assert evidence[4]["compactness_gain"] > 0.5
    assert evidence[4]["density_gain"] > 0.0
    assert evidence[4]["wire_gain"] > 0.0
    assert evidence[4]["pressure_support"] > 0.0
    assert evidence[4]["confidence"] >= 0.54


def test_proximity_without_connectivity_does_not_create_hierarchy(monkeypatch):
    plc = SimpleNamespace(
        hard_macro_indices=np.arange(8, dtype=np.int64),
        soft_macro_indices=np.zeros(0, dtype=np.int64),
        modules_w_pins=[_PlacedMacro(f"m{index}", (index % 4, index // 4)) for index in range(8)],
        get_canvas_width_height=lambda: (10.0, 10.0),
    )
    monkeypatch.setattr(cluster_module, "_build_wl_cache", lambda _plc: _cache_from_ref_nets([]))

    parents, children, parent_children, child_parent, evidence = derive_one_level_hard_subclusters(
        plc,
        8,
        {0: np.arange(8, dtype=np.int64)},
        max_fanout=8,
        hard_sizes=np.ones((8, 2), dtype=np.float64),
        min_parent_size=8,
        min_child_size=2,
        max_cut_ratio=0.15,
    )

    assert not parents
    assert not children
    assert not parent_children
    assert not child_parent
    assert not evidence


def test_connected_but_spatially_interleaved_split_is_rejected(monkeypatch):
    positions = [
        (float(2 * index), 0.0) if index < 4 else (float(2 * (index - 4) + 1), 0.0)
        for index in range(8)
    ]
    plc = SimpleNamespace(
        hard_macro_indices=np.arange(8, dtype=np.int64),
        soft_macro_indices=np.zeros(0, dtype=np.int64),
        modules_w_pins=[_PlacedMacro(f"m{index}", positions[index]) for index in range(8)],
        get_canvas_width_height=lambda: (10.0, 4.0),
    )
    graph = _two_community_graph()
    ref_nets = [list(edge) for edge in graph]
    monkeypatch.setattr(
        cluster_module, "_build_wl_cache", lambda _plc: _cache_from_ref_nets(ref_nets)
    )

    parents, children, parent_children, child_parent, evidence = derive_one_level_hard_subclusters(
        plc,
        8,
        {0: np.arange(8, dtype=np.int64)},
        max_fanout=8,
        hard_sizes=np.ones((8, 2), dtype=np.float64),
        min_parent_size=8,
        min_child_size=2,
        max_cut_ratio=0.15,
        min_compactness_gain=0.50,
    )

    assert not parents
    assert not children
    assert not parent_children
    assert not child_parent
    assert not evidence
