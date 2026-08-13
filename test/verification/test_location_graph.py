import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from placer.local_search.location_graph import LocationAwareGraph


class _Module:
    def __init__(self, name):
        self.name = name

    def get_name(self):
        return self.name


def _fixture():
    plc = SimpleNamespace(
        hard_macro_indices=[10, 11, 12],
        soft_macro_indices=[20, 21],
        modules_w_pins=[_Module(f"module_{index}") for index in range(22)],
        _wl_vec_cache={
            "net_starts": np.array([0, 2, 4]),
            "net_lengths": np.array([2, 2, 3]),
            "net_weights": np.array([2.0, 1.0, 1.0]),
            "ref_idx": np.array([10, 11, 11, 20, 11, 12, 21]),
        },
    )
    hierarchy = SimpleNamespace(
        labels=np.array([0, 0, 1]),
        clusters={0: np.array([0, 1]), 1: np.array([2])},
        cluster_softs={0: np.array([3])},
        bridge_softs={1: np.array([0, 1])},
        subcluster_labels=np.array([4, 4, 5]),
        subclusters={4: np.array([0, 1]), 5: np.array([2])},
        parent_labels=np.array([7, 7, 7]),
        edges=[SimpleNamespace(src=0, dst=1, weight=1.0)],
    )
    positions = np.array([[1.0, 1.0], [3.0, 1.0], [8.0, 8.0], [2.0, 2.0], [5.0, 5.0]])
    sizes = np.ones((5, 2), dtype=np.float64)
    graph = LocationAwareGraph.build(plc, hierarchy, positions, sizes, max_fanout=3)
    hierarchy.location_graph = graph
    return graph, hierarchy, positions


def _graph():
    return _fixture()[0]


def test_location_graph_keeps_macro_cluster_roles_and_adjacency():
    graph, hierarchy, _positions = _fixture()

    assert graph.macros[0].name == "module_10"
    assert graph.macros[0].kind == "hard"
    assert graph.cluster_of(0) == 0
    assert graph.cluster_of(3) == 0
    assert graph.macros[3].subcluster_id == 4
    assert graph.macros[4].bridge_clusters == (0, 1)
    assert graph.cluster_of(4) == -1
    assert graph.neighbors(0)[1] == 2.0
    assert graph.cluster_neighbors(0) == {1: 1.0}
    assert graph.active_edges is hierarchy.edges
    graph._rebuild_cluster_adjacency()
    assert graph.cluster_neighbors(0) == {1: 1.0}
    assert graph.summary() == {
        "revision": 1,
        "macros": 5,
        "hard_macros": 3,
        "soft_macros": 2,
        "owned_macros": 4,
        "bridge_softs": 1,
        "macro_edges": 5,
        "clusters": 2,
        "cluster_edges": 1,
    }


def test_location_graph_synchronizes_nodes_cluster_geometry_and_spatial_queries():
    graph = _graph()
    original = graph.macros[0]
    revision = graph.revision
    hard = np.array([[2.0, 2.0], [4.0, 2.0], [9.0, 9.0]])
    soft = np.array([[3.0, 3.0], [6.0, 6.0]])

    graph.synchronize(hard, soft)

    assert graph.macros[0] is original
    assert graph.revision == revision + 1
    assert np.allclose(graph.macros[0].position, [2.0, 2.0])
    assert np.allclose(graph.clusters[0].centroid, [3.0, 7.0 / 3.0])
    assert np.allclose(graph.clusters[0].bbox, [1.5, 1.5, 4.5, 3.5])
    assert graph.macros_in_box([1.5, 1.5, 3.5, 3.5], cluster_id=0) == (0, 3)

    graph.update_macros([1, 4], [[5.0, 2.0], [7.0, 7.0]])
    assert graph.revision == revision + 2
    assert np.allclose(graph.macros[1].position, [5.0, 2.0])
    assert np.allclose(graph.macros[4].position, [7.0, 7.0])
    assert np.allclose(graph.clusters[0].bbox, [1.5, 1.5, 5.5, 3.5])


def test_location_graph_returns_decaying_hard_graph_hops_within_one_cluster():
    graph = _graph()

    members, scales = graph.graph_hop_profile(
        np.array([0]),
        np.array([0, 1, 2]),
        max_hops=2,
        decay=0.5,
    )

    assert np.array_equal(members, [0, 1, 2])
    assert np.allclose(scales, [1.0, 0.5, 0.25])


def test_location_graph_analyzes_capacity_frontiers_routing_and_visual_payload():
    graph = _graph()
    routing = graph.analyze_state(np.array([[3.0, 1.0], [0.5, 0.25]]), 10.0, 10.0)

    assert routing["internal_cluster"] == 3.0
    assert routing["cross_cluster"] == 0.5
    assert routing["bridge_soft"] == 1.0
    assert graph.clusters[0].metrics["hard_area"] == 2.0
    assert graph.clusters[0].metrics["free_capacity"] > 0.0
    assert set(graph.clusters[0].side_capacity) == {"left", "right", "bottom", "top"}

    frontier = graph.frontier_records(0, 1)
    assert {row["index"] for row in frontier} == {0, 1, 3}
    assert all("cut_gain" in row and "capacity_ratio" in row for row in frontier)
    payload = graph.to_visualizer_payload()
    assert payload["analysis_revision"] == graph.revision
    assert payload["routing_attribution"]["cross_cluster"] == 0.5
    assert payload["frontiers"]


def test_location_graph_connected_wave_dynamic_region_order_and_rollback():
    graph = _graph()
    graph.analyze_state(np.ones((2, 2)), 10.0, 10.0)
    bundle = graph.connected_frontier_bundle(1, 1, max_members=3)
    assert bundle[0] == 1
    assert len(bundle) >= 2
    assert all(graph.cluster_of(index) == 0 for index in bundle)

    members, scales = graph.directional_graph_profile(
        np.array([0]), np.array([0, 1, 2]), max_hops=2, decay=0.5
    )
    assert np.array_equal(members, [0, 1, 2])
    assert np.all(scales > 0.0)
    assert scales[0] > scales[1] > scales[2]
    assert graph.dynamic_cluster_bounds(0, 10.0, 10.0).shape == (4,)
    assert sorted(graph.legalization_order([2, 1, 0]).tolist()) == [0, 1, 2]

    old = graph.macros[0].position.copy()
    graph.checkpoint("before")
    graph.update_macros([0], [[7.0, 7.0]])
    assert graph.rollback_graph()
    assert np.allclose(graph.macros[0].position, old)
