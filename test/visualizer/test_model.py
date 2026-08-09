from visualizer.model import (
    collapse_synthetic_groups,
    extract_real_nets,
    filter_real_nets,
    hierarchy_centroid_edges,
    hierarchy_color,
)


def _metadata():
    return {
        "macro_names": ["a", "b", "c"],
        "net_nodes": [[0, 1], [0, 1, 2], [2, 3]],
        "net_weights": [1.0, 5.0, 2.0],
        "port_positions": [[99.0, 2.0]],
        "macro_pin_offsets": [[[1.0, 0.0]], [], []],
    }


def test_real_net_topology_ports_offsets_and_stable_filter():
    nets = extract_real_nets(_metadata())
    assert len(nets) == 3
    assert nets[0].endpoints[0].offset == (1.0, 0.0)
    assert nets[2].endpoints[1].port == 0
    assert [net.index for net in filter_real_nets(nets, 1)] == [2]
    # Selected incidence is additive and output remains original stable order.
    assert [net.index for net in filter_real_nets(nets, 1, selected_macro=2)] == [1, 2]


def test_synthetic_collapse_hierarchy_edges_and_colours_are_deterministic():
    hierarchy = {
        "leaf_clusters": {"2": [0, 1], "5": [2]},
        "edges": [[2, 5, 3.5]],
    }
    assert collapse_synthetic_groups(hierarchy, 12) == [
        {"cluster": 2, "members": (0, 1), "weight": 12}
    ]
    assert hierarchy_centroid_edges(hierarchy) == [(2, 5, 3.5)]
    assert hierarchy_color(2) == hierarchy_color(2)
    assert hierarchy_color(2) != hierarchy_color(5)
