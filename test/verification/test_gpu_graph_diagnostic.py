"""Parity and snapshot checks for the offline GPU graph experiment."""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from test.diagnostic import profile_gpu_graphs as diagnostic
from test.verification.test_location_graph import _fixture


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_edge_counts_weights_duplicate_pins_ports_and_raw_fanout(normalize, empty):
    # The final net exceeds the raw pin cutoff even though it has only two owners.
    nets = [] if empty else [[0, 0, 1], [0, 1], [0, 1, 2], [3], [4, 4], [0, 1, 0, 1, 0]]
    lengths = np.array([len(net) for net in nets], dtype=np.int64)
    refs = np.array([node for net in nets for node in net], dtype=np.int64)
    starts = np.cumsum(lengths) - lengths
    weights = np.array([] if empty else [0.1, 0.2, 0.3, 8.0, 9.0, 20.0])
    args = (refs, starts, lengths, weights, np.array([0, 1, 2, -1, 3]), 5, 4, normalize)
    expected = diagnostic.python_edges(*args)
    actual = diagnostic.numba_edges(*args)
    assert diagnostic.edge_parity(expected, actual)["exact"]
    vector = diagnostic.numpy_edges(diagnostic.pack_edges(args))
    assert np.array_equal(expected[0], vector[0])
    assert np.array_equal(expected[1], vector[1])
    np.testing.assert_allclose(expected[2], vector[2], rtol=1e-14, atol=1e-14)
    if not empty:
        assert expected[0].tolist() == [1, 2, 7]
        assert expected[1].tolist() == [3, 1, 1]
    if torch.cuda.is_available():
        cuda = diagnostic.readback(
            diagnostic.cuda_edges(diagnostic.upload_edges(diagnostic.pack_edges(args)))
        )
        assert np.array_equal(expected[0], cuda[0])
        assert np.array_equal(expected[1], cuda[1])
        np.testing.assert_allclose(expected[2], cuda[2], rtol=1e-14, atol=1e-14)


def test_weight_accumulation_preserves_net_order_and_flags_near_ties():
    args = (
        np.array([0, 1] * 3),
        np.array([0, 2, 4]),
        np.array([2, 2, 2]),
        np.array([1e16, 1.0, -1e16]),
        np.array([0, 1]),
        2,
        2,
        False,
    )
    expected = diagnostic.python_edges(*args)
    assert expected[2].tolist() == [0.0]
    assert diagnostic.edge_parity(expected, diagnostic.numba_edges(*args))["exact"]
    changed = (expected[0], expected[1], np.array([1e-16]))
    assert not diagnostic.edge_parity(expected, changed)["exact"]
    assert not diagnostic.edge_parity(expected, (np.array([99]), *expected[1:]))["exact"]


def test_affinity_snapshots_follow_ownership_and_rollback_without_mutating_graph():
    graph, hierarchy, positions = _fixture()
    graph.analyze_state(np.ones((2, 2)), 10.0, 10.0)
    initial = graph.to_visualizer_payload()

    def check():
        before = graph.to_visualizer_payload()
        args, groups = diagnostic.pack_affinity(graph)
        expected = diagnostic.python_affinity(args)
        for candidate in (diagnostic.numba_affinity(*args), diagnostic.numpy_affinity(args)):
            assert diagnostic.affinity_parity(graph, args, groups, expected, candidate)["exact"]
        if torch.cuda.is_available():
            candidate = diagnostic.cuda_affinity(diagnostic.upload_affinity(args)).cpu().numpy()
            assert diagnostic.affinity_parity(graph, args, groups, expected, candidate)["exact"]
        assert graph.to_visualizer_payload() == before
        assert graph.active_edges is hierarchy.edges
        return args, expected

    old_args, old_values = check()
    graph.checkpoint("diagnostic fixture")
    graph.reassign_macros({3: 1})  # Soft transfer must leave hard hierarchy edges alone.
    assert graph.cluster_neighbors(0) == {1: 1.0}
    check()
    assert np.array_equal(diagnostic.numba_affinity(*old_args), old_values)
    assert graph.rollback_graph()
    args, values = check()
    assert np.array_equal(values, old_values)
    graph.checkpoint("hard transfer")
    graph.reassign_macros({1: 1})
    check()
    assert graph.rollback_graph()
    check()
    assert graph.to_visualizer_payload()["clusters"] == initial["clusters"]
    assert np.array_equal(positions, np.array([node.position for node in graph.macros.values()]))


@pytest.mark.parametrize("incremental", [False, True])
def test_compiled_split_respects_ties_isolates_and_minimum_size(incremental):
    for edges in (
        {},
        {(0, 1): 2.0, (2, 3): 2.0, (1, 2): 0.25},
        {(i, j): 1.0 for i in range(6) for j in range(i + 1, 6)},
    ):
        for minimum in (2, 4):
            members = np.array([5, 0, 3, 1, 4, 2])
            areas = np.ones(6)
            expected = diagnostic.clustering._balanced_graph_split(
                members, edges, areas, min_size=minimum
            )
            actual = diagnostic.compiled_split(
                members, edges, areas, min_size=minimum, incremental=incremental
            )
            assert diagnostic.split_equal(expected, actual)


def test_measure_reports_first_call_and_requested_warm_samples():
    calls = []
    result = diagnostic.measure(lambda: calls.append(1), 3)
    assert len(calls) == 5
    assert len(result["samples_ms"]) == 3
    assert result["median_ms"] >= 0


def test_affinity_parity_rejects_a_changed_complete_frontier_order():
    graph, _, _ = _fixture()
    args, groups = diagnostic.pack_affinity(graph)
    reference = diagnostic.python_affinity(args)
    changed = reference.copy()
    source, destination, start, end = next(group for group in groups if group[3] - group[2] > 1)
    last = graph.frontier_records(source, destination)[-1]["index"]
    query = next(q for q in range(start, end) if args[4][q, 0] == last)
    changed[query, 2] += 1e9
    result = diagnostic.affinity_parity(graph, args, groups, reference, changed)
    assert result["reference_matches_production"]
    assert not result["rank_equal"]
    assert not result["exact"]
