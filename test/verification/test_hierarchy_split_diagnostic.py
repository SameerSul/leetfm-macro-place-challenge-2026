"""Parity checks for the retained CPU hierarchy-split experiment."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from test.diagnostic import profile_hierarchy_splits as diagnostic


def test_compiled_split_respects_ties_isolates_and_minimum_size():
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
            actual = diagnostic.compiled_split(members, edges, areas, min_size=minimum)
            assert diagnostic.split_equal(expected, actual)


def test_measure_reports_first_call_and_requested_warm_samples():
    calls = []
    result = diagnostic.measure(lambda: calls.append(1), 3)
    assert len(calls) == 5
    assert len(result["samples_ms"]) == 3
    assert result["median_ms"] >= 0
