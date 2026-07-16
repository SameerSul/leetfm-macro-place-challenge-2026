"""Correctness checks for deterministic constraint-graph legalization."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from placer.legalize.constraint_graph import _will_legalize_constraint_graph  # noqa: E402


def _assert_legal(pos: np.ndarray, sizes: np.ndarray, cw: float, ch: float) -> None:
    half = sizes * 0.5
    assert np.all(pos[:, 0] - half[:, 0] >= 0.0)
    assert np.all(pos[:, 0] + half[:, 0] <= cw)
    assert np.all(pos[:, 1] - half[:, 1] >= 0.0)
    assert np.all(pos[:, 1] + half[:, 1] <= ch)
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            separated = (
                abs(float(pos[i, 0] - pos[j, 0])) >= half[i, 0] + half[j, 0] + 0.05
                or abs(float(pos[i, 1] - pos[j, 1])) >= half[i, 1] + half[j, 1] + 0.05
            )
            assert separated, (i, j, pos[i], pos[j])


def test_constraint_graph_resolves_overlaps_deterministically():
    pos = np.array(
        [
            [20.0, 20.0],
            [24.0, 20.0],
            [20.0, 24.0],
            [24.0, 24.0],
            [62.0, 65.0],
        ],
        dtype=np.float64,
    )
    sizes = np.array(
        [[10.0, 10.0], [12.0, 8.0], [8.0, 12.0], [11.0, 11.0], [9.0, 7.0]],
        dtype=np.float64,
    )
    movable = np.ones(len(pos), dtype=bool)
    args = (pos, movable, sizes, sizes[:, 0] * 0.5, sizes[:, 1] * 0.5, 100.0, 90.0)
    actual, stats = _will_legalize_constraint_graph(*args, len(pos))
    repeated, repeated_stats = _will_legalize_constraint_graph(*args, len(pos))

    np.testing.assert_array_equal(actual, repeated)
    assert stats["initial_overlaps"] > 0
    assert stats["final_overlaps"] == 0
    assert repeated_stats["final_overlaps"] == 0
    _assert_legal(actual, sizes, 100.0, 90.0)


def test_constraint_graph_preserves_fixed_macro():
    pos = np.array([[30.0, 30.0], [30.0, 30.0], [34.0, 34.0]], dtype=np.float64)
    sizes = np.full((3, 2), 10.0, dtype=np.float64)
    movable = np.array([False, True, True])
    actual, stats = _will_legalize_constraint_graph(
        pos,
        movable,
        sizes,
        sizes[:, 0] * 0.5,
        sizes[:, 1] * 0.5,
        80.0,
        80.0,
        len(pos),
    )

    np.testing.assert_array_equal(actual[0], pos[0])
    assert stats["infeasible"] is False
    assert stats["final_overlaps"] == 0
    _assert_legal(actual, sizes, 80.0, 80.0)
