"""Parity checks for the Stage 3 legalization kernel."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from placer.legalize import spiral  # noqa: E402


def _legalize(pos, movable, sizes, order, use_numba):
    original = spiral.HAS_NUMBA
    spiral.HAS_NUMBA = use_numba
    try:
        return spiral._will_legalize(
            pos,
            movable,
            sizes,
            sizes[:, 0] * 0.5,
            sizes[:, 1] * 0.5,
            120.0,
            90.0,
            len(pos),
            order=order,
        )
    finally:
        spiral.HAS_NUMBA = original


def test_spiral_legalizer_jit_matches_numpy_reference():
    for seed in range(5):
        rng = np.random.RandomState(seed)
        n = 36
        sizes = rng.uniform(2.0, 12.0, size=(n, 2)).astype(np.float64)
        pos = np.column_stack(
            [rng.uniform(-4.0, 124.0, size=n), rng.uniform(-4.0, 94.0, size=n)]
        ).astype(np.float64)
        movable = rng.uniform(size=n) > 0.15
        order = rng.permutation(n).tolist()
        actual = _legalize(pos, movable, sizes, order, True)
        expected = _legalize(pos, movable, sizes, order, False)
        np.testing.assert_array_equal(actual, expected)
