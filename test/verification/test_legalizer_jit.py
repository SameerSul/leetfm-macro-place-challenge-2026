"""Parity checks for Stage 3 legalization and synthetic-clearance kernels."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from placer.legalize import spiral  # noqa: E402
from placer.pipeline.segments import floorplan_seed  # noqa: E402


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


def _clearance_reference(hard, eligible, temp_hw, temp_hh):
    delta = np.zeros_like(hard)
    for i in range(len(hard)):
        for j in range(i + 1, len(hard)):
            move_i = bool(eligible[i])
            move_j = bool(eligible[j])
            if not (move_i or move_j):
                continue
            dx = float(hard[i, 0] - hard[j, 0])
            dy = float(hard[i, 1] - hard[j, 1])
            overlap_x = float(temp_hw[i] + temp_hw[j] - abs(dx))
            overlap_y = float(temp_hh[i] + temp_hh[j] - abs(dy))
            if overlap_x <= 0.0 or overlap_y <= 0.0:
                continue
            if overlap_x <= overlap_y:
                push = np.array(
                    [0.5 * overlap_x * (1.0 if dx >= 0.0 else -1.0), 0.0],
                    dtype=np.float64,
                )
            else:
                push = np.array(
                    [0.0, 0.5 * overlap_y * (1.0 if dy >= 0.0 else -1.0)],
                    dtype=np.float64,
                )
            if move_i and move_j:
                delta[i] += push
                delta[j] -= push
            elif move_i:
                delta[i] += 2.0 * push
            else:
                delta[j] -= 2.0 * push
    return delta


def test_synthetic_clearance_jit_matches_scalar_reference():
    rng = np.random.RandomState(73)
    hard = rng.uniform(0.0, 40.0, size=(80, 2)).astype(np.float64)
    half_sizes = rng.uniform(1.0, 8.0, size=(80, 2)).astype(np.float64)
    eligible = rng.uniform(size=80) > 0.35
    expected = _clearance_reference(hard, eligible, half_sizes[:, 0], half_sizes[:, 1])
    actual = np.empty_like(hard)
    floorplan_seed._synthetic_clearance_delta_jit(
        hard, eligible, half_sizes[:, 0], half_sizes[:, 1], actual
    )
    np.testing.assert_array_equal(actual, expected)
