import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from placer.scoring.incremental import (
    IncrementalScorer,
    _changed_density_cost_jit,
    _group_density_cost_jit,
)


def test_group_density_tail_matches_full_partition_and_clears_scratch():
    rng = np.random.default_rng(903)
    for size in (10, 11, 100, 961):
        base = rng.integers(0, 8, size=size).astype(np.float64)
        order = np.argsort(base)[::-1].copy()
        count = size // 10
        mark = np.zeros(size, dtype=np.uint8)
        scratch = np.empty(size + count)
        for trial in range(24):
            values = base.copy()
            indices = rng.choice(size, size=trial % size, replace=False)
            values[indices] = rng.uniform(-1.0e-15, 12.0, size=indices.size)
            if trial == 0:
                values[:] = 0.0
            if trial == 1:
                values[:] = 0.0
                values[0] = -1.0e-15
            nonzero = values[values != 0.0]
            top = min(count, nonzero.size)
            expected = (
                (0.5 * np.partition(nonzero, nonzero.size - top)[-top:].sum() / 2.5 / count)
                if top
                else 0.0
            )
            actual = _changed_density_cost_jit(values, base, order, count, 2.5, mark, scratch)
            np.testing.assert_allclose(actual, expected, rtol=1.0e-14, atol=1.0e-15)
            assert not np.any(mark)


def test_group_density_rectangles_match_scalar_updates_without_mutating_occupancy():
    rng = np.random.default_rng(904)
    half_sizes = rng.uniform(0.1, 3.0, (6, 2))
    scorer = SimpleNamespace(
        _dens_half=dict(enumerate(half_sizes)),
        dens_grid_w=1.25,
        dens_grid_h=2.0,
        dens_grid_col=10,
        dens_grid_row=7,
        _dens_empty_idx=np.empty(0, dtype=np.int64),
        _dens_empty_area=np.empty(0),
    )
    base = rng.uniform(0.0, 12.0, 70)
    order = np.argsort(base)[::-1].copy()
    mark = np.zeros(70, dtype=np.uint8)
    scratch, candidates = np.empty(70), np.empty(77)
    touched = np.empty(70, dtype=np.int64)
    for _ in range(12):
        occupied = base.copy()
        occupied[::3] += 1.0e-13  # A cached order can precede scalar roundoff.
        before = occupied.copy()
        old_xy = rng.uniform(-5.0, 18.0, (6, 2))
        new_xy = rng.uniform(-5.0, 18.0, (6, 2))
        expected_grid = occupied.copy()
        for xy, operation in ((old_xy, np.subtract.at), (new_xy, np.add.at)):
            for module, (x, y) in enumerate(xy):
                indices, area = IncrementalScorer._macro_occ(scorer, module, x, y)
                operation(expected_grid, indices, area)
        expected = 0.5 * np.partition(expected_grid, 63)[63:].sum() / 2.5 / 7
        actual = _group_density_cost_jit(
            old_xy,
            new_xy,
            half_sizes,
            occupied,
            base,
            order,
            7,
            2.5,
            1.25,
            2.0,
            7,
            10,
            mark,
            scratch,
            touched,
            candidates,
        )
        np.testing.assert_array_equal(occupied, before)
        np.testing.assert_array_equal(scratch, expected_grid)
        np.testing.assert_allclose(actual, expected, rtol=1.0e-14)
        assert not np.any(mark)
