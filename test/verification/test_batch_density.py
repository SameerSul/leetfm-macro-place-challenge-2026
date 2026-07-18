import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from placer.scoring.incremental import _batch_density_costs


def _reference_density_costs(grids, density_count, grid_area):
    values = np.empty(grids.shape[0], dtype=np.float64)
    for index, occupied in enumerate(grids):
        nonzero = occupied[occupied != 0.0]
        if nonzero.size == 0:
            values[index] = 0.0
        elif grids.shape[1] < 10:
            values[index] = 0.5 * float(nonzero.mean() / grid_area)
        else:
            count = min(int(density_count), nonzero.size)
            top = np.partition(nonzero, nonzero.size - count)[nonzero.size - count :]
            values[index] = 0.5 * float(top.sum()) / grid_area / density_count
    return values


def test_numba_batched_density_reduction_matches_scalar_semantics():
    rng = np.random.default_rng(97)
    grids = rng.normal(size=(7, 31)).astype(np.float64)
    grids[np.abs(grids) < 0.45] = 0.0
    grids[0] = 0.0
    grids[1, :2] = [3.0, -1.0]

    expected = _reference_density_costs(grids, density_count=5, grid_area=2.75)
    actual = _batch_density_costs(grids, density_count=5, grid_area=2.75)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-14)


def test_numba_batched_density_reduction_handles_tiny_grids():
    grids = np.array(
        [
            [0.0, 2.0, 0.0, 4.0, 6.0, 0.0, 8.0, 0.0, 10.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    expected = _reference_density_costs(grids, density_count=0, grid_area=4.0)
    actual = _batch_density_costs(grids, density_count=0, grid_area=4.0)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-14)
