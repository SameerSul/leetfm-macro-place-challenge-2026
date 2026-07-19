import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from placer.scoring.incremental import _batch_congestion_costs


@pytest.mark.parametrize("cells", [8, 800, 4410])
def test_batch_congestion_inplace_reduction_matches_copying_reference(cells):
    rng = np.random.default_rng(73 + cells)
    values = rng.integers(0, 32, size=(12, cells)).astype(np.float64) / 7.0
    count = int(cells * 0.05)
    if count == 0:
        expected = values.max(axis=1)
    else:
        expected = np.partition(values, cells - count, axis=1)[:, -count:].sum(axis=1) / count

    working = values.copy()
    actual = _batch_congestion_costs(working)

    assert np.array_equal(actual, expected)
    if count:
        assert not np.array_equal(working, values)
