"""Verify commit-scoped swap-tail baseline reuse and invalidation."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from placer.scoring.incremental import IncrementalScorer


def test_swap_tail_baseline_reuses_then_rebuilds_cached_state():
    scorer = IncrementalScorer.__new__(IncrementalScorer)
    scorer._swap_tail_baseline_cache = None
    scorer.V_smoothed = np.array([[1.0, 2.0]], dtype=np.float64)
    scorer.H_smoothed = np.array([[3.0, 4.0]], dtype=np.float64)
    scorer.V_macro_flat = np.array([0.5, 1.0], dtype=np.float64)
    scorer.H_macro_flat = np.array([1.5, 2.0], dtype=np.float64)
    scorer.grid_v_routes = 2.0
    scorer.grid_h_routes = 4.0
    scorer.grid_occupied = np.array([0.0, 5.0, 2.0], dtype=np.float64)

    first = scorer._swap_tail_baseline()
    second = scorer._swap_tail_baseline()

    assert second is first
    assert first["density_nonzero"] == 2
    assert first["density_sum"] == 7.0

    scorer.V_smoothed[0, 0] = 9.0
    scorer.grid_occupied[0] = 4.0
    assert scorer._swap_tail_baseline() is first

    scorer._invalidate_swap_tail_baseline()
    rebuilt = scorer._swap_tail_baseline()

    assert rebuilt is not first
    assert rebuilt["congestion"][0] == 9.25
    assert rebuilt["density_nonzero"] == 3
    assert rebuilt["density_sum"] == 11.0
