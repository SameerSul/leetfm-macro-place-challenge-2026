import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from placer.scoring.incremental import IncrementalScorer


def _reference_h(raw, c_lo, c_hi, capacity, window_lo, window_hi, window_count):
    sub = raw[:, c_lo : c_hi + 1] / capacity
    weighted = sub / window_count[:, None]
    prefix = np.empty((raw.shape[0] + 1, sub.shape[1]), dtype=np.float64)
    prefix[0, :] = 0.0
    np.cumsum(weighted, axis=0, out=prefix[1:, :])
    return prefix[window_hi + 1] - prefix[window_lo]


def _reference_v(raw, r_lo, r_hi, capacity, window_lo, window_hi, window_count):
    sub = raw[r_lo : r_hi + 1, :] / capacity
    weighted = sub / window_count[None, :]
    prefix = np.empty((sub.shape[0], raw.shape[1] + 1), dtype=np.float64)
    prefix[:, 0] = 0.0
    np.cumsum(weighted, axis=1, out=prefix[:, 1:])
    return prefix[:, window_hi + 1] - prefix[:, window_lo]


def test_numba_incremental_resmoothing_matches_vectorized_reference():
    rng = np.random.default_rng(23)
    rows, cols, radius = 11, 13, 2
    hard_raw = rng.normal(size=(rows, cols)).astype(np.float64)
    vert_raw = rng.normal(size=(rows, cols)).astype(np.float64)
    row_ids = np.arange(rows, dtype=np.int64)
    col_ids = np.arange(cols, dtype=np.int64)

    scorer = IncrementalScorer.__new__(IncrementalScorer)
    scorer.grid_row = rows
    scorer.grid_col = cols
    scorer.grid_h_routes = 7.25
    scorer.grid_v_routes = 9.5
    scorer.smooth_range = radius
    scorer.H_flat = hard_raw.ravel().copy()
    scorer.V_flat = vert_raw.ravel().copy()
    scorer.H_smoothed = np.full((rows, cols), np.nan, dtype=np.float64)
    scorer.V_smoothed = np.full((rows, cols), np.nan, dtype=np.float64)
    scorer._sm_row_lp = np.maximum(row_ids - radius, 0)
    scorer._sm_row_up = np.minimum(row_ids + radius, rows - 1)
    scorer._sm_row_cnt = (scorer._sm_row_up - scorer._sm_row_lp + 1).astype(np.float64)
    scorer._sm_col_lp = np.maximum(col_ids - radius, 0)
    scorer._sm_col_up = np.minimum(col_ids + radius, cols - 1)
    scorer._sm_col_cnt = (scorer._sm_col_up - scorer._sm_col_lp + 1).astype(np.float64)
    scorer._resmooth_h_prefix = np.empty(rows + 1, dtype=np.float64)
    scorer._resmooth_v_prefix = np.empty(cols + 1, dtype=np.float64)

    r_lo, r_hi, c_lo, c_hi = 3, 8, 4, 10
    scorer._resmooth_bbox(r_lo, r_hi, c_lo, c_hi)

    expected_h = _reference_h(
        hard_raw,
        c_lo,
        c_hi,
        scorer.grid_h_routes,
        scorer._sm_row_lp,
        scorer._sm_row_up,
        scorer._sm_row_cnt,
    )
    expected_v = _reference_v(
        vert_raw,
        r_lo,
        r_hi,
        scorer.grid_v_routes,
        scorer._sm_col_lp,
        scorer._sm_col_up,
        scorer._sm_col_cnt,
    )
    np.testing.assert_array_equal(scorer.H_smoothed[:, c_lo : c_hi + 1], expected_h)
    np.testing.assert_array_equal(scorer.V_smoothed[r_lo : r_hi + 1, :], expected_v)
    assert np.isnan(scorer.H_smoothed[:, :c_lo]).all()
    assert np.isnan(scorer.V_smoothed[:r_lo, :]).all()
