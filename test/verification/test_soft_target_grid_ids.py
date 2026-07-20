"""Parity checks for stable integer-grid soft target preparation."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from placer.local_search.relocation import (  # noqa: E402
    _dedupe_targets_xy,
    _point_in_region_mask,
    _soft_targets_from_grid_ids,
)


def _legacy_targets(ids, half_w, half_h, cw, ch, nr, nc, region_mask, allowed):
    cell_w = cw / nc
    cell_h = ch / nr
    targets = []
    hierarchy_rejects = 0
    for grid_id in ids:
        col = int(grid_id) % nc
        row = int(grid_id) // nc
        x = float(np.clip((col + 0.5) * cell_w, half_w, cw - half_w))
        y = float(np.clip((row + 0.5) * cell_h, half_h, ch - half_h))
        if not _point_in_region_mask(region_mask, x, y, cw, ch):
            continue
        if allowed is not None and not bool(allowed(x, y)):
            hierarchy_rejects += 1
            continue
        targets.append((x, y))
    before_dedup = len(targets)
    targets = _dedupe_targets_xy(targets)
    return targets, hierarchy_rejects, before_dedup - targets.shape[0]


def test_integer_grid_targets_match_stable_coordinate_deduplication():
    cases = [
        (3.0, 4.0, 100.0, 80.0, 8, 10),
        (17.0, 14.0, 100.0, 80.0, 8, 10),
        (49.0, 39.0, 100.0, 80.0, 8, 10),
        (50.0, 40.0, 100.0, 80.0, 8, 10),
        (4.0, 4.0, 9.0, 9.0, 1, 1),
    ]
    for half_w, half_h, cw, ch, nr, nc in cases:
        ids = np.asarray(list(range(nr * nc)) + [0, nr * nc - 1, 0], dtype=np.int64)
        region_mask = np.ones((nr, nc), dtype=bool)
        if region_mask.size > 1:
            region_mask.ravel()[1::5] = False
        allowed = lambda x, y: x + 0.5 * y < 0.92 * (cw + 0.5 * ch)
        expected, expected_hierarchy, expected_dedup = _legacy_targets(
            ids,
            half_w,
            half_h,
            cw,
            ch,
            nr,
            nc,
            region_mask,
            allowed,
        )
        stamps = np.zeros((nr + 2) * (nc + 2), dtype=np.int64)
        actual, hierarchy_rejects, dedup_rejects = _soft_targets_from_grid_ids(
            ids,
            half_w=half_w,
            half_h=half_h,
            cw=cw,
            ch=ch,
            nr=nr,
            nc=nc,
            stamps=stamps,
            generation=7,
            region_mask=region_mask,
            candidate_allowed=allowed,
        )
        assert np.array_equal(actual, expected)
        assert hierarchy_rejects == expected_hierarchy
        assert dedup_rejects == expected_dedup


def test_grid_target_stamps_are_generation_scoped():
    stamps = np.zeros((4 + 2) * (5 + 2), dtype=np.int64)
    kwargs = dict(
        half_w=2.0,
        half_h=2.0,
        cw=50.0,
        ch=40.0,
        nr=4,
        nc=5,
        stamps=stamps,
    )
    first, _, _ = _soft_targets_from_grid_ids(np.array([0, 0, 1]), generation=1, **kwargs)
    second, _, _ = _soft_targets_from_grid_ids(np.array([0, 1]), generation=2, **kwargs)
    assert first.shape[0] == 2
    assert np.array_equal(second, first)
