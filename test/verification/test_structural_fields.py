import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from placer.local_search.structural_fields import (
    combined_structural_penalty,
    edge_keepout_penalty,
    grid_alignment_penalty,
    notch_penalty,
)


def test_edge_keepout_penalty_prefers_interior_spacing():
    sizes = np.array([[10.0, 10.0], [10.0, 10.0]])
    edge = np.array([[6.0, 50.0], [50.0, 50.0]])
    interior = np.array([[20.0, 50.0], [50.0, 50.0]])

    assert edge_keepout_penalty(edge, sizes, 100.0, 100.0, keepout=10.0) > edge_keepout_penalty(
        interior, sizes, 100.0, 100.0, keepout=10.0
    )


def test_grid_alignment_penalty_prefers_cell_centers():
    sizes = np.array([[4.0, 4.0], [4.0, 4.0]])
    aligned = np.array([[5.0, 5.0], [15.0, 15.0]])
    off_grid = np.array([[9.5, 5.0], [15.0, 10.5]])

    assert grid_alignment_penalty(aligned, sizes, 20.0, 20.0, grid_cols=2, grid_rows=2) == 0.0
    assert grid_alignment_penalty(
        off_grid, sizes, 20.0, 20.0, grid_cols=2, grid_rows=2
    ) > 0.0


def test_notch_penalty_detects_narrow_channels():
    sizes = np.array([[20.0, 20.0], [20.0, 20.0], [10.0, 10.0]])
    narrow = np.array([[20.0, 50.0], [43.0, 50.0], [80.0, 80.0]])
    wide = np.array([[20.0, 50.0], [60.0, 50.0], [80.0, 80.0]])

    assert notch_penalty(narrow, sizes, 100.0, 100.0, notch_window=8.0) > notch_penalty(
        wide, sizes, 100.0, 100.0, notch_window=8.0
    )


def test_combined_structural_penalty_is_weighted_sum_directionally():
    sizes = np.array([[10.0, 10.0], [10.0, 10.0]])
    better = np.array([[37.5, 37.5], [62.5, 62.5]])
    worse = np.array([[6.0, 6.0], [58.0, 50.0]])

    assert combined_structural_penalty(
        worse,
        sizes,
        100.0,
        100.0,
        grid_cols=4,
        grid_rows=4,
        keepout=10.0,
        notch_window=8.0,
    ) > combined_structural_penalty(
        better,
        sizes,
        100.0,
        100.0,
        grid_cols=4,
        grid_rows=4,
        keepout=10.0,
        notch_window=8.0,
    )
