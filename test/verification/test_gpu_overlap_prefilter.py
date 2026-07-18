"""Parity checks for the retained diagnostic CUDA swap-legality prefilter."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from placer.local_search import hierarchy_swaps
from placer.shared.geometry import separation_matrices
from utils.config import _GPU_BACKEND

pytestmark = pytest.mark.skipif(_GPU_BACKEND != "cuda", reason="requires CUDA")


def _hard_fixture():
    sizes = np.array(
        [[8.0, 6.0], [6.0, 10.0], [7.0, 7.0], [4.0, 5.0], [9.0, 4.0], [5.0, 8.0]],
        dtype=np.float64,
    )
    hard_pos = np.array(
        [[10.0, 10.0], [28.0, 14.0], [46.0, 31.0], [64.0, 20.0], [79.0, 44.0], [34.0, 58.0]],
        dtype=np.float64,
    )
    return hard_pos, sizes, sizes[:, 0] / 2.0, sizes[:, 1] / 2.0


def test_cuda_hard_hard_prefilter_matches_numba_legality(monkeypatch):
    hard_pos, sizes, hw, hh = _hard_fixture()
    sep_x, sep_y = separation_matrices(sizes)
    cand = np.array([1, 2, 3, 4, 5], dtype=np.int64)

    monkeypatch.delenv("HIER_GPU_EXPERIMENT", raising=False)
    expected = hierarchy_swaps._legal_hard_hard_candidates(
        hard_pos, sep_x, sep_y, hw, hh, 100.0, 100.0, 0, cand
    )

    monkeypatch.setenv("HIER_GPU_EXPERIMENT", "overlap_prefilter")
    monkeypatch.setattr(hierarchy_swaps.const, "HIER_GPU_OVERLAP_PREFILTER_MIN_CANDIDATES", 1)
    actual = hierarchy_swaps._legal_hard_hard_candidates(
        hard_pos, sep_x, sep_y, hw, hh, 100.0, 100.0, 0, cand
    )

    np.testing.assert_array_equal(actual, expected)


def test_cuda_hard_soft_prefilter_matches_numba_legality(monkeypatch):
    hard_pos, sizes, hw, hh = _hard_fixture()
    soft_pos = np.array(
        [[14.0, 71.0], [31.0, 14.0], [51.0, 48.0], [70.0, 76.0], [92.0, 25.0]],
        dtype=np.float64,
    )
    soft_hw = np.array([3.0, 7.0, 4.0, 3.0, 4.0], dtype=np.float64)
    soft_hh = np.array([4.0, 3.0, 4.0, 6.0, 4.0], dtype=np.float64)
    sep_x, sep_y = separation_matrices(sizes)
    cand = np.arange(soft_pos.shape[0], dtype=np.int64)

    monkeypatch.delenv("HIER_GPU_EXPERIMENT", raising=False)
    expected = hierarchy_swaps._legal_hard_soft_candidates(
        hard_pos,
        soft_pos,
        sep_x,
        sep_y,
        hw,
        hh,
        soft_hw,
        soft_hh,
        100.0,
        100.0,
        0,
        cand,
    )

    monkeypatch.setenv("HIER_GPU_EXPERIMENT", "overlap_prefilter")
    monkeypatch.setattr(hierarchy_swaps.const, "HIER_GPU_OVERLAP_PREFILTER_MIN_CANDIDATES", 1)
    actual = hierarchy_swaps._legal_hard_soft_candidates(
        hard_pos,
        soft_pos,
        sep_x,
        sep_y,
        hw,
        hh,
        soft_hw,
        soft_hh,
        100.0,
        100.0,
        0,
        cand,
    )

    np.testing.assert_array_equal(actual, expected)
