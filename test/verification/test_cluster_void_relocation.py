import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from placer.local_search.cluster_void_relocation import (
    _void_cluster_relocation,
    find_large_macro_voids,
)


class _Scorer:
    def __init__(self):
        self.grid_occupied = np.zeros(400, dtype=np.float64)
        self.dens_grid_area = 1.0
        self.committed = None

    def congestion_field(self):
        field = np.zeros((20, 20), dtype=np.float64)
        field[0:5, :] = 1.0
        return field

    def score_move_group(self, members, new_hard, soft_indices, new_soft):
        return 0.9

    def commit_move_group(self, members, new_hard, soft_indices, new_soft):
        self.committed = (
            np.asarray(members).copy(),
            np.asarray(new_hard).copy(),
            np.asarray(soft_indices).copy(),
            np.asarray(new_soft).copy(),
        )


def test_large_macro_voids_find_opposing_edge_corridor():
    hard = np.array([[2.0, 10.0], [18.0, 10.0], [10.0, 2.0]])
    hw = np.array([2.0, 2.0, 0.5])
    hh = np.array([4.0, 4.0, 0.5])

    voids = find_large_macro_voids(
        hard,
        hw,
        hh,
        large_area_percentile=70.0,
        min_width=2.0,
        min_height=2.0,
    )

    assert len(voids) == 1
    assert voids[0]["orientation"] == "horizontal"
    assert np.allclose(voids[0]["rect"], [4.0, 6.0, 16.0, 14.0])


def test_void_relocation_moves_whole_leaf_and_owned_soft_toward_graph_centroid():
    hard = np.array([[2.0, 10.0], [18.0, 10.0], [8.0, 2.0], [9.0, 2.0]])
    soft = np.array([[8.5, 2.5]])
    hw = np.array([2.0, 2.0, 0.5, 0.5])
    hh = np.array([4.0, 4.0, 0.5, 0.5])
    soft_hw = np.array([0.5])
    soft_hh = np.array([0.5])
    scorer = _Scorer()
    edges = [
        SimpleNamespace(src=1, dst=0, weight=1.0),
        SimpleNamespace(src=1, dst=2, weight=1.0),
    ]

    moved_hard, moved_soft, accepts, score = _void_cluster_relocation(
        hard.copy(),
        soft.copy(),
        hw,
        hh,
        soft_hw,
        soft_hh,
        20.0,
        20.0,
        4,
        SimpleNamespace(grid_rows=20, grid_cols=20),
        scorer,
        1.0,
        clusters={0: np.array([0]), 1: np.array([2, 3]), 2: np.array([1])},
        cluster_softs={1: np.array([4])},
        edges=edges,
        movable_h=np.array([False, False, True, True]),
        movable_soft=np.array([True]),
        candidate_allowed=lambda trial_hard, trial_soft: True,
        large_area_percentile=70.0,
        min_gap_cells=2,
        min_field_drop=0.01,
        min_gain=0.0001,
    )

    assert accepts == 1
    assert score == 0.9
    assert scorer.committed is not None
    assert np.allclose(np.mean(moved_hard[[2, 3]], axis=0), [10.0, 10.0])
    assert 4.0 <= moved_soft[0, 0] <= 16.0
    assert 6.0 <= moved_soft[0, 1] <= 14.0


def test_void_relocation_respects_hierarchy_gate():
    hard = np.array([[2.0, 10.0], [18.0, 10.0], [8.0, 2.0], [9.0, 2.0]])
    scorer = _Scorer()
    result = _void_cluster_relocation(
        hard.copy(),
        np.zeros((0, 2), dtype=np.float64),
        np.array([2.0, 2.0, 0.5, 0.5]),
        np.array([4.0, 4.0, 0.5, 0.5]),
        np.zeros(0),
        np.zeros(0),
        20.0,
        20.0,
        4,
        SimpleNamespace(grid_rows=20, grid_cols=20),
        scorer,
        1.0,
        clusters={0: np.array([0]), 1: np.array([2, 3]), 2: np.array([1])},
        cluster_softs={},
        edges=[SimpleNamespace(src=1, dst=0, weight=1.0)],
        movable_h=np.array([False, False, True, True]),
        movable_soft=np.zeros(0, dtype=bool),
        candidate_allowed=lambda trial_hard, trial_soft: False,
    )

    assert result[2] == 0
    assert scorer.committed is None
    assert _void_cluster_relocation.last_stats["hierarchy_rejects"] >= 1
