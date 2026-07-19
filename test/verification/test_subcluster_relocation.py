import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import placer.local_search.subcluster_relocation as relocation_module
from placer.local_search.subcluster_relocation import _subcluster_relocation


class _GroupScorer:
    def __init__(self):
        self.scored = []
        self.committed = None

    def score_move_group(self, hard_indices, hard_xy, soft_indices, soft_xy):
        self.scored.append(
            (
                np.asarray(hard_indices).copy(),
                np.asarray(hard_xy).copy(),
                np.asarray(soft_indices).copy(),
                np.asarray(soft_xy).copy(),
            )
        )
        return 9.0

    def commit_move_group(self, hard_indices, hard_xy, soft_indices, soft_xy):
        self.committed = (
            np.asarray(hard_indices).copy(),
            np.asarray(hard_xy).copy(),
            np.asarray(soft_indices).copy(),
            np.asarray(soft_xy).copy(),
        )


def _run(
    monkeypatch,
    *,
    candidate_allowed=lambda _hard, _soft: True,
    max_scored=8,
    top_swaps=0,
):
    field = np.zeros((4, 10), dtype=np.float64)
    field[:, :3] = 1.0
    monkeypatch.setattr(
        relocation_module,
        "weighted_congestion_field",
        lambda _scorer, _rows, _cols: field,
    )
    hard = np.asarray([[1.0, 1.0], [2.0, 1.0], [8.0, 3.0], [9.0, 3.0]])
    soft = np.asarray([[1.5, 1.5]])
    hw = np.full(4, 0.2)
    hh = np.full(4, 0.2)
    soft_hw = np.asarray([0.1])
    soft_hh = np.asarray([0.1])
    hard_region = np.tile(np.asarray([0.2, 0.2, 9.8, 3.8]), (4, 1))
    soft_region = np.tile(np.asarray([0.1, 0.1, 9.9, 3.9]), (1, 1))
    scorer = _GroupScorer()
    old_hard = hard.copy()
    old_soft = soft.copy()

    result = _subcluster_relocation(
        hard,
        soft,
        hw,
        hh,
        soft_hw,
        soft_hh,
        10.0,
        4.0,
        4,
        SimpleNamespace(grid_rows=4, grid_cols=10),
        scorer,
        10.0,
        child_clusters={0: np.asarray([0, 1]), 1: np.asarray([2, 3])},
        parent_clusters={0: np.arange(4)},
        parent_children={0: (0, 1)},
        cluster_softs={0: np.asarray([4])},
        movable_h=np.ones(4, dtype=bool),
        soft_movable=np.ones(1, dtype=bool),
        hard_parent_region=hard_region,
        soft_parent_region=soft_region,
        candidate_allowed=candidate_allowed,
        top_children=1,
        top_swaps=top_swaps,
        n_anchors=1,
        shift_fractions=(1.0,),
        min_field_drop=0.1,
        max_scored=max_scored,
    )
    return result, scorer, old_hard, old_soft


def test_child_relocation_moves_owned_soft_rigidly_and_commits_once(monkeypatch):
    (hard, soft, accepts, score), scorer, old_hard, old_soft = _run(monkeypatch)

    assert accepts == 1
    assert score == 9.0
    assert scorer.committed is not None
    assert np.allclose(hard[1] - hard[0], old_hard[1] - old_hard[0])
    assert np.allclose(soft[0] - hard[0], old_soft[0] - old_hard[0])
    assert np.array_equal(hard[2:], old_hard[2:])
    assert _subcluster_relocation.last_stats["scored"] >= 1


def test_child_relocation_checks_complete_hierarchy_before_exact_score(monkeypatch):
    (hard, soft, accepts, score), scorer, old_hard, old_soft = _run(
        monkeypatch,
        candidate_allowed=lambda _hard, _soft: False,
    )

    assert accepts == 0
    assert score == 10.0
    assert not scorer.scored
    assert np.array_equal(hard, old_hard)
    assert np.array_equal(soft, old_soft)
    assert _subcluster_relocation.last_stats["hierarchy_rejects"] >= 1


def test_child_relocation_honors_zero_exact_score_quota(monkeypatch):
    (_hard, _soft, accepts, score), scorer, _old_hard, _old_soft = _run(
        monkeypatch,
        max_scored=0,
    )

    assert accepts == 0
    assert score == 10.0
    assert not scorer.scored
    assert _subcluster_relocation.last_stats["quota_exhausted"] is True


def test_sibling_slot_swap_is_exact_scored_as_one_complete_state(monkeypatch):
    (_hard, _soft, accepts, score), scorer, _old_hard, _old_soft = _run(
        monkeypatch,
        max_scored=16,
        top_swaps=1,
    )

    assert accepts == 1
    assert score == 9.0
    assert scorer.committed is not None
    assert _subcluster_relocation.last_stats["swap_candidates"] == 1
    assert _subcluster_relocation.last_stats["swap_legal"] == 1
    assert _subcluster_relocation.last_stats["swap_scored"] == 1
