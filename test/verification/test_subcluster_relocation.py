import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import placer.local_search.subcluster_relocation as relocation_module
from placer.local_search.clusters import compute_region_bbox
from placer.local_search.subcluster_relocation import (
    _deep_cluster_internal_relief,
    _deep_cluster_margin_fractions,
    _subcluster_relocation,
)


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


def test_deep_cluster_margin_blends_graph_congestion_and_density_pressure():
    margins = _deep_cluster_margin_fractions(
        {0: np.asarray([0, 1]), 1: np.asarray([2, 3])},
        {0: 1.0, 1: 1.0},
        {0: 1.0, 1: 1.0},
        {0: 0.0, 1: 1.0},
        base_margin=0.01,
        extra_margin=0.03,
        congestion_weight=0.45,
        density_weight=0.35,
        graph_weight=0.20,
    )

    assert margins[0] >= 0.01
    assert margins[1] > margins[0]
    assert margins[1] <= 0.04 + 1.0e-12


def test_cluster_specific_margin_changes_only_its_child_box():
    hard = np.asarray([[2.0, 2.0], [3.0, 2.0], [7.0, 2.0], [8.0, 2.0]])
    sizes = np.ones((4, 2), dtype=np.float64)
    hw = np.full(4, 0.5)
    hh = np.full(4, 0.5)
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    clusters = {0: np.asarray([0, 1]), 1: np.asarray([2, 3])}

    region = compute_region_bbox(
        hard,
        sizes,
        hw,
        hh,
        10.0,
        5.0,
        4,
        labels,
        clusters,
        margin=0.0,
        singleton_window=0.0,
        cluster_margins={0: 0.01, 1: 0.05},
    )

    child_0_width = region[0, 2] - region[0, 0]
    child_1_width = region[2, 2] - region[2, 0]
    assert child_1_width > child_0_width
    assert np.all(region[:, 0] <= hard[:, 0])
    assert np.all(region[:, 2] >= hard[:, 0])


def test_deep_internal_relief_uses_graph_priority_and_enforces_both_swap_boxes(monkeypatch):
    field = np.ones((4, 10), dtype=np.float64)
    monkeypatch.setattr(
        relocation_module,
        "_congestion_field",
        lambda _scorer, _rows, _cols: field,
    )
    monkeypatch.setattr(
        relocation_module,
        "_density_field",
        lambda _scorer, _rows, _cols: field,
    )
    called_hard_masks = []
    checked_swap_quality = []

    def fake_hard_relocation(
        pos,
        _sizes,
        _hw,
        _hh,
        _cw,
        _ch,
        movable,
        _n,
        _plc,
        _benchmark,
        _scorer,
        initial_score,
        *,
        candidate_allowed,
        **kwargs,
    ):
        selected = np.flatnonzero(movable)
        called_hard_masks.append(selected.copy())
        index = int(selected[0])
        assert candidate_allowed(index, float(pos[index, 0]), float(pos[index, 1]))
        fake_hard_relocation.last_stats = {
            "candidates": 1,
            "legal": 0,
            "scored": 0,
            "hierarchy_rejects": 0,
        }
        return pos, 0, float(initial_score)

    fake_hard_relocation.last_stats = {}

    def fake_soft_relocation(
        pos,
        _soft_hw,
        _soft_hh,
        _cw,
        _ch,
        _n,
        _plc,
        _benchmark,
        _scorer,
        initial_score,
        **kwargs,
    ):
        fake_soft_relocation.last_stats = {
            "candidates": 0,
            "legal": 0,
            "scored": 0,
            "hierarchy_rejects": 0,
        }
        return pos, 0, float(initial_score)

    fake_soft_relocation.last_stats = {}

    def fake_swap(
        hard_pos,
        soft_pos,
        _sizes,
        _hw,
        _hh,
        _soft_hw,
        _soft_hh,
        _cw,
        _ch,
        movable_h,
        _soft_movable,
        _benchmark,
        _scorer,
        initial_score,
        _hard_region,
        _soft_region,
        *,
        hierarchy_quality_fn,
        **kwargs,
    ):
        selected = np.flatnonzero(movable_h)
        trial = hard_pos.copy()
        trial[selected] = trial[selected[::-1]]
        checked_swap_quality.append(float(hierarchy_quality_fn(trial)))
        return hard_pos, soft_pos, 0, float(initial_score), {"hh_scores": 0}

    monkeypatch.setattr(relocation_module, "_relocation_moves", fake_hard_relocation)
    monkeypatch.setattr(relocation_module, "_soft_relocation_moves", fake_soft_relocation)
    monkeypatch.setattr(relocation_module, "_region_bounded_swap_relief", fake_swap)

    hard = np.asarray([[1.0, 1.0], [2.0, 1.0], [7.0, 3.0], [9.0, 3.0]])
    soft = np.empty((0, 2), dtype=np.float64)
    hard_region = np.asarray(
        [
            [0.5, 0.5, 2.5, 1.5],
            [0.5, 0.5, 2.5, 1.5],
            [6.5, 2.5, 7.5, 3.5],
            [8.5, 2.5, 9.5, 3.5],
        ]
    )
    soft_region = np.empty((0, 4), dtype=np.float64)
    result = _deep_cluster_internal_relief(
        hard,
        soft,
        np.full((4, 2), 0.4),
        np.full(4, 0.2),
        np.full(4, 0.2),
        np.empty(0),
        np.empty(0),
        10.0,
        4.0,
        4,
        object(),
        SimpleNamespace(grid_rows=4, grid_cols=10),
        object(),
        10.0,
        child_clusters={0: np.asarray([0, 1]), 1: np.asarray([2, 3])},
        cluster_softs={},
        subcluster_labels=np.asarray([0, 0, 1, 1]),
        graph_edges=[],
        graph_confidence={},
        graph_tension={0: 0.0, 1: 1.0},
        seed_hard_xy=hard.copy(),
        movable_h=np.ones(4, dtype=bool),
        soft_movable=np.empty(0, dtype=bool),
        hard_region=hard_region,
        soft_region=soft_region,
        candidate_allowed=lambda _hard, _soft: True,
        top_children=1,
        max_scored=8,
    )

    assert result[2] == 0
    assert called_hard_masks
    assert all(np.array_equal(mask, np.asarray([2, 3])) for mask in called_hard_masks)
    assert checked_swap_quality == [1.0]
    assert _deep_cluster_internal_relief.last_stats["graph_prioritized_children"] == 1
