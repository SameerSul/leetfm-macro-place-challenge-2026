import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from placer.local_search.compound_relocation import _compound_soft_relocation


class _FakeScorer:
    def __init__(self, field):
        self._field = np.asarray(field, dtype=np.float64)
        self.scored = []
        self.committed = None

    def congestion_field(self):
        return self._field

    def score_move_soft_group(self, indices, targets):
        self.scored.append((np.asarray(indices).copy(), np.asarray(targets).copy()))
        return 0.9

    def commit_move_soft_group(self, indices, targets):
        self.committed = (np.asarray(indices).copy(), np.asarray(targets).copy())


def _run(candidate_allowed, *, max_scored=None):
    soft = np.array([[70.0, 60.0], [74.0, 62.0], [72.0, 66.0]], dtype=np.float64)
    original = soft.copy()
    half = np.full(3, 1.0, dtype=np.float64)
    region = np.tile(np.array([20.0, 20.0, 90.0, 90.0]), (3, 1))
    field = np.tile(np.array([0.0, 0.1, 0.8, 1.0]), (4, 1))
    scorer = _FakeScorer(field)
    benchmark = SimpleNamespace(grid_rows=4, grid_cols=4, name="synthetic")

    moved, accepts, score = _compound_soft_relocation(
        soft,
        half,
        half,
        100.0,
        100.0,
        5,
        benchmark,
        scorer,
        1.0,
        cluster_softs={3: np.array([5, 6, 7])},
        soft_movable=np.ones(3, dtype=bool),
        region_bbox=region,
        candidate_allowed=candidate_allowed,
        top_groups=1,
        group_size=3,
        n_anchors=1,
        shift_fractions=(1.0,),
        min_field_drop=0.01,
        min_gain=0.001,
        max_scored=max_scored,
    )
    return original, moved, accepts, score, scorer, region


def _run_with_low_role_evidence():
    soft = np.array([[70.0, 60.0], [74.0, 62.0], [72.0, 66.0]], dtype=np.float64)
    half = np.full(3, 1.0, dtype=np.float64)
    region = np.tile(np.array([20.0, 20.0, 90.0, 90.0]), (3, 1))
    scorer = _FakeScorer(np.tile(np.array([0.0, 0.1, 0.8, 1.0]), (4, 1)))
    benchmark = SimpleNamespace(grid_rows=4, grid_cols=4, name="synthetic")
    moved, accepts, score = _compound_soft_relocation(
        soft,
        half,
        half,
        100.0,
        100.0,
        5,
        benchmark,
        scorer,
        1.0,
        cluster_softs={3: np.array([5, 6, 7])},
        soft_role_evidence={index: {"confidence": "low", "role": "owned"} for index in range(3)},
        soft_movable=np.ones(3, dtype=bool),
        region_bbox=region,
    )
    return moved, accepts, score, scorer


def test_compound_soft_move_scores_and_commits_only_the_completed_group():
    allowed_trials = []
    original, moved, accepts, score, scorer, region = _run(
        lambda trial: allowed_trials.append(trial.copy()) or True
    )

    assert accepts == 1
    assert score == 0.9
    assert len(scorer.scored) >= 1
    assert scorer.committed is not None
    indices, targets = scorer.committed
    assert indices.size >= 2
    np.testing.assert_allclose(targets - targets[0], original[indices] - original[indices[0]])
    np.testing.assert_allclose(moved[indices], targets)
    assert np.all(targets[:, 0] >= region[indices, 0])
    assert np.all(targets[:, 0] <= region[indices, 2])
    assert allowed_trials


def test_compound_soft_move_applies_hierarchy_gate_before_exact_scoring():
    original, moved, accepts, score, scorer, _region = _run(lambda _trial: False)

    assert accepts == 0
    assert score == 1.0
    np.testing.assert_allclose(moved, original)
    assert scorer.scored == []
    assert scorer.committed is None
    assert _compound_soft_relocation.last_stats["hierarchy_rejects"] >= 1


def test_compound_soft_move_honors_zero_exact_score_quota():
    original, moved, accepts, score, scorer, _region = _run(
        lambda _trial: True,
        max_scored=0,
    )

    assert accepts == 0
    assert score == 1.0
    np.testing.assert_allclose(moved, original)
    assert scorer.scored == []
    assert scorer.committed is None
    assert _compound_soft_relocation.last_stats["quota_exhausted"] is True


def test_low_confidence_flat_ownership_does_not_form_a_compound_group():
    moved, accepts, score, scorer = _run_with_low_role_evidence()

    assert accepts == 0
    assert score == 1.0
    assert scorer.scored == []
    assert _compound_soft_relocation.last_stats["groups"] == 0
