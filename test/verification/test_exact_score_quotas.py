import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from placer.local_search.hierarchy_swaps import (
    _new_stats,
    _region_bounded_swap_relief,
    _stable_exact_prefixes,
    _try_soft_soft,
)
from placer.shared.geometry import separation_matrices
from placer.local_search.relocation import _relocation_moves, _soft_relocation_moves
from utils import constants as const


class _MoveScorer:
    def __init__(self, hard, soft, sizes, field):
        self.committed_hard_pos = np.asarray(hard, dtype=np.float64).copy()
        self.committed_soft_pos = np.asarray(soft, dtype=np.float64).copy()
        self.benchmark = SimpleNamespace(macro_sizes=torch.tensor(sizes, dtype=torch.float32))
        self.num_soft = int(self.committed_soft_pos.shape[0])
        self._field = np.asarray(field, dtype=np.float64)
        self.scored = 0

    def congestion_field(self):
        return self._field

    def _prepare_move(self, index):
        return ("hard", int(index))

    def _prepare_move_soft(self, index):
        return ("soft", int(index))

    def _trial_many_at(self, _prep, targets):
        self.scored += int(len(targets))
        return np.full(len(targets), 2.0, dtype=np.float64)

    def _trial_many_at_soft(self, _prep, targets):
        self.scored += int(len(targets))
        return np.full(len(targets), 2.0, dtype=np.float64)

    def _revert_prep(self, _prep):
        return None

    def _revert_prep_soft(self, _prep):
        return None

    def wl_delta_move_soft(self, _index, _target):
        return 0.0


class _SwapScorer:
    def __init__(self, field):
        self._field = np.asarray(field, dtype=np.float64)
        self.scored = 0

    def congestion_field(self):
        return self._field

    def score_swap_hard_hard_many(self, _index, targets):
        self.scored += int(len(targets))
        return np.full(len(targets), 2.0, dtype=np.float64)

    def score_swap_hard_soft_many(self, _index, targets):
        self.scored += int(len(targets))
        return np.full(len(targets), 2.0, dtype=np.float64)

    def score_swap_soft_soft_many(self, _index, targets):
        self.scored += int(len(targets))
        return np.full(len(targets), 2.0, dtype=np.float64)


class _FirstSoftSwapWinsScorer:
    def __init__(self):
        self.batch_sizes = []
        self.commits = []

    def score_swap_soft_soft_many(self, index, targets):
        self.batch_sizes.append(int(len(targets)))
        scores = np.full(len(targets), 2.0, dtype=np.float64)
        if len(self.batch_sizes) == 1:
            scores[0] = 0.5
        return scores

    def commit_swap_soft_soft(self, index, target):
        self.commits.append((int(index), int(target)))


def test_hard_and_soft_relocation_stop_at_exact_score_quota():
    sizes = np.array([[10.0, 10.0], [4.0, 4.0]], dtype=np.float64)
    benchmark = SimpleNamespace(
        grid_rows=2,
        grid_cols=2,
        macro_sizes=torch.tensor(sizes, dtype=torch.float32),
    )
    hard = np.array([[75.0, 75.0]], dtype=np.float64)
    soft = np.array([[75.0, 75.0]], dtype=np.float64)
    scorer = _MoveScorer(hard, soft, sizes, [[0.0, 0.1], [0.2, 1.0]])

    _relocation_moves(
        hard.copy(),
        sizes[:1],
        sizes[:1, 0] / 2.0,
        sizes[:1, 1] / 2.0,
        100.0,
        100.0,
        np.array([True]),
        1,
        None,
        benchmark,
        scorer,
        1.0,
        top_hot=1,
        n_targets=4,
        max_scored=2,
    )
    assert _relocation_moves.last_stats["scored"] == 2
    assert _relocation_moves.last_stats["quota_exhausted"] is True

    scorer.scored = 0
    _soft_relocation_moves(
        soft.copy(),
        sizes[1:, 0] / 2.0,
        sizes[1:, 1] / 2.0,
        100.0,
        100.0,
        1,
        None,
        benchmark,
        scorer,
        1.0,
        top_hot=1,
        n_targets=4,
        max_scored=2,
    )
    assert _soft_relocation_moves.last_stats["scored"] == 2
    assert _soft_relocation_moves.last_stats["quota_exhausted"] is True
    assert scorer.scored == 2


def test_region_swaps_share_one_exact_score_quota_across_move_types(monkeypatch):
    hard = np.array([[75.0, 75.0], [25.0, 25.0]], dtype=np.float64)
    soft = np.array([[25.0, 75.0], [75.0, 25.0]], dtype=np.float64)
    sizes = np.full((2, 2), 10.0, dtype=np.float64)
    soft_sizes = np.full((2, 2), 4.0, dtype=np.float64)
    scorer = _SwapScorer([[0.0, 0.2], [0.5, 1.0]])
    benchmark = SimpleNamespace(grid_rows=2, grid_cols=2, name="quota_test")
    hard_separation = separation_matrices(sizes)
    monkeypatch.setattr(
        "placer.local_search.hierarchy_swaps.separation_matrices",
        lambda _sizes: (_ for _ in ()).throw(AssertionError("cache was not reused")),
    )

    _, _, _, _, stats = _region_bounded_swap_relief(
        hard.copy(),
        soft.copy(),
        sizes,
        sizes[:, 0] / 2.0,
        sizes[:, 1] / 2.0,
        soft_sizes[:, 0] / 2.0,
        soft_sizes[:, 1] / 2.0,
        100.0,
        100.0,
        np.ones(2, dtype=bool),
        np.ones(2, dtype=bool),
        benchmark,
        scorer,
        1.0,
        None,
        None,
        hard_k=4,
        soft_k=4,
        max_scored=2,
        hard_separation=hard_separation,
    )

    scored = int(stats["hh_scores"] + stats["hs_scores"] + stats["ss_scores"])
    exact_scored = int(
        stats["hh_exact_scores"] + stats["hs_exact_scores"] + stats["ss_exact_scores"]
    )
    assert scored == 2
    assert exact_scored == 2
    assert stats["hh_avoided_scores"] == 0
    assert stats["hs_avoided_scores"] == 0
    assert stats["ss_avoided_scores"] == 0
    assert scorer.scored == 2


def test_region_swap_exact_prefixes_preserve_stable_candidate_order():
    candidates = [(index, bool(index % 2)) for index in range(11)]

    chunks = list(_stable_exact_prefixes(candidates, 4))

    assert chunks == [candidates[:4], candidates[4:]]
    assert [candidate for chunk in chunks for candidate in chunk] == candidates
    assert list(_stable_exact_prefixes(candidates[:3], 4)) == [candidates[:3]]
    assert list(_stable_exact_prefixes([], 4)) == []


def test_soft_swap_prefix_winner_avoids_only_the_untouched_suffix():
    prefix = int(const.HIER_SWAP_SOFT_SOFT_EXACT_PREFIX)
    count = prefix + 4
    soft_pos = np.column_stack(
        [np.arange(count, dtype=np.float64) * 10.0 + 5.0, np.full(count, 5.0)]
    )
    scorer = _FirstSoftSwapWinsScorer()
    stats = _new_stats()

    accepts, best_score = _try_soft_soft(
        soft_pos,
        np.ones(count, dtype=np.float64),
        np.ones(count, dtype=np.float64),
        float(count * 10),
        10.0,
        np.ones(count, dtype=bool),
        scorer,
        1.0,
        np.arange(count, dtype=np.float64).reshape(1, count),
        None,
        count,
        0.0,
        0.0,
        1.0e-5,
        0.0,
        0.0,
        stats,
        None,
        "prefix_test",
        "density",
    )

    assert accepts == 1
    assert best_score == 0.5
    assert scorer.batch_sizes[0] == prefix
    assert len(scorer.commits) == 1
    assert stats["ss_avoided_scores"] == 3
    assert stats["ss_exact_scores"] == sum(scorer.batch_sizes)
