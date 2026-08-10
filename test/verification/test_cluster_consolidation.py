import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from placer.local_search.cluster_consolidation import _small_cluster_consolidation


def _assembly_score(hard, soft):
    center = np.mean(hard[:2], axis=0)
    return float(
        np.mean(np.linalg.norm(hard[:2] - center, axis=1))
        + np.mean(np.linalg.norm(soft[:1] - center, axis=1))
    )


class _AssemblyScorer:
    def __init__(self, hard, soft):
        self.hard = hard.copy()
        self.soft = soft.copy()
        self.commits = 0

    def score_move_group(self, hard_indices, hard_xy, soft_indices, soft_xy):
        trial_hard = self.hard.copy()
        trial_soft = self.soft.copy()
        trial_hard[np.asarray(hard_indices, dtype=np.int64)] = hard_xy
        trial_soft[np.asarray(soft_indices, dtype=np.int64)] = soft_xy
        return _assembly_score(trial_hard, trial_soft)

    def commit_move_group(self, hard_indices, hard_xy, soft_indices, soft_xy):
        self.hard[np.asarray(hard_indices, dtype=np.int64)] = hard_xy
        self.soft[np.asarray(soft_indices, dtype=np.int64)] = soft_xy
        self.commits += 1


def test_small_cluster_consolidation_improves_structure_and_exact_score():
    hard = np.asarray([[4.0, 10.0], [10.0, 10.0], [17.0, 17.0]])
    soft = np.asarray([[13.0, 10.0]])
    scorer = _AssemblyScorer(hard, soft)
    initial = _assembly_score(hard, soft)
    hard_region = np.tile(np.asarray([0.5, 0.5, 19.5, 19.5]), (3, 1))
    soft_region = np.tile(np.asarray([0.5, 0.5, 19.5, 19.5]), (1, 1))

    new_hard, new_soft, accepts, score = _small_cluster_consolidation(
        hard.copy(),
        soft.copy(),
        np.full(3, 0.5),
        np.full(3, 0.5),
        np.full(1, 0.5),
        np.full(1, 0.5),
        20.0,
        20.0,
        3,
        scorer,
        initial,
        clusters={0: np.asarray([0, 1]), 1: np.asarray([2])},
        cluster_softs={0: np.asarray([3])},
        edges=(),
        movable_h=np.ones(3, dtype=bool),
        movable_soft=np.ones(1, dtype=bool),
        hard_region=hard_region,
        soft_region=soft_region,
        candidate_allowed=lambda _hard, _soft: True,
        structural_score_fn=_assembly_score,
        top_clusters=1,
        compact_scales=(0.8,),
        neighbor_shift_fractions=(0.0,),
        soft_scales=(0.5,),
        max_scored=4,
    )

    assert accepts == 1
    assert scorer.commits == 1
    assert score < initial
    assert _assembly_score(new_hard, new_soft) < initial
    assert np.linalg.norm(new_hard[0] - new_hard[1]) < np.linalg.norm(hard[0] - hard[1])
    assert np.linalg.norm(new_soft[0] - new_hard[:2].mean(axis=0)) < np.linalg.norm(
        soft[0] - hard[:2].mean(axis=0)
    )


def test_small_cluster_consolidation_exchanges_interleaved_leaf_slots():
    hard = np.asarray([[2.0, 10.0], [4.0, 10.0], [16.0, 10.0], [18.0, 10.0]])
    soft = np.asarray([[3.0, 10.0], [17.0, 10.0]])

    def slot_score(trial_hard, _trial_soft):
        left = np.mean(trial_hard[:2], axis=0)
        right = np.mean(trial_hard[2:], axis=0)
        return float(abs(left[0] - 17.0) + abs(right[0] - 3.0))

    class SlotScorer:
        def __init__(self):
            self.hard = hard.copy()
            self.soft = soft.copy()
            self.commits = 0

        def score_move_group(self, hard_indices, hard_xy, soft_indices, soft_xy):
            trial_hard = self.hard.copy()
            trial_soft = self.soft.copy()
            trial_hard[np.asarray(hard_indices, dtype=np.int64)] = hard_xy
            trial_soft[np.asarray(soft_indices, dtype=np.int64)] = soft_xy
            return slot_score(trial_hard, trial_soft)

        def commit_move_group(self, hard_indices, hard_xy, soft_indices, soft_xy):
            self.hard[np.asarray(hard_indices, dtype=np.int64)] = hard_xy
            self.soft[np.asarray(soft_indices, dtype=np.int64)] = soft_xy
            self.commits += 1

    scorer = SlotScorer()
    region = np.tile(np.asarray([0.5, 0.5, 19.5, 19.5]), (4, 1))
    soft_region = np.tile(np.asarray([0.5, 0.5, 19.5, 19.5]), (2, 1))
    new_hard, new_soft, accepts, score = _small_cluster_consolidation(
        hard.copy(),
        soft.copy(),
        np.full(4, 0.4),
        np.full(4, 0.4),
        np.full(2, 0.4),
        np.full(2, 0.4),
        20.0,
        20.0,
        4,
        scorer,
        slot_score(hard, soft),
        clusters={0: np.asarray([0, 1]), 1: np.asarray([2, 3])},
        cluster_softs={0: np.asarray([4]), 1: np.asarray([5])},
        edges=(),
        movable_h=np.ones(4, dtype=bool),
        movable_soft=np.ones(2, dtype=bool),
        hard_region=region,
        soft_region=soft_region,
        candidate_allowed=lambda _hard, _soft: True,
        structural_score_fn=slot_score,
        top_clusters=2,
        top_slot_clusters=2,
        compact_scales=(0.9,),
        neighbor_shift_fractions=(0.0,),
        soft_scales=(1.0,),
        max_scored=8,
    )

    assert accepts == 1
    assert scorer.commits == 1
    assert score == 0.0
    assert np.mean(new_hard[:2, 0]) == 17.0
    assert np.mean(new_hard[2:, 0]) == 3.0
    assert new_soft[:, 0].tolist() == [17.0, 3.0]
    assert _small_cluster_consolidation.last_stats["slot_accepts"] == 1
