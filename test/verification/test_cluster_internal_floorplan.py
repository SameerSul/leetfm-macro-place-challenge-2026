import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from placer.local_search.cluster_internal_floorplan import (
    _cluster_topologies,
    _owned_soft_targets,
    _topology_aware_cluster_floorplan,
    _topology_targets,
)


class _FakePlc:
    hard_macro_indices = [10, 11, 12, 13]
    soft_macro_indices = [20]

    def __init__(self):
        # Internal 0-1 and 1-2 nets, external 2-3 net, and owned soft 0-4 net.
        refs = np.asarray([10, 11, 11, 12, 12, 13, 10, 20], dtype=np.int64)
        starts = np.asarray([0, 2, 4, 6], dtype=np.int64)
        lengths = np.asarray([2, 2, 2, 2], dtype=np.int64)
        self._wl_vec_cache = {
            "ref_idx": refs,
            "net_starts": starts,
            "net_lengths": lengths,
            "net_weights": np.asarray([4.0, 2.0, 6.0, 3.0]),
        }


def _fixture():
    hard = np.asarray([[3.0, 5.0], [5.0, 5.0], [7.0, 5.0], [17.0, 5.0]])
    soft = np.asarray([[3.0, 7.0]])
    labels = np.asarray([0, 0, 0, 1], dtype=np.int64)
    clusters = {0: np.asarray([0, 1, 2]), 1: np.asarray([3])}
    cluster_softs = {0: np.asarray([4])}
    topology = _cluster_topologies(_FakePlc(), hard, soft, labels, clusters, cluster_softs, 4)[0]
    return hard, soft, labels, clusters, cluster_softs, topology


def test_cluster_topology_separates_internal_boundary_and_owned_soft_demand():
    hard, _soft, _labels, _clusters, _cluster_softs, topology = _fixture()

    assert topology.internal[0, 1] == 4.0
    assert topology.internal[1, 2] == 2.0
    assert topology.external_weight.tolist() == [0.0, 0.0, 6.0]
    assert np.allclose(topology.external_sum[2] / topology.external_weight[2], hard[3])
    assert topology.soft_hard[0, 0] == 3.0
    assert topology.boundary_ratio[2] > topology.boundary_ratio[0]


def test_topology_targets_put_external_macro_on_facing_boundary_with_channel():
    hard, _soft, _labels, _clusters, _cluster_softs, topology = _fixture()
    hw = np.full(4, 0.4)
    hh = np.full(4, 0.4)
    region = np.tile(np.asarray([0.5, 0.5, 14.5, 9.5]), (4, 1))

    targets = _topology_targets(
        topology,
        hard,
        hw,
        hh,
        region,
        utilization=0.55,
        boundary_threshold=0.35,
        boundary_channel_frac=0.04,
        transform="identity",
    )

    assert targets.shape == (3, 2)
    assert targets[2, 0] > np.mean(targets[:2, 0])
    assert targets[2, 0] < region[2, 2]


def test_owned_soft_uses_connected_hard_barycenter():
    hard, soft, _labels, _clusters, _cluster_softs, topology = _fixture()
    moved = hard[:3].copy()
    moved[0] = [10.0, 6.0]
    soft_region = np.asarray([[0.5, 0.5, 19.5, 9.5]])

    target = _owned_soft_targets(topology, hard[:3], moved, soft, soft_region)

    assert target.shape == (1, 2)
    assert np.linalg.norm(target[0] - moved[0]) < np.linalg.norm(soft[0] - moved[0])


class _AcceptingScorer:
    def __init__(self):
        self.commits = 0

    def score_move_group(self, *_args):
        return 0.5

    def commit_move_group(self, *_args):
        self.commits += 1


def test_topology_floorplan_exact_gates_and_commits_complete_cluster_state():
    hard, soft, labels, clusters, cluster_softs, _topology = _fixture()
    scorer = _AcceptingScorer()
    hard_region = np.tile(np.asarray([0.5, 0.5, 14.5, 9.5]), (4, 1))
    soft_region = np.asarray([[0.5, 0.5, 14.5, 9.5]])

    new_hard, new_soft, accepts, score = _topology_aware_cluster_floorplan(
        _FakePlc(),
        hard.copy(),
        soft.copy(),
        np.full(4, 0.25),
        np.full(4, 0.25),
        20.0,
        10.0,
        4,
        scorer,
        1.0,
        labels=labels,
        clusters=clusters,
        cluster_softs=cluster_softs,
        movable_h=np.ones(4, dtype=bool),
        movable_soft=np.ones(1, dtype=bool),
        hard_region=hard_region,
        soft_region=soft_region,
        candidate_allowed=lambda _hard, _soft: True,
        top_clusters=1,
        utilization_variants=(0.55,),
        transforms=("identity",),
        max_scored=1,
    )

    assert accepts == 1
    assert score == 0.5
    assert scorer.commits == 1
    assert not np.allclose(new_hard[:3], hard[:3])
    assert not np.allclose(new_soft, soft)
