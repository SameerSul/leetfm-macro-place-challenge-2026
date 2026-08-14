import sys
from pathlib import Path

import numpy as np
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from placer.local_search.adjacent_cluster_transfer import (
    _restore_hierarchy,
    _snapshot_hierarchy,
    _transfer_ownership,
    run_adjacent_cluster_transfer,
)
import placer.local_search.adjacent_cluster_transfer as transfer_module
from test_location_graph import _fixture


def test_transfer_ownership_updates_hierarchy_and_persistent_graph_transactionally():
    graph, hierarchy, positions = _fixture()
    snapshot = _snapshot_hierarchy(hierarchy)

    _transfer_ownership(hierarchy, {1: 1, 3: 1}, n_hard=3)

    assert hierarchy.labels.tolist() == [0, 1, 1]
    assert np.array_equal(hierarchy.clusters[0], [0])
    assert np.array_equal(hierarchy.clusters[1], [1, 2])
    assert hierarchy.cluster_softs[0].size == 0
    assert np.array_equal(hierarchy.cluster_softs[1], [3])
    assert graph.cluster_of(1) == 1
    assert graph.cluster_of(3) == 1
    assert graph.clusters[0].hard_members == (0,)
    assert graph.clusters[1].hard_members == (1, 2)
    assert graph.active_edges is hierarchy.edges
    assert [(edge.src, edge.dst, edge.weight) for edge in hierarchy.edges] == [(0, 1, 2.0)]

    _restore_hierarchy(
        hierarchy,
        snapshot,
        positions[:3],
        positions[3:],
        changed_indices=(1, 3),
    )

    assert hierarchy.labels.tolist() == [0, 0, 1]
    assert np.array_equal(hierarchy.clusters[0], [0, 1])
    assert np.array_equal(hierarchy.clusters[1], [2])
    assert np.array_equal(hierarchy.cluster_softs[0], [3])
    assert graph.cluster_of(1) == 0
    assert graph.cluster_of(3) == 0
    assert [(edge.src, edge.dst, edge.weight) for edge in hierarchy.edges] == [(0, 1, 1.0)]


def test_soft_ownership_transfer_does_not_change_hard_hierarchy_edges():
    graph, hierarchy, _positions = _fixture()
    before = list(hierarchy.edges)

    _transfer_ownership(hierarchy, {3: 1}, n_hard=3)

    assert hierarchy.edges == before
    assert graph.active_edges is hierarchy.edges


def test_adjacent_transfer_commits_proxy_and_density_congestion_winner(monkeypatch):
    graph, hierarchy, positions = _fixture()
    hard, soft = positions[:3].copy(), positions[3:].copy()

    class _Scorer:
        def __init__(self, plc, benchmark, placement):
            self.plc = plc
            self.grid_row = 2
            self.grid_col = 2
            self.grid_occupied = np.zeros(4)
            self.dens_grid_area = 1.0

    proposal = {
        "kind": "hard_hard_swap",
        "moves": ((1, hard[2].copy()), (2, hard[1].copy())),
        "assignments": {1: 1, 2: 0},
        "rank": -1.0,
    }

    monkeypatch.setattr(transfer_module, "IncrementalScorer", _Scorer)
    monkeypatch.setattr(
        transfer_module,
        "_field_values",
        lambda scorer, placement, hard: np.ones(placement.shape[0]),
    )
    monkeypatch.setattr(
        transfer_module,
        "weighted_congestion_field",
        lambda scorer, nr, nc: np.zeros((nr, nc)),
    )
    monkeypatch.setattr(
        transfer_module,
        "_generate_intercluster_proposals",
        lambda *args, **kwargs: [proposal],
    )
    intra_calls = []

    def intra_proposals(*args, **kwargs):
        intra_calls.append(True)
        return []

    monkeypatch.setattr(transfer_module, "_generate_intracluster_proposals", intra_proposals)

    def components(placement, benchmark, plc):
        moved = bool(np.allclose(np.asarray(placement)[1], hard[2]))
        return {
            "wirelength": 0.2,
            "density": 0.4 if moved else 1.0,
            "congestion": 0.6 if moved else 1.0,
            "proxy": 0.7 if moved else 1.2,
        }

    monkeypatch.setattr(transfer_module, "exact_proxy_components", components)
    plc = SimpleNamespace(width=10.0, height=10.0)
    moved_hard, moved_soft, score, _scorer, stats = run_adjacent_cluster_transfer(
        hard,
        soft,
        np.full(3, 0.1),
        np.full(3, 0.1),
        np.full(2, 0.1),
        np.full(2, 0.1),
        10.0,
        10.0,
        np.ones(3, dtype=bool),
        np.ones(2, dtype=bool),
        hierarchy,
        plc,
        SimpleNamespace(),
        1.2,
        candidate_allowed=lambda trial_hard, trial_soft: True,
        deadline=None,
        min_proxy_gain=0.1,
        min_density_congestion_gain=0.1,
        max_inter_scored=2,
        max_intra_scored=1,
        max_inter_accepts=1,
    )

    assert stats["inter_accepts"] == 1
    assert score == 0.7
    assert np.allclose(moved_hard[1], hard[2])
    assert np.allclose(moved_hard[2], hard[1])
    assert np.allclose(moved_soft, soft)
    assert hierarchy.labels.tolist() == [0, 1, 0]
    assert graph.cluster_of(1) == 1
    assert graph.cluster_of(2) == 0
    assert intra_calls == [True]
