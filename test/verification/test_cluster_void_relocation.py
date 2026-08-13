import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from placer.local_search.cluster_void_relocation import (
    _graph_taper_profile,
    _void_cluster_relocation,
    _soft_routing_units,
    find_large_macro_voids,
)
from placer.local_search.soft_hierarchy import SoftBundle


class _Scorer:
    def __init__(self, plc=None):
        self.grid_occupied = np.zeros(400, dtype=np.float64)
        self.dens_grid_area = 1.0
        self.committed = None
        self.plc = plc
        self.soft_commits = 0

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

    def score_move_soft_group(self, soft_indices, new_soft):
        return 0.9 - 0.1 * self.soft_commits

    def commit_move_soft_group(self, soft_indices, new_soft):
        self.soft_commits += 1
        self.committed = (
            np.asarray(soft_indices).copy(),
            np.asarray(new_soft).copy(),
        )


class _ExpansionScorer(_Scorer):
    def congestion_field(self):
        field = np.zeros((20, 20), dtype=np.float64)
        field[:, :10] = 1.0
        return field


class _GraphExpansionScorer(_ExpansionScorer):
    def score_move_group(self, members, new_hard, soft_indices, new_soft):
        return 0.8 if len(members) > 1 else 0.9


def test_graph_taper_profile_follows_low_fanout_edges_with_decaying_shift():
    class _LocationGraph:
        def directional_graph_profile(self, boundary, allowed, *, max_hops, decay):
            assert np.array_equal(boundary, [0])
            assert np.array_equal(allowed, [0, 1, 2, 3, 4])
            assert max_hops == 3
            assert decay == 0.5
            return np.array([0, 1, 2, 3]), np.array([1.0, 0.5, 0.25, 0.125])

    members, scales = _graph_taper_profile(
        np.array([0, 1, 2, 3, 4]),
        np.array([0]),
        max_hops=3,
        decay=0.5,
        location_graph=_LocationGraph(),
    )

    assert np.array_equal(members, [0, 1, 2, 3])
    assert np.allclose(scales, [1.0, 0.5, 0.25, 0.125])


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


def test_large_macro_voids_include_canvas_edge_pockets():
    hard = np.array([[10.0, 10.0]])
    voids = find_large_macro_voids(
        hard,
        np.array([2.0]),
        np.array([4.0]),
        canvas_width=20.0,
        canvas_height=20.0,
        min_width=2.0,
        min_height=2.0,
    )

    assert len(voids) == 4
    assert {void["kind"] for void in voids} == {"edge"}
    assert any(np.allclose(void["rect"], [0.0, 6.0, 8.0, 14.0]) for void in voids)


def test_large_macro_voids_subtract_intervening_hard_blockage():
    hard = np.array([[2.0, 10.0], [18.0, 10.0], [10.0, 10.0]])
    voids = find_large_macro_voids(
        hard,
        np.array([2.0, 2.0, 1.0]),
        np.array([4.0, 4.0, 1.0]),
        large_area_percentile=70.0,
        min_width=1.0,
        min_height=1.0,
    )

    assert voids
    for void in voids:
        x0, y0, x1, y1 = void["rect"]
        assert x1 <= 9.0 or x0 >= 11.0 or y1 <= 9.0 or y0 >= 11.0


def test_soft_routing_units_keep_roles_separate_and_add_residual_fallbacks():
    cache = {
        "net_starts": np.array([0]),
        "net_lengths": np.array([2]),
        "net_weights": np.array([1.0]),
        "ref_idx": np.array([104, 105]),
    }
    plc = SimpleNamespace(
        soft_macro_indices=list(range(100, 107)),
        _wl_vec_cache=cache,
    )
    bundle = SoftBundle(
        members=np.array([2, 3]),
        source="soft_only_connectivity",
        key="stable",
        score=0.8,
    )
    units = _soft_routing_units(
        plc,
        np.zeros((7, 2)),
        {0: np.array([2])},
        {1: np.array([0])},
        [bundle],
        np.ones(7, dtype=bool),
        2,
        np.arange(7, dtype=np.float64),
        max_fanout=16,
        max_cohort=8,
        top_cohorts=12,
        top_singletons=16,
    )

    by_kind = {unit["kind"]: tuple(unit["indices"]) for unit in units}
    assert by_kind["stable_bundle"] == (2, 3)
    assert by_kind["routing_cohort"] == (4, 5)
    assert by_kind["soft_singleton"] == (6,)
    assert all(0 not in unit["indices"] and 1 not in unit["indices"] for unit in units)


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


def test_void_relocation_expands_existing_cluster_boundary_into_clear_corridor():
    hard = np.array([[8.0, 10.0], [4.0, 10.0]])
    scorer = _ExpansionScorer()

    moved_hard, _, accepts, score = _void_cluster_relocation(
        hard.copy(),
        np.zeros((0, 2), dtype=np.float64),
        np.array([2.0, 0.5]),
        np.array([4.0, 0.5]),
        np.zeros(0),
        np.zeros(0),
        20.0,
        20.0,
        2,
        SimpleNamespace(grid_rows=20, grid_cols=20),
        scorer,
        1.0,
        clusters={0: np.array([0, 1])},
        cluster_softs={},
        edges=[],
        movable_h=np.array([True, True]),
        movable_soft=np.zeros(0, dtype=bool),
        candidate_allowed=lambda trial_hard, trial_soft: True,
        min_field_drop=0.01,
        max_scored=96,
    )

    assert accepts >= 1
    assert score == 0.9
    assert moved_hard[0, 0] > hard[0, 0]
    assert np.allclose(moved_hard[1], hard[1])
    assert _void_cluster_relocation.last_stats["expansion_accepts"] == 1


def test_void_relocation_can_select_graph_tapered_boundary_expansion():
    hard = np.array([[8.0, 10.0], [4.0, 10.0], [2.0, 10.0]])
    plc = SimpleNamespace(
        hard_macro_indices=[100, 101, 102],
        _wl_vec_cache={
            "net_starts": np.array([0, 2]),
            "net_lengths": np.array([2, 2]),
            "net_weights": np.ones(2),
            "ref_idx": np.array([100, 101, 101, 102]),
        },
    )
    scorer = _GraphExpansionScorer(plc)

    class _LocationGraph:
        def synchronize(self, hard_positions, soft_positions):
            return None

        def directional_graph_profile(self, boundary, allowed, *, max_hops, decay):
            return np.array([0, 1, 2]), np.array([1.0, decay, decay**2])

    moved_hard, _, accepts, score = _void_cluster_relocation(
        hard.copy(),
        np.zeros((0, 2), dtype=np.float64),
        np.array([2.0, 0.5, 0.5]),
        np.array([4.0, 0.5, 0.5]),
        np.zeros(0),
        np.zeros(0),
        20.0,
        20.0,
        3,
        SimpleNamespace(grid_rows=20, grid_cols=20),
        scorer,
        1.0,
        location_graph=_LocationGraph(),
        clusters={0: np.array([0, 1, 2])},
        cluster_softs={},
        edges=[],
        movable_h=np.ones(3, dtype=bool),
        movable_soft=np.zeros(0, dtype=bool),
        candidate_allowed=lambda trial_hard, trial_soft: True,
        min_field_drop=0.01,
        max_accepts=1,
        max_scored=96,
    )

    shifts = moved_hard[:, 0] - hard[:, 0]
    assert accepts == 1
    assert score == 0.8
    assert shifts[0] > shifts[1] > shifts[2] > 0.0
    assert _void_cluster_relocation.last_stats["graph_taper_accepts"] == 1


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


def test_void_relocation_can_commit_multiple_residual_soft_singletons():
    cache = {
        "net_starts": np.zeros(0, dtype=np.int64),
        "net_lengths": np.zeros(0, dtype=np.int64),
        "net_weights": np.zeros(0, dtype=np.float64),
        "ref_idx": np.zeros(0, dtype=np.int64),
    }
    plc = SimpleNamespace(soft_macro_indices=[0, 1], hard_macro_indices=[], _wl_vec_cache=cache)
    scorer = _Scorer(plc)
    hard = np.array([[2.0, 10.0], [18.0, 10.0]])
    soft = np.array([[8.0, 2.0], [12.0, 2.0]])

    moved_hard, moved_soft, accepts, score = _void_cluster_relocation(
        hard.copy(),
        soft.copy(),
        np.array([2.0, 2.0]),
        np.array([4.0, 4.0]),
        np.array([0.5, 0.5]),
        np.array([0.5, 0.5]),
        20.0,
        20.0,
        2,
        SimpleNamespace(grid_rows=20, grid_cols=20),
        scorer,
        1.0,
        clusters={0: np.array([0]), 1: np.array([1])},
        cluster_softs={},
        edges=[],
        movable_h=np.array([False, False]),
        movable_soft=np.array([True, True]),
        candidate_allowed=lambda trial_hard, trial_soft: True,
        min_field_drop=0.01,
        max_accepts=2,
        max_scored=64,
    )

    assert accepts == 2
    assert score == 0.8
    assert np.allclose(moved_hard, hard)
    assert np.all(moved_soft[:, 1] >= 6.5)
    assert _void_cluster_relocation.last_stats["accepted_kinds"]
