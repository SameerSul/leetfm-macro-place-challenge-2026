import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from placer.local_search import cluster_tile_rearrange as tiles
from placer.routing.apply import _smooth_routing_cong_vec


def test_directional_tail_attribution_matches_evaluator_smoothing_at_edges():
    congestion = np.arange(24, dtype=np.float64)
    density = np.arange(12, dtype=np.float64)
    original = density.copy()
    scorer = SimpleNamespace(
        grid_row=3, grid_col=4, smooth_range=2, grid_h_routes=7.0,
        grid_v_routes=11.0, dens_grid_area=4.0,
        _swap_tail_baseline=lambda: dict(
            congestion=congestion, congestion_order=np.argsort(congestion)[::-1],
            density=density, density_order=np.argsort(density)[::-1], density_nonzero=11,
        ),
    )
    for peak in (0, 3, 12, 23):
        congestion[peak] = 100
        view = tiles._tail_view(scorer)
        raw = np.arange(12, dtype=np.float64) + 1
        for direction, axis, capacity in (("h", True, 7), ("v", False, 11)):
            smoothed = _smooth_routing_cong_vec(raw, 3, 4, 2, axis)
            assert np.isclose(raw @ view["route_" + direction], smoothed @ view[direction])
        assert np.isclose(view["density"].sum(), 0.25 / 4)
        congestion[peak] = peak
    np.testing.assert_array_equal(density, original)


def test_incident_route_can_attribute_hotspot_away_from_macro(monkeypatch):
    def route(plc, struct, multiplier, horizontal, vertical):
        horizontal[0] += 2

    monkeypatch.setattr(tiles, "_build_net_routing_struct", lambda plc, nets: nets)
    monkeypatch.setattr(tiles, "_apply_net_routing_struct", route)
    monkeypatch.setattr(tiles, "_apply_macro_routing_subset", lambda *args: None)
    scorer = SimpleNamespace(
        hard_indices=[], soft_indices=[20], n_hard=0, grid_row=2, grid_col=2, plc=None,
        _touched_nets_many=lambda modules: np.array([0]),
        _hard_slot_array=lambda *args: np.array([], dtype=int),
        _macro_occ=lambda *args: (np.array([3]), np.array([1.0])),
    )
    view = {key: np.zeros(4) for key in ("h", "v", "route_h", "route_v", "density")}
    view["route_h"][0] = 0.25
    assert tiles._unit_pressure(scorer, np.array([0]), np.array([[9., 9.]]), view) == 0.5
    view["route_h"][0], view["route_v"][0] = 0, 0.25
    assert tiles._unit_pressure(scorer, np.array([0]), np.array([[9., 9.]]), view) == 0


def _hierarchy(n, softs=0):
    return SimpleNamespace(
        labels=np.zeros(n, dtype=int), subcluster_labels=np.zeros(n, dtype=int),
        cluster_softs={0: np.arange(n, n + softs)}, subcluster_softs={0: np.arange(n, n + softs)},
        bridge_softs={}, subcluster_bridge_softs={}, parent_bridge_softs={},
        soft_bundles=[], soft_connectivity_bundles=[], soft_bundle_evidence=[],
        active_soft_bundles=[], soft_only_bundles=[],
        location_graph=SimpleNamespace(
            neighbors=lambda i: {1 - i: 1.0} if i < 2 else {}, synchronize=lambda *args: None,
        ),
    )


def test_leaf_units_preserve_bundles_fixed_macros_bridges_and_children():
    hierarchy = _hierarchy(3, 5)
    hierarchy.subcluster_labels[2] = 1
    bundle = SimpleNamespace(members=(0, 1), source="path")
    hierarchy.soft_bundles = hierarchy.active_soft_bundles = [bundle]
    hierarchy.bridge_softs = {2: (0, 1)}
    movable = np.ones(8, dtype=bool)
    movable[7] = False
    groups = tiles._leaf_units(hierarchy, movable, 3)
    assert [u.tolist() for u in groups[0, 0]] == [[0], [1], [6], [3, 4]]
    assert (0, 1) not in groups  # A single-unit child cannot join another child.
    movable[3] = False
    assert all(3 not in u and 4 not in u for u in tiles._leaf_units(hierarchy, movable, 3)[0, 0])
    movable[3] = True
    bundle.source = "soft_only_connectivity"
    assert all(3 not in u and 4 not in u for u in tiles._leaf_units(hierarchy, movable, 3)[0, 0])


def test_joint_rearrangement_requires_fresh_gain_and_contract(monkeypatch):
    baseline = np.array([[2., 2.], [6., 2.], [9., 9.]])
    benchmark = SimpleNamespace(
        macro_fixed=torch.tensor([False, False, True]), macro_sizes=torch.tensor([[1., 1.], [2., 1.], [1., 1.]]),
        canvas_width=10., canvas_height=10.,
    )
    benchmark.get_movable_mask = lambda: ~benchmark.macro_fixed
    hierarchy = _hierarchy(3)
    resets = []

    def jointly_better(pos):
        return np.array_equal(pos[:2], baseline[[1, 0]])

    class Scorer:
        hard_indices, soft_indices = [0, 1, 2], []
        grid_w = grid_h = 1.
        dens_grid_area = 1.
        wl_normalizer = 100.

        def __init__(self, plc, benchmark, positions):
            self.positions = positions.copy()
            resets.append(positions.copy())

        def _macro_occ(self, module, x, y):
            return np.array([int(y) * 10 + int(x)]), np.array([1.])

        def score_move_group(self, hidx, hxy, sidx, sxy):
            pos = self.positions.copy()
            pos[hidx] = hxy
            return 0.9 if jointly_better(pos) else 1.1

        def commit_move_group(self, hidx, hxy, sidx, sxy):
            self.positions[hidx] = hxy

    monkeypatch.setattr(tiles, "IncrementalScorer", Scorer)
    monkeypatch.setattr(tiles, "_tail_view", lambda scorer: {"field": np.zeros((10, 10))})
    monkeypatch.setattr(tiles, "_unit_pressure", lambda *args: 1.)
    region = np.tile([0.5, 0.5, 9.5, 9.5], (3, 1))
    for contract, fresh_score in ((True, 0.9), (False, 0.9), (True, 1.01)):
        monkeypatch.setattr(tiles, "_exact_proxy", lambda p, *args: fresh_score if jointly_better(p.numpy()) else 1.)
        hard, soft, before, after, stats = tiles.final_cluster_tile_relief(
            baseline.copy(), np.empty((0, 2)), benchmark, None, hierarchy,
            lambda h, s: contract, hard_region=region, soft_region=np.empty((0, 4)),
        )
        keep = contract and fresh_score < 1
        assert bool(stats["accepts"]) == keep
        assert before == 1 and after == (fresh_score if keep else before)
        assert stats["scored"] <= 64 and stats["patches"] <= 4
        np.testing.assert_array_equal(hard[2], baseline[2])
        np.testing.assert_array_equal(resets[-1], np.vstack([hard, soft]).astype(np.float32))
        if keep:
            np.testing.assert_array_equal(hard[:2], baseline[[1, 0]])
        else:
            np.testing.assert_array_equal(hard, baseline)

    # A profitable swap cannot move a macro out of its frozen center region.
    region[0, 2] = 3.0
    monkeypatch.setattr(tiles, "_exact_proxy", lambda p, *args: 0.9 if jointly_better(p.numpy()) else 1.)
    hard, _, _, _, stats = tiles.final_cluster_tile_relief(
        baseline.copy(), np.empty((0, 2)), benchmark, None, hierarchy,
        lambda h, s: True, hard_region=region, soft_region=np.empty((0, 4)),
    )
    assert stats["accepts"] == 0
    np.testing.assert_array_equal(hard, baseline)

    # No search or scorer construction when the borrowed allowance is spent.
    prior_resets = len(resets)
    _, _, before, after, stats = tiles.final_cluster_tile_relief(
        baseline, np.empty((0, 2)), benchmark, None, hierarchy,
        lambda h, s: True, hard_region=region, soft_region=np.empty((0, 4)), deadline=0.0,
    )
    assert len(resets) == prior_resets and before is None and after is None
    assert stats["scored"] == 0

    # Recheck hard legality after the final output conversion/clamp as well.
    region[0, 2] = 9.5
    original_clamp = tiles.clamp_in_bounds

    def invalid_output(p, b):
        p = original_clamp(p, b)
        if jointly_better(p.numpy()):
            p[0] = torch.tensor([8.8, 9.0])
        return p

    monkeypatch.setattr(tiles, "clamp_in_bounds", invalid_output)
    monkeypatch.setattr(tiles, "_exact_proxy", lambda p, *args: 1. if np.array_equal(p.numpy(), baseline) else 0.9)
    hard, _, _, _, stats = tiles.final_cluster_tile_relief(
        baseline, np.empty((0, 2)), benchmark, None, hierarchy,
        lambda h, s: True, hard_region=region, soft_region=np.empty((0, 4)),
    )
    assert stats["accepts"] == 0
    np.testing.assert_array_equal(hard, baseline)
