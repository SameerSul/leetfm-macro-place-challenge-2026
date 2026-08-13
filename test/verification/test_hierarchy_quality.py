import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from utils import constants as const

from placer.local_search.hierarchy_quality import (
    hierarchy_island_contract,
    hierarchy_island_limits,
    hierarchy_island_metrics,
    HIERARCHY_VECTOR_METRICS,
    _neighbor_impurity,
    _neighbor_impurity_reference,
    hierarchy_quality_vector,
    hierarchy_vector_contract,
    hierarchy_vector_limits,
    hierarchy_vector_margins,
)


def test_island_contract_catches_one_scattered_cluster_hidden_by_averages():
    hard = np.asarray([[1.0, 1.0], [2.0, 1.0], [8.0, 8.0], [9.0, 8.0]])
    soft = np.asarray([[1.5, 1.5]])
    sizes = np.ones((4, 2))
    clusters = {0: np.asarray([0, 1]), 1: np.asarray([2, 3])}
    cluster_softs = {0: np.asarray([4])}
    reference = hierarchy_island_metrics(hard, soft, clusters, cluster_softs, sizes, 10.0, 10.0)
    limits = hierarchy_island_limits(
        reference,
        {0: 1.0, 1: 1.0},
        "hierarchy_path_tags",
        distance_absolute_slack=0.0,
    )

    scattered = hard.copy()
    scattered[1] = [6.0, 1.0]
    candidate = hierarchy_island_metrics(
        scattered, soft, clusters, cluster_softs, sizes, 10.0, 10.0
    )
    passed, violations = hierarchy_island_contract(candidate, limits)

    assert not passed
    assert "island_0_spread" in violations


def test_island_contract_is_confidence_calibrated():
    hard = np.asarray([[1.0, 1.0], [2.0, 1.0], [8.0, 8.0], [9.0, 8.0]])
    sizes = np.ones((4, 2))
    clusters = {0: np.asarray([0, 1]), 1: np.asarray([2, 3])}
    reference = hierarchy_island_metrics(hard, np.zeros((0, 2)), clusters, {}, sizes, 10.0, 10.0)
    limits = hierarchy_island_limits(
        reference,
        {0: 0.9, 1: 0.1},
        "hierarchy_oversized_connectivity",
        strict_confidence=0.65,
    )

    assert limits[0]["tier"] == 2.0
    assert 1 not in limits


from placer.pipeline.segments.floorplan_seed import (
    _hard_placement_is_legal,
    repair_seed_to_contract,
    select_initial_recurrent_leaves,
    select_recursive_prototype_leaves,
    select_seed_candidate,
    should_run_initial_recurrent,
)


def test_recursive_prototype_leaf_selection_is_stable_and_excludes_fixed_members():
    clusters = {
        7: np.array([0, 1]),
        3: np.array([2, 3]),
        9: np.array([4, 5]),
    }
    hard = np.array([[1.0, 1.0], [2.0, 1.0], [5.0, 1.0], [6.0, 1.0], [8.0, 1.0], [9.0, 1.0]])
    half = np.full(6, 0.4)
    movable = np.array([True, True, True, True, False, True])

    selected = select_recursive_prototype_leaves(
        clusters,
        {7: 0.8, 3: 0.8, 9: 1.0},
        movable,
        hard,
        half,
        half,
        max_leaves=2,
    )

    assert selected == [3, 7]
    assert 9 not in selected


def test_recursive_prototype_leaf_selection_requires_positive_confidence():
    selected = select_recursive_prototype_leaves(
        {0: np.array([0, 1])},
        {0: 0.0},
        np.ones(2, dtype=bool),
        np.array([[1.0, 1.0], [2.0, 1.0]]),
        np.full(2, 0.4),
        np.full(2, 0.4),
        max_leaves=1,
    )

    assert selected == []


def test_recursive_prototype_leaf_selection_excludes_oversized_leaf():
    members = np.arange(17, dtype=np.int64)
    selected = select_recursive_prototype_leaves(
        {0: members},
        {0: 1.0},
        np.ones(17, dtype=bool),
        np.column_stack((np.arange(17, dtype=np.float64), np.zeros(17))),
        np.full(17, 0.4),
        np.full(17, 0.4),
        max_leaves=1,
        max_members=16,
    )

    assert selected == []


def test_initial_recurrent_leaf_selection_spreads_strong_anchors_across_canvas():
    clusters = {
        0: np.array([0, 1]),
        1: np.array([2, 3]),
        2: np.array([4, 5]),
        3: np.array([6, 7]),
    }
    hard = np.array(
        [
            [1.0, 1.0],
            [2.0, 1.0],
            [4.0, 1.0],
            [5.0, 1.0],
            [49.0, 1.0],
            [50.0, 1.0],
            [98.0, 1.0],
            [99.0, 1.0],
        ]
    )
    selected = select_initial_recurrent_leaves(
        clusters,
        {0: 0.90, 1: 0.85, 2: 0.80, 3: 0.75},
        np.ones(8, dtype=bool),
        hard,
        np.full(8, 0.4),
        np.full(8, 0.4),
        canvas_width=100.0,
        canvas_height=100.0,
        max_leaves=3,
    )

    assert selected == [0, 3, 2]


def test_initial_recurrent_requires_contract_failure_and_large_proxy_gap():
    assert should_run_initial_recurrent(
        1.0,
        1.2,
        dreamplace_contract_passed=False,
        minimum_proxy_advantage=0.15,
    )
    assert not should_run_initial_recurrent(
        1.0,
        1.2,
        dreamplace_contract_passed=True,
        minimum_proxy_advantage=0.15,
    )
    assert not should_run_initial_recurrent(
        1.05,
        1.2,
        dreamplace_contract_passed=False,
        minimum_proxy_advantage=0.15,
    )


def _vector(hard, soft):
    clusters = {0: np.array([0, 1]), 1: np.array([2, 3])}
    owned = {0: np.array([4]), 1: np.array([5])}
    bridges = {2: np.array([0, 1])}
    edges = [(0, 1, 3.0)]
    return hierarchy_quality_vector(
        np.asarray(hard, dtype=np.float64),
        np.asarray(soft, dtype=np.float64),
        clusters,
        owned,
        bridges,
        edges,
        100.0,
        100.0,
    )


def test_hierarchy_vector_prefers_compact_pure_clusters_and_soft_roles():
    coherent = _vector(
        [[20, 20], [24, 20], [70, 20], [74, 20]],
        [[22, 22], [72, 22], [47, 20]],
    )
    mixed = _vector(
        [[20, 20], [70, 20], [24, 20], [74, 20]],
        [[72, 70], [22, 70], [47, 70]],
    )

    assert coherent["composite"] < mixed["composite"]
    assert coherent["neighbor_impurity"] < mixed["neighbor_impurity"]
    assert coherent["owned_soft_distance"] < mixed["owned_soft_distance"]
    assert coherent["bridge_soft_distance"] < mixed["bridge_soft_distance"]


def test_numba_neighbor_impurity_matches_stable_sort_for_sparse_clustered_indices():
    rng = np.random.default_rng(73)
    hard = rng.normal(size=(17, 2)).astype(np.float64)
    clustered = np.array([1, 2, 4, 7, 8, 9, 12, 16], dtype=np.int64)
    labels = np.full(hard.shape[0], -1, dtype=np.int64)
    labels[clustered] = np.array([0, 0, 0, 1, 1, 2, 2, 2], dtype=np.int64)
    own_sizes = np.zeros(hard.shape[0], dtype=np.int64)
    for cluster_id in np.unique(labels[clustered]):
        members = clustered[labels[clustered] == cluster_id]
        own_sizes[members] = members.size

    expected = _neighbor_impurity_reference(hard, clustered, labels, own_sizes)
    actual = _neighbor_impurity(hard, clustered, labels, own_sizes)

    assert actual == expected


def test_numba_neighbor_impurity_preserves_stable_tie_order():
    hard = np.array([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [3.0, 0.0]])
    clustered = np.arange(hard.shape[0], dtype=np.int64)
    labels = np.array([0, 1, 0, 1], dtype=np.int64)
    own_sizes = np.array([2, 2, 2, 2], dtype=np.int64)

    expected = _neighbor_impurity_reference(hard, clustered, labels, own_sizes)
    actual = _neighbor_impurity(hard, clustered, labels, own_sizes)

    assert expected == 0.5
    assert actual == expected


def test_seed_selector_uses_proxy_within_best_hierarchy_band():
    rows = [
        {"name": "best_hq", "score": 1.20, "hierarchy_composite": 0.100},
        {"name": "balanced", "score": 1.10, "hierarchy_composite": 0.108},
        {"name": "proxy_only", "score": 1.00, "hierarchy_composite": 0.140},
    ]

    proxy = select_seed_candidate(
        rows, hierarchy_first=False, absolute_slack=0.01, relative_slack=0.0
    )
    hierarchy = select_seed_candidate(
        rows, hierarchy_first=True, absolute_slack=0.01, relative_slack=0.0
    )

    assert proxy["name"] == "proxy_only"
    assert hierarchy["name"] == "balanced"


def test_production_seed_selection_preserves_contract_valid_spread():
    """Production ranks passing seeds by proxy instead of compactness alone."""
    assert const.HIER_SEED_HIERARCHY_SELECT is False


def test_seed_selector_prefers_contract_headroom_inside_proxy_band():
    limits = {key: 1.0 for key in HIERARCHY_VECTOR_METRICS}
    tight = {key: 0.95 for key in HIERARCHY_VECTOR_METRICS}
    roomy = {key: 0.60 for key in HIERARCHY_VECTOR_METRICS}
    rows = [
        {
            "name": "tight",
            "score": 1.00,
            "hierarchy_composite": 0.30,
            "hierarchy_vector": tight,
        },
        {
            "name": "roomy",
            "score": 1.03,
            "hierarchy_composite": 0.20,
            "hierarchy_vector": roomy,
        },
    ]

    selected = select_seed_candidate(
        rows,
        hierarchy_first=False,
        absolute_slack=0.0,
        relative_slack=0.0,
        component_absolute_slack=limits,
        component_relative_slack=0.0,
        component_reference_name="tight",
        component_reference_vector={key: 0.0 for key in HIERARCHY_VECTOR_METRICS},
        headroom_aware=True,
        proxy_band_relative=0.05,
    )

    assert selected["name"] == "roomy"


def test_seed_selector_does_not_buy_headroom_outside_proxy_band():
    rows = [
        {"name": "proxy", "score": 1.00, "hierarchy_composite": 0.30},
        {"name": "hierarchy", "score": 1.20, "hierarchy_composite": 0.10},
    ]

    selected = select_seed_candidate(
        rows,
        hierarchy_first=False,
        absolute_slack=0.0,
        relative_slack=0.0,
        headroom_aware=True,
        proxy_band_relative=0.05,
    )

    assert selected["name"] == "proxy"


def test_component_contract_rejects_one_dimension_regression():
    reference = _vector(
        [[20, 20], [24, 20], [70, 20], [74, 20]],
        [[22, 22], [72, 22], [47, 20]],
    )
    worse_soft = dict(reference)
    worse_soft["owned_soft_distance"] += 0.05
    rows = [
        {
            "name": "initial",
            "score": 1.10,
            "hierarchy_composite": reference["composite"],
            "hierarchy_vector": reference,
        },
        {
            "name": "proxy_only",
            "score": 1.00,
            "hierarchy_composite": reference["composite"],
            "hierarchy_vector": worse_soft,
        },
    ]
    slack = {key: 0.01 for key in HIERARCHY_VECTOR_METRICS}

    selected = select_seed_candidate(
        rows,
        hierarchy_first=False,
        absolute_slack=0.0,
        relative_slack=0.0,
        component_absolute_slack=slack,
        component_relative_slack=0.0,
    )

    assert selected["name"] == "initial"
    assert rows[1]["hierarchy_contract_eligible"] is False
    assert "owned_soft_distance" in rows[1]["hierarchy_contract_violations"]


def test_component_contract_uses_independent_relative_and_absolute_limits():
    reference = {key: 0.1 for key in HIERARCHY_VECTOR_METRICS}
    limits = hierarchy_vector_limits(
        reference,
        {key: 0.01 for key in HIERARCHY_VECTOR_METRICS},
        0.2,
    )
    candidate = dict(reference)
    candidate["edge_stretch"] = 0.119

    passed, violations = hierarchy_vector_contract(candidate, limits)

    assert passed
    assert violations == {}
    assert limits["edge_stretch"] == 0.12000000000000001


def test_seed_selector_can_use_stricter_external_reference_vector():
    row_reference = {key: 0.105 for key in HIERARCHY_VECTOR_METRICS}
    external_reference = {key: 0.10 for key in HIERARCHY_VECTOR_METRICS}
    candidate = dict(external_reference)
    candidate["worst_cluster_spread"] = 0.115
    rows = [
        {
            "name": "initial",
            "score": 1.10,
            "hierarchy_composite": 0.105,
            "hierarchy_vector": row_reference,
        },
        {
            "name": "proxy",
            "score": 1.00,
            "hierarchy_composite": 0.10,
            "hierarchy_vector": candidate,
        },
    ]

    selected = select_seed_candidate(
        rows,
        hierarchy_first=False,
        absolute_slack=0.0,
        relative_slack=0.0,
        component_absolute_slack={key: 0.01 for key in HIERARCHY_VECTOR_METRICS},
        component_relative_slack=0.0,
        component_reference_vector=external_reference,
    )

    assert selected["name"] == "initial"
    assert rows[1]["hierarchy_contract_eligible"] is False
    assert rows[0]["hierarchy_contract_reference_vector"] == external_reference


def test_raw_seed_reference_requires_legal_hard_placement():
    hw = np.asarray([1.0, 1.0], dtype=np.float64)
    hh = np.asarray([1.0, 1.0], dtype=np.float64)

    assert _hard_placement_is_legal(
        np.asarray([[2.0, 2.0], [5.0, 5.0]], dtype=np.float64),
        hw,
        hh,
        8.0,
        8.0,
    )
    assert not _hard_placement_is_legal(
        np.asarray([[2.0, 2.0], [3.0, 2.0]], dtype=np.float64),
        hw,
        hh,
        8.0,
        8.0,
    )


def test_seed_selector_can_anchor_an_illegal_initial_case_to_dreamplace():
    dreamplace_vector = {key: 0.10 for key in HIERARCHY_VECTOR_METRICS}
    initial_vector = dict(dreamplace_vector)
    initial_vector["edge_stretch"] = 0.30
    rows = [
        {
            "name": "initial",
            "score": 5.0,
            "hierarchy_composite": 0.30,
            "hierarchy_vector": initial_vector,
        },
        {
            "name": "dreamplace",
            "score": 1.0,
            "hierarchy_composite": 0.10,
            "hierarchy_vector": dreamplace_vector,
        },
    ]

    selected = select_seed_candidate(
        rows,
        hierarchy_first=False,
        absolute_slack=0.0,
        relative_slack=0.0,
        component_absolute_slack={key: 0.01 for key in HIERARCHY_VECTOR_METRICS},
        component_relative_slack=0.0,
        component_reference_name="dreamplace",
        component_reference_vector=dreamplace_vector,
    )

    assert selected["name"] == "dreamplace"
    assert rows[0]["hierarchy_contract_eligible"] is False
    assert selected["hierarchy_contract_reference"] == "dreamplace"


def test_seed_contract_repair_finds_furthest_passing_interpolation():
    source_hard = np.asarray([[1.0, 0.0]], dtype=np.float64)
    reference_hard = np.asarray([[0.0, 0.0]], dtype=np.float64)
    empty_soft = np.empty((0, 2), dtype=np.float64)
    limits = {key: 1.0 for key in HIERARCHY_VECTOR_METRICS}
    limits["edge_stretch"] = 0.60
    calls = []

    def legalize(hard, soft):
        calls.append(float(hard[0, 0]))
        return hard.copy(), soft.copy()

    def vector(hard, _soft):
        result = {key: 0.0 for key in HIERARCHY_VECTOR_METRICS}
        result["edge_stretch"] = float(hard[0, 0])
        result["composite"] = result["edge_stretch"]
        return result

    repaired = repair_seed_to_contract(
        source_hard,
        empty_soft,
        reference_hard,
        empty_soft,
        legalize_fn=legalize,
        vector_fn=vector,
        limits=limits,
        refine_rounds=5,
    )

    assert repaired is not None
    hard, soft, repaired_vector, fraction, attempts = repaired
    assert soft.shape == (0, 2)
    assert np.allclose(hard, [[fraction, 0.0]])
    assert 0.59 < fraction <= 0.60
    assert repaired_vector["edge_stretch"] <= limits["edge_stretch"]
    assert attempts == len(calls)


def test_component_margins_are_positive_headroom_and_negative_violations():
    candidate = {key: 0.10 for key in HIERARCHY_VECTOR_METRICS}
    limits = {key: 0.12 for key in HIERARCHY_VECTOR_METRICS}
    candidate["bridge_soft_distance"] = 0.13

    margins = hierarchy_vector_margins(candidate, limits)

    assert np.isclose(margins["cluster_compactness"], 0.02)
    assert np.isclose(margins["bridge_soft_distance"], -0.01)
