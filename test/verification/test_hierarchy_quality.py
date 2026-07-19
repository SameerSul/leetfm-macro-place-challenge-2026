import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from placer.local_search.hierarchy_quality import (
    HIERARCHY_VECTOR_METRICS,
    _neighbor_impurity,
    _neighbor_impurity_reference,
    hierarchy_quality_vector,
    hierarchy_vector_contract,
    hierarchy_vector_limits,
    hierarchy_vector_margins,
)
from placer.pipeline.segments.floorplan_seed import (
    _hard_placement_is_legal,
    select_seed_candidate,
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


def test_component_margins_are_positive_headroom_and_negative_violations():
    candidate = {key: 0.10 for key in HIERARCHY_VECTOR_METRICS}
    limits = {key: 0.12 for key in HIERARCHY_VECTOR_METRICS}
    candidate["bridge_soft_distance"] = 0.13

    margins = hierarchy_vector_margins(candidate, limits)

    assert np.isclose(margins["cluster_compactness"], 0.02)
    assert np.isclose(margins["bridge_soft_distance"], -0.01)
