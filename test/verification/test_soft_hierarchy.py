import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from placer.local_search.soft_hierarchy import (
    SoftBundle,
    combine_soft_bundle_evidence,
    infer_connectivity_soft_bundles,
    infer_path_soft_bundles,
    select_high_confidence_soft_bundles,
    soft_bundle_confidence,
)
from placer.local_search.clusters import classify_soft_role_evidence


def test_path_soft_bundles_choose_useful_explicit_prefixes():
    bundles = infer_path_soft_bundles(
        [
            "top/cpu0/cache/group0",
            "top/cpu0/cache/group1",
            "top/cpu1/cache/group0",
            "top/cpu1/cache/group1",
            "top/io/uart/group0",
        ],
        min_coverage=0.5,
    )

    assert [bundle.key for bundle in bundles] == ["top/cpu0/cache", "top/cpu1/cache"]
    assert all(bundle.source == "path" for bundle in bundles)
    assert np.array_equal(bundles[0].members, np.array([0, 1]))
    assert np.array_equal(bundles[1].members, np.array([2, 3]))


def test_flat_soft_names_do_not_create_false_bundles():
    assert infer_path_soft_bundles(["Grp_0", "Grp_1", "Grp_2"]) == ()


def test_connectivity_soft_bundles_require_mutually_strong_repeated_edges():
    bundles = infer_connectivity_soft_bundles(
        6,
        [
            [0, 1],
            [0, 1],
            [0, 1, 2],
            [1, 2],
            [1, 2],
            [2, 3],
            [4, 5],
        ],
    )

    assert len(bundles) == 1
    assert bundles[0].source == "connectivity"
    assert np.array_equal(bundles[0].members, np.array([0, 1, 2]))
    assert 0.0 < bundles[0].score < 0.9


def test_combined_bundle_evidence_rewards_common_hard_ownership():
    path = SoftBundle(np.array([0, 1]), "path", "top/ip", 0.95)
    linked = SoftBundle(np.array([2, 3]), "connectivity", "softnet:2,3", 0.72)
    unowned = SoftBundle(np.array([4, 5]), "connectivity", "softnet:4,5", 0.72)

    bundles = combine_soft_bundle_evidence(
        (path,),
        (linked, unowned),
        {7: np.array([12, 13])},
        {},
        n_hard=10,
    )

    assert bundles[0].source == "path"
    assert bundles[0].score == 0.95
    assert bundles[1].source == "connectivity+owner"
    assert 0.75 <= bundles[1].score < 0.9
    assert bundles[2].source == "connectivity"
    assert bundles[2].score == 0.72


def test_only_high_confidence_bundles_are_selected_for_compound_moves():
    bundles = (
        SoftBundle(np.array([0, 1]), "path", "top/ip", 0.95),
        SoftBundle(np.array([2, 3]), "connectivity+owner", "softnet:2,3", 0.90),
        SoftBundle(np.array([4, 5]), "connectivity", "softnet:4,5", 0.80),
        SoftBundle(np.array([6, 7]), "connectivity", "softnet:6,7", 0.60),
    )

    selected = select_high_confidence_soft_bundles(bundles)

    assert [bundle.confidence for bundle in selected] == ["high", "high"]
    assert soft_bundle_confidence(0.80) == "medium"
    assert soft_bundle_confidence(0.60) == "low"


def test_connectivity_plus_owner_evidence_stays_medium_without_explicit_path():
    linked = SoftBundle(np.array([2, 3]), "connectivity+owner", "softnet:2,3", 0.89)

    assert soft_bundle_confidence(linked.score) == "medium"
    assert select_high_confidence_soft_bundles((linked,)) == ()


def test_flat_soft_role_confidence_requires_repeated_hard_affinity():
    single_owner = classify_soft_role_evidence([(3, 1)])
    repeated_owner = classify_soft_role_evidence([(3, 3)])
    single_bridge = classify_soft_role_evidence([(3, 1), (4, 1)])
    repeated_bridge = classify_soft_role_evidence([(3, 3), (4, 3)])

    assert single_owner["role"] == "owned"
    assert single_owner["confidence"] == "low"
    assert repeated_owner["confidence"] == "medium"
    assert single_bridge["role"] == "bridge"
    assert single_bridge["confidence"] == "low"
    assert repeated_bridge["confidence"] == "medium"
    assert all(
        row["score"] < 0.90
        for row in (single_owner, repeated_owner, single_bridge, repeated_bridge)
    )
