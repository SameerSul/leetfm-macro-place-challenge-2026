import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from placer.local_search.hierarchy_quality import hierarchy_quality_vector
from placer.pipeline.segments.floorplan_seed import select_seed_candidate


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
