import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import placer.pipeline.segments.floorplan_post_coldspot as final_stage


def test_final_soft_relief_freezes_hierarchy_and_requires_verified_gain(monkeypatch):
    hard = np.array([[1.0, 1.0]])
    soft = np.full((8, 2), 2.0)
    benchmark = SimpleNamespace(
        macro_fixed=torch.tensor([False] * 7 + [True, False]),
        macro_sizes=torch.ones((9, 2)), canvas_width=10.0, canvas_height=10.0,
    )
    benchmark.get_movable_mask = lambda: ~benchmark.macro_fixed
    hierarchy = SimpleNamespace(
        cluster_softs={0: [1]}, bridge_softs={1: (0, 1)},
        subcluster_softs={0: [3]}, subcluster_bridge_softs={},
        parent_cluster_softs={}, parent_bridge_softs={3: (0, 1)},
        soft_bundles=[SimpleNamespace(members=[4])],
        soft_connectivity_bundles=[], soft_bundle_evidence=[],
        active_soft_bundles=[], soft_only_bundles=[],
    )
    resets = []

    class Scorer:
        soft_indices = np.arange(1, 9)

        def __init__(self, plc, benchmark, positions):
            resets.append(positions.copy())

        def _touched_nets_only(self, index):
            return np.array([0] if index == 6 else [], dtype=np.int64)

    def relocate(positions, *args, **kwargs):
        assert np.flatnonzero(kwargs["soft_movable"]).tolist() == [7]
        assert kwargs["max_scored"] == 64
        relocate.last_stats = {"scored": 4, "accepts": 1}
        # The final transaction must protect frozen rows even from a bad proposal.
        proposed = positions + 0.1
        proposed[7, 0] = 100.0
        return proposed, 1, 0.9

    monkeypatch.setattr(final_stage, "IncrementalScorer", Scorer)
    monkeypatch.setattr(final_stage, "_soft_relocation_moves", relocate)
    monkeypatch.setattr(final_stage, "_build_wl_cache", lambda plc: {
        "ref_idx": np.array([0, 6]), "net_starts": np.array([0]),
        "net_lengths": np.array([2]),
    })
    for contract_passes, verified_score in ((True, 0.99), (True, 1.01), (False, 0.99)):
        scores = iter((1.0, verified_score))
        def exact_score(positions, *args):
            assert float(positions[:, 0].max()) <= 9.5
            return next(scores)

        monkeypatch.setattr(final_stage, "_exact_proxy", exact_score)

        def contract(h, s):
            np.testing.assert_array_equal(h, hard)
            np.testing.assert_array_equal(s[:7], soft[:7])
            return contract_passes

        result, before, after, stats = final_stage._final_free_soft_relief(
            hard, soft, benchmark, SimpleNamespace(hard_macro_indices=[0]),
            hierarchy, 0.5, contract,
        )
        kept = contract_passes and verified_score < 1.0
        assert before == 1.0  # Recomputed, not the supplied stale score.
        assert after == (verified_score if kept else before)
        assert stats["accepts"] == int(kept)
        assert stats["eligible"] == 1
        np.testing.assert_array_equal(result[:7], soft[:7])
        assert bool(np.any(result[7] != soft[7])) == kept
        np.testing.assert_array_equal(resets[-1], np.vstack([hard, result]).astype(np.float32))
