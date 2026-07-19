import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from placer.local_search.relocation import _relocation_moves


class _FakeScorer:
    def __init__(self, hard_pos, sizes, field):
        self.committed_hard_pos = np.asarray(hard_pos, dtype=np.float64).copy()
        self.committed_soft_pos = np.empty((0, 2), dtype=np.float64)
        self.benchmark = SimpleNamespace(macro_sizes=torch.tensor(sizes, dtype=torch.float32))
        self._field = np.asarray(field, dtype=np.float64)
        self.exact_batches = 0

    def congestion_field(self):
        return self._field

    def _prepare_move(self, _index):
        raise AssertionError("hierarchy-rejected candidates must not reach exact scoring")

    def _trial_many_at(self, _prep, _targets):
        self.exact_batches += 1
        raise AssertionError("hierarchy-rejected candidates must not reach exact scoring")


def test_hard_relocation_hierarchy_gate_runs_before_exact_scoring():
    hard = np.array([[75.0, 75.0]], dtype=np.float64)
    sizes = np.array([[10.0, 10.0]], dtype=np.float64)
    scorer = _FakeScorer(hard, sizes, [[0.0, 0.0], [0.0, 1.0]])
    benchmark = SimpleNamespace(
        grid_rows=2,
        grid_cols=2,
        macro_sizes=torch.tensor(sizes, dtype=torch.float32),
    )

    moved, accepts, score = _relocation_moves(
        hard.copy(),
        sizes,
        sizes[:, 0] / 2.0,
        sizes[:, 1] / 2.0,
        100.0,
        100.0,
        np.array([True]),
        1,
        None,
        benchmark,
        scorer,
        1.0,
        top_hot=1,
        n_targets=4,
        candidate_allowed=lambda _index, _x, _y: False,
    )

    np.testing.assert_allclose(moved, hard)
    assert accepts == 0
    assert score == 1.0
    assert scorer.exact_batches == 0
    assert _relocation_moves.last_stats["hierarchy_rejects"] >= 1
    assert _relocation_moves.last_stats["scored"] == 0
