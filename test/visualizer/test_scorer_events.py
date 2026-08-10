import numpy as np
import pytest
import torch

from macro_place.loader import load_benchmark_from_dir
from placer.scoring.exact import _exact_proxy
from placer.scoring.incremental import IncrementalScorer


@pytest.fixture(scope="module")
def ibm01_dir():
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "external/MacroPlacement/Testcases/ICCAD04/ibm01"
    if not (path / "netlist.pb.txt").exists():
        pytest.skip("ICCAD04 submodule is unavailable")
    return path


def _load(path):
    benchmark, plc = load_benchmark_from_dir(path.as_posix())
    positions = benchmark.macro_positions.numpy().astype(np.float64)
    _exact_proxy(torch.tensor(positions, dtype=torch.float32), benchmark, plc)
    return benchmark, plc, positions


def test_commit_primitives_emit_output_indices_and_exact_metrics(ibm01_dir):
    benchmark, plc, positions = _load(ibm01_dir)
    rows = []
    scorer = IncrementalScorer(plc, benchmark, positions, event_sink=rows.append)
    hard = next(i for i in range(benchmark.num_hard_macros) if not benchmark.macro_fixed[i])
    old = scorer.committed_hard_pos[hard].copy()
    scorer.commit_move(hard, old + [0.001, 0.001])
    assert rows[-1]["indices"] == [hard]
    assert rows[-1]["move_kind"] == "hard_move"

    prep_target = scorer.committed_hard_pos[hard] + [0.001, 0.001]
    prep = scorer._prepare_move(hard)
    scorer._commit_after_prep(prep, prep_target)
    assert rows[-1]["indices"] == [hard]

    if benchmark.num_soft_macros >= 2:
        scorer.commit_swap_soft_soft(0, 1)
        assert rows[-1]["indices"] == [benchmark.num_hard_macros, benchmark.num_hard_macros + 1]
        assert rows[-1]["move_kind"] == "soft_soft_swap"
    if benchmark.num_hard_macros >= 2:
        scorer.commit_swap_hard_hard(0, 1)
        assert rows[-1]["indices"] == [0, 1]
        assert rows[-1]["move_kind"] == "hard_hard_swap"
    if benchmark.num_soft_macros:
        scorer.commit_swap_hard_soft(hard, 0)
        assert rows[-1]["indices"] == [hard, benchmark.num_hard_macros]
        assert rows[-1]["move_kind"] == "hard_soft_swap"

    full = np.vstack([scorer.committed_hard_pos, scorer.committed_soft_pos]).astype(np.float32)
    exact_proxy = float(_exact_proxy(torch.tensor(full), benchmark, plc))
    assert rows[-1]["metrics"]["proxy"] == pytest.approx(exact_proxy, abs=1e-7)


def test_disabled_sink_leaves_committed_state_unchanged(ibm01_dir):
    benchmark_a, plc_a, positions_a = _load(ibm01_dir)
    benchmark_b, plc_b, positions_b = _load(ibm01_dir)
    scorer_a = IncrementalScorer(plc_a, benchmark_a, positions_a, event_sink=None)
    scorer_b = IncrementalScorer(plc_b, benchmark_b, positions_b, event_sink=lambda _event: None)
    hard = next(i for i in range(benchmark_a.num_hard_macros) if not benchmark_a.macro_fixed[i])
    target = scorer_a.committed_hard_pos[hard] + [0.001, 0.001]
    scorer_a.commit_move(hard, target)
    scorer_b.commit_move(hard, target)
    np.testing.assert_array_equal(scorer_a.committed_hard_pos, scorer_b.committed_hard_pos)
    np.testing.assert_array_equal(scorer_a.committed_soft_pos, scorer_b.committed_soft_pos)
    assert scorer_a.total_wl_raw == scorer_b.total_wl_raw


def test_visualizer_metrics_accept_hierarchy_contract_mapping(ibm01_dir):
    benchmark, plc, positions = _load(ibm01_dir)
    scorer = IncrementalScorer(
        plc,
        benchmark,
        positions,
        hierarchy_metric=lambda _hard, _soft: {
            "hierarchy": 0.25,
            "hierarchy_hard_containment": 0.50,
            "hierarchy_hard_limit": 0.60,
            "hierarchy_hard_headroom": 0.10,
        },
    )

    metrics = scorer.visualizer_metrics()

    assert metrics["hierarchy"] == 0.25
    assert metrics["hierarchy_hard_containment"] == 0.50
    assert metrics["hierarchy_hard_limit"] == 0.60
    assert metrics["hierarchy_hard_headroom"] == 0.10
