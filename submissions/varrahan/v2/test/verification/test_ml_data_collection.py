import json

import pytest
import torch

from macro_place.benchmark import Benchmark
import placer.ml.data_collection as data_collection
from placer.ml.data_collection import CandidateTrace
from placer.ml.dataset import add_group_relevance, load_candidates, trace_summary


def _benchmark():
    return Benchmark(
        name="ibm01",
        canvas_width=10.0,
        canvas_height=12.0,
        num_macros=2,
        macro_positions=torch.zeros((2, 2)),
        macro_sizes=torch.ones((2, 2)),
        macro_fixed=torch.zeros(2, dtype=torch.bool),
        macro_names=["h0", "s0"],
        num_nets=1,
        net_nodes=[torch.tensor([0, 1])],
        net_weights=torch.ones(1),
        grid_rows=4,
        grid_cols=5,
        num_hard_macros=1,
        num_soft_macros=1,
    )


def test_candidate_trace_writes_training_grade_ranker_label(tmp_path):
    path = tmp_path / "moves.jsonl"
    trace = CandidateTrace(str(path), flush_rows=1, run_id="test-run")
    trace.start_benchmark(
        benchmark=_benchmark(),
        seed=42,
        config={"time_budget_s": 150.0},
        effective_budget_s=120.0,
        benchmark_index=3,
    )
    trace.set_context(phase="r2", r2_round=2, pass_name="hard_relocation")
    group_id = trace.next_group_id("hard_relocation")
    trace.record(
        operator="hard_relocation",
        field="congestion",
        group_id=group_id,
        state_score=1.25,
        trial_score=1.20,
        candidate_rank=2,
        group_size=8,
        candidate_source="cold_cell",
        features={"dx_norm": 0.1},
    )

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    row = rows[-1]
    assert rows[0]["row_type"] == "event"
    assert rows[0]["event"] == "benchmark_start"
    assert row["schema_version"] == 2
    assert row["row_type"] == "candidate"
    assert row["run_id"] == "test-run"
    assert row["benchmark"] == "ibm01"
    assert row["seed"] == 42
    assert row["num_hard"] == 1
    assert row["num_soft"] == 1
    assert row["config_hash"]
    assert row["context"]["phase"] == "r2"
    assert row["context"]["r2_round"] == 2
    assert row["group_id"] == group_id
    assert row["score_gain"] == pytest.approx(0.05)
    assert row["improves"] is True
    assert row["candidate_rank"] == 2
    assert row["group_size"] == 8
    assert row["candidate_source"] == "cold_cell"
    assert row["features"]["dx_norm"] == 0.1

    flat = load_candidates([path])[0]
    assert flat["num_hard"] == 1
    assert flat["grid_cols"] == 5


def test_candidate_trace_records_group_summary_event(tmp_path):
    path = tmp_path / "moves.jsonl"
    trace = CandidateTrace(str(path), flush_rows=1, run_id="test-run")
    trace.event(
        "candidate_group_summary",
        operator="hard_2opt",
        generated=12,
        scored=4,
        rejected_overlap=8,
    )

    row = json.loads(path.read_text().strip())
    assert row["row_type"] == "event"
    assert row["event"] == "candidate_group_summary"
    assert row["data"]["generated"] == 12
    assert row["data"]["scored"] == 4
    assert row["data"]["rejected_overlap"] == 8


def test_dataset_loader_flattens_candidates_and_adds_group_relevance(tmp_path):
    path = tmp_path / "moves.jsonl"
    trace = CandidateTrace(str(path), flush_rows=1, run_id="test-run")
    trace.start_benchmark(
        benchmark=_benchmark(),
        seed=42,
        config={},
        effective_budget_s=120.0,
        benchmark_index=0,
    )
    group_id = trace.next_group_id("hard_relocation")
    for rank, score in enumerate((1.30, 1.20, 1.25)):
        trace.record(
            operator="hard_relocation",
            field="congestion",
            group_id=group_id,
            state_score=1.30,
            trial_score=score,
            candidate_rank=rank,
            group_size=3,
            candidate_source="cold_cell",
            features={"dx_norm": rank / 10},
        )

    rows = add_group_relevance(load_candidates([path]))
    assert [row["relevance"] for row in rows] == [0, 31, 16]
    assert rows[1]["feature.dx_norm"] == 0.1
    summary = trace_summary([path])
    assert summary["row_types"] == {"event": 1, "candidate": 3}
    assert summary["operators"] == {"hard_relocation": 3}
    assert summary["candidate_groups"] == 1


def test_compressed_trace_round_trip(tmp_path):
    path = tmp_path / "moves.jsonl.gz"
    trace = CandidateTrace(str(path), flush_rows=1, run_id="compressed-run")
    trace.record(
        operator="hard_2opt",
        field="spatial",
        group_id=trace.next_group_id("hard_2opt"),
        state_score=1.0,
        trial_score=0.9,
        features={"distance_norm": 0.2},
    )

    rows = load_candidates([path])
    assert len(rows) == 1
    assert rows[0]["run_id"] == "compressed-run"
    assert rows[0]["feature.distance_norm"] == 0.2


def test_forked_writer_moves_to_child_specific_path(tmp_path, monkeypatch):
    trace = CandidateTrace(str(tmp_path / "moves.jsonl.gz"), run_id="fork-run")
    trace._rows.append({"inherited": True})
    monkeypatch.setattr("placer.ml.data_collection.os.getpid", lambda: 999)

    trace._after_fork_child()

    assert trace.path.name == "moves.jsonl.pid-999.gz"
    assert trace._rows == []


def test_global_trace_is_disabled_without_environment_variable(monkeypatch):
    monkeypatch.delenv("ML_TRACE_PATH", raising=False)
    monkeypatch.setattr(data_collection, "_TRACE", None)
    monkeypatch.setattr(data_collection, "_TRACE_INITIALIZED", False)

    assert data_collection.get_candidate_trace() is None
