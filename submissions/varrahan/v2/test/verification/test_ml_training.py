import json
import subprocess
import sys

import pytest

from placer.ml.data_collection import CandidateTrace
from placer.ml.dataset import flatten_candidate
from placer.ml.modeling import ModelBank
from placer.ml.train import ranking_metrics, split_rows


def _write_candidate(trace, *, benchmark, run_id, group, rank, gain, dx):
    trace.run_id = run_id
    trace._benchmark = {
        "benchmark": benchmark,
        "benchmark_index": 0,
        "seed": 1,
        "effective_budget_s": 120.0,
        "num_hard": 1,
        "num_soft": 1,
        "num_nets": 1,
        "grid_rows": 4,
        "grid_cols": 4,
        "canvas_width": 10.0,
        "canvas_height": 10.0,
        "config_hash": "cfg",
    }
    state = 1.0
    trace.record(
        operator="hard_relocation",
        field="congestion",
        group_id=group,
        state_score=state,
        trial_score=state - gain,
        candidate_rank=rank,
        group_size=3,
        candidate_source="cold_cell",
        features={
            "accepted_in_pass": 0,
            "source_hot_rank_norm": 0.0,
            "net_degree": 1,
            "net_degree_log1p": 0.693,
            "net_degree_norm": 1.0,
            "macro_w_norm": 0.1,
            "macro_h_norm": 0.1,
            "x_norm": 0.1,
            "y_norm": 0.1,
            "target_x_norm": 0.2 + dx,
            "target_y_norm": 0.2,
            "dx_norm": dx,
            "dy_norm": 0.0,
            "source_field_norm": 0.8,
            "target_field_norm": 0.2,
            "source_congestion_norm": 0.8,
            "target_congestion_norm": 0.2,
            "source_density_norm": 0.5,
            "target_density_norm": 0.2,
            "target_cold_rank_norm": rank / 3,
        },
    )


def _make_trace(path):
    trace = CandidateTrace(str(path), flush_rows=1, run_id="run-a")
    for run_id, bench in (("run-a", "ibm01"), ("run-b", "ibm02"), ("run-c", "ng45_a")):
        for group_i in range(4):
            group = f"{run_id}:hard_relocation:{group_i}"
            for rank, gain in enumerate((-0.01, 0.01, 0.03)):
                _write_candidate(
                    trace,
                    benchmark=bench,
                    run_id=run_id,
                    group=group,
                    rank=rank,
                    gain=gain,
                    dx=float(rank) / 10,
                )
    trace.flush()


def test_split_rows_holds_out_ng_benchmarks(tmp_path):
    path = tmp_path / "trace.jsonl"
    _make_trace(path)
    rows = []
    for line in path.read_text().splitlines():
        rows.append(flatten_candidate(json.loads(line)))

    split = split_rows(rows, seed=1, valid_fraction=0.5, test_benchmark_prefix="ng")

    assert {row["benchmark"] for row in split["test"]} == {"ng45_a"}
    assert not ({row["run_id"] for row in split["train"]} & {row["run_id"] for row in split["valid"]})


def test_ranking_metrics_reports_recall_and_regret():
    rows = [
        {"run_id": "r", "group_id": "g", "score_gain": -1.0},
        {"run_id": "r", "group_id": "g", "score_gain": 2.0},
        {"run_id": "r", "group_id": "g", "score_gain": 1.0},
    ]
    metrics = ranking_metrics(rows, [0.0, 0.1, 0.9], top_ks=(1, 2))

    assert metrics["groups"] == 1
    assert metrics["improving_recall@1"] == 1.0
    assert metrics["best_recall@1"] == 0.0
    assert metrics["mean_regret@1"] == pytest.approx(1.0)
    assert metrics["best_recall@2"] == 1.0


def test_training_cli_writes_xgboost_manifest(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    out_dir = tmp_path / "models"
    _make_trace(trace_path)

    cmd = [
        sys.executable,
        "-m",
        "placer.ml.train",
        str(trace_path),
        "--output-dir",
        str(out_dir),
        "--operators",
        "hard_relocation",
        "--objective",
        "regressor",
        "--rounds",
        "3",
        "--top-k",
        "1,2",
        "--seed",
        "1",
    ]
    subprocess.run(cmd, check=True)

    manifest = json.loads((out_dir / "manifest.json").read_text())
    metrics = json.loads((out_dir / "metrics.json").read_text())
    assert manifest["models"][0]["operator"] == "hard_relocation"
    assert (out_dir / manifest["models"][0]["model_path"]).exists()
    assert metrics["operators"]["hard_relocation"]["status"] == "trained"

    bank = ModelBank.from_manifest(out_dir / "manifest.json")
    scores = bank.require("hard_relocation").scores(
        [
            {
                "features": {
                    "accepted_in_pass": 0,
                    "source_hot_rank_norm": 0.0,
                    "dx_norm": 0.3,
                }
            }
        ]
    )
    assert len(scores) == 1
