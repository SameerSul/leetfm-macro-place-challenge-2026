import json

import pytest

import placer.ml.shadow as shadow
from placer.ml.data_collection import CandidateTrace


@pytest.fixture(autouse=True)
def _reset_shadow(monkeypatch):
    monkeypatch.delenv("ML_MODEL_MANIFEST", raising=False)
    monkeypatch.delenv("ML_SHADOW_TOP_K", raising=False)
    monkeypatch.delenv("ML_FILTER_OPERATORS", raising=False)
    monkeypatch.delenv("ML_FILTER_TOP_K", raising=False)
    monkeypatch.delenv("ML_FILTER_KEEP_HEURISTIC_FIRST", raising=False)
    monkeypatch.delenv("ML_FILTER_PROTECT_HEURISTIC_FIRST", raising=False)
    monkeypatch.delenv("ML_FILTER_MIN_GENERATED", raising=False)
    monkeypatch.delenv("ML_FILTER_MIN_SCORE_GAP", raising=False)
    monkeypatch.delenv("ML_FILTER_CALIBRATION_GROUPS", raising=False)
    monkeypatch.delenv("ML_FILTER_CALIBRATION_MAX_MISSES", raising=False)
    monkeypatch.delenv("ML_FILTER_CALIBRATION_IMPROVING_ONLY", raising=False)
    shadow._reset_shadow_model_bank_for_tests()
    yield
    shadow._reset_shadow_model_bank_for_tests()


def _write_manifest(tmp_path):
    model_path = tmp_path / "linear.json"
    model_path.write_text(
        json.dumps({"intercept": 0.0, "weights": [1.0]}),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "operator": "hard_relocation",
                        "backend": "linear_json",
                        "feature_names": ["dx_norm"],
                        "model_path": model_path.name,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_shadow_rank_group_is_disabled_without_manifest():
    out = shadow.shadow_rank_group(
        operator="hard_relocation",
        candidates=[{"features": {"dx_norm": 1.0}, "score_gain": 1.0}],
    )

    assert out is None


def test_shadow_rank_group_records_model_event(tmp_path, monkeypatch):
    manifest_path = _write_manifest(tmp_path)
    monkeypatch.setenv("ML_MODEL_MANIFEST", str(manifest_path))
    monkeypatch.setenv("ML_SHADOW_TOP_K", "1,2")
    trace = CandidateTrace(str(tmp_path / "trace.jsonl"), flush_rows=1, run_id="run")

    data = shadow.shadow_rank_group(
        operator="hard_relocation",
        candidates=[
            {"features": {"dx_norm": 0.1}, "score_gain": 0.05},
            {"features": {"dx_norm": 0.9}, "score_gain": -0.01},
            {"features": {"dx_norm": 0.2}, "score_gain": 0.10},
        ],
        trace=trace,
        field="congestion",
        group_id="g1",
    )

    assert data["best_exact_model_rank"] == 2
    assert data["best_recall@1"] is False
    assert data["best_recall@2"] is True
    assert data["mean_regret@1"] == pytest.approx(0.11)
    assert data["predicted_top_indices"] == [1, 2, 0]

    row = json.loads((tmp_path / "trace.jsonl").read_text())
    assert row["row_type"] == "event"
    assert row["event"] == "ml_shadow_group"
    assert row["data"]["operator"] == "hard_relocation"
    assert row["data"]["group_id"] == "g1"
    assert row["data"]["model_backend"] == "linear_json"
    assert row["data"]["best_exact_model_rank"] == 2


def test_shadow_rank_group_ignores_missing_operator_model(tmp_path, monkeypatch):
    manifest_path = _write_manifest(tmp_path)
    monkeypatch.setenv("ML_MODEL_MANIFEST", str(manifest_path))

    assert shadow.shadow_rank_group(
        operator="soft_relocation",
        candidates=[{"features": {"dx_norm": 1.0}, "score_gain": 1.0}],
    ) is None


def test_filter_candidate_indices_is_disabled_by_default(tmp_path, monkeypatch):
    manifest_path = _write_manifest(tmp_path)
    monkeypatch.setenv("ML_MODEL_MANIFEST", str(manifest_path))

    selected = shadow.filter_candidate_indices(
        operator="hard_relocation",
        candidates=[
            {"features": {"dx_norm": 0.1}},
            {"features": {"dx_norm": 0.9}},
        ],
    )

    assert selected == [0, 1]


def test_filter_candidate_indices_selects_model_top_k_in_original_order(tmp_path, monkeypatch):
    manifest_path = _write_manifest(tmp_path)
    monkeypatch.setenv("ML_MODEL_MANIFEST", str(manifest_path))
    monkeypatch.setenv("ML_FILTER_OPERATORS", "hard_relocation")
    monkeypatch.setenv("ML_FILTER_TOP_K", "2")
    monkeypatch.setenv("ML_FILTER_KEEP_HEURISTIC_FIRST", "1")
    trace = CandidateTrace(str(tmp_path / "trace.jsonl"), flush_rows=1, run_id="run")

    selected = shadow.filter_candidate_indices(
        operator="hard_relocation",
        candidates=[
            {"features": {"dx_norm": 0.1}},
            {"features": {"dx_norm": 0.9}},
            {"features": {"dx_norm": 0.2}},
        ],
        trace=trace,
        field="congestion",
        group_id="g1",
    )

    assert selected == [0, 1]
    row = json.loads((tmp_path / "trace.jsonl").read_text())
    assert row["event"] == "ml_filter_group"
    assert row["data"]["applied"] is True
    assert row["data"]["generated"] == 3
    assert row["data"]["selected"] == 2
    assert row["data"]["skipped"] == 1
    assert row["data"]["selected_indices"] == [0, 1]


def test_filter_candidate_indices_protects_heuristic_frontier(tmp_path, monkeypatch):
    manifest_path = _write_manifest(tmp_path)
    monkeypatch.setenv("ML_MODEL_MANIFEST", str(manifest_path))
    monkeypatch.setenv("ML_FILTER_OPERATORS", "hard_relocation")
    monkeypatch.setenv("ML_FILTER_TOP_K", "1")
    monkeypatch.setenv("ML_FILTER_KEEP_HEURISTIC_FIRST", "0")
    monkeypatch.setenv("ML_FILTER_PROTECT_HEURISTIC_FIRST", "2")
    trace = CandidateTrace(str(tmp_path / "trace.jsonl"), flush_rows=1, run_id="run")

    selected = shadow.filter_candidate_indices(
        operator="hard_relocation",
        candidates=[
            {"features": {"dx_norm": 0.1}},
            {"features": {"dx_norm": 0.2}},
            {"features": {"dx_norm": 0.9}},
            {"features": {"dx_norm": 0.3}},
        ],
        trace=trace,
        field="congestion",
        group_id="g1",
    )

    assert selected == [0, 1, 2]
    row = json.loads((tmp_path / "trace.jsonl").read_text())
    assert row["data"]["generated"] == 4
    assert row["data"]["selected"] == 3
    assert row["data"]["skipped"] == 1
    assert row["data"]["top_k"] == 1
    assert row["data"]["keep_heuristic_first"] == 0
    assert row["data"]["protect_heuristic_first"] == 2
    assert row["data"]["selected_indices"] == [0, 1, 2]


def test_filter_candidate_indices_skips_small_groups(tmp_path, monkeypatch):
    manifest_path = _write_manifest(tmp_path)
    monkeypatch.setenv("ML_MODEL_MANIFEST", str(manifest_path))
    monkeypatch.setenv("ML_FILTER_OPERATORS", "hard_relocation")
    monkeypatch.setenv("ML_FILTER_TOP_K", "1")
    monkeypatch.setenv("ML_FILTER_MIN_GENERATED", "4")
    trace = CandidateTrace(str(tmp_path / "trace.jsonl"), flush_rows=1, run_id="run")

    selected = shadow.filter_candidate_indices(
        operator="hard_relocation",
        candidates=[
            {"features": {"dx_norm": 0.1}},
            {"features": {"dx_norm": 0.9}},
            {"features": {"dx_norm": 0.2}},
        ],
        trace=trace,
        field="congestion",
        group_id="g1",
    )

    assert selected == [0, 1, 2]
    row = json.loads((tmp_path / "trace.jsonl").read_text())
    assert row["data"]["applied"] is False
    assert row["data"]["reason"] == "below_min_generated"
    assert row["data"]["min_generated"] == 4
    assert row["data"]["selected"] == 3
    assert row["data"]["skipped"] == 0


def test_filter_candidate_indices_skips_low_confidence_groups(tmp_path, monkeypatch):
    manifest_path = _write_manifest(tmp_path)
    monkeypatch.setenv("ML_MODEL_MANIFEST", str(manifest_path))
    monkeypatch.setenv("ML_FILTER_OPERATORS", "hard_relocation")
    monkeypatch.setenv("ML_FILTER_TOP_K", "1")
    monkeypatch.setenv("ML_FILTER_MIN_SCORE_GAP", "0.5")
    trace = CandidateTrace(str(tmp_path / "trace.jsonl"), flush_rows=1, run_id="run")

    selected = shadow.filter_candidate_indices(
        operator="hard_relocation",
        candidates=[
            {"features": {"dx_norm": 0.1}},
            {"features": {"dx_norm": 0.3}},
            {"features": {"dx_norm": 0.2}},
        ],
        trace=trace,
        field="congestion",
        group_id="g1",
    )

    assert selected == [0, 1, 2]
    row = json.loads((tmp_path / "trace.jsonl").read_text())
    assert row["data"]["applied"] is False
    assert row["data"]["reason"] == "below_min_score_gap"
    assert row["data"]["min_score_gap"] == 0.5
    assert row["data"]["score_gap"] == pytest.approx(0.1)
    assert row["data"]["selected"] == 3
    assert row["data"]["skipped"] == 0


def test_filter_candidate_indices_calibrates_before_filtering(tmp_path, monkeypatch):
    manifest_path = _write_manifest(tmp_path)
    monkeypatch.setenv("ML_MODEL_MANIFEST", str(manifest_path))
    monkeypatch.setenv("ML_FILTER_OPERATORS", "hard_relocation")
    monkeypatch.setenv("ML_FILTER_TOP_K", "1")
    monkeypatch.setenv("ML_FILTER_CALIBRATION_GROUPS", "1")
    trace = CandidateTrace(str(tmp_path / "trace.jsonl"), flush_rows=1, run_id="run")
    candidates = [
        {"features": {"dx_norm": 0.1}},
        {"features": {"dx_norm": 0.9}},
        {"features": {"dx_norm": 0.2}},
    ]

    selected = shadow.filter_candidate_indices(
        operator="hard_relocation",
        candidates=candidates,
        trace=trace,
        field="congestion",
        group_id="g1",
    )

    assert selected == [0, 1, 2]
    shadow.update_filter_calibration(
        operator="hard_relocation",
        candidates=[
            {**candidates[0], "score_gain": 0.0},
            {**candidates[1], "score_gain": 1.0},
            {**candidates[2], "score_gain": 0.5},
        ],
        trace=trace,
        field="congestion",
        group_id="g1",
    )

    selected = shadow.filter_candidate_indices(
        operator="hard_relocation",
        candidates=candidates,
        trace=trace,
        field="congestion",
        group_id="g2",
    )

    assert selected == [1]
    rows = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert rows[0]["data"]["reason"] == "calibrating"
    assert rows[1]["event"] == "ml_filter_calibration"
    assert rows[1]["data"]["enabled"] is True
    assert rows[2]["data"]["applied"] is True


def test_filter_candidate_indices_fails_closed_after_calibration_miss(tmp_path, monkeypatch):
    manifest_path = _write_manifest(tmp_path)
    monkeypatch.setenv("ML_MODEL_MANIFEST", str(manifest_path))
    monkeypatch.setenv("ML_FILTER_OPERATORS", "hard_relocation")
    monkeypatch.setenv("ML_FILTER_TOP_K", "1")
    monkeypatch.setenv("ML_FILTER_CALIBRATION_GROUPS", "1")
    trace = CandidateTrace(str(tmp_path / "trace.jsonl"), flush_rows=1, run_id="run")
    candidates = [
        {"features": {"dx_norm": 0.1}},
        {"features": {"dx_norm": 0.9}},
        {"features": {"dx_norm": 0.2}},
    ]

    assert shadow.filter_candidate_indices(
        operator="hard_relocation",
        candidates=candidates,
        trace=trace,
        field="congestion",
        group_id="g1",
    ) == [0, 1, 2]
    shadow.update_filter_calibration(
        operator="hard_relocation",
        candidates=[
            {**candidates[0], "score_gain": 1.0},
            {**candidates[1], "score_gain": 0.0},
            {**candidates[2], "score_gain": 0.5},
        ],
        trace=trace,
        field="congestion",
        group_id="g1",
    )

    selected = shadow.filter_candidate_indices(
        operator="hard_relocation",
        candidates=candidates,
        trace=trace,
        field="congestion",
        group_id="g2",
    )

    assert selected == [0, 1, 2]
    rows = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert rows[1]["data"]["failed"] is True
    assert rows[2]["data"]["reason"] == "calibration_failed"


def test_filter_candidate_indices_falls_back_when_operator_model_missing(tmp_path, monkeypatch):
    manifest_path = _write_manifest(tmp_path)
    monkeypatch.setenv("ML_MODEL_MANIFEST", str(manifest_path))
    monkeypatch.setenv("ML_FILTER_OPERATORS", "soft_relocation")

    selected = shadow.filter_candidate_indices(
        operator="soft_relocation",
        candidates=[
            {"features": {"dx_norm": 0.1}},
            {"features": {"dx_norm": 0.9}},
        ],
    )

    assert selected == [0, 1]
