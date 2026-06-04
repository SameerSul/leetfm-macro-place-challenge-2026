import json

import pytest

from placer.ml.data_collection import CandidateTrace


def test_candidate_trace_writes_ranker_label(tmp_path):
    path = tmp_path / "moves.jsonl"
    trace = CandidateTrace(str(path), flush_rows=1)
    group_id = trace.next_group_id("hard_relocation")
    trace.record(
        benchmark="ibm01",
        operator="hard_relocation",
        field="congestion",
        group_id=group_id,
        state_score=1.25,
        trial_score=1.20,
        features={"dx_norm": 0.1},
    )

    row = json.loads(path.read_text().strip())
    assert row["group_id"] == group_id
    assert row["score_gain"] == pytest.approx(0.05)
    assert row["improves"] is True
    assert row["features"]["dx_norm"] == 0.1
