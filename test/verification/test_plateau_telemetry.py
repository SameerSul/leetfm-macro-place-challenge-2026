import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_plateau_telemetry import aggregate, load_rows


def test_plateau_analysis_filters_provenance_and_flags_repeated_zero_yield(tmp_path):
    path = tmp_path / "trace.jsonl"
    rows = [
        {
            "event": "hier_plateau_telemetry",
            "run_id": "wanted",
            "code_revision": "abc",
            "benchmark": "ibm01",
            "plateau_pass": "dead_pass",
            "proxy_gain": 0.0,
            "elapsed_s": 1.0,
            "accepts": 0,
            "candidates": 10,
        },
        {
            "event": "hier_plateau_telemetry",
            "run_id": "other",
            "code_revision": "abc",
            "benchmark": "ibm01",
            "plateau_pass": "productive",
            "proxy_gain": 0.1,
            "elapsed_s": 1.0,
            "accepts": 1,
            "candidates": 10,
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    args = SimpleNamespace(run_id="wanted", revision=None, benchmark=None)

    summary = aggregate(load_rows([path], args), min_runs=1)

    assert [row["pass"] for row in summary] == ["dead_pass"]
    assert summary[0]["recommendation"] == "skip_candidate"
