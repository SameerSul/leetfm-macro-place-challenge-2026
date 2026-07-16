import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from analyze_plateau_telemetry import aggregate, load_rows
from placer.local_search import gnn_trace
from utils import constants

REMOVED_DEFAULT_ON_GATES = (
    "HIER_MEDIUM_SOFT_CONTINUATION",
    "HIER_REGION_COMPONENT_EXPAND",
    "HIER_REGION_WEAK_HOT_REQUIRE_RELEASE_CANDIDATE",
    "HIER_DECOMPRESS_LOCAL_COMPONENT",
    "HIER_DECOMPRESS_FEASIBILITY_FILTER",
    "HIER_DECOMPRESS_GRAPH_SURVIVOR",
    "HIER_GRAPH_TENSION_ORDER",
    "HIER_SWAP_GRAPH_MASK_AWARE",
    "HIER_SWAP_GRAPH_FALLBACK",
    "HIER_FINAL_HIER_AUDIT_ROLLBACK",
    "HIER_SMALL_DESIGN_POLISH",
    "HIER_COLD_COMPONENT_TARGETS",
)
REMOVED_DEFAULT_ON_ENV_GATES = (
    "HIER_DREAMPLACE_BB",
    "HIER_DREAMPLACE_CACHE",
    "HIER_ADAPTIVE_PASSES",
    "HIER_PLATEAU_TRACE",
    "HIER_PLATEAU_TRACE_BUFFERED",
    "HIER_GNN_COLDSPOT_SKIP_MICRO",
)


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


def test_promoted_production_features_no_longer_have_boolean_gates():
    for name in REMOVED_DEFAULT_ON_GATES:
        assert not hasattr(constants, name)


def test_production_source_no_longer_reads_default_on_env_gates():
    source = "\n".join(path.read_text() for path in (ROOT / "src").rglob("*.py"))
    for name in REMOVED_DEFAULT_ON_ENV_GATES:
        assert f'"{name}"' not in source
        assert f"'{name}'" not in source


def test_plateau_telemetry_always_records_even_with_legacy_disable_env(tmp_path, monkeypatch):
    path = tmp_path / "plateau.jsonl"
    monkeypatch.setenv("HIER_PLATEAU_TRACE", "0")
    monkeypatch.setenv("HIER_PLATEAU_TRACE_PATH", str(path))
    gnn_trace._PLATEAU_BUFFER.clear()
    gnn_trace._PLATEAU_BUFFER_PATH = None

    gnn_trace.log_plateau_event("hier_plateau_telemetry", benchmark="test")
    gnn_trace.flush_plateau_events()

    row = json.loads(path.read_text())
    assert row["event"] == "hier_plateau_telemetry"
    assert row["benchmark"] == "test"
