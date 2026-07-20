import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from analyze_plateau_telemetry import (
    aggregate,
    aggregate_coverage,
    aggregate_quotas,
    aggregate_stages,
    load_rows,
    load_stage_rows,
)
from placer.local_search import plateau_telemetry
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
    assert summary[0]["retained_gain_per_scored"] is None


def test_stage_analysis_aggregates_stage_timings(tmp_path):
    path = tmp_path / "trace.jsonl"
    rows = [
        {
            "event": "hier_stage_timing",
            "run_id": "wanted",
            "code_revision": "abc",
            "worktree_fingerprint": "fingerprint",
            "benchmark": "ibm01",
            "stage": "seed_prescore",
            "elapsed_s": 1.25,
        },
        {
            "event": "hier_stage_timing",
            "run_id": "wanted",
            "code_revision": "abc",
            "worktree_fingerprint": "fingerprint",
            "benchmark": "ibm02",
            "stage": "seed_prescore",
            "elapsed_s": 0.75,
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    args = SimpleNamespace(
        run_id="wanted",
        revision=None,
        worktree_fingerprint="fingerprint",
        benchmark=None,
    )

    summary = aggregate_stages(load_stage_rows([path], args), min_runs=1)

    assert summary == [
        {
            "stage": "seed_prescore",
            "runs": 2,
            "benchmarks": 2,
            "elapsed_s": 2.0,
            "average_s": 1.0,
            "maximum_s": 1.25,
        }
    ]


def test_coverage_analysis_reconciles_exclusive_phases_to_api_boundary():
    rows = [
        {
            "run_id": "run",
            "benchmark": "ibm01",
            "stage": "placer_api_total",
            "elapsed_s": 10.0,
        },
        {
            "run_id": "run",
            "benchmark": "ibm01",
            "stage": "hierarchy_floorplan_total",
            "elapsed_s": 9.5,
        },
    ]
    rows.extend(
        {
            "run_id": "run",
            "benchmark": "ibm01",
            "stage": stage,
            "elapsed_s": elapsed,
        }
        for stage, elapsed in (
            ("hierarchy_setup_total", 1.0),
            ("seed_portfolio_total", 2.0),
            ("hierarchy_search_total", 4.0),
            ("coldspot_total", 1.0),
            ("post_coldspot_total", 1.0),
        )
    )

    summary = aggregate_coverage(rows, min_runs=1)

    assert len(summary) == 1
    assert summary[0]["phase_s"] == 9.0
    assert summary[0]["floorplan_gap_s"] == 0.5
    assert summary[0]["api_boundary_s"] == 0.5
    assert summary[0]["phase_coverage"] == 9.0 / 9.5


def test_quota_analysis_aggregates_repeated_passes_per_benchmark():
    rows = [
        {
            "benchmark": "ibm01",
            "plateau_pass": "region_soft_relocation",
            "scored": 7,
            "elapsed_s": 0.2,
            "retained_proxy_gain": 0.01,
            "exact_quota_limit": 12,
            "exact_quota_exhausted": False,
        },
        {
            "benchmark": "ibm01",
            "plateau_pass": "region_soft_relocation",
            "scored": 5,
            "elapsed_s": 0.1,
            "retained_proxy_gain": 0.02,
            "exact_quota_limit": 12,
            "exact_quota_exhausted": True,
        },
        {
            "benchmark": "ibm02",
            "plateau_pass": "region_soft_relocation",
            "scored": 4,
            "elapsed_s": 0.1,
            "retained_proxy_gain": 0.03,
            "exact_quota_limit": 12,
            "exact_quota_exhausted": False,
        },
    ]

    summary = aggregate_quotas(rows, min_runs=1)

    assert len(summary) == 1
    assert summary[0]["configured_limit"] == 12
    assert summary[0]["maximum_scored"] == 12
    assert summary[0]["total_scored"] == 16
    assert summary[0]["exhausted_benchmarks"] == 1


def test_promoted_production_features_no_longer_have_boolean_gates():
    for name in REMOVED_DEFAULT_ON_GATES:
        assert not hasattr(constants, name)


def test_production_source_no_longer_reads_default_on_env_gates():
    source = "\n".join(path.read_text() for path in (ROOT / "src").rglob("*.py"))
    for name in REMOVED_DEFAULT_ON_ENV_GATES:
        assert f'"{name}"' not in source
        assert f"'{name}'" not in source


def test_learned_ranker_hooks_are_absent_from_production_source():
    source = "\n".join(path.read_text() for path in (ROOT / "src").rglob("*.py"))
    assert "HIER_GNN_" not in source
    assert "gnn_ranker" not in source
    assert "gnn_trace" not in source


def test_plateau_telemetry_always_records_even_with_legacy_disable_env(tmp_path, monkeypatch):
    path = tmp_path / "plateau.jsonl"
    monkeypatch.setenv("HIER_PLATEAU_TRACE", "0")
    monkeypatch.setenv("HIER_PLATEAU_TRACE_PATH", str(path))
    plateau_telemetry._PLATEAU_BUFFER.clear()
    plateau_telemetry._PLATEAU_BUFFER_PATH = None

    plateau_telemetry.log_plateau_event("hier_plateau_telemetry", benchmark="test")
    plateau_telemetry.flush_plateau_events()

    row = json.loads(path.read_text())
    assert row["event"] == "hier_plateau_telemetry"
    assert row["benchmark"] == "test"


def test_plateau_telemetry_records_worktree_fingerprint_override(tmp_path, monkeypatch):
    path = tmp_path / "plateau.jsonl"
    monkeypatch.setenv("HIER_PLATEAU_TRACE_PATH", str(path))
    monkeypatch.setenv("VIVAPLACE_WORKTREE_FINGERPRINT", "test-dirty-state")
    plateau_telemetry._PLATEAU_BUFFER.clear()
    plateau_telemetry._PLATEAU_BUFFER_PATH = None
    plateau_telemetry._WORKTREE_PROVENANCE = None

    plateau_telemetry.log_plateau_event("hier_plateau_telemetry", benchmark="test")
    plateau_telemetry.flush_plateau_events()

    row = json.loads(path.read_text())
    assert row["worktree_dirty"] is True
    assert row["worktree_fingerprint"] == "test-dirty-state"
    plateau_telemetry._WORKTREE_PROVENANCE = None
