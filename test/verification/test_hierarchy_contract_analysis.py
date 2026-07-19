import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_hierarchy_contract import aggregate_contract, load_contract_rows


def _row(
    benchmark,
    compactness,
    *,
    event="hierarchy_contract_audit",
    selected=True,
    provenance="inferred",
):
    reference = {
        "cluster_compactness": 0.10,
        "worst_cluster_spread": 0.20,
        "neighbor_impurity": 0.30,
        "edge_stretch": 0.10,
        "owned_soft_distance": 0.10,
        "bridge_soft_distance": 0.10,
    }
    vector = {
        **reference,
        "cluster_compactness": compactness,
        "clustered_hard_count": 10.0,
        "clustered_hard_fraction": 1.0,
        "edge_count": 1.0,
        "owned_soft_count": 2.0,
        "bridge_soft_count": 1.0,
        "soft_coverage": 1.0,
    }
    return {
        "event": event,
        "stage": "final",
        "benchmark": benchmark,
        "selected": selected,
        "coverage_scope": "high",
        "hierarchy_provenance": provenance,
        "vector": vector,
        "reference_vector": reference,
        "limits": {key: value + 0.02 for key, value in reference.items()},
    }


def test_contract_summary_reports_headroom_and_replayed_failure():
    rows = [_row("safe", 0.11), _row("tight", 0.119)]

    summary = aggregate_contract(
        rows,
        absolute_slack={
            "cluster_compactness": 0.015,
            "worst_cluster_spread": 0.015,
            "neighbor_impurity": 0.05,
            "edge_stretch": 0.015,
            "owned_soft_distance": 0.015,
            "bridge_soft_distance": 0.015,
        },
        relative_slack=0.0,
    )

    assert summary["passed"] == 1
    assert summary["failed"] == 1
    assert summary["selected_passed"] == 1
    assert summary["selected_failed"] == 1
    assert summary["selected_violation_components"] == {"cluster_compactness": 1}
    assert summary["selected_failures"] == [
        {
            "benchmark": "tight",
            "candidate": "final",
            "violations": ["cluster_compactness"],
        }
    ]
    compactness = summary["components"]["cluster_compactness"]
    assert compactness["tightest_benchmark"] == "tight"
    assert compactness["near_limit_rows"] == 1
    assert compactness["violation_rows"] == 1
    assert abs(compactness["required_absolute_at_relative"] - 0.019) < 1.0e-12


def test_contract_loader_filters_event_stage_and_benchmark(tmp_path):
    path = tmp_path / "trace.jsonl"
    rows = [
        _row("wanted", 0.11),
        _row("other", 0.11),
        _row("wanted", 0.11, event="hierarchy_truth_audit"),
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    args = SimpleNamespace(
        event="contract",
        stage="final",
        benchmark=["wanted"],
        run_id=None,
        revision=None,
        worktree_fingerprint=None,
    )

    loaded = load_contract_rows([path], args)

    assert len(loaded) == 1
    assert loaded[0]["benchmark"] == "wanted"


def test_contract_summary_separates_rejected_candidates_from_selected_failures():
    rows = [
        _row("design", 0.11, selected=True, provenance="hierarchy_path_tags"),
        _row("design", 0.13, selected=False, provenance="hierarchy_oversized_connectivity"),
    ]

    summary = aggregate_contract(
        rows,
        absolute_slack={
            "cluster_compactness": 0.015,
            "worst_cluster_spread": 0.015,
            "neighbor_impurity": 0.05,
            "edge_stretch": 0.015,
            "owned_soft_distance": 0.015,
            "bridge_soft_distance": 0.015,
        },
        relative_slack=0.0,
    )

    assert summary["failed"] == 1
    assert summary["selected_passed"] == 1
    assert summary["selected_failed"] == 0
    assert summary["provenance"] == {"explicit": 1, "inferred": 1}
