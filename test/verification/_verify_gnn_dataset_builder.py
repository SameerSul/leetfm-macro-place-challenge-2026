"""Verify schema-v1 GNN trace dataset building.

uv run python test/verification/_verify_gnn_dataset_builder.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gnn"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from build_gnn_dataset import (  # noqa: E402
    CANDIDATE_FEATURES,
    EDGE_FEATURES,
    MACRO_NET_EDGE_FEATURES,
    NET_NODE_FEATURES,
    NODE_FEATURES,
    build_dataset,
)


def _write_trace(path: Path) -> None:
    rows = [
        {
            "schema_version": 1,
            "time_s": 1.0,
            "event": "hier_relocation_candidates",
            "benchmark": "ibm01",
            "kind": "hard_propose_all",
            "field": "congestion",
            "initial_proxy": 1.0,
            "candidates": [
                {
                    "macro": 0,
                    "candidate_rank": 0,
                    "target_index": 7,
                    "score": 0.9,
                    "local_field": 1.2,
                    "target_field": 0.4,
                    "structural_delta": -0.01,
                    "x": 1.0,
                    "y": 1.0,
                }
            ],
        },
        {
            "schema_version": 1,
            "time_s": 2.0,
            "event": "hier_relocation_result",
            "benchmark": "ibm01",
            "kind": "hard_propose_all",
            "field": "congestion",
            "accepted": [
                {
                    "macro": 0,
                    "target_index": 7,
                    "x": 1.0,
                    "y": 1.0,
                    "new_proxy": 0.99,
                    "proxy_delta": -0.01,
                }
            ],
        },
        {
            "schema_version": 1,
            "time_s": 3.0,
            "event": "hier_decompression_candidate",
            "benchmark": "ibm01",
            "operator": "cluster_decompression",
            "candidate_id": 0,
            "cluster": 0,
            "movable_count": 3,
            "member_count": 4,
            "soft_count": 2,
            "expansion_factor": 1.08,
            "axis_scale": [1.08, 1.02],
            "hierarchy_quality_before": 0.4,
            "hierarchy_quality_after": 0.41,
            "hierarchy_quality_delta": 0.01,
            "old_proxy": 1.0,
            "candidate_proxy": 1.01,
            "proxy_delta": 0.01,
            "accepted": False,
            "rejection_reason": "exact_proxy_failed",
        },
        {
            "schema_version": 1,
            "time_s": 4.0,
            "event": "hier_swap_candidates",
            "benchmark": "ibm01",
            "operator": "region_swaps",
            "kind": "hard_soft",
            "field": "density",
            "source": 1,
            "initial_proxy": 1.0,
            "candidate_count": 1,
            "candidates": [
                {
                    "candidate_rank": 0,
                    "target": 2,
                    "source_field": 1.1,
                    "target_field": 0.3,
                    "outside_region": False,
                    "legal": True,
                    "old_proxy": 1.0,
                    "candidate_proxy": 0.98,
                    "proxy_delta": -0.02,
                    "accepted": True,
                    "rejection_reason": None,
                }
            ],
        },
        {
            "schema_version": 1,
            "time_s": 5.0,
            "event": "hier_coldspot_candidate",
            "benchmark": "ibm01",
            "operator": "coldspot_tightening",
            "kind": "coldspot_kick",
            "field": "congestion",
            "candidate_id": 0,
            "candidate_pool_id": 0,
            "candidate_pool_size": 2,
            "selector_enabled": True,
            "selector_rank": 1,
            "selected_by_gnn": False,
            "is_noop": False,
            "cluster": 0,
            "field_gap": 0.2,
            "min_field_gap": 0.02,
            "old_proxy": 1.0,
            "candidate_proxy": 1.03,
            "proxy_delta": 0.03,
            "hierarchy_quality_before": 0.4,
            "hierarchy_quality_after": 0.39,
            "hierarchy_quality_delta": -0.01,
            "movable_count": 3,
            "member_area": 8.0,
            "cluster_heat": 1.2,
            "anchor_x": 3.0,
            "anchor_y": 4.0,
            "window_microns": 2.0,
            "window_cells": 3,
            "target_density": 0.65,
            "pick": "random",
            "soft_count": 2,
            "soft_moved": 1,
            "hard_disp_mean": 0.4,
            "hard_disp_max": 0.9,
            "soft_disp_mean": 0.2,
            "soft_disp_max": 0.5,
            "cluster_cx_before": 1.0,
            "cluster_cy_before": 2.0,
            "cluster_cx_after": 1.4,
            "cluster_cy_after": 2.2,
            "cluster_bbox_before": [0.0, 0.0, 2.0, 2.0],
            "cluster_bbox_after": [0.0, 0.0, 3.0, 2.0],
            "accepted": False,
            "rejection_reason": "proxy_budget_failed",
        },
    ]
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _args(trace_path: Path, out: Path) -> Namespace:
    return Namespace(
        trace_dir=None,
        trace_path=trace_path,
        out=out,
        benchmark_root=[ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04"],
        benchmark=["ibm01"],
        schema_out=None,
    )


def _assert_equal(a, b, prefix="") -> None:
    if torch.is_tensor(a):
        assert torch.equal(a, b), prefix
    elif isinstance(a, list):
        assert a == b, prefix


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        trace = tmp / "trace.jsonl"
        _write_trace(trace)
        first = build_dataset(_args(trace, tmp / "a.pt"))
        second = build_dataset(_args(trace, tmp / "b.pt"))

    assert first["metadata"]["num_graphs"] == 1
    assert first["metadata"]["num_examples"] == 4
    assert first["metadata"]["num_accepted"] == 2
    graph = first["graphs"][0]
    assert graph["node_features"].shape[1] == len(NODE_FEATURES)
    assert graph["edge_features"].shape[1] == len(EDGE_FEATURES)
    assert graph["edge_index"].shape[0] == 2
    assert graph["net_node_features"].shape[1] == len(NET_NODE_FEATURES)
    assert graph["macro_net_edge_index"].shape[0] == 2
    assert graph["macro_net_edge_features"].shape[1] == len(MACRO_NET_EDGE_FEATURES)
    assert graph["macro_net_edge_index"].shape[1] == graph["macro_net_edge_features"].shape[0]
    examples = first["examples"]
    assert examples["features"].shape == (4, len(CANDIDATE_FEATURES))
    assert examples["accepted"].tolist() == [True, False, True, False]
    assert examples["proxy_delta_known"].tolist() == [True, True, True, True]
    assert examples["candidate_pool_id"].tolist() == [-1, -1, -1, 0]
    assert examples["candidate_id"].tolist() == [0, 0, 0, 0]

    for key in (
        "node_features",
        "edge_index",
        "edge_features",
        "net_node_features",
        "macro_net_edge_index",
        "macro_net_edge_features",
        "macro_cluster",
        "cluster_node",
    ):
        _assert_equal(first["graphs"][0][key], second["graphs"][0][key], key)
    for key, val in examples.items():
        _assert_equal(val, second["examples"][key], key)

    print("GNN DATASET BUILDER VERIFICATION PASSED")


if __name__ == "__main__":
    main()
