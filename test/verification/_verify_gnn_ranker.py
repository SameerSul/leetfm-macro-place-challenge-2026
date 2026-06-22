"""Verify Stage-G4 macro-net graph ranker training.

uv run python test/verification/_verify_gnn_ranker.py
"""

from __future__ import annotations

import sys
import tempfile
from argparse import Namespace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gnn"))

from train_gnn_ranker import train_and_evaluate  # noqa: E402


def _graph(name: str) -> dict:
    return {
        "benchmark": name,
        "canvas": torch.tensor([10.0, 10.0]),
        "num_hard_macros": 2,
        "num_soft_macros": 0,
        "num_macros": 2,
        "num_clusters": 0,
        "node_features": torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 1.0] + [0.1] * 13,
                [1.0, 0.0, 0.0, 0.0, 1.0] + [0.8] * 13,
            ],
            dtype=torch.float32,
        ),
        "edge_index": torch.zeros((2, 0), dtype=torch.long),
        "edge_features": torch.zeros((0, 6), dtype=torch.float32),
        "net_node_features": torch.tensor([[1.0, 1.0, 1.0, 0.2, 0.2, 0.4]]),
        "macro_net_edge_index": torch.tensor([[0, 1], [0, 0]], dtype=torch.long),
        "macro_net_edge_features": torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        "macro_cluster": torch.tensor([-1, -1], dtype=torch.long),
        "cluster_node": torch.zeros(0, dtype=torch.long),
        "bridge_softs": {},
        "macro_names": ["a", "b"],
    }


def _write_dataset(path: Path) -> None:
    rows = []
    accepted = []
    graph_id = []
    benchmark = []
    source = []
    target = []
    operator = []
    kind = []
    trace_file = []
    trace_line = []
    for gid, name in enumerate(("train_bench", "val_bench")):
        for pool in range(8):
            for rank in range(4):
                is_good = rank == 0
                rows.append([0.0, 0.0, 0.0, rank / 512.0, 1.0 - rank * 0.2] + [0.0] * 22)
                accepted.append(is_good)
                graph_id.append(gid)
                benchmark.append(name)
                source.append(pool % 2)
                target.append((pool + 1) % 2)
                operator.append("region_swaps")
                kind.append("hard_hard")
                trace_file.append(f"{name}.jsonl")
                trace_line.append(pool)
    torch.save(
        {
            "metadata": {
                "dataset_schema_version": 2,
                "trace_schema_version": 1,
                "benchmarks": ["train_bench", "val_bench"],
                "num_graphs": 2,
                "num_examples": len(rows),
                "num_accepted": sum(accepted),
            },
            "feature_schema": {
                "node_features": [f"n{i}" for i in range(18)],
                "net_node_features": [f"net{i}" for i in range(6)],
                "macro_net_edge_features": [f"e{i}" for i in range(7)],
                "candidate_features": [f"c{i}" for i in range(27)],
            },
            "graphs": [_graph("train_bench"), _graph("val_bench")],
            "examples": {
                "graph_id": torch.tensor(graph_id, dtype=torch.long),
                "source_node": torch.tensor(source, dtype=torch.long),
                "target_node": torch.tensor(target, dtype=torch.long),
                "features": torch.tensor(rows, dtype=torch.float32),
                "accepted": torch.tensor(accepted, dtype=torch.bool),
                "proxy_delta": torch.zeros(len(rows), dtype=torch.float32),
                "proxy_delta_known": torch.ones(len(rows), dtype=torch.bool),
                "rejection_id": torch.zeros(len(rows), dtype=torch.long),
                "operator": operator,
                "kind": kind,
                "benchmark": benchmark,
                "trace_file": trace_file,
                "trace_line": torch.tensor(trace_line, dtype=torch.long),
            },
        },
        path,
    )


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        dataset = tmp / "dataset.pt"
        out_dir = tmp / "g4"
        _write_dataset(dataset)
        result = train_and_evaluate(
            Namespace(
                dataset=dataset,
                out_dir=out_dir,
                g3_model=None,
                train_benchmark=["train_bench"],
                val_benchmark=["val_bench"],
                epochs=5,
                lr=1e-3,
                hidden_size=8,
                layers=1,
                seed=5,
            )
        )
        assert result["stage"] == "G4"
        assert "g4_macro_net" in result["metrics"]
        assert result["metrics"]["g4_macro_net"]["validation"]["examples"] == 32.0
        assert result["validation_inference_s"] >= 0.0
        for name in (
            "feature_schema.json",
            "train_config.json",
            "splits.json",
            "trace_manifest.json",
            "metrics.json",
            "model.pt",
            "README.md",
        ):
            assert (out_dir / name).exists(), name
    print("GNN RANKER VERIFICATION PASSED")


if __name__ == "__main__":
    main()
