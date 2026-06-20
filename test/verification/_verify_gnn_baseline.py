"""Verify Stage-G3 candidate baseline training.

uv run python test/verification/_verify_gnn_baseline.py
"""

from __future__ import annotations

import sys
import tempfile
from argparse import Namespace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gnn"))
sys.path.insert(0, str(ROOT))

from train_gnn_baseline import train_and_evaluate  # noqa: E402


def _write_dataset(path: Path) -> None:
    rows = []
    accepted = []
    benchmarks = []
    operators = []
    kinds = []
    trace_files = []
    trace_lines = []
    source_nodes = []
    target_nodes = []
    proxy_delta = []
    proxy_delta_known = []
    rejection_id = []

    for bench_i, benchmark in enumerate(("ibm01", "ibm10")):
        for pool_i in range(10):
            for rank in range(5):
                is_good = rank == 0
                score = 1.0 - rank * 0.2
                rows.append(
                    [
                        1.0,
                        1.0,
                        1.0,
                        rank / 512.0,
                        score,
                        0.1 * rank,
                        -score,
                    ]
                    + [0.0] * 20
                )
                accepted.append(is_good)
                benchmarks.append(benchmark)
                operators.append("relocation" if pool_i % 2 == 0 else "region_swaps")
                kinds.append("hard_propose_all")
                trace_files.append(f"{benchmark}.jsonl")
                trace_lines.append(bench_i * 100 + pool_i)
                source_nodes.append(pool_i)
                target_nodes.append(rank)
                proxy_delta.append(-score if is_good else score)
                proxy_delta_known.append(True)
                rejection_id.append(0 if is_good else 9)

    payload = {
        "metadata": {
            "dataset_schema_version": 1,
            "trace_schema_version": 1,
            "trace_files": ["ibm01.jsonl", "ibm10.jsonl"],
            "benchmarks": ["ibm01", "ibm10"],
            "num_graphs": 2,
            "num_examples": len(rows),
            "num_accepted": sum(accepted),
        },
        "feature_schema": {
            "candidate_features": [f"f{i}" for i in range(27)],
            "node_features": [],
            "edge_features": [],
            "operator_ids": {"relocation": 1, "region_swaps": 3},
        },
        "graphs": [],
        "examples": {
            "features": torch.tensor(rows, dtype=torch.float32),
            "accepted": torch.tensor(accepted, dtype=torch.bool),
            "benchmark": benchmarks,
            "operator": operators,
            "kind": kinds,
            "trace_file": trace_files,
            "trace_line": torch.tensor(trace_lines, dtype=torch.long),
            "source_node": torch.tensor(source_nodes, dtype=torch.long),
            "target_node": torch.tensor(target_nodes, dtype=torch.long),
            "proxy_delta": torch.tensor(proxy_delta, dtype=torch.float32),
            "proxy_delta_known": torch.tensor(proxy_delta_known, dtype=torch.bool),
            "rejection_id": torch.tensor(rejection_id, dtype=torch.long),
        },
    }
    torch.save(payload, path)


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        dataset = tmp / "dataset.pt"
        out_dir = tmp / "model"
        _write_dataset(dataset)
        result = train_and_evaluate(
            Namespace(
                dataset=dataset,
                out_dir=out_dir,
                train_benchmark=["ibm01"],
                val_benchmark=["ibm10"],
                holdout_benchmark=[],
                epochs=30,
                lr=5e-3,
                hidden_size=8,
                seed=3,
            )
        )

        assert result["stage"] == "G3"
        assert result["splits"]["train"] == ["ibm01"]
        assert result["splits"]["validation"] == ["ibm10"]
        for model in ("trace_order", "existing_score", "logistic", "mlp"):
            assert model in result["metrics"]
            assert result["metrics"][model]["validation"]["examples"] == 50.0
            assert "relocation" in result["metrics"][model]["by_operator"]
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

    print("GNN BASELINE VERIFICATION PASSED")


if __name__ == "__main__":
    main()
