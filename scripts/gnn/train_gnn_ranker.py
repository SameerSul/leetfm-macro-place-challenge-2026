#!/usr/bin/env python3
"""Train and evaluate the first Stage-G4 macro-net graph ranker.

The model is intentionally small and CPU-oriented. It consumes the v2 dataset
from `scripts/build_gnn_dataset.py`, encodes macro/cluster nodes plus net nodes,
and scores existing hierarchy candidates offline. It does not integrate with
placement.
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from train_gnn_baseline import (  # noqa: E402
    MlpRanker,
    _benchmark_split,
    _group_metrics,
    _load_dataset,
    _mask_by_benchmark,
    _score_existing_score,
    _score_trace_order,
    _standardize,
    _summarize,
    _train_model,
    _write_json,
)


class MacroNetLayer(nn.Module):
    def __init__(self, hidden: int, edge_features: int) -> None:
        super().__init__()
        self.edge_gate = nn.Sequential(nn.Linear(edge_features, hidden), nn.Sigmoid())
        self.net_update = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU())
        self.node_update = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU())

    def forward(
        self,
        node_h: torch.Tensor,
        net_h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if edge_index.numel() == 0:
            return node_h, net_h
        macro_idx = edge_index[0].long()
        net_idx = edge_index[1].long()
        gate = self.edge_gate(edge_attr)

        net_msg = torch.zeros_like(net_h)
        net_msg.index_add_(0, net_idx, node_h[macro_idx] * gate)
        net_count = torch.zeros((net_h.shape[0], 1), dtype=net_h.dtype, device=net_h.device)
        net_count.index_add_(0, net_idx, torch.ones((net_idx.numel(), 1), device=net_h.device))
        net_msg = net_msg / net_count.clamp_min(1.0)
        net_new = self.net_update(torch.cat([net_h, net_msg], dim=1))

        node_msg = torch.zeros_like(node_h)
        node_msg.index_add_(0, macro_idx, net_new[net_idx] * gate)
        node_count = torch.zeros((node_h.shape[0], 1), dtype=node_h.dtype, device=node_h.device)
        node_count.index_add_(
            0,
            macro_idx,
            torch.ones((macro_idx.numel(), 1), device=node_h.device),
        )
        node_msg = node_msg / node_count.clamp_min(1.0)
        node_new = self.node_update(torch.cat([node_h, node_msg], dim=1))
        return node_new, net_new


class MacroNetRanker(nn.Module):
    def __init__(
        self,
        node_features: int,
        net_features: int,
        edge_features: int,
        candidate_features: int,
        hidden: int,
        layers: int,
    ) -> None:
        super().__init__()
        self.node_in = nn.Sequential(nn.Linear(node_features, hidden), nn.ReLU())
        self.net_in = nn.Sequential(nn.Linear(net_features, hidden), nn.ReLU())
        self.layers = nn.ModuleList(MacroNetLayer(hidden, edge_features) for _ in range(layers))
        self.scorer = nn.Sequential(
            nn.Linear(hidden * 2 + candidate_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def encode_graph(self, graph: dict[str, Any]) -> torch.Tensor:
        node_h = self.node_in(graph["node_features"].float())
        net_h = self.net_in(graph["net_node_features"].float())
        edge_index = graph["macro_net_edge_index"]
        edge_attr = graph["macro_net_edge_features"].float()
        for layer in self.layers:
            node_h, net_h = layer(node_h, net_h, edge_index, edge_attr)
        return node_h

    def forward(self, dataset: dict[str, Any], indices: torch.Tensor) -> torch.Tensor:
        examples = dataset["examples"]
        features = examples["features"].float()
        graph_id = examples["graph_id"].long()
        source = examples["source_node"].long()
        target = examples["target_node"].long()
        out = torch.zeros(indices.numel(), dtype=torch.float32)
        for gid, graph in enumerate(dataset["graphs"]):
            local_mask = graph_id[indices] == gid
            if not bool(local_mask.any()):
                continue
            local_indices = indices[local_mask]
            node_h = self.encode_graph(graph)
            src = source[local_indices].clamp_min(0)
            src_h = node_h[src]
            tgt_raw = target[local_indices]
            tgt_h = torch.zeros_like(src_h)
            has_target = tgt_raw >= 0
            if bool(has_target.any()):
                tgt_h[has_target] = node_h[tgt_raw[has_target]]
            score_in = torch.cat([src_h, tgt_h, features[local_indices]], dim=1)
            out[local_mask] = self.scorer(score_in).squeeze(-1)
        return out


def _code_fingerprint() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _train_graph_model(
    model: MacroNetRanker,
    dataset: dict[str, Any],
    train_idx: torch.Tensor,
    labels: torch.Tensor,
    *,
    epochs: int,
    lr: float,
) -> MacroNetRanker:
    pos = float(labels[train_idx].sum().item())
    neg = float(train_idx.numel() - pos)
    pos_weight = torch.tensor([max(1.0, neg / max(pos, 1.0))], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(max(1, epochs)):
        model.train()
        opt.zero_grad(set_to_none=True)
        scores = model(dataset, train_idx)
        loss = loss_fn(scores, labels[train_idx].float())
        loss.backward()
        opt.step()
    return model.eval()


def _load_g3_scores(
    path: Path | None,
    dataset: dict[str, Any],
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
) -> torch.Tensor | None:
    if path is None:
        return None
    artifact = _load_dataset(path)
    state = artifact.get("models", {}).get("mlp")
    mean = artifact.get("feature_mean")
    std = artifact.get("feature_std")
    if state is None or mean is None or std is None:
        raise ValueError(f"{path} is not a compatible G3 baseline artifact")
    features = dataset["examples"]["features"].float()
    model = MlpRanker(features.shape[1], int(artifact.get("config", {}).get("hidden_size", 32)))
    model.load_state_dict(state)
    model.eval()
    scores = torch.zeros(features.shape[0], dtype=torch.float32)
    with torch.no_grad():
        x_train = (features[train_mask] - mean) / std.clamp_min(1e-6)
        x_val = (features[val_mask] - mean) / std.clamp_min(1e-6)
        scores[train_mask] = model(x_train)
        scores[val_mask] = model(x_val)
    return scores


def train_and_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    dataset = _load_dataset(args.dataset)
    examples = dataset["examples"]
    features = examples["features"].float()
    labels = examples["accepted"].bool()
    benchmarks = [str(b) for b in examples["benchmark"]]
    train_bench, val_bench = _benchmark_split(benchmarks, args.train_benchmark, args.val_benchmark)
    train_mask = _mask_by_benchmark(examples, set(train_bench))
    val_mask = _mask_by_benchmark(examples, set(val_bench))
    train_idx = train_mask.nonzero(as_tuple=False).flatten()
    val_idx = val_mask.nonzero(as_tuple=False).flatten()
    if train_idx.numel() == 0 or val_idx.numel() == 0:
        raise ValueError("split produced no training or validation examples")

    # Train a fresh candidate-only MLP on the same split for an apples-to-apples G3 reference.
    x_train, x_val, _, _ = _standardize(features[train_mask], features[val_mask])
    g3_mlp = _train_model(
        MlpRanker(features.shape[1], args.hidden_size),
        x_train,
        labels[train_mask].float(),
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
    )
    g3_scores = torch.zeros(features.shape[0], dtype=torch.float32)
    with torch.no_grad():
        g3_scores[train_mask] = g3_mlp(x_train)
        g3_scores[val_mask] = g3_mlp(x_val)

    artifact_scores = _load_g3_scores(args.g3_model, dataset, train_mask, val_mask)
    model = MacroNetRanker(
        node_features=dataset["graphs"][0]["node_features"].shape[1],
        net_features=dataset["graphs"][0]["net_node_features"].shape[1],
        edge_features=dataset["graphs"][0]["macro_net_edge_features"].shape[1],
        candidate_features=features.shape[1],
        hidden=args.hidden_size,
        layers=args.layers,
    )
    start = time.time()
    model = _train_graph_model(model, dataset, train_idx, labels, epochs=args.epochs, lr=args.lr)
    runtime_s = time.time() - start
    graph_scores = torch.zeros(features.shape[0], dtype=torch.float32)
    with torch.no_grad():
        graph_scores[train_mask] = model(dataset, train_idx)
        inference_start = time.perf_counter()
        graph_scores[val_mask] = model(dataset, val_idx)
        validation_inference_s = time.perf_counter() - inference_start

    val_list = val_idx.tolist()
    score_sets = {
        "trace_order": _score_trace_order(examples),
        "existing_score": _score_existing_score(examples),
        "g3_mlp_retrained": g3_scores,
        "g4_macro_net": graph_scores,
    }
    if artifact_scores is not None:
        score_sets["g3_mlp_artifact"] = artifact_scores

    metrics = {
        name: {
            "validation": _summarize(examples, val_list, scores, labels),
            "by_operator": _group_metrics(examples, val_list, scores, labels, "operator"),
            "by_benchmark": _group_metrics(examples, val_list, scores, labels, "benchmark"),
        }
        for name, scores in score_sets.items()
    }

    result = {
        "stage": "G4",
        "dataset": str(args.dataset),
        "metadata": dataset.get("metadata", {}),
        "splits": {"train": train_bench, "validation": val_bench, "holdout": []},
        "config": {
            "model": "macro_net_ranker",
            "hidden_size": args.hidden_size,
            "graph_layers": args.layers,
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "seed": args.seed,
            "g3_model": str(args.g3_model) if args.g3_model else None,
        },
        "runtime_s": runtime_s,
        "validation_inference_s": validation_inference_s,
        "metrics": metrics,
        "promotion_decision": "default_off",
        "code_fingerprint": _code_fingerprint(),
    }
    if args.out_dir:
        _write_artifacts(args.out_dir, dataset, model, result)
    return result


def _write_artifacts(
    out_dir: Path,
    dataset: dict[str, Any],
    model: MacroNetRanker,
    result: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "metrics.json", result)
    _write_json(out_dir / "train_config.json", result["config"])
    _write_json(out_dir / "splits.json", result["splits"])
    _write_json(
        out_dir / "trace_manifest.json",
        {
            "trace_files": result.get("metadata", {}).get("trace_files", []),
            "dataset": result["dataset"],
            "code_fingerprint": result["code_fingerprint"],
        },
    )
    _write_json(out_dir / "feature_schema.json", dataset.get("feature_schema", {}))
    torch.save(
        {
            "model": model.state_dict(),
            "config": result["config"],
            "feature_schema": dataset.get("feature_schema", {}),
        },
        out_dir / "model.pt",
    )
    summary = result["metrics"]["g4_macro_net"]["validation"]
    readme = [
        "# Stage G4 Macro-Net Ranker",
        "",
        "Default-off offline graph-ranker artifact. It is not integrated with placement.",
        "",
        f"- Dataset: `{result['dataset']}`",
        f"- Train benchmarks: {', '.join(result['splits']['train'])}",
        f"- Validation benchmarks: {', '.join(result['splits']['validation'])}",
        f"- Validation top-4 recall: `{summary['top4_recall']:.4f}`",
        f"- Validation MRR: `{summary['mrr']:.4f}`",
        "- Promotion decision: `default_off`.",
        "",
        "See `metrics.json` for operator-level comparisons against G3.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--g3-model", type=Path, help="Optional accepted G3 model.pt artifact")
    parser.add_argument("--train-benchmark", action="append")
    parser.add_argument("--val-benchmark", action="append")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    result = train_and_evaluate(args)
    g4 = result["metrics"]["g4_macro_net"]["validation"]
    g3 = result["metrics"]["g3_mlp_retrained"]["validation"]
    print(
        "G4 validation: "
        f"g4_top4={g4['top4_recall']:.4f}, "
        f"g3_top4={g3['top4_recall']:.4f}, "
        f"g4_mrr={g4['mrr']:.4f}, "
        f"g3_mrr={g3['mrr']:.4f}"
    )


if __name__ == "__main__":
    main()
