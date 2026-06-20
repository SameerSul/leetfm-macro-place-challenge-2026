#!/usr/bin/env python3
"""Train and evaluate Stage-G3 candidate-feature baselines.

This script consumes the framework-neutral dataset from
`scripts/build_gnn_dataset.py`. It does not run placement and does not enable
inference. The goal is to prove whether schema-v1 candidate labels are learnable
before adding graph-model complexity.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


DEFAULT_TOP_K = (1, 4, 8, 16)


def _load_dataset(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _as_list(values: Any) -> list[Any]:
    if isinstance(values, list):
        return values
    if torch.is_tensor(values):
        return values.detach().cpu().tolist()
    return list(values)


def _examples(dataset: dict[str, Any]) -> dict[str, Any]:
    examples = dataset.get("examples")
    if not isinstance(examples, dict):
        raise ValueError("dataset is missing examples dictionary")
    required = {"features", "accepted", "benchmark", "operator", "trace_file", "trace_line"}
    missing = required - set(examples)
    if missing:
        raise ValueError(f"dataset examples missing required keys: {sorted(missing)}")
    return examples


def _benchmark_split(
    benchmarks: list[str],
    train: list[str] | None,
    val: list[str] | None,
) -> tuple[list[str], list[str]]:
    names = sorted(set(benchmarks))
    if not names:
        raise ValueError("dataset has no benchmark examples")
    if train or val:
        train_set = sorted(set(train or []) & set(names))
        val_set = sorted(set(val or []) & set(names))
        if not train_set:
            train_set = sorted(set(names) - set(val_set))
        if not val_set:
            val_set = sorted(set(names) - set(train_set))
    else:
        val_set = [names[-1]]
        train_set = names[:-1] or names
    overlap = set(train_set) & set(val_set)
    if overlap:
        raise ValueError(f"train/validation benchmark overlap is not allowed: {sorted(overlap)}")
    if not train_set or not val_set:
        raise ValueError("need at least one train and one validation benchmark")
    return train_set, val_set


def _mask_by_benchmark(examples: dict[str, Any], names: set[str]) -> torch.Tensor:
    return torch.tensor([str(b) in names for b in examples["benchmark"]], dtype=torch.bool)


def _standardize(
    x_train: torch.Tensor,
    x_val: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = x_train.mean(dim=0)
    std = x_train.std(dim=0).clamp_min(1e-6)
    return (x_train - mean) / std, (x_val - mean) / std, mean, std


class LogisticRanker(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.net = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MlpRanker(nn.Module):
    def __init__(self, n_features: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _train_model(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    epochs: int,
    lr: float,
    seed: int,
) -> nn.Module:
    torch.manual_seed(seed)
    pos = float(y.sum().item())
    neg = float(y.numel() - pos)
    pos_weight = torch.tensor([max(1.0, neg / max(pos, 1.0))], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(max(1, epochs)):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y.float())
        loss.backward()
        opt.step()
    return model.eval()


def _pool_keys(examples: dict[str, Any], indices: list[int]) -> dict[tuple[Any, ...], list[int]]:
    trace_line = _as_list(examples["trace_line"])
    source_node = _as_list(
        examples.get("source_node", torch.zeros(len(trace_line), dtype=torch.long))
    )
    keys: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for i in indices:
        key = (
            str(examples["trace_file"][i]),
            int(trace_line[i]),
            str(examples["operator"][i]),
            str(examples["kind"][i]),
            int(source_node[i]),
        )
        keys[key].append(i)
    return keys


def _ranking_metrics(
    examples: dict[str, Any],
    indices: list[int],
    scores: torch.Tensor,
    *,
    top_k: tuple[int, ...] = DEFAULT_TOP_K,
) -> dict[str, float]:
    accepted = _as_list(examples["accepted"])
    pools = _pool_keys(examples, indices)
    out = {f"top{k}_recall": 0.0 for k in top_k}
    out.update({"mrr": 0.0, "mean_accepted_rank": 0.0, "pools": float(len(pools))})
    accepted_total = 0
    reciprocal_sum = 0.0
    rank_sum = 0.0
    for members in pools.values():
        order = sorted(members, key=lambda idx: float(scores[idx]), reverse=True)
        accepted_ranks = [rank for rank, idx in enumerate(order, 1) if bool(accepted[idx])]
        if not accepted_ranks:
            continue
        accepted_total += len(accepted_ranks)
        best_rank = min(accepted_ranks)
        reciprocal_sum += 1.0 / best_rank
        rank_sum += sum(accepted_ranks)
        rank_set = set(accepted_ranks)
        for k in top_k:
            out[f"top{k}_recall"] += sum(1 for rank in rank_set if rank <= k)
    if accepted_total:
        for k in top_k:
            out[f"top{k}_recall"] /= accepted_total
        pools_with_accepts = sum(1 for m in pools.values() if any(accepted[i] for i in m))
        out["mrr"] = reciprocal_sum / max(pools_with_accepts, 1)
        out["mean_accepted_rank"] = rank_sum / accepted_total
    out["accepted"] = float(accepted_total)
    return out


def _binary_metrics(labels: torch.Tensor, scores: torch.Tensor) -> dict[str, float | None]:
    probs = torch.sigmoid(scores)
    pred = probs >= 0.5
    labels_bool = labels.bool()
    acc = float((pred == labels_bool).float().mean().item()) if labels.numel() else 0.0
    pos = scores[labels_bool]
    neg = scores[~labels_bool]
    auc: float | None = None
    if pos.numel() and neg.numel():
        cmp = (pos[:, None] > neg[None, :]).float()
        ties = (pos[:, None] == neg[None, :]).float() * 0.5
        auc = float((cmp + ties).mean().item())
    return {"accuracy": acc, "roc_auc": auc}


def _correlation(a: torch.Tensor, b: torch.Tensor) -> float | None:
    if a.numel() < 2:
        return None
    ax = a.float() - a.float().mean()
    bx = b.float() - b.float().mean()
    denom = torch.linalg.norm(ax) * torch.linalg.norm(bx)
    if float(denom.item()) <= 1e-12:
        return None
    return float((ax @ bx / denom).item())


def _score_trace_order(examples: dict[str, Any]) -> torch.Tensor:
    features = examples["features"].float()
    if features.shape[1] > 3:
        return -features[:, 3]
    return -torch.arange(features.shape[0], dtype=torch.float32)


def _score_existing_score(examples: dict[str, Any]) -> torch.Tensor:
    features = examples["features"].float()
    if features.shape[1] > 6:
        return -features[:, 6]
    return _score_trace_order(examples)


def _summarize(
    examples: dict[str, Any],
    indices: list[int],
    scores: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, Any]:
    idx_tensor = torch.tensor(indices, dtype=torch.long)
    local_labels = labels[idx_tensor]
    local_scores = scores[idx_tensor]
    known = examples["proxy_delta_known"][idx_tensor].bool()
    proxy_delta = examples["proxy_delta"][idx_tensor].float()
    result: dict[str, Any] = {}
    result.update(_ranking_metrics(examples, indices, scores))
    result.update(_binary_metrics(local_labels, local_scores))
    result["proxy_delta_pearson"] = _correlation(local_scores[known], -proxy_delta[known])
    result["examples"] = float(len(indices))
    result["accepted_rate"] = float(local_labels.float().mean().item()) if len(indices) else 0.0
    return result


def _group_metrics(
    examples: dict[str, Any],
    indices: list[int],
    scores: torch.Tensor,
    labels: torch.Tensor,
    key_name: str,
) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(list)
    values = examples[key_name]
    for i in indices:
        groups[str(values[i])].append(i)
    return {
        name: _summarize(examples, group_indices, scores, labels)
        for name, group_indices in sorted(groups.items())
    }


def _code_fingerprint() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def train_and_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = _load_dataset(args.dataset)
    examples = _examples(dataset)
    features = examples["features"].float()
    labels = examples["accepted"].bool()
    benchmarks = [str(b) for b in examples["benchmark"]]
    train_bench, val_bench = _benchmark_split(benchmarks, args.train_benchmark, args.val_benchmark)
    train_mask = _mask_by_benchmark(examples, set(train_bench))
    val_mask = _mask_by_benchmark(examples, set(val_bench))
    train_idx = train_mask.nonzero(as_tuple=False).flatten().tolist()
    val_idx = val_mask.nonzero(as_tuple=False).flatten().tolist()
    if not train_idx or not val_idx:
        raise ValueError("split produced no training or validation examples")

    x_train, x_val, mean, std = _standardize(features[train_mask], features[val_mask])
    y_train = labels[train_mask].float()
    n_features = int(features.shape[1])

    models: dict[str, nn.Module] = {
        "logistic": LogisticRanker(n_features),
        "mlp": MlpRanker(n_features, args.hidden_size),
    }
    trained: dict[str, nn.Module] = {}
    val_scores_by_model: dict[str, torch.Tensor] = {
        "trace_order": _score_trace_order(examples),
        "existing_score": _score_existing_score(examples),
    }
    train_started = time.time()
    for name, model in models.items():
        trained[name] = _train_model(
            model,
            x_train,
            y_train,
            epochs=args.epochs,
            lr=args.lr,
            seed=args.seed,
        )
        all_scores = torch.zeros(features.shape[0], dtype=torch.float32)
        with torch.no_grad():
            all_scores[val_mask] = trained[name](x_val)
            all_scores[train_mask] = trained[name](x_train)
        val_scores_by_model[name] = all_scores

    metrics = {}
    for name, scores in val_scores_by_model.items():
        metrics[name] = {
            "validation": _summarize(examples, val_idx, scores, labels),
            "by_operator": _group_metrics(examples, val_idx, scores, labels, "operator"),
            "by_benchmark": _group_metrics(examples, val_idx, scores, labels, "benchmark"),
        }

    result = {
        "stage": "G3",
        "dataset": str(args.dataset),
        "metadata": dataset.get("metadata", {}),
        "splits": {
            "train": train_bench,
            "validation": val_bench,
            "holdout": args.holdout_benchmark or [],
        },
        "config": {
            "models": ["logistic", "mlp"],
            "hidden_size": args.hidden_size,
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "seed": args.seed,
            "standardized": True,
        },
        "runtime_s": time.time() - train_started,
        "metrics": metrics,
        "promotion_decision": "default_off",
        "code_fingerprint": _code_fingerprint(),
    }

    if args.out_dir:
        _write_artifacts(args.out_dir, dataset, trained, mean, std, result)
    return result


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_artifacts(
    out_dir: Path,
    dataset: dict[str, Any],
    models: dict[str, nn.Module],
    mean: torch.Tensor,
    std: torch.Tensor,
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
            "models": {name: model.state_dict() for name, model in models.items()},
            "feature_mean": mean,
            "feature_std": std,
            "config": result["config"],
            "feature_schema": dataset.get("feature_schema", {}),
        },
        out_dir / "model.pt",
    )
    readme = [
        "# Stage G3 Candidate Baseline",
        "",
        "Default-off offline baseline artifact for hierarchy GNN candidate ranking.",
        "",
        f"- Dataset: `{result['dataset']}`",
        f"- Train benchmarks: {', '.join(result['splits']['train'])}",
        f"- Validation benchmarks: {', '.join(result['splits']['validation'])}",
        "- Models: logistic regression and small MLP over candidate scalar features.",
        "- Placement behavior: unchanged; this artifact is not integrated for inference.",
        "",
        "See `metrics.json` for ranking and class-balance results.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Stage-G2 dataset .pt")
    parser.add_argument("--out-dir", type=Path, help="Optional artifact output directory")
    parser.add_argument("--metrics-out", type=Path, help="Optional metrics JSON output path")
    parser.add_argument("--train-benchmark", action="append", help="Training benchmark name")
    parser.add_argument("--val-benchmark", action="append", help="Validation benchmark name")
    parser.add_argument(
        "--holdout-benchmark",
        action="append",
        help="Recorded holdout benchmark name",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    result = train_and_evaluate(args)
    if args.metrics_out:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        _write_json(args.metrics_out, result)

    summary = result["metrics"]["mlp"]["validation"]
    trace_summary = result["metrics"]["trace_order"]["validation"]
    print(
        "G3 baseline validation: "
        f"mlp_top4={summary['top4_recall']:.4f}, "
        f"trace_top4={trace_summary['top4_recall']:.4f}, "
        f"mlp_mrr={summary['mrr']:.4f}, "
        f"trace_mrr={trace_summary['mrr']:.4f}"
    )


if __name__ == "__main__":
    main()
