#!/usr/bin/env python3
"""Diagnose coldspot kick candidate selector quality from a GNN dataset.

The report is offline-only. It groups `coldspot_tightening` candidates by
candidate pool, then compares trace order, cheap field-delta order, stable
random order, exact-proxy oracle order, and optionally one GNN model.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gnn"))

from train_gnn_baseline import (
    _load_dataset,
    _score_existing_score,
    _score_trace_order,
)  # noqa: E402
from train_gnn_ranker import MacroNetRanker  # noqa: E402


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return list(value)


def _adjust_features(features: torch.Tensor, feature_count: int) -> torch.Tensor:
    if features.shape[1] == feature_count:
        return features
    if features.shape[1] > feature_count:
        return features[:, :feature_count].contiguous()
    pad = torch.zeros((features.shape[0], feature_count - features.shape[1]), dtype=features.dtype)
    return torch.cat([features, pad], dim=1)


def _load_model_scores(path: Path, dataset: dict[str, Any]) -> torch.Tensor:
    artifact = _load_dataset(path)
    config = artifact["config"]
    feature_count = len(artifact["feature_schema"]["candidate_features"])
    graph0 = dataset["graphs"][0]
    model = MacroNetRanker(
        node_features=graph0["node_features"].shape[1],
        net_features=graph0["net_node_features"].shape[1],
        edge_features=graph0["macro_net_edge_features"].shape[1],
        candidate_features=feature_count,
        hidden=int(config.get("hidden_size", 32)),
        layers=int(config.get("graph_layers", 2)),
    )
    model.load_state_dict(artifact["model"])
    model.eval()
    scored_dataset = dict(dataset)
    scored_examples = dict(dataset["examples"])
    scored_examples["features"] = _adjust_features(
        dataset["examples"]["features"].float(), feature_count
    )
    scored_dataset["examples"] = scored_examples
    idx = torch.arange(scored_examples["features"].shape[0], dtype=torch.long)
    with torch.no_grad():
        return model(scored_dataset, idx).float()


def _coldspot_indices(examples: dict[str, Any], benchmarks: set[str] | None) -> list[int]:
    out = []
    for i, op in enumerate(examples["operator"]):
        if str(op) != "coldspot_tightening":
            continue
        if benchmarks and str(examples["benchmark"][i]) not in benchmarks:
            continue
        out.append(i)
    return out


def _pool_keys(examples: dict[str, Any], indices: list[int]) -> dict[tuple[Any, ...], list[int]]:
    trace_line = _as_list(examples["trace_line"])
    source_node = _as_list(
        examples.get("source_node", torch.zeros(len(trace_line), dtype=torch.long))
    )
    candidate_pool_id = _as_list(
        examples.get("candidate_pool_id", torch.full((len(trace_line),), -1, dtype=torch.long))
    )
    pools: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for idx in indices:
        pool_id = int(candidate_pool_id[idx])
        if pool_id >= 0:
            key = (str(examples["trace_file"][idx]), str(examples["benchmark"][idx]), pool_id)
        else:
            key = (
                str(examples["trace_file"][idx]),
                int(trace_line[idx]),
                str(examples["benchmark"][idx]),
                int(source_node[idx]),
            )
        pools[key].append(idx)
    return pools


def _stable_random_scores(n: int, seed: int) -> torch.Tensor:
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    return torch.rand(n, generator=gen)


def _oracle_scores(examples: dict[str, Any]) -> torch.Tensor:
    known = examples["proxy_delta_known"].bool()
    gain = (-examples["proxy_delta"].float()).clamp_min(0.0)
    return torch.where(known, gain, torch.full_like(gain, -1.0))


def _summarize(
    examples: dict[str, Any],
    indices: list[int],
    scores: dict[str, torch.Tensor],
    *,
    top_k: tuple[int, ...],
) -> dict[str, Any]:
    accepted = examples["accepted"].bool()
    proxy_delta = examples["proxy_delta"].float()
    known = examples["proxy_delta_known"].bool()
    pools = _pool_keys(examples, indices)
    out: dict[str, Any] = {
        "examples": len(indices),
        "pools": len(pools),
        "accepted": float(accepted[torch.tensor(indices)].sum().item()) if indices else 0.0,
        "models": {},
    }
    total_gain = 0.0
    pools_with_gain = 0
    for members in pools.values():
        gains = [max(0.0, -float(proxy_delta[i])) if bool(known[i]) else 0.0 for i in members]
        total_gain += sum(gains)
        if max(gains or [0.0]) > 0.0:
            pools_with_gain += 1
    out["positive_gain_total"] = total_gain
    out["pools_with_positive_gain"] = pools_with_gain

    for name, score in scores.items():
        metrics = {f"top{k}_best_gain_recall": 0.0 for k in top_k}
        metrics.update({f"top{k}_gain_recall": 0.0 for k in top_k})
        metrics["mean_best_gain_rank"] = 0.0
        rank_count = 0
        for members in pools.values():
            order = sorted(members, key=lambda idx: float(score[idx]), reverse=True)
            gains = {
                idx: max(0.0, -float(proxy_delta[idx])) if bool(known[idx]) else 0.0
                for idx in members
            }
            best_gain = max(gains.values() or [0.0])
            if best_gain > 0.0:
                best = min(
                    (idx for idx, gain in gains.items() if gain == best_gain),
                    key=lambda idx: order.index(idx),
                )
                best_rank = order.index(best) + 1
                metrics["mean_best_gain_rank"] += float(best_rank)
                rank_count += 1
                for k in top_k:
                    if best_rank <= k:
                        metrics[f"top{k}_best_gain_recall"] += 1.0
            for k in top_k:
                top = order[:k]
                metrics[f"top{k}_gain_recall"] += sum(gains[idx] for idx in top)
        for k in top_k:
            metrics[f"top{k}_best_gain_recall"] = (
                metrics[f"top{k}_best_gain_recall"] / pools_with_gain if pools_with_gain else 0.0
            )
            metrics[f"top{k}_gain_recall"] = (
                metrics[f"top{k}_gain_recall"] / total_gain if total_gain else 0.0
            )
        metrics["mean_best_gain_rank"] = (
            metrics["mean_best_gain_rank"] / rank_count if rank_count else None
        )
        out["models"][name] = metrics
    return out


def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    dataset = _load_dataset(args.dataset)
    examples = dataset["examples"]
    benchmarks = set(args.benchmark or [])
    indices = _coldspot_indices(examples, benchmarks or None)
    scores = {
        "trace_order": _score_trace_order(examples),
        "field_delta": _score_existing_score(examples),
        "stable_random": _stable_random_scores(len(examples["benchmark"]), args.seed),
        "exact_proxy_oracle": _oracle_scores(examples),
    }
    if args.model:
        scores["gnn_model"] = _load_model_scores(args.model, dataset)
    top_k = tuple(sorted({int(k) for k in args.top_k}))
    report = {
        "dataset": str(args.dataset),
        "model": str(args.model) if args.model else None,
        "benchmarks": sorted(benchmarks) if benchmarks else "all",
        "top_k": list(top_k),
        "overall": _summarize(examples, indices, scores, top_k=top_k),
        "by_benchmark": {},
    }
    for benchmark in sorted({str(examples["benchmark"][i]) for i in indices}):
        bench_indices = [i for i in indices if str(examples["benchmark"][i]) == benchmark]
        report["by_benchmark"][benchmark] = _summarize(examples, bench_indices, scores, top_k=top_k)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, help="Optional GNN ranker artifact")
    parser.add_argument("--benchmark", action="append")
    parser.add_argument("--top-k", type=int, action="append", default=[1, 4])
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    report = diagnose(args)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    models = report["overall"]["models"]
    parts = [
        f"{name}:top1={vals.get('top1_best_gain_recall', 0.0):.4f},"
        f"top4={vals.get('top4_best_gain_recall', 0.0):.4f}"
        for name, vals in models.items()
    ]
    print(
        "Coldspot selector diagnostic: "
        f"pools={report['overall']['pools']}, "
        f"gain_pools={report['overall']['pools_with_positive_gain']}; " + " | ".join(parts)
    )


if __name__ == "__main__":
    main()
