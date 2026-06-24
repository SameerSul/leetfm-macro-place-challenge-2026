#!/usr/bin/env python3
"""Diagnose why offline GNN ranking may not improve closed-loop placement.

The report compares heuristic, G3, and G4 orderings inside candidate pools. It
adds value-oriented metrics based on exact proxy delta where available, so we can
see whether a ranker captures high-gain moves rather than merely accepted moves.
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
sys.path.insert(0, str(ROOT / "scripts"))

from train_gnn_baseline import (  # noqa: E402
    MlpRanker,
    _load_dataset,
    _score_existing_score,
    _score_trace_order,
)
from train_gnn_ranker import MacroNetRanker  # noqa: E402


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return list(value)


def _pool_keys(examples: dict[str, Any], indices: list[int]) -> dict[tuple[Any, ...], list[int]]:
    trace_line = _as_list(examples["trace_line"])
    source_node = _as_list(
        examples.get("source_node", torch.zeros(len(trace_line), dtype=torch.long))
    )
    pools: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for idx in indices:
        key = (
            str(examples["trace_file"][idx]),
            int(trace_line[idx]),
            str(examples["operator"][idx]),
            str(examples["kind"][idx]),
            int(source_node[idx]),
        )
        pools[key].append(idx)
    return pools


def _load_g3_scores(path: Path, dataset: dict[str, Any]) -> torch.Tensor:
    artifact = _load_dataset(path)
    state = artifact.get("models", {}).get("mlp")
    mean = artifact.get("feature_mean")
    std = artifact.get("feature_std")
    if state is None or mean is None or std is None:
        raise ValueError(f"{path} is not a compatible G3 baseline artifact")
    features = dataset["examples"]["features"].float()
    feature_count = int(artifact.get("feature_mean", features).numel())
    features = _adjust_features(features, feature_count)
    model = MlpRanker(feature_count, int(artifact.get("config", {}).get("hidden_size", 32)))
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        return model((features - mean) / std.clamp_min(1e-6)).float()


def _load_g4_scores(path: Path, dataset: dict[str, Any]) -> torch.Tensor:
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
    idx = torch.arange(dataset["examples"]["features"].shape[0], dtype=torch.long)
    with torch.no_grad():
        return model(scored_dataset, idx).float()


def _adjust_features(features: torch.Tensor, feature_count: int) -> torch.Tensor:
    if features.shape[1] == feature_count:
        return features
    if features.shape[1] > feature_count:
        return features[:, :feature_count].contiguous()
    pad = torch.zeros((features.shape[0], feature_count - features.shape[1]), dtype=features.dtype)
    return torch.cat([features, pad], dim=1)


def _scope_indices(examples: dict[str, Any], benchmarks: set[str] | None) -> list[int]:
    if not benchmarks:
        return list(range(len(examples["benchmark"])))
    return [i for i, name in enumerate(examples["benchmark"]) if str(name) in benchmarks]


def _summarize_pools(
    examples: dict[str, Any],
    indices: list[int],
    scores: dict[str, torch.Tensor],
    *,
    top_k: int,
) -> dict[str, Any]:
    accepted = examples["accepted"].bool()
    proxy_delta = examples["proxy_delta"].float()
    known = examples["proxy_delta_known"].bool()
    pools = _pool_keys(examples, indices)
    names = list(scores)
    totals = {
        name: {
            "topk_accepts": 0.0,
            "topk_gain": 0.0,
            "best_gain_rank_sum": 0.0,
            "best_gain_rank_count": 0.0,
            "topk_overlap_with_trace": 0.0,
            "topk_overlap_count": 0.0,
        }
        for name in names
    }
    accepted_total = 0.0
    positive_gain_total = 0.0
    pools_with_gain = 0.0

    for members in pools.values():
        member_tensor = torch.tensor(members, dtype=torch.long)
        member_accept = accepted[member_tensor]
        member_known = known[member_tensor]
        gains = torch.where(member_known, (-proxy_delta[member_tensor]).clamp_min(0.0), 0.0)
        accepted_total += float(member_accept.sum().item())
        positive_gain_total += float(gains.sum().item())
        if float(gains.max().item()) > 0.0:
            pools_with_gain += 1.0
        trace_order = sorted(
            members, key=lambda idx: float(scores["trace_order"][idx]), reverse=True
        )
        trace_top = set(trace_order[:top_k])
        for name, score in scores.items():
            order = sorted(members, key=lambda idx: float(score[idx]), reverse=True)
            top = order[:top_k]
            top_set = set(top)
            totals[name]["topk_accepts"] += float(accepted[torch.tensor(top)].sum().item())
            totals[name]["topk_gain"] += sum(
                max(0.0, -float(proxy_delta[i])) for i in top if bool(known[i])
            )
            if float(gains.max().item()) > 0.0:
                best_member = members[int(torch.argmax(gains).item())]
                totals[name]["best_gain_rank_sum"] += float(order.index(best_member) + 1)
                totals[name]["best_gain_rank_count"] += 1.0
            if name != "trace_order":
                totals[name]["topk_overlap_with_trace"] += len(top_set & trace_top) / max(top_k, 1)
                totals[name]["topk_overlap_count"] += 1.0

    out: dict[str, Any] = {
        "examples": len(indices),
        "pools": len(pools),
        "accepted": accepted_total,
        "positive_gain_total": positive_gain_total,
        "pools_with_positive_gain": pools_with_gain,
        "models": {},
    }
    for name, vals in totals.items():
        accept_recall = vals["topk_accepts"] / accepted_total if accepted_total else 0.0
        gain_recall = vals["topk_gain"] / positive_gain_total if positive_gain_total else 0.0
        rank_count = vals["best_gain_rank_count"]
        overlap_count = vals["topk_overlap_count"]
        out["models"][name] = {
            "topk_accept_recall": accept_recall,
            "topk_gain_recall": gain_recall,
            "topk_gain": vals["topk_gain"],
            "mean_best_gain_rank": vals["best_gain_rank_sum"] / rank_count if rank_count else None,
            "topk_overlap_with_trace": (
                vals["topk_overlap_with_trace"] / overlap_count if overlap_count else None
            ),
        }
    return out


def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    dataset = _load_dataset(args.dataset)
    examples = dataset["examples"]
    scores = {
        "trace_order": _score_trace_order(examples),
        "existing_score": _score_existing_score(examples),
        "g3_mlp": _load_g3_scores(args.g3_model, dataset),
        "g4_macro_net": _load_g4_scores(args.g4_model, dataset),
    }
    benchmarks = set(args.benchmark or [])
    indices = _scope_indices(examples, benchmarks or None)
    by_operator: dict[str, Any] = {}
    for operator in sorted({str(examples["operator"][i]) for i in indices}):
        op_indices = [i for i in indices if str(examples["operator"][i]) == operator]
        by_operator[operator] = _summarize_pools(examples, op_indices, scores, top_k=args.top_k)
    by_benchmark: dict[str, Any] = {}
    for benchmark in sorted({str(examples["benchmark"][i]) for i in indices}):
        bench_indices = [i for i in indices if str(examples["benchmark"][i]) == benchmark]
        by_benchmark[benchmark] = _summarize_pools(
            examples, bench_indices, scores, top_k=args.top_k
        )
    return {
        "dataset": str(args.dataset),
        "g3_model": str(args.g3_model),
        "g4_model": str(args.g4_model),
        "top_k": args.top_k,
        "benchmarks": sorted(benchmarks) if benchmarks else "all",
        "overall": _summarize_pools(examples, indices, scores, top_k=args.top_k),
        "by_operator": by_operator,
        "by_benchmark": by_benchmark,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--g3-model", type=Path, required=True)
    parser.add_argument("--g4-model", type=Path, required=True)
    parser.add_argument("--benchmark", action="append")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    report = diagnose(args)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    overall = report["overall"]["models"]
    print(
        "GNN diagnostic: "
        f"trace_gain@{args.top_k}={overall['trace_order']['topk_gain_recall']:.4f}, "
        f"g3_gain@{args.top_k}={overall['g3_mlp']['topk_gain_recall']:.4f}, "
        f"g4_gain@{args.top_k}={overall['g4_macro_net']['topk_gain_recall']:.4f}, "
        f"g4_overlap={overall['g4_macro_net']['topk_overlap_with_trace']:.4f}"
    )


if __name__ == "__main__":
    main()
