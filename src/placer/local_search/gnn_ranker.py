"""Default-off GNN candidate reordering hooks.

The ranker is advisory only: it can reorder an existing candidate list, but all
deterministic legality and exact-proxy gates remain downstream.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

_FALSE = {"0", "false", "False", "no", "NO", "off", ""}
_CACHE: dict[tuple[str, str], tuple[Any, dict[str, Any], dict[str, Any]]] = {}


def gnn_rank_enabled(operator: str) -> bool:
    if os.environ.get("HIER_GNN_RANK", "0").strip() in _FALSE:
        return False
    raw_ops = os.environ.get("HIER_GNN_OPERATORS", "relocation").strip()
    ops = {x.strip() for x in raw_ops.split(",") if x.strip()}
    return not ops or operator in ops


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_ranker(benchmark_name: str) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    model_path = Path(
        os.environ.get(
            "HIER_GNN_MODEL",
            "ml_data/beyondppa_gnn/models/20260619_g4_macro_net_ranker_v1/model.pt",
        )
    )
    key = (str(model_path), benchmark_name)
    if key in _CACHE:
        return _CACHE[key]

    import sys

    root = _repo_root()
    scripts = root / "scripts" / "gnn"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from build_gnn_dataset import _build_graph, _candidate_feature  # type: ignore
    from train_gnn_ranker import MacroNetRanker  # type: ignore

    artifact = torch.load(model_path, map_location="cpu", weights_only=False)
    bench_root = Path(
        os.environ.get(
            "HIER_GNN_BENCHMARK_ROOT",
            str(root / "external" / "MacroPlacement" / "Testcases" / "ICCAD04"),
        )
    )
    graph = _build_graph(benchmark_name, [bench_root])
    config = artifact["config"]
    model = MacroNetRanker(
        node_features=graph["node_features"].shape[1],
        net_features=graph["net_node_features"].shape[1],
        edge_features=graph["macro_net_edge_features"].shape[1],
        candidate_features=len(artifact["feature_schema"]["candidate_features"]),
        hidden=int(config.get("hidden_size", 32)),
        layers=int(config.get("graph_layers", 2)),
    )
    model.load_state_dict(artifact["model"])
    model.eval()
    loaded = (model, graph, {"candidate_feature": _candidate_feature})
    _CACHE[key] = loaded
    return loaded


def reorder_hard_relocation_proposals(
    proposals: list[dict[str, Any]],
    *,
    benchmark_name: str,
    field: str,
) -> list[dict[str, Any]]:
    if not proposals or not gnn_rank_enabled("relocation"):
        return proposals
    try:
        model, graph, helpers = _load_ranker(benchmark_name)
        candidate_feature = helpers["candidate_feature"]
        rows = []
        for p in proposals:
            x, y = p["xy"]
            rows.append(
                {
                    "event": "hier_relocation_candidate",
                    "operator": "relocation",
                    "kind": "hard_propose_all",
                    "field": field,
                    "macro": int(p.get("i", -1)),
                    "candidate_rank": int(p.get("candidate_rank", -1)),
                    "target_index": int(p.get("target_index", -1)),
                    "score": float(p.get("score", 0.0)),
                    "local_field": float(p.get("local_field", 0.0)),
                    "target_field": float(p.get("target_field", 0.0)),
                    "structural_delta": float(p.get("structural_delta", 0.0)),
                    "x": float(x),
                    "y": float(y),
                }
            )
        features = torch.tensor(
            [candidate_feature(row, graph) for row in rows], dtype=torch.float32
        )
        source_node = torch.tensor([int(p.get("i", 0)) for p in proposals], dtype=torch.long)
        target_node = torch.full((len(proposals),), -1, dtype=torch.long)
        dataset = {
            "graphs": [graph],
            "examples": {
                "features": features,
                "graph_id": torch.zeros(len(proposals), dtype=torch.long),
                "source_node": source_node,
                "target_node": target_node,
            },
        }
        with torch.no_grad():
            scores = model(dataset, torch.arange(len(proposals), dtype=torch.long))
        for p, score in zip(proposals, scores.tolist()):
            p["gnn_score"] = float(score)
        original_order = {id(p): i for i, p in enumerate(proposals)}
        top_k = int(os.environ.get("HIER_GNN_TOP_K", "0") or "0")
        ranked = sorted(
            proposals,
            key=lambda p: (
                -float(p.get("gnn_score", 0.0)),
                float(p.get("score", 0.0)),
                int(p.get("candidate_rank", 0)),
                original_order[id(p)],
            ),
        )
        preserve_top_n = max(0, int(os.environ.get("HIER_GNN_PRESERVE_TOP_N", "0") or "0"))
        if preserve_top_n > 0:
            prefix = proposals[:preserve_top_n]
            prefix_ids = {id(p) for p in prefix}
            ranked_remainder = [p for p in ranked if id(p) not in prefix_ids]
            top_k = int(os.environ.get("HIER_GNN_TOP_K", "0") or "0")
            if top_k > 0:
                selected = ranked_remainder[:top_k]
                selected_ids = {id(p) for p in selected} | prefix_ids
                return prefix + selected + [p for p in proposals if id(p) not in selected_ids]
            return prefix + ranked_remainder
        if top_k > 0:
            selected = ranked[:top_k]
            selected_ids = {id(p) for p in selected}
            return selected + [p for p in proposals if id(p) not in selected_ids]
        return ranked
    except Exception as exc:
        for p in proposals:
            p["gnn_rank_error"] = str(exc)
        return proposals
