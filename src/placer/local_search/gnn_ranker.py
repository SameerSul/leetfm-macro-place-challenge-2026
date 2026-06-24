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


def gnn_coldspot_select_enabled() -> bool:
    return os.environ.get("HIER_GNN_COLDSPOT_SELECT", "0").strip() not in _FALSE


def gnn_coldspot_oracle_enabled() -> bool:
    return os.environ.get("HIER_GNN_COLDSPOT_ORACLE", "0").strip() not in _FALSE


def gnn_coldspot_kicks(default: int = 8) -> int:
    return max(1, int(os.environ.get("HIER_GNN_COLDSPOT_KICKS", str(default)) or str(default)))


def gnn_coldspot_skip_micro(default: bool = True) -> bool:
    raw = os.environ.get("HIER_GNN_COLDSPOT_SKIP_MICRO", "1" if default else "0").strip()
    return raw not in _FALSE


def _load_ranker(
    benchmark_name: str,
    *,
    model_env: str = "HIER_GNN_MODEL",
    default_model: str = "ml_data/beyondppa_gnn/models/20260619_g4_macro_net_ranker_v1/model.pt",
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    raw_model = os.environ.get(model_env, default_model).strip()
    if not raw_model:
        raise ValueError(f"{model_env} is required")
    model_path = Path(raw_model)
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
    loaded[2]["candidate_feature_count"] = len(artifact["feature_schema"]["candidate_features"])
    _CACHE[key] = loaded
    return loaded


def _artifact_features(rows: list[dict[str, Any]], graph: dict[str, Any], helpers: dict[str, Any]):
    candidate_feature = helpers["candidate_feature"]
    feature_count = int(helpers.get("candidate_feature_count", 0))
    vectors = [candidate_feature(row, graph) for row in rows]
    if feature_count <= 0:
        return torch.tensor(vectors, dtype=torch.float32)
    adjusted = []
    for vec in vectors:
        if len(vec) < feature_count:
            vec = vec + [0.0] * (feature_count - len(vec))
        elif len(vec) > feature_count:
            vec = vec[:feature_count]
        adjusted.append(vec)
    return torch.tensor(adjusted, dtype=torch.float32)


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
        features = _artifact_features(rows, graph, helpers)
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


def reorder_region_swap_candidates(
    candidates: list[dict[str, Any]],
    *,
    benchmark_name: str,
    kind: str,
    field: str,
    source: int,
) -> list[dict[str, Any]]:
    """Rank region-bounded swap candidates before exact scoring."""
    if not candidates or not gnn_rank_enabled("region_swaps"):
        return candidates
    try:
        model, graph, helpers = _load_ranker(benchmark_name)
        rows = []
        for cand in candidates:
            row = dict(cand)
            row.update(
                {
                    "event": "hier_swap_candidates",
                    "operator": "region_swaps",
                    "kind": kind,
                    "field": field,
                    "source": int(source),
                }
            )
            rows.append(row)
        features = _artifact_features(rows, graph, helpers)
        n_hard = int(graph["num_hard_macros"])
        source_nodes = []
        target_nodes = []
        for row in rows:
            src = int(row.get("source", -1))
            tgt = int(row.get("target", -1))
            if kind == "soft_soft":
                src += n_hard
                tgt += n_hard
            elif kind == "hard_soft":
                tgt += n_hard
            source_nodes.append(max(src, 0))
            target_nodes.append(tgt if tgt >= 0 else -1)
        dataset = {
            "graphs": [graph],
            "examples": {
                "features": features,
                "graph_id": torch.zeros(len(candidates), dtype=torch.long),
                "source_node": torch.tensor(source_nodes, dtype=torch.long),
                "target_node": torch.tensor(target_nodes, dtype=torch.long),
            },
        }
        with torch.no_grad():
            scores = model(dataset, torch.arange(len(candidates), dtype=torch.long))
        for cand, score in zip(candidates, scores.tolist()):
            cand["gnn_score"] = float(score)
        original_order = {id(c): i for i, c in enumerate(candidates)}
        ranked = sorted(
            candidates,
            key=lambda c: (
                -float(c.get("gnn_score", 0.0)),
                int(c.get("candidate_rank", 0)),
                int(c.get("target", -1)),
                original_order[id(c)],
            ),
        )
        preserve_top_n = max(0, int(os.environ.get("HIER_GNN_SWAP_PRESERVE_TOP_N", "0") or "0"))
        if preserve_top_n > 0:
            prefix = candidates[:preserve_top_n]
            prefix_ids = {id(c) for c in prefix}
            ranked_remainder = [c for c in ranked if id(c) not in prefix_ids]
            top_k = max(0, int(os.environ.get("HIER_GNN_SWAP_TOP_K", "0") or "0"))
            if top_k > 0:
                selected = ranked_remainder[:top_k]
                selected_ids = {id(c) for c in selected} | prefix_ids
                return prefix + selected + [c for c in candidates if id(c) not in selected_ids]
            return prefix + ranked_remainder
        top_k = max(0, int(os.environ.get("HIER_GNN_SWAP_TOP_K", "0") or "0"))
        if top_k > 0:
            selected = ranked[:top_k]
            selected_ids = {id(c) for c in selected}
            return selected + [c for c in candidates if id(c) not in selected_ids]
        return ranked
    except Exception as exc:
        for cand in candidates:
            cand["gnn_rank_error"] = str(exc)
        return candidates


def rank_coldspot_kick_candidates(
    candidates: list[dict[str, Any]],
    *,
    benchmark_name: str,
) -> list[dict[str, Any]]:
    """Rank generated coldspot kick outcomes with the default-off GNN selector."""
    if not candidates or not gnn_coldspot_select_enabled():
        return candidates
    try:
        model, graph, helpers = _load_ranker(
            benchmark_name,
            model_env="HIER_GNN_COLDSPOT_MODEL",
            default_model="",
        )
        rows = []
        for cand in candidates:
            row = dict(cand.get("trace", {}))
            row.update(
                {
                    "event": "hier_coldspot_candidate",
                    "operator": "coldspot_tightening",
                    "kind": "coldspot_kick",
                    "field": "congestion",
                    "candidate_rank": int(cand.get("candidate_rank", row.get("candidate_rank", 0))),
                    "old_proxy": cand.get("old_proxy"),
                    "candidate_proxy": cand.get("candidate_proxy"),
                    "proxy_delta": cand.get("proxy_delta"),
                    "hierarchy_quality_before": cand.get("hierarchy_quality_before"),
                    "hierarchy_quality_after": cand.get("hierarchy_quality_after"),
                    "hierarchy_quality_delta": cand.get("hierarchy_quality_delta"),
                    "legal": True,
                }
            )
            rows.append(row)
        features = _artifact_features(rows, graph, helpers)
        source_node = []
        clusters = graph["cluster_node"]
        for row in rows:
            cid = int(row.get("cluster", -1))
            if 0 <= cid < int(clusters.numel()):
                source_node.append(int(clusters[cid]))
            else:
                source_node.append(0)
        dataset = {
            "graphs": [graph],
            "examples": {
                "features": features,
                "graph_id": torch.zeros(len(candidates), dtype=torch.long),
                "source_node": torch.tensor(source_node, dtype=torch.long),
                "target_node": torch.full((len(candidates),), -1, dtype=torch.long),
            },
        }
        with torch.no_grad():
            scores = model(dataset, torch.arange(len(candidates), dtype=torch.long))
        for cand, score in zip(candidates, scores.tolist()):
            cand["gnn_score"] = float(score)
        original_order = {id(c): i for i, c in enumerate(candidates)}
        ranked = sorted(
            candidates,
            key=lambda c: (
                -float(c.get("gnn_score", 0.0)),
                bool(c.get("is_noop", False)),
                float(c.get("trace", {}).get("score", 0.0)),
                int(c.get("candidate_rank", 0)),
                original_order[id(c)],
            ),
        )
        top_k = max(1, int(os.environ.get("HIER_GNN_COLDSPOT_TOP_K", "1") or "1"))
        selected = ranked[:top_k]
        selected_ids = {id(c) for c in selected}
        return selected + [c for c in ranked if id(c) not in selected_ids]
    except Exception as exc:
        for cand in candidates:
            cand["gnn_rank_error"] = str(exc)
        active = [c for c in candidates if not c.get("is_noop", False)]
        noop = [c for c in candidates if c.get("is_noop", False)]
        return active + noop
