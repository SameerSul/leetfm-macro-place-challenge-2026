#!/usr/bin/env python3
"""Build schema-v1 hierarchy GNN datasets from JSONL traces.

The output is a CPU `torch.save` payload with benchmark graphs and candidate
examples. It is intentionally framework-neutral: PyTorch Geometric can wrap the
stored tensors later, but the builder itself only requires torch/numpy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from macro_place.loader import load_benchmark_from_dir  # noqa: E402
from placer.local_search.clusters import (  # noqa: E402
    cluster_max_fanout,
    cluster_min_edge,
    derive_hard_clusters,
    derive_soft_cluster_roles,
)
from placer.scoring.wirelength import _build_wl_cache  # noqa: E402

DATASET_SCHEMA_VERSION = 2

NODE_FEATURES = [
    "is_hard_macro",
    "is_soft_macro",
    "is_cluster",
    "is_fixed",
    "is_movable",
    "width_norm",
    "height_norm",
    "area_norm",
    "x_norm",
    "y_norm",
    "dist_left_norm",
    "dist_right_norm",
    "dist_bottom_norm",
    "dist_top_norm",
    "grid_offset_x",
    "grid_offset_y",
    "cluster_id_norm",
    "cluster_size_norm",
]

EDGE_FEATURES = [
    "is_net_edge",
    "is_membership_edge",
    "is_spatial_edge",
    "net_weight_norm",
    "fanout_norm",
    "distance_norm",
]

NET_NODE_FEATURES = [
    "degree_norm",
    "macro_degree_norm",
    "net_weight_norm",
    "hpwl_x_norm",
    "hpwl_y_norm",
    "hpwl_norm",
]

MACRO_NET_EDGE_FEATURES = [
    "pin_offset_x_norm",
    "pin_offset_y_norm",
    "abs_pin_offset_x_norm",
    "abs_pin_offset_y_norm",
    "net_weight_norm",
    "fanout_norm",
    "is_driver_pin",
]

CANDIDATE_FEATURES = [
    "operator_id",
    "kind_id",
    "field_id",
    "candidate_rank_norm",
    "source_field",
    "target_field",
    "score",
    "structural_delta",
    "outside_region",
    "legal",
    "old_proxy",
    "candidate_proxy",
    "proxy_delta_known",
    "target_x_norm",
    "target_y_norm",
    "hierarchy_quality_before",
    "hierarchy_quality_after",
    "hierarchy_quality_delta",
    "expansion_factor",
    "axis_scale_x",
    "axis_scale_y",
    "field_gap",
    "cluster_heat",
    "window_cells_norm",
    "movable_count_norm",
    "member_count_norm",
    "soft_count_norm",
]

OPERATOR_IDS = {
    "relocation": 1,
    "cluster_decompression": 2,
    "region_swaps": 3,
    "coldspot_tightening": 4,
}

KIND_IDS = {
    "hard_propose_all": 1,
    "hard_sequential": 2,
    "soft_sequential": 3,
    "hard_hard": 4,
    "hard_soft": 5,
    "soft_soft": 6,
}

FIELD_IDS = {
    "congestion": 1,
    "density": 2,
}

REJECTION_IDS = {
    None: 0,
    "illegal_overlap": 1,
    "out_of_bounds": 2,
    "out_of_hierarchy_region": 3,
    "hierarchy_quality_failed": 4,
    "exact_proxy_failed": 5,
    "proxy_budget_failed": 6,
    "field_gap_below_threshold": 7,
    "no_eligible_cluster": 8,
    "not_scored": 9,
}


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    try:
        val = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(val):
        return float(default)
    return val


def _norm_xy(x: Any, y: Any, cw: float, ch: float) -> tuple[float, float]:
    return _as_float(x) / max(cw, 1e-9), _as_float(y) / max(ch, 1e-9)


def _benchmark_dir(name: str, roots: list[Path]) -> Path:
    for root in roots:
        direct = root / name
        if (direct / "netlist.pb.txt").exists():
            return direct
        if root.name == name and (root / "netlist.pb.txt").exists():
            return root
        if name.endswith("_ng45"):
            design = name[: -len("_ng45")]
            ng45 = root / design / "netlist" / "output_CT_Grouping"
            if (ng45 / "netlist.pb.txt").exists():
                return ng45
    raise FileNotFoundError(f"Could not find benchmark '{name}' under: {roots}")


def _trace_files(trace_dir: Path | None, trace_path: Path | None) -> list[Path]:
    if trace_path is not None:
        return [trace_path]
    if trace_dir is None:
        raise ValueError("Provide --trace-dir or --trace-path")
    return sorted(p for p in trace_dir.glob("*.jsonl") if p.is_file())


def _read_trace_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if int(row.get("schema_version", 0)) != 1:
                    raise ValueError(f"{path}:{line_no}: expected schema_version=1")
                row["_trace_file"] = str(path)
                row["_trace_line"] = line_no
                rows.append(row)
    return rows


def _spatial_edges(
    pos: np.ndarray, start: int, end: int, k: int = 4
) -> list[tuple[int, int, list[float]]]:
    edges: list[tuple[int, int, list[float]]] = []
    if end - start <= 1:
        return edges
    sub = pos[start:end]
    span = max(float(np.ptp(pos[:, 0])), float(np.ptp(pos[:, 1])), 1.0)
    for local_i, xy in enumerate(sub):
        d = np.hypot(sub[:, 0] - xy[0], sub[:, 1] - xy[1])
        order = np.argsort(d)
        for local_j in order[1 : k + 1]:
            dist = float(d[local_j] / span)
            i, j = start + int(local_i), start + int(local_j)
            edges.append((i, j, [0.0, 0.0, 1.0, 0.0, 0.0, dist]))
    return edges


def _module_xy(
    plc: Any, ref: int, b_to_macro: dict[int, int], pos: np.ndarray
) -> tuple[float, float]:
    macro_i = b_to_macro.get(int(ref))
    if macro_i is not None and 0 <= macro_i < pos.shape[0]:
        return float(pos[macro_i, 0]), float(pos[macro_i, 1])
    module = plc.modules_w_pins[int(ref)]
    return float(getattr(module, "x", 0.0)), float(getattr(module, "y", 0.0))


def _macro_net_graph(
    plc: Any,
    cache: dict[str, Any],
    pos: np.ndarray,
    cw: float,
    ch: float,
    b_to_macro: dict[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_fanout = max(int(cache["net_lengths"].max()) if cache["net_lengths"].size else 1, 1)
    max_weight = max(float(cache["net_weights"].max()) if cache["net_weights"].size else 1.0, 1.0)
    max_dim = max(cw, ch, 1e-9)
    net_rows: list[list[float]] = []
    incidence_rows: list[tuple[int, int, list[float]]] = []

    for net_i, start_raw in enumerate(cache["net_starts"]):
        start = int(start_raw)
        length = int(cache["net_lengths"][net_i])
        end = start + length
        refs = [int(r) for r in cache["ref_idx"][start:end]]
        pin_x = np.zeros(length, dtype=np.float64)
        pin_y = np.zeros(length, dtype=np.float64)
        macro_refs = []
        for local_i, ref in enumerate(refs):
            base_x, base_y = _module_xy(plc, ref, b_to_macro, pos)
            pin_x[local_i] = base_x + float(cache["x_off"][start + local_i])
            pin_y[local_i] = base_y + float(cache["y_off"][start + local_i])
            if ref in b_to_macro:
                macro_refs.append((local_i, b_to_macro[ref]))

        hpwl_x = float((pin_x.max() - pin_x.min()) / max(cw, 1e-9)) if length else 0.0
        hpwl_y = float((pin_y.max() - pin_y.min()) / max(ch, 1e-9)) if length else 0.0
        weight = float(cache["net_weights"][net_i] / max_weight)
        fanout = float(length / max_fanout)
        macro_degree = float(len({m for _, m in macro_refs}) / max(max_fanout, 1))
        net_rows.append(
            [
                fanout,
                macro_degree,
                weight,
                hpwl_x,
                hpwl_y,
                float((hpwl_x + hpwl_y) * weight),
            ]
        )

        seen_macro_pin: set[tuple[int, int]] = set()
        for local_i, macro_i in macro_refs:
            key = (int(macro_i), int(net_i))
            if key in seen_macro_pin:
                continue
            seen_macro_pin.add(key)
            x_off = float(cache["x_off"][start + local_i] / max_dim)
            y_off = float(cache["y_off"][start + local_i] / max_dim)
            incidence_rows.append(
                (
                    int(macro_i),
                    int(net_i),
                    [
                        x_off,
                        y_off,
                        abs(x_off),
                        abs(y_off),
                        weight,
                        fanout,
                        1.0 if local_i == 0 else 0.0,
                    ],
                )
            )

    if incidence_rows:
        incidence_rows.sort(key=lambda e: (e[0], e[1], e[2]))
        edge_index = torch.tensor(
            [[e[0], e[1]] for e in incidence_rows], dtype=torch.long
        ).t().contiguous()
        edge_features = torch.tensor([e[2] for e in incidence_rows], dtype=torch.float32)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_features = torch.zeros((0, len(MACRO_NET_EDGE_FEATURES)), dtype=torch.float32)

    return torch.tensor(net_rows, dtype=torch.float32), edge_index, edge_features


def _build_graph(name: str, bench_roots: list[Path]) -> dict[str, Any]:
    bench_dir = _benchmark_dir(name, bench_roots)
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))
    n = int(benchmark.num_hard_macros)
    ns = int(benchmark.num_soft_macros)
    nm = int(benchmark.num_macros)
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)

    pos = benchmark.macro_positions.detach().cpu().numpy().astype(np.float64)
    sizes = benchmark.macro_sizes.detach().cpu().numpy().astype(np.float64)
    fixed = benchmark.macro_fixed.detach().cpu().numpy().astype(bool)
    movable = benchmark.get_movable_mask().detach().cpu().numpy().astype(bool)
    labels, clusters = derive_hard_clusters(
        plc,
        n,
        n_soft=ns,
        max_fanout=cluster_max_fanout(),
        min_edge=cluster_min_edge(),
    )
    owned_softs, bridge_softs = derive_soft_cluster_roles(
        plc,
        n,
        ns,
        labels,
        max_fanout=cluster_max_fanout(),
    )

    macro_cluster = np.full(nm, -1, dtype=np.int64)
    cluster_sizes = np.zeros(max(len(clusters), 1), dtype=np.float64)
    for cid, members in clusters.items():
        members = np.asarray(members, dtype=np.int64)
        macro_cluster[members] = int(cid)
        cluster_sizes[int(cid)] = float(members.size)
    for cid, softs in owned_softs.items():
        macro_cluster[np.asarray(softs, dtype=np.int64)] = int(cid)

    px = cw / max(int(benchmark.grid_cols), 1)
    py = ch / max(int(benchmark.grid_rows), 1)
    gx = (np.floor(pos[:, 0] / px) + 0.5) * px
    gy = (np.floor(pos[:, 1] / py) + 0.5) * py
    area_norm = max(cw * ch, 1e-9)
    max_dim = max(cw, ch, 1e-9)
    num_clusters = max(len(clusters), 1)

    node_rows = []
    for i in range(nm):
        cid = int(macro_cluster[i])
        csize = 0.0 if cid < 0 else cluster_sizes[cid] / max(nm, 1)
        node_rows.append(
            [
                1.0 if i < n else 0.0,
                1.0 if i >= n else 0.0,
                0.0,
                float(fixed[i]),
                float(movable[i]),
                float(sizes[i, 0] / max_dim),
                float(sizes[i, 1] / max_dim),
                float((sizes[i, 0] * sizes[i, 1]) / area_norm),
                float(pos[i, 0] / max(cw, 1e-9)),
                float(pos[i, 1] / max(ch, 1e-9)),
                float(pos[i, 0] / max(cw, 1e-9)),
                float((cw - pos[i, 0]) / max(cw, 1e-9)),
                float(pos[i, 1] / max(ch, 1e-9)),
                float((ch - pos[i, 1]) / max(ch, 1e-9)),
                float(abs(pos[i, 0] - gx[i]) / max(0.5 * px, 1e-9)),
                float(abs(pos[i, 1] - gy[i]) / max(0.5 * py, 1e-9)),
                0.0 if cid < 0 else float((cid + 1) / (num_clusters + 1)),
                float(csize),
            ]
        )

    cluster_node = {}
    for cid, members in sorted(clusters.items()):
        members = np.asarray(members, dtype=np.int64)
        cpos = pos[members].mean(axis=0)
        csize = float(members.size / max(nm, 1))
        cluster_node[int(cid)] = len(node_rows)
        node_rows.append(
            [
                0.0,
                0.0,
                1.0,
                0.0,
                1.0,
                0.0,
                0.0,
                float(np.sum(sizes[members, 0] * sizes[members, 1]) / area_norm),
                float(cpos[0] / max(cw, 1e-9)),
                float(cpos[1] / max(ch, 1e-9)),
                float(cpos[0] / max(cw, 1e-9)),
                float((cw - cpos[0]) / max(cw, 1e-9)),
                float(cpos[1] / max(ch, 1e-9)),
                float((ch - cpos[1]) / max(ch, 1e-9)),
                0.0,
                0.0,
                float((int(cid) + 1) / (num_clusters + 1)),
                csize,
            ]
        )

    edge_rows: list[tuple[int, int, list[float]]] = []
    cache = _build_wl_cache(plc)
    hard_b_to_a = {int(b): int(a) for a, b in enumerate(plc.hard_macro_indices)}
    soft_b_to_a = {int(b): int(n + a) for a, b in enumerate(plc.soft_macro_indices)}
    b_to_macro = {**hard_b_to_a, **soft_b_to_a}
    max_fanout = max(int(cache["net_lengths"].max()) if cache["net_lengths"].size else 1, 1)
    max_weight = max(float(cache["net_weights"].max()) if cache["net_weights"].size else 1.0, 1.0)
    net_node_features, macro_net_edge_index, macro_net_edge_features = _macro_net_graph(
        plc, cache, pos, cw, ch, b_to_macro
    )
    for net_i, start in enumerate(cache["net_starts"]):
        length = int(cache["net_lengths"][net_i])
        refs = [int(r) for r in cache["ref_idx"][int(start) : int(start) + length]]
        macros = sorted({b_to_macro[r] for r in refs if r in b_to_macro})
        if len(macros) < 2:
            continue
        weight = float(cache["net_weights"][net_i] / max_weight)
        fanout = float(length / max_fanout)
        for a_i in range(len(macros)):
            for b_i in range(a_i + 1, len(macros)):
                a, b = macros[a_i], macros[b_i]
                feat = [1.0, 0.0, 0.0, weight, fanout, 0.0]
                edge_rows.append((a, b, feat))
                edge_rows.append((b, a, feat))

    for macro_i, cid in enumerate(macro_cluster):
        if cid < 0 or int(cid) not in cluster_node:
            continue
        cnode = cluster_node[int(cid)]
        feat = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        edge_rows.append((macro_i, cnode, feat))
        edge_rows.append((cnode, macro_i, feat))

    edge_rows.extend(_spatial_edges(pos, 0, nm, k=4))
    edge_rows.sort(key=lambda e: (e[0], e[1], e[2]))
    edge_index = torch.tensor([[e[0], e[1]] for e in edge_rows], dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor([e[2] for e in edge_rows], dtype=torch.float32)

    return {
        "benchmark": name,
        "benchmark_dir": str(bench_dir),
        "canvas": torch.tensor([cw, ch], dtype=torch.float32),
        "num_hard_macros": n,
        "num_soft_macros": ns,
        "num_macros": nm,
        "num_clusters": len(clusters),
        "node_features": torch.tensor(node_rows, dtype=torch.float32),
        "edge_index": edge_index,
        "edge_features": edge_attr,
        "net_node_features": net_node_features,
        "macro_net_edge_index": macro_net_edge_index,
        "macro_net_edge_features": macro_net_edge_features,
        "macro_cluster": torch.tensor(macro_cluster, dtype=torch.long),
        "cluster_node": torch.tensor(
            [cluster_node.get(cid, -1) for cid in range(num_clusters)], dtype=torch.long
        ),
        "bridge_softs": {
            int(k): torch.tensor(v, dtype=torch.long) for k, v in bridge_softs.items()
        },
        "macro_names": list(benchmark.macro_names),
    }


def _candidate_feature(row: dict[str, Any], graph: dict[str, Any]) -> list[float]:
    cw, ch = [float(v) for v in graph["canvas"].tolist()]
    operator = row.get("operator") or row.get("event", "")
    if row.get("event") == "hier_relocation_candidate":
        operator = "relocation"
    target_x, target_y = _norm_xy(row.get("x"), row.get("y"), cw, ch)
    axis = row.get("axis_scale") or [0.0, 0.0]
    return [
        float(OPERATOR_IDS.get(str(operator), 0)),
        float(KIND_IDS.get(str(row.get("kind", "")), 0)),
        float(FIELD_IDS.get(str(row.get("field", "")), 0)),
        _as_float(row.get("candidate_rank"), -1.0) / 512.0,
        _as_float(row.get("source_field", row.get("local_field"))),
        _as_float(row.get("target_field")),
        _as_float(row.get("score")),
        _as_float(row.get("structural_delta")),
        float(bool(row.get("outside_region", False))),
        float(bool(row.get("legal", True))),
        _as_float(row.get("old_proxy")),
        _as_float(row.get("candidate_proxy")),
        float(row.get("candidate_proxy") is not None or row.get("proxy_delta") is not None),
        target_x,
        target_y,
        _as_float(row.get("hierarchy_quality_before")),
        _as_float(row.get("hierarchy_quality_after")),
        _as_float(row.get("hierarchy_quality_delta")),
        _as_float(row.get("expansion_factor"), 1.0),
        _as_float(axis[0] if len(axis) > 0 else 0.0),
        _as_float(axis[1] if len(axis) > 1 else 0.0),
        _as_float(row.get("field_gap")),
        _as_float(row.get("cluster_heat")),
        _as_float(row.get("window_cells")) / max(int(graph["num_macros"]), 1),
        _as_float(row.get("movable_count")) / max(int(graph["num_macros"]), 1),
        _as_float(row.get("member_count")) / max(int(graph["num_macros"]), 1),
        _as_float(row.get("soft_count")) / max(int(graph["num_soft_macros"]), 1),
    ]


def _source_target(row: dict[str, Any], graph: dict[str, Any]) -> tuple[int, int]:
    nm = int(graph["num_macros"])
    clusters = graph["cluster_node"]
    if "macro" in row:
        return int(row["macro"]), -1
    if row.get("kind") in {"hard_hard", "hard_soft", "soft_soft"}:
        source = int(row.get("source", -1))
        target = int(row.get("target", -1))
        if row.get("kind") == "hard_soft" and target >= 0:
            target += int(graph["num_hard_macros"])
        if row.get("kind") == "soft_soft":
            source += int(graph["num_hard_macros"])
            target += int(graph["num_hard_macros"])
        return source, target
    if "cluster" in row:
        cid = int(row.get("cluster", -1))
        if 0 <= cid < int(clusters.numel()):
            return int(clusters[cid]), -1
    return -1 if nm == 0 else 0, -1


def _flush_relocation_pending(
    pending: dict[str, Any] | None,
    accepted: list[dict[str, Any]],
    examples: list[dict[str, Any]],
    graph_ids: dict[str, int],
    graphs: dict[str, dict[str, Any]],
) -> None:
    if pending is None:
        return
    benchmark = str(pending.get("benchmark", ""))
    if benchmark not in graphs:
        return
    graph = graphs[benchmark]
    accepted_keys = {
        (
            int(a.get("macro", a.get("soft_macro", -1))),
            int(a.get("target_index", -1)),
            round(_as_float(a.get("x")), 6),
            round(_as_float(a.get("y")), 6),
        ): a
        for a in accepted
    }
    for cand in pending.get("candidates", []):
        row = dict(cand)
        row["event"] = "hier_relocation_candidate"
        row["operator"] = "relocation"
        row["kind"] = pending.get("kind", "")
        row["field"] = pending.get("field", "")
        row["old_proxy"] = pending.get("initial_proxy")
        key = (
            int(row.get("macro", -1)),
            int(row.get("target_index", -1)),
            round(_as_float(row.get("x")), 6),
            round(_as_float(row.get("y")), 6),
        )
        hit = accepted_keys.get(key)
        accepted_flag = hit is not None
        if hit is not None:
            row["candidate_proxy"] = hit.get("new_proxy")
            row["proxy_delta"] = hit.get("proxy_delta")
        source, target = _source_target(row, graph)
        examples.append(
            {
                "graph_id": graph_ids[benchmark],
                "benchmark": benchmark,
                "operator": "relocation",
                "kind": row.get("kind", ""),
                "source_node": source,
                "target_node": target,
                "features": _candidate_feature(row, graph),
                "accepted": bool(accepted_flag),
                "proxy_delta": _as_float(row.get("proxy_delta")),
                "proxy_delta_known": bool(row.get("proxy_delta") is not None),
                "rejection_id": 0 if accepted_flag else REJECTION_IDS["not_scored"],
                "trace_file": pending.get("_trace_file", ""),
                "trace_line": int(pending.get("_trace_line", 0)),
            }
        )


def _build_examples(
    rows: list[dict[str, Any]],
    graphs: dict[str, dict[str, Any]],
    graph_ids: dict[str, int],
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    pending_reloc: dict[str, Any] | None = None
    for row in rows:
        event = row.get("event")
        benchmark = str(row.get("benchmark", ""))
        if event == "hier_relocation_candidates":
            _flush_relocation_pending(pending_reloc, [], examples, graph_ids, graphs)
            pending_reloc = row
            continue
        if event == "hier_relocation_result":
            if pending_reloc is not None and pending_reloc.get("benchmark") == benchmark:
                _flush_relocation_pending(
                    pending_reloc, list(row.get("accepted", [])), examples, graph_ids, graphs
                )
                pending_reloc = None
            continue
        if benchmark not in graphs:
            continue
        graph = graphs[benchmark]
        direct_rows: list[dict[str, Any]] = []
        if event in {"hier_decompression_candidate", "hier_coldspot_candidate"}:
            direct_rows = [row]
        elif event == "hier_swap_candidates":
            for cand in row.get("candidates", []):
                merged = dict(cand)
                merged.update(
                    {
                        "event": event,
                        "operator": row.get("operator"),
                        "kind": row.get("kind"),
                        "field": row.get("field"),
                        "source": row.get("source"),
                        "benchmark": benchmark,
                        "_trace_file": row.get("_trace_file", ""),
                        "_trace_line": row.get("_trace_line", 0),
                    }
                )
                direct_rows.append(merged)
        for cand in direct_rows:
            source, target = _source_target(cand, graph)
            accepted = bool(cand.get("accepted", False))
            examples.append(
                {
                    "graph_id": graph_ids[benchmark],
                    "benchmark": benchmark,
                    "operator": str(cand.get("operator", "")),
                    "kind": str(cand.get("kind", "")),
                    "source_node": source,
                    "target_node": target,
                    "features": _candidate_feature(cand, graph),
                    "accepted": accepted,
                    "proxy_delta": _as_float(cand.get("proxy_delta")),
                    "proxy_delta_known": bool(cand.get("proxy_delta") is not None),
                    "rejection_id": REJECTION_IDS.get(cand.get("rejection_reason"), 0),
                    "trace_file": cand.get("_trace_file", ""),
                    "trace_line": int(cand.get("_trace_line", 0)),
                }
            )
    _flush_relocation_pending(pending_reloc, [], examples, graph_ids, graphs)
    return examples


def _stack_examples(examples: list[dict[str, Any]]) -> dict[str, Any]:
    if not examples:
        return {
            "graph_id": torch.zeros(0, dtype=torch.long),
            "source_node": torch.zeros(0, dtype=torch.long),
            "target_node": torch.zeros(0, dtype=torch.long),
            "features": torch.zeros((0, len(CANDIDATE_FEATURES)), dtype=torch.float32),
            "accepted": torch.zeros(0, dtype=torch.bool),
            "proxy_delta": torch.zeros(0, dtype=torch.float32),
            "proxy_delta_known": torch.zeros(0, dtype=torch.bool),
            "rejection_id": torch.zeros(0, dtype=torch.long),
            "operator": [],
            "kind": [],
            "benchmark": [],
            "trace_file": [],
            "trace_line": torch.zeros(0, dtype=torch.long),
        }
    return {
        "graph_id": torch.tensor([e["graph_id"] for e in examples], dtype=torch.long),
        "source_node": torch.tensor([e["source_node"] for e in examples], dtype=torch.long),
        "target_node": torch.tensor([e["target_node"] for e in examples], dtype=torch.long),
        "features": torch.tensor([e["features"] for e in examples], dtype=torch.float32),
        "accepted": torch.tensor([e["accepted"] for e in examples], dtype=torch.bool),
        "proxy_delta": torch.tensor([e["proxy_delta"] for e in examples], dtype=torch.float32),
        "proxy_delta_known": torch.tensor(
            [e["proxy_delta_known"] for e in examples], dtype=torch.bool
        ),
        "rejection_id": torch.tensor([e["rejection_id"] for e in examples], dtype=torch.long),
        "operator": [e["operator"] for e in examples],
        "kind": [e["kind"] for e in examples],
        "benchmark": [e["benchmark"] for e in examples],
        "trace_file": [e["trace_file"] for e in examples],
        "trace_line": torch.tensor([e["trace_line"] for e in examples], dtype=torch.long),
    }


def _write_feature_schema(path: Path) -> None:
    schema = {
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "trace_schema_version": 1,
        "node_features": NODE_FEATURES,
        "edge_features": EDGE_FEATURES,
        "net_node_features": NET_NODE_FEATURES,
        "macro_net_edge_features": MACRO_NET_EDGE_FEATURES,
        "candidate_features": CANDIDATE_FEATURES,
        "operator_ids": OPERATOR_IDS,
        "kind_ids": KIND_IDS,
        "field_ids": FIELD_IDS,
        "rejection_ids": {str(k): v for k, v in REJECTION_IDS.items()},
    }
    path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    trace_paths = _trace_files(args.trace_dir, args.trace_path)
    rows = _read_trace_rows(trace_paths)
    benchmark_names = sorted({str(r.get("benchmark", "")) for r in rows if r.get("benchmark")})
    if args.benchmark:
        keep = set(args.benchmark)
        rows = [r for r in rows if r.get("benchmark") in keep]
        benchmark_names = sorted(keep & set(benchmark_names))
    bench_roots = [Path(p) for p in args.benchmark_root]
    graphs = {name: _build_graph(name, bench_roots) for name in benchmark_names}
    graph_ids = {name: i for i, name in enumerate(sorted(graphs))}
    graph_list = [graphs[name] for name in sorted(graphs)]
    examples_raw = _build_examples(rows, graphs, graph_ids)
    examples = _stack_examples(examples_raw)
    return {
        "metadata": {
            "dataset_schema_version": DATASET_SCHEMA_VERSION,
            "trace_schema_version": 1,
            "graph_schema": "macro_cluster_plus_macro_net_v2",
            "trace_files": [str(p) for p in trace_paths],
            "benchmarks": sorted(graphs),
            "num_graphs": len(graph_list),
            "num_examples": int(examples["features"].shape[0]),
            "num_accepted": int(examples["accepted"].sum().item()),
        },
        "feature_schema": {
            "node_features": NODE_FEATURES,
            "edge_features": EDGE_FEATURES,
            "net_node_features": NET_NODE_FEATURES,
            "macro_net_edge_features": MACRO_NET_EDGE_FEATURES,
            "candidate_features": CANDIDATE_FEATURES,
            "operator_ids": OPERATOR_IDS,
            "kind_ids": KIND_IDS,
            "field_ids": FIELD_IDS,
            "rejection_ids": {str(k): v for k, v in REJECTION_IDS.items()},
        },
        "graphs": graph_list,
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, help="Directory of schema-v1 JSONL traces")
    parser.add_argument("--trace-path", type=Path, help="Single schema-v1 JSONL trace")
    parser.add_argument("--out", type=Path, required=True, help="Output dataset .pt path")
    parser.add_argument(
        "--benchmark-root",
        action="append",
        default=[ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04"],
        help="Benchmark root containing benchmark-name subdirectories",
    )
    parser.add_argument("--benchmark", action="append", help="Limit to this benchmark name")
    parser.add_argument(
        "--schema-out",
        type=Path,
        help="Feature schema JSON path. Defaults to feature_schema.json next to --out",
    )
    args = parser.parse_args()

    if bool(args.trace_dir) == bool(args.trace_path):
        parser.error("provide exactly one of --trace-dir or --trace-path")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    schema_out = args.schema_out or (args.out.parent / "feature_schema.json")
    dataset = build_dataset(args)
    torch.save(dataset, args.out)
    _write_feature_schema(schema_out)
    meta = dataset["metadata"]
    print(
        f"wrote {args.out} with {meta['num_graphs']} graph(s), "
        f"{meta['num_examples']} examples, {meta['num_accepted']} accepted"
    )
    print(f"wrote {schema_out}")


if __name__ == "__main__":
    main()
