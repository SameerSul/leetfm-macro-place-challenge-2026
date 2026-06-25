#!/usr/bin/env python3
"""Summarize hierarchy graph-tension traces."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _read_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line_no, raw in enumerate(f, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
                row["_path"] = str(path)
                row["_line"] = int(line_no)
                rows.append(row)
    return rows


def _num(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _bench(row: dict[str, Any]) -> str:
    return str(row.get("benchmark") or row.get("benchmark_name") or "unknown")


def _candidate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if row.get("event") in {
            "hier_decompression_candidate",
            "hier_coldspot_candidate",
            "hier_swap_candidate",
        }:
            out.append(row)
    return out


def _print_tension_events(rows: list[dict[str, Any]], limit: int) -> None:
    events = [row for row in rows if row.get("event") == "hier_graph_tension"]
    print(f"graph_tension_events: {len(events)}")
    by_bench = Counter(_bench(row) for row in events)
    for bench, count in sorted(by_bench.items()):
        print(f"  {bench}: {count}")
    if not events:
        return
    print("top_tension_snapshots:")
    for row in events[: max(0, limit)]:
        clusters = row.get("top_clusters") or []
        scores = row.get("top_scores") or []
        pairs = ", ".join(
            f"{int(cid)}:{float(score):.3f}" for cid, score in zip(clusters, scores)
        )
        print(f"  {_bench(row)} {row.get('label', '')}: {pairs}")


def _print_candidate_summary(candidates: list[dict[str, Any]]) -> None:
    with_tension = [row for row in candidates if "graph_tension" in row]
    egonet = [row for row in candidates if bool(row.get("egonet_candidate"))]
    graph_anchor = [row for row in candidates if bool(row.get("graph_anchor_enabled"))]
    prefiltered = [
        row
        for row in candidates
        if bool(row.get("prefiltered"))
        or str(row.get("rejection_reason") or "").startswith("prefilter_")
    ]
    feasibility = [
        row
        for row in candidates
        if bool(row.get("feasibility_rejected"))
        or str(row.get("rejection_reason") or "") == "feasibility_blocked"
    ]
    graph_delta = [row for row in candidates if "graph_candidate_delta" in row]
    graph_delta_rank = [row for row in candidates if "graph_delta_rank_penalty" in row]
    graph_rescue = [row for row in candidates if bool(row.get("graph_rescue_attempted"))]
    graph_survivor = [row for row in candidates if bool(row.get("graph_survivor_attempted"))]
    print(f"candidate_rows: {len(candidates)}")
    print(f"candidate_rows_with_graph_tension: {len(with_tension)}")
    print(f"egonet_candidate_rows: {len(egonet)}")
    print(f"graph_anchor_candidate_rows: {len(graph_anchor)}")
    print(f"prefiltered_candidate_rows: {len(prefiltered)}")
    print(f"feasibility_rejected_rows: {len(feasibility)}")
    print(f"graph_delta_candidate_rows: {len(graph_delta)}")
    print(f"graph_delta_ranked_rows: {len(graph_delta_rank)}")
    print(f"graph_rescue_attempted_rows: {len(graph_rescue)}")
    print(f"graph_survivor_attempted_rows: {len(graph_survivor)}")
    if not with_tension:
        if egonet:
            _print_egonet_summary(egonet)
            _print_egonet_rejections(egonet)
        if graph_anchor:
            _print_graph_anchor_summary(graph_anchor)
        if prefiltered:
            _print_prefilter_summary(prefiltered)
        if feasibility:
            _print_feasibility_summary(feasibility)
        if graph_delta:
            _print_graph_delta_summary(graph_delta)
        if graph_delta_rank:
            _print_graph_delta_rank_summary(graph_delta_rank)
        if graph_rescue:
            _print_graph_rescue_summary(graph_rescue)
        if graph_survivor:
            _print_graph_survivor_summary(graph_survivor)
        return

    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in with_tension:
        buckets[(_bench(row), str(row.get("operator", row.get("event", "unknown"))))].append(row)

    print("candidate_summary:")
    for (bench, operator), rows in sorted(buckets.items()):
        accepted = [row for row in rows if bool(row.get("accepted"))]
        committed = [row for row in rows if bool(row.get("committed"))]
        exact = [row for row in rows if row.get("proxy_delta") is not None]
        gain = sum(max(0.0, -_num(row, "proxy_delta")) for row in committed)
        avg_tension = sum(_num(row, "graph_tension") for row in rows) / max(1, len(rows))
        accepted_tension = (
            sum(_num(row, "graph_tension") for row in accepted) / len(accepted)
            if accepted
            else 0.0
        )
        committed_tension = (
            sum(_num(row, "graph_tension") for row in committed) / len(committed)
            if committed
            else 0.0
        )
        print(
            f"  {bench} {operator}: rows={len(rows)} exact={len(exact)} "
            f"accepted={len(accepted)} committed={len(committed)} "
            f"committed_gain={gain:.6f} avg_tension={avg_tension:.3f} "
            f"accepted_tension={accepted_tension:.3f} committed_tension={committed_tension:.3f}"
        )
    if egonet:
        _print_egonet_summary(egonet)
        _print_egonet_rejections(egonet)
    if graph_anchor:
        _print_graph_anchor_summary(graph_anchor)
    if prefiltered:
        _print_prefilter_summary(prefiltered)
    if feasibility:
        _print_feasibility_summary(feasibility)
    if graph_delta:
        _print_graph_delta_summary(graph_delta)
    if graph_delta_rank:
        _print_graph_delta_rank_summary(graph_delta_rank)
    if graph_rescue:
        _print_graph_rescue_summary(graph_rescue)
    if graph_survivor:
        _print_graph_survivor_summary(graph_survivor)


def _print_egonet_summary(rows: list[dict[str, Any]]) -> None:
    print("egonet_summary:")
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[_bench(row)].append(row)
    for bench, bench_rows in sorted(buckets.items()):
        accepted = [row for row in bench_rows if bool(row.get("accepted"))]
        committed = [row for row in bench_rows if bool(row.get("committed"))]
        exact = [row for row in bench_rows if row.get("proxy_delta") is not None]
        gain = sum(max(0.0, -_num(row, "proxy_delta")) for row in committed)
        avg_neighbors = sum(_num(row, "egonet_neighbor_count") for row in bench_rows) / max(
            1, len(bench_rows)
        )
        avg_members = sum(_num(row, "egonet_member_count") for row in bench_rows) / max(
            1, len(bench_rows)
        )
        avg_soft = sum(_num(row, "soft_count") for row in bench_rows) / max(1, len(bench_rows))
        avg_hard_disp = sum(_num(row, "hard_disp_mean") for row in bench_rows) / max(
            1, len(bench_rows)
        )
        avg_proxy_delta = sum(_num(row, "proxy_delta") for row in exact) / max(1, len(exact))
        avg_required_gain = sum(_num(row, "required_min_gain") for row in exact) / max(
            1, len(exact)
        )
        avg_quality_delta = sum(_num(row, "hierarchy_quality_delta") for row in exact) / max(
            1, len(exact)
        )
        rejection_counts = Counter(
            str(row.get("rejection_reason") or "unknown")
            for row in bench_rows
            if not bool(row.get("accepted"))
        )
        top_reject = ", ".join(f"{k}:{v}" for k, v in rejection_counts.most_common(4))
        print(
            f"  {bench}: rows={len(bench_rows)} exact={len(exact)} "
            f"accepted={len(accepted)} committed={len(committed)} "
            f"committed_gain={gain:.6f} avg_neighbors={avg_neighbors:.2f} "
            f"avg_members={avg_members:.1f} avg_soft={avg_soft:.1f} "
            f"avg_hard_disp={avg_hard_disp:.2f} avg_proxy_delta={avg_proxy_delta:.4f} "
            f"avg_required_gain={avg_required_gain:.4f} "
            f"avg_quality_delta={avg_quality_delta:.4f} rejections=[{top_reject}]"
        )


def _print_egonet_rejections(rows: list[dict[str, Any]]) -> None:
    exact_rows = [row for row in rows if row.get("proxy_delta") is not None]
    if not exact_rows:
        return
    print("egonet_rejection_detail:")
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in exact_rows:
        reason = "accepted" if bool(row.get("accepted")) else str(row.get("rejection_reason"))
        buckets[(_bench(row), reason)].append(row)
    for (bench, reason), bench_rows in sorted(buckets.items()):
        avg_proxy_delta = sum(_num(row, "proxy_delta") for row in bench_rows) / max(
            1, len(bench_rows)
        )
        avg_required_gain = sum(_num(row, "required_min_gain") for row in bench_rows) / max(
            1, len(bench_rows)
        )
        avg_quality_delta = sum(_num(row, "hierarchy_quality_delta") for row in bench_rows) / max(
            1, len(bench_rows)
        )
        avg_members = sum(_num(row, "egonet_member_count") for row in bench_rows) / max(
            1, len(bench_rows)
        )
        avg_soft = sum(_num(row, "soft_count") for row in bench_rows) / max(1, len(bench_rows))
        avg_neighbor_hard = sum(_num(row, "egonet_neighbor_hard_count") for row in bench_rows) / max(
            1, len(bench_rows)
        )
        avg_hard_disp = sum(_num(row, "hard_disp_mean") for row in bench_rows) / max(
            1, len(bench_rows)
        )
        print(
            f"  {bench} {reason}: rows={len(bench_rows)} "
            f"avg_proxy_delta={avg_proxy_delta:.4f} "
            f"avg_required_gain={avg_required_gain:.4f} "
            f"avg_quality_delta={avg_quality_delta:.4f} "
            f"avg_members={avg_members:.1f} avg_neighbor_hard={avg_neighbor_hard:.1f} "
            f"avg_soft={avg_soft:.1f} avg_hard_disp={avg_hard_disp:.2f}"
        )


def _print_graph_anchor_summary(rows: list[dict[str, Any]]) -> None:
    print("graph_anchor_summary:")
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[_bench(row)].append(row)
    for bench, bench_rows in sorted(buckets.items()):
        accepted = [row for row in bench_rows if bool(row.get("accepted"))]
        committed = [row for row in bench_rows if bool(row.get("committed"))]
        exact = [row for row in bench_rows if row.get("proxy_delta") is not None]
        gain = sum(max(0.0, -_num(row, "proxy_delta")) for row in committed)
        avg_weight = sum(_num(row, "graph_anchor_weight") for row in bench_rows) / max(
            1, len(bench_rows)
        )
        avg_distance = sum(_num(row, "graph_anchor_distance") for row in bench_rows) / max(
            1, len(bench_rows)
        )
        avg_proxy_delta = sum(_num(row, "proxy_delta") for row in exact) / max(1, len(exact))
        rejection_counts = Counter(
            str(row.get("rejection_reason") or "unknown")
            for row in bench_rows
            if not bool(row.get("accepted"))
        )
        top_reject = ", ".join(f"{k}:{v}" for k, v in rejection_counts.most_common(4))
        print(
            f"  {bench}: rows={len(bench_rows)} exact={len(exact)} "
            f"accepted={len(accepted)} committed={len(committed)} "
            f"committed_gain={gain:.6f} avg_weight={avg_weight:.3f} "
            f"avg_distance={avg_distance:.2f} avg_proxy_delta={avg_proxy_delta:.4f} "
            f"rejections=[{top_reject}]"
        )


def _print_prefilter_summary(rows: list[dict[str, Any]]) -> None:
    print("prefilter_summary:")
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(_bench(row), str(row.get("operator", row.get("event", "unknown"))))].append(row)
    for (bench, operator), bench_rows in sorted(buckets.items()):
        avg_tension = sum(_num(row, "graph_tension") for row in bench_rows) / max(
            1, len(bench_rows)
        )
        avg_relief = sum(_num(row, "local_relief") for row in bench_rows) / max(
            1, len(bench_rows)
        )
        reasons = Counter(str(row.get("rejection_reason") or "unknown") for row in bench_rows)
        top_reject = ", ".join(f"{k}:{v}" for k, v in reasons.most_common(4))
        print(
            f"  {bench} {operator}: rows={len(bench_rows)} "
            f"avg_tension={avg_tension:.3f} avg_local_relief={avg_relief:.4f} "
            f"rejections=[{top_reject}]"
        )


def _print_feasibility_summary(rows: list[dict[str, Any]]) -> None:
    print("feasibility_summary:")
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(_bench(row), str(row.get("operator", row.get("event", "unknown"))))].append(row)
    for (bench, operator), bench_rows in sorted(buckets.items()):
        avg_tension = sum(_num(row, "graph_tension") for row in bench_rows) / max(
            1, len(bench_rows)
        )
        avg_free = sum(_num(row, "feasible_free_ratio") for row in bench_rows) / max(
            1, len(bench_rows)
        )
        avg_blockage = sum(_num(row, "feasible_blockage_ratio") for row in bench_rows) / max(
            1, len(bench_rows)
        )
        reasons = Counter(str(row.get("rejection_reason") or "unknown") for row in bench_rows)
        top_reject = ", ".join(f"{k}:{v}" for k, v in reasons.most_common(4))
        print(
            f"  {bench} {operator}: rows={len(bench_rows)} "
            f"avg_tension={avg_tension:.3f} avg_free_ratio={avg_free:.3f} "
            f"avg_blockage={avg_blockage:.3f} rejections=[{top_reject}]"
        )


def _print_graph_delta_summary(rows: list[dict[str, Any]]) -> None:
    print("graph_delta_summary:")
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(_bench(row), str(row.get("operator", row.get("event", "unknown"))))].append(row)
    for (bench, operator), bench_rows in sorted(buckets.items()):
        exact = [row for row in bench_rows if row.get("proxy_delta") is not None]
        accepted = [row for row in bench_rows if bool(row.get("accepted"))]
        committed = [row for row in bench_rows if bool(row.get("committed"))]
        avg_delta = sum(_num(row, "graph_candidate_delta") for row in bench_rows) / max(
            1, len(bench_rows)
        )
        avg_stretch = sum(_num(row, "edge_stretch_delta") for row in bench_rows) / max(
            1, len(bench_rows)
        )
        avg_corridor = sum(_num(row, "corridor_congestion_delta") for row in bench_rows) / max(
            1, len(bench_rows)
        )
        accepted_delta = (
            sum(_num(row, "graph_candidate_delta") for row in accepted) / len(accepted)
            if accepted
            else 0.0
        )
        committed_delta = (
            sum(_num(row, "graph_candidate_delta") for row in committed) / len(committed)
            if committed
            else 0.0
        )
        avg_proxy_delta = sum(_num(row, "proxy_delta") for row in exact) / max(1, len(exact))
        print(
            f"  {bench} {operator}: rows={len(bench_rows)} exact={len(exact)} "
            f"accepted={len(accepted)} committed={len(committed)} "
            f"avg_graph_delta={avg_delta:.4f} avg_stretch_delta={avg_stretch:.4f} "
            f"avg_corridor_delta={avg_corridor:.4f} accepted_delta={accepted_delta:.4f} "
            f"committed_delta={committed_delta:.4f} avg_proxy_delta={avg_proxy_delta:.4f}"
        )


def _print_graph_delta_rank_summary(rows: list[dict[str, Any]]) -> None:
    print("graph_delta_rank_summary:")
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(_bench(row), str(row.get("operator", row.get("event", "unknown"))))].append(row)
    for (bench, operator), bench_rows in sorted(buckets.items()):
        penalties = [_num(row, "graph_delta_rank_penalty") for row in bench_rows]
        proxies = [_num(row, "graph_delta_rank_proxy") for row in bench_rows]
        weights = [_num(row, "graph_delta_rank_weight") for row in bench_rows]
        accepted = [row for row in bench_rows if bool(row.get("accepted"))]
        committed = [row for row in bench_rows if bool(row.get("committed"))]
        print(
            f"  {bench} {operator}: rows={len(bench_rows)} "
            f"avg_weight={sum(weights) / max(1, len(weights)):.6f} "
            f"avg_penalty={sum(penalties) / max(1, len(penalties)):.6f} "
            f"max_penalty={max(penalties) if penalties else 0.0:.6f} "
            f"avg_rank_proxy={sum(proxies) / max(1, len(proxies)):.4f} "
            f"accepted={len(accepted)} committed={len(committed)}"
        )


def _print_graph_rescue_summary(rows: list[dict[str, Any]]) -> None:
    print("graph_rescue_summary:")
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(_bench(row), str(row.get("operator", row.get("event", "unknown"))))].append(row)
    for (bench, operator), bench_rows in sorted(buckets.items()):
        used = [row for row in bench_rows if bool(row.get("graph_rescue_used"))]
        accepted = [row for row in bench_rows if bool(row.get("accepted"))]
        attempts = sum(int(_num(row, "graph_rescue_attempts")) for row in bench_rows)
        avg_trigger_delta = sum(_num(row, "graph_rescue_trigger_delta") for row in bench_rows) / max(
            1, len(bench_rows)
        )
        reasons = Counter(str(row.get("graph_rescue_trigger") or "unknown") for row in bench_rows)
        top_reasons = ", ".join(f"{k}:{v}" for k, v in reasons.most_common(4))
        print(
            f"  {bench} {operator}: rows={len(bench_rows)} used={len(used)} "
            f"accepted={len(accepted)} attempts={attempts} "
            f"avg_trigger_delta={avg_trigger_delta:.4f} triggers=[{top_reasons}]"
        )


def _print_graph_survivor_summary(rows: list[dict[str, Any]]) -> None:
    print("graph_survivor_summary:")
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(_bench(row), str(row.get("operator", row.get("event", "unknown"))))].append(row)
    for (bench, operator), bench_rows in sorted(buckets.items()):
        used = [row for row in bench_rows if bool(row.get("graph_survivor_used"))]
        accepted = [row for row in bench_rows if bool(row.get("accepted"))]
        trials = sum(int(_num(row, "graph_survivor_trials")) for row in bench_rows)
        avg_pre = sum(_num(row, "graph_survivor_pre_score") for row in bench_rows) / max(
            1, len(bench_rows)
        )
        avg_delta = sum(_num(row, "graph_candidate_delta") for row in bench_rows) / max(
            1, len(bench_rows)
        )
        print(
            f"  {bench} {operator}: rows={len(bench_rows)} used={len(used)} "
            f"accepted={len(accepted)} trials={trials} avg_pre_score={avg_pre:.4f} "
            f"avg_graph_delta={avg_delta:.4f}"
        )


def _print_rejections(candidates: list[dict[str, Any]], limit: int) -> None:
    rows = [row for row in candidates if "graph_tension" in row and not bool(row.get("accepted"))]
    counter = Counter(str(row.get("rejection_reason") or "unknown") for row in rows)
    if not counter:
        return
    print("rejections_with_graph_tension:")
    for reason, count in counter.most_common(max(1, limit)):
        print(f"  {reason}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", nargs="+", type=Path, help="JSONL trace file(s)")
    parser.add_argument("--snapshot-limit", type=int, default=8)
    parser.add_argument("--rejection-limit", type=int, default=12)
    args = parser.parse_args()

    rows = _read_rows(args.trace)
    candidates = _candidate_rows(rows)
    _print_tension_events(rows, args.snapshot_limit)
    _print_candidate_summary(candidates)
    _print_rejections(candidates, args.rejection_limit)


if __name__ == "__main__":
    main()
