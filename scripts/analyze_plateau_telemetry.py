#!/usr/bin/env python3
"""Aggregate hierarchy pass yield from plateau telemetry JSONL files."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

DEFAULT_PATH = Path("ml_data/plateau_telemetry/plateau_telemetry.jsonl")


def _quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(1.0, float(fraction))) * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _matches(row: dict, args) -> bool:
    if getattr(args, "run_id", None) and str(row.get("run_id", "legacy")) != args.run_id:
        return False
    if (
        getattr(args, "revision", None)
        and str(row.get("code_revision", "unknown")) != args.revision
    ):
        return False
    fingerprint = getattr(args, "worktree_fingerprint", None)
    if fingerprint and str(row.get("worktree_fingerprint", "legacy")) != fingerprint:
        return False
    if getattr(args, "benchmark", None) and str(row.get("benchmark", "")) not in args.benchmark:
        return False
    return True


def _load_event_rows(paths: list[Path], args, event: str) -> list[dict]:
    rows = []
    for path in paths:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
                if row.get("event") == event and _matches(row, args):
                    rows.append(row)
    return rows


def load_rows(paths: list[Path], args) -> list[dict]:
    return _load_event_rows(paths, args, "hier_plateau_telemetry")


def load_stage_rows(paths: list[Path], args) -> list[dict]:
    return _load_event_rows(paths, args, "hier_stage_timing")


def aggregate(rows: list[dict], min_runs: int) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["plateau_pass"])].append(row)
    output = []
    for name, items in sorted(grouped.items()):
        runs = len(items)
        if runs < min_runs:
            continue
        elapsed = sum(float(item.get("elapsed_s", 0.0)) for item in items)
        proposed_gain = sum(
            float(item.get("proposed_proxy_gain", item.get("proxy_gain", 0.0))) for item in items
        )
        retained_gain = sum(
            float(item.get("retained_proxy_gain", item.get("proxy_gain", 0.0))) for item in items
        )
        proposed_accepts = sum(
            int(item.get("proposed_accepts", item.get("accepts", 0))) for item in items
        )
        retained_accepts = sum(
            int(item.get("retained_accepts", item.get("accepts", 0))) for item in items
        )
        candidates = sum(int(item.get("candidates", 0)) for item in items)
        scored = sum(int(item.get("scored", 0)) for item in items)
        rollback_count = sum(1 for item in items if bool(item.get("audit_rollback", False)))
        audit_rebuild_s = sum(float(item.get("audit_rebuild_s", 0.0)) for item in items)
        discarded_gain = max(0.0, proposed_gain - retained_gain)
        zero_retained_gain = sum(
            float(item.get("retained_proxy_gain", item.get("proxy_gain", 0.0))) < 0.00005
            for item in items
        )
        zero_proposed_gain = sum(
            float(item.get("proposed_proxy_gain", item.get("proxy_gain", 0.0))) < 0.00005
            for item in items
        )
        zero_accept = sum(int(item.get("accepts", 0)) == 0 for item in items)
        retained_gain_per_s = retained_gain / max(elapsed, 1.0e-12)
        retained_gain_per_scored = retained_gain / scored if scored > 0 else None
        rollback_fraction = rollback_count / runs
        audit_rebuild_ratio = audit_rebuild_s / max(elapsed, 1.0e-12)
        retained_to_proposed_gain = (
            retained_gain / max(proposed_gain, 1.0e-12) if proposed_gain > 0 else 1.0
        )
        skip_candidate = (
            elapsed >= 0.5 and zero_retained_gain / runs >= 0.8 and retained_gain_per_s < 0.00001
        )
        output.append(
            {
                "pass": name,
                "runs": runs,
                "benchmarks": len({str(item.get("benchmark", "")) for item in items}),
                "candidates": candidates,
                "accepts": retained_accepts,
                "proposed_accepts": proposed_accepts,
                "proxy_gain": retained_gain,
                "proposed_proxy_gain": proposed_gain,
                "rollback_count": rollback_count,
                "rollback_fraction": rollback_fraction,
                "audit_rebuild_s": audit_rebuild_s,
                "audit_rebuild_ratio": audit_rebuild_ratio,
                "discarded_proxy_gain": discarded_gain,
                "retained_to_proposed_gain": retained_to_proposed_gain,
                "retained_gain_per_scored": retained_gain_per_scored,
                "elapsed_s": elapsed,
                "gain_per_s": retained_gain_per_s,
                "zero_gain_fraction": zero_retained_gain / runs,
                "zero_proposed_gain_fraction": zero_proposed_gain / runs,
                "zero_accept_fraction": zero_accept / runs,
                "recommendation": "skip_candidate" if skip_candidate else "retain_or_measure",
            }
        )
    return sorted(
        output,
        key=lambda row: (
            (
                float(row["retained_gain_per_scored"])
                if row["retained_gain_per_scored"] is not None
                else float("-inf")
            ),
            row["gain_per_s"],
            row["proxy_gain"],
        ),
        reverse=True,
    )


def print_table(rows: list[dict]) -> None:
    header = (
        f"{'pass':26} {'runs':>4} {'benches':>7} {'gain':>10} {'seconds':>8} "
        f"{'g/s':>10} {'g/sc':>10} {'rollback%':>9} {'rebuild/s':>9} {'zero%':>7} recommendation"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        gain_per_scored = row["retained_gain_per_scored"]
        gain_per_scored_text = (
            f"{float(gain_per_scored):10.7f}" if gain_per_scored is not None else f"{'n/a':>10}"
        )
        print(
            f"{row['pass'][:26]:26} {row['runs']:4d} {row['benchmarks']:7d} "
            f"{row['proxy_gain']:10.6f} {row['elapsed_s']:8.2f} {row['gain_per_s']:10.7f} "
            f"{gain_per_scored_text} {100.0 * row['rollback_fraction']:8.1f}% "
            f"{row['audit_rebuild_ratio']:9.7f} {100.0 * row['zero_gain_fraction']:6.1f}% "
            f"{row['recommendation']}"
        )


def aggregate_quotas(rows: list[dict], min_runs: int) -> list[dict]:
    """Summarize exact-scored candidate use per pass and benchmark."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["plateau_pass"])].append(row)
    output = []
    for name, items in sorted(grouped.items()):
        if len(items) < min_runs:
            continue
        by_benchmark: dict[str, dict] = defaultdict(
            lambda: {"scored": 0, "elapsed_s": 0.0, "gain": 0.0, "exhausted": False}
        )
        limits = set()
        for item in items:
            benchmark = str(item.get("benchmark", ""))
            summary = by_benchmark[benchmark]
            summary["scored"] += int(item.get("scored", 0))
            summary["elapsed_s"] += float(item.get("elapsed_s", 0.0))
            summary["gain"] += float(item.get("retained_proxy_gain", item.get("proxy_gain", 0.0)))
            summary["exhausted"] = bool(
                summary["exhausted"] or item.get("exact_quota_exhausted", False)
            )
            limit = item.get("exact_quota_limit")
            if limit is not None:
                limits.add(int(limit))
        scored = [int(item["scored"]) for item in by_benchmark.values()]
        elapsed = [float(item["elapsed_s"]) for item in by_benchmark.values()]
        configured_limit = next(iter(limits)) if len(limits) == 1 else None
        output.append(
            {
                "pass": name,
                "runs": len(items),
                "benchmarks": len(by_benchmark),
                "configured_limit": configured_limit,
                "minimum_scored": min(scored, default=0),
                "median_scored": _quantile(scored, 0.50),
                "p90_scored": _quantile(scored, 0.90),
                "maximum_scored": max(scored, default=0),
                "total_scored": sum(scored),
                "exhausted_benchmarks": sum(
                    bool(item["exhausted"]) for item in by_benchmark.values()
                ),
                "p90_elapsed_s": _quantile(elapsed, 0.90),
                "maximum_elapsed_s": max(elapsed, default=0.0),
                "proxy_gain": sum(float(item["gain"]) for item in by_benchmark.values()),
            }
        )
    return sorted(output, key=lambda row: (row["maximum_scored"], row["pass"]), reverse=True)


def print_quota_table(rows: list[dict]) -> None:
    header = (
        f"{'pass':34} {'runs':>4} {'benches':>7} {'limit':>8} {'min':>8} "
        f"{'median':>8} {'p90':>8} {'max':>8} {'exhaust':>8} {'p90 s':>8} {'max s':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        limit = row["configured_limit"]
        limit_text = str(limit) if limit is not None else "n/a"
        print(
            f"{row['pass'][:34]:34} {row['runs']:4d} {row['benchmarks']:7d} "
            f"{limit_text:>8} {row['minimum_scored']:8d} "
            f"{float(row['median_scored'] or 0.0):8.0f} "
            f"{float(row['p90_scored'] or 0.0):8.0f} {row['maximum_scored']:8d} "
            f"{row['exhausted_benchmarks']:8d} {float(row['p90_elapsed_s'] or 0.0):8.3f} "
            f"{row['maximum_elapsed_s']:8.3f}"
        )


def aggregate_stages(rows: list[dict], min_runs: int) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("stage", "unknown"))].append(row)
    output = []
    for name, items in sorted(grouped.items()):
        runs = len(items)
        if runs < min_runs:
            continue
        elapsed = [float(item.get("elapsed_s", 0.0)) for item in items]
        output.append(
            {
                "stage": name,
                "runs": runs,
                "benchmarks": len({str(item.get("benchmark", "")) for item in items}),
                "elapsed_s": sum(elapsed),
                "average_s": sum(elapsed) / runs,
                "maximum_s": max(elapsed, default=0.0),
            }
        )
    return sorted(output, key=lambda row: row["elapsed_s"], reverse=True)


def print_stage_table(rows: list[dict]) -> None:
    header = (
        f"{'stage':32} {'runs':>5} {'benches':>7} {'seconds':>10} {'average':>10} {'maximum':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['stage'][:32]:32} {row['runs']:5d} {row['benchmarks']:7d} "
            f"{row['elapsed_s']:10.2f} {row['average_s']:10.4f} {row['maximum_s']:10.4f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, default=[DEFAULT_PATH])
    parser.add_argument("--run-id")
    parser.add_argument("--revision")
    parser.add_argument("--worktree-fingerprint")
    parser.add_argument("--benchmark", action="append")
    parser.add_argument("--min-runs", type=int, default=1)
    parser.add_argument("--stages", action="store_true")
    parser.add_argument("--quotas", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.stages and args.quotas:
        parser.error("--stages and --quotas are mutually exclusive")
    if args.stages:
        rows = aggregate_stages(load_stage_rows(args.paths, args), max(1, args.min_runs))
    elif args.quotas:
        rows = aggregate_quotas(load_rows(args.paths, args), max(1, args.min_runs))
    else:
        rows = aggregate(load_rows(args.paths, args), max(1, args.min_runs))
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
    elif args.stages:
        print_stage_table(rows)
    elif args.quotas:
        print_quota_table(rows)
    else:
        print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
