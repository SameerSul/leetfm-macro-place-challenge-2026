#!/usr/bin/env python3
"""Aggregate hierarchy pass yield from plateau telemetry JSONL files."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

DEFAULT_PATH = Path("ml_data/beyondppa_gnn/plateau/plateau_telemetry.jsonl")


def _matches(row: dict, args) -> bool:
    if args.run_id and str(row.get("run_id", "legacy")) != args.run_id:
        return False
    if args.revision and str(row.get("code_revision", "unknown")) != args.revision:
        return False
    if args.benchmark and str(row.get("benchmark", "")) not in args.benchmark:
        return False
    return row.get("event") == "hier_plateau_telemetry" and row.get("plateau_pass")


def load_rows(paths: list[Path], args) -> list[dict]:
    rows = []
    for path in paths:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
                if _matches(row, args):
                    rows.append(row)
    return rows


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
        gain = sum(float(item.get("proxy_gain", 0.0)) for item in items)
        accepts = sum(int(item.get("accepts", 0)) for item in items)
        candidates = sum(int(item.get("candidates", 0)) for item in items)
        zero_gain = sum(float(item.get("proxy_gain", 0.0)) < 0.00005 for item in items)
        zero_accept = sum(int(item.get("accepts", 0)) == 0 for item in items)
        gain_per_s = gain / max(elapsed, 1.0e-12)
        skip_candidate = elapsed >= 0.5 and zero_gain / runs >= 0.8 and gain_per_s < 0.00001
        output.append(
            {
                "pass": name,
                "runs": runs,
                "benchmarks": len({str(item.get("benchmark", "")) for item in items}),
                "candidates": candidates,
                "accepts": accepts,
                "proxy_gain": gain,
                "elapsed_s": elapsed,
                "gain_per_s": gain_per_s,
                "zero_gain_fraction": zero_gain / runs,
                "zero_accept_fraction": zero_accept / runs,
                "recommendation": "skip_candidate" if skip_candidate else "retain_or_measure",
            }
        )
    return output


def print_table(rows: list[dict]) -> None:
    header = (
        f"{'pass':34} {'runs':>4} {'benches':>7} {'gain':>10} {'seconds':>9} "
        f"{'gain/s':>10} {'zero%':>7} recommendation"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['pass'][:34]:34} {row['runs']:4d} {row['benchmarks']:7d} "
            f"{row['proxy_gain']:10.6f} {row['elapsed_s']:9.2f} {row['gain_per_s']:10.7f} "
            f"{100.0 * row['zero_gain_fraction']:6.1f}% {row['recommendation']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, default=[DEFAULT_PATH])
    parser.add_argument("--run-id")
    parser.add_argument("--revision")
    parser.add_argument("--benchmark", action="append")
    parser.add_argument("--min-runs", type=int, default=1)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    rows = aggregate(load_rows(args.paths, args), max(1, args.min_runs))
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
