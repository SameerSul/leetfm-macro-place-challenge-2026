#!/usr/bin/env python3
"""Analyze hierarchy-contract headroom across attributable telemetry traces."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from placer.local_search.hierarchy_quality import (  # noqa: E402
    HIERARCHY_VECTOR_METRICS,
    hierarchy_coverage_scope,
    hierarchy_vector_contract,
    hierarchy_vector_limits,
    hierarchy_vector_margins,
)
from utils import constants as const  # noqa: E402

DEFAULT_PATH = Path("ml_data/plateau_telemetry/plateau_telemetry.jsonl")
DEFAULT_ABSOLUTE_SLACK = dict(const.HIER_VECTOR_CONTRACT_ABS_SLACK)
DEFAULT_RELATIVE_SLACK = float(const.HIER_VECTOR_CONTRACT_REL_SLACK)


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
    if getattr(args, "revision", None) and str(row.get("code_revision", "")) != args.revision:
        return False
    fingerprint = getattr(args, "worktree_fingerprint", None)
    if fingerprint and str(row.get("worktree_fingerprint", "legacy")) != fingerprint:
        return False
    benchmarks = getattr(args, "benchmark", None)
    if benchmarks and str(row.get("benchmark", "")) not in benchmarks:
        return False
    stage = getattr(args, "stage", "final")
    if stage != "all" and str(row.get("stage", "")) != stage:
        return False
    return True


def load_contract_rows(paths: list[Path], args) -> list[dict]:
    event = (
        "hierarchy_truth_audit"
        if getattr(args, "event", "contract") == "truth"
        else "hierarchy_contract_audit"
    )
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


def _coverage_scope(row: dict) -> str:
    if row.get("coverage_scope"):
        return str(row["coverage_scope"])
    return hierarchy_coverage_scope(row.get("coverage", row.get("vector", {})))


def _provenance(row: dict) -> str:
    value = row.get("hierarchy_provenance", row.get("provenance", "unknown"))
    if isinstance(value, dict):
        value = value.get("source", "unknown")
    value = str(value)
    if "path_tags" in value:
        return "explicit"
    if value.startswith("hierarchy_"):
        return "inferred"
    return value


def _component_relevant(row: dict, metric: str) -> bool:
    vector = row.get("vector", {})
    if metric in {"cluster_compactness", "worst_cluster_spread", "neighbor_impurity"}:
        return float(vector.get("clustered_hard_count", 0.0)) > 0.0
    if metric == "edge_stretch":
        return float(vector.get("edge_count", 0.0)) > 0.0
    if metric == "owned_soft_distance":
        return float(vector.get("owned_soft_count", 0.0)) > 0.0
    if metric == "bridge_soft_distance":
        return float(vector.get("bridge_soft_count", 0.0)) > 0.0
    return True


def _limits_for_row(
    row: dict,
    *,
    absolute_slack: dict[str, float] | None,
    relative_slack: float | None,
) -> tuple[dict[str, float], dict[str, float], bool, dict[str, float]]:
    vector = row.get("vector", {})
    reference = row.get("reference_vector", {})
    if absolute_slack is None and relative_slack is None:
        limits = {str(key): float(value) for key, value in row.get("limits", {}).items()}
    else:
        limits = hierarchy_vector_limits(
            reference,
            absolute_slack or DEFAULT_ABSOLUTE_SLACK,
            DEFAULT_RELATIVE_SLACK if relative_slack is None else float(relative_slack),
        )
    passed, violations = hierarchy_vector_contract(vector, limits)
    margins = hierarchy_vector_margins(vector, limits)
    return limits, margins, bool(passed), violations


def aggregate_contract(
    rows: list[dict],
    *,
    absolute_slack: dict[str, float] | None = None,
    relative_slack: float | None = None,
) -> dict:
    active_absolute = absolute_slack or DEFAULT_ABSOLUTE_SLACK
    active_relative = DEFAULT_RELATIVE_SLACK if relative_slack is None else float(relative_slack)
    evaluated = []
    for source in rows:
        row = dict(source)
        limits, margins, passed, violations = _limits_for_row(
            row,
            absolute_slack=absolute_slack,
            relative_slack=relative_slack,
        )
        row["_limits"] = limits
        row["_margins"] = margins
        row["_passed"] = passed
        row["_violations"] = violations
        evaluated.append(row)

    component_rows = {}
    for metric in HIERARCHY_VECTOR_METRICS:
        relevant = [row for row in evaluated if _component_relevant(row, metric)]
        margins = [float(row["_margins"][metric]) for row in relevant]
        utilization = []
        required_absolute = 0.0
        tightest = None
        for row in relevant:
            value = float(row.get("vector", {}).get(metric, 0.0))
            reference = float(row.get("reference_vector", {}).get(metric, 0.0))
            limit = float(row["_limits"][metric])
            allowance = max(limit - reference, 1.0e-15)
            used = value - reference
            utilization.append(used / allowance)
            if used > active_relative * abs(reference) + 1.0e-12:
                required_absolute = max(required_absolute, used)
            candidate = (float(row["_margins"][metric]), str(row.get("benchmark", "")))
            if tightest is None or candidate < tightest:
                tightest = candidate
        configured_absolute = float(active_absolute[metric])
        component_rows[metric] = {
            "relevant_rows": len(relevant),
            "minimum_margin": min(margins) if margins else None,
            "p10_margin": _quantile(margins, 0.10),
            "median_margin": _quantile(margins, 0.50),
            "maximum_utilization": max(utilization) if utilization else None,
            "p90_utilization": _quantile(utilization, 0.90),
            "near_limit_rows": sum(value >= 0.80 for value in utilization),
            "violation_rows": sum(value > 1.0 + 1.0e-12 for value in utilization),
            "required_absolute_at_relative": float(required_absolute),
            "configured_absolute": configured_absolute,
            "absolute_reserve": configured_absolute - required_absolute,
            "tightest_benchmark": tightest[1] if tightest else None,
        }

    violation_counts = Counter()
    selected_violation_counts = Counter()
    for row in evaluated:
        violation_counts.update(str(key) for key in row["_violations"])
        if row.get("selected", False):
            selected_violation_counts.update(str(key) for key in row["_violations"])
    selected_rows = [row for row in evaluated if row.get("selected", False)]
    selected_failures = [
        {
            "benchmark": str(row.get("benchmark", "")),
            "candidate": str(row.get("candidate", row.get("stage", ""))),
            "violations": sorted(str(key) for key in row["_violations"]),
        }
        for row in selected_rows
        if not row["_passed"]
    ]
    selected_failures.sort(key=lambda row: (row["benchmark"], row["candidate"]))
    return {
        "rows": len(evaluated),
        "benchmarks": sorted({str(row.get("benchmark", "")) for row in evaluated}),
        "passed": sum(bool(row["_passed"]) for row in evaluated),
        "failed": sum(not bool(row["_passed"]) for row in evaluated),
        "selected": len(selected_rows),
        "selected_passed": sum(bool(row["_passed"]) for row in selected_rows),
        "selected_failed": sum(not bool(row["_passed"]) for row in selected_rows),
        "selected_failures": selected_failures,
        "coverage": dict(sorted(Counter(_coverage_scope(row) for row in evaluated).items())),
        "provenance": dict(sorted(Counter(_provenance(row) for row in evaluated).items())),
        "violation_components": dict(sorted(violation_counts.items())),
        "selected_violation_components": dict(sorted(selected_violation_counts.items())),
        "relative_slack": active_relative,
        "absolute_slack": {key: float(active_absolute[key]) for key in HIERARCHY_VECTOR_METRICS},
        "components": component_rows,
    }


def _format_number(value: float | None, width: int = 9) -> str:
    return f"{float(value):{width}.5f}" if value is not None else f"{'n/a':>{width}}"


def print_report(summary: dict) -> None:
    print(
        f"rows={summary['rows']} designs={len(summary['benchmarks'])} "
        f"passed={summary['passed']} failed={summary['failed']} selected={summary['selected']} "
        f"selected_passed={summary['selected_passed']} "
        f"selected_failed={summary['selected_failed']}"
    )
    print(f"coverage={summary['coverage']} provenance={summary['provenance']}")
    print(f"relative_slack={summary['relative_slack']:.5f}")
    header = (
        f"{'component':24} {'rows':>5} {'min margin':>10} {'p10':>9} "
        f"{'max use':>9} {'p90 use':>9} {'near':>5} {'fail':>5} "
        f"{'req abs':>9} {'cfg abs':>9} {'reserve':>9} tightest"
    )
    print(header)
    print("-" * len(header))
    for metric in HIERARCHY_VECTOR_METRICS:
        row = summary["components"][metric]
        print(
            f"{metric:24} {row['relevant_rows']:5d} "
            f"{_format_number(row['minimum_margin'], 10)} "
            f"{_format_number(row['p10_margin'])} "
            f"{_format_number(row['maximum_utilization'])} "
            f"{_format_number(row['p90_utilization'])} "
            f"{row['near_limit_rows']:5d} {row['violation_rows']:5d} "
            f"{row['required_absolute_at_relative']:9.5f} "
            f"{row['configured_absolute']:9.5f} {row['absolute_reserve']:9.5f} "
            f"{row['tightest_benchmark'] or 'n/a'}"
        )
    if summary["violation_components"]:
        print(f"violations={summary['violation_components']}")
    if summary["selected_violation_components"]:
        print(f"selected_violations={summary['selected_violation_components']}")
        details = ", ".join(
            f"{row['benchmark']}:{row['candidate']}[{','.join(row['violations'])}]"
            for row in summary["selected_failures"]
        )
        print(f"selected_failures={details}")


def _parse_absolute_slack(values: list[str] | None) -> dict[str, float] | None:
    if not values:
        return None
    result = dict(DEFAULT_ABSOLUTE_SLACK)
    for value in values:
        name, separator, raw = value.partition("=")
        if not separator or name not in HIERARCHY_VECTOR_METRICS:
            raise ValueError(f"invalid --absolute-slack {value!r}; expected metric=value")
        result[name] = max(0.0, float(raw))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, default=[DEFAULT_PATH])
    parser.add_argument("--event", choices=("contract", "truth"), default="contract")
    parser.add_argument("--stage", choices=("final", "seed_candidate", "all"), default="final")
    parser.add_argument("--run-id")
    parser.add_argument("--revision")
    parser.add_argument("--worktree-fingerprint")
    parser.add_argument("--benchmark", action="append")
    parser.add_argument("--relative-slack", type=float)
    parser.add_argument("--absolute-slack", action="append", metavar="METRIC=VALUE")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        absolute_slack = _parse_absolute_slack(args.absolute_slack)
    except ValueError as exc:
        parser.error(str(exc))
    summary = aggregate_contract(
        load_contract_rows(args.paths, args),
        absolute_slack=absolute_slack,
        relative_slack=args.relative_slack,
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_report(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
