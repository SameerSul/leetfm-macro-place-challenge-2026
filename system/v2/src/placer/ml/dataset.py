"""Utilities for validating and flattening candidate-trace JSONL files."""

import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable


IDENTITY_COLUMNS = (
    "run_id",
    "benchmark",
    "benchmark_index",
    "seed",
    "config_hash",
    "operator",
    "field",
    "group_id",
    "candidate_rank",
    "group_size",
    "candidate_source",
)

BENCHMARK_COLUMNS = (
    "effective_budget_s",
    "num_hard",
    "num_soft",
    "num_nets",
    "grid_rows",
    "grid_cols",
    "canvas_width",
    "canvas_height",
)

LABEL_COLUMNS = ("state_score", "trial_score", "score_gain", "improves")


def iter_trace_rows(paths: Iterable[str | Path]):
    """Yield decoded rows from one or more trace files."""
    for path in paths:
        path = Path(path)
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSONL row") from exc


def flatten_candidate(row: dict) -> dict:
    """Flatten one candidate row into a tabular feature/label dictionary."""
    if row.get("row_type") != "candidate":
        raise ValueError("flatten_candidate requires row_type='candidate'")
    flat = {
        key: row.get(key)
        for key in IDENTITY_COLUMNS + BENCHMARK_COLUMNS + LABEL_COLUMNS
    }
    flat.update({f"context.{key}": value for key, value in row.get("context", {}).items()})
    flat.update({f"feature.{key}": value for key, value in row.get("features", {}).items()})
    return flat


def load_candidates(paths: Iterable[str | Path], operator: str | None = None) -> list[dict]:
    """Load and flatten candidate rows, optionally filtering to one operator."""
    rows = []
    for row in iter_trace_rows(paths):
        if row.get("row_type") != "candidate":
            continue
        if operator is not None and row.get("operator") != operator:
            continue
        rows.append(flatten_candidate(row))
    return rows


def add_group_relevance(rows: list[dict], max_relevance: int = 31) -> list[dict]:
    """Add ordinal LambdaMART relevance derived only within each decision group.

    The worst candidate receives 0 and the best receives `max_relevance`.
    Ties receive the same relevance. Singleton groups receive 0 because they
    contain no ranking signal.
    """
    groups = defaultdict(list)
    for row in rows:
        groups[(row["run_id"], row["group_id"])].append(row)

    for group in groups.values():
        unique_gains = sorted({float(row["score_gain"]) for row in group})
        if len(unique_gains) <= 1:
            for row in group:
                row["relevance"] = 0
            continue
        gain_rank = {gain: rank for rank, gain in enumerate(unique_gains)}
        denominator = len(unique_gains) - 1
        for row in group:
            rank = gain_rank[float(row["score_gain"])]
            row["relevance"] = round(max_relevance * rank / denominator)
    return rows


def trace_summary(paths: Iterable[str | Path]) -> dict:
    """Summarize row/operator counts and candidate-group completeness."""
    row_types = defaultdict(int)
    operators = defaultdict(int)
    groups = defaultdict(int)
    for row in iter_trace_rows(paths):
        row_type = row.get("row_type", "unknown")
        row_types[row_type] += 1
        if row_type == "candidate":
            operators[row.get("operator", "unknown")] += 1
            groups[(row.get("run_id"), row.get("group_id"))] += 1
    return {
        "row_types": dict(row_types),
        "operators": dict(operators),
        "candidate_groups": len(groups),
        "singleton_groups": sum(count == 1 for count in groups.values()),
    }
