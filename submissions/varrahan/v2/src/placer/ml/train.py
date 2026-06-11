"""Offline training and evaluation for candidate-ranker models.

This module is intentionally not imported by the placer runtime. Use it as:

    python -m placer.ml.train TRACE.jsonl.gz ... --output-dir /tmp/ml_models
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from placer.ml.dataset import add_group_relevance, flatten_candidate, iter_trace_rows
from placer.ml.modeling import (
    OPERATORS,
    ModelSpec,
    build_training_matrix,
)


def load_operator_rows(
    paths: Iterable[str | Path],
    operators: Sequence[str],
    *,
    max_rows_per_operator: int = 0,
    always_include_benchmarks: Sequence[str] = (),
) -> dict[str, list[dict]]:
    """Load flattened candidate rows from traces, capped per operator if set.

    Rows from ``always_include_benchmarks`` do not count against the cap. This is
    useful for exact held-out test benchmarks: a cap should bound training volume,
    not accidentally discard the test split because it appears late in the files.
    """
    wanted = set(operators)
    always_include = {str(name).lower() for name in always_include_benchmarks}
    rows = {operator: [] for operator in operators}
    capped_counts = {operator: 0 for operator in operators}
    for row in iter_trace_rows(paths):
        if row.get("row_type") != "candidate":
            continue
        operator = row.get("operator")
        if operator not in wanted:
            continue
        benchmark = str(row.get("benchmark") or "").lower()
        is_always_included = benchmark in always_include
        if (
            max_rows_per_operator
            and not is_always_included
            and capped_counts[operator] >= max_rows_per_operator
        ):
            if not always_include and all(
                capped_counts[op] >= max_rows_per_operator for op in rows
            ):
                break
            continue
        rows[operator].append(flatten_candidate(row))
        if not is_always_included:
            capped_counts[operator] += 1
    return rows


def split_rows(
    rows: Sequence[Mapping],
    *,
    seed: int = 0,
    valid_fraction: float = 0.15,
    test_benchmark_prefix: str | None = "ng",
    test_benchmarks: Sequence[str] = (),
) -> dict[str, list[Mapping]]:
    """Split by benchmark/run, never by row."""
    test_prefix = (test_benchmark_prefix or "").lower()
    exact_test_benchmarks = {str(name).lower() for name in test_benchmarks}
    key_to_rows = defaultdict(list)
    for row in rows:
        key_to_rows[(row.get("benchmark"), row.get("run_id"))].append(row)

    train_keys = []
    test_keys = []
    for key in key_to_rows:
        benchmark = str(key[0] or "").lower()
        if benchmark in exact_test_benchmarks or (
            test_prefix and benchmark.startswith(test_prefix)
        ):
            test_keys.append(key)
        else:
            train_keys.append(key)

    rng = random.Random(seed)
    rng.shuffle(train_keys)
    n_valid = int(round(len(train_keys) * valid_fraction))
    if len(train_keys) > 1:
        n_valid = max(1, min(n_valid, len(train_keys) - 1))
    valid_keys = set(train_keys[:n_valid])
    train_keys = set(train_keys[n_valid:])
    test_keys = set(test_keys)

    splits = {"train": [], "valid": [], "test": []}
    for key, group_rows in key_to_rows.items():
        if key in test_keys:
            splits["test"].extend(group_rows)
        elif key in valid_keys:
            splits["valid"].extend(group_rows)
        elif key in train_keys:
            splits["train"].extend(group_rows)
        else:
            splits["train"].extend(group_rows)
    return splits


def ranking_metrics(
    rows: Sequence[Mapping],
    scores: Sequence[float],
    *,
    top_ks: Sequence[int] = (1, 3, 5, 10, 16),
) -> dict:
    """Compute group-level ranking metrics from candidate rows and predictions."""
    by_group = defaultdict(list)
    for row, score in zip(rows, scores):
        by_group[(row.get("run_id"), row.get("group_id"))].append((row, float(score)))

    metrics = {
        "groups": len(by_group),
        "rows": len(rows),
        "improving_groups": 0,
    }
    recall_improve = {int(k): 0 for k in top_ks}
    recall_best = {int(k): 0 for k in top_ks}
    regret_sum = {int(k): 0.0 for k in top_ks}

    for group in by_group.values():
        gains = [float(row["score_gain"]) for row, _ in group]
        best_gain = max(gains)
        has_improving = best_gain > 0.0
        if has_improving:
            metrics["improving_groups"] += 1
        best_indices = {i for i, gain in enumerate(gains) if gain == best_gain}
        pred_order = sorted(range(len(group)), key=lambda i: (-group[i][1], i))

        for k in top_ks:
            kk = min(int(k), len(pred_order))
            chosen = pred_order[:kk]
            chosen_best = max(gains[i] for i in chosen) if chosen else float("-inf")
            regret_sum[int(k)] += max(0.0, best_gain - chosen_best)
            if has_improving and any(gains[i] > 0.0 for i in chosen):
                recall_improve[int(k)] += 1
            if any(i in best_indices for i in chosen):
                recall_best[int(k)] += 1

    denom = max(metrics["groups"], 1)
    improve_denom = max(metrics["improving_groups"], 1)
    for k in top_ks:
        kk = int(k)
        metrics[f"best_recall@{kk}"] = recall_best[kk] / denom
        metrics[f"improving_recall@{kk}"] = recall_improve[kk] / improve_denom
        metrics[f"mean_regret@{kk}"] = regret_sum[kk] / denom
    return metrics


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    if not y_true:
        return float("nan")
    err = [(float(a) - float(b)) ** 2 for a, b in zip(y_true, y_pred)]
    return math.sqrt(sum(err) / len(err))


def _to_dmatrix(matrix, *, ranking: bool = False):
    import xgboost as xgb

    dmat = xgb.DMatrix(
        np.asarray(matrix.X, dtype=np.float32),
        label=np.asarray(matrix.y, dtype=np.float32),
        feature_names=list(matrix.feature_names),
    )
    if ranking:
        dmat.set_group(matrix.group_sizes)
    return dmat


def train_xgboost(train_matrix, valid_matrix, *, objective: str, rounds: int, seed: int):
    """Train one XGBoost booster."""
    import xgboost as xgb

    ranking = objective == "ranker"
    params = {
        "tree_method": "hist",
        "max_depth": 6,
        "eta": 0.08,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "seed": seed,
        "nthread": 0,
        "objective": "rank:ndcg" if ranking else "reg:squarederror",
        "eval_metric": "ndcg@10" if ranking else "rmse",
    }
    train_dmat = _to_dmatrix(train_matrix, ranking=ranking)
    evals = [(train_dmat, "train")]
    if valid_matrix.X:
        evals.append((_to_dmatrix(valid_matrix, ranking=ranking), "valid"))
    return xgb.train(params, train_dmat, num_boost_round=rounds, evals=evals, verbose_eval=False)


def _predict(booster, matrix) -> list[float]:
    import xgboost as xgb

    dmat = xgb.DMatrix(
        np.asarray(matrix.X, dtype=np.float32),
        feature_names=list(matrix.feature_names),
    )
    return [float(x) for x in booster.predict(dmat)]


def train_operator(
    rows: Sequence[Mapping],
    operator: str,
    *,
    objective: str,
    output_dir: Path,
    top_ks: Sequence[int],
    seed: int,
    rounds: int,
    valid_fraction: float,
    test_benchmark_prefix: str | None,
    test_benchmarks: Sequence[str],
) -> dict:
    if not rows:
        return {"operator": operator, "status": "skipped", "reason": "no rows"}
    split = split_rows(
        rows,
        seed=seed,
        valid_fraction=valid_fraction,
        test_benchmark_prefix=test_benchmark_prefix,
        test_benchmarks=test_benchmarks,
    )
    if objective == "ranker":
        for name in ("train", "valid", "test"):
            split[name] = add_group_relevance(list(split[name]))
        label = "relevance"
    else:
        label = "score_gain"

    train_matrix = build_training_matrix(split["train"], operator, label=label)
    valid_matrix = build_training_matrix(split["valid"], operator, label=label)
    test_matrix = build_training_matrix(split["test"], operator, label=label)
    if not train_matrix.X:
        return {"operator": operator, "status": "skipped", "reason": "empty train split"}

    booster = train_xgboost(
        train_matrix,
        valid_matrix,
        objective=objective,
        rounds=rounds,
        seed=seed,
    )

    model_name = f"{operator}.{objective}.xgb.json"
    model_path = output_dir / model_name
    booster.save_model(model_path)

    metrics = {
        "operator": operator,
        "objective": objective,
        "status": "trained",
        "rows": {name: len(split[name]) for name in ("train", "valid", "test")},
        "groups": {
            "train": len(train_matrix.group_sizes),
            "valid": len(valid_matrix.group_sizes),
            "test": len(test_matrix.group_sizes),
        },
        "model_path": model_name,
        "features": list(train_matrix.feature_names),
    }
    for split_name, matrix in (("train", train_matrix), ("valid", valid_matrix), ("test", test_matrix)):
        if not matrix.X:
            continue
        preds = _predict(booster, matrix)
        metrics[split_name] = {
            "rmse_score_gain": rmse(
                [float(row["score_gain"]) for row in matrix.rows],
                preds,
            ),
            **ranking_metrics(matrix.rows, preds, top_ks=top_ks),
        }

    spec = ModelSpec(
        operator=operator,
        backend="xgboost_json",
        feature_names=train_matrix.feature_names,
        model_path=model_name,
        top_k_default=max(top_ks) if top_ks else None,
        keep_heuristic_first=2,
        random_exploration_fraction=0.05,
    )
    return {"spec": spec, "metrics": metrics}


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split(",") if part.strip())


def _parse_csv_strings(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", help="Trace JSONL/JSONL.GZ files")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--operators", default=",".join(OPERATORS))
    parser.add_argument("--objective", choices=("ranker", "regressor"), default="ranker")
    parser.add_argument("--rounds", type=int, default=80)
    parser.add_argument("--max-rows-per-operator", type=int, default=0)
    parser.add_argument("--valid-fraction", type=float, default=0.15)
    parser.add_argument("--test-benchmark-prefix", default="ng")
    parser.add_argument(
        "--test-benchmarks",
        default="",
        help="Comma-separated exact benchmark names to hold out as test data.",
    )
    parser.add_argument("--top-k", default="1,3,5,10,16")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    operators = tuple(op for op in args.operators.split(",") if op)
    unknown = [op for op in operators if op not in OPERATORS]
    if unknown:
        raise SystemExit(f"unknown operators: {unknown}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    top_ks = _parse_csv_ints(args.top_k)
    test_benchmarks = _parse_csv_strings(args.test_benchmarks)

    rows_by_operator = load_operator_rows(
        args.traces,
        operators,
        max_rows_per_operator=args.max_rows_per_operator,
        always_include_benchmarks=test_benchmarks,
    )

    manifest = {"models": []}
    metrics = {
        "objective": args.objective,
        "test_benchmark_prefix": args.test_benchmark_prefix,
        "test_benchmarks": list(test_benchmarks),
        "operators": {},
    }
    for operator in operators:
        result = train_operator(
            rows_by_operator[operator],
            operator,
            objective=args.objective,
            output_dir=output_dir,
            top_ks=top_ks,
            seed=args.seed,
            rounds=args.rounds,
            valid_fraction=args.valid_fraction,
            test_benchmark_prefix=args.test_benchmark_prefix,
            test_benchmarks=test_benchmarks,
        )
        if result.get("spec") is not None:
            spec = result["spec"]
            manifest["models"].append(
                {
                    "operator": spec.operator,
                    "backend": spec.backend,
                    "feature_names": list(spec.feature_names),
                    "model_path": spec.model_path,
                    "top_k_default": spec.top_k_default,
                    "keep_heuristic_first": spec.keep_heuristic_first,
                    "random_exploration_fraction": spec.random_exploration_fraction,
                }
            )
            metrics["operators"][operator] = result["metrics"]
        else:
            metrics["operators"][operator] = result

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
