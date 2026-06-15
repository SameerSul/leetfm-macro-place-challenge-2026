"""Per-benchmark filter recall + group-richness diagnostic.

Runs the deployed hard_relocation ranker over wide32_nofilter traces (which score
all 32 candidates and label every one) and reports, per benchmark:

- best_recall@K: fraction of groups whose true-best-gain candidate is in the
  model's top-K (i.e. pruning to K would still score the best move).
- improving group richness: how many *improving* candidates exist per group.
  If a benchmark has many improving candidates per group, pruning removes
  alternative accepts and can divert the trajectory even when recall is high.

This separates "the filter drops the best move" (a recall problem, fixable by
retraining) from "the filter removes useful tail diversity" (a trajectory
problem, where a single fixed K cannot win every benchmark).

Usage:
    PYTHONPATH=src \
    uv run python test/diagnostic/_filter_recall_by_benchmark.py
"""

from __future__ import annotations

import collections
import glob
import json
import gzip
import statistics
from pathlib import Path

from placer.ml.dataset import flatten_candidate
from placer.ml.modeling import ModelBank

ROOT = Path(__file__).resolve().parents[2]
TRACE_GLOB = str(ROOT / "ml_data" / "wide32_nofilter" / "*.jsonl.gz")
MANIFEST = ROOT / "ml_data" / "models" / "clean-wide32-holdout-ibm13-001" / "manifest.json"
OPERATOR = "hard_relocation"
TOP_KS = (5, 10, 16, 20, 24)


def _iter_rows(path: str):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> int:
    bank = ModelBank.from_manifest(MANIFEST)
    ranker = bank.get(OPERATOR)
    if ranker is None:
        raise SystemExit(f"no model for {OPERATOR} in {MANIFEST}")

    # group_id -> {benchmark, candidates:[flattened rows]}
    groups: dict[str, dict] = {}
    for path in glob.glob(TRACE_GLOB):
        for row in _iter_rows(path):
            if row.get("row_type") != "candidate" or row.get("operator") != OPERATOR:
                continue
            gid = row.get("group_id")
            g = groups.get(gid)
            if g is None:
                g = {"benchmark": row.get("benchmark"), "cands": []}
                groups[gid] = g
            g["cands"].append(flatten_candidate(row))

    # Per-benchmark accumulators.
    per = collections.defaultdict(
        lambda: {
            "groups": 0,
            "improving_groups": 0,
            "improving_counts": [],  # improving candidates per (improving) group
            "recall_best": {k: 0 for k in TOP_KS},
            "recall_best_improving": {k: 0 for k in TOP_KS},
        }
    )

    for g in groups.values():
        cands = g["cands"]
        if len(cands) < 2:
            continue
        bench = g["benchmark"]
        acc = per[bench]
        gains = [float(c.get("score_gain", 0.0)) for c in cands]
        best_gain = max(gains)
        improving = best_gain > 0.0
        n_improving = sum(1 for x in gains if x > 0.0)
        best_idx = {i for i, x in enumerate(gains) if x == best_gain}

        scores = ranker.scores(cands)
        order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))

        acc["groups"] += 1
        if improving:
            acc["improving_groups"] += 1
            acc["improving_counts"].append(n_improving)
        for k in TOP_KS:
            chosen = set(order[:k])
            if best_idx & chosen:
                acc["recall_best"][k] += 1
                if improving:
                    acc["recall_best_improving"][k] += 1

    print(f"operator={OPERATOR}  manifest={MANIFEST.name}")
    print(f"trace={TRACE_GLOB}")
    print()
    for bench in sorted(per):
        a = per[bench]
        ng = max(a["groups"], 1)
        nig = max(a["improving_groups"], 1)
        rich = a["improving_counts"]
        rich_mean = statistics.mean(rich) if rich else 0.0
        rich_med = statistics.median(rich) if rich else 0.0
        frac_rich = (sum(1 for x in rich if x > 16) / len(rich)) if rich else 0.0
        print(f"== {bench} ==  groups={a['groups']}  improving_groups={a['improving_groups']}")
        print(
            "   improving-per-group: mean=%.2f median=%.1f  frac(>16 improving)=%.3f"
            % (rich_mean, rich_med, frac_rich)
        )
        rb = "  ".join(f"@{k}={a['recall_best'][k] / ng:.4f}" for k in TOP_KS)
        rbi = "  ".join(f"@{k}={a['recall_best_improving'][k] / nig:.4f}" for k in TOP_KS)
        print(f"   best_recall          {rb}")
        print(f"   best_recall(improv)  {rbi}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
