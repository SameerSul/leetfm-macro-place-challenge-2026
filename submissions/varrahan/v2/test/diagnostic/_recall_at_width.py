"""Recall@K-vs-pool-width feasibility study for a GNN routing-fill prefilter.

Question this answers (the kill-switch before building any GNN): as the candidate
pool widens, (1) does the *legal* pool on IBM even get wide, and (2) does a cheap
surrogate keep the true-best move inside its top-K? If recall collapses with
width, a 1000-wide GNN prefilter is dead; if it holds, the direction is alive.

Inputs are wide-pool traces produced by re-running the placer with
ML_HARD_RELOCATION_N_TARGETS={64,128,256} and the tracer on, filter OFF (so every
legal candidate is exact-scored and labelled with score_gain). Filenames are
expected to encode benchmark + width, e.g. ibm13_w128.jsonl.gz.

The surrogate is the deployed XGBoost ranker (trained on 32-wide pools). Scoring a
wider pool with it is a width-extrapolation test on purpose. ibm13 is the honest
holdout for clean-wide32-holdout-ibm13; other benchmarks were in its train set, so
their recall is optimistic — read ibm13 as the generalization signal.

Usage:
    PYTHONPATH=submissions/varrahan/v2/src \
    uv run python submissions/varrahan/v2/test/diagnostic/_recall_at_width.py \
      submissions/varrahan/v2/ml_data/recall_study/*.jsonl.gz
"""

from __future__ import annotations

import gzip
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from placer.ml.dataset import flatten_candidate
from placer.ml.modeling import ModelBank

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "ml_data" / "models" / "clean-wide32-holdout-ibm13-001" / "manifest.json"
OPERATOR = "hard_relocation"
KS = (1, 3, 5, 10, 20, 50)
WIDTH_THRESHOLDS = (16, 32, 64, 128)


def _iter_rows(path: str):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _pct(values, q):
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))]


def analyze_file(path: str, ranker) -> None:
    m = re.search(r"([A-Za-z0-9_]+?)_w(\d+)", Path(path).name)
    bench = m.group(1) if m else Path(path).stem
    width_req = m.group(2) if m else "?"

    # group_id -> list of flattened candidates
    groups: dict[str, list] = defaultdict(list)
    for row in _iter_rows(path):
        if row.get("row_type") == "candidate" and row.get("operator") == OPERATOR:
            groups[row.get("group_id")].append(flatten_candidate(row))

    all_widths, imp_widths = [], []
    improving = 0
    recall = {k: 0 for k in KS}
    regret = {k: 0.0 for k in KS}
    for cands in groups.values():
        w = len(cands)
        all_widths.append(w)
        if w < 2:
            continue
        gains = [float(c.get("score_gain", 0.0)) for c in cands]
        best = max(gains)
        if best <= 0.0:
            continue  # non-improving group: ranking is moot, the move is rejected
        improving += 1
        imp_widths.append(w)
        best_idx = {i for i, g in enumerate(gains) if g == best}
        scores = ranker.scores(cands)
        order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
        for k in KS:
            chosen = order[:k]
            if best_idx & set(chosen):
                recall[k] += 1
            regret[k] += max(0.0, best - max((gains[i] for i in chosen), default=0.0))

    ng = len(all_widths)
    nimp = max(improving, 1)
    print(f"== {bench}  N_TARGETS={width_req}  ({Path(path).name}) ==")
    print(f"   groups={ng}  improving={improving} ({100*improving/max(ng,1):.1f}%)")
    if all_widths:
        print(
            "   legal-pool width (ALL groups):     median=%d  p90=%d  max=%d"
            % (statistics.median(all_widths), _pct(all_widths, 0.9), max(all_widths))
        )
    if imp_widths:
        print(
            "   legal-pool width (IMPROVING grps): median=%d  p90=%d  max=%d"
            % (statistics.median(imp_widths), _pct(imp_widths, 0.9), max(imp_widths))
        )
        for t in WIDTH_THRESHOLDS:
            frac = sum(1 for w in imp_widths if w > t) / len(imp_widths)
            print(f"     frac improving groups with width > {t:>3}: {frac:.3f}")
    rec = "  ".join(f"@{k}={recall[k]/nimp:.3f}" for k in KS)
    reg10 = regret[10] / nimp
    print(f"   improving_recall   {rec}")
    print(f"   mean_regret@10={reg10:.2e}   mean_regret@5={regret[5]/nimp:.2e}")
    print()


def main(argv) -> int:
    # Any arg ending in manifest.json overrides the surrogate; the rest are traces.
    manifest = MANIFEST
    traces = []
    for a in argv:
        if a.endswith("manifest.json"):
            manifest = Path(a)
        else:
            traces.append(a)
    paths = traces or [str(p) for p in (ROOT / "ml_data" / "recall_study").glob("*.jsonl.gz")]
    if not paths:
        raise SystemExit("no trace files found")
    ranker = ModelBank.from_manifest(manifest).get(OPERATOR)
    if ranker is None:
        raise SystemExit(f"no {OPERATOR} model in {manifest}")
    print(f"surrogate = {manifest.parent.name}")
    print(f"ibm13 = honest holdout; other benchmarks may be in train (optimistic)\n")
    for path in sorted(paths):
        analyze_file(path, ranker)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
