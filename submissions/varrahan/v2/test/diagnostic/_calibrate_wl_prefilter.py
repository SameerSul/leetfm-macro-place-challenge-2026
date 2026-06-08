"""Calibrate the WL-delta prefilter thresholds for hard_2opt / soft_2opt.

Reads a traced run (with the prefilter disabled, so every candidate is scored and
its wl_delta + score_gain recorded) and, per operator, sweeps the threshold T,
reporting the tradeoff:

  - reject%  : fraction of candidates with wl_delta > T  (= score calls skipped)
  - lost%    : fraction of IMPROVING candidates (score_gain>0) with wl_delta > T
               (= real gains the prefilter would drop — want ~0)
  - max wl_d among improving candidates = the safe lower bound for T.

Pick the smallest T with lost% ≈ 0 (maximizes reject% without dropping gains).

    PYTHONPATH=submissions/varrahan/v2/src uv run python \
      submissions/varrahan/v2/test/diagnostic/_calibrate_wl_prefilter.py \
      submissions/varrahan/v2/ml_data/calib_hard2opt_ibm13.jsonl.gz
"""
import gzip
import json
import sys
from collections import defaultdict

THRESHOLDS = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
OPERATORS = ("hard_2opt", "soft_2opt", "soft_relocation")


def _iter(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def main(argv):
    paths = argv or ["submissions/varrahan/v2/ml_data/calib_hard2opt_ibm13.jsonl.gz"]
    # per op: list of (wl_delta, score_gain)
    rows = defaultdict(list)
    for p in paths:
        for r in _iter(p):
            if r.get("row_type") != "candidate":
                continue
            op = r.get("operator")
            if op not in OPERATORS:
                continue
            f = r.get("features", {})
            wl = f.get("wl_delta")
            if wl is None:
                continue
            rows[op].append((float(wl), float(r.get("score_gain", 0.0))))

    for op in OPERATORS:
        data = rows[op]
        if not data:
            print(f"== {op}: no rows with wl_delta ==\n")
            continue
        n = len(data)
        imp = [(w, g) for w, g in data if g > 0.0]
        nimp = len(imp)
        imp_wl = [w for w, _ in imp]
        max_imp_wl = max(imp_wl) if imp_wl else 0.0
        # p99 of improving wl_delta (robust safe bound vs a single outlier)
        p99 = sorted(imp_wl)[min(len(imp_wl) - 1, int(0.99 * len(imp_wl)))] if imp_wl else 0.0
        print(f"== {op} ==  candidates={n}  improving={nimp} ({100*nimp/n:.1f}%)")
        print(f"   improving wl_delta: max={max_imp_wl:.2e}  p99={p99:.2e}")
        print(f"   {'T':>8} {'reject%':>8} {'lost%':>7} {'lostN':>6}")
        for T in THRESHOLDS:
            rej = sum(1 for w, _ in data if w > T) / n
            lost = sum(1 for w, g in imp if w > T)
            lostpct = lost / max(nimp, 1)
            print(f"   {T:8.0e} {100*rej:7.1f}% {100*lostpct:6.1f}% {lost:6d}")
        print()


if __name__ == "__main__":
    main(sys.argv[1:])
