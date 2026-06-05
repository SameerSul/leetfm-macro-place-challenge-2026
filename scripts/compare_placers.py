#!/usr/bin/env python3
"""
Compare multiple placement submissions side-by-side on IBM ICCAD04 benchmarks.

Runs each placer on the selected benchmarks and prints a ranked comparison
table showing proxy cost, improvement over SA/RePlAce baselines, and timing.
Results are saved to results/comparison_<timestamp>.json for tracking.

Usage:
    # Compare sameer_v1 vs will_seed on ibm01
    uv run python scripts/compare_placers.py submissions/sameer_v1/placer.py submissions/will_seed/placer.py

    # Compare on all 17 benchmarks
    uv run python scripts/compare_placers.py --all submissions/sameer_v1/placer.py submissions/will_seed/placer.py

    # Compare on specific benchmarks
    uv run python scripts/compare_placers.py -b ibm01 ibm02 ibm03 submissions/sameer_v1/placer.py submissions/will_seed/placer.py

    # Quick mode: one benchmark, suppress verbose output
    uv run python scripts/compare_placers.py -b ibm01 --quiet submissions/sameer_v1/placer.py
"""

import argparse
import importlib.util
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Re-use harness primitives ────────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from macro_place.evaluate import (
    IBM_BENCHMARKS,
    SA_BASELINES,
    REPLACE_BASELINES,
    evaluate_benchmark,
)

# Current known leaderboard (update as new submissions arrive)
KNOWN_LEADERBOARD = {
    "will_seed": {
        "ibm01": 1.2920, "ibm02": 1.7621, "ibm03": 1.7133, "ibm04": 1.4832,
        "ibm06": 2.1817, "ibm07": 1.8451, "ibm08": 1.8156, "ibm09": 1.2919,
        "ibm10": 1.8897, "ibm11": 1.6036, "ibm12": 2.5017, "ibm13": 1.7567,
        "ibm14": 2.0539, "ibm15": 2.0699, "ibm16": 2.0121, "ibm17": 3.3138,
        "ibm18": 2.4894,
    },
    "SA_baseline": SA_BASELINES,
    "RePlAce_baseline": REPLACE_BASELINES,
}


# ── Placer loading ───────────────────────────────────────────────────────────

def _load_placer(path: Path):
    path = path.resolve()
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    if spec is None:
        raise RuntimeError(f"Failed to load placer from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for attr in vars(mod).values():
        if (
            isinstance(attr, type)
            and attr.__module__ == path.stem
            and callable(getattr(attr, "place", None))
        ):
            return attr()
    raise RuntimeError(
        f"No placer class with place(self, benchmark) -> Tensor found in {path}"
    )


# ── Formatting helpers ───────────────────────────────────────────────────────

def _pct(val, ref):
    """Positive = improvement over ref."""
    if ref is None or ref == 0:
        return None
    return (ref - val) / ref * 100


def _fmt_pct(p):
    if p is None:
        return "    -    "
    sign = "+" if p >= 0 else ""
    return f"{sign}{p:6.1f}%"


def _avg(values):
    v = [x for x in values if x is not None]
    return sum(v) / len(v) if v else None


# ── Per-benchmark result table ───────────────────────────────────────────────

def _print_per_benchmark(benchmarks, all_results, names):
    """all_results[name][bmark] = result dict"""
    col_w = 10
    cols = len(names)

    def header_row(label):
        return f"{'':>12}" + "".join(f"  {n[:col_w]:>{col_w}}" for n in label)

    print()
    print("=" * (14 + (col_w + 2) * cols))
    print(f"  Proxy Cost by Benchmark")
    print("=" * (14 + (col_w + 2) * cols))
    print(header_row(names))
    print(header_row(["SA_base", "RePlAce"] + [""] * max(0, cols - 2)))
    print("-" * (14 + (col_w + 2) * cols))

    for bmark in benchmarks:
        sa = SA_BASELINES.get(bmark)
        rep = REPLACE_BASELINES.get(bmark)
        row = f"  {bmark:>10}"
        vals = []
        for name in names:
            r = all_results.get(name, {}).get(bmark)
            if r is not None:
                vals.append(r["proxy_cost"])
                row += f"  {r['proxy_cost']:>{col_w}.4f}"
            else:
                vals.append(None)
                row += f"  {'-':>{col_w}}"
        # Indicate the winner
        valid_vals = [(v, i) for i, v in enumerate(vals) if v is not None]
        if valid_vals:
            best_idx = min(valid_vals, key=lambda x: x[0])[1]
            # Add star to winner column
        print(row)

    print("-" * (14 + (col_w + 2) * cols))

    # Averages
    row = f"  {'AVG':>10}"
    for name in names:
        res = all_results.get(name, {})
        vals = [res[b]["proxy_cost"] for b in benchmarks if b in res and res[b] is not None]
        if vals:
            row += f"  {sum(vals)/len(vals):>{col_w}.4f}"
        else:
            row += f"  {'-':>{col_w}}"
    print(row)
    print()


# ── Summary ranking table ────────────────────────────────────────────────────

def _print_ranking(benchmarks, all_results, names):
    print("=" * 72)
    print("  Leaderboard Ranking (avg proxy cost, lower = better)")
    print("=" * 72)
    print(f"  {'Rank':>4}  {'Placer':<28}  {'Avg Proxy':>10}  {'vs SA':>8}  {'vs RePlAce':>10}  {'Time':>8}")
    print("-" * 72)

    rows = []
    for name in names:
        res = all_results.get(name, {})
        proxies = [res[b]["proxy_cost"] for b in benchmarks if b in res and res[b] is not None]
        runtimes = [res[b]["runtime"] for b in benchmarks if b in res and res[b] is not None]
        if not proxies:
            continue
        avg_p = sum(proxies) / len(proxies)
        avg_t = sum(runtimes)

        sa_vals = [SA_BASELINES[b] for b in benchmarks if b in SA_BASELINES]
        rep_vals = [REPLACE_BASELINES[b] for b in benchmarks if b in REPLACE_BASELINES]
        avg_sa = sum(sa_vals) / len(sa_vals) if sa_vals else None
        avg_rep = sum(rep_vals) / len(rep_vals) if rep_vals else None

        rows.append((avg_p, name, avg_t, avg_sa, avg_rep))

    rows.sort(key=lambda x: x[0])

    for rank, (avg_p, name, avg_t, avg_sa, avg_rep) in enumerate(rows, 1):
        vs_sa = _fmt_pct(_pct(avg_p, avg_sa))
        vs_rep = _fmt_pct(_pct(avg_p, avg_rep))
        print(f"  {rank:>4}  {name:<28}  {avg_p:>10.4f}  {vs_sa}  {vs_rep}  {avg_t:>7.0f}s")

    # Also show published leaderboard entries for context
    print()
    print("  -- Published baselines --")
    leaderboard_rows = []
    for lname, scores in KNOWN_LEADERBOARD.items():
        proxies = [scores.get(b) for b in benchmarks if scores.get(b) is not None]
        if not proxies:
            continue
        avg_p = sum(proxies) / len(proxies)
        sa_vals = [SA_BASELINES[b] for b in benchmarks if b in SA_BASELINES]
        rep_vals = [REPLACE_BASELINES[b] for b in benchmarks if b in REPLACE_BASELINES]
        avg_sa = sum(sa_vals) / len(sa_vals) if sa_vals else None
        avg_rep = sum(rep_vals) / len(rep_vals) if rep_vals else None
        leaderboard_rows.append((avg_p, lname, avg_sa, avg_rep))

    leaderboard_rows.sort(key=lambda x: x[0])
    for avg_p, lname, avg_sa, avg_rep in leaderboard_rows:
        vs_sa = _fmt_pct(_pct(avg_p, avg_sa))
        vs_rep = _fmt_pct(_pct(avg_p, avg_rep))
        print(f"  {'-':>4}  {lname:<28}  {avg_p:>10.4f}  {vs_sa}  {vs_rep}  {'(published)':>8}")

    print("=" * 72)
    print()


# ── Save results ─────────────────────────────────────────────────────────────

def _save_results(all_results, benchmarks, names, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"comparison_{ts}.json"

    serializable = {}
    for name in names:
        serializable[name] = {}
        for bmark in benchmarks:
            r = all_results.get(name, {}).get(bmark)
            if r is not None:
                serializable[name][bmark] = {
                    "proxy_cost": float(r["proxy_cost"]),
                    "wirelength": float(r["wirelength"]),
                    "density": float(r["density"]),
                    "congestion": float(r["congestion"]),
                    "overlaps": int(r["overlaps"]),
                    "runtime": float(r["runtime"]),
                    "valid": bool(r["valid"]),
                    "sa_baseline": r.get("sa_baseline"),
                    "replace_baseline": r.get("replace_baseline"),
                }

    with open(out_path, "w") as f:
        json.dump({
            "timestamp": ts,
            "benchmarks": benchmarks,
            "placers": names,
            "results": serializable,
        }, f, indent=2)

    print(f"  Results saved to: {out_path}")
    return out_path


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="compare_placers",
        description="Compare multiple placer submissions side-by-side.",
    )
    parser.add_argument(
        "placers",
        nargs="+",
        help="Paths to placer .py files to compare.",
    )
    parser.add_argument(
        "--benchmark", "-b",
        nargs="+",
        default=None,
        metavar="BENCHMARK",
        help="Specific benchmark(s) to run (e.g. ibm01 ibm02). Default: ibm01.",
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Run on all 17 IBM benchmarks.",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress per-run verbose output.",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory to save JSON results (default: results/).",
    )
    args = parser.parse_args()

    if args.all:
        benchmarks = IBM_BENCHMARKS
    elif args.benchmark:
        benchmarks = args.benchmark
    else:
        benchmarks = ["ibm01"]

    testcase_root = Path("external/MacroPlacement/Testcases/ICCAD04")
    if not testcase_root.exists():
        print(f"Error: {testcase_root} not found. Run: git submodule update --init external/MacroPlacement")
        sys.exit(1)

    placer_paths = [Path(p) for p in args.placers]
    placers = {}
    for path in placer_paths:
        try:
            placers[path.stem if path.stem != "placer" else path.parent.name] = _load_placer(path)
        except Exception as e:
            print(f"  ERROR loading {path}: {e}")
            sys.exit(1)

    names = list(placers.keys())
    all_results = {name: {} for name in names}

    print()
    print("=" * 72)
    print(f"  Macro Placement Comparison  ({datetime.now().strftime('%Y-%m-%d %H:%M')})")
    print(f"  Benchmarks: {', '.join(benchmarks)}")
    print(f"  Placers:    {', '.join(names)}")
    print("=" * 72)
    print()

    for bmark in benchmarks:
        print(f"  [{bmark}]")
        for name, placer in placers.items():
            if not args.quiet:
                print(f"    {name}...", end=" ", flush=True)
            try:
                result = evaluate_benchmark(placer, bmark, str(testcase_root))
                all_results[name][bmark] = result
                if not args.quiet:
                    status = "VALID" if result["overlaps"] == 0 else f"INVALID ({result['overlaps']} overlaps)"
                    print(f"proxy={result['proxy_cost']:.4f}  (wl={result['wirelength']:.3f} "
                          f"den={result['density']:.3f} cong={result['congestion']:.3f})  "
                          f"{status}  [{result['runtime']:.1f}s]")
            except Exception as e:
                print(f"    ERROR on {bmark}: {e}")
                all_results[name][bmark] = None
        print()

    # Print comparison tables
    _print_per_benchmark(benchmarks, all_results, names)
    _print_ranking(benchmarks, all_results, names)

    # Save results
    _save_results(all_results, benchmarks, names, Path(args.output_dir))


if __name__ == "__main__":
    main()
