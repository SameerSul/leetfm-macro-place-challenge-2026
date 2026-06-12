"""Per-metric impact analysis across benchmark results.

Breaks each benchmark's placed proxy cost into its weighted components
(proxy = 1.0*wirelength + 0.5*density + 0.5*congestion) and compares against
the initial placement, so it is visible WHICH metric is responsible for the
remaining cost on each benchmark - and whether the problem cases share a
common metric. Use this to decide what the next placer change should target.

Inputs are the results JSONs written by run_synthetic.py:
    results.json        the synthetic anti-overfitting suite
    results_ibm.json    the 17 IBM benchmarks (run_synthetic.py --ibm)

Usage:
    uv run python system/v2/test/benchmarks/analyze_impact.py
    uv run python .../analyze_impact.py --no-synthetic   # IBM only
    uv run python .../analyze_impact.py --no-ibm         # synthetic only
    uv run python .../analyze_impact.py --worst 5        # deep-dive count
"""

import argparse
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent

W_WL, W_DEN, W_CONG = 1.0, 0.5, 0.5
METRICS = ("wirelength", "density", "congestion")
WEIGHTS = {"wirelength": W_WL, "density": W_DEN, "congestion": W_CONG}


def weighted(costs):
    """Weighted per-metric contributions to the proxy cost."""
    return {m: WEIGHTS[m] * costs[m] for m in METRICS}


def load_group(path, group):
    if not path.exists():
        return []
    entries = json.loads(path.read_text())
    rows = []
    for e in entries:
        if "placed" not in e:
            continue
        rows.append({
            "name": e["name"],
            "group": group,
            "axis": e.get("axis", ""),
            "valid": e.get("valid", True),
            "initial": e["initial"],
            "placed": e["placed"],
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-synthetic", action="store_true",
                        help="exclude the synthetic suite from the analysis")
    parser.add_argument("--no-ibm", action="store_true",
                        help="exclude the IBM benchmarks from the analysis")
    parser.add_argument("--synthetic-results", default=str(OUT / "results.json"))
    parser.add_argument("--ibm-results", default=str(OUT / "results_ibm.json"))
    parser.add_argument("--worst", type=int, default=3,
                        help="how many worst benchmarks to deep-dive (default 3)")
    args = parser.parse_args()

    rows = []
    if not args.no_synthetic:
        path = Path(args.synthetic_results)
        got = load_group(path, "synthetic")
        if not got:
            print(f"note: no synthetic results at {path} (run run_synthetic.py)")
        rows += got
    if not args.no_ibm:
        path = Path(args.ibm_results)
        got = load_group(path, "ibm")
        if not got:
            print(f"note: no IBM results at {path} (run run_synthetic.py --ibm)")
        rows += got
    if not rows:
        print("nothing to analyze")
        return

    # ---- Per-benchmark table: weighted shares of the placed proxy + deltas ----
    print("\nproxy = 1.0*wl + 0.5*den + 0.5*cong   |  share = weighted part / proxy")
    print("delta columns are weighted (placed - initial): negative = the placer improved it")
    print("-" * 118)
    print(f"{'benchmark':>16} {'grp':>4} {'proxy':>8} {'run avg':>8} | "
          f"{'wl':>7} {'den/2':>7} {'cong/2':>7} | "
          f"{'wl%':>5} {'den%':>5} {'cong%':>5} | "
          f"{'d_wl':>8} {'d_den':>8} {'d_cong':>8} | dominant")
    print("-" * 118)

    running_sum = 0.0
    for i, r in enumerate(sorted(rows, key=lambda r: (r["group"], r["name"]))):
        w = weighted(r["placed"])
        w0 = weighted(r["initial"])
        proxy = r["placed"]["proxy"]
        running_sum += proxy
        shares = {m: 100.0 * w[m] / proxy for m in METRICS}
        deltas = {m: w[m] - w0[m] for m in METRICS}
        dominant = max(METRICS, key=lambda m: w[m])
        flag = "" if r["valid"] else "  INVALID"
        print(f"{r['name']:>16} {r['group'][:4]:>4} {proxy:>8.4f} {running_sum/(i+1):>8.4f} | "
              f"{w['wirelength']:>7.4f} {w['density']:>7.4f} {w['congestion']:>7.4f} | "
              f"{shares['wirelength']:>4.0f}% {shares['density']:>4.0f}% {shares['congestion']:>4.0f}% | "
              f"{deltas['wirelength']:>+8.4f} {deltas['density']:>+8.4f} {deltas['congestion']:>+8.4f} | "
              f"{dominant}{flag}")

    # ---- Group + combined averages ----
    print("-" * 118)
    groups = sorted({r["group"] for r in rows})
    for g in groups + ["ALL"]:
        sel = rows if g == "ALL" else [r for r in rows if r["group"] == g]
        n = len(sel)
        avg = sum(r["placed"]["proxy"] for r in sel) / n
        avg_init = sum(r["initial"]["proxy"] for r in sel) / n
        avg_w = {m: sum(weighted(r["placed"])[m] for r in sel) / n for m in METRICS}
        print(f"{g:>16} avg proxy {avg:>8.4f}  (initial {avg_init:.4f}, n={n})  "
              f"avg weighted: wl {avg_w['wirelength']:.4f} | den {avg_w['density']:.4f} "
              f"| cong {avg_w['congestion']:.4f}")

    # ---- Diagnosis: which metric to target next ----
    n = len(rows)
    avg_share = {m: sum(100.0 * weighted(r["placed"])[m] / r["placed"]["proxy"]
                        for r in rows) / n for m in METRICS}
    # Relative improvement: how much of each metric's initial weighted cost the
    # placer removed. A metric with a big share AND low improvement is the lever.
    rel_impr = {}
    for m in METRICS:
        init_total = sum(weighted(r["initial"])[m] for r in rows)
        placed_total = sum(weighted(r["placed"])[m] for r in rows)
        rel_impr[m] = 100.0 * (init_total - placed_total) / max(init_total, 1e-12)

    print("\n=== Diagnosis ===")
    for m in METRICS:
        print(f"  {m:>11}: {avg_share[m]:5.1f}% of remaining proxy cost, "
              f"{rel_impr[m]:+6.1f}% improved vs initial")

    worst = sorted(rows, key=lambda r: -r["placed"]["proxy"])[: args.worst]
    print(f"\n  worst {len(worst)} benchmarks and their dominant metric:")
    for r in worst:
        w = weighted(r["placed"])
        dominant = max(METRICS, key=lambda m: w[m])
        share = 100.0 * w[dominant] / r["placed"]["proxy"]
        axis = f"  [{r['axis']}]" if r["axis"] else ""
        print(f"    {r['name']:>16} proxy={r['placed']['proxy']:.4f} -> "
              f"{dominant} ({share:.0f}%){axis}")

    common = {max(METRICS, key=lambda m: weighted(r["placed"])[m]) for r in worst}
    target = max(METRICS, key=lambda m: avg_share[m])
    if len(common) == 1:
        print(f"\n  common cause across the worst cases: {common.pop().upper()}")
    print(f"  => next change should target: {target.upper()} "
          f"({avg_share[target]:.0f}% of remaining cost, "
          f"only {rel_impr[target]:.0f}% improved so far)")


if __name__ == "__main__":
    main()
