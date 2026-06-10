"""Evaluate a placer on the synthetic anti-overfitting benchmark suite.

For each synthetic benchmark this scores the seed placement, runs the placer,
scores the result, and saves side-by-side visualizations (placement + density
heatmap + congestion heatmap) so the impact of each benchmark axis on the
proxy cost is visible.

Outputs (under this directory):
    vis/<name>_initial.png   the benchmark itself (seed placement)
    vis/<name>_placed.png    the placer's result
    results.json             per-benchmark cost breakdown

Usage:
    uv run python submissions/varrahan/v2/test/benchmarks/run_synthetic.py
    uv run python .../run_synthetic.py -b syn02_fixed
    uv run python .../run_synthetic.py --placer submissions/varrahan/v1/placer.py
    uv run python .../run_synthetic.py --budget 60 --skip-initial-vis
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from macro_place.evaluate import _load_placer  # noqa: E402
from macro_place.loader import load_benchmark  # noqa: E402
from macro_place.objective import compute_proxy_cost  # noqa: E402
from macro_place.utils import validate_placement, visualize_placement  # noqa: E402


def bounds_violations(placement, benchmark):
    """Count macros out of canvas bounds and the worst overhang in microns."""
    half = benchmark.macro_sizes / 2
    lo = placement - half
    hi = placement + half
    over = torch.zeros(placement.shape[0])
    over = torch.maximum(over, -lo[:, 0])
    over = torch.maximum(over, -lo[:, 1])
    over = torch.maximum(over, hi[:, 0] - benchmark.canvas_width)
    over = torch.maximum(over, hi[:, 1] - benchmark.canvas_height)
    out = over > 1e-6
    return int(out.sum()), float(over.max())


def costs_summary(costs):
    return {
        "proxy": round(float(costs["proxy_cost"]), 4),
        "wirelength": round(float(costs["wirelength_cost"]), 4),
        "density": round(float(costs["density_cost"]), 4),
        "congestion": round(float(costs["congestion_cost"]), 4),
        "overlaps": int(costs["overlap_count"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--placer",
        default=str(ROOT / "submissions/varrahan/v2/src/main.py"),
        help="placer .py file (default: v2 main.py)",
    )
    parser.add_argument("-b", "--benchmark", help="run a single benchmark by name")
    parser.add_argument(
        "--budget", type=float, default=150.0,
        help="per-benchmark placer time budget in seconds (default 150)",
    )
    parser.add_argument(
        "--skip-initial-vis", action="store_true",
        help="skip rendering the seed-placement visualization",
    )
    parser.add_argument(
        "--initial-only", action="store_true",
        help="only score + visualize the seed placements (no placer run)",
    )
    parser.add_argument(
        "--ibm", action="store_true",
        help="run on the 17 IBM ICCAD04 benchmarks instead of the synthetic "
             "suite (writes results_ibm.json, same schema - feeds "
             "analyze_impact.py's IBM group)",
    )
    args = parser.parse_args()

    if args.ibm:
        cases_dir = ROOT / "external/MacroPlacement/Testcases/ICCAD04"
    else:
        cases_dir = OUT / "testcases"
    cases = sorted(cases_dir.glob("*/netlist.pb.txt"))
    if args.benchmark:
        cases = [c for c in cases if c.parent.name == args.benchmark]
    if not cases:
        print("no testcases found - run generate_benchmarks.py first")
        sys.exit(1)

    vis_dir = OUT / "vis"
    vis_dir.mkdir(exist_ok=True)

    placer = None
    if not args.initial_only:
        placer = _load_placer(Path(args.placer))
        if hasattr(placer, "time_budget_s"):
            placer.time_budget_s = args.budget

    results = []
    for netlist in cases:
        name = netlist.parent.name
        axis = ""
        meta_path = OUT / "metadata" / f"{name}.json"
        if meta_path.exists():
            axis = json.loads(meta_path.read_text()).get("axis", "")

        print(f"\n=== {name} ===")
        if axis:
            print(f"  axis: {axis}")
        benchmark, plc = load_benchmark(str(netlist), str(netlist.parent / "initial.plc"))
        # v2 resolves its exact-scoring plc by benchmark name, which only works
        # for ICCAD04/NG45 paths; hand it the plc directly instead
        benchmark._cached_plc = plc

        entry = {"name": name, "axis": axis}

        t = time.time()
        init_costs = compute_proxy_cost(benchmark.macro_positions, benchmark, plc)
        entry["initial"] = costs_summary(init_costs)
        print(f"  initial: {entry['initial']}  [scored in {time.time() - t:.0f}s]")
        if not args.skip_initial_vis:
            visualize_placement(
                benchmark.macro_positions, benchmark,
                save_path=str(vis_dir / f"{name}_initial.png"), plc=plc,
            )

        if placer is not None:
            t = time.time()
            placement = placer.place(benchmark)
            runtime = time.time() - t
            is_valid, violations = validate_placement(placement, benchmark)
            final_costs = compute_proxy_cost(placement, benchmark, plc)
            entry["placed"] = costs_summary(final_costs)
            entry["runtime_s"] = round(runtime, 1)
            entry["valid"] = bool(is_valid)
            n_oob, worst_oob = bounds_violations(placement, benchmark)
            if n_oob:
                entry["out_of_bounds"] = {"count": n_oob, "worst_um": round(worst_oob, 3)}
                print(f"  out-of-bounds: {n_oob} macros, worst overhang {worst_oob:.3f}um")
            entry["delta_vs_initial"] = round(
                entry["placed"]["proxy"] - entry["initial"]["proxy"], 4
            )
            status = "VALID" if is_valid else f"INVALID ({violations[:2]})"
            print(
                f"  placed:  {entry['placed']}  [{runtime:.0f}s]  {status}  "
                f"delta={entry['delta_vs_initial']:+.4f}"
            )
            visualize_placement(
                placement, benchmark,
                save_path=str(vis_dir / f"{name}_placed.png"), plc=plc,
            )
        results.append(entry)

    if len(results) > 1:
        print("\n" + "-" * 96)
        header = f"{'benchmark':>16}  {'init proxy':>10}"
        if not args.initial_only:
            header += f"  {'placed':>8}  {'delta':>8}  {'overlaps':>8}  {'valid':>6}  {'time':>6}"
        print(header)
        print("-" * 96)
        for e in results:
            row = f"{e['name']:>16}  {e['initial']['proxy']:>10.4f}"
            if "placed" in e:
                row += (
                    f"  {e['placed']['proxy']:>8.4f}  {e['delta_vs_initial']:>+8.4f}"
                    f"  {e['placed']['overlaps']:>8}  {str(e['valid']):>6}  {e.get('runtime_s', 0):>5.0f}s"
                )
            print(row)
        placed = [e for e in results if "placed" in e]
        if placed:
            avg = sum(e["placed"]["proxy"] for e in placed) / len(placed)
            print("-" * 96)
            print(f"{'AVG placed proxy':>28}  {avg:.4f}")

    results_path = OUT / ("results_ibm.json" if args.ibm else "results.json")
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nresults -> {results_path}\nvisualizations -> {vis_dir}/")


if __name__ == "__main__":
    main()
