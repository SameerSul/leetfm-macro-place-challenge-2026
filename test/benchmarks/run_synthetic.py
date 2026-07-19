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
    uv run python test/benchmarks/run_synthetic.py
    uv run python .../run_synthetic.py -b syn02_fixed
    uv run python .../run_synthetic.py --placer src/main.py
    uv run python .../run_synthetic.py --budget 60 --skip-initial-vis
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from macro_place.evaluate import _load_placer  # noqa: E402
from macro_place.loader import load_benchmark  # noqa: E402
from macro_place.objective import compute_proxy_cost  # noqa: E402
from macro_place.utils import validate_placement, visualize_placement  # noqa: E402
from placer.local_search.hierarchy_model import HierarchyModel  # noqa: E402
from placer.local_search.hierarchy_quality import (  # noqa: E402
    hierarchy_quality_vector,
    hierarchy_vector_contract,
    hierarchy_vector_limits,
    hierarchy_vector_margins,
)
from placer.local_search.plateau_telemetry import log_plateau_event  # noqa: E402
from utils import constants as const  # noqa: E402


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


def _pair_count(count):
    return int(count) * max(0, int(count) - 1) // 2


def _inference_agreement(plc, benchmark, hard_truth):
    """Compare inferred hard clusters with the generator's known communities."""
    n = int(benchmark.num_hard_macros)
    sizes = benchmark.macro_sizes[:n].detach().cpu().numpy().astype(np.float64)
    model = HierarchyModel.build(plc, n, int(benchmark.num_soft_macros), hard_sizes=sizes)
    inferred = np.asarray(model.labels, dtype=np.int64)
    truth = np.asarray(hard_truth, dtype=np.int64)
    covered = inferred >= 0
    contingency = Counter((int(inferred[i]), int(truth[i])) for i in np.flatnonzero(covered))
    inferred_counts = Counter(int(value) for value in inferred[covered])
    truth_counts = Counter(int(value) for value in truth)
    true_positive_pairs = sum(_pair_count(value) for value in contingency.values())
    inferred_pairs = sum(_pair_count(value) for value in inferred_counts.values())
    truth_pairs = sum(_pair_count(value) for value in truth_counts.values())
    majority = sum(
        max(
            (count for (group, _truth), count in contingency.items() if group == inferred_id),
            default=0,
        )
        for inferred_id in inferred_counts
    )
    return {
        "hard_coverage": float(np.mean(covered)) if n else 0.0,
        "cluster_purity": float(majority / max(int(np.count_nonzero(covered)), 1)),
        "pair_precision": float(true_positive_pairs / max(inferred_pairs, 1)),
        "pair_recall": float(true_positive_pairs / max(truth_pairs, 1)),
        "inferred_clusters": int(len(inferred_counts)),
        "truth_clusters": int(len(truth_counts)),
        "hierarchy_source": str(model.cluster_source),
    }


def _truth_hierarchy_audit(name, axis, metadata, benchmark, plc, initial, placed):
    """Evaluate final placement against the generator's known hierarchy."""
    truth = metadata.get("hierarchy_truth")
    if not isinstance(truth, dict):
        return None
    n = int(benchmark.num_hard_macros)
    n_soft = int(benchmark.num_soft_macros)
    hard_truth = np.asarray(truth.get("hard_cluster", []), dtype=np.int64)
    soft_truth = np.asarray(truth.get("soft_cluster", []), dtype=np.int64)
    if hard_truth.size != n or soft_truth.size != n_soft:
        return None
    cluster_ids = sorted(set(int(value) for value in hard_truth))
    clusters = {cid: np.flatnonzero(hard_truth == cid) for cid in cluster_ids}
    cluster_softs = {
        cid: n + np.flatnonzero(soft_truth == cid)
        for cid in cluster_ids
        if np.any(soft_truth == cid)
    }
    initial_np = initial.detach().cpu().numpy().astype(np.float64)
    placed_np = placed.detach().cpu().numpy().astype(np.float64)
    vector_initial = hierarchy_quality_vector(
        initial_np[:n],
        initial_np[n : n + n_soft],
        clusters,
        cluster_softs,
        {},
        (),
        float(benchmark.canvas_width),
        float(benchmark.canvas_height),
    )
    vector_final = hierarchy_quality_vector(
        placed_np[:n],
        placed_np[n : n + n_soft],
        clusters,
        cluster_softs,
        {},
        (),
        float(benchmark.canvas_width),
        float(benchmark.canvas_height),
    )
    limits = hierarchy_vector_limits(
        vector_initial,
        const.HIER_VECTOR_CONTRACT_ABS_SLACK,
        float(const.HIER_VECTOR_CONTRACT_REL_SLACK),
    )
    passed, violations = hierarchy_vector_contract(vector_final, limits)
    agreement = _inference_agreement(plc, benchmark, hard_truth)
    result = {
        "passed": bool(passed),
        "reference_overlaps": int(metadata.get("seed_overlaps", 0)),
        "vector": vector_final,
        "reference_vector": vector_initial,
        "limits": limits,
        "margins": hierarchy_vector_margins(vector_final, limits),
        "violations": violations,
        "inference_agreement": agreement,
    }
    log_plateau_event(
        "hierarchy_truth_audit",
        benchmark=str(name),
        stage="final",
        candidate="final_placement",
        reference="generated_initial",
        selected=True,
        passed=bool(passed),
        axis=str(axis),
        hierarchy_source="synthetic_ground_truth",
        hierarchy_provenance="synthetic_ground_truth",
        coverage_scope="truth",
        coverage={"clustered_hard_fraction": 1.0, "soft_coverage": 1.0},
        reference_overlaps=int(metadata.get("seed_overlaps", 0)),
        vector=vector_final,
        reference_vector=vector_initial,
        limits=limits,
        margins=result["margins"],
        violations=violations,
        inference_agreement=agreement,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--placer",
        default=str(ROOT / "src/main.py"),
        help="placer .py file (default: v2 main.py)",
    )
    parser.add_argument("-b", "--benchmark", help="run a single benchmark by name")
    parser.add_argument(
        "--budget",
        type=float,
        default=150.0,
        help="per-benchmark placer time budget in seconds (default 150)",
    )
    parser.add_argument(
        "--skip-initial-vis",
        action="store_true",
        help="skip rendering the seed-placement visualization",
    )
    parser.add_argument(
        "--skip-vis",
        action="store_true",
        help="skip all placement visualizations",
    )
    parser.add_argument(
        "--initial-only",
        action="store_true",
        help="only score + visualize the seed placements (no placer run)",
    )
    parser.add_argument(
        "--ibm",
        action="store_true",
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
        metadata = {}
        meta_path = OUT / "metadata" / f"{name}.json"
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text())
        axis = metadata.get("axis", "")

        print(f"\n=== {name} ===")
        if axis:
            print(f"  axis: {axis}")
        benchmark, plc = load_benchmark(str(netlist), str(netlist.parent / "initial.plc"))
        # Give the hierarchy path both the generated source files for
        # DREAMPlace and the already-loaded exact scorer.
        benchmark._source_dir = netlist.parent
        benchmark._cached_plc = plc

        entry = {"name": name, "axis": axis}

        t = time.time()
        init_costs = compute_proxy_cost(benchmark.macro_positions, benchmark, plc)
        entry["initial"] = costs_summary(init_costs)
        print(f"  initial: {entry['initial']}  [scored in {time.time() - t:.0f}s]")
        if not args.skip_vis and not args.skip_initial_vis:
            visualize_placement(
                benchmark.macro_positions,
                benchmark,
                save_path=str(vis_dir / f"{name}_initial.png"),
                plc=plc,
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
            truth_audit = _truth_hierarchy_audit(
                name,
                axis,
                metadata,
                benchmark,
                plc,
                benchmark.macro_positions,
                placement,
            )
            if truth_audit is not None:
                entry["hierarchy_truth"] = truth_audit
            status = "VALID" if is_valid else f"INVALID ({violations[:2]})"
            print(
                f"  placed:  {entry['placed']}  [{runtime:.0f}s]  {status}  "
                f"delta={entry['delta_vs_initial']:+.4f}"
            )
            if not args.skip_vis:
                visualize_placement(
                    placement,
                    benchmark,
                    save_path=str(vis_dir / f"{name}_placed.png"),
                    plc=plc,
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
