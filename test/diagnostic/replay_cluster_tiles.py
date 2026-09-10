"""Paired cluster-tile trials on captured final placements and immutable contracts."""

import argparse
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from macro_place.evaluate import NG45_BENCHMARKS
from macro_place.loader import load_benchmark, load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from macro_place.utils import validate_placement
from placer.local_search.cluster_decompress import hierarchy_quality_metric
from placer.local_search.cluster_tile_rearrange import final_cluster_tile_relief, _tail_view, _unit_pressure
from placer.local_search.hierarchy_quality import (
    hierarchy_island_contract, hierarchy_island_metrics,
    hierarchy_quality_vector, hierarchy_vector_contract,
)
from placer.plc.placement import clamp_in_bounds, _ensure_pos_cache
from placer.scoring.exact import _exact_proxy
from placer.scoring.incremental import IncrementalScorer
from test.benchmarks.run_synthetic import _truth_hierarchy_audit
from test.verification import _verify_ng45_hierarchy_tags as tags


def load(name):
    if name.startswith("ibm"):
        return load_benchmark_from_dir(str(ROOT / "external/MacroPlacement/Testcases/ICCAD04" / name))
    source = (ROOT / "test/benchmarks/testcases" / name if name.startswith("syn")
              else Path(NG45_BENCHMARKS[name]))
    return load_benchmark(str(source / "netlist.pb.txt"), str(source / "initial.plc"))


def contract_for(data, benchmark):
    hierarchy, limits = data["hierarchy"], data["limits"]
    cw, ch = float(benchmark.canvas_width), float(benchmark.canvas_height)
    n = benchmark.num_hard_macros

    def allowed(hard, soft):
        if hierarchy_quality_metric(hard, hierarchy.clusters) > data["hard_quality_limit"]:
            return False
        projections = [(hierarchy.clusters, hierarchy.cluster_softs, hierarchy.bridge_softs,
                        hierarchy.edges, limits["hierarchy_contract_limits"])]
        if limits["subhierarchy_contract_active"]:
            projections.extend([
                (hierarchy.subclusters, hierarchy.subcluster_softs, hierarchy.subcluster_bridge_softs,
                 hierarchy.subcluster_edges, limits["subcluster_hierarchy_contract_limits"]),
                (hierarchy.parent_clusters, hierarchy.parent_cluster_softs, hierarchy.parent_bridge_softs,
                 hierarchy.parent_edges, limits["parent_hierarchy_contract_limits"]),
            ])
        for clusters, softs, bridges, edges, bound in projections:
            if clusters:
                vector = hierarchy_quality_vector(hard, soft, clusters, softs, bridges, edges, cw, ch)
                if not hierarchy_vector_contract(vector, bound)[0]:
                    return False
        metrics = hierarchy_island_metrics(
            hard, soft, hierarchy.clusters, hierarchy.cluster_softs,
            benchmark.macro_sizes.numpy()[:n], cw, ch, diagnostics=False,
        )
        return hierarchy_island_contract(metrics, limits["seed_island_limits"])[0]

    return allowed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--placements", type=Path, help="Require replay to match these saved pipeline outputs")
    parser.add_argument("names", nargs="*")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.inputs.glob("*_final.pkl"))
    rows, ng45 = [], {}
    for path in paths:
        name = path.name.removesuffix("_final.pkl")
        if args.names and name not in args.names:
            continue
        data = pickle.loads(path.read_bytes())
        benchmark, plc = load(name)
        before = clamp_in_bounds(torch.from_numpy(np.vstack([data["hard"], data["soft"]]).astype(np.float32)), benchmark)
        _exact_proxy(before, benchmark, plc)
        scorer = IncrementalScorer(plc, benchmark, before.numpy())
        arrays = [getattr(scorer, key) for key in (
            "H_flat", "V_flat", "H_macro_flat", "V_macro_flat", "H_smoothed",
            "V_smoothed", "grid_occupied", "committed_hard_pos", "committed_soft_pos",
        )] + [_ensure_pos_cache(plc)]
        snapshots = [array.copy() for array in arrays]
        metrics = scorer.visualizer_metrics()
        view = _tail_view(scorer)
        pressure = _unit_pressure(scorer, np.arange(len(before)), before.numpy(), view)
        assert pressure == _unit_pressure(scorer, np.arange(len(before)), before.numpy(), view)
        assert metrics == scorer.visualizer_metrics()
        for array, snapshot in zip(arrays, snapshots):
            np.testing.assert_array_equal(array, snapshot)
        allowed = contract_for(data, benchmark)
        assert allowed(before[:len(data["hard"])].numpy(), before[len(data["hard"]):].numpy()), name
        start = time.monotonic()
        hard, soft, _, _, stats = final_cluster_tile_relief(
            data["hard"], data["soft"], benchmark, plc, data["hierarchy"], allowed,
            hard_region=data["hard_region"], soft_region=data["region"],
            deadline=start + args.seconds if args.seconds > 0 else None,
        )
        elapsed = time.monotonic() - start
        after = clamp_in_bounds(torch.from_numpy(np.vstack([hard, soft]).astype(np.float32)), benchmark)
        if args.placements:
            np.testing.assert_array_equal(after.numpy(), np.load(args.placements / f"{name}.npy"))
        assert allowed(after[:len(hard)].numpy(), after[len(hard):].numpy()), name
        np.testing.assert_array_equal(before[benchmark.macro_fixed], after[benchmark.macro_fixed])
        # Fresh external evaluator state for both saved-coordinate comparisons.
        fresh_benchmark, fresh_plc = load(name)
        scores = []
        for positions in (before, after):
            valid, issues = validate_placement(positions, fresh_benchmark)
            assert valid, (name, issues)
            costs = compute_proxy_cost(positions.double(), fresh_benchmark, fresh_plc)
            assert costs["overlap_count"] == 0, name
            scores.append(float(costs["proxy_cost"]))
        assert scores[1] <= scores[0] + 1e-7, (name, scores)
        row = dict(benchmark=name, before=scores[0], after=scores[1],
                   gain=scores[0] - scores[1], elapsed_s=elapsed,
                   attribution_preserves_state=True, **stats)
        if name.startswith("syn"):
            metadata = json.loads((ROOT / "test/benchmarks/metadata" / f"{name}.json").read_text())
            row["truth"] = _truth_hierarchy_audit(
                name, metadata.get("axis", ""), metadata, fresh_benchmark, fresh_plc,
                fresh_benchmark.macro_positions, after,
            )
            assert row["truth"]["passed"], name
        if name in NG45_BENCHMARKS:
            ng45[name] = dict(benchmark=fresh_benchmark, placement=after, valid=True, proxy_cost=scores[1])
        rows.append(row)
        np.save(args.out / f"{name}.npy", after.numpy())
        (args.out / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(json.dumps(row), flush=True)
    if ng45:
        previous_argv, previous_evaluate = sys.argv, tags.evaluate_benchmark
        try:
            tags.evaluate_benchmark = lambda placer, name, *args, **kwargs: ng45[name]
            sys.argv = ["_verify_ng45_hierarchy_tags.py", *ng45]
            tags.main()
        finally:
            sys.argv, tags.evaluate_benchmark = previous_argv, previous_evaluate


if __name__ == "__main__":
    main()
