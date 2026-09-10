"""Same-source controls for the September 9 speed changes; no production switches."""

import argparse
import ast
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "ml_data/speed_equivalence/20260909"
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]


def reference_module(relative):
    path = HERE / "before" / relative
    spec = importlib.util.spec_from_file_location("speed_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference_group():
    from placer.scoring import incremental

    path = HERE / "before/src/placer/scoring/incremental.py"
    tree = ast.parse(path.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "IncrementalScorer"
    )
    node = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "_score_multi_move"
    )
    namespace = dict(vars(incremental))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[node.name]


def install_reference():
    from placer.local_search import hierarchy_quality, cluster_void_relocation
    from placer.scoring.incremental import IncrementalScorer

    old_metrics = reference_module(
        "src/placer/local_search/hierarchy_quality.py"
    ).hierarchy_island_metrics
    old_void = reference_module(
        "src/placer/local_search/cluster_void_relocation.py"
    )._void_cluster_relocation

    def metrics(*args, diagnostics=True, **kwargs):
        return old_metrics(*args, **kwargs)

    replacements = {
        hierarchy_quality.hierarchy_island_metrics: metrics,
        cluster_void_relocation._void_cluster_relocation: old_void,
    }
    for name, module in list(sys.modules.items()):
        if name.startswith("placer.") and module is not None:
            for key, value in list(vars(module).items()):
                if callable(value):
                    replacement = (
                        replacements.get(value) if isinstance(value, type(metrics)) else None
                    )
                    if replacement is not None:
                        setattr(module, key, replacement)
    IncrementalScorer._score_multi_move = reference_group()


def compare_outputs(control, candidate):
    import numpy as np

    before = {r["benchmark"]: r for r in json.loads((control / "results.json").read_text())}
    after = json.loads((candidate / "results.json").read_text())
    assert set(before) == {r["benchmark"] for r in after}, "benchmark coverage differs"
    assert all(r["valid"] and r["overlaps"] == 0 for r in [*before.values(), *after])
    audits = []
    for directory in (control, candidate):
        audits.append(
            {
                r["benchmark"]: r
                for line in (directory / "trace.jsonl").read_text().splitlines()
                if (r := json.loads(line)).get("event") == "hierarchy_contract_audit"
                and r.get("stage") == "final"
            }
        )
    rows = []
    for r in after:
        name = r["benchmark"]
        x, y = [np.load(directory / f"{name}.npy") for directory in (control, candidate)]
        a, b = [table[name] for table in audits]
        hierarchy_equal = all(
            a[key] == b[key]
            for key in ("vector", "parent_vector", "subcluster_vector", "island_metrics")
        )
        rows.append(
            dict(
                benchmark=name,
                proxy_delta=r["proxy_cost"] - before[name]["proxy_cost"],
                coordinates_equal=bool(np.array_equal(x, y)),
                hierarchy_equal=hierarchy_equal,
                audits_pass=bool(a["passed"] and b["passed"]),
                before_s=before[name]["runtime"],
                after_s=r["runtime"],
            )
        )
    report = dict(
        control=str(control),
        candidate=str(candidate),
        designs=rows,
        proxy_regressions=[r["benchmark"] for r in rows if r["proxy_delta"] > 1.0e-7],
        changed_coordinates=[r["benchmark"] for r in rows if not r["coordinates_equal"]],
        changed_hierarchy=[r["benchmark"] for r in rows if not r["hierarchy_equal"]],
        before_s=sum(r["before_s"] for r in rows),
        after_s=sum(r["after_s"] for r in rows),
    )
    (candidate / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "designs"}, indent=2))
    assert all(r["audits_pass"] for r in rows)
    assert not report["proxy_regressions"], report["proxy_regressions"]
    return report


def audit_saved(directory, names=()):
    import numpy as np
    import torch
    from macro_place.evaluate import NG45_BENCHMARKS
    from macro_place.loader import load_benchmark, load_benchmark_from_dir
    from macro_place.objective import compute_proxy_cost
    from macro_place.utils import validate_placement
    from test.benchmarks.run_synthetic import _truth_hierarchy_audit
    from test.verification import _verify_ng45_hierarchy_tags as tags

    rows, ng45 = [], {}
    for result in json.loads((directory / "results.json").read_text()):
        name = result["benchmark"]
        if names and name not in names:
            continue
        if name in NG45_BENCHMARKS or name.startswith("syn"):
            source = (
                Path(NG45_BENCHMARKS[name])
                if name in NG45_BENCHMARKS
                else ROOT / "test/benchmarks/testcases" / name
            )
            benchmark, plc = load_benchmark(
                str(source / "netlist.pb.txt"), str(source / "initial.plc")
            )
        else:
            benchmark, plc = load_benchmark_from_dir(
                str(ROOT / "external/MacroPlacement/Testcases/ICCAD04" / name)
            )
        placement = torch.from_numpy(np.load(directory / f"{name}.npy"))
        valid, violations = validate_placement(placement, benchmark)
        # Synthetic runs use fp64 field scoring. Promote the same saved fp32
        # coordinates to avoid losing precision in the scalar wirelength sum.
        score_input = placement.double() if name.startswith("syn") else placement
        costs = compute_proxy_cost(score_input, benchmark, plc)
        assert valid and costs["overlap_count"] == 0, (name, violations)
        delta = float(costs["proxy_cost"] - result["proxy_cost"])
        assert abs(delta) <= 1.0e-7, (name, delta)
        row = dict(benchmark=name, valid=True, score_verified=True, score_delta=delta)
        if name.startswith("syn"):
            metadata = json.loads((ROOT / "test/benchmarks/metadata" / f"{name}.json").read_text())
            row["truth"] = _truth_hierarchy_audit(
                name,
                metadata.get("axis", ""),
                metadata,
                benchmark,
                plc,
                benchmark.macro_positions,
                placement,
            )
            assert row["truth"]["passed"], name
        if name in NG45_BENCHMARKS:
            ng45[name] = dict(
                benchmark=benchmark, placement=placement, valid=True, proxy_cost=costs["proxy_cost"]
            )
        rows.append(row)
        print(f"Saved-placement audit passed: {name}; score delta={delta:.3g}", flush=True)
    if ng45:
        previous_argv = sys.argv
        previous_evaluate = tags.evaluate_benchmark
        try:
            tags.evaluate_benchmark = lambda placer, name, *args, **kwargs: ng45[name]
            sys.argv = ["_verify_ng45_hierarchy_tags.py", *ng45]
            tags.main()
        finally:
            sys.argv = previous_argv
            tags.evaluate_benchmark = previous_evaluate
    (directory / "independent_audits.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(
        f"Independent saved-placement audits passed: {len(rows)} designs; {len(ng45)} NG45 tag checks."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--no-deadlines", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--compare", type=Path)
    parser.add_argument("names", nargs="*")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ["HIER_PLATEAU_TRACE_PATH"] = str(args.out / "trace.jsonl")
    os.environ["HIER_DIAGNOSTIC_NO_DEADLINES"] = str(int(args.no_deadlines))
    if args.compare:
        compare_outputs(args.compare, args.out)
        return
    if args.audit_only:
        audit_saved(args.out, args.names)
        return
    if not args.names:
        parser.error("provide benchmark names or ibm/ng45/synthetic")
    import numpy as np
    from main import MacroPlacer
    from macro_place.evaluate import IBM_BENCHMARKS, NG45_BENCHMARKS, evaluate_benchmark
    from macro_place.loader import load_benchmark
    from macro_place.objective import compute_proxy_cost
    from macro_place.utils import validate_placement

    if args.reference:
        install_reference()
    names = []
    for name in args.names:
        names.extend(
            IBM_BENCHMARKS
            if name == "ibm"
            else (
                list(NG45_BENCHMARKS)
                if name == "ng45"
                else (
                    sorted(
                        p.parent.name
                        for p in (ROOT / "test/benchmarks/testcases").glob("*/netlist.pb.txt")
                    )
                    if name == "synthetic"
                    else [name]
                )
            )
        )
    placer = MacroPlacer()
    rows = []
    for name in names:
        if name.startswith("syn"):
            source = ROOT / "test/benchmarks/testcases" / name
            benchmark, plc = load_benchmark(
                str(source / "netlist.pb.txt"), str(source / "initial.plc")
            )
            benchmark._source_dir, benchmark._cached_plc = source, plc
            compute_proxy_cost(benchmark.macro_positions, benchmark, plc)
            start = time.perf_counter()
            placement = placer.place(benchmark)
            runtime = time.perf_counter() - start
            valid, violations = validate_placement(placement, benchmark)
            costs = compute_proxy_cost(placement, benchmark, plc)
            result = {
                "runtime": runtime,
                "valid": valid,
                "placement": placement,
                "benchmark": benchmark,
                "plc": plc,
                "proxy_cost": costs["proxy_cost"],
                "wirelength": costs["wirelength_cost"],
                "density": costs["density_cost"],
                "congestion": costs["congestion_cost"],
                "overlaps": costs["overlap_count"],
            }
        else:
            result = evaluate_benchmark(
                placer,
                name,
                str(ROOT / "external/MacroPlacement/Testcases/ICCAD04"),
                ng45_dir=NG45_BENCHMARKS.get(name),
            )
        row = {
            key: float(result[key])
            for key in ("proxy_cost", "wirelength", "density", "congestion", "overlaps", "runtime")
        }
        row.update(benchmark=name, valid=bool(result["valid"]))
        rows.append(row)
        np.save(args.out / f"{name}.npy", result["placement"].detach().cpu().numpy())
        (args.out / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
        print("RESULT " + json.dumps(row), flush=True)
        assert row["valid"] and row["overlaps"] == 0, row


if __name__ == "__main__":
    main()
