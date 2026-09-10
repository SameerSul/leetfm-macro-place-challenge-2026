"""Replay archived source and isolated pass removals using the existing evaluator runner."""

import argparse
import ast
import importlib.abc
import importlib.machinery
import inspect
import os
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]


def source_overlay(directory, ablation=None):
    """Compile archived source with original filenames so runtime paths stay valid."""
    class Loader(importlib.machinery.SourceFileLoader):
        def get_code(self, fullname):
            original = Path(self.path)
            archived = directory / original.relative_to(ROOT)
            source = archived.read_bytes()
            if ablation and original.name == "hierarchy_floorplan.py":
                tree = ast.parse(source)
                function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                                and n.name == "run_hierarchy_floorplan")
                start_name, pass_name = (
                    ("deep_before", "deep_cluster_internal") if ablation == "deep"
                    else ("consolidation_before", "small_cluster_consolidation")
                )
                start = next(i for i, n in enumerate(function.body)
                             if isinstance(n, ast.Assign)
                             and any(isinstance(t, ast.Name) and t.id == start_name
                                     for t in n.targets))
                end = next(i for i, n in enumerate(function.body[start:], start)
                           if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                           and isinstance(n.value.func, ast.Name)
                           and n.value.func.id == "_record_plateau"
                           and isinstance(n.value.args[0], ast.Constant)
                           and n.value.args[0].value == pass_name)
                del function.body[start:end + 1]
                source = tree
            return compile(source, str(original), "exec")

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec and spec.origin:
                original = Path(spec.origin)
                if original.is_relative_to(ROOT / "src") and original.suffix == ".py":
                    spec.loader = Loader(fullname, str(original))
                    return spec
            return None

    sys.meta_path.insert(0, Finder())


def capture_final_inputs(directory):
    from placer.pipeline.segments import floorplan_post_coldspot as module

    original = module._final_free_soft_relief
    signature = inspect.signature(original)
    keys = {
        "hierarchy_contract_limits", "seed_island_limits", "subhierarchy_contract_active",
        "subcluster_hierarchy_contract_limits", "parent_hierarchy_contract_limits",
    }

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        values = signature.bind(*args, **kwargs).arguments
        benchmark = values["benchmark"]
        limits, seen = {}, set()

        def visit(function):
            if id(function) in seen or not inspect.isfunction(function):
                return
            seen.add(id(function))
            for key, value in inspect.getclosurevars(function).nonlocals.items():
                if key in keys:
                    limits[key] = value
                elif inspect.isfunction(value):
                    visit(value)

        visit(values["candidate_allowed"])
        name = str(getattr(benchmark, "_hierarchy_trace_name", benchmark.name))
        payload = dict(
            hard=values["hard"].copy(), soft=result[0].copy(),
            hierarchy=values["hierarchy"], region=values.get("region_bbox"),
            hard_region=inspect.currentframe().f_back.f_locals["region"].copy(),
            hard_quality_limit=inspect.currentframe().f_back.f_locals["audit_limit"],
            limits=limits, proxy=float(result[2]),
        )
        (directory / f"{name}_final.pkl").write_bytes(pickle.dumps(payload))
        return result

    module._final_free_soft_relief = capture


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--ablate", choices=("deep", "consolidation"))
    parser.add_argument("--capture-final", action="store_true")
    args, remaining = parser.parse_known_args()
    if args.source:
        source_overlay(args.source.resolve(), args.ablate)
    elif args.ablate:
        parser.error("--ablate requires --source")
    if args.capture_final:
        out = Path(remaining[remaining.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        capture_final_inputs(out)
    if args.source:
        import hashlib
        digest = hashlib.sha256()
        for path in sorted((args.source / "src").rglob("*.py")):
            digest.update(str(path.relative_to(args.source)).encode())
            digest.update(path.read_bytes())
        os.environ["VIVAPLACE_WORKTREE_FINGERPRINT"] = digest.hexdigest()[:16]
    from test.diagnostic.run_speed_comparison import main as run
    sys.argv = [sys.argv[0], *remaining]
    run()


if __name__ == "__main__":
    main()
