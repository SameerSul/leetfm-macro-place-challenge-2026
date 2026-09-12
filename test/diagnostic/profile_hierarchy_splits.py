"""Replay captured hierarchy splits with the promising compiled CPU control.

This diagnostic never installs a production backend. Equality of every captured
partition and cut ratio is required before reporting the timing comparison.
"""

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import platform
import time
import statistics

import numpy as np
from numba import njit, __version__ as numba_version
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from macro_place.evaluate import IBM_BENCHMARKS, NG45_BENCHMARKS
from macro_place.loader import load_benchmark_from_dir
from placer.local_search import clusters as clustering
from placer.local_search.hierarchy_model import HierarchyModel


def measure(fn, repeats):
    """Separate first-call cost from warm wall time."""
    samples = []
    for _ in range(repeats + 2):
        start = time.perf_counter()
        fn()
        samples.append(1000 * (time.perf_counter() - start))
    return {
        "first_ms": samples[0],
        "median_ms": statistics.median(samples[2:]),
        "samples_ms": samples[2:],
    }


@njit(cache=True)
def split_sums(matrix, candidates, left, right):
    result = np.zeros((len(candidates), 2))
    for i, node in enumerate(candidates):
        for other in left:
            result[i, 0] += matrix[node, other]
        for other in right:
            if other != node:
                result[i, 1] += matrix[node, other]
    return result


def compiled_split(members, edge_weight, areas, *, min_size):
    """Keep Python's set iteration and tie order; compile only numeric reductions.

    Python's compensated sum and compiled additions can still differ. The
    diagnostic rejects unequal partitions/cuts rather than asserting equivalence.
    """
    members = np.asarray(members, dtype=np.int64)
    # ponytail: dense O(N²) scratch; use sparse rows if larger designs exhaust memory.
    matrix = np.zeros((len(areas), len(areas)))
    for (a, b), weight in edge_weight.items():
        matrix[a, b] = matrix[b, a] = weight
    degree = {}
    total = 0.0
    for i in members:
        for j in members:
            if j <= i or matrix[i, j] <= 0:
                continue
            weight = float(matrix[i, j])
            degree[i] = degree.get(i, 0.0) + weight
            degree[j] = degree.get(j, 0.0) + weight
            total += weight
    if total <= 0 or len(members) < 2 * min_size:
        return None
    target = 0.5 * float(np.sum(areas[members]))
    seed = int(max(members, key=lambda x: (degree.get(int(x), 0.0), areas[int(x)], -int(x))))
    left = {seed}
    right = set({int(x) for x in members})
    right.remove(seed)
    area = float(areas[seed])
    while len(left) < len(members) - min_size and area < target:
        candidates = np.asarray(sorted(right), dtype=np.int64)
        sums = split_sums(matrix, candidates, np.asarray(list(left)), np.asarray(list(right)))
        best = None
        for node, (to_left, to_right) in zip(candidates, sums):
            penalty = abs((area + float(areas[node])) - target) / max(target, 1.0)
            score = float(to_left) - 0.35 * float(to_right) - 0.05 * penalty
            row = (score, degree.get(node, 0.0), -penalty, -int(node), int(node))
            if best is None or row > best:
                best = row
        if best is None:
            break
        node = best[-1]
        left.add(node)
        right.remove(node)
        area += float(areas[node])
    if len(left) < min_size or len(right) < min_size:
        return None
    left, right = np.asarray(sorted(left)), np.asarray(sorted(right))
    cut = 0.0
    for a in left:
        for b in right:
            cut += float(matrix[a, b])
    return left, right, float(cut / max(total, 1e-12))


def split_equal(reference, candidate):
    if reference is None or candidate is None:
        return reference is None and candidate is None
    return all(np.array_equal(a, b) for a, b in zip(reference, candidate))


def benchmark_report(name, repeats):
    path = (
        ROOT / NG45_BENCHMARKS[name]
        if name in NG45_BENCHMARKS
        else ROOT / "external/MacroPlacement/Testcases/ICCAD04" / name
    )
    with contextlib.redirect_stdout(io.StringIO()):
        benchmark, plc = load_benchmark_from_dir(str(path))
    n = benchmark.num_hard_macros
    positions = benchmark.macro_positions.numpy().copy()
    sizes = benchmark.macro_sizes.numpy().astype(np.float64)
    original = clustering._balanced_graph_split
    splits = []

    def capture(members, edges, areas, *, min_size):
        result = original(members, edges, areas, min_size=min_size)
        splits.append((members.copy(), dict(edges), areas.copy(), min_size, result))
        return result

    clustering._balanced_graph_split = capture
    try:
        HierarchyModel.build(plc, n, benchmark.num_soft_macros, hard_sizes=sizes[:n])
    finally:
        clustering._balanced_graph_split = original
    report = dict(
        benchmark=name,
        splits=[],
        input_sha256={
            filename: hashlib.sha256((path / filename).read_bytes()).hexdigest()
            for filename in ("netlist.pb.txt", "initial.plc")
        },
    )
    for members, edges, areas, minimum, expected in splits:
        row = {"members": len(members), "backends": {}}
        for mode, function in (("python", original), ("compiled", compiled_split)):

            def run():
                return function(members, edges, areas, min_size=minimum)

            exact = split_equal(expected, run())
            assert exact, (name, mode, len(members))
            row["backends"][mode] = dict(timing=measure(run, repeats), exact=exact)
        report["splits"].append(row)
    assert np.array_equal(positions, benchmark.macro_positions.numpy())
    report["read_only_verified"] = True
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", default=["ibm10", "ibm17", "nvdla"])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if not set(args.names) <= set(IBM_BENCHMARKS) | set(NG45_BENCHMARKS):
        parser.error("names must be known IBM or NG45 benchmarks")
    args.out.mkdir(parents=True, exist_ok=False)
    paths = [
        Path(__file__),
        ROOT / "src/placer/local_search/clusters.py",
        ROOT / "src/placer/local_search/hierarchy_model.py",
        ROOT / "src/utils/constants.py",
    ]
    result = dict(
        schema_version=1,
        python=platform.python_version(),
        numpy=np.__version__,
        numba=numba_version,
        repeats=args.repeats,
        scope="offline CPU splits; no production speedup claim",
        source_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        },
        benchmarks=[],
    )
    for name in args.names:
        row = benchmark_report(name, args.repeats)
        result["benchmarks"].append(row)
        (args.out / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(
            json.dumps(dict(benchmark=name, captured_splits=len(row["splits"]), exact=True)),
            flush=True,
        )


if __name__ == "__main__":
    main()
