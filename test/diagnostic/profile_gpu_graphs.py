"""Compare graph primitives on identical inputs; never install a production backend.

Run from the repository root with --help. Timings include warm medians and raw
samples; CUDA lifecycle measurements include packing, upload, and readback.
"""

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import numpy as np
from numba import njit, __version__ as numba_version
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from macro_place.evaluate import IBM_BENCHMARKS, NG45_BENCHMARKS
from macro_place.loader import load_benchmark_from_dir
from placer.local_search import clusters as clustering
from placer.local_search.hierarchy_model import HierarchyModel
from placer.local_search.location_graph import LocationAwareGraph
from placer.scoring.wirelength import _build_wl_cache
from utils import constants


def measure(fn, repeats, *, cuda=False):
    """Separate first-call cost from synchronized warm wall time."""
    samples = []
    for _ in range(repeats + 2):
        if cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        if cuda:
            torch.cuda.synchronize()
        samples.append(1000 * (time.perf_counter() - start))
    return {
        "first_ms": samples[0],
        "median_ms": statistics.median(samples[2:]),
        "samples_ms": samples[2:],
    }


def edge_inputs(plc, count, fanout, normalize):
    cache = _build_wl_cache(plc)
    modules = list(plc.hard_macro_indices) + list(plc.soft_macro_indices)
    lookup = np.full(len(plc.modules_w_pins), -1, dtype=np.int64)
    lookup[modules[:count]] = np.arange(count)
    return (
        cache["ref_idx"],
        cache["net_starts"],
        cache["net_lengths"],
        cache["net_weights"],
        lookup,
        count,
        fanout,
        normalize,
    )


def python_edges(refs, starts, lengths, weights, lookup, count, fanout, normalize):
    """Reference net-order accumulation, with distinct hard/macro projections."""
    edges = {}
    for start, length, weight in zip(starts, lengths, weights):
        if not 2 <= length <= fanout:
            continue
        ids = sorted({int(lookup[r]) for r in refs[start : start + length] if lookup[r] >= 0})
        weight = float(weight)
        if normalize:
            weight = max(0.0, weight) / max(len(ids) - 1, 1)
        for i, left in enumerate(ids):
            for right in ids[i + 1 :]:
                key = left * count + right
                old_count, old_weight = edges.get(key, (0, 0.0))
                edges[key] = old_count + 1, old_weight + weight
    keys = np.asarray(sorted(edges), dtype=np.int64)
    return (
        keys,
        np.asarray([edges[k][0] for k in keys], dtype=np.int64),
        np.asarray([edges[k][1] for k in keys], dtype=np.float64),
    )


@njit(cache=True)
def numba_edges(refs, starts, lengths, weights, lookup, count, fanout, normalize):
    # ponytail: bounded fanout scratch; use chunked emission for much larger netlists.
    capacity = len(starts) * fanout * (fanout - 1) // 2
    keys = np.empty(capacity, np.int64)
    values = np.empty(capacity, np.float64)
    ids = np.empty(fanout, np.int64)
    used = 0
    for net in range(len(starts)):
        length = lengths[net]
        if length < 2 or length > fanout:
            continue
        size = 0
        for pin in range(starts[net], starts[net] + length):
            node = lookup[refs[pin]]
            if node < 0:
                continue
            duplicate = False
            for j in range(size):
                if ids[j] == node:
                    duplicate = True
                    break
            if not duplicate:
                ids[size] = node
                size += 1
        weight = weights[net]
        if normalize:
            weight = max(0.0, weight) / max(size - 1, 1)
        for i in range(size):
            for j in range(i + 1, size):
                a, b = ids[i], ids[j]
                keys[used] = min(a, b) * count + max(a, b)
                values[used] = weight
                used += 1
    order = np.argsort(keys[:used], kind="mergesort")
    unique = np.empty(used, np.int64)
    counts = np.empty(used, np.int64)
    sums = np.empty(used, np.float64)
    size = 0
    for offset in order:
        key = keys[offset]
        if size == 0 or unique[size - 1] != key:
            unique[size] = key
            counts[size] = 1
            sums[size] = 0.0
            size += 1
        else:
            counts[size - 1] += 1
        sums[size - 1] += values[offset]
    return unique[:size], counts[:size], sums[:size]


def pack_edges(args):
    refs, starts, lengths, weights, lookup, count, fanout, normalize = args
    keep = (lengths >= 2) & (lengths <= fanout)
    offsets = np.arange(fanout)
    indices = starts[keep, None] + offsets[None, :]
    rows = np.full(indices.shape, -1, dtype=np.int64)
    valid = offsets[None, :] < lengths[keep, None]
    rows[valid] = lookup[refs[indices[valid]]]
    return rows, weights[keep], count, normalize


def numpy_edges(packed):
    rows, weights, count, normalize = packed
    rows = np.sort(rows, axis=1)
    unique = rows >= 0
    unique[:, 1:] &= rows[:, 1:] != rows[:, :-1]
    if normalize:
        weights = np.maximum(weights, 0) / np.maximum(unique.sum(axis=1) - 1, 1)
    left, right = np.triu_indices(rows.shape[1], 1)
    valid = unique[:, left] & unique[:, right]
    keys = (rows[:, left] * count + rows[:, right])[valid]
    values = np.broadcast_to(weights[:, None], valid.shape)[valid]
    order = np.argsort(keys, kind="stable")
    keys, values = keys[order], values[order]
    unique_keys, starts, counts = np.unique(keys, return_index=True, return_counts=True)
    sums = np.add.reduceat(values, starts) if starts.size else np.zeros(0)
    return unique_keys, counts, sums


def upload_edges(packed):
    rows, weights, count, normalize = packed
    return (
        torch.as_tensor(rows, device="cuda"),
        torch.as_tensor(weights, device="cuda"),
        count,
        normalize,
    )


def cuda_edges(packed):
    rows, weights, count, normalize = packed
    rows = rows.sort(dim=1).values
    unique = rows >= 0
    unique[:, 1:] &= rows[:, 1:] != rows[:, :-1]
    if normalize:
        weights = weights.clamp_min(0) / (unique.sum(dim=1) - 1).clamp_min(1)
    left, right = torch.triu_indices(rows.shape[1], rows.shape[1], 1, device="cuda")
    valid = unique[:, left] & unique[:, right]
    keys = (rows[:, left] * count + rows[:, right])[valid]
    values = weights[:, None].expand_as(valid)[valid]
    keys, inverse, counts = torch.unique(keys, return_inverse=True, return_counts=True)
    sums = torch.zeros(len(keys), dtype=torch.float64, device="cuda")
    sums.scatter_add_(0, inverse, values)
    return keys, counts, sums


def readback(values):
    return tuple(value.cpu().numpy() for value in values)


def edge_parity(reference, candidate):
    keys = np.array_equal(reference[0], candidate[0])
    counts = keys and np.array_equal(reference[1], candidate[1])
    weights = keys and np.array_equal(reference[2], candidate[2])
    error = float(np.max(np.abs(reference[2] - candidate[2]), initial=0)) if keys else None
    return {
        "keys_equal": keys,
        "counts_equal": counts,
        "weights_equal": weights,
        "max_weight_error": error,
        "exact": bool(keys and counts and weights),
    }


def pack_affinity(graph):
    """CSR follows original dictionary insertion order for the CPU reference."""
    cluster_ids = sorted(graph.clusters)
    owner_column = {cid: i for i, cid in enumerate(cluster_ids)}
    owners = np.asarray(
        [owner_column.get(node.cluster_id, -1) for node in graph.macros.values()], dtype=np.int64
    )
    offsets, neighbors, weights = [0], [], []
    for node in graph.macros.values():
        neighbors.extend(node.neighbors)
        weights.extend(node.neighbors.values())
        offsets.append(len(neighbors))
    # Match the same-parent eligibility in adjacent_cluster_transfer.
    from placer.local_search.adjacent_cluster_transfer import _cluster_parent

    pairs = [
        (edge.src, edge.dst)
        for edge in graph.active_edges
        if _cluster_parent(graph, edge.src) == _cluster_parent(graph, edge.dst)
    ]
    queries, groups = [], []
    for left, right in sorted(pairs):
        for source, destination in ((left, right), (right, left)):
            start = len(queries)
            queries.extend(
                (index, owner_column[source], owner_column[destination])
                for index in graph.clusters[source].members
            )
            groups.append((source, destination, start, len(queries)))
    return (
        np.asarray(offsets, dtype=np.int64),
        np.asarray(neighbors, dtype=np.int64),
        np.asarray(weights, dtype=np.float64),
        owners,
        np.asarray(queries, dtype=np.int64).reshape(-1, 3),
        len(cluster_ids),
    ), groups


def python_affinity(args):
    offsets, neighbors, weights, owners, queries, _ = args
    output = np.zeros((len(queries), 3))
    for q, (node, source, destination) in enumerate(queries):
        for edge in range(offsets[node], offsets[node + 1]):
            owner, weight = owners[neighbors[edge]], float(weights[edge])
            if owner == source:
                output[q, 0] += weight
            else:
                output[q, 1] += weight
                if owner == destination:
                    output[q, 2] += weight
    return output


@njit(cache=True)
def numba_affinity(offsets, neighbors, weights, owners, queries, cluster_count):
    by_owner = np.zeros((len(owners), cluster_count))
    external = np.zeros(len(owners))
    for node in range(len(owners)):
        for edge in range(offsets[node], offsets[node + 1]):
            owner = owners[neighbors[edge]]
            if owner >= 0:
                by_owner[node, owner] += weights[edge]
            if owner != owners[node]:
                external[node] += weights[edge]
    result = np.empty((len(queries), 3))
    for i in range(len(queries)):
        node, source, destination = queries[i]
        result[i, 0] = by_owner[node, source]
        result[i, 1] = external[node]
        result[i, 2] = by_owner[node, destination]
    return result


def numpy_affinity(args):
    offsets, neighbors, weights, owners, queries, cluster_count = args
    nodes = np.repeat(np.arange(len(owners)), np.diff(offsets))
    by_owner = np.zeros((len(owners), cluster_count))
    target_owner = owners[neighbors]
    owned = target_owner >= 0
    np.add.at(by_owner, (nodes[owned], target_owner[owned]), weights[owned])
    external = np.zeros(len(owners))
    outside = target_owner != owners[nodes]
    np.add.at(external, nodes[outside], weights[outside])
    node, source, destination = queries.T
    return np.column_stack((by_owner[node, source], external[node], by_owner[node, destination]))


def upload_affinity(args):
    offsets, neighbors, weights, owners, queries, cluster_count = args
    rows = np.repeat(np.arange(len(owners)), np.diff(offsets))
    indices = torch.as_tensor(np.stack((rows, neighbors)), device="cuda")
    values = torch.as_tensor(weights, device="cuda")
    adjacency = torch.sparse_coo_tensor(indices, values, (len(owners), len(owners))).coalesce()
    adjacency = adjacency.to_sparse_csr()
    labels = torch.as_tensor(owners, device="cuda")
    columns = torch.arange(cluster_count, device="cuda")
    membership = labels[:, None] == columns[None, :]
    masks = torch.cat((membership, ~membership), dim=1).to(torch.float64)
    return adjacency, masks, torch.as_tensor(queries, device="cuda"), cluster_count


def cuda_affinity(device):
    adjacency, masks, queries, cluster_count = device
    sums = torch.sparse.mm(adjacency, masks)
    node, source, destination = queries.T
    return torch.stack(
        (sums[node, source], sums[node, source + cluster_count], sums[node, destination]), dim=1
    )


def affinity_parity(graph, args, groups, reference, candidate):
    """Recompute the actual frontier score and compare its complete ordering."""
    reference_matches = True
    rank_equal = True
    for source, destination, start, end in groups:
        rows = graph.frontier_records(source, destination)
        by_id = {row["index"]: row for row in rows}
        ranked = []
        for q in range(start, end):
            index = int(args[4][q, 0])
            row = by_id[index]
            expected = [row["internal_weight"], row["external_weight"], row["destination_weight"]]
            reference_matches &= np.array_equal(reference[q], expected)
            internal, external, dest = map(float, candidate[q])
            score = (
                2.0 * max(0.0, dest - internal)
                + external
                + max(0.0, row["pressure_relief"])
                + row["facing"]
                + min(row["capacity_ratio"], 4.0) * 0.25
            )
            ranked.append((-score, -row["facing"], index))
        rank_equal &= [row["index"] for row in rows] == [r[2] for r in sorted(ranked)]
    exact = np.array_equal(reference, candidate)
    return {
        "reference_matches_production": bool(reference_matches),
        "weights_equal": exact,
        "rank_equal": bool(rank_equal),
        "exact": bool(exact and rank_equal and reference_matches),
        "max_weight_error": float(np.max(np.abs(reference - candidate), initial=0)),
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


def compiled_split(members, edge_weight, areas, *, min_size, incremental=False):
    """Keep Python's set iteration and tie order; compile only numeric reductions.

    Python's compensated sum and compiled additions can still differ. The
    diagnostic rejects unequal partitions/cuts rather than asserting equivalence.
    """
    members = np.asarray(members, dtype=np.int64)
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
    if incremental:
        accum = split_sums(matrix, members, np.array([seed]), np.asarray(list(right)))
        totals = {int(node): accum[i].copy() for i, node in enumerate(members)}
    while len(left) < len(members) - min_size and area < target:
        candidates = np.asarray(sorted(right), dtype=np.int64)
        sums = (
            np.asarray([totals[int(node)] for node in candidates])
            if incremental
            else split_sums(matrix, candidates, np.asarray(list(left)), np.asarray(list(right)))
        )
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
        if incremental:
            for other in right:
                totals[other][0] += matrix[other, node]
                totals[other][1] -= matrix[other, node]
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


def construction_report(args, backends, repeats):
    reference = python_edges(*args)
    packed = pack_edges(args)
    methods = {
        "python": lambda: python_edges(*args),
        "numpy": lambda: numpy_edges(pack_edges(args)),
        "numba": lambda: numba_edges(*args),
    }
    report = {"edges": len(reference[0]), "pack": measure(lambda: pack_edges(args), repeats)}
    if "cuda" in backends:
        device = upload_edges(packed)
        methods["cuda"] = lambda: readback(cuda_edges(upload_edges(pack_edges(args))))
        report["cuda_upload"] = measure(lambda: upload_edges(packed), repeats, cuda=True)
        report["cuda_resident"] = measure(lambda: cuda_edges(device), repeats, cuda=True)
        result = cuda_edges(device)
        report["cuda_readback"] = measure(lambda: readback(result), repeats, cuda=True)
    report["backends"] = {}
    for name in backends:
        fn = methods[name]
        timing = measure(fn, repeats, cuda=name == "cuda")
        checks = [edge_parity(reference, fn()) for _ in range(3)]
        report["backends"][name] = {"timing": timing, "parity": checks}
    return report


def benchmark_report(name, backends, repeats):
    if "cuda" in backends:
        torch.cuda.reset_peak_memory_stats()
    path = (
        ROOT / NG45_BENCHMARKS[name]
        if name in NG45_BENCHMARKS
        else ROOT / "external/MacroPlacement/Testcases/ICCAD04" / name
    )
    with contextlib.redirect_stdout(io.StringIO()):
        benchmark, plc = load_benchmark_from_dir(str(path))
    n = benchmark.num_hard_macros
    positions = benchmark.macro_positions.numpy().astype(np.float64)
    sizes = benchmark.macro_sizes.numpy().astype(np.float64)
    original = clustering._balanced_graph_split
    splits = []

    def capture(members, edges, areas, *, min_size):
        result = original(members, edges, areas, min_size=min_size)
        splits.append((members.copy(), dict(edges), areas.copy(), min_size, result))
        return result

    clustering._balanced_graph_split = capture
    start = time.perf_counter()
    try:
        hierarchy = HierarchyModel.build(plc, n, benchmark.num_soft_macros, hard_sizes=sizes[:n])
    finally:
        clustering._balanced_graph_split = original
    hierarchy_ms = 1000 * (time.perf_counter() - start)
    fanout = max(hierarchy.max_fanout, constants.HIER_SOFT_ROLE_MAX_FANOUT)
    graph = LocationAwareGraph.build(plc, hierarchy, positions, sizes, max_fanout=fanout)
    graph.analyze_state(
        np.zeros((benchmark.grid_rows, benchmark.grid_cols)),
        benchmark.canvas_width,
        benchmark.canvas_height,
    )
    state_before = graph.to_visualizer_payload()
    report = {
        "benchmark": name,
        "graph": graph.summary(),
        "hierarchy_capture_ms": hierarchy_ms,
        "input_sha256": {},
        "construction": {},
        "splits": [],
    }
    for filename in ("netlist.pb.txt", "initial.plc"):
        report["input_sha256"][filename] = hashlib.sha256(
            (path / filename).read_bytes()
        ).hexdigest()
    for projection, count, limit, normalize in (
        ("hard", n, hierarchy.max_fanout, False),
        ("macro", len(positions), fanout, True),
    ):
        args = edge_inputs(plc, count, limit, normalize)
        report["construction"][projection] = construction_report(args, backends, repeats)
        ref = python_edges(*args)
        if projection == "hard":
            counts, weights = clustering._hard_edge_maps(plc, n, limit)
        else:
            weights = {
                (i, j): weight
                for i, node in graph.macros.items()
                for j, weight in node.neighbors.items()
                if i < j
            }
        keys = np.asarray([a * count + b for a, b in sorted(weights)], dtype=np.int64)
        values = np.asarray([weights[pair] for pair in sorted(weights)])
        assert np.array_equal(ref[0], keys) and np.array_equal(ref[2], values)
        if projection == "hard":
            assert np.array_equal(ref[1], [counts[pair] for pair in sorted(weights)])
        report["construction"][projection]["reference_matches_production"] = True

    args, groups = pack_affinity(graph)
    reference = python_affinity(args)
    methods = {
        "python": lambda: python_affinity(pack_affinity(graph)[0]),
        "numpy": lambda: numpy_affinity(pack_affinity(graph)[0]),
        "numba": lambda: numba_affinity(*pack_affinity(graph)[0]),
    }
    affinity = {
        "queries": len(args[4]),
        "directed_pairs": len(groups),
        "pack": measure(lambda: pack_affinity(graph), repeats),
        "backends": {},
    }
    if "cuda" in backends:
        device = upload_affinity(args)
        methods["cuda"] = (
            lambda: cuda_affinity(upload_affinity(pack_affinity(graph)[0])).cpu().numpy()
        )
        affinity["cuda_upload_and_sparse_build"] = measure(
            lambda: upload_affinity(args), repeats, cuda=True
        )
        affinity["cuda_resident"] = measure(lambda: cuda_affinity(device), repeats, cuda=True)
        values = cuda_affinity(device)
        affinity["cuda_readback"] = measure(lambda: values.cpu().numpy(), repeats, cuda=True)
        affinity["resident_reuse"] = {}
        # Repeated identical queries are synthetic: CPU can cache their answer too.
        for batches in (1, 16, 64):

            def cpu_reuse():
                packed = pack_affinity(graph)[0]
                return [numba_affinity(*packed) for _ in range(batches)]

            def gpu_reuse():
                device = upload_affinity(pack_affinity(graph)[0])
                return torch.stack([cuda_affinity(device) for _ in range(batches)]).cpu().numpy()

            def cpu_cached():
                values = numba_affinity(*pack_affinity(graph)[0])
                return [values] * batches

            affinity["resident_reuse"][str(batches)] = {
                "numba": measure(cpu_reuse, repeats),
                "numba_cached": measure(cpu_cached, repeats),
                "cuda": measure(gpu_reuse, repeats, cuda=True),
            }
    for backend in backends:
        fn = methods[backend]
        affinity["backends"][backend] = {
            "timing": measure(fn, repeats, cuda=backend == "cuda"),
            "parity": [affinity_parity(graph, args, groups, reference, fn()) for _ in range(3)],
        }
    report["affinity"] = affinity
    leaves = [
        np.asarray(cluster.hard_members, dtype=np.int64)
        for cluster in graph.clusters.values()
        if cluster.hard_members
    ]
    report["directional_cpu"] = measure(
        lambda: [
            graph.directional_graph_profile(members[:1], members, max_hops=3, decay=0.55)
            for members in leaves
        ],
        repeats,
    )
    for members, edges, areas, minimum, expected in splits:
        row = {"members": len(members), "backends": {}}
        for mode in ("python", "compiled", "incremental"):
            fn = (
                (lambda: original(members, edges, areas, min_size=minimum))
                if mode == "python"
                else (
                    lambda: compiled_split(
                        members, edges, areas, min_size=minimum, incremental=mode == "incremental"
                    )
                )
            )
            row["backends"][mode] = {
                "timing": measure(fn, repeats),
                "exact": split_equal(expected, fn()),
            }
        report["splits"].append(row)
    assert graph.active_edges is hierarchy.edges
    assert graph.to_visualizer_payload() == state_before
    assert np.array_equal(benchmark.macro_positions.numpy(), positions)
    report["read_only_verified"] = True
    report["snapshot"] = "initial.plc geometry, zero heat, current hierarchy ownership"
    if "cuda" in backends:
        report["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    report["cuda_gate"] = []
    for workload, result in [*report["construction"].items(), ("affinity", affinity)]:
        rows = result["backends"]
        if "cuda" not in rows or "numba" not in rows:
            continue
        speedup = rows["numba"]["timing"]["median_ms"] / rows["cuda"]["timing"]["median_ms"]
        parity = all(check["exact"] for check in rows["cuda"]["parity"])
        report["cuda_gate"].append(
            {
                "workload": workload,
                "speedup_over_numba": speedup,
                "exact": parity,
                "advance": parity and speedup >= 1.25,
            }
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", default=["ibm10", "ibm17", "nvdla"])
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=("python", "numpy", "numba", "cuda"),
        default=["python", "numpy", "numba", "cuda"],
    )
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if not set(args.names) <= set(IBM_BENCHMARKS) | set(NG45_BENCHMARKS):
        parser.error("names must be known IBM or NG45 benchmarks")
    if "cuda" in args.backends and not torch.cuda.is_available():
        parser.error("CUDA unavailable; select --backends python numpy numba")
    args.out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    if "cuda" in args.backends:
        torch.empty(1, device="cuda")
        torch.cuda.synchronize()
    context_ms = 1000 * (time.perf_counter() - start)
    paths = [
        Path(__file__),
        ROOT / "src/placer/local_search/clusters.py",
        ROOT / "src/placer/local_search/location_graph.py",
    ]
    result = {
        "schema_version": 1,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "numba": numba_version,
        "torch_cpu_threads": torch.get_num_threads(),
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if "cuda" in args.backends else "cpu",
        "cuda_context_ms": context_ms,
        "repeats": args.repeats,
        "backends": args.backends,
        "scope": "offline primitives; advance means eligible for a later placement comparison",
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        },
        "revision": subprocess.check_output(
            ["rtk", "proxy", "git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "benchmarks": [],
    }
    source_hash = hashlib.sha256()
    for path in sorted((ROOT / "src").rglob("*.py")):
        source_hash.update(str(path.relative_to(ROOT)).encode() + b"\0" + path.read_bytes())
    result["src_tree_sha256"] = source_hash.hexdigest()
    for name in args.names:
        row = benchmark_report(name, args.backends, args.repeats)
        result["benchmarks"].append(row)
        (args.out / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"benchmark": name, "cuda_gate": row["cuda_gate"]}), flush=True)


if __name__ == "__main__":
    main()
