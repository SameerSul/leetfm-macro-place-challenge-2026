"""Paired kernel parity and timing on saved production placements."""

import json
import statistics
import time
import numpy as np
import run_speed_comparison as run
from placer import _patch_plc_wirelength, _patch_plc_density, _patch_plc_congestion, _exact_proxy
from placer.scoring.incremental import IncrementalScorer
from placer.local_search import hierarchy_quality as hq, cluster_void_relocation as void
from placer.local_search.hierarchy_model import HierarchyModel
from macro_place.loader import load_benchmark_from_dir

old_group = run.reference_group()
old_hq = run.reference_module("src/placer/local_search/hierarchy_quality.py")
old_void = run.reference_module("src/placer/local_search/cluster_void_relocation.py")
rows = []
for name in ("ibm04", "ibm10", "ibm12"):
    bench, plc = load_benchmark_from_dir(f"external/MacroPlacement/Testcases/ICCAD04/{name}")
    _patch_plc_wirelength(plc)
    _patch_plc_congestion(plc)
    _patch_plc_density(plc)
    _exact_proxy(bench.macro_positions, bench, plc)
    pos = np.load(run.ROOT / f"ml_data/proxy_individual/20260908/12_ibm/{name}.npy").astype(
        np.float64
    )
    scorer = IncrementalScorer(plc, bench, pos)
    n = bench.num_hard_macros
    sizes = bench.macro_sizes.numpy().astype(np.float64)
    model = HierarchyModel.build(plc, n, bench.num_soft_macros, hard_sizes=sizes[:n])
    metric_args = (
        pos[:n],
        pos[n:],
        model.clusters,
        model.cluster_softs,
        sizes[:n],
        bench.canvas_width,
        bench.canvas_height,
    )
    full = old_hq.hierarchy_island_metrics(*metric_args)
    fast = hq.hierarchy_island_metrics(*metric_args, diagnostics=False)
    assert fast == {
        cid: {key: row[key] for key in hq.HIERARCHY_ISLAND_METRICS} for cid, row in full.items()
    }
    assert hq.hierarchy_island_metrics(*metric_args) == full
    metric_times = [[], []]
    for _ in range(5):
        for variant in (0, 1):
            start = time.perf_counter()
            if variant == 0:
                old_hq.hierarchy_island_metrics(*metric_args)
            else:
                hq.hierarchy_island_metrics(*metric_args, diagnostics=False)
            metric_times[variant].append(time.perf_counter() - start)
    hw, hh = sizes[:n, 0] / 2, sizes[:n, 1] / 2
    args = (pos[:n], hw, hh)
    kwargs = dict(
        min_width=2 * bench.canvas_width / bench.grid_cols,
        min_height=2 * bench.canvas_height / bench.grid_rows,
        max_voids=96,
    )
    references = [
        old_void.find_large_macro_voids(
            *args, canvas_width=bench.canvas_width, canvas_height=bench.canvas_height, **kwargs
        ),
        old_void.find_large_macro_voids(*args, subtract_blockages=False, **kwargs),
    ]
    bases = void._large_macro_void_bases(*args, bench.canvas_width, bench.canvas_height, 70.0)
    candidates = [
        void._select_macro_voids(bases, *args, **kwargs, subtract_blockages=True),
        void._select_macro_voids(
            [b for b in bases if b["kind"] == "interior"], *args, **kwargs, subtract_blockages=False
        ),
    ]
    for reference, candidate in zip(references, candidates):
        assert len(reference) == len(candidate)
        for left, right in zip(reference, candidate):
            for key in left:
                np.testing.assert_array_equal(left[key], right[key])
    target_times = [[], []]
    for index in range(min(12, bench.num_soft_macros)):
        indices = np.arange(index, min(index + 4, bench.num_soft_macros))
        start = time.perf_counter()
        reference = old_void._routing_target(plc, pos[:n], pos[n:], indices)
        target_times[0].append(time.perf_counter() - start)
        start = time.perf_counter()
        candidate = void._routing_target(scorer, pos[:n], pos[n:], indices)
        target_times[1].append(time.perf_counter() - start)
        np.testing.assert_array_equal(candidate, reference)
    rng = np.random.default_rng(901)
    group_times = [[], []]
    by_size = {}
    max_error = 0.0
    # Warm both implementations before timed comparisons.
    modules = [int(scorer.soft_indices[0])]
    old = pos[n : n + 1].copy()
    for function in (old_group, IncrementalScorer._score_multi_move):
        function(scorer, modules, old, old + 0.01, [])
    for trial in range(60):
        nh = (0, 1, 2, 8)[trial % 4]
        ns = (1, 4, 8, 16)[trial % 4]
        hi = rng.choice(n, nh, replace=False)
        si = rng.choice(bench.num_soft_macros, ns, replace=False)
        modules = [int(scorer.hard_indices[i]) for i in hi] + [
            int(scorer.soft_indices[i]) for i in si
        ]
        old = np.vstack([scorer.committed_hard_pos[hi], scorer.committed_soft_pos[si]])
        new = old + rng.uniform(-0.05, 0.05, size=old.shape) * [
            bench.canvas_width,
            bench.canvas_height,
        ]
        slots = scorer._hard_slot_array(*hi)
        scorer._route_struct_many(modules)
        density = scorer.grid_occupied.copy()
        state = {
            key: getattr(scorer, key).copy()
            for key in (
                "H_flat",
                "V_flat",
                "H_macro_flat",
                "V_macro_flat",
                "H_smoothed",
                "V_smoothed",
                "committed_hard_pos",
                "committed_soft_pos",
            )
        }
        scores = [None, None]
        for variant in ((trial // 4) % 2, 1 - (trial // 4) % 2):
            scorer.grid_occupied[:] = density
            function = (old_group, IncrementalScorer._score_multi_move)[variant]
            start = time.perf_counter()
            scores[variant] = function(scorer, modules, old, new, slots)
            elapsed = time.perf_counter() - start
            group_times[variant].append(elapsed)
            by_size.setdefault(len(modules), [[], []])[variant].append(elapsed)
            for key, before in state.items():
                np.testing.assert_array_equal(getattr(scorer, key), before)
            np.testing.assert_allclose(scorer.grid_occupied, density, rtol=0.0, atol=1.0e-12)
            for module, xy in zip(modules, old):
                np.testing.assert_array_equal(plc.modules_w_pins[module].get_pos(), xy)
        max_error = max(max_error, abs(scores[0] - scores[1]))
        assert abs(scores[0] - scores[1]) < 1.0e-12, (name, trial, scores)
        scorer.grid_occupied[:] = density
        if trial % 10 == 9:
            scorer.commit_move_group(hi, new[:nh], si, new[nh:])
    row = dict(
        benchmark=name,
        max_group_error=max_error,
        metrics_median_s=[statistics.median(t) for t in metric_times],
        routing_targets_s=[sum(t) for t in target_times],
        group_median_s=[statistics.median(t) for t in group_times],
        group_total_s=[sum(t) for t in group_times],
        by_size={size: [sum(t) for t in ts] for size, ts in by_size.items()},
    )
    rows.append(row)
    print("KERNEL " + json.dumps(row), flush=True)
    (run.HERE / "kernels.json").write_text(json.dumps(rows, indent=2) + "\n")
