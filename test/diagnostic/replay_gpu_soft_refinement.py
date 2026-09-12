"""Offline coordinated soft refinement with resident PyTorch CPU/CUDA tensors.

Uses smooth wirelength, rectangular density, and a differentiable RUDY surrogate.
Every returned candidate passes fresh float32 exact scoring and the saved contract.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from dreamplace_rudy import blockage_map
from replay_cluster_tiles import contract_for, load
from macro_place.objective import compute_proxy_cost
from macro_place.utils import validate_placement
from placer.local_search.cluster_tile_rearrange import _leaf_units
from placer.plc.placement import _ensure_pos_cache
from placer.scoring.exact import _exact_proxy
from placer.scoring.incremental import IncrementalScorer
from placer.scoring.wirelength import _build_wl_cache


def movable_bounds(data, benchmark, baseline, max_shift):
    """Freeze hard/fixed/bridge/bundled macros; bound owned singleton soft moves."""
    n = benchmark.num_hard_macros
    units = _leaf_units(data["hierarchy"], ~benchmark.macro_fixed.numpy(), n)
    indices = sorted(
        int(unit[0]) for rows in units.values() for unit in rows if len(unit) == 1 and unit[0] >= n
    )
    indices = np.asarray(indices, dtype=np.int64)
    canvas = np.array([benchmark.canvas_width, benchmark.canvas_height])
    cell = canvas / np.array([benchmark.grid_cols, benchmark.grid_rows])
    sizes = benchmark.macro_sizes.numpy()
    region = np.asarray(data["region"])[indices - n]
    lower = np.maximum(region[:, :2], sizes[indices] / 2)
    upper = np.minimum(region[:, 2:], canvas - sizes[indices] / 2)
    lo = np.maximum((lower - baseline[indices]) / cell, -max_shift)
    hi = np.minimum((upper - baseline[indices]) / cell, max_shift)
    valid = np.all((lo <= 0) & (hi >= 0), axis=1)
    return indices[valid], lo[valid], hi[valid]


def net_terms(pins, lengths, pin_net, weights, gamma, x_edges, y_edges, capacities):
    """Smooth HPWL and directional rectangular routing demand on physical nets."""
    maximum = torch.segment_reduce(pins, "max", lengths=lengths, unsafe=True)
    minimum = torch.segment_reduce(pins, "min", lengths=lengths, unsafe=True)
    high = maximum.detach()
    low = minimum.detach()
    pos_exp = torch.exp((pins - high[pin_net]) / gamma)
    neg_exp = torch.exp((low[pin_net] - pins) / gamma)
    smooth = (
        high
        - low
        + gamma
        * (
            torch.log(torch.segment_reduce(pos_exp, "sum", lengths=lengths, unsafe=True))
            + torch.log(torch.segment_reduce(neg_exp, "sum", lengths=lengths, unsafe=True))
        )
    )
    wl = (smooth.sum(1) * weights).sum()
    # ponytail: rectangular routes are a surrogate; exact evaluator gates all output.
    cell = torch.stack((x_edges[1] - x_edges[0], y_edges[1] - y_edges[0]))
    center = (maximum + minimum) / 2
    span = torch.maximum(maximum - minimum, cell / 4)
    lower, upper = center - span / 2, center + span / 2
    dx = (
        torch.minimum(upper[:, 0, None], x_edges[None, 1:])
        - torch.maximum(lower[:, 0, None], x_edges[None, :-1])
    ).clamp_min(0)
    dy = (
        torch.minimum(upper[:, 1, None], y_edges[None, 1:])
        - torch.maximum(lower[:, 1, None], y_edges[None, :-1])
    ).clamp_min(0)
    horizontal = dx.T @ (dy * (weights / span[:, 1])[:, None]) / cell.prod() / capacities[0]
    vertical = (dx * (weights / span[:, 0])[:, None]).T @ dy / cell.prod() / capacities[1]
    return wl, horizontal, vertical


def smoothing_matrix(count, radius, device):
    indices = torch.arange(count, device=device)
    matrix = ((indices[:, None] - indices[None, :]).abs() <= radius).double()
    return matrix / matrix.sum(0)


def propose(data, benchmark, plc, baseline, device, steps=24, max_shift=0.5):
    """Optimize one fixed snapshot and download four completed candidates together."""
    started = time.perf_counter()
    indices, lower, upper = movable_bounds(data, benchmark, baseline, max_shift)
    if not len(indices):
        return [], dict(eligible=0, total_s=time.perf_counter() - started)
    _exact_proxy(torch.from_numpy(baseline), benchmark, plc)
    scorer = IncrementalScorer(plc, benchmark, baseline)
    cache = _build_wl_cache(plc)
    assert np.all(cache["net_lengths"] > 0)
    assert int(cache["net_lengths"].sum()) == len(cache["ref_idx"])
    module_positions = _ensure_pos_cache(plc).copy()
    modules = np.array(list(benchmark.hard_macro_indices) + list(benchmark.soft_macro_indices))
    pack_s = time.perf_counter() - started

    def tensor(value, dtype=torch.float64):
        return torch.as_tensor(value, dtype=dtype, device=device)

    base, sizes = tensor(baseline), tensor(benchmark.macro_sizes.numpy())
    canvas = tensor([benchmark.canvas_width, benchmark.canvas_height])
    cell = canvas / tensor([benchmark.grid_cols, benchmark.grid_rows])
    selected = tensor(indices, torch.long)
    modules_t = tensor(modules, torch.long)
    reference = tensor(module_positions)
    refs = tensor(cache["ref_idx"], torch.long)
    offsets = tensor(np.column_stack([cache["x_off"], cache["y_off"]]))
    lengths = tensor(cache["net_lengths"], torch.long)
    pin_net = tensor(cache["pin_to_net"], torch.long)
    weights = tensor(cache["net_weights"])
    x_edges = torch.linspace(
        0,
        float(benchmark.canvas_width),
        benchmark.grid_cols + 1,
        device=device,
        dtype=torch.float64,
    )
    y_edges = torch.linspace(
        0,
        float(benchmark.canvas_height),
        benchmark.grid_rows + 1,
        device=device,
        dtype=torch.float64,
    )
    capacities = tensor([plc.hroutes_per_micron, plc.vroutes_per_micron])
    lo, hi = tensor(lower), tensor(upper)
    n = benchmark.num_hard_macros
    occupancy = blockage_map(base[:n] - sizes[:n] / 2, base[:n] + sizes[:n] / 2, x_edges, y_edges)
    hblock = occupancy * plc.hrouting_alloc / plc.hroutes_per_micron
    vblock = occupancy * plc.vrouting_alloc / plc.vroutes_per_micron
    sx = smoothing_matrix(benchmark.grid_cols, plc.smooth_range, device)
    sy = smoothing_matrix(benchmark.grid_rows, plc.smooth_range, device)
    gamma = float(min(scorer.grid_w, scorer.grid_h)) / 4
    shift = torch.zeros((len(indices), 2), device=device, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([shift], lr=0.03)
    if device == "cuda":
        torch.cuda.synchronize()
    setup_s = time.perf_counter() - started - pack_s
    optimize_start = time.perf_counter()
    candidates = []
    checkpoints = {max(1, steps // 6), max(1, steps // 3), max(1, steps * 2 // 3), steps}
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        positions = base.index_copy(0, selected, base[selected] + shift * cell)
        locations = reference.index_copy(0, modules_t, positions)
        pins = locations[refs] + offsets
        wl, horizontal, vertical = net_terms(
            pins,
            lengths,
            pin_net,
            weights,
            gamma,
            x_edges,
            y_edges,
            capacities,
        )
        density = blockage_map(positions - sizes / 2, positions + sizes / 2, x_edges, y_edges)
        congestion = torch.cat(
            ((horizontal @ sy.T + hblock).flatten(), (sx @ vertical + vblock).flatten())
        )
        loss = (
            wl / scorer.wl_normalizer
            + 0.25 * density.flatten().topk(max(1, density.numel() // 10)).values.mean()
            + 0.5 * congestion.topk(max(1, congestion.numel() // 20)).values.mean()
        )
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            shift.copy_(torch.maximum(lo, torch.minimum(shift, hi)))
            if step in checkpoints:
                candidates.append(base.index_copy(0, selected, base[selected] + shift * cell))
    if device == "cuda":
        torch.cuda.synchronize()
    optimize_s = time.perf_counter() - optimize_start
    download_start = time.perf_counter()
    candidates = torch.stack(candidates).cpu().numpy().astype(np.float32)
    if not np.isfinite(candidates).all():
        raise ValueError("non-finite refinement candidate")
    frozen = np.ones(len(baseline), dtype=bool)
    frozen[indices] = False
    np.testing.assert_array_equal(
        candidates[:, frozen], np.broadcast_to(baseline[frozen], candidates[:, frozen].shape)
    )
    return candidates, dict(
        eligible=len(indices),
        steps=steps,
        pack_s=pack_s,
        setup_s=setup_s,
        optimize_s=optimize_s,
        download_s=time.perf_counter() - download_start,
        total_s=time.perf_counter() - started,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--devices", nargs="+", choices=("cpu", "cuda"), default=["cpu", "cuda"])
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--max-shift-cells", type=float, default=0.5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("names", nargs="+")
    args = parser.parse_args()
    if (
        args.steps < 1
        or args.repeats < 1
        or not np.isfinite(args.max_shift_cells)
        or args.max_shift_cells <= 0
    ):
        parser.error("steps, repeats, and max-shift-cells must be positive")
    args.out.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    (args.out / "source.json").write_text(
        json.dumps(
            dict(
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                torch=torch.__version__,
                numpy=np.__version__,
                steps=args.steps,
                max_shift_cells=args.max_shift_cells,
                repeats=args.repeats,
            ),
            indent=2,
        )
        + "\n"
    )
    rows = []
    for name in args.names:
        data = pickle.loads((args.inputs / f"{name}_final.pkl").read_bytes())
        baseline = np.load(args.inputs / f"{name}.npy")
        benchmark, plc = load(name)
        allowed = contract_for(data, benchmark)
        n = benchmark.num_hard_macros
        assert allowed(baseline[:n], baseline[n:]), name
        valid, issues = validate_placement(torch.from_numpy(baseline), benchmark)
        assert valid, (name, issues)
        before = float(_exact_proxy(torch.from_numpy(baseline), benchmark, plc))
        external_scores = {}
        external_components = {}

        def external_score(positions):
            key = positions.tobytes()
            if key not in external_scores:
                fresh_benchmark, fresh_plc = load(name)
                costs = compute_proxy_cost(
                    torch.from_numpy(positions).double(), fresh_benchmark, fresh_plc
                )
                assert costs["overlap_count"] == 0
                external_scores[key] = float(costs["proxy_cost"])
                external_components[key] = {
                    term: float(costs[term + "_cost"])
                    for term in ("wirelength", "density", "congestion")
                }
            return external_scores[key]

        for device in args.devices:
            samples, previous = [], None
            for _ in range(args.repeats):
                # Rebuild independent scoring state for every identical-input trial.
                benchmark, plc = load(name)
                candidates, timing = propose(
                    data,
                    benchmark,
                    plc,
                    baseline,
                    device,
                    steps=args.steps,
                    max_shift=args.max_shift_cells,
                )
                if previous is not None:
                    np.testing.assert_array_equal(candidates, previous)
                previous = candidates
                samples.append(timing)
                print(
                    "PROPOSE "
                    + json.dumps(
                        dict(benchmark=name, device=device, repetition=len(samples), **timing)
                    ),
                    flush=True,
                )
            best, best_score = baseline, before
            trials = []
            validation_start = time.perf_counter()
            for candidate in candidates:
                np.testing.assert_array_equal(candidate[:n], baseline[:n])
                valid, _ = validate_placement(
                    torch.from_numpy(candidate), benchmark, check_overlaps=False
                )
                contract = bool(allowed(candidate[:n], candidate[n:]))
                trial = dict(valid=bool(valid), contract=contract)
                if valid and contract:
                    fresh_benchmark, fresh_plc = load(name)
                    trial.update(
                        proxy=float(
                            _exact_proxy(torch.from_numpy(candidate), fresh_benchmark, fresh_plc)
                        ),
                        overlaps=0,
                    )
                    if trial["overlaps"] == 0 and trial["proxy"] < best_score - 1e-6:
                        best, best_score = candidate, trial["proxy"]
                trials.append(trial)
            assert allowed(best[:n], best[n:])
            np.testing.assert_array_equal(best[:n], baseline[:n])
            np.testing.assert_array_equal(
                best[benchmark.macro_fixed.numpy()], baseline[benchmark.macro_fixed.numpy()]
            )
            external_before, external_after = external_score(baseline), external_score(best)
            assert abs(external_after - best_score) < 1e-7, (name, external_after, best_score)
            assert external_after <= external_before + 1e-7
            row = dict(
                benchmark=name,
                device=device,
                before=before,
                after=best_score,
                gain=before - best_score,
                external_before=external_before,
                external_after=external_after,
                before_components=external_components[baseline.tobytes()],
                after_components=external_components[best.tobytes()],
                samples=samples,
                trials=trials,
                validation_s=time.perf_counter() - validation_start,
                input_sha256=hashlib.sha256(baseline.tobytes()).hexdigest(),
            )
            rows.append(row)
            np.save(args.out / f"{name}_{device}.npy", best)
            (args.out / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
