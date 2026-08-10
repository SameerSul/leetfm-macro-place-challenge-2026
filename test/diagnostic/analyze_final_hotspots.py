"""Run VivaPlace and attribute exact proxy-tail congestion and density hotspots.

Usage:
    uv run python test/diagnostic/analyze_final_hotspots.py --all
    uv run python test/diagnostic/analyze_final_hotspots.py -b ibm10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from macro_place.evaluate import IBM_BENCHMARKS, _load_placer
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from macro_place.utils import validate_placement
from placer.local_search.hierarchy_model import HierarchyModel
from placer.plc.placement import _ensure_pos_cache
from placer.routing.apply import _apply_net_routing_struct, _build_net_routing_struct
from placer.scoring.congestion import _patch_plc_congestion
from placer.scoring.density import _patch_plc_density
from placer.scoring.incremental import IncrementalScorer
from placer.scoring.wirelength import _build_wl_cache, _patch_plc_wirelength


def _components(mask: np.ndarray, values: np.ndarray, limit: int = 4) -> list[dict]:
    """Return the highest-pressure four-neighbour components of a tail mask."""
    nr, nc = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    rows = []
    for r0 in range(nr):
        for c0 in range(nc):
            if seen[r0, c0] or not mask[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            cells = []
            while stack:
                r, c = stack.pop()
                cells.append(r * nc + c)
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= rr < nr and 0 <= cc < nc and mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            indices = np.asarray(cells, dtype=np.int64)
            rr = indices // nc
            cc = indices % nc
            vals = values.ravel()[indices]
            rows.append(
                {
                    "indices": indices,
                    "cells": int(indices.size),
                    "r0": int(rr.min()),
                    "r1": int(rr.max()),
                    "c0": int(cc.min()),
                    "c1": int(cc.max()),
                    "peak": float(vals.max()),
                    "mean": float(vals.mean()),
                    "pressure": float(vals.sum()),
                }
            )
    rows.sort(key=lambda row: (-row["pressure"], -row["peak"], row["r0"], row["c0"]))
    return rows[: max(1, int(limit))]


def _expanded_mask(indices: np.ndarray, nr: int, nc: int, radius: int) -> np.ndarray:
    mask = np.zeros((nr, nc), dtype=bool)
    for index in indices:
        r, c = divmod(int(index), nc)
        mask[
            max(0, r - radius) : min(nr, r + radius + 1),
            max(0, c - radius) : min(nc, c + radius + 1),
        ] = True
    return mask


def _macro_attribution(
    component: dict,
    positions: np.ndarray,
    sizes: np.ndarray,
    names: list[str],
    cluster_by_output: np.ndarray,
    n_hard: int,
    grid_w: float,
    grid_h: float,
    nr: int,
    nc: int,
) -> dict:
    """Attribute occupied component-cell area to exact macro rectangles."""
    mask = np.zeros((nr, nc), dtype=bool)
    mask.ravel()[component["indices"]] = True
    attributed = []
    hard_area = 0.0
    soft_area = 0.0
    cluster_area: Counter[int] = Counter()
    for output, ((x, y), (width, height)) in enumerate(zip(positions, sizes)):
        x0, x1 = float(x - width / 2.0), float(x + width / 2.0)
        y0, y1 = float(y - height / 2.0), float(y + height / 2.0)
        c0 = max(0, int(np.floor(x0 / grid_w)))
        c1 = min(nc - 1, int(np.floor(np.nextafter(x1, -np.inf) / grid_w)))
        r0 = max(0, int(np.floor(y0 / grid_h)))
        r1 = min(nr - 1, int(np.floor(np.nextafter(y1, -np.inf) / grid_h)))
        if c1 < c0 or r1 < r0 or not np.any(mask[r0 : r1 + 1, c0 : c1 + 1]):
            continue
        area = 0.0
        for r in range(r0, r1 + 1):
            oy = max(0.0, min(y1, (r + 1) * grid_h) - max(y0, r * grid_h))
            for c in range(c0, c1 + 1):
                if not mask[r, c]:
                    continue
                ox = max(0.0, min(x1, (c + 1) * grid_w) - max(x0, c * grid_w))
                area += ox * oy
        if area <= 0.0:
            continue
        is_hard = output < n_hard
        if is_hard:
            hard_area += area
        else:
            soft_area += area
        cid = int(cluster_by_output[output])
        if cid >= 0:
            cluster_area[cid] += area
        attributed.append(
            {
                "output": int(output),
                "name": names[output],
                "kind": "hard" if is_hard else "soft",
                "cluster": cid,
                "area": float(area),
            }
        )
    attributed.sort(key=lambda row: (-row["area"], row["output"]))
    return {
        "macro_count": len(attributed),
        "hard_area": hard_area,
        "soft_area": soft_area,
        "soft_area_share": soft_area / max(hard_area + soft_area, 1.0e-12),
        "cluster_count": len(cluster_area),
        "cluster_area": [
            {"cluster": int(cid), "area": float(area)} for cid, area in cluster_area.most_common(8)
        ],
        "top_macros": attributed[:12],
    }


def _net_attribution(
    component: dict,
    plc,
    cluster_by_module: dict[int, int],
    name_by_module: dict[int, str],
    nr: int,
    nc: int,
) -> dict:
    """Route candidate nets exactly and retain those touching the hotspot."""
    cache = _build_wl_cache(plc)
    target = _expanded_mask(component["indices"], nr, nc, int(plc.smooth_range))
    target_flat = target.ravel()
    unique_ref = cache["unique_ref"]
    inv = cache["ref_inv"]
    pos = np.asarray(_ensure_pos_cache(plc), dtype=np.float64)
    grid_w = float(plc.width / nc)
    grid_h = float(plc.height / nr)
    target_rr, target_cc = np.nonzero(target)
    target_r0, target_r1 = int(target_rr.min()), int(target_rr.max())
    target_c0, target_c1 = int(target_cc.min()), int(target_cc.max())
    implicated = []
    class_demand: Counter[str] = Counter()
    relation_demand: Counter[str] = Counter()
    for net_index, start_raw in enumerate(cache["net_starts"]):
        start = int(start_raw)
        length = int(cache["net_lengths"][net_index])
        if length < 2:
            continue
        pin_indices = np.arange(start, start + length, dtype=np.int64)
        modules = unique_ref[inv[pin_indices]].astype(np.int64, copy=False)
        pin_x = pos[modules, 0] + cache["x_off"][pin_indices]
        pin_y = pos[modules, 1] + cache["y_off"][pin_indices]
        cols = np.clip((pin_x / grid_w).astype(np.int64), 0, nc - 1)
        rows = np.clip((pin_y / grid_h).astype(np.int64), 0, nr - 1)
        if (
            cols.max() < target_c0
            or cols.min() > target_c1
            or rows.max() < target_r0
            or rows.min() > target_r1
        ):
            continue
        hard_clusters = sorted(
            {
                cluster_by_module[int(module)]
                for module in modules
                if cluster_by_module.get(int(module), -1) >= 0
            }
        )
        if len(hard_clusters) == 1:
            traffic_class = "internal"
        elif len(hard_clusters) >= 2:
            traffic_class = "cross_cluster"
        else:
            traffic_class = "unassigned_soft_or_io"
        h_raw = np.zeros(nr * nc, dtype=np.float64)
        v_raw = np.zeros(nr * nc, dtype=np.float64)
        struct = _build_net_routing_struct(plc, np.asarray([net_index], dtype=np.int64))
        _apply_net_routing_struct(plc, struct, 1.0, h_raw, v_raw)
        demand = float(h_raw[target_flat].sum() + v_raw[target_flat].sum())
        if demand <= 0.0:
            continue
        class_demand[traffic_class] += demand
        relation = "-".join(str(cid) for cid in hard_clusters[:4]) or "unassigned"
        relation_demand[relation] += demand
        implicated.append(
            {
                "net": int(net_index),
                "fanout": length,
                "demand": demand,
                "class": traffic_class,
                "clusters": hard_clusters,
                "modules": [
                    name_by_module.get(int(module), str(int(module))) for module in modules[:8]
                ],
            }
        )
    implicated.sort(key=lambda row: (-row["demand"], row["net"]))
    total = float(sum(class_demand.values()))
    return {
        "routed_net_count": len(implicated),
        "total_demand": total,
        "demand_by_class": {
            key: {"demand": float(value), "share": float(value / max(total, 1.0e-12))}
            for key, value in class_demand.most_common()
        },
        "top_cluster_relations": [
            {"relation": key, "demand": float(value)}
            for key, value in relation_demand.most_common(10)
        ],
        "top_nets": implicated[:12],
    }


def _physical_box(component: dict, grid_w: float, grid_h: float, cw: float, ch: float) -> dict:
    x0, x1 = component["c0"] * grid_w, (component["c1"] + 1) * grid_w
    y0, y1 = component["r0"] * grid_h, (component["r1"] + 1) * grid_h
    return {
        "x0": float(x0),
        "y0": float(y0),
        "x1": float(x1),
        "y1": float(y1),
        "canvas_fraction": [float(x0 / cw), float(y0 / ch), float(x1 / cw), float(y1 / ch)],
    }


def _analyze_result(name: str, benchmark, plc, placement, costs: dict, hierarchy) -> dict:
    positions = placement.detach().cpu().numpy().astype(np.float64)
    sizes = benchmark.macro_sizes.detach().cpu().numpy().astype(np.float64)
    n_hard = int(benchmark.num_hard_macros)
    n_soft = int(benchmark.num_soft_macros)
    nr, nc = int(benchmark.grid_rows), int(benchmark.grid_cols)
    cw, ch = float(benchmark.canvas_width), float(benchmark.canvas_height)
    scorer = IncrementalScorer(plc, benchmark, positions.copy())
    h_grid = scorer.H_smoothed + (scorer.H_macro_flat / scorer.grid_h_routes).reshape(nr, nc)
    v_grid = scorer.V_smoothed + (scorer.V_macro_flat / scorer.grid_v_routes).reshape(nr, nc)
    all_congestion = np.concatenate([v_grid.ravel(), h_grid.ravel()])
    cong_count = max(1, int(all_congestion.size * 0.05))
    cong_order = np.argsort(all_congestion, kind="stable")[-cong_count:]
    v_tail = np.zeros(nr * nc, dtype=bool)
    h_tail = np.zeros(nr * nc, dtype=bool)
    v_tail[cong_order[cong_order < nr * nc]] = True
    h_indices = cong_order[cong_order >= nr * nc] - nr * nc
    h_tail[h_indices] = True
    v_tail = v_tail.reshape(nr, nc)
    h_tail = h_tail.reshape(nr, nc)
    congestion_mask = v_tail | h_tail
    congestion_value = np.maximum(v_grid, h_grid)

    density_ratio = (scorer.grid_occupied / scorer.dens_grid_area).reshape(nr, nc)
    nz = np.flatnonzero(scorer.grid_occupied > 0.0)
    density_count = max(1, int(np.floor(nr * nc * 0.1)))
    density_take = min(density_count, nz.size)
    density_indices = (
        nz[np.argsort(scorer.grid_occupied[nz], kind="stable")[-density_take:]]
        if density_take
        else np.zeros(0, dtype=np.int64)
    )
    density_mask = np.zeros((nr, nc), dtype=bool)
    density_mask.ravel()[density_indices] = True

    hard_modules = [int(module) for module in plc.hard_macro_indices[:n_hard]]
    soft_modules = [int(module) for module in plc.soft_macro_indices[:n_soft]]
    output_modules = hard_modules + soft_modules
    name_by_module = {
        module: str(plc.modules_w_pins[module].get_name()) for module in output_modules
    }
    names = [name_by_module[module] for module in output_modules]
    cluster_by_output = np.full(n_hard + n_soft, -1, dtype=np.int64)
    cluster_by_output[:n_hard] = np.asarray(hierarchy.labels[:n_hard], dtype=np.int64)
    for cid, full_indices in hierarchy.cluster_softs.items():
        for output in np.asarray(full_indices, dtype=np.int64):
            if n_hard <= output < n_hard + n_soft:
                cluster_by_output[int(output)] = int(cid)
    cluster_by_module = {
        module: int(cluster_by_output[output]) for output, module in enumerate(output_modules)
    }

    congestion_components = _components(congestion_mask, congestion_value)
    for rank, component in enumerate(congestion_components, 1):
        indices = component["indices"]
        cell_h = h_grid.ravel()[indices]
        cell_v = v_grid.ravel()[indices]
        cell_hm = (scorer.H_macro_flat / scorer.grid_h_routes)[indices]
        cell_vm = (scorer.V_macro_flat / scorer.grid_v_routes)[indices]
        dominant_h = cell_h >= cell_v
        total = np.where(dominant_h, cell_h, cell_v)
        blockage = np.where(dominant_h, cell_hm, cell_vm)
        component.update(
            {
                "rank": rank,
                "box": _physical_box(component, scorer.dens_grid_w, scorer.dens_grid_h, cw, ch),
                "horizontal_tail_cells": int(np.count_nonzero(h_tail.ravel()[indices])),
                "vertical_tail_cells": int(np.count_nonzero(v_tail.ravel()[indices])),
                "macro_blockage_share": float(np.sum(blockage) / max(np.sum(total), 1.0e-12)),
                "macros": _macro_attribution(
                    component,
                    positions,
                    sizes,
                    names,
                    cluster_by_output,
                    n_hard,
                    scorer.dens_grid_w,
                    scorer.dens_grid_h,
                    nr,
                    nc,
                ),
                "nets": _net_attribution(component, plc, cluster_by_module, name_by_module, nr, nc),
            }
        )
        component.pop("indices")

    density_components = _components(density_mask, density_ratio)
    for rank, component in enumerate(density_components, 1):
        component.update(
            {
                "rank": rank,
                "box": _physical_box(component, scorer.dens_grid_w, scorer.dens_grid_h, cw, ch),
                "macros": _macro_attribution(
                    component,
                    positions,
                    sizes,
                    names,
                    cluster_by_output,
                    n_hard,
                    scorer.dens_grid_w,
                    scorer.dens_grid_h,
                    nr,
                    nc,
                ),
            }
        )
        component.pop("indices")

    tail_union = congestion_mask | density_mask
    overlap = congestion_mask & density_mask
    return {
        "benchmark": name,
        "proxy": float(costs["proxy_cost"]),
        "wirelength": float(costs["wirelength_cost"]),
        "density": float(costs["density_cost"]),
        "congestion": float(costs["congestion_cost"]),
        "grid": [nr, nc],
        "canvas": [cw, ch],
        "tail": {
            "congestion_directional_cells": cong_count,
            "congestion_spatial_cells": int(np.count_nonzero(congestion_mask)),
            "density_cells": int(np.count_nonzero(density_mask)),
            "overlap_cells": int(np.count_nonzero(overlap)),
            "overlap_jaccard": float(
                np.count_nonzero(overlap) / max(np.count_nonzero(tail_union), 1)
            ),
        },
        "congestion_hotspots": congestion_components,
        "density_hotspots": density_components,
    }


def _write_markdown(report: dict, path: Path) -> None:
    rows = report["designs"]
    lines = [
        "# Final congestion and density hotspot analysis",
        "",
        "Congestion regions use the evaluator's exact top-5% directional tail. Density regions use its top-10% occupied-cell tail.",
        "",
        "| Design | Proxy | Cong. | Density | Worst congestion region (canvas fractions x0,y0–x1,y1) | Cause | Worst density region |",
        "|---|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        cong = row["congestion_hotspots"][0] if row["congestion_hotspots"] else None
        dens = row["density_hotspots"][0] if row["density_hotspots"] else None
        if cong:
            f = cong["box"]["canvas_fraction"]
            classes = cong["nets"]["demand_by_class"]
            cross = float(classes.get("cross_cluster", {}).get("share", 0.0))
            internal = float(classes.get("internal", {}).get("share", 0.0))
            unassigned = float(classes.get("unassigned_soft_or_io", {}).get("share", 0.0))
            cause = (
                f"unassigned {unassigned:.0%}, internal {internal:.0%}, "
                f"cross {cross:.0%}, blockage {cong['macro_blockage_share']:.0%}"
            )
            cong_box = f"{f[0]:.2f},{f[1]:.2f}–{f[2]:.2f},{f[3]:.2f}; peak {cong['peak']:.2f}"
        else:
            cause, cong_box = "-", "-"
        if dens:
            f = dens["box"]["canvas_fraction"]
            dens_box = f"{f[0]:.2f},{f[1]:.2f}–{f[2]:.2f},{f[3]:.2f}; peak {dens['peak']:.2f}×"
        else:
            dens_box = "-"
        lines.append(
            f"| {row['benchmark']} | {row['proxy']:.4f} | {row['congestion']:.3f} | {row['density']:.3f} | {cong_box} | {cause} | {dens_box} |"
        )
    lines.extend(
        [
            "",
            "## Suite summary",
            "",
            f"- Average proxy: {report['summary']['proxy']:.6f}",
            f"- Average congestion contribution: {report['summary']['congestion_contribution']:.6f}",
            f"- Average density contribution: {report['summary']['density_contribution']:.6f}",
            f"- Average wirelength contribution: {report['summary']['wirelength_contribution']:.6f}",
            f"- Mean top-tail congestion/density spatial overlap: {report['summary']['mean_tail_overlap_jaccard']:.3f}",
            "",
            "The adjacent JSON contains every reported component's physical box, macro/cluster area attribution, exact routed-net attribution, and top implicated nets/macros.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--benchmark", choices=IBM_BENCHMARKS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--placer", default="src/main.py")
    parser.add_argument("--output-dir", default="ml_data/hotspot_analysis")
    args = parser.parse_args()
    names = IBM_BENCHMARKS if args.all else [args.benchmark or "ibm10"]
    placer = _load_placer(Path(args.placer))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    designs = []
    placements = {}
    started = time.time()
    for name in names:
        print(f"{name}: placing", flush=True)
        benchmark_dir = Path("external/MacroPlacement/Testcases/ICCAD04") / name
        benchmark, plc = load_benchmark_from_dir(str(benchmark_dir))
        _patch_plc_wirelength(plc)
        _patch_plc_congestion(plc, benchmark)
        _patch_plc_density(plc, benchmark)
        hierarchy = HierarchyModel.build(
            plc,
            int(benchmark.num_hard_macros),
            int(benchmark.num_soft_macros),
            hard_sizes=benchmark.macro_sizes[: benchmark.num_hard_macros].numpy(),
        )
        placement = placer.place(benchmark)
        valid, violations = validate_placement(placement, benchmark)
        if not valid:
            raise RuntimeError(f"{name} invalid: {violations}")
        costs = compute_proxy_cost(placement, benchmark, plc)
        print(f"{name}: analyzing proxy={costs['proxy_cost']:.4f}", flush=True)
        designs.append(_analyze_result(name, benchmark, plc, placement, costs, hierarchy))
        placements[name] = placement.detach().cpu().numpy().astype(np.float32)
    proxy = float(np.mean([row["proxy"] for row in designs]))
    wirelength = float(np.mean([row["wirelength"] for row in designs]))
    density = float(np.mean([row["density"] for row in designs]))
    congestion = float(np.mean([row["congestion"] for row in designs]))
    report = {
        "run_id": run_id,
        "elapsed_s": time.time() - started,
        "designs": designs,
        "summary": {
            "proxy": proxy,
            "wirelength_contribution": wirelength,
            "density_contribution": 0.5 * density,
            "congestion_contribution": 0.5 * congestion,
            "mean_tail_overlap_jaccard": float(
                np.mean([row["tail"]["overlap_jaccard"] for row in designs])
            ),
        },
    }
    json_path = output_dir / f"{run_id}.json"
    md_path = output_dir / f"{run_id}.md"
    npz_path = output_dir / f"{run_id}-placements.npz"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_markdown(report, md_path)
    np.savez_compressed(npz_path, **placements)
    print(f"report: {json_path}")
    print(f"summary: {md_path}")
    print(f"placements: {npz_path}")


if __name__ == "__main__":
    main()
