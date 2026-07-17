"""Hierarchy floorplan pipeline segment extracted from macro_placer."""

import os
import time

import numpy as np
import torch
from macro_place.benchmark import Benchmark

from utils import constants as const
from utils.config import _log
from placer.scoring.exact import _exact_proxy


def run_hierarchy_floorplan(benchmark: Benchmark) -> "torch.Tensor | None":
    """Non-proxy hierarchy-preserving placement.

    Grouped DREAMPlace derives a hierarchical global placement, cluster-
    consecutive legalization keeps each subsystem together, and bounded
    cleanup recovers some congestion without making proxy score the primary
    objective.
    """
    from dreamplace_bridge.run_bridge import (  # noqa: E402
        run_dreamplace,
        is_available as _dp_available,
    )
    from placer.legalize.spiral import _will_legalize
    from placer.local_search.cluster_decompress import (
        _cluster_decompression_relief,
        hierarchy_quality_metric,
    )
    from placer.local_search.compound_relocation import _compound_soft_relocation
    from placer.local_search.fields import _congestion_field
    from placer.local_search.plateau_telemetry import log_plateau_event
    from placer.local_search.graph_tension import (
        cluster_graph_tension,
        hard_tension_from_labels,
    )
    from placer.local_search.hierarchy_swaps import _region_bounded_swap_relief
    from placer.local_search.hierarchy_model import HierarchyModel
    from placer.local_search.hierarchy_quality import (
        hierarchy_quality_vector,
        hierarchy_vector_contract,
        hierarchy_vector_limits,
    )
    from placer.local_search.region_expand import expand_regions_by_congestion
    from placer.local_search.relocation import (
        _micro_shift_polish,
        _relocation_moves,
        _soft_relocation_moves,
    )
    from placer.pipeline.hierarchy_context import (
        PassContext,
        PlacementState,
        PlateauTelemetry,
    )
    from placer.scoring.incremental import IncrementalScorer
    from placer.scoring.wirelength import _build_wl_cache
    from placer.plc.loader import _load_plc, _resolve_benchmark_dir
    from placer.pipeline.segments.floorplan_seed import run_seed_portfolio
    from placer.pipeline.segments.floorplan_coldspot import run_coldspot_tightening
    from placer.pipeline.segments.floorplan_post_coldspot import (
        run_post_coldspot_finalize,
    )

    if not _dp_available():
        return None
    benchmark_dir = _resolve_benchmark_dir(benchmark.name, benchmark)
    if benchmark_dir is None:
        return None

    diagnostic_no_deadlines = os.environ.get("HIER_DIAGNOSTIC_NO_DEADLINES", "0").strip() in {
        "1",
        "true",
        "TRUE",
        "yes",
        "YES",
        "on",
    }

    def _deadline(seconds: float, outer: "float | None" = None) -> "float | None":
        if diagnostic_no_deadlines:
            return None
        now_deadline = time.monotonic() + float(seconds)
        return min(outer, now_deadline) if outer is not None else now_deadline

    def _additive_spare(deadline: "float | None") -> bool:
        return deadline is None or time.monotonic() + float(
            const.HIER_ADDITIVE_MIN_SPARE_S
        ) < float(deadline)

    def _proxy_components(hard_xy, soft_xy) -> dict[str, float]:
        full = torch.tensor(
            np.vstack([hard_xy, soft_xy]).astype(np.float32),
            dtype=torch.float32,
        )
        _exact_proxy(full, benchmark, plc)
        return {
            "wirelength": float(plc.get_cost()),
            "density": float(plc.get_density_cost()),
            "congestion": float(plc.get_congestion_cost()),
        }

    def _component_cleanup_bias(hard_xy, soft_xy) -> dict[str, float | bool]:
        comp = _proxy_components(hard_xy, soft_xy)
        density = float(comp["density"])
        congestion = float(comp["congestion"])
        dominates = congestion > density + float(const.HIER_COMPONENT_CONG_DOMINANCE)
        return {
            "enabled": True,
            "congestion_dominates": bool(dominates),
            "wirelength": float(comp["wirelength"]),
            "density": density,
            "congestion": congestion,
        }

    hier_soft_barrier_gain = max(
        0.0,
        float(os.environ.get("HIER_SOFT_BARRIER_GAIN", const.HIER_SOFT_BARRIER_GAIN)),
    )
    hier_region_heat_frac = float(const.HIER_REGION_HEAT_FRAC)
    hier_region_heat_pct = float(const.HIER_REGION_HEAT_HOT_PCT)
    hier_region_heat_escape = float(const.HIER_REGION_HEAT_ESCAPE_MIN)
    hier_micro_shift_radius = max(1, int(const.HIER_MICRO_SHIFT_RADIUS))
    hier_micro_shift_top = max(1, int(const.HIER_MICRO_SHIFT_TOP))
    hier_micro_shift_min_gain = float(const.HIER_MICRO_SHIFT_MIN_GAIN)
    plc = _load_plc(benchmark.name, benchmark)
    n = benchmark.num_hard_macros
    n_soft = benchmark.num_soft_macros
    cw, ch = float(benchmark.canvas_width), float(benchmark.canvas_height)
    sizes = benchmark.macro_sizes.numpy().astype(np.float64)
    hw, hh = sizes[:n, 0] / 2.0, sizes[:n, 1] / 2.0
    soft_hw, soft_hh = sizes[n : n + n_soft, 0] / 2.0, sizes[n : n + n_soft, 1] / 2.0
    movable = benchmark.get_movable_mask().numpy()
    gw = max(1, int(const.HIER_GROUP_WEIGHT))
    pass_context = PassContext(
        benchmark_name=str(benchmark.name),
        diagnostic_no_deadlines=bool(diagnostic_no_deadlines),
    )

    hierarchy = HierarchyModel.build(plc, n, n_soft, hard_sizes=sizes[:n])
    labels = hierarchy.labels
    clusters = hierarchy.clusters
    csofts = hierarchy.cluster_softs
    bridge_softs = hierarchy.bridge_softs

    plateau_records: list[PlateauTelemetry] = []

    def _record_plateau(
        pass_name: str,
        before: float,
        after: float,
        accepts: int,
        elapsed_s: float,
        *,
        candidates: int = 0,
        legal: int = 0,
        scored: int = 0,
        **extra,
    ) -> PlateauTelemetry:
        record = PlateauTelemetry(
            name=pass_name,
            proxy_before=float(before),
            proxy_after=float(after),
            elapsed_s=float(elapsed_s),
            candidates=int(candidates),
            legal=int(legal),
            scored=int(scored),
            accepts=int(accepts),
            extra=extra,
        )
        plateau_records.append(record)
        plateaued = record.plateaued(
            float(const.HIER_PLATEAU_ACCEPT_RATE),
            float(const.HIER_PLATEAU_PROXY_GAIN),
        )
        log_plateau_event(
            "hier_plateau_telemetry",
            benchmark=pass_context.benchmark_name,
            diagnostic_no_deadlines=pass_context.diagnostic_no_deadlines,
            plateaued=bool(plateaued),
            **record.to_trace_kwargs(),
        )
        return record

    def _is_plateau(record: PlateauTelemetry | None) -> bool:
        if record is None:
            return False
        return record.plateaued(
            float(const.HIER_PLATEAU_ACCEPT_RATE),
            float(const.HIER_PLATEAU_PROXY_GAIN),
        )

    adaptive_floor_proxy_gain = float(const.HIER_PLATEAU_PROXY_GAIN)
    hier_micro_shift_min_gain = max(
        float(hier_micro_shift_min_gain),
        adaptive_floor_proxy_gain,
    )

    def _adaptive_pass_gain(before: float, after: float) -> bool:
        return float(before) - float(after) > adaptive_floor_proxy_gain

    def _has_spare(deadline: "float | None", reserve_s: float) -> bool:
        return deadline is None or time.monotonic() + float(reserve_s) < float(deadline)

    def _env_float(name: str, default: float) -> float:
        raw = os.environ.get(name)
        if raw is None or not raw.strip():
            return float(default)
        try:
            return float(raw)
        except ValueError:
            _log(f"  [hier] env parse fallback: {name}={raw!r}, using {float(default)}")
            return float(default)

    def _env_int(name: str, default: int) -> int:
        raw = os.environ.get(name)
        if raw is None or not raw.strip():
            return int(default)
        try:
            return int(raw)
        except ValueError:
            _log(f"  [hier] env parse fallback: {name}={raw!r}, using {int(default)}")
            return int(default)

    def _empty_pass_stats() -> dict[str, int]:
        return {"candidates": 0, "legal": 0, "scored": 0, "accepts": 0}

    def _accum_pass_stats(total: dict[str, int], stats: dict) -> None:
        for key in ("candidates", "legal", "scored", "accepts"):
            total[key] += int(stats.get(key, 0))

    def _hard_legality_margin(hard_xy, eps: float) -> dict[str, float]:
        if hard_xy.shape[0] == 0:
            return {
                "pair_margin": float("inf"),
                "bounds_margin": float("inf"),
                "min_margin": float("inf"),
            }
        bounds_x = np.minimum(hard_xy[:, 0] - hw, cw - hw - hard_xy[:, 0])
        bounds_y = np.minimum(hard_xy[:, 1] - hh, ch - hh - hard_xy[:, 1])
        bounds_margin = float(min(np.min(bounds_x), np.min(bounds_y)))
        if hard_xy.shape[0] < 2:
            pair_margin = float("inf")
        else:
            dx = np.abs(hard_xy[:, None, 0] - hard_xy[None, :, 0])
            dy = np.abs(hard_xy[:, None, 1] - hard_xy[None, :, 1])
            sep_x = hw[:, None] + hw[None, :] + float(eps)
            sep_y = hh[:, None] + hh[None, :] + float(eps)
            clear = np.maximum(dx - sep_x, dy - sep_y)
            np.fill_diagonal(clear, np.inf)
            pair_margin = float(np.min(clear))
        return {
            "pair_margin": pair_margin,
            "bounds_margin": bounds_margin,
            "min_margin": min(pair_margin, bounds_margin),
        }

    groups = hierarchy.dreamplace_groups(plc, n)

    def _full_tensor(hard_xy, soft_xy):
        return torch.tensor(np.vstack([hard_xy, soft_xy]).astype(np.float32), dtype=torch.float32)

    def _hard_connectivity_pressure() -> np.ndarray:
        pressure = np.zeros(n, dtype=np.float64)
        cache = _build_wl_cache(plc)
        ref_idx = cache["ref_idx"]
        net_starts = cache["net_starts"]
        net_lengths = cache["net_lengths"]
        net_weights = cache["net_weights"]
        b_to_a = {int(b): a for a, b in enumerate(plc.hard_macro_indices)}
        max_fanout = hierarchy.max_fanout
        for net_i in range(len(net_starts)):
            length = int(net_lengths[net_i])
            if length < 2 or length > max_fanout:
                continue
            start = int(net_starts[net_i])
            hard_a = [b_to_a[int(r)] for r in ref_idx[start : start + length] if int(r) in b_to_a]
            if not hard_a:
                continue
            add = float(net_weights[net_i]) / max(1.0, float(len(hard_a) - 1))
            for i in hard_a:
                pressure[int(i)] += add
        return pressure

    area = sizes[:n, 0] * sizes[:n, 1]
    pressure = _hard_connectivity_pressure()

    def _member_key(i: int) -> tuple[float, float]:
        return (-(pressure[i] * area[i]), -area[i])

    def _cluster_key(mem: np.ndarray) -> tuple[float, int]:
        return (-float(np.sum(pressure[mem] * area[mem])), -int(mem.size))

    order = []
    for mem in sorted(clusters.values(), key=_cluster_key):
        order += sorted((int(x) for x in mem), key=_member_key)
    order += sorted([i for i in range(n) if labels[i] < 0], key=_member_key)
    try:
        hard, soft, s_score, seed_rows = run_seed_portfolio(
            benchmark=benchmark,
            plc=plc,
            benchmark_dir=benchmark_dir,
            n=n,
            n_soft=n_soft,
            clusters=clusters,
            order=order,
            sizes=sizes,
            hw=hw,
            hh=hh,
            soft_hw=soft_hw,
            soft_hh=soft_hh,
            movable=movable,
            groups=groups,
            csofts=csofts,
            bridge_softs=bridge_softs,
            hierarchy_edges=hierarchy.edges,
            cw=cw,
            ch=ch,
            const=const,
            logger=_log,
            run_dreamplace=run_dreamplace,
            will_legalize=_will_legalize,
            exact_proxy_fn=_exact_proxy,
            soft_relocation_fn=_soft_relocation_moves,
            incremental_scorer_cls=IncrementalScorer,
            group_weight=gw,
            random_seed=1000,
            scratch_root="/tmp/dreamplace_v1_hier",
        )
    except Exception as exc:
        _log(f"  [hier] DREAMPlace failed: {type(exc).__name__}: {exc}")
        return None
    selected_seed_name = str(seed_rows[0].get("name", "dreamplace")) if seed_rows else "dreamplace"

    legal = hard
    s_pos = soft.copy()
    seed_hard_for_tension = legal.copy()
    state = PlacementState(legal.copy(), s_pos.copy(), float(s_score))
    pos = state.full()
    scorer = IncrementalScorer(plc, benchmark, pos.copy())
    soft_mov = movable[n : n + n_soft]
    seed_hierarchy_quality = hierarchy_quality_metric(legal, clusters)
    seed_hierarchy_vector = dict(
        seed_rows[0].get(
            "hierarchy_vector",
            hierarchy_quality_vector(
                legal,
                s_pos,
                clusters,
                csofts,
                bridge_softs,
                hierarchy.edges,
                cw,
                ch,
            ),
        )
    )
    seed_hierarchy_vector_limits = hierarchy_vector_limits(
        seed_hierarchy_vector,
        const.HIER_VECTOR_CONTRACT_ABS_SLACK,
        float(const.HIER_VECTOR_CONTRACT_REL_SLACK),
    )

    graph_tension_enabled = int(n) >= int(getattr(const, "HIER_GRAPH_TENSION_HARD_MIN", 0)) and int(
        n
    ) <= int(getattr(const, "HIER_GRAPH_TENSION_HARD_MAX", 1000000))
    graph_tension_weight = (
        max(
            0.0,
            _env_float(
                "HIER_GRAPH_TENSION_WEIGHT",
                getattr(const, "HIER_GRAPH_TENSION_WEIGHT", 0.0),
            ),
        )
        if graph_tension_enabled
        else 0.0
    )
    graph_tension_decomp_weight = (
        max(
            0.0,
            _env_float(
                "HIER_GRAPH_TENSION_DECOMP_WEIGHT",
                float(getattr(const, "HIER_GRAPH_TENSION_DECOMP_WEIGHT", graph_tension_weight)),
            ),
        )
        if graph_tension_enabled
        else 0.0
    )
    graph_tension_coldspot_weight = (
        max(
            0.0,
            _env_float(
                "HIER_GRAPH_TENSION_COLDSPOT_WEIGHT",
                float(getattr(const, "HIER_GRAPH_TENSION_COLDSPOT_WEIGHT", graph_tension_weight)),
            ),
        )
        if graph_tension_enabled
        else 0.0
    )
    graph_tension_swap_weight = (
        max(
            0.0,
            _env_float(
                "HIER_GRAPH_TENSION_SWAP_WEIGHT",
                float(getattr(const, "HIER_GRAPH_TENSION_SWAP_WEIGHT", 0.0)),
            ),
        )
        if graph_tension_enabled
        else 0.0
    )
    graph_swap_delta_weight = (
        max(
            0.0,
            _env_float(
                "HIER_SWAP_GRAPH_DELTA_WEIGHT",
                float(getattr(const, "HIER_SWAP_GRAPH_DELTA_WEIGHT", 0.0)),
            ),
        )
        if graph_tension_enabled
        else 0.0
    )
    graph_swap_mask_penalty_weight = (
        max(
            0.0,
            _env_float(
                "HIER_SWAP_GRAPH_MASK_PENALTY_WEIGHT",
                float(getattr(const, "HIER_SWAP_GRAPH_MASK_PENALTY_WEIGHT", 0.0)),
            ),
        )
        if graph_tension_enabled
        else 0.0
    )
    graph_swap_delta_samples = max(
        2,
        int(
            _env_int(
                "HIER_SWAP_GRAPH_DELTA_SAMPLES",
                int(getattr(const, "HIER_GRAPH_TENSION_CORRIDOR_SAMPLES", 9)),
            )
        ),
    )
    graph_swap_fallback_budget_s = max(
        0.0,
        _env_float(
            "HIER_SWAP_GRAPH_FALLBACK_BUDGET_S",
            float(getattr(const, "HIER_SWAP_GRAPH_FALLBACK_BUDGET_S", 2.5)),
        ),
    )
    graph_tension_active = max(
        graph_tension_decomp_weight,
        graph_tension_coldspot_weight,
        graph_tension_swap_weight,
    )

    def _graph_tension(hard_xy: np.ndarray, field=None) -> dict[int, float]:
        if graph_tension_active <= 0.0:
            return {}
        return cluster_graph_tension(
            hard_xy,
            clusters,
            hierarchy.edges,
            cw=float(cw),
            ch=float(ch),
            field=field,
            seed_hard_xy=seed_hard_for_tension,
            confidence=hierarchy.cluster_confidence,
            samples=max(2, int(getattr(const, "HIER_GRAPH_TENSION_CORRIDOR_SAMPLES", 9))),
        )

    def _hard_graph_tension(hard_xy: np.ndarray, field=None) -> np.ndarray:
        return hard_tension_from_labels(labels, _graph_tension(hard_xy, field), int(n))

    def _build_graph_swap_mask(
        *,
        hard_xy: np.ndarray,
        graph_weight: float,
    ) -> tuple[np.ndarray | None, dict[str, object]]:
        max_edges = max(
            0,
            _env_int(
                "HIER_SWAP_GRAPH_MASK_MAX_EDGES",
                int(getattr(const, "HIER_SWAP_GRAPH_MASK_MAX_EDGES", 0)),
            ),
        )
        pad_cells = max(
            0,
            _env_int(
                "HIER_SWAP_GRAPH_MASK_PAD_CELLS",
                int(getattr(const, "HIER_SWAP_GRAPH_MASK_PAD_CELLS", 1)),
            ),
        )
        if graph_weight <= 0.0:
            return None, {
                "enabled": False,
                "graph_weight": float(graph_weight),
                "reason": "zero_graph_tension_weight",
            }
        if not hierarchy.edges or not clusters:
            return None, {
                "enabled": False,
                "graph_weight": float(graph_weight),
                "reason": "no_graph_edges",
            }

        nr_, nc_ = int(benchmark.grid_rows), int(benchmark.grid_cols)
        if nr_ <= 0 or nc_ <= 0:
            return None, {
                "enabled": False,
                "graph_weight": float(graph_weight),
                "reason": "invalid_grid",
            }

        try:
            from placer.pipeline.segments.floorplan_coldspot_utils import dilate_cell_mask
        except Exception:
            return None, {
                "enabled": False,
                "graph_weight": float(graph_weight),
                "reason": "missing_dilate",
            }

        cell_w, cell_h = float(cw) / nc_, float(ch) / nr_
        if cell_w <= 0.0 or cell_h <= 0.0:
            return None, {
                "enabled": False,
                "graph_weight": float(graph_weight),
                "reason": "invalid_cell_size",
            }

        def _macro_cells(raw_mem: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
            mem = np.asarray(raw_mem, dtype=np.int64)
            mem = mem[(mem >= 0) & (mem < n)]
            if mem.size == 0:
                return None
            movable_mem = mem[movable[mem]]
            if movable_mem.size:
                mem = movable_mem
            x = np.clip((hard_xy[mem, 0] / cell_w), 0.0, float(nc_ - 1))
            y = np.clip((hard_xy[mem, 1] / cell_h), 0.0, float(nr_ - 1))
            return np.rint(x).astype(np.int64), np.rint(y).astype(np.int64)

        centers: dict[int, tuple[int, int]] = {}
        for cid, raw_mem in clusters.items():
            cells = _macro_cells(raw_mem)
            if cells is None:
                continue
            cols, rows = cells
            centers[int(cid)] = (int(float(rows.mean())), int(float(cols.mean())))

        if not centers:
            return None, {
                "enabled": False,
                "graph_weight": float(graph_weight),
                "reason": "no_cluster_cells",
            }

        edges = sorted(
            hierarchy.edges,
            key=lambda edge: float(getattr(edge, "weight", 0.0)),
            reverse=True,
        )
        if max_edges > 0:
            edges = edges[:max_edges]

        def _draw_line(
            mask: np.ndarray,
            r0: int,
            c0: int,
            r1: int,
            c1: int,
        ) -> None:
            dr = abs(int(r1) - int(r0))
            dc = abs(int(c1) - int(c0))
            steps = max(dr, dc, 1)
            rr = np.clip(
                np.rint(np.linspace(float(r0), float(r1), steps + 1)).astype(np.int64),
                0,
                nr_ - 1,
            )
            cc = np.clip(
                np.rint(np.linspace(float(c0), float(c1), steps + 1)).astype(np.int64),
                0,
                nc_ - 1,
            )
            mask[rr, cc] = True

        graph_mask = np.zeros((nr_, nc_), dtype=np.bool_)
        top_edges: list[tuple[int, int, float]] = []
        top_weights: list[float] = []
        for edge in edges:
            a = int(getattr(edge, "src", -1))
            b = int(getattr(edge, "dst", -1))
            if a not in centers or b not in centers:
                continue
            ar, ac = centers[a]
            br, bc = centers[b]
            if ar == br and ac == bc:
                continue
            w = float(getattr(edge, "weight", 0.0))
            top_edges.append((a, b, w))
            top_weights.append(w)
            _draw_line(graph_mask, ar, ac, br, bc)

        for r, c in centers.values():
            graph_mask[int(r), int(c)] = True

        if not graph_mask.any():
            return None, {
                "enabled": False,
                "graph_weight": float(graph_weight),
                "reason": "no_drawn_corridors",
                "requested_edges": int(len(hierarchy.edges)),
                "used_edges": int(len(top_edges)),
            }

        if pad_cells > 0:
            graph_mask = dilate_cell_mask(graph_mask, pad_cells, nr_, nc_)

        used_edges = int(len(top_edges))
        return graph_mask, {
            "enabled": True,
            "graph_weight": float(graph_weight),
            "requested_edges": int(len(edges) if max_edges > 0 else len(hierarchy.edges)),
            "used_edges": used_edges,
            "requested_max_edges": max_edges,
            "pad_cells": pad_cells,
            "cell_count": int(np.count_nonzero(graph_mask)),
            "top_edges": top_edges[: max(1, min(8, used_edges))],
            "top_weights": sorted(top_weights, reverse=True)[: max(1, min(8, len(top_weights)))],
            "min_weight": float(min(top_weights)) if top_weights else 0.0,
            "max_weight": float(max(top_weights)) if top_weights else 0.0,
        }

    def _log_swap_graph_mask(mask_info: dict[str, object]) -> None:
        if not mask_info:
            return
        if bool(mask_info.get("enabled", False)):
            _log(
                "  [hier] swap graph-mask: "
                f"enabled edges={int(mask_info.get('used_edges', 0))}/"
                f"{int(mask_info.get('requested_edges', 0))}, "
                f"cells={int(mask_info.get('cell_count', 0))}, "
                f"pad_cells={int(mask_info.get('pad_cells', 0))}"
            )
        else:
            _log(
                "  [hier] swap graph-mask disabled: "
                f"reason={str(mask_info.get('reason', 'disabled'))}, "
                f"graph_weight={float(mask_info.get('graph_weight', 0.0)):.4f}"
            )

    pre_relief = s_score
    region = None
    soft_region = None
    nr, nc = int(benchmark.grid_rows), int(benchmark.grid_cols)
    base_heat_field = _congestion_field(scorer, nr, nc)
    cluster_heat = None
    swap_mask, swap_mask_info = _build_graph_swap_mask(
        hard_xy=legal,
        graph_weight=float(graph_tension_swap_weight),
    )
    _log_swap_graph_mask(swap_mask_info)

    def _cluster_heat(field):
        if field is None or not clusters:
            return None
        cell_w, cell_h = float(cw) / nc, float(ch) / nr
        out = {}
        for cid, mem in clusters.items():
            mem = np.asarray(mem, dtype=np.int64)
            if mem.size == 0:
                continue
            ci = np.clip((legal[mem, 0] / cell_w).astype(np.int64), 0, nc - 1)
            ri = np.clip((legal[mem, 1] / cell_h).astype(np.int64), 0, nr - 1)
            out[int(cid)] = float(field[ri, ci].mean())
        return out

    if hier_region_heat_frac > 0.0:
        cluster_heat = _cluster_heat(base_heat_field)
    region = hierarchy.hard_regions(
        legal,
        sizes[:n],
        hw,
        hh,
        cw,
        ch,
        n,
        cluster_heat=cluster_heat,
        heat_expand_frac=hier_region_heat_frac,
        heat_hot_percentile=hier_region_heat_pct,
        heat_escape_min=hier_region_heat_escape,
    )
    soft_region = hierarchy.soft_regions(
        legal,
        s_pos,
        sizes[:n],
        hw,
        hh,
        soft_hw,
        soft_hh,
        cw,
        ch,
        n,
        cluster_heat=cluster_heat,
        heat_expand_frac=hier_region_heat_frac,
        heat_hot_percentile=hier_region_heat_pct,
        heat_escape_min=hier_region_heat_escape,
    )
    expand_field = base_heat_field
    weak_hot_enabled = bool(getattr(const, "HIER_REGION_WEAK_HOT_RESHAPE", False))
    weak_hot_small_shape = bool(
        int(n) >= int(const.HIER_REGION_WEAK_HOT_HARD_MIN)
        and int(n) <= int(const.HIER_REGION_WEAK_HOT_HARD_MAX)
        and int(n) + int(n_soft) <= int(const.HIER_REGION_WEAK_HOT_MACRO_MAX)
        and bool(np.all(np.asarray(movable[:n], dtype=bool)))
    )
    weak_hot_candidate_clusters = None
    if weak_hot_enabled and weak_hot_small_shape:
        confidence = getattr(hierarchy, "cluster_confidence", None) or {}
        weakest_k = max(0, int(const.HIER_SMALL_DESIGN_RELEASE_WEAKEST_K))
        if weakest_k > 0:
            weakest_rows = sorted(
                (
                    (float(conf), int(cid))
                    for cid, conf in confidence.items()
                    if int(cid) in clusters
                    and float(conf) <= float(const.HIER_REGION_WEAK_CONFIDENCE_MAX)
                ),
                key=lambda item: (item[0], item[1]),
            )[:weakest_k]
            weak_hot_candidate_clusters = [int(cid) for _conf, cid in weakest_rows]
    if weak_hot_enabled and not weak_hot_candidate_clusters:
        weak_hot_enabled = False
    region, soft_region, n_expanded = expand_regions_by_congestion(
        region,
        soft_region,
        legal,
        s_pos,
        clusters,
        csofts,
        bridge_softs,
        hw,
        hh,
        soft_hw,
        soft_hh,
        cw,
        ch,
        expand_field,
        hot_percentile=float(const.HIER_REGION_EXPAND_HOT_PCT),
        max_expand_frac=float(const.HIER_REGION_EXPAND_FRAC),
        side_band=max(1, int(const.HIER_REGION_EXPAND_BAND)),
        cluster_confidence=(
            hierarchy.cluster_confidence if weak_hot_enabled and weak_hot_small_shape else None
        ),
        weak_confidence_max=float(const.HIER_REGION_WEAK_CONFIDENCE_MAX),
        weak_hot_extra_frac=float(const.HIER_REGION_WEAK_HOT_EXTRA_FRAC),
        weak_hot_max_clusters=max(0, int(const.HIER_REGION_WEAK_HOT_MAX_CLUSTERS)),
        weak_hot_side_floor=float(const.HIER_REGION_WEAK_HOT_SIDE_FLOOR),
        weak_candidate_clusters=weak_hot_candidate_clusters,
        component_cold_percentile=float(getattr(const, "HIER_REGION_COMPONENT_COLD_PCT", 45.0)),
        component_min_cells=max(1, int(getattr(const, "HIER_REGION_COMPONENT_MIN_CELLS", 4))),
        component_max_distance_cells=max(
            0,
            int(getattr(const, "HIER_REGION_COMPONENT_MAX_DISTANCE_CELLS", 4)),
        ),
        graph_edges=hierarchy.edges if graph_tension_enabled else None,
        graph_component_weight=(
            max(
                0.0,
                float(
                    os.environ.get(
                        "HIER_REGION_GRAPH_COMPONENT_WEIGHT",
                        str(getattr(const, "HIER_REGION_GRAPH_COMPONENT_WEIGHT", 0.0)),
                    )
                ),
            )
            if graph_tension_enabled
            else 0.0
        ),
    )
    region_expand_stats = getattr(expand_regions_by_congestion, "last_stats", {})
    if n_expanded:
        weak_hot_reshaped = int(region_expand_stats.get("weak_hot_reshaped", 0))
        component_expanded = int(region_expand_stats.get("component_expanded", 0))
        suffix_parts = []
        if weak_hot_reshaped:
            suffix_parts.append(f"weak_hot={weak_hot_reshaped}")
        if component_expanded:
            suffix_parts.append(f"component={component_expanded}")
        graph_component_expanded = int(region_expand_stats.get("graph_component_expanded", 0))
        if graph_component_expanded:
            suffix_parts.append(f"graph_component={graph_component_expanded}")
        suffix = f", {', '.join(suffix_parts)}" if suffix_parts else ""
        _log(f"  [hier] congestion-expanded regions: {n_expanded} clusters{suffix}")
    bias = float(const.REGION_BIAS)
    escape_min = float(const.HIER_REGION_ESCAPE_MIN)
    rounds = max(1, int(const.HIER_REGION_ROUNDS))
    rdeadline = _deadline(float(const.HIER_REGION_BUDGET_S))
    h_pos = legal.copy()
    full = np.vstack([h_pos, s_pos]).astype(np.float64)
    r_score = float(_exact_proxy(torch.tensor(full, dtype=torch.float32), benchmark, plc))
    rscorer = IncrementalScorer(plc, benchmark, full.copy())
    best_h, best_s, best_score = h_pos.copy(), s_pos.copy(), r_score
    audit_budget = max(
        0.0,
        float(getattr(const, "HIER_FINAL_HIER_AUDIT_MAX_DEGRADATION", 0.0)),
    )
    audit_limit = float(seed_hierarchy_quality) + audit_budget
    audit_h, audit_s, audit_score = h_pos.copy(), s_pos.copy(), float(r_score)
    audit_quality = float(seed_hierarchy_quality)

    def _hierarchy_vector(hard_xy, soft_xy):
        return hierarchy_quality_vector(
            hard_xy,
            soft_xy,
            clusters,
            csofts,
            bridge_softs,
            hierarchy.edges,
            cw,
            ch,
        )

    def _vector_contract(hard_xy, soft_xy):
        vector = _hierarchy_vector(hard_xy, soft_xy)
        passed, violations = hierarchy_vector_contract(vector, seed_hierarchy_vector_limits)
        return passed, vector, violations

    def _hard_valid(hard_xy):
        if hard_xy.shape[0] == 0:
            return True
        if np.any(hard_xy[:, 0] < hw - 1e-6) or np.any(hard_xy[:, 0] > cw - hw + 1e-6):
            return False
        if np.any(hard_xy[:, 1] < hh - 1e-6) or np.any(hard_xy[:, 1] > ch - hh + 1e-6):
            return False
        dx = np.abs(hard_xy[:, None, 0] - hard_xy[None, :, 0])
        dy = np.abs(hard_xy[:, None, 1] - hard_xy[None, :, 1])
        ok = (dx + 1e-6 >= (hw[:, None] + hw[None, :])) | (dy + 1e-6 >= (hh[:, None] + hh[None, :]))
        np.fill_diagonal(ok, True)
        return bool(ok.all())

    def _maybe_update_audit_checkpoint(hard_xy, soft_xy, score):
        nonlocal audit_h, audit_s, audit_score, audit_quality
        if not _hard_valid(hard_xy):
            return False
        quality = hierarchy_quality_metric(hard_xy, clusters)
        if quality > audit_limit:
            return False
        vector_passed, vector, _violations = _vector_contract(hard_xy, soft_xy)
        if not vector_passed:
            return False
        if float(score) < float(audit_score) - 1e-9 or audit_quality > audit_limit:
            audit_h, audit_s, audit_score = hard_xy.copy(), soft_xy.copy(), float(score)
            audit_quality = float(quality)
            return True
        return False

    def _restore_audit_checkpoint(label: str) -> bool:
        nonlocal h_pos, s_pos, r_score, rscorer
        if audit_quality > audit_limit or not _hard_valid(audit_h):
            return False
        old_quality = hierarchy_quality_metric(h_pos, clusters)
        vector_passed, _old_vector, violations = _vector_contract(h_pos, s_pos)
        if old_quality <= audit_limit and vector_passed:
            return False
        h_pos = audit_h.copy()
        s_pos = audit_s.copy()
        r_score = float(audit_score)
        rscorer = IncrementalScorer(
            plc,
            benchmark,
            np.vstack([h_pos, s_pos]).astype(np.float64),
        )
        _log(
            f"  [hier] audit checkpoint restore after {label}: "
            f"hard_quality={old_quality:.5f}/{audit_limit:.5f}, "
            f"vector_violations={','.join(sorted(violations)) or 'none'}"
        )
        return True

    def _enforce_audit_checkpoint(label: str) -> bool:
        _maybe_update_audit_checkpoint(h_pos, s_pos, r_score)
        return _restore_audit_checkpoint(label)

    for round_idx in range(rounds):
        if rdeadline is not None and time.monotonic() >= rdeadline:
            break
        round_start = float(r_score)
        before_micro = r_score
        micro_acc = 0
        micro_t0 = time.monotonic()
        for use_density in (False, True):
            micro_before = float(r_score)
            h_pos, s_pos, got, r_score = _micro_shift_polish(
                h_pos,
                s_pos,
                sizes[:n],
                hw,
                hh,
                soft_hw,
                soft_hh,
                cw,
                ch,
                movable[:n],
                soft_mov,
                n,
                plc,
                benchmark,
                rscorer,
                r_score,
                hard_region=region,
                soft_region=soft_region,
                deadline=rdeadline,
                radius_cells=hier_micro_shift_radius,
                top_hot=hier_micro_shift_top,
                min_gain=hier_micro_shift_min_gain,
                use_density=use_density,
            )
            micro_acc += got
            if not _adaptive_pass_gain(micro_before, r_score):
                break
        if micro_acc:
            _log(
                f"  [hier] micro-shift polish: {micro_acc} accepts, "
                f"proxy {before_micro:.4f}->{r_score:.4f}"
            )
        _record_plateau(
            "micro_shift",
            before_micro,
            r_score,
            micro_acc,
            time.monotonic() - micro_t0,
            round=int(round_idx),
        )
        _enforce_audit_checkpoint("micro_shift")
        reloc_acc = 0
        reloc_stats = _empty_pass_stats()
        before_reloc = r_score
        reloc_t0 = time.monotonic()
        for use_density in (False, True):
            reloc_before = float(r_score)
            h_pos, got, r_score = _relocation_moves(
                h_pos,
                sizes[:n],
                hw,
                hh,
                cw,
                ch,
                movable[:n],
                n,
                plc,
                benchmark,
                rscorer,
                r_score,
                deadline=rdeadline,
                top_hot=128,
                n_targets=16,
                use_density=use_density,
                propose_all=False,
                propose_top_m=None,
                region_bbox=region,
                region_bias=bias,
                region_escape_min=escape_min,
            )
            reloc_acc += got
            _accum_pass_stats(
                reloc_stats,
                getattr(_relocation_moves, "last_stats", {}),
            )
            if not _adaptive_pass_gain(reloc_before, r_score):
                break
        _record_plateau(
            "region_hard_relocation",
            before_reloc,
            r_score,
            reloc_acc,
            time.monotonic() - reloc_t0,
            candidates=reloc_stats["candidates"],
            legal=reloc_stats["legal"],
            scored=reloc_stats["scored"],
            round=int(round_idx),
        )
        _enforce_audit_checkpoint("region_hard_relocation")
        soft_reloc_acc = 0
        soft_reloc_stats = _empty_pass_stats()
        before_soft_reloc = r_score
        soft_reloc_t0 = time.monotonic()
        for use_density in (False, True):
            soft_before = float(r_score)
            s_pos, got, r_score = _soft_relocation_moves(
                s_pos,
                soft_hw,
                soft_hh,
                cw,
                ch,
                n,
                plc,
                benchmark,
                rscorer,
                r_score,
                deadline=rdeadline,
                top_hot=1024,
                n_targets=6,
                soft_movable=soft_mov,
                use_density=use_density,
                region_bbox=soft_region,
                region_bias=bias,
                region_escape_min=escape_min,
                accept_min_gain=hier_soft_barrier_gain,
            )
            soft_reloc_acc += got
            _accum_pass_stats(
                soft_reloc_stats,
                getattr(_soft_relocation_moves, "last_stats", {}),
            )
            if not _adaptive_pass_gain(soft_before, r_score):
                break
        _record_plateau(
            "region_soft_relocation",
            before_soft_reloc,
            r_score,
            soft_reloc_acc,
            time.monotonic() - soft_reloc_t0,
            candidates=soft_reloc_stats["candidates"],
            legal=soft_reloc_stats["legal"],
            scored=soft_reloc_stats["scored"],
            round=int(round_idx),
        )
        _enforce_audit_checkpoint("region_soft_relocation")
        round_accepts = int(micro_acc + reloc_acc + soft_reloc_acc)
        if not _adaptive_pass_gain(round_start, r_score):
            _log(
                f"  [hier] region-round adaptation: early exit after round {round_idx}, "
                f"gain={round_start - float(r_score):.4f}, accepts={round_accepts}"
            )
            break
        if _hard_valid(h_pos) and r_score < best_score - 1e-9:
            best_h, best_s, best_score = h_pos.copy(), s_pos.copy(), r_score
        _enforce_audit_checkpoint("region_round")
    pre_decomp_score = r_score
    decomp_gap = float("inf")
    decomp_skip = False
    decomp_field = _congestion_field(rscorer, nr, nc)
    if decomp_field is not None and clusters:
        cell_w, cell_h = cw / nc, ch / nr
        heats = []
        for mem in clusters.values():
            mem = np.asarray(mem, dtype=np.int64)
            mem = mem[(mem >= 0) & (mem < n)]
            if mem.size == 0:
                continue
            ci = np.clip((h_pos[mem, 0] / cell_w).astype(np.int64), 0, nc - 1)
            ri = np.clip((h_pos[mem, 1] / cell_h).astype(np.int64), 0, nr - 1)
            heats.append(float(decomp_field[ri, ci].mean()))
        if heats:
            decomp_gap = float(max(heats) - np.min(decomp_field))
            decomp_skip = decomp_gap < float(const.HIER_DECOMPRESS_MIN_FIELD_GAP)
    if decomp_skip:
        hq = hierarchy_quality_metric(h_pos, clusters)
        _log(
            f"  [hier] cluster decompression skipped: "
            f"field_gap={decomp_gap:.4f} < {float(const.HIER_DECOMPRESS_MIN_FIELD_GAP):.4f}"
        )
        _record_plateau(
            "cluster_decompression",
            pre_decomp_score,
            r_score,
            0,
            0.0,
            quality=float(hq),
            skipped_by_field_gate=True,
            field_gap=float(decomp_gap),
        )
    else:
        decomp_min_gain = max(
            float(const.HIER_DECOMPRESS_MIN_GAIN),
            adaptive_floor_proxy_gain,
        )
        d_deadline = _deadline(float(const.HIER_DECOMPRESS_BUDGET_S), rdeadline)
        decomp_t0 = time.monotonic()
        d_acc = 0
        decomp_rounds = max(1, int(const.HIER_DECOMPRESS_ROUNDS))
        decomp_hq = hierarchy_quality_metric(h_pos, clusters)
        decomp_priority = _graph_tension(h_pos, decomp_field)
        for _ in range(decomp_rounds):
            if d_deadline is not None and time.monotonic() >= d_deadline:
                break
            decomp_before = float(r_score)
            trial_h, trial_s, trial_acc, trial_score, trial_hq = _cluster_decompression_relief(
                h_pos,
                s_pos,
                sizes[:n],
                hw,
                hh,
                soft_hw,
                soft_hh,
                cw,
                ch,
                movable[:n],
                soft_mov,
                n,
                clusters,
                csofts,
                bridge_softs,
                region,
                soft_region,
                plc,
                benchmark,
                decomp_before,
                deadline=d_deadline,
                rounds=1,
                hot_percentile=float(const.HIER_DECOMPRESS_HOT_PCT),
                quality_budget=float(const.HIER_QUALITY_BUDGET),
                min_proxy_gain=decomp_min_gain,
                anisotropic=True,
                anisotropic_band=max(1, int(const.HIER_DECOMPRESS_ANISO_BAND)),
                anisotropic_secondary=float(const.HIER_DECOMPRESS_ANISO_SECONDARY),
                local_component_cold_percentile=float(
                    getattr(const, "HIER_DECOMPRESS_LOCAL_COLD_PCT", 45.0)
                ),
                local_component_min_cells=max(
                    1,
                    int(getattr(const, "HIER_DECOMPRESS_LOCAL_MIN_CELLS", 4)),
                ),
                local_component_max_distance_cells=max(
                    0,
                    int(getattr(const, "HIER_DECOMPRESS_LOCAL_MAX_DISTANCE_CELLS", 4)),
                ),
                local_component_shift_frac=max(
                    0.0,
                    float(getattr(const, "HIER_DECOMPRESS_LOCAL_SHIFT_FRAC", 0.0)),
                ),
                cluster_priority=decomp_priority,
                cluster_priority_weight=graph_tension_decomp_weight,
                graph_edges=hierarchy.edges,
                seed_hard_xy=seed_hard_for_tension,
                graph_confidence=hierarchy.cluster_confidence,
            )
            weak_decomp = bool(trial_acc) and float(trial_score) > decomp_before - decomp_min_gain
            if not _hard_valid(trial_h) or weak_decomp or int(trial_acc) <= 0:
                break
            h_pos, s_pos = trial_h, trial_s
            r_score = float(trial_score)
            decomp_hq = float(trial_hq)
            d_acc += int(trial_acc)
            if _hard_valid(h_pos) and r_score < best_score - 1e-9:
                best_h, best_s, best_score = h_pos.copy(), s_pos.copy(), r_score
            _enforce_audit_checkpoint("cluster_decompression")
            if not _adaptive_pass_gain(decomp_before, r_score):
                break
        if d_acc and _hard_valid(h_pos):
            full = np.vstack([h_pos, s_pos]).astype(np.float64)
            rscorer = IncrementalScorer(plc, benchmark, full.copy())
            if _hard_valid(h_pos) and r_score < best_score - 1e-9:
                best_h, best_s, best_score = h_pos.copy(), s_pos.copy(), r_score
            _enforce_audit_checkpoint("cluster_decompression")
            _log(
                f"  [hier] cluster decompression: {d_acc} accepts, "
                f"quality={decomp_hq:.4f}, proxy={r_score:.4f}"
            )
            _record_plateau(
                "cluster_decompression",
                pre_decomp_score,
                r_score,
                d_acc,
                time.monotonic() - decomp_t0,
                quality=float(decomp_hq),
                rolled_back=False,
            )
    if (
        n_soft
        and bool(np.any(soft_mov))
        and _has_spare(rdeadline, float(const.HIER_INTERLEAVED_SOFT_REPAIR_MIN_SPARE_S))
    ):
        interleaved_soft_min_gain = max(
            float(const.HIER_INTERLEAVED_SOFT_REPAIR_MIN_GAIN),
            adaptive_floor_proxy_gain,
        )
        inter_soft_deadline = _deadline(
            float(const.HIER_INTERLEAVED_SOFT_REPAIR_BUDGET_S), rdeadline
        )
        pre_inter_soft_score = r_score
        inter_soft_acc = 0
        inter_soft_stats = _empty_pass_stats()
        inter_soft_t0 = time.monotonic()
        for use_density in (False, True):
            inter_before = float(r_score)
            s_pos, got, r_score = _soft_relocation_moves(
                s_pos,
                soft_hw,
                soft_hh,
                cw,
                ch,
                n,
                plc,
                benchmark,
                rscorer,
                r_score,
                deadline=inter_soft_deadline,
                top_hot=max(1, int(const.HIER_INTERLEAVED_SOFT_REPAIR_TOP_K)),
                n_targets=max(1, int(const.HIER_INTERLEAVED_SOFT_REPAIR_TARGETS)),
                soft_movable=soft_mov,
                use_density=use_density,
                region_bbox=soft_region,
                region_bias=bias,
                region_escape_min=escape_min,
                accept_min_gain=max(
                    float(interleaved_soft_min_gain),
                    hier_soft_barrier_gain,
                ),
                wl_prefilter=float(const.HIER_STRONG_SOFT_REPAIR_WL_PREFILTER),
            )
            inter_soft_acc += got
            _accum_pass_stats(
                inter_soft_stats,
                getattr(_soft_relocation_moves, "last_stats", {}),
            )
            if not _adaptive_pass_gain(inter_before, r_score):
                break
            if inter_soft_deadline is not None and time.monotonic() >= inter_soft_deadline:
                break
        if _hard_valid(h_pos) and r_score < best_score - 1e-9:
            best_h, best_s, best_score = h_pos.copy(), s_pos.copy(), r_score
        _enforce_audit_checkpoint("interleaved_soft_repair")
        _log(
            f"  [hier] interleaved soft repair: {inter_soft_acc} accepts, "
            f"proxy {pre_inter_soft_score:.4f}->{r_score:.4f}"
        )
        _record_plateau(
            "interleaved_soft_repair",
            pre_inter_soft_score,
            r_score,
            inter_soft_acc,
            time.monotonic() - inter_soft_t0,
            candidates=inter_soft_stats["candidates"],
            legal=inter_soft_stats["legal"],
            scored=inter_soft_stats["scored"],
            quality=hierarchy_quality_metric(h_pos, clusters),
        )
    swap_record: PlateauTelemetry | None = None
    post_hard_record: PlateauTelemetry | None = None
    plateau_escape_ran = False
    swap_rounds = max(1, int(const.HIER_REGION_SWAP_ROUNDS))
    swap_deadline = _deadline(float(const.HIER_REGION_SWAP_BUDGET_S), rdeadline)
    hard_k = max(1, int(const.HIER_HARD_SWAP_K))
    soft_k = max(1, int(const.HIER_SOFT_SWAP_K))
    if _additive_spare(swap_deadline):
        extra_k = max(0, int(const.HIER_ADDITIVE_SWAP_EXTRA_K))
        hard_k += extra_k
        soft_k += extra_k
    swap_min_gain = max(float(const.HIER_SWAP_MIN_GAIN), adaptive_floor_proxy_gain)
    swap_min_field = float(const.HIER_SWAP_MIN_FIELD_RELIEF)
    enable_hh = True
    enable_hs = True
    enable_ss = True
    swap_stats = {
        "hh_scores": 0,
        "hh_accepts": 0,
        "hh_escape_accepts": 0,
        "hs_scores": 0,
        "hs_accepts": 0,
        "hs_escape_accepts": 0,
        "ss_scores": 0,
        "ss_accepts": 0,
        "ss_escape_accepts": 0,
        "proxy_gain": 0.0,
    }
    swap_acc = 0
    swap_round_micro_acc = 0
    pre_swap_score = r_score
    swap_t0 = time.monotonic()
    swap_fallback_used = False
    fields = (False, True)
    for _swap_round in range(swap_rounds):
        if swap_deadline is not None and time.monotonic() >= swap_deadline:
            break
        swap_round_start = float(r_score)
        swap_round_start_acc = int(swap_acc)
        swap_round_masked_accepts = 0
        for use_density in fields:
            swap_before = float(r_score)
            swap_field = (
                _congestion_field(rscorer, nr, nc) if graph_tension_swap_weight > 0.0 else None
            )
            hard_graph_priority = (
                _hard_graph_tension(h_pos, swap_field)
                if graph_tension_swap_weight > 0.0 and swap_field is not None
                else None
            )
            h_pos, s_pos, got, r_score, stats = _region_bounded_swap_relief(
                h_pos,
                s_pos,
                sizes[:n],
                hw,
                hh,
                soft_hw,
                soft_hh,
                cw,
                ch,
                movable[:n],
                soft_mov,
                benchmark,
                rscorer,
                r_score,
                region,
                soft_region,
                deadline=swap_deadline,
                rounds=1,
                hard_k=hard_k,
                soft_k=soft_k,
                region_bias=bias,
                escape_min=escape_min,
                min_gain=swap_min_gain,
                soft_barrier_gain=hier_soft_barrier_gain,
                min_field_relief=swap_min_field,
                enable_hh=enable_hh,
                enable_hs=enable_hs,
                enable_ss=enable_ss,
                use_density=use_density,
                hierarchy_quality_fn=lambda cand_h: hierarchy_quality_metric(cand_h, clusters),
                hierarchy_quality_limit=audit_limit,
                hard_priority=hard_graph_priority,
                priority_weight=graph_tension_swap_weight,
                region_mask=swap_mask,
                graph_clusters=clusters,
                graph_labels=labels,
                graph_edges=hierarchy.edges,
                graph_confidence=hierarchy.cluster_confidence,
                seed_hard_xy=seed_hard_for_tension,
                graph_delta_weight=graph_swap_delta_weight,
                graph_delta_samples=graph_swap_delta_samples,
                graph_mask_penalty_weight=graph_swap_mask_penalty_weight,
            )
            swap_acc += got
            swap_round_masked_accepts += int(got)
            for k, v in stats.items():
                swap_stats[k] += v
            _enforce_audit_checkpoint("region_swaps")
            if not _adaptive_pass_gain(swap_before, r_score):
                break
        if (
            swap_round_masked_accepts <= 0
            and swap_mask is not None
            and bool(graph_swap_fallback_budget_s > 0.0)
            and not swap_fallback_used
            and _has_spare(swap_deadline, graph_swap_fallback_budget_s)
        ):
            swap_fallback_used = True
            fallback_deadline = _deadline(graph_swap_fallback_budget_s, swap_deadline)
            if fallback_deadline is not None:
                fallback_before = float(r_score)
                h_pos, s_pos, got, r_score, fallback_stats = _region_bounded_swap_relief(
                    h_pos,
                    s_pos,
                    sizes[:n],
                    hw,
                    hh,
                    soft_hw,
                    soft_hh,
                    cw,
                    ch,
                    movable[:n],
                    soft_mov,
                    benchmark,
                    rscorer,
                    r_score,
                    region,
                    soft_region,
                    deadline=fallback_deadline,
                    rounds=1,
                    hard_k=hard_k,
                    soft_k=soft_k,
                    region_bias=bias,
                    escape_min=escape_min,
                    min_gain=swap_min_gain,
                    soft_barrier_gain=hier_soft_barrier_gain,
                    min_field_relief=swap_min_field,
                    enable_hh=enable_hh,
                    enable_hs=enable_hs,
                    enable_ss=enable_ss,
                    use_density=False,
                    hierarchy_quality_fn=lambda cand_h: hierarchy_quality_metric(cand_h, clusters),
                    hierarchy_quality_limit=audit_limit,
                    hard_priority=hard_graph_priority,
                    priority_weight=graph_tension_swap_weight,
                    region_mask=None,
                    graph_clusters=clusters,
                    graph_labels=labels,
                    graph_edges=hierarchy.edges,
                    graph_confidence=hierarchy.cluster_confidence,
                    seed_hard_xy=seed_hard_for_tension,
                    graph_delta_weight=graph_swap_delta_weight,
                    graph_delta_samples=graph_swap_delta_samples,
                    graph_mask_penalty_weight=0.0,
                )
                if got:
                    for k, v in fallback_stats.items():
                        swap_stats[k] += v
                swap_acc += int(got)
                _log(
                    "  [hier] region swap graph-mask fallback: "
                    f"{int(got)} accepts, proxy"
                    f" {fallback_before:.4f}->{r_score:.4f},"
                    " budget="
                    f"{float(graph_swap_fallback_budget_s):.2f}s"
                )
                _enforce_audit_checkpoint("region_swaps_graph_fallback")
        for micro_density in (False, True):
            swap_micro_before = float(r_score)
            h_pos, s_pos, got, r_score = _micro_shift_polish(
                h_pos,
                s_pos,
                sizes[:n],
                hw,
                hh,
                soft_hw,
                soft_hh,
                cw,
                ch,
                movable[:n],
                soft_mov,
                n,
                plc,
                benchmark,
                rscorer,
                r_score,
                hard_region=region,
                soft_region=soft_region,
                deadline=swap_deadline,
                radius_cells=hier_micro_shift_radius,
                top_hot=hier_micro_shift_top,
                min_gain=hier_micro_shift_min_gain,
                use_density=micro_density,
            )
            swap_round_micro_acc += got
            _enforce_audit_checkpoint("swap_round_micro_shift")
            if not _adaptive_pass_gain(swap_micro_before, r_score):
                break
        if not _adaptive_pass_gain(swap_round_start, r_score):
            _log(
                f"  [hier] swap-round adaptation: early exit after round {_swap_round}, "
                f"gain={swap_round_start - float(r_score):.4f}, "
                f"accepts={int(swap_acc - swap_round_start_acc)}"
            )
            break
    if not _hard_valid(h_pos):
        h_pos, s_pos, r_score = best_h.copy(), best_s.copy(), best_score
        full = np.vstack([h_pos, s_pos]).astype(np.float64)
        rscorer = IncrementalScorer(plc, benchmark, full.copy())
    if _hard_valid(h_pos) and r_score < best_score - 1e-9:
        best_h, best_s, best_score = h_pos.copy(), s_pos.copy(), r_score
    _enforce_audit_checkpoint("region_swaps")
    escape_accepts = (
        swap_stats["hh_escape_accepts"]
        + swap_stats["hs_escape_accepts"]
        + swap_stats["ss_escape_accepts"]
    )
    _log(
        f"  [hier] region swaps: {swap_acc} accepts "
        f"(hh {swap_stats['hh_accepts']}/{swap_stats['hh_scores']}, "
        f"hs {swap_stats['hs_accepts']}/{swap_stats['hs_scores']}, "
        f"ss {swap_stats['ss_accepts']}/{swap_stats['ss_scores']}, "
        f"esc {escape_accepts}, "
        f"gain {swap_stats['proxy_gain']:.4f}, "
        f"round_micro {swap_round_micro_acc}), proxy={r_score:.4f}"
    )
    swap_scored = int(swap_stats["hh_scores"] + swap_stats["hs_scores"] + swap_stats["ss_scores"])
    swap_record = _record_plateau(
        "region_swaps",
        pre_swap_score,
        r_score,
        swap_acc,
        time.monotonic() - swap_t0,
        candidates=swap_scored,
        legal=swap_scored,
        scored=swap_scored,
        quality=hierarchy_quality_metric(h_pos, clusters),
        stats=swap_stats,
    )
    plateau_escape_min_gain = max(
        float(const.HIER_PLATEAU_ESCAPE_MIN_GAIN),
        adaptive_floor_proxy_gain,
    )
    if n_soft and bool(np.any(soft_mov)) and _is_plateau(swap_record):
        escape_budget = float(const.HIER_PLATEAU_ESCAPE_BUDGET_S)
        escape_spare = float(const.HIER_PLATEAU_ESCAPE_MIN_SPARE_S)
        run_escape = _has_spare(rdeadline, escape_spare)
        schedule_payload = {
            "benchmark": pass_context.benchmark_name,
            "diagnostic_no_deadlines": pass_context.diagnostic_no_deadlines,
            "pass_name": "plateau_escape_soft_relocation",
            "run": bool(run_escape),
            "has_spare": bool(run_escape),
            "trigger_pass": "region_swaps",
            "budget_s": escape_budget,
            "min_spare_s": escape_spare,
        }
        log_plateau_event("hier_budget_schedule", **schedule_payload)
        if run_escape:
            plateau_escape_ran = True
            escape_deadline = _deadline(escape_budget, rdeadline)
            pre_escape_score = r_score
            escape_acc = 0
            escape_stats = _empty_pass_stats()
            escape_t0 = time.monotonic()
            for use_density in (False, True):
                pre_escape = float(r_score)
                s_pos, got, r_score = _soft_relocation_moves(
                    s_pos,
                    soft_hw,
                    soft_hh,
                    cw,
                    ch,
                    n,
                    plc,
                    benchmark,
                    rscorer,
                    r_score,
                    deadline=escape_deadline,
                    top_hot=max(1, int(const.HIER_PLATEAU_ESCAPE_SOFT_TOP_K)),
                    n_targets=max(1, int(const.HIER_PLATEAU_ESCAPE_SOFT_TARGETS)),
                    soft_movable=soft_mov,
                    use_density=use_density,
                    region_bbox=soft_region,
                    region_bias=bias,
                    region_escape_min=escape_min,
                    accept_min_gain=max(
                        float(plateau_escape_min_gain),
                        hier_soft_barrier_gain,
                    ),
                )
                escape_acc += got
                _accum_pass_stats(
                    escape_stats,
                    getattr(_soft_relocation_moves, "last_stats", {}),
                )
                if not _adaptive_pass_gain(pre_escape, r_score):
                    break
                if escape_deadline is not None and time.monotonic() >= escape_deadline:
                    break
            if _hard_valid(h_pos) and r_score < best_score - 1e-9:
                best_h, best_s, best_score = h_pos.copy(), s_pos.copy(), r_score
            _enforce_audit_checkpoint("plateau_escape_soft_relocation")
            _log(
                f"  [hier] plateau escape soft relocation: {escape_acc} accepts, "
                f"proxy {pre_escape_score:.4f}->{r_score:.4f}"
            )
            _record_plateau(
                "plateau_escape_soft_relocation",
                pre_escape_score,
                r_score,
                escape_acc,
                time.monotonic() - escape_t0,
                candidates=escape_stats["candidates"],
                legal=escape_stats["legal"],
                scored=escape_stats["scored"],
                quality=hierarchy_quality_metric(h_pos, clusters),
            )
    post_micro_deadline = _deadline(float(const.HIER_POST_SWAP_MICRO_SHIFT_BUDGET_S), rdeadline)
    pre_post_micro_score = r_score
    post_micro_acc = 0
    post_micro_t0 = time.monotonic()
    for use_density in (False, True):
        post_micro_before = float(r_score)
        h_pos, s_pos, got, r_score = _micro_shift_polish(
            h_pos,
            s_pos,
            sizes[:n],
            hw,
            hh,
            soft_hw,
            soft_hh,
            cw,
            ch,
            movable[:n],
            soft_mov,
            n,
            plc,
            benchmark,
            rscorer,
            r_score,
            hard_region=region,
            soft_region=soft_region,
            deadline=post_micro_deadline,
            radius_cells=hier_micro_shift_radius,
            top_hot=hier_micro_shift_top,
            min_gain=hier_micro_shift_min_gain,
            use_density=use_density,
        )
        post_micro_acc += got
        if not _adaptive_pass_gain(post_micro_before, r_score):
            break
    if not _hard_valid(h_pos):
        h_pos, s_pos, r_score = best_h.copy(), best_s.copy(), best_score
        full = np.vstack([h_pos, s_pos]).astype(np.float64)
        rscorer = IncrementalScorer(plc, benchmark, full.copy())
        post_micro_acc = 0
    if _hard_valid(h_pos) and r_score < best_score - 1e-9:
        best_h, best_s, best_score = h_pos.copy(), s_pos.copy(), r_score
    _enforce_audit_checkpoint("post_swap_micro_shift")
    _log(
        f"  [hier] post-swap micro-shift replay: {post_micro_acc} accepts, "
        f"proxy {pre_post_micro_score:.4f}->{r_score:.4f}"
    )
    _record_plateau(
        "post_swap_micro_shift",
        pre_post_micro_score,
        r_score,
        post_micro_acc,
        time.monotonic() - post_micro_t0,
        quality=hierarchy_quality_metric(h_pos, clusters),
    )
    superseded_soft_schedule = {
        "benchmark": pass_context.benchmark_name,
        "diagnostic_no_deadlines": pass_context.diagnostic_no_deadlines,
        "pass_name": "post_swap_soft_relocation",
        "run": False,
        "reason": "superseded_by_plateau_escape_after_zero_of_34_clean_runs",
        "reinvest_pass": "deadline_and_final_audit_headroom",
    }
    log_plateau_event("hier_budget_schedule", **superseded_soft_schedule)
    _log("  [hier] post-swap soft relocation: skipped, superseded by plateau escape")
    compound_record: PlateauTelemetry | None = None
    if (
        n_soft
        and bool(np.any(soft_mov))
        and _has_spare(rdeadline, float(const.HIER_COMPOUND_SOFT_MIN_SPARE_S))
    ):
        compound_deadline = _deadline(float(const.HIER_COMPOUND_SOFT_BUDGET_S), rdeadline)
        pre_compound_score = float(r_score)
        compound_t0 = time.monotonic()
        s_pos, compound_acc, r_score = _compound_soft_relocation(
            s_pos,
            soft_hw,
            soft_hh,
            cw,
            ch,
            n,
            benchmark,
            rscorer,
            r_score,
            cluster_softs=csofts,
            bridge_softs=bridge_softs,
            soft_bundles=hierarchy.active_soft_bundles,
            soft_movable=soft_mov,
            region_bbox=soft_region,
            candidate_allowed=lambda trial_soft: _vector_contract(h_pos, trial_soft)[0],
            deadline=compound_deadline,
            top_groups=max(1, int(const.HIER_COMPOUND_SOFT_TOP_GROUPS)),
            group_size=max(2, int(const.HIER_COMPOUND_SOFT_GROUP_SIZE)),
            cold_percentile=float(const.HIER_COMPOUND_SOFT_COLD_PCT),
            max_components=max(1, int(const.HIER_COMPOUND_SOFT_MAX_COMPONENTS)),
            min_component_cells=max(1, int(const.HIER_COMPOUND_SOFT_MIN_COMPONENT_CELLS)),
            n_anchors=max(1, int(const.HIER_COMPOUND_SOFT_ANCHORS)),
            shift_fractions=const.HIER_COMPOUND_SOFT_SHIFT_FRACTIONS,
            min_field_drop=float(const.HIER_COMPOUND_SOFT_MIN_FIELD_DROP),
            min_gain=max(float(const.HIER_COMPOUND_SOFT_MIN_GAIN), adaptive_floor_proxy_gain),
        )
        if _hard_valid(h_pos) and r_score < best_score - 1e-9:
            best_h, best_s, best_score = h_pos.copy(), s_pos.copy(), r_score
        _enforce_audit_checkpoint("compound_soft_relocation")
        compound_stats = getattr(_compound_soft_relocation, "last_stats", {})
        _log(
            f"  [hier] compound soft relocation: {compound_acc} accepts, "
            f"proxy {pre_compound_score:.4f}->{r_score:.4f}"
        )
        compound_record = _record_plateau(
            "compound_soft_relocation",
            pre_compound_score,
            r_score,
            compound_acc,
            time.monotonic() - compound_t0,
            candidates=int(compound_stats.get("candidates", 0)),
            legal=(
                int(compound_stats.get("candidates", 0))
                - int(compound_stats.get("hierarchy_rejects", 0))
                - int(compound_stats.get("field_rejects", 0))
            ),
            scored=int(compound_stats.get("scored", 0)),
            quality=hierarchy_quality_metric(h_pos, clusters),
            hierarchy_rejects=int(compound_stats.get("hierarchy_rejects", 0)),
            field_rejects=int(compound_stats.get("field_rejects", 0)),
            groups=int(compound_stats.get("groups", 0)),
            best_candidate_gain=float(compound_stats.get("best_candidate_gain", 0.0)),
        )
    post_escape_trigger = "post_swap_soft_relocation_superseded"
    if (
        not plateau_escape_ran
        and post_escape_trigger is not None
        and n_soft
        and bool(np.any(soft_mov))
    ):
        escape_budget = float(const.HIER_PLATEAU_ESCAPE_BUDGET_S)
        escape_spare = float(const.HIER_PLATEAU_ESCAPE_MIN_SPARE_S)
        run_escape = _has_spare(rdeadline, escape_spare)
        schedule_payload = {
            "benchmark": pass_context.benchmark_name,
            "diagnostic_no_deadlines": pass_context.diagnostic_no_deadlines,
            "pass_name": "plateau_escape_post_soft_relocation",
            "run": bool(run_escape),
            "has_spare": bool(run_escape),
            "trigger_pass": post_escape_trigger,
            "budget_s": escape_budget,
            "min_spare_s": escape_spare,
        }
        log_plateau_event("hier_budget_schedule", **schedule_payload)
        if run_escape:
            plateau_escape_ran = True
            escape_deadline = _deadline(escape_budget, rdeadline)
            pre_escape_score = r_score
            escape_acc = 0
            escape_stats = _empty_pass_stats()
            escape_t0 = time.monotonic()
            for use_density in (False, True):
                pre_escape = float(r_score)
                s_pos, got, r_score = _soft_relocation_moves(
                    s_pos,
                    soft_hw,
                    soft_hh,
                    cw,
                    ch,
                    n,
                    plc,
                    benchmark,
                    rscorer,
                    r_score,
                    deadline=escape_deadline,
                    top_hot=max(1, int(const.HIER_PLATEAU_ESCAPE_SOFT_TOP_K)),
                    n_targets=max(1, int(const.HIER_PLATEAU_ESCAPE_SOFT_TARGETS)),
                    soft_movable=soft_mov,
                    use_density=use_density,
                    region_bbox=soft_region,
                    region_bias=bias,
                    region_escape_min=escape_min,
                    accept_min_gain=max(
                        float(plateau_escape_min_gain),
                        hier_soft_barrier_gain,
                    ),
                )
                escape_acc += got
                _accum_pass_stats(
                    escape_stats,
                    getattr(_soft_relocation_moves, "last_stats", {}),
                )
                if not _adaptive_pass_gain(pre_escape, r_score):
                    break
                if escape_deadline is not None and time.monotonic() >= escape_deadline:
                    break
            if _hard_valid(h_pos) and r_score < best_score - 1e-9:
                best_h, best_s, best_score = h_pos.copy(), s_pos.copy(), r_score
            _enforce_audit_checkpoint("plateau_escape_post_soft_relocation")
            _log(
                f"  [hier] plateau escape post-soft relocation: {escape_acc} accepts, "
                f"proxy {pre_escape_score:.4f}->{r_score:.4f}"
            )
            _record_plateau(
                "plateau_escape_post_soft_relocation",
                pre_escape_score,
                r_score,
                escape_acc,
                time.monotonic() - escape_t0,
                candidates=escape_stats["candidates"],
                legal=escape_stats["legal"],
                scored=escape_stats["scored"],
                quality=hierarchy_quality_metric(h_pos, clusters),
            )
    if n_soft and bool(np.any(soft_mov)):
        component_bias = _component_cleanup_bias(h_pos, s_pos)
        strong_budget = float(const.HIER_STRONG_SOFT_REPAIR_BUDGET_S)
        min_spare = float(const.HIER_STRONG_SOFT_REPAIR_MIN_SPARE_S)
        has_spare = _has_spare(rdeadline, min_spare)
        cleanup_reserved = _has_spare(
            rdeadline,
            float(const.HIER_COMPONENT_RESERVED_CLEANUP_S),
        )
        component_soft_trigger = bool(
            component_bias["enabled"]
            and component_bias["congestion_dominates"]
            and cleanup_reserved
        )
        plateau_trigger = (
            _is_plateau(swap_record)
            or _is_plateau(post_hard_record)
            or _is_plateau(compound_record)
        )
        useful_soft_trigger = bool(
            compound_record is None
            or compound_record.accepts > 0
            or compound_record.proxy_gain >= float(const.HIER_STRONG_SOFT_REPAIR_MIN_GAIN)
        )
        plateau_soft_bonus = bool(
            plateau_trigger
            and _has_spare(
                rdeadline,
                float(const.HIER_PLATEAU_SOFT_REPAIR_BONUS_MIN_SPARE_S),
            )
        )
        scheduled_strong_budget = strong_budget + (
            float(const.HIER_PLATEAU_SOFT_REPAIR_BONUS_BUDGET_S) if plateau_soft_bonus else 0.0
        )
        scheduled_strong_rounds = max(1, int(const.HIER_STRONG_SOFT_REPAIR_ROUNDS)) + (
            max(0, int(const.HIER_PLATEAU_SOFT_REPAIR_BONUS_ROUNDS)) if plateau_soft_bonus else 0
        )
        total_macros = max(1, int(n) + int(n_soft))
        nets_per_macro = float(getattr(benchmark, "num_nets", 0)) / float(total_macros)
        medium_soft_shape = bool(
            int(n) >= int(const.HIER_MEDIUM_SOFT_HARD_MIN)
            and int(n) <= int(const.HIER_MEDIUM_SOFT_HARD_MAX)
            and total_macros >= int(const.HIER_MEDIUM_SOFT_MACRO_MIN)
            and total_macros <= int(const.HIER_MEDIUM_SOFT_MACRO_MAX)
            and nets_per_macro >= float(const.HIER_MEDIUM_SOFT_NETS_PER_MACRO_MIN)
            and nets_per_macro <= float(const.HIER_MEDIUM_SOFT_NETS_PER_MACRO_MAX)
        )
        run_strong_soft = has_spare and (
            plateau_trigger or useful_soft_trigger or component_soft_trigger
        )
        schedule_payload = {
            "benchmark": pass_context.benchmark_name,
            "diagnostic_no_deadlines": pass_context.diagnostic_no_deadlines,
            "pass_name": "strong_soft_repair",
            "run": bool(run_strong_soft),
            "has_spare": bool(has_spare),
            "plateau_trigger": bool(plateau_trigger),
            "plateau_soft_bonus": bool(plateau_soft_bonus),
            "useful_soft_trigger": bool(useful_soft_trigger),
            "component_soft_trigger": bool(component_soft_trigger),
            "component_wirelength": float(component_bias["wirelength"]),
            "component_density": float(component_bias["density"]),
            "component_congestion": float(component_bias["congestion"]),
            "budget_s": scheduled_strong_budget,
            "min_spare_s": min_spare,
            "rounds": int(scheduled_strong_rounds),
            "medium_soft_shape": bool(medium_soft_shape),
            "nets_per_macro": float(nets_per_macro),
        }
        log_plateau_event("hier_budget_schedule", **schedule_payload)
        if run_strong_soft:
            strong_deadline = _deadline(scheduled_strong_budget, rdeadline)
            pre_strong_score = r_score
            strong_acc = 0
            strong_stats = _empty_pass_stats()
            strong_t0 = time.monotonic()
            strong_min_gain = max(
                float(const.HIER_STRONG_SOFT_REPAIR_MIN_GAIN),
                adaptive_floor_proxy_gain,
            )
            for _strong_round in range(scheduled_strong_rounds):
                strong_round_before = float(r_score)
                for use_density in (False, True):
                    strong_before = float(r_score)
                    s_pos, got, r_score = _soft_relocation_moves(
                        s_pos,
                        soft_hw,
                        soft_hh,
                        cw,
                        ch,
                        n,
                        plc,
                        benchmark,
                        rscorer,
                        r_score,
                        deadline=strong_deadline,
                        top_hot=max(1, int(const.HIER_STRONG_SOFT_REPAIR_TOP_K)),
                        n_targets=max(1, int(const.HIER_STRONG_SOFT_REPAIR_TARGETS)),
                        soft_movable=soft_mov,
                        use_density=use_density,
                        region_bbox=soft_region,
                        region_bias=bias,
                        region_escape_min=escape_min,
                        accept_min_gain=max(
                            float(strong_min_gain),
                            hier_soft_barrier_gain,
                        ),
                        wl_prefilter=float(const.HIER_STRONG_SOFT_REPAIR_WL_PREFILTER),
                    )
                    strong_acc += got
                    _accum_pass_stats(
                        strong_stats,
                        getattr(_soft_relocation_moves, "last_stats", {}),
                    )
                    if not _adaptive_pass_gain(strong_before, r_score):
                        break
                    if strong_deadline is not None and time.monotonic() >= strong_deadline:
                        break
                if not _adaptive_pass_gain(strong_round_before, r_score):
                    break
                if strong_deadline is not None and time.monotonic() >= strong_deadline:
                    break
            if _hard_valid(h_pos) and r_score < best_score - 1e-9:
                best_h, best_s, best_score = h_pos.copy(), s_pos.copy(), r_score
            _enforce_audit_checkpoint("strong_soft_repair")
            _log(
                f"  [hier] strong soft repair: {strong_acc} accepts, "
                f"proxy {pre_strong_score:.4f}->{r_score:.4f}"
            )
            _record_plateau(
                "strong_soft_repair",
                pre_strong_score,
                r_score,
                strong_acc,
                time.monotonic() - strong_t0,
                candidates=strong_stats["candidates"],
                legal=strong_stats["legal"],
                scored=strong_stats["scored"],
                quality=hierarchy_quality_metric(h_pos, clusters),
            )
            strong_gain = max(0.0, float(pre_strong_score) - float(r_score))
            medium_has_spare = _has_spare(
                rdeadline,
                float(const.HIER_MEDIUM_SOFT_MIN_SPARE_S),
            )
            run_medium_soft = bool(
                medium_soft_shape
                and medium_has_spare
                and strong_gain >= float(const.HIER_MEDIUM_SOFT_TRIGGER_GAIN)
            )
            medium_schedule = {
                "benchmark": pass_context.benchmark_name,
                "diagnostic_no_deadlines": pass_context.diagnostic_no_deadlines,
                "pass_name": "medium_soft_continuation",
                "run": bool(run_medium_soft),
                "has_spare": bool(medium_has_spare),
                "strong_soft_gain": float(strong_gain),
                "trigger_gain": float(const.HIER_MEDIUM_SOFT_TRIGGER_GAIN),
                "medium_soft_shape": bool(medium_soft_shape),
                "num_hard": int(n),
                "num_soft": int(n_soft),
                "num_macros": int(total_macros),
                "nets_per_macro": float(nets_per_macro),
                "budget_s": float(const.HIER_MEDIUM_SOFT_BUDGET_S),
                "min_spare_s": float(const.HIER_MEDIUM_SOFT_MIN_SPARE_S),
                "rounds": int(const.HIER_MEDIUM_SOFT_ROUNDS),
            }
            log_plateau_event("hier_budget_schedule", **medium_schedule)
            if run_medium_soft:
                medium_deadline = _deadline(float(const.HIER_MEDIUM_SOFT_BUDGET_S), rdeadline)
                pre_medium_score = float(r_score)
                medium_acc = 0
                medium_stats = _empty_pass_stats()
                medium_t0 = time.monotonic()
                medium_min_gain = max(
                    float(const.HIER_MEDIUM_SOFT_MIN_GAIN),
                    adaptive_floor_proxy_gain,
                )
                for _medium_round in range(max(1, int(const.HIER_MEDIUM_SOFT_ROUNDS))):
                    medium_round_before = float(r_score)
                    for use_density in (False, True):
                        medium_before = float(r_score)
                        s_pos, got, r_score = _soft_relocation_moves(
                            s_pos,
                            soft_hw,
                            soft_hh,
                            cw,
                            ch,
                            n,
                            plc,
                            benchmark,
                            rscorer,
                            r_score,
                            deadline=medium_deadline,
                            top_hot=max(1, int(const.HIER_MEDIUM_SOFT_TOP_K)),
                            n_targets=max(1, int(const.HIER_MEDIUM_SOFT_TARGETS)),
                            soft_movable=soft_mov,
                            use_density=use_density,
                            region_bbox=soft_region,
                            region_bias=bias,
                            region_escape_min=escape_min,
                            accept_min_gain=max(
                                float(medium_min_gain),
                                hier_soft_barrier_gain,
                            ),
                            wl_prefilter=float(const.HIER_STRONG_SOFT_REPAIR_WL_PREFILTER),
                        )
                        medium_acc += got
                        _accum_pass_stats(
                            medium_stats,
                            getattr(_soft_relocation_moves, "last_stats", {}),
                        )
                        if not _adaptive_pass_gain(medium_before, r_score):
                            break
                        if medium_deadline is not None and time.monotonic() >= medium_deadline:
                            break
                    if not _adaptive_pass_gain(medium_round_before, r_score):
                        break
                    if medium_deadline is not None and time.monotonic() >= medium_deadline:
                        break
                if _hard_valid(h_pos) and r_score < best_score - 1e-9:
                    best_h, best_s, best_score = h_pos.copy(), s_pos.copy(), r_score
                _enforce_audit_checkpoint("medium_soft_continuation")
                _log(
                    f"  [hier] medium soft continuation: {medium_acc} accepts, "
                    f"proxy {pre_medium_score:.4f}->{r_score:.4f}"
                )
                _record_plateau(
                    "medium_soft_continuation",
                    pre_medium_score,
                    r_score,
                    medium_acc,
                    time.monotonic() - medium_t0,
                    candidates=medium_stats["candidates"],
                    legal=medium_stats["legal"],
                    scored=medium_stats["scored"],
                    quality=hierarchy_quality_metric(h_pos, clusters),
                    strong_soft_gain=float(strong_gain),
                    nets_per_macro=float(nets_per_macro),
                )
        else:
            reason = "insufficient_spare" if not has_spare else "no_trigger"
            _log(
                "  [hier] strong soft repair skipped: "
                f"{reason}, spare_min={min_spare:.1f}s, budget={strong_budget:.1f}s"
            )
    legal_candidate = _will_legalize(
        h_pos,
        movable[:n],
        sizes[:n],
        hw,
        hh,
        cw,
        ch,
        n,
        deadline=_deadline(30),
        order=order,
    )
    legal_score = float(
        _exact_proxy(
            torch.tensor(
                np.vstack([legal_candidate, s_pos]).astype(np.float32),
                dtype=torch.float32,
            ),
            benchmark,
            plc,
        )
    )
    if legal_score <= best_score + 1e-9:
        legal = legal_candidate
        best_h, best_s, best_score = legal.copy(), s_pos.copy(), legal_score
        _maybe_update_audit_checkpoint(legal, s_pos, legal_score)
    elif _hard_valid(best_h):
        legal, s_pos = best_h.copy(), best_s.copy()
    else:
        legal = legal_candidate
        _maybe_update_audit_checkpoint(legal, s_pos, legal_score)
    legal, s_pos, cur_proxy, _ = run_coldspot_tightening(
        benchmark=benchmark,
        plc=plc,
        clusters=clusters,
        csofts=csofts,
        bridge_softs=bridge_softs,
        movable=movable,
        n=int(n),
        n_soft=int(n_soft),
        sizes=sizes,
        hw=hw,
        hh=hh,
        soft_hw=soft_hw,
        soft_hh=soft_hh,
        cw=float(cw),
        ch=float(ch),
        region=region,
        soft_region=soft_region,
        legal=legal,
        s_pos=s_pos,
        const=const,
        log_fn=_log,
        hard_valid_fn=_hard_valid,
        deadline_fn=_deadline,
        hierarchy_quality_metric_fn=hierarchy_quality_metric,
        hier_soft_barrier_gain=float(hier_soft_barrier_gain),
        hier_micro_shift_radius=int(hier_micro_shift_radius),
        hier_micro_shift_top=int(hier_micro_shift_top),
        hier_micro_shift_min_gain=float(hier_micro_shift_min_gain),
        graph_tension_fn=_graph_tension,
        graph_tension_weight=graph_tension_coldspot_weight,
        graph_edges=hierarchy.edges,
        seed_hard_xy=seed_hard_for_tension,
        graph_confidence=hierarchy.cluster_confidence,
    )
    _maybe_update_audit_checkpoint(legal, s_pos, cur_proxy)

    return run_post_coldspot_finalize(
        benchmark=benchmark,
        plc=plc,
        clusters=clusters,
        csofts=csofts,
        bridge_softs=bridge_softs,
        hierarchy=hierarchy,
        hierarchy_quality_metric_fn=hierarchy_quality_metric,
        selected_seed_name=selected_seed_name,
        pre_relief=pre_relief,
        seed_hierarchy_quality=float(seed_hierarchy_quality),
        seed_hierarchy_vector=seed_hierarchy_vector,
        legal=legal,
        s_pos=s_pos,
        cur_proxy=cur_proxy,
        best_h=best_h,
        best_s=best_s,
        best_score=best_score,
        audit_h=audit_h,
        audit_s=audit_s,
        audit_score=audit_score,
        movable=movable,
        n=int(n),
        n_soft=int(n_soft),
        sizes=sizes,
        hw=hw,
        hh=hh,
        soft_hw=soft_hw,
        soft_hh=soft_hh,
        cw=float(cw),
        ch=float(ch),
        region=region,
        soft_region=soft_region,
        const=const,
        group_weight=int(gw),
        log_fn=_log,
        record_plateau_fn=_record_plateau,
        hard_valid_fn=_hard_valid,
        deadline_fn=_deadline,
    )
