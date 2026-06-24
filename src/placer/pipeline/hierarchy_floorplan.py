"""Hierarchy floorplan pipeline segment extracted from macro_placer."""

import os
import time

import numpy as np
import torch
from macro_place.benchmark import Benchmark

from utils import constants as const
from utils.config import _GPU_BACKEND, _log
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
    from placer.local_search.fields import _congestion_field
    from placer.local_search.gnn_trace import (
        log_gnn_event,
        log_plateau_event,
    )
    from placer.local_search.hierarchy_swaps import _region_bounded_swap_relief
    from placer.local_search.hierarchy_model import HierarchyModel
    from placer.local_search.region_expand import expand_regions_by_congestion
    from placer.local_search.relocation import (
        _micro_shift_polish,
        _relocation_moves,
        _soft_relocation_moves,
    )
    from placer.pipeline.hierarchy_context import (
        PassContext,
        PassResult,
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

    def _auto_cuda_flag(value) -> bool:
        if isinstance(value, str) and value.lower() == "auto":
            return _GPU_BACKEND == "cuda"
        return bool(value)

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

    hier_post_reloc_top_m = const.HIER_POST_RELOC_PROPOSE_TOP_M
    hier_reloc_propose_hot_k = max(1, int(const.HIER_RELOC_PROPOSE_HOT_K))
    hier_post_reloc_propose = _auto_cuda_flag(const.HIER_POST_RELOC_PROPOSE_ALL)
    hier_post_soft_reloc_top_k = max(1, int(const.HIER_POST_SOFT_RELOC_TOP_K))
    hier_post_soft_reloc_min_gain = float(const.HIER_POST_SOFT_RELOC_MIN_GAIN)
    hier_reloc_propose_min_gain = float(const.HIER_RELOC_PROPOSE_MIN_GAIN)
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
        canvas_width=cw,
        canvas_height=ch,
        num_hard=n,
        num_soft=n_soft,
        diagnostic_no_deadlines=bool(diagnostic_no_deadlines),
    )

    hierarchy = HierarchyModel.build(plc, n, n_soft, hard_sizes=sizes[:n])
    labels = hierarchy.labels
    clusters = hierarchy.clusters
    csofts = hierarchy.cluster_softs
    bridge_softs = hierarchy.bridge_softs

    def _trace_pass(pass_name: str, before: float, after: float, accepts: int, **extra) -> None:
        quality = extra.pop("quality", None)
        result = PassResult(
            name=pass_name,
            proxy_before=float(before),
            proxy_after=float(after),
            accepts=int(accepts),
            quality=quality,
            extra=extra,
        )
        log_gnn_event(
            "hier_pass_result",
            benchmark=pass_context.benchmark_name,
            diagnostic_no_deadlines=pass_context.diagnostic_no_deadlines,
            **result.to_trace_kwargs(),
        )

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
        log_gnn_event(
            "hier_plateau_telemetry",
            benchmark=pass_context.benchmark_name,
            diagnostic_no_deadlines=pass_context.diagnostic_no_deadlines,
            **record.to_trace_kwargs(),
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

    adaptive_passes = (
        os.environ.get("HIER_ADAPTIVE_PASSES", "1").strip().lower()
        not in {"0", "false", "no", "off", "disable"}
    )
    adaptive_floor_proxy_gain = float(const.HIER_PLATEAU_PROXY_GAIN)
    if adaptive_passes:
        hier_micro_shift_min_gain = max(
            float(hier_micro_shift_min_gain),
            adaptive_floor_proxy_gain,
        )

    def _adaptive_pass_gain(before: float, after: float) -> bool:
        if not adaptive_passes:
            return True
        return float(before) - float(after) > adaptive_floor_proxy_gain

    def _has_spare(deadline: "float | None", reserve_s: float) -> bool:
        return deadline is None or time.monotonic() + float(reserve_s) < float(deadline)

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
        return torch.tensor(
            np.vstack([hard_xy, soft_xy]).astype(np.float32), dtype=torch.float32
        )

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
            hard_a = [
                b_to_a[int(r)] for r in ref_idx[start : start + length] if int(r) in b_to_a
            ]
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
    selected_seed_name = (
        str(seed_rows[0].get("name", "dreamplace")) if seed_rows else "dreamplace"
    )

    legal = hard
    s_pos = soft.copy()
    state = PlacementState(legal.copy(), s_pos.copy(), float(s_score))
    pos = state.full()
    scorer = IncrementalScorer(plc, benchmark, pos.copy())
    soft_mov = movable[n : n + n_soft]

    pre_relief = s_score
    region = None
    soft_region = None
    nr, nc = int(benchmark.grid_rows), int(benchmark.grid_cols)
    base_heat_field = _congestion_field(scorer, nr, nc)
    cluster_heat = None

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
    )
    if n_expanded:
        _log(f"  [hier] congestion-expanded regions: {n_expanded} clusters")
    bias = float(const.REGION_BIAS)
    escape_min = float(const.HIER_REGION_ESCAPE_MIN)
    rounds = max(1, int(const.HIER_REGION_ROUNDS))
    rdeadline = _deadline(float(const.HIER_REGION_BUDGET_S))
    h_pos = legal.copy()
    full = np.vstack([h_pos, s_pos]).astype(np.float64)
    r_score = float(_exact_proxy(torch.tensor(full, dtype=torch.float32), benchmark, plc))
    rscorer = IncrementalScorer(plc, benchmark, full.copy())
    best_h, best_s, best_score = h_pos.copy(), s_pos.copy(), r_score

    def _hard_valid(hard_xy):
        if hard_xy.shape[0] == 0:
            return True
        if np.any(hard_xy[:, 0] < hw - 1e-6) or np.any(hard_xy[:, 0] > cw - hw + 1e-6):
            return False
        if np.any(hard_xy[:, 1] < hh - 1e-6) or np.any(hard_xy[:, 1] > ch - hh + 1e-6):
            return False
        dx = np.abs(hard_xy[:, None, 0] - hard_xy[None, :, 0])
        dy = np.abs(hard_xy[:, None, 1] - hard_xy[None, :, 1])
        ok = (dx + 1e-6 >= (hw[:, None] + hw[None, :])) | (
            dy + 1e-6 >= (hh[:, None] + hh[None, :])
        )
        np.fill_diagonal(ok, True)
        return bool(ok.all())

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
        _trace_pass(
            "micro_shift",
            before_micro,
            r_score,
            micro_acc,
            quality=hierarchy_quality_metric(h_pos, clusters),
        )
        _record_plateau(
            "micro_shift",
            before_micro,
            r_score,
            micro_acc,
            time.monotonic() - micro_t0,
            round=int(round_idx),
        )
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
        round_accepts = int(micro_acc + reloc_acc + soft_reloc_acc)
        if adaptive_passes and not _adaptive_pass_gain(round_start, r_score):
            _log(
                f"  [hier] region-round adaptation: early exit after round {round_idx}, "
                f"gain={round_start - float(r_score):.4f}, accepts={round_accepts}"
            )
            break
        if _hard_valid(h_pos) and r_score < best_score - 1e-9:
            best_h, best_s, best_score = h_pos.copy(), s_pos.copy(), r_score
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
        _trace_pass(
            "cluster_decompression",
            pre_decomp_score,
            r_score,
            0,
            quality=float(hq),
            skipped_by_field_gate=True,
            field_gap=float(decomp_gap),
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
        decomp_min_gain = float(const.HIER_DECOMPRESS_MIN_GAIN)
        if adaptive_passes:
            decomp_min_gain = max(decomp_min_gain, adaptive_floor_proxy_gain)
        d_deadline = _deadline(float(const.HIER_DECOMPRESS_BUDGET_S), rdeadline)
        decomp_t0 = time.monotonic()
        d_acc = 0
        decomp_rounds = max(1, int(const.HIER_DECOMPRESS_ROUNDS))
        decomp_hq = hierarchy_quality_metric(h_pos, clusters)
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
            if not _adaptive_pass_gain(decomp_before, r_score):
                break
        if d_acc and _hard_valid(h_pos):
            full = np.vstack([h_pos, s_pos]).astype(np.float64)
            rscorer = IncrementalScorer(plc, benchmark, full.copy())
            if _hard_valid(h_pos) and r_score < best_score - 1e-9:
                best_h, best_s, best_score = h_pos.copy(), s_pos.copy(), r_score
            _log(
                f"  [hier] cluster decompression: {d_acc} accepts, "
                f"quality={decomp_hq:.4f}, proxy={r_score:.4f}"
            )
            _trace_pass(
                "cluster_decompression",
                pre_decomp_score,
                r_score,
                d_acc,
                quality=float(decomp_hq),
                rolled_back=False,
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
        interleaved_soft_min_gain = float(const.HIER_INTERLEAVED_SOFT_REPAIR_MIN_GAIN)
        if adaptive_passes:
            interleaved_soft_min_gain = max(
                interleaved_soft_min_gain,
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
        _log(
            f"  [hier] interleaved soft repair: {inter_soft_acc} accepts, "
            f"proxy {pre_inter_soft_score:.4f}->{r_score:.4f}"
        )
        _trace_pass(
            "interleaved_soft_repair",
            pre_inter_soft_score,
            r_score,
            inter_soft_acc,
            quality=hierarchy_quality_metric(h_pos, clusters),
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
    post_soft_record: PlateauTelemetry | None = None
    plateau_escape_ran = False
    swap_rounds = max(1, int(const.HIER_REGION_SWAP_ROUNDS))
    swap_deadline = _deadline(float(const.HIER_REGION_SWAP_BUDGET_S), rdeadline)
    hard_k = max(1, int(const.HIER_HARD_SWAP_K))
    soft_k = max(1, int(const.HIER_SOFT_SWAP_K))
    if _additive_spare(swap_deadline):
        extra_k = max(0, int(const.HIER_ADDITIVE_SWAP_EXTRA_K))
        hard_k += extra_k
        soft_k += extra_k
    swap_min_gain = float(const.HIER_SWAP_MIN_GAIN)
    if adaptive_passes:
        swap_min_gain = max(swap_min_gain, adaptive_floor_proxy_gain)
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
    fields = (False, True)
    for _swap_round in range(swap_rounds):
        if swap_deadline is not None and time.monotonic() >= swap_deadline:
            break
        swap_round_start = float(r_score)
        swap_round_start_acc = int(swap_acc)
        for use_density in fields:
            swap_before = float(r_score)
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
            )
            swap_acc += got
            for k, v in stats.items():
                swap_stats[k] += v
            if not _adaptive_pass_gain(swap_before, r_score):
                break
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
            if not _adaptive_pass_gain(swap_micro_before, r_score):
                break
        if adaptive_passes and not _adaptive_pass_gain(swap_round_start, r_score):
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
    _trace_pass(
        "region_swaps",
        pre_swap_score,
        r_score,
        swap_acc,
        stats=swap_stats,
        round_micro_accepts=int(swap_round_micro_acc),
        quality=hierarchy_quality_metric(h_pos, clusters),
    )
    swap_scored = int(
        swap_stats["hh_scores"] + swap_stats["hs_scores"] + swap_stats["ss_scores"]
    )
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
    plateau_escape_min_gain = float(const.HIER_PLATEAU_ESCAPE_MIN_GAIN)
    if adaptive_passes:
        plateau_escape_min_gain = max(plateau_escape_min_gain, adaptive_floor_proxy_gain)
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
        log_gnn_event("hier_budget_schedule", **schedule_payload)
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
                    gpu_batch_rank=True,
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
            _log(
                f"  [hier] plateau escape soft relocation: {escape_acc} accepts, "
                f"proxy {pre_escape_score:.4f}->{r_score:.4f}"
            )
            escape_record = _record_plateau(
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
            _trace_pass(
                "plateau_escape_soft_relocation",
                pre_escape_score,
                r_score,
                escape_acc,
                quality=hierarchy_quality_metric(h_pos, clusters),
                plateau=bool(_is_plateau(escape_record)),
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
    _log(
        f"  [hier] post-swap micro-shift replay: {post_micro_acc} accepts, "
        f"proxy {pre_post_micro_score:.4f}->{r_score:.4f}"
    )
    _trace_pass(
        "post_swap_micro_shift",
        pre_post_micro_score,
        r_score,
        post_micro_acc,
        quality=hierarchy_quality_metric(h_pos, clusters),
    )
    _record_plateau(
        "post_swap_micro_shift",
        pre_post_micro_score,
        r_score,
        post_micro_acc,
        time.monotonic() - post_micro_t0,
        quality=hierarchy_quality_metric(h_pos, clusters),
    )
    if hier_post_reloc_propose:
        post_reloc_propose_min_gain = float(hier_reloc_propose_min_gain)
        if adaptive_passes:
            post_reloc_propose_min_gain = max(
                post_reloc_propose_min_gain,
                adaptive_floor_proxy_gain,
            )
        post_deadline = _deadline(float(const.HIER_POST_RELOC_PROPOSE_BUDGET_S), rdeadline)
        pre_post_score = r_score
        post_t0 = time.monotonic()
        h_pos, post_acc, r_score = _relocation_moves(
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
            deadline=post_deadline,
            top_hot=hier_reloc_propose_hot_k,
            n_targets=16,
            use_density=False,
            propose_all=True,
            propose_top_m=hier_post_reloc_top_m,
            region_bbox=region,
            region_bias=bias,
            region_escape_min=escape_min,
            propose_accept_min_gain=post_reloc_propose_min_gain,
        )
        if not _hard_valid(h_pos):
            h_pos, s_pos, r_score = best_h.copy(), best_s.copy(), best_score
            full = np.vstack([h_pos, s_pos]).astype(np.float64)
            rscorer = IncrementalScorer(plc, benchmark, full.copy())
            post_acc = 0
        if _hard_valid(h_pos) and r_score < best_score - 1e-9:
            best_h, best_s, best_score = h_pos.copy(), s_pos.copy(), r_score
        _log(
            f"  [hier] post-swap hard propose-all: {post_acc} accepts, "
            f"proxy {pre_post_score:.4f}->{r_score:.4f}"
        )
        _trace_pass(
            "post_swap_hard_propose_all",
            pre_post_score,
            r_score,
            post_acc,
            quality=hierarchy_quality_metric(h_pos, clusters),
        )
        post_stats = getattr(_relocation_moves, "last_stats", {})
        post_hard_record = _record_plateau(
            "post_swap_hard_propose_all",
            pre_post_score,
            r_score,
            post_acc,
            time.monotonic() - post_t0,
            candidates=int(post_stats.get("candidates", 0)),
            legal=int(post_stats.get("legal", 0)),
            scored=int(post_stats.get("scored", 0)),
            quality=hierarchy_quality_metric(h_pos, clusters),
        )
    post_soft_deadline = _deadline(float(const.HIER_POST_SOFT_RELOC_BUDGET_S), rdeadline)
    pre_post_soft_score = r_score
    post_soft_acc = 0
    post_soft_t0 = time.monotonic()
    post_soft_min_gain = float(hier_post_soft_reloc_min_gain)
    if adaptive_passes:
        post_soft_min_gain = max(post_soft_min_gain, adaptive_floor_proxy_gain)
    post_soft_stats = _empty_pass_stats()
    for use_density in (False, True):
        post_soft_before = float(r_score)
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
            deadline=post_soft_deadline,
            top_hot=hier_post_soft_reloc_top_k,
            n_targets=6,
            soft_movable=soft_mov,
            use_density=use_density,
            region_bbox=soft_region,
            region_bias=bias,
            region_escape_min=escape_min,
            accept_min_gain=max(
                float(post_soft_min_gain),
                hier_soft_barrier_gain,
            ),
        )
        post_soft_acc += got
        _accum_pass_stats(
            post_soft_stats,
            getattr(_soft_relocation_moves, "last_stats", {}),
        )
        if not _adaptive_pass_gain(post_soft_before, r_score):
            break
    if _hard_valid(h_pos) and r_score < best_score - 1e-9:
        best_h, best_s, best_score = h_pos.copy(), s_pos.copy(), r_score
    _log(
        f"  [hier] post-swap soft relocation: {post_soft_acc} accepts, "
        f"proxy {pre_post_soft_score:.4f}->{r_score:.4f}"
    )
    _trace_pass(
        "post_swap_soft_relocation",
        pre_post_soft_score,
        r_score,
        post_soft_acc,
        quality=hierarchy_quality_metric(h_pos, clusters),
    )
    post_soft_record = _record_plateau(
        "post_swap_soft_relocation",
        pre_post_soft_score,
        r_score,
        post_soft_acc,
        time.monotonic() - post_soft_t0,
        candidates=post_soft_stats["candidates"],
        legal=post_soft_stats["legal"],
        scored=post_soft_stats["scored"],
        quality=hierarchy_quality_metric(h_pos, clusters),
    )
    post_escape_trigger = None
    if _is_plateau(post_soft_record):
        post_escape_trigger = "post_swap_soft_relocation"
    elif _is_plateau(post_hard_record):
        post_escape_trigger = "post_swap_hard_propose_all"
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
        log_gnn_event("hier_budget_schedule", **schedule_payload)
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
                    gpu_batch_rank=True,
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
            _log(
                f"  [hier] plateau escape post-soft relocation: {escape_acc} accepts, "
                f"proxy {pre_escape_score:.4f}->{r_score:.4f}"
            )
            escape_record = _record_plateau(
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
            _trace_pass(
                "plateau_escape_post_soft_relocation",
                pre_escape_score,
                r_score,
                escape_acc,
                quality=hierarchy_quality_metric(h_pos, clusters),
                plateau=bool(_is_plateau(escape_record)),
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
            or _is_plateau(post_soft_record)
        )
        useful_soft_trigger = bool(
            post_soft_record is None
            or post_soft_record.accepts > 0
            or post_soft_record.proxy_gain >= float(const.HIER_STRONG_SOFT_REPAIR_MIN_GAIN)
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
            max(0, int(const.HIER_PLATEAU_SOFT_REPAIR_BONUS_ROUNDS))
            if plateau_soft_bonus
            else 0
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
        }
        log_gnn_event("hier_budget_schedule", **schedule_payload)
        log_plateau_event("hier_budget_schedule", **schedule_payload)
        if run_strong_soft:
            strong_deadline = _deadline(scheduled_strong_budget, rdeadline)
            pre_strong_score = r_score
            strong_acc = 0
            strong_stats = _empty_pass_stats()
            strong_t0 = time.monotonic()
            strong_min_gain = float(const.HIER_STRONG_SOFT_REPAIR_MIN_GAIN)
            if adaptive_passes:
                strong_min_gain = max(strong_min_gain, adaptive_floor_proxy_gain)
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
                if adaptive_passes and not _adaptive_pass_gain(strong_round_before, r_score):
                    break
                if strong_deadline is not None and time.monotonic() >= strong_deadline:
                    break
            if _hard_valid(h_pos) and r_score < best_score - 1e-9:
                best_h, best_s, best_score = h_pos.copy(), s_pos.copy(), r_score
            _log(
                f"  [hier] strong soft repair: {strong_acc} accepts, "
                f"proxy {pre_strong_score:.4f}->{r_score:.4f}"
            )
            strong_record = _record_plateau(
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
            _trace_pass(
                "strong_soft_repair",
                pre_strong_score,
                r_score,
                strong_acc,
                quality=hierarchy_quality_metric(h_pos, clusters),
                plateau=bool(_is_plateau(strong_record)),
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
    elif _hard_valid(best_h):
        legal, s_pos = best_h.copy(), best_s.copy()
    else:
        legal = legal_candidate
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
        trace_pass_fn=_trace_pass,
        record_plateau_fn=_record_plateau,
        hard_valid_fn=_hard_valid,
        deadline_fn=_deadline,
        hierarchy_quality_metric_fn=hierarchy_quality_metric,
        hier_soft_barrier_gain=float(hier_soft_barrier_gain),
        hier_micro_shift_radius=int(hier_micro_shift_radius),
        hier_micro_shift_top=int(hier_micro_shift_top),
        hier_micro_shift_min_gain=float(hier_micro_shift_min_gain),
    )

    return run_post_coldspot_finalize(
        benchmark=benchmark,
        plc=plc,
        clusters=clusters,
        csofts=csofts,
        bridge_softs=bridge_softs,
        hierarchy=hierarchy,
        hierarchy_quality_metric_fn=hierarchy_quality_metric,
        selected_seed_name=selected_seed_name,
        seed_rows=seed_rows,
        pre_relief=pre_relief,
        legal=legal,
        s_pos=s_pos,
        cur_proxy=cur_proxy,
        best_h=best_h,
        best_s=best_s,
        best_score=best_score,
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
        region_deadline=rdeadline,
        log_fn=_log,
        trace_pass_fn=_trace_pass,
        record_plateau_fn=_record_plateau,
        hard_valid_fn=_hard_valid,
        deadline_fn=_deadline,
    )
