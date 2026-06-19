"""Main macro-placement pipeline."""

import random
import time
from typing import List, Optional

import numpy as np
import torch
from macro_place.benchmark import Benchmark

from utils import constants as const
from utils.config import _GPU_BACKEND, _GPU_DEVICE_NAME, _log
from placer.scoring.exact import _exact_proxy


class MacroPlacer:
    """Hierarchy-preserving macro placer."""

    def __init__(
        self,
        n_restarts: int = 50,
        noise_fracs: Optional[List[float]] = None,
        seed: int = 42,
        time_budget_s: float = const.TIME_BUDGET_S,
    ):
        # Kept for API compatibility with previous experiments and harnesses.
        self.n_restarts = n_restarts
        self.noise_fracs = noise_fracs or [
            0.02,
            0.04,
            0.06,
            0.08,
            0.01,
            0.03,
            0.05,
            0.07,
            0.09,
            0.06,
            0.06,
            0.04,
            0.10,
            0.12,
            0.08,
            0.025,
            0.035,
            0.045,
            0.055,
            0.065,
            0.075,
            0.15,
            0.20,
            0.10,
            0.05,
            0.06,
            0.07,
            0.03,
            0.04,
            0.02,
            0.005,
            0.010,
            0.015,
            0.030,
            0.050,
        ]
        self.seed = seed
        self.time_budget_s = time_budget_s

        self._benchmarks_done: int = 0
        self._total_place_time_s: float = 0.0

    @staticmethod
    def _clamp_in_bounds(pl: torch.Tensor, benchmark: Benchmark) -> torch.Tensor:
        """Keep movable macro centers inside the canvas."""
        sizes = benchmark.macro_sizes
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        hw = sizes[:, 0] / 2.0
        hh = sizes[:, 1] / 2.0
        mov = benchmark.get_movable_mask().to(torch.bool)
        out = pl.clone()
        cx = torch.minimum(torch.maximum(out[:, 0], hw), cw - hw)
        cy = torch.minimum(torch.maximum(out[:, 1], hh), ch - hh)
        out[:, 0] = torch.where(mov, cx, out[:, 0])
        out[:, 1] = torch.where(mov, cy, out[:, 1])
        return out

    def _hierarchy_floorplan(self, benchmark: Benchmark) -> "torch.Tensor | None":
        """Non-proxy hierarchy-preserving placement.

        Grouped DREAMPlace derives a hierarchical global placement, cluster-
        consecutive legalization keeps each subsystem together, and bounded
        cleanup recovers some congestion without making proxy score the primary
        objective.
        """
        from dreamplace_bridge.run_bridge import (  # noqa: E402
            run_dreamplace,
            is_available as _dp_available,
            dreamplace_design_name,
        )
        from placer.legalize.spiral import _will_legalize
        from placer.local_search.clusters import (
            cluster_max_fanout,
            cluster_min_edge,
            compute_region_bbox,
            compute_soft_region_bbox,
            derive_cluster_softs,
            derive_hard_clusters,
            derive_soft_cluster_roles,
            hier_region_density,
            hier_region_margin,
            hier_region_singleton,
        )
        from placer.local_search.cluster_decompress import (
            _cluster_decompression_relief,
            hierarchy_quality_metric,
        )
        from placer.local_search.fields import _congestion_field
        from placer.local_search.gnn_trace import log_gnn_event
        from placer.local_search.hierarchy_swaps import _region_bounded_swap_relief
        from placer.local_search.region_expand import expand_regions_by_congestion
        from placer.local_search.relocation import (
            _micro_shift_polish,
            _relocation_moves,
            _soft_relocation_moves,
        )
        from placer.scoring.incremental import IncrementalScorer
        from placer.scoring.wirelength import _build_wl_cache
        from placer.plc.loader import _load_plc, _resolve_benchmark_dir

        if not _dp_available():
            return None
        benchmark_dir = _resolve_benchmark_dir(benchmark.name, benchmark)
        if benchmark_dir is None:
            return None

        def _auto_cuda_flag(value) -> bool:
            if isinstance(value, str) and value.lower() == "auto":
                return _GPU_BACKEND == "cuda"
            return bool(value)

        hier_post_reloc_top_m = const.HIER_POST_RELOC_PROPOSE_TOP_M
        hier_reloc_propose_hot_k = max(1, int(const.HIER_RELOC_PROPOSE_HOT_K))
        hier_post_reloc_propose = _auto_cuda_flag(const.HIER_POST_RELOC_PROPOSE_ALL)
        hier_post_soft_reloc = const.HIER_POST_SOFT_RELOC
        hier_post_soft_reloc_top_k = max(1, int(const.HIER_POST_SOFT_RELOC_TOP_K))
        hier_post_soft_reloc_min_gain = float(const.HIER_POST_SOFT_RELOC_MIN_GAIN)
        hier_reloc_propose_min_gain = float(const.HIER_RELOC_PROPOSE_MIN_GAIN)
        hier_region_heat_frac = float(const.HIER_REGION_HEAT_FRAC)
        hier_region_heat_pct = float(const.HIER_REGION_HEAT_HOT_PCT)
        hier_region_heat_escape = float(const.HIER_REGION_HEAT_ESCAPE_MIN)
        hier_micro_shift = const.HIER_MICRO_SHIFT
        hier_micro_shift_radius = max(1, int(const.HIER_MICRO_SHIFT_RADIUS))
        hier_micro_shift_top = max(1, int(const.HIER_MICRO_SHIFT_TOP))
        hier_micro_shift_min_gain = float(const.HIER_MICRO_SHIFT_MIN_GAIN)
        hier_post_swap_micro = const.HIER_POST_SWAP_MICRO_SHIFT
        hier_post_coldspot_micro = const.HIER_POST_COLDSPOT_MICRO_SHIFT
        hier_decompress_aniso = const.HIER_DECOMPRESS_ANISO
        plc = _load_plc(benchmark.name, benchmark)
        n = benchmark.num_hard_macros
        n_soft = benchmark.num_soft_macros
        cw, ch = float(benchmark.canvas_width), float(benchmark.canvas_height)
        sizes = benchmark.macro_sizes.numpy().astype(np.float64)
        hw, hh = sizes[:n, 0] / 2.0, sizes[:n, 1] / 2.0
        soft_hw, soft_hh = sizes[n : n + n_soft, 0] / 2.0, sizes[n : n + n_soft, 1] / 2.0
        movable = benchmark.get_movable_mask().numpy()
        gw = max(1, int(const.HIER_GROUP_WEIGHT))

        labels, clusters = derive_hard_clusters(
            plc,
            n,
            n_soft=n_soft,
            max_fanout=cluster_max_fanout(),
            min_edge=cluster_min_edge(),
        )

        def _trace_pass(pass_name: str, before: float, after: float, accepts: int, **extra) -> None:
            log_gnn_event(
                "hier_pass_result",
                benchmark=benchmark.name,
                hierarchy_pass=pass_name,
                proxy_before=float(before),
                proxy_after=float(after),
                proxy_delta=float(after) - float(before),
                accepts=int(accepts),
                **extra,
            )

        bridge_ratio = float(const.HIER_BRIDGE_SOFT_RATIO)
        if const.HIER_BRIDGE_SOFTS:
            csofts, bridge_softs = derive_soft_cluster_roles(
                plc,
                n,
                n_soft,
                labels,
                max_fanout=cluster_max_fanout(),
                bridge_ratio=bridge_ratio,
            )
        else:
            csofts = derive_cluster_softs(plc, n, n_soft, labels)
            bridge_softs = {}
        hmi, smi = plc.hard_macro_indices, plc.soft_macro_indices
        groups = []
        for cid, mem in clusters.items():
            names = [plc.modules_w_pins[hmi[int(a)]].get_name() for a in mem]
            for p in csofts.get(cid, []):
                names.append(plc.modules_w_pins[smi[int(p) - n]].get_name())
            groups.append(names)

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
            max_fanout = cluster_max_fanout()
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

        if const.HIER_LEGALIZE_CONNECTIVITY_ORDER:
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
        else:
            order = []
            for mem in sorted(clusters.values(), key=lambda m: -m.size):
                order += sorted((int(x) for x in mem), key=lambda i: -(sizes[i, 0] * sizes[i, 1]))
            order += sorted(
                [i for i in range(n) if labels[i] < 0],
                key=lambda i: -(sizes[i, 0] * sizes[i, 1]),
            )

        def _prepare_dreamplace_candidate(
            *,
            group_weight: int,
            random_seed: int,
            scratch_root: str,
        ):
            raw_hard, raw_soft = run_dreamplace(
                str(benchmark_dir),
                plc=plc,
                scratch_root=scratch_root,
                iterations=300,
                num_threads=2,
                random_seed=random_seed,
                soft_macros_movable=True,
                cluster_groups=(groups or None),
                group_weight=group_weight,
                return_full=True,
            )
            legal_hard = _will_legalize(
                raw_hard.copy(),
                movable[:n],
                sizes[:n],
                hw,
                hh,
                cw,
                ch,
                n,
                deadline=time.monotonic() + 120,
                order=order,
            )
            legal_hard = _will_legalize(
                legal_hard,
                movable[:n],
                sizes[:n],
                hw,
                hh,
                cw,
                ch,
                n,
                deadline=time.monotonic() + 120,
                order=None,
            )

            cand_pos = np.vstack([legal_hard, raw_soft]).astype(np.float64)
            cand_soft = raw_soft.copy()
            cand_score = float(
                _exact_proxy(torch.tensor(cand_pos, dtype=torch.float32), benchmark, plc)
            )
            cand_scorer = IncrementalScorer(plc, benchmark, cand_pos.copy())
            soft_mov_local = movable[n : n + n_soft]
            for use_density in (False, True):
                cand_soft, _, cand_score = _soft_relocation_moves(
                    cand_soft,
                    soft_hw,
                    soft_hh,
                    cw,
                    ch,
                    n,
                    plc,
                    benchmark,
                    cand_scorer,
                    cand_score,
                    deadline=time.monotonic() + 30,
                    top_hot=1024,
                    n_targets=6,
                    soft_movable=soft_mov_local,
                    use_density=use_density,
                )
            return legal_hard, cand_soft, cand_score

        try:
            hard, soft, s_score = _prepare_dreamplace_candidate(
                group_weight=gw,
                random_seed=1000,
                scratch_root="/tmp/dreamplace_v1_hier",
            )
        except Exception as exc:
            _log(f"  [hier] DREAMPlace failed: {type(exc).__name__}: {exc}")
            return None

        legal = hard
        s_pos = soft.copy()
        pos = np.vstack([legal, s_pos]).astype(np.float64)
        scorer = IncrementalScorer(plc, benchmark, pos.copy())
        soft_mov = movable[n : n + n_soft]

        pre_relief = s_score
        region = None
        soft_region = None
        if const.HIER_REGION_RELIEF:
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
            region = compute_region_bbox(
                legal,
                sizes[:n],
                hw,
                hh,
                cw,
                ch,
                n,
                labels,
                clusters,
                target_density=hier_region_density(),
                margin=hier_region_margin(),
                singleton_window=hier_region_singleton(),
                cluster_heat=cluster_heat,
                heat_expand_frac=hier_region_heat_frac,
                heat_hot_percentile=hier_region_heat_pct,
                heat_escape_min=hier_region_heat_escape,
            )
            soft_region = compute_soft_region_bbox(
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
                clusters,
                csofts,
                bridge_softs=bridge_softs,
                target_density=hier_region_density(),
                margin=hier_region_margin(),
                singleton_window=hier_region_singleton(),
                cluster_heat=cluster_heat,
                heat_expand_frac=hier_region_heat_frac,
                heat_hot_percentile=hier_region_heat_pct,
                heat_escape_min=hier_region_heat_escape,
            )
            if const.HIER_CONG_EXPAND_REGIONS:
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
            rdeadline = time.monotonic() + float(const.HIER_REGION_BUDGET_S)
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

            for _ in range(rounds):
                if time.monotonic() >= rdeadline:
                    break
                if hier_micro_shift:
                    before_micro = r_score
                    micro_acc = 0
                    for use_density in (False, True):
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
                reloc_acc = 0
                for use_density in (False, True):
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
                soft_reloc_acc = 0
                for use_density in (False, True):
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
                    )
                    soft_reloc_acc += got
            if _hard_valid(h_pos) and r_score < best_score - 1e-9:
                best_h, best_s, best_score = h_pos.copy(), s_pos.copy(), r_score
            if const.HIER_DECOMPRESS:
                pre_decomp_h, pre_decomp_s, pre_decomp_score = h_pos.copy(), s_pos.copy(), r_score
                d_deadline = min(
                    rdeadline,
                    time.monotonic() + float(const.HIER_DECOMPRESS_BUDGET_S),
                )
                h_pos, s_pos, d_acc, r_score, hq = _cluster_decompression_relief(
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
                    r_score,
                    deadline=d_deadline,
                    rounds=max(1, int(const.HIER_DECOMPRESS_ROUNDS)),
                    hot_percentile=float(const.HIER_DECOMPRESS_HOT_PCT),
                    quality_budget=float(const.HIER_QUALITY_BUDGET),
                    min_proxy_gain=float(const.HIER_DECOMPRESS_MIN_GAIN),
                    anisotropic=hier_decompress_aniso,
                    anisotropic_band=max(1, int(const.HIER_DECOMPRESS_ANISO_BAND)),
                    anisotropic_secondary=float(const.HIER_DECOMPRESS_ANISO_SECONDARY),
                )
                invalid_decomp = not _hard_valid(h_pos)
                weak_decomp = (
                    d_acc
                    and const.HIER_ROLLBACK_WEAK_DECOMP
                    and r_score > pre_decomp_score - float(const.HIER_DECOMPRESS_MIN_GAIN)
                )
                if invalid_decomp or weak_decomp:
                    h_pos, s_pos, r_score = pre_decomp_h, pre_decomp_s, pre_decomp_score
                    d_acc = 0
                if d_acc:
                    full = np.vstack([h_pos, s_pos]).astype(np.float64)
                    rscorer = IncrementalScorer(plc, benchmark, full.copy())
                if _hard_valid(h_pos) and r_score < best_score - 1e-9:
                    best_h, best_s, best_score = h_pos.copy(), s_pos.copy(), r_score
                _log(
                    f"  [hier] cluster decompression: {d_acc} accepts, "
                    f"quality={hq:.4f}, proxy={r_score:.4f}"
                )
                _trace_pass(
                    "cluster_decompression",
                    pre_decomp_score,
                    r_score,
                    d_acc,
                    quality=float(hq),
                    rolled_back=bool(invalid_decomp or weak_decomp),
                )
            if const.HIER_REGION_SWAPS:
                swap_rounds = max(1, int(const.HIER_REGION_SWAP_ROUNDS))
                swap_deadline = min(
                    rdeadline,
                    time.monotonic() + float(const.HIER_REGION_SWAP_BUDGET_S),
                )
                hard_k = max(1, int(const.HIER_HARD_SWAP_K))
                soft_k = max(1, int(const.HIER_SOFT_SWAP_K))
                swap_min_gain = float(const.HIER_SWAP_MIN_GAIN)
                swap_min_field = float(const.HIER_SWAP_MIN_FIELD_RELIEF)
                enable_hh = const.HIER_SWAP_HH
                enable_hs = const.HIER_SWAP_HS
                enable_ss = const.HIER_SWAP_SS
                use_density_swaps = const.HIER_SWAP_DENSITY_FIELD
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
                pre_swap_score = r_score
                fields = (False, True) if use_density_swaps else (False,)
                for use_density in fields:
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
                        rounds=swap_rounds,
                        hard_k=hard_k,
                        soft_k=soft_k,
                        region_bias=bias,
                        escape_min=escape_min,
                        min_gain=swap_min_gain,
                        min_field_relief=swap_min_field,
                        enable_hh=enable_hh,
                        enable_hs=enable_hs,
                        enable_ss=enable_ss,
                        use_density=use_density,
                    )
                    swap_acc += got
                    for k, v in stats.items():
                        swap_stats[k] += v
                if not _hard_valid(h_pos):
                    h_pos, s_pos, r_score = best_h.copy(), best_s.copy(), best_score
                    full = np.vstack([h_pos, s_pos]).astype(np.float64)
                    rscorer = IncrementalScorer(plc, benchmark, full.copy())
                if _hard_valid(h_pos) and r_score < best_score - 1e-9:
                    best_h, best_s, best_score = h_pos.copy(), s_pos.copy(), r_score
                _log(
                    f"  [hier] region swaps: {swap_acc} accepts "
                    f"(hh {swap_stats['hh_accepts']}/{swap_stats['hh_scores']}, "
                    f"hs {swap_stats['hs_accepts']}/{swap_stats['hs_scores']}, "
                    f"ss {swap_stats['ss_accepts']}/{swap_stats['ss_scores']}, "
                    f"esc {swap_stats['hh_escape_accepts'] + swap_stats['hs_escape_accepts'] + swap_stats['ss_escape_accepts']}, "
                    f"gain {swap_stats['proxy_gain']:.4f}), proxy={r_score:.4f}"
                )
                _trace_pass(
                    "region_swaps",
                    pre_swap_score,
                    r_score,
                    swap_acc,
                    stats=swap_stats,
                    quality=hierarchy_quality_metric(h_pos, clusters),
                )
            if hier_post_swap_micro:
                post_micro_deadline = min(
                    rdeadline,
                    time.monotonic() + float(const.HIER_POST_SWAP_MICRO_SHIFT_BUDGET_S),
                )
                pre_post_micro_score = r_score
                post_micro_acc = 0
                for use_density in (False, True):
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
            if hier_post_reloc_propose:
                post_deadline = min(
                    rdeadline,
                    time.monotonic() + float(const.HIER_POST_RELOC_PROPOSE_BUDGET_S),
                )
                pre_post_score = r_score
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
                    propose_accept_min_gain=hier_reloc_propose_min_gain,
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
            if hier_post_soft_reloc:
                post_soft_deadline = min(
                    rdeadline,
                    time.monotonic() + float(const.HIER_POST_SOFT_RELOC_BUDGET_S),
                )
                pre_post_soft_score = r_score
                post_soft_acc = 0
                for use_density in (False, True):
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
                        accept_min_gain=hier_post_soft_reloc_min_gain,
                    )
                    post_soft_acc += got
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
            legal_candidate = _will_legalize(
                h_pos,
                movable[:n],
                sizes[:n],
                hw,
                hh,
                cw,
                ch,
                n,
                deadline=time.monotonic() + 30,
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

        if const.HIER_COLDSPOT_KICK:
            from placer.local_search.lsmc_explore import _coldspot_cluster_kick

            ck_budget = float(const.HIER_COLDSPOT_BUDGET)
            ck_total = float(const.HIER_COLDSPOT_TOTAL)
            ck_min_gain = float(const.HIER_COLDSPOT_MIN_GAIN)
            ck_quality_budget = float(const.HIER_COLDSPOT_QUALITY_BUDGET)
            ck_rounds = max(1, int(const.HIER_COLDSPOT_ROUNDS))
            ck_min_field_gap = float(const.HIER_COLDSPOT_MIN_FIELD_GAP)
            ck_deadline = time.monotonic() + float(const.HIER_COLDSPOT_BUDGET_S)
            nr, nc = int(benchmark.grid_rows), int(benchmark.grid_cols)
            soft_mov = movable[n : n + n_soft]

            def _full(h, sft):
                return torch.tensor(np.vstack([h, sft]).astype(np.float32), dtype=torch.float32)

            def _min_window_avg(field: np.ndarray, win_cells: int) -> float:
                w = int(max(1, min(win_cells, nr, nc)))
                rows, cols = nr - w + 1, nc - w + 1
                if rows <= 0 or cols <= 0:
                    return float(np.min(field))
                integ = np.zeros((nr + 1, nc + 1), dtype=np.float64)
                integ[1:, 1:] = np.cumsum(np.cumsum(field, axis=0), axis=1)
                win_sum = (
                    integ[w : w + rows, w : w + cols]
                    - integ[0:rows, w : w + cols]
                    - integ[w : w + rows, 0:cols]
                    + integ[0:rows, 0:cols]
                )
                return float(np.min(win_sum) / float(w * w))

            def _coldspot_field_gap(field: np.ndarray) -> float:
                cell_w, cell_h = cw / nc, ch / nr
                mcol = np.clip((cur_h[:, 0] / cell_w).astype(np.int64), 0, nc - 1)
                mrow = np.clip((cur_h[:, 1] / cell_h).astype(np.int64), 0, nr - 1)
                macro_cong = field[mrow, mcol]
                best_gap = -np.inf
                for members in clusters.values():
                    members = members[movable[:n][members]]
                    if members.size < 2 or members.size > 64:
                        continue
                    member_area = float(np.sum(sizes[members, 0] * sizes[members, 1]))
                    win_microns = float(np.sqrt(member_area / 0.65))
                    win_cells = max(1, int(np.ceil(win_microns / min(cell_w, cell_h))))
                    gap = float(np.mean(macro_cong[members])) - _min_window_avg(field, win_cells)
                    if gap > best_gap:
                        best_gap = gap
                return best_gap

            cur_h, cur_s = legal.copy(), s_pos.copy()
            base_proxy = float(_exact_proxy(_full(cur_h, cur_s), benchmark, plc))
            cur_proxy, cur_quality = base_proxy, hierarchy_quality_metric(cur_h, clusters)
            ck_scorer = IncrementalScorer(
                plc,
                benchmark,
                np.vstack([cur_h, cur_s]).astype(np.float64),
            )
            ck_rng = np.random.default_rng(0)
            ck_acc = 0
            for _ in range(ck_rounds):
                if time.monotonic() >= ck_deadline:
                    break
                field = _congestion_field(ck_scorer, nr, nc)
                if field is None:
                    break
                field_gap = _coldspot_field_gap(field)
                if field_gap < ck_min_field_gap:
                    log_gnn_event(
                        "hier_coldspot_candidate",
                        benchmark=benchmark.name,
                        operator="coldspot_tightening",
                        candidate_id=int(ck_acc),
                        field_gap=float(field_gap),
                        min_field_gap=float(ck_min_field_gap),
                        old_proxy=float(cur_proxy),
                        candidate_proxy=None,
                        proxy_delta=None,
                        hierarchy_quality_before=float(cur_quality),
                        hierarchy_quality_after=None,
                        hierarchy_quality_delta=None,
                        accepted=False,
                        rejection_reason="field_gap_below_threshold",
                    )
                    _log(
                        f"  [hier] coldspot tightening: skipped, "
                        f"field_gap={field_gap:.4f} < {ck_min_field_gap:.4f}"
                    )
                    break
                res = _coldspot_cluster_kick(
                    cur_h,
                    sizes[:n],
                    hw,
                    hh,
                    cw,
                    ch,
                    movable[:n],
                    n,
                    clusters,
                    csofts,
                    cur_s,
                    soft_hw,
                    soft_hh,
                    soft_mov,
                    field,
                    nr,
                    nc,
                    ck_rng,
                    deadline=ck_deadline,
                    pick="random",
                    return_trace=True,
                )
                if res is None:
                    log_gnn_event(
                        "hier_coldspot_candidate",
                        benchmark=benchmark.name,
                        operator="coldspot_tightening",
                        candidate_id=int(ck_acc),
                        field_gap=float(field_gap),
                        min_field_gap=float(ck_min_field_gap),
                        old_proxy=float(cur_proxy),
                        candidate_proxy=None,
                        proxy_delta=None,
                        hierarchy_quality_before=float(cur_quality),
                        hierarchy_quality_after=None,
                        hierarchy_quality_delta=None,
                        accepted=False,
                        rejection_reason="no_eligible_cluster",
                    )
                    continue
                kh, ks, ck_trace = res
                ks = ks if ks is not None else cur_s
                kproxy = float(_exact_proxy(_full(kh, ks), benchmark, plc))
                kquality = hierarchy_quality_metric(kh, clusters)
                accepted = (
                    kquality <= cur_quality + ck_quality_budget
                    and kproxy <= cur_proxy + ck_budget
                    and kproxy <= base_proxy + ck_total
                    and kproxy < cur_proxy - ck_min_gain
                )
                if kquality > cur_quality + ck_quality_budget:
                    reason = "hierarchy_quality_failed"
                elif kproxy > cur_proxy + ck_budget or kproxy > base_proxy + ck_total:
                    reason = "proxy_budget_failed"
                elif kproxy >= cur_proxy - ck_min_gain:
                    reason = "exact_proxy_failed"
                else:
                    reason = "accepted"
                log_gnn_event(
                    "hier_coldspot_candidate",
                    benchmark=benchmark.name,
                    operator="coldspot_tightening",
                    candidate_id=int(ck_acc),
                    field_gap=float(field_gap),
                    min_field_gap=float(ck_min_field_gap),
                    old_proxy=float(cur_proxy),
                    candidate_proxy=float(kproxy),
                    proxy_delta=float(kproxy) - float(cur_proxy),
                    hierarchy_quality_before=float(cur_quality),
                    hierarchy_quality_after=float(kquality),
                    hierarchy_quality_delta=float(kquality) - float(cur_quality),
                    accepted=bool(accepted),
                    rejection_reason=None if accepted else reason,
                    **ck_trace,
                )
                if accepted:
                    cur_h, cur_s, cur_proxy, cur_quality = kh, ks, kproxy, kquality
                    ck_scorer = IncrementalScorer(
                        plc,
                        benchmark,
                        np.vstack([cur_h, cur_s]).astype(np.float64),
                    )
                    ck_acc += 1
            legal, s_pos = cur_h, cur_s
            _log(
                f"  [hier] coldspot tightening: {ck_acc} accepts, "
                f"quality={cur_quality:.4f}, proxy {base_proxy:.4f}->{cur_proxy:.4f}"
            )
            _trace_pass(
                "coldspot_tightening",
                base_proxy,
                cur_proxy,
                ck_acc,
                quality=float(cur_quality),
            )

            if hier_post_coldspot_micro and region is not None and soft_region is not None:
                post_ck_micro_deadline = min(
                    ck_deadline,
                    time.monotonic() + float(const.HIER_POST_COLDSPOT_MICRO_SHIFT_BUDGET_S),
                )
                post_ck_micro_acc = 0
                pre_post_ck_micro_score = cur_proxy
                full = np.vstack([legal, s_pos]).astype(np.float64)
                ck_scorer = IncrementalScorer(plc, benchmark, full.copy())
                for use_density in (False, True):
                    legal, s_pos, got, cur_proxy = _micro_shift_polish(
                        legal,
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
                        ck_scorer,
                        cur_proxy,
                        hard_region=region,
                        soft_region=soft_region,
                        deadline=post_ck_micro_deadline,
                        radius_cells=hier_micro_shift_radius,
                        top_hot=hier_micro_shift_top,
                        min_gain=hier_micro_shift_min_gain,
                        use_density=use_density,
                    )
                    post_ck_micro_acc += got
                _log(
                    f"  [hier] post-coldspot micro-shift replay: {post_ck_micro_acc} accepts, "
                    f"proxy {pre_post_ck_micro_score:.4f}->{cur_proxy:.4f}"
                )
                _trace_pass(
                    "post_coldspot_micro_shift",
                    pre_post_ck_micro_score,
                    cur_proxy,
                    post_ck_micro_acc,
                    quality=hierarchy_quality_metric(legal, clusters),
                )

        out = torch.tensor(np.vstack([legal, s_pos]).astype(np.float32), dtype=torch.float32)
        proxy = float(_exact_proxy(out, benchmark, plc))
        log_gnn_event(
            "hier_final",
            benchmark=benchmark.name,
            proxy=float(proxy),
            pre_relief_proxy=float(pre_relief),
            hierarchy_quality=float(hierarchy_quality_metric(legal, clusters)),
            clusters=int(len(clusters)),
            group_weight=int(gw),
        )
        _log(
            f"  [hier] {len(clusters)} clusters, weight={gw}: proxy={proxy:.4f} "
            f"(pre-relief {pre_relief:.4f}; hierarchy-preserving NON-proxy mode)"
        )
        return out

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        return self._clamp_in_bounds(self._place_impl(benchmark), benchmark)

    def _place_impl(self, benchmark: Benchmark) -> torch.Tensor:
        np.random.seed(self.seed)
        random.seed(self.seed)

        _log(f"[GPU] backend={_GPU_BACKEND} device={_GPU_DEVICE_NAME} | benchmark={benchmark.name}")

        t0 = time.monotonic()
        hier = self._hierarchy_floorplan(benchmark)
        if hier is None:
            raise RuntimeError(
                "hierarchy floorplan path unavailable; proxy fallback has been removed"
            )
        self._total_place_time_s += time.monotonic() - t0
        self._benchmarks_done += 1
        return hier
