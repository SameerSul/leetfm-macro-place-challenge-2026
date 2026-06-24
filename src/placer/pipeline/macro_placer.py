"""Main macro-placement pipeline."""

import os
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
        from placer.local_search.cluster_decompress import (
            _cluster_decompression_relief,
            hierarchy_quality_breakdown,
            hierarchy_quality_metric,
        )
        from placer.local_search.fields import _congestion_field
        from placer.local_search.gnn_trace import (
            flush_plateau_events,
            log_gnn_event,
            log_plateau_event,
        )
        from placer.local_search.hierarchy_swaps import _region_bounded_swap_relief
        from placer.local_search.hierarchy_model import HierarchyModel
        from placer.local_search.region_expand import expand_regions_by_congestion
        from placer.local_search.survivor_search import _parallel_survivor_search
        from placer.local_search.relocation import (
            _micro_shift_polish,
            _relocation_moves,
            _soft_relocation_moves,
        )
        from placer.pipeline.hierarchy_context import (
            PassContext,
            PassResult,
            PlateauTelemetry,
            PlacementState,
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

        def _clip_seed(hard_xy, soft_xy):
            hard_xy = hard_xy.copy()
            soft_xy = soft_xy.copy()
            hard_mov = movable[:n]
            hard_xy[hard_mov, 0] = np.clip(hard_xy[hard_mov, 0], hw[hard_mov], cw - hw[hard_mov])
            hard_xy[hard_mov, 1] = np.clip(hard_xy[hard_mov, 1], hh[hard_mov], ch - hh[hard_mov])
            if n_soft:
                soft_mov_local = movable[n : n + n_soft]
                soft_xy[soft_mov_local, 0] = np.clip(
                    soft_xy[soft_mov_local, 0],
                    soft_hw[soft_mov_local],
                    cw - soft_hw[soft_mov_local],
                )
                soft_xy[soft_mov_local, 1] = np.clip(
                    soft_xy[soft_mov_local, 1],
                    soft_hh[soft_mov_local],
                    ch - soft_hh[soft_mov_local],
                )
            return hard_xy, soft_xy

        def _legalize_seed(name: str, hard_xy, soft_xy, *, budget_s: float = 60.0):
            hard_xy, soft_xy = _clip_seed(hard_xy, soft_xy)
            seed_deadline = time.monotonic() + float(budget_s)
            legal_hard = _will_legalize(
                hard_xy,
                movable[:n],
                sizes[:n],
                hw,
                hh,
                cw,
                ch,
                n,
                deadline=seed_deadline,
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
                deadline=seed_deadline,
                order=None,
            )
            full = np.vstack([legal_hard, soft_xy]).astype(np.float64)
            score = float(_exact_proxy(torch.tensor(full, dtype=torch.float32), benchmark, plc))
            return {
                "name": name,
                "hard": legal_hard,
                "soft": soft_xy,
                "score": score,
            }

        def _expanded_seed(base_hard, base_soft):
            hard = base_hard.copy()
            soft = base_soft.copy()
            frac = float(const.HIER_SEED_EXPANSION_FRAC)
            hard_mov = movable[:n]
            if np.any(hard_mov):
                center = np.mean(hard[hard_mov], axis=0)
                hard[hard_mov] = center + (1.0 + frac) * (hard[hard_mov] - center)
            if n_soft and np.any(movable[n : n + n_soft]):
                soft_mov_local = movable[n : n + n_soft]
                center = np.mean(soft[soft_mov_local], axis=0)
                soft[soft_mov_local] = center + (1.0 + frac) * (soft[soft_mov_local] - center)
            return hard, soft

        def _synthetic_clearance_seed(base_hard, base_soft):
            hard = base_hard.copy()
            hard_mov = movable[:n]
            area = sizes[:n, 0] * sizes[:n, 1]
            area_limit = float(np.percentile(area, float(const.HIER_SEED_CLEARANCE_AREA_PCT)))
            eligible = hard_mov & (area <= area_limit)
            if not np.any(eligible):
                return hard, base_soft.copy()
            temp_hw = hw.copy()
            temp_hh = hh.copy()
            temp_hw[eligible] *= 1.0 + float(const.HIER_SEED_CLEARANCE_FRAC)
            temp_hh[eligible] *= 1.0 + float(const.HIER_SEED_CLEARANCE_FRAC)
            iters = max(1, int(const.HIER_SEED_CLEARANCE_ITERS))
            for _ in range(iters):
                delta = np.zeros_like(hard)
                for i in range(n):
                    for j in range(i + 1, n):
                        move_i = bool(eligible[i])
                        move_j = bool(eligible[j])
                        if not (move_i or move_j):
                            continue
                        dx = float(hard[i, 0] - hard[j, 0])
                        dy = float(hard[i, 1] - hard[j, 1])
                        ox = float(temp_hw[i] + temp_hw[j] - abs(dx))
                        oy = float(temp_hh[i] + temp_hh[j] - abs(dy))
                        if ox <= 0.0 or oy <= 0.0:
                            continue
                        if ox <= oy:
                            sx = 1.0 if dx >= 0.0 else -1.0
                            push = np.array([0.5 * ox * sx, 0.0], dtype=np.float64)
                        else:
                            sy = 1.0 if dy >= 0.0 else -1.0
                            push = np.array([0.0, 0.5 * oy * sy], dtype=np.float64)
                        if move_i and move_j:
                            delta[i] += push
                            delta[j] -= push
                        elif move_i:
                            delta[i] += 2.0 * push
                        elif move_j:
                            delta[j] -= 2.0 * push
                hard[eligible] += 0.5 * delta[eligible]
                hard, _soft = _clip_seed(hard, base_soft)
            return hard, base_soft.copy()

        def _has_explicit_path_tags() -> bool:
            try:
                hard_b = list(plc.hard_macro_indices[:n])
                tagged = sum(
                    1 for idx in hard_b if "/" in str(plc.modules_w_pins[int(idx)].get_name())
                )
            except Exception:
                return False
            min_group = max(2, int(const.HIER_TAG_PREFIX_MIN_GROUP))
            return tagged >= max(min_group, int(0.5 * n))

        def _route_channel_seed(base_hard, base_soft):
            hard = base_hard.copy()
            soft = base_soft.copy()
            hard_mov = movable[:n].astype(bool)
            soft_mov = movable[n : n + n_soft].astype(bool)
            min_cluster = max(2, int(const.HIER_SEED_ROUTE_CHANNEL_MIN_CLUSTER))
            lane_frac = max(0.0, float(const.HIER_SEED_ROUTE_CHANNEL_LANE_FRAC))
            push_frac = max(0.0, float(const.HIER_SEED_ROUTE_CHANNEL_PUSH_FRAC))
            max_shift_frac = max(0.0, float(const.HIER_SEED_ROUTE_CHANNEL_MAX_SHIFT_FRAC))
            if lane_frac <= 0.0 or push_frac <= 0.0 or max_shift_frac <= 0.0:
                return hard, soft

            def _channel_delta(xy, local_hw, local_hh, center, span_x, span_y, index_bias):
                dx = xy[:, 0] - center[0]
                dy = xy[:, 1] - center[1]
                sx = np.where(dx >= 0.0, 1.0, -1.0)
                sy = np.where(dy >= 0.0, 1.0, -1.0)
                sx = np.where(np.abs(dx) > 1.0e-9, sx, np.where(index_bias % 2 == 0, 1.0, -1.0))
                sy = np.where(np.abs(dy) > 1.0e-9, sy, np.where(index_bias % 3 == 0, 1.0, -1.0))
                lane_x = max(float(np.median(local_hw)) * 0.75, float(span_x) * lane_frac)
                lane_y = max(float(np.median(local_hh)) * 0.75, float(span_y) * lane_frac)
                max_x = float(span_x) * max_shift_frac
                max_y = float(span_y) * max_shift_frac
                push_x = np.maximum(0.0, lane_x - np.abs(dx)) * push_frac
                push_y = np.maximum(0.0, lane_y - np.abs(dy)) * push_frac
                out = np.zeros_like(xy)
                out[:, 0] = sx * np.minimum(push_x, max_x)
                out[:, 1] = sy * np.minimum(push_y, max_y)
                return out

            for cid, mem in clusters.items():
                mem = np.asarray(mem, dtype=np.int64)
                if mem.size < min_cluster:
                    continue
                active = mem[hard_mov[mem]]
                if active.size == 0:
                    continue
                left = float(np.min(hard[mem, 0] - hw[mem]))
                right = float(np.max(hard[mem, 0] + hw[mem]))
                bottom = float(np.min(hard[mem, 1] - hh[mem]))
                top = float(np.max(hard[mem, 1] + hh[mem]))
                span_x = max(1.0, right - left)
                span_y = max(1.0, top - bottom)
                center = np.array([(left + right) * 0.5, (bottom + top) * 0.5], dtype=np.float64)
                hard[active] += _channel_delta(
                    hard[active],
                    hw[active],
                    hh[active],
                    center,
                    span_x,
                    span_y,
                    active,
                )
                if n_soft:
                    owned = np.asarray(csofts.get(int(cid), []), dtype=np.int64) - n
                    owned = owned[(owned >= 0) & (owned < n_soft)]
                    if owned.size:
                        owned = owned[soft_mov[owned]]
                    if owned.size:
                        soft[owned] += _channel_delta(
                            soft[owned],
                            soft_hw[owned],
                            soft_hh[owned],
                            center,
                            span_x,
                            span_y,
                            owned + mem.size,
                        )
            return _clip_seed(hard, soft)

        def _select_seed_portfolio(dp_hard, dp_soft, dp_score):
            initial = benchmark.macro_positions.detach().cpu().numpy().astype(np.float64)
            init_hard = initial[:n].copy()
            init_soft = initial[n : n + n_soft].copy()
            rows = [
                {"name": "dreamplace", "hard": dp_hard, "soft": dp_soft, "score": float(dp_score)}
            ]
            raw_candidates = []
            raw_candidates.append(("initial", init_hard, init_soft))
            for alpha in tuple(float(a) for a in const.HIER_SEED_BLEND_ALPHAS):
                hard = (1.0 - alpha) * dp_hard + alpha * init_hard
                soft = (1.0 - alpha) * dp_soft + alpha * init_soft
                raw_candidates.append((f"blend_{alpha:.2f}", hard, soft))
            raw_candidates.append(("expand", *_expanded_seed(dp_hard, dp_soft)))
            raw_candidates.append(
                ("synthetic_clearance", *_synthetic_clearance_seed(dp_hard, dp_soft))
            )
            if _has_explicit_path_tags():
                raw_candidates.append(("route_channel", *_route_channel_seed(dp_hard, dp_soft)))
            for name, cand_h, cand_s in raw_candidates:
                try:
                    rows.append(_legalize_seed(name, cand_h, cand_s, budget_s=45.0))
                except Exception as exc:
                    _log(f"  [hier] seed {name} failed prescore: {type(exc).__name__}: {exc}")
            rows.sort(key=lambda r: (float(r["score"]), str(r["name"])))
            summary = ", ".join(f"{r['name']}={float(r['score']):.4f}" for r in rows)
            _log(f"  [hier] seed portfolio prescore: {summary}; selected={rows[0]['name']}")
            return rows[0]["hard"], rows[0]["soft"], float(rows[0]["score"]), rows

        try:
            hard, soft, s_score = _prepare_dreamplace_candidate(
                group_weight=gw,
                random_seed=1000,
                scratch_root="/tmp/dreamplace_v1_hier",
            )
        except Exception as exc:
            _log(f"  [hier] DREAMPlace failed: {type(exc).__name__}: {exc}")
            return None

        hard, soft, s_score, seed_rows = _select_seed_portfolio(hard, soft, s_score)
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
            before_micro = r_score
            micro_acc = 0
            micro_t0 = time.monotonic()
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
            if _hard_valid(h_pos) and r_score < best_score - 1e-9:
                best_h, best_s, best_score = h_pos.copy(), s_pos.copy(), r_score
        pre_decomp_h, pre_decomp_s, pre_decomp_score = h_pos.copy(), s_pos.copy(), r_score
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
            d_deadline = _deadline(float(const.HIER_DECOMPRESS_BUDGET_S), rdeadline)
            decomp_t0 = time.monotonic()
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
                anisotropic=True,
                anisotropic_band=max(1, int(const.HIER_DECOMPRESS_ANISO_BAND)),
                anisotropic_secondary=float(const.HIER_DECOMPRESS_ANISO_SECONDARY),
            )
            invalid_decomp = not _hard_valid(h_pos)
            weak_decomp = d_acc and r_score > pre_decomp_score - float(
                const.HIER_DECOMPRESS_MIN_GAIN
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
                _record_plateau(
                    "cluster_decompression",
                    pre_decomp_score,
                    r_score,
                    d_acc,
                    time.monotonic() - decomp_t0,
                    quality=float(hq),
                    rolled_back=bool(invalid_decomp or weak_decomp),
                )
        if (
            n_soft
            and bool(np.any(soft_mov))
            and _has_spare(rdeadline, float(const.HIER_INTERLEAVED_SOFT_REPAIR_MIN_SPARE_S))
        ):
            inter_soft_deadline = _deadline(
                float(const.HIER_INTERLEAVED_SOFT_REPAIR_BUDGET_S), rdeadline
            )
            pre_inter_soft_score = r_score
            inter_soft_acc = 0
            inter_soft_stats = _empty_pass_stats()
            inter_soft_t0 = time.monotonic()
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
                    deadline=inter_soft_deadline,
                    top_hot=max(1, int(const.HIER_INTERLEAVED_SOFT_REPAIR_TOP_K)),
                    n_targets=max(1, int(const.HIER_INTERLEAVED_SOFT_REPAIR_TARGETS)),
                    soft_movable=soft_mov,
                    use_density=use_density,
                    region_bbox=soft_region,
                    region_bias=bias,
                    region_escape_min=escape_min,
                    accept_min_gain=max(
                        float(const.HIER_INTERLEAVED_SOFT_REPAIR_MIN_GAIN),
                        hier_soft_barrier_gain,
                    ),
                    wl_prefilter=float(const.HIER_STRONG_SOFT_REPAIR_WL_PREFILTER),
                )
                inter_soft_acc += got
                _accum_pass_stats(
                    inter_soft_stats,
                    getattr(_soft_relocation_moves, "last_stats", {}),
                )
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
            for micro_density in (False, True):
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
                            float(const.HIER_PLATEAU_ESCAPE_MIN_GAIN),
                            hier_soft_barrier_gain,
                        ),
                        gpu_batch_rank=True,
                    )
                    escape_acc += got
                    _accum_pass_stats(
                        escape_stats,
                        getattr(_soft_relocation_moves, "last_stats", {}),
                    )
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
        _record_plateau(
            "post_swap_micro_shift",
            pre_post_micro_score,
            r_score,
            post_micro_acc,
            time.monotonic() - post_micro_t0,
            quality=hierarchy_quality_metric(h_pos, clusters),
        )
        if hier_post_reloc_propose:
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
        post_soft_stats = _empty_pass_stats()
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
                accept_min_gain=max(
                    hier_post_soft_reloc_min_gain,
                    hier_soft_barrier_gain,
                ),
            )
            post_soft_acc += got
            _accum_pass_stats(
                post_soft_stats,
                getattr(_soft_relocation_moves, "last_stats", {}),
            )
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
                            float(const.HIER_PLATEAU_ESCAPE_MIN_GAIN),
                            hier_soft_barrier_gain,
                        ),
                        gpu_batch_rank=True,
                    )
                    escape_acc += got
                    _accum_pass_stats(
                        escape_stats,
                        getattr(_soft_relocation_moves, "last_stats", {}),
                    )
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
                for _strong_round in range(scheduled_strong_rounds):
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
                            deadline=strong_deadline,
                            top_hot=max(1, int(const.HIER_STRONG_SOFT_REPAIR_TOP_K)),
                            n_targets=max(1, int(const.HIER_STRONG_SOFT_REPAIR_TARGETS)),
                            soft_movable=soft_mov,
                            use_density=use_density,
                            region_bbox=soft_region,
                            region_bias=bias,
                            region_escape_min=escape_min,
                            accept_min_gain=max(
                                float(const.HIER_STRONG_SOFT_REPAIR_MIN_GAIN),
                                hier_soft_barrier_gain,
                            ),
                            wl_prefilter=float(const.HIER_STRONG_SOFT_REPAIR_WL_PREFILTER),
                        )
                        strong_acc += got
                        _accum_pass_stats(
                            strong_stats,
                            getattr(_soft_relocation_moves, "last_stats", {}),
                        )
                        if strong_deadline is not None and time.monotonic() >= strong_deadline:
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

        from placer.local_search.gnn_ranker import (
            gnn_coldspot_kicks,
            gnn_coldspot_oracle_enabled,
            gnn_coldspot_select_enabled,
            gnn_coldspot_skip_micro,
            rank_coldspot_kick_candidates,
        )
        from placer.local_search.lsmc_explore import _coldspot_cluster_kick_candidates

        ck_budget = float(const.HIER_COLDSPOT_BUDGET)
        ck_total = float(const.HIER_COLDSPOT_TOTAL)
        ck_min_gain = float(const.HIER_COLDSPOT_MIN_GAIN)
        ck_quality_budget = float(const.HIER_COLDSPOT_QUALITY_BUDGET)
        ck_rounds = max(1, int(const.HIER_COLDSPOT_ROUNDS))
        ck_min_field_gap = float(const.HIER_COLDSPOT_MIN_FIELD_GAP)
        ck_min_field_gap = max(
            ck_min_field_gap,
            float(const.HIER_COLDSPOT_STRONG_MIN_FIELD_GAP),
        )
        ck_deadline = _deadline(float(const.HIER_COLDSPOT_BUDGET_S))
        ck_gnn_select = gnn_coldspot_select_enabled()
        ck_oracle = gnn_coldspot_oracle_enabled()
        ck_gnn_kicks = gnn_coldspot_kicks() if (ck_gnn_select or ck_oracle) else 1
        if not (ck_gnn_select or ck_oracle):
            ck_gnn_kicks = max(
                ck_gnn_kicks,
                max(1, int(const.HIER_COLDSPOT_GRAPH_SELECT_CANDIDATES)),
            )
        ck_gnn_top_k = max(1, int(os.environ.get("HIER_GNN_COLDSPOT_TOP_K", "1") or "1"))
        ck_graph_top_k = max(1, int(const.HIER_COLDSPOT_GRAPH_SELECT_TOP_K))
        ck_skip_micro = ck_gnn_select and gnn_coldspot_skip_micro()
        nr, nc = int(benchmark.grid_rows), int(benchmark.grid_cols)
        soft_mov = movable[n : n + n_soft]
        coldspot_candidate_softs = csofts
        if bridge_softs:
            bridge_by_cluster: dict[int, list[int]] = {}
            for soft_k, soft_cids in bridge_softs.items():
                k = int(soft_k)
                if k < 0 or k >= n_soft or not bool(soft_mov[k]):
                    continue
                for cid_for_soft in np.asarray(soft_cids, dtype=np.int64):
                    bridge_by_cluster.setdefault(int(cid_for_soft), []).append(n + k)
            if bridge_by_cluster:
                merged_softs: dict[int, np.ndarray] = {}
                for cid in clusters.keys():
                    parts = []
                    owned = np.asarray(csofts.get(int(cid), []), dtype=np.int64)
                    if owned.size:
                        parts.append(owned)
                    bridge = np.asarray(
                        bridge_by_cluster.get(int(cid), []),
                        dtype=np.int64,
                    )
                    if bridge.size:
                        parts.append(bridge)
                    if parts:
                        merged_softs[int(cid)] = np.unique(np.concatenate(parts)).astype(np.int64)
                if merged_softs:
                    coldspot_candidate_softs = merged_softs

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

        def _remember_cold_cells(field: np.ndarray) -> np.ndarray:
            cold_pct = float(const.HIER_COLDSPOT_MEMORY_COLD_PCT)
            thresh = float(np.percentile(field, np.clip(cold_pct, 0.0, 100.0)))
            return np.asarray(field <= thresh, dtype=bool)

        def _occupied_cells(hard_xy: np.ndarray, soft_xy: np.ndarray) -> np.ndarray:
            occupied = np.zeros((nr, nc), dtype=bool)
            cell_w, cell_h = cw / nc, ch / nr

            def _mark(pos, half_w, half_h):
                for x, y, mx, my in zip(pos[:, 0], pos[:, 1], half_w, half_h):
                    c0 = int(np.floor((float(x) - float(mx)) / cell_w))
                    c1 = int(np.floor((float(x) + float(mx)) / cell_w))
                    r0 = int(np.floor((float(y) - float(my)) / cell_h))
                    r1 = int(np.floor((float(y) + float(my)) / cell_h))
                    c0 = max(0, min(nc - 1, c0))
                    c1 = max(0, min(nc - 1, c1))
                    r0 = max(0, min(nr - 1, r0))
                    r1 = max(0, min(nr - 1, r1))
                    occupied[r0 : r1 + 1, c0 : c1 + 1] = True

            if hard_xy.size:
                _mark(hard_xy, hw, hh)
            if soft_xy.size:
                _mark(soft_xy, soft_hw, soft_hh)
            return occupied

        def _bbox_cell_mask(xlo: float, ylo: float, xhi: float, yhi: float) -> np.ndarray:
            mask = np.zeros((nr, nc), dtype=bool)
            cell_w, cell_h = cw / nc, ch / nr
            c0 = int(np.floor(xlo / cell_w))
            c1 = int(np.floor(xhi / cell_w))
            r0 = int(np.floor(ylo / cell_h))
            r1 = int(np.floor(yhi / cell_h))
            c0 = max(0, min(nc - 1, c0))
            c1 = max(0, min(nc - 1, c1))
            r0 = max(0, min(nr - 1, r0))
            r1 = max(0, min(nr - 1, r1))
            mask[r0 : r1 + 1, c0 : c1 + 1] = True
            return mask

        def _dilate_cell_mask(mask: np.ndarray, radius: int) -> np.ndarray:
            radius = max(0, int(radius))
            out = np.asarray(mask, dtype=bool).copy()
            if radius == 0 or not out.any():
                return out
            rows, cols = np.where(out)
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    rr = rows + dr
                    cc = cols + dc
                    valid = (rr >= 0) & (rr < nr) & (cc >= 0) & (cc < nc)
                    out[rr[valid], cc[valid]] = True
            return out

        def _expand_bbox_to_adjacent_cold(
            xlo: float,
            ylo: float,
            xhi: float,
            yhi: float,
            hard_xy: np.ndarray,
            soft_xy: np.ndarray,
        ) -> "tuple[float, float, float, float, int, np.ndarray, np.ndarray]":
            seed_mask = _bbox_cell_mask(xlo, ylo, xhi, yhi)
            empty = np.zeros((nr, nc), dtype=bool)
            if not cold_memory.any():
                return xlo, ylo, xhi, yhi, 0, seed_mask, empty

            occupied = _occupied_cells(hard_xy, soft_xy)
            open_cold = cold_memory & ~occupied
            if not open_cold.any():
                return xlo, ylo, xhi, yhi, 0, seed_mask, empty

            cell_w, cell_h = cw / nc, ch / nr
            c0 = int(np.floor(xlo / cell_w))
            c1 = int(np.floor(xhi / cell_w))
            r0 = int(np.floor(ylo / cell_h))
            r1 = int(np.floor(yhi / cell_h))
            c0 = max(0, min(nc - 1, c0))
            c1 = max(0, min(nc - 1, c1))
            r0 = max(0, min(nr - 1, r0))
            r1 = max(0, min(nr - 1, r1))

            max_dist = max(1, int(const.HIER_COLDSPOT_ADAPTIVE_MAX_CELLS))
            seen = np.zeros((nr, nc), dtype=bool)
            queue: list[tuple[int, int, int]] = []
            for rr in range(max(0, r0 - 1), min(nr, r1 + 2)):
                for cc in range(max(0, c0 - 1), min(nc, c1 + 2)):
                    adjacent = rr < r0 or rr > r1 or cc < c0 or cc > c1
                    if adjacent and open_cold[rr, cc]:
                        seen[rr, cc] = True
                        queue.append((rr, cc, 0))

            reached: list[tuple[int, int]] = []
            head = 0
            while head < len(queue):
                rr, cc, dist = queue[head]
                head += 1
                reached.append((rr, cc))
                if dist >= max_dist:
                    continue
                for nr2, nc2 in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if nr2 < 0 or nr2 >= nr or nc2 < 0 or nc2 >= nc or seen[nr2, nc2]:
                        continue
                    if not open_cold[nr2, nc2]:
                        continue
                    seen[nr2, nc2] = True
                    queue.append((nr2, nc2, dist + 1))

            if not reached:
                return xlo, ylo, xhi, yhi, 0, seed_mask, empty
            rows = np.array([p[0] for p in reached], dtype=np.int64)
            cols = np.array([p[1] for p in reached], dtype=np.int64)
            reached_mask = np.zeros((nr, nc), dtype=bool)
            reached_mask[rows, cols] = True
            xlo = min(xlo, float(cols.min()) * cell_w)
            ylo = min(ylo, float(rows.min()) * cell_h)
            xhi = max(xhi, float(cols.max() + 1) * cell_w)
            yhi = max(yhi, float(rows.max() + 1) * cell_h)
            graph_mask = seed_mask | reached_mask
            return (
                max(0.0, xlo),
                max(0.0, ylo),
                min(cw, xhi),
                min(ch, yhi),
                len(reached),
                graph_mask,
                open_cold,
            )

        def _coldspot_local_regions(
            hard_xy: np.ndarray,
            soft_xy: np.ndarray,
            cid: int,
        ) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict] | None":
            members = np.asarray(clusters.get(int(cid), []), dtype=np.int64)
            members = members[(members >= 0) & (members < n)]
            members = members[movable[:n][members]]
            if members.size == 0:
                return None

            owned_soft = np.asarray(csofts.get(int(cid), []), dtype=np.int64) - n
            owned_soft = owned_soft[(owned_soft >= 0) & (owned_soft < n_soft)]
            bridge_local = [
                int(k)
                for k, cids_for_soft in bridge_softs.items()
                if int(cid) in {int(v) for v in np.asarray(cids_for_soft, dtype=np.int64)}
            ]
            soft_seed = np.unique(
                np.concatenate([owned_soft, np.asarray(bridge_local, dtype=np.int64)])
                if bridge_local
                else owned_soft
            )
            if soft_seed.size:
                soft_seed = soft_seed[soft_mov[soft_seed]]

            hard_xlo = float(np.min(hard_xy[members, 0] - hw[members]))
            hard_ylo = float(np.min(hard_xy[members, 1] - hh[members]))
            hard_xhi = float(np.max(hard_xy[members, 0] + hw[members]))
            hard_yhi = float(np.max(hard_xy[members, 1] + hh[members]))
            xlo, ylo, xhi, yhi = hard_xlo, hard_ylo, hard_xhi, hard_yhi
            if soft_seed.size:
                xlo = min(xlo, float(np.min(soft_xy[soft_seed, 0] - soft_hw[soft_seed])))
                ylo = min(ylo, float(np.min(soft_xy[soft_seed, 1] - soft_hh[soft_seed])))
                xhi = max(xhi, float(np.max(soft_xy[soft_seed, 0] + soft_hw[soft_seed])))
                yhi = max(yhi, float(np.max(soft_xy[soft_seed, 1] + soft_hh[soft_seed])))

            (
                xlo,
                ylo,
                xhi,
                yhi,
                adaptive_cold_cells,
                graph_mask,
                open_cold_mask,
            ) = _expand_bbox_to_adjacent_cold(
                xlo,
                ylo,
                xhi,
                yhi,
                hard_xy,
                soft_xy,
            )

            cell_w, cell_h = cw / nc, ch / nr
            hard_core_span = max(hard_xhi - hard_xlo, hard_yhi - hard_ylo)
            min_pad = max(cell_w, cell_h) * max(0.0, float(const.HIER_COLDSPOT_LOCAL_MIN_PAD_CELLS))
            max_pad = max(cw, ch) * max(0.0, float(const.HIER_COLDSPOT_LOCAL_MAX_PAD_FRAC))
            pad = max(
                min_pad,
                hard_core_span * max(0.0, float(const.HIER_COLDSPOT_LOCAL_HARD_PAD_FRAC)),
            )
            if max_pad > 0.0:
                pad = min(pad, max_pad)
            pad_cells = int(np.ceil(pad / max(min(cell_w, cell_h), 1e-9)))
            region_mask = _dilate_cell_mask(graph_mask, pad_cells)
            xlo = max(0.0, xlo - pad)
            ylo = max(0.0, ylo - pad)
            xhi = min(cw, xhi + pad)
            yhi = min(ch, yhi + pad)
            target_mask = region_mask & ~_occupied_cells(hard_xy, soft_xy)
            if open_cold_mask.any():
                target_mask = target_mask & (open_cold_mask | graph_mask)
            target_pool = np.flatnonzero(target_mask.ravel()).astype(np.int64)

            if n_soft:
                inside_soft = np.where(
                    soft_mov
                    & (soft_xy[:, 0] >= xlo)
                    & (soft_xy[:, 0] <= xhi)
                    & (soft_xy[:, 1] >= ylo)
                    & (soft_xy[:, 1] <= yhi)
                )[0]
                local_soft = np.unique(np.concatenate([soft_seed, inside_soft]))
            else:
                local_soft = np.zeros(0, dtype=np.int64)

            hard_region = np.column_stack([hw, hh, cw - hw, ch - hh]).astype(np.float64)
            for i in members:
                hard_region[i] = (
                    max(hw[i], xlo + hw[i]),
                    max(hh[i], ylo + hh[i]),
                    min(cw - hw[i], xhi - hw[i]),
                    min(ch - hh[i], yhi - hh[i]),
                )
                if hard_region[i, 0] > hard_region[i, 2]:
                    hard_region[i, 0] = hard_region[i, 2] = float(hard_xy[i, 0])
                if hard_region[i, 1] > hard_region[i, 3]:
                    hard_region[i, 1] = hard_region[i, 3] = float(hard_xy[i, 1])

            soft_region = np.column_stack([soft_hw, soft_hh, cw - soft_hw, ch - soft_hh]).astype(
                np.float64
            )
            for k in local_soft:
                soft_region[k] = (
                    max(soft_hw[k], xlo + soft_hw[k]),
                    max(soft_hh[k], ylo + soft_hh[k]),
                    min(cw - soft_hw[k], xhi - soft_hw[k]),
                    min(ch - soft_hh[k], yhi - soft_hh[k]),
                )
                if soft_region[k, 0] > soft_region[k, 2]:
                    soft_region[k, 0] = soft_region[k, 2] = float(soft_xy[k, 0])
                if soft_region[k, 1] > soft_region[k, 3]:
                    soft_region[k, 1] = soft_region[k, 3] = float(soft_xy[k, 1])

            hard_mask = np.zeros(n, dtype=bool)
            hard_mask[members] = True
            soft_mask = np.zeros(n_soft, dtype=bool)
            if local_soft.size:
                soft_mask[local_soft] = True
            stats = {
                "local_region_pad": float(pad),
                "local_region_pad_cells": int(pad_cells),
                "local_region_hard_core_span": float(hard_core_span),
                "adaptive_cold_cells": int(adaptive_cold_cells),
                "graph_region_cells": int(np.count_nonzero(region_mask)),
                "graph_target_cells": int(target_pool.size),
                "adaptive_region_xlo": float(xlo),
                "adaptive_region_ylo": float(ylo),
                "adaptive_region_xhi": float(xhi),
                "adaptive_region_yhi": float(yhi),
            }
            return (
                hard_region,
                soft_region,
                hard_mask,
                soft_mask,
                stats,
                target_pool,
                region_mask,
            )

        def _graph_candidate_score(cand: dict) -> float:
            trace = cand.get("trace", {})
            if cand.get("is_noop", False):
                return -1.0e30
            source = float(trace.get("source_field", trace.get("cluster_heat", 0.0)) or 0.0)
            target = float(trace.get("target_field", source) or source)
            relief = source - target
            target_cells = float(trace.get("graph_target_cells", 0) or 0)
            region_cells = float(trace.get("graph_region_cells", 0) or 0)
            adaptive_cells = float(trace.get("adaptive_cold_cells", 0) or 0)
            hard_disp = float(trace.get("hard_disp_mean", 0.0) or 0.0)
            score = relief
            score += 0.020 * np.log1p(target_cells)
            score += 0.010 * np.log1p(adaptive_cells)
            score += 0.002 * np.log1p(region_cells)
            score -= 0.001 * hard_disp
            cand["graph_score"] = float(score)
            trace["graph_score"] = float(score)
            return float(score)

        def _rank_graph_coldspot_candidates(candidates: list[dict]) -> list[dict]:
            scored = []
            for idx, cand in enumerate(candidates):
                scored.append((_graph_candidate_score(cand), idx, cand))
            scored.sort(key=lambda row: (-row[0], row[1]))
            return [cand for _score, _idx, cand in scored]

        def _hot_cluster_fallback_candidates(
            field: np.ndarray | None,
            hard_xy: np.ndarray,
            top_k: int,
        ) -> list[dict]:
            if field is not None:
                cell_w, cell_h = cw / nc, ch / nr
                mcol = np.clip((hard_xy[:, 0] / cell_w).astype(np.int64), 0, nc - 1)
                mrow = np.clip((hard_xy[:, 1] / cell_h).astype(np.int64), 0, nr - 1)
                macro_field = field[mrow, mcol]
            else:
                macro_field = np.zeros(n, dtype=np.float64)

            records = []
            for cid, raw_members in clusters.items():
                members = np.asarray(raw_members, dtype=np.int64)
                members = members[(members >= 0) & (members < n)]
                members = members[movable[:n][members]]
                if members.size < 2 or members.size > 64:
                    continue
                area = float(np.sum(sizes[members, 0] * sizes[members, 1]))
                heat = float(np.mean(macro_field[members])) if field is not None else 0.0
                cx = float(np.mean(hard_xy[members, 0]))
                cy = float(np.mean(hard_xy[members, 1]))
                xlo = float(np.min(hard_xy[members, 0] - hw[members]))
                ylo = float(np.min(hard_xy[members, 1] - hh[members]))
                xhi = float(np.max(hard_xy[members, 0] + hw[members]))
                yhi = float(np.max(hard_xy[members, 1] + hh[members]))
                records.append(
                    {
                        "cluster": int(cid),
                        "members": int(members.size),
                        "cluster_area": float(area),
                        "cluster_heat": float(heat),
                        "source_field": float(heat),
                        "target_field": float(heat),
                        "cluster_cx_before": float(cx),
                        "cluster_cy_before": float(cy),
                        "cluster_cx_after": float(cx),
                        "cluster_cy_after": float(cy),
                        "cluster_bbox_before": (xlo, ylo, xhi, yhi),
                        "cluster_bbox_after": (xlo, ylo, xhi, yhi),
                        "hard_disp_mean": 0.0,
                        "hard_disp_max": 0.0,
                        "hard_dx_mean": 0.0,
                        "hard_dy_mean": 0.0,
                        "soft_moved": 0,
                        "soft_disp_mean": 0.0,
                        "soft_disp_max": 0.0,
                    }
                )
            records.sort(
                key=lambda row: (
                    -float(row["cluster_heat"]),
                    -float(row["cluster_area"]),
                    int(row["cluster"]),
                )
            )
            return records[: max(0, int(top_k))]

        def _refine_coldspot_candidate(
            hard_xy: np.ndarray,
            soft_xy: np.ndarray,
            trace: dict,
        ) -> "tuple[np.ndarray, np.ndarray, float, dict]":
            score = float(_exact_proxy(_full(hard_xy, soft_xy), benchmark, plc))
            if ck_deadline is not None and time.monotonic() >= ck_deadline:
                return hard_xy, soft_xy, score, {}

            local = _coldspot_local_regions(hard_xy, soft_xy, int(trace.get("cluster", -1)))
            if local is None:
                return hard_xy, soft_xy, score, {}
            (
                local_h_region,
                local_s_region,
                local_h_mask,
                local_s_mask,
                local_stats,
                local_target_pool,
                local_region_mask,
            ) = local
            if not local_h_mask.any() and not local_s_mask.any():
                return hard_xy, soft_xy, score, {}

            refined_h = hard_xy.copy()
            refined_s = soft_xy.copy()
            scorer = IncrementalScorer(
                plc,
                benchmark,
                np.vstack([refined_h, refined_s]).astype(np.float64),
            )
            start_score = score
            hard_escape = 1.0e9
            soft_escape = max(0.0, float(const.HIER_COLDSPOT_LOCAL_SOFT_ESCAPE_MIN))
            stats_total = {
                "swap_accepts": 0,
                "hard_reloc_accepts": 0,
                "soft_reloc_accepts": 0,
            }
            local_hard_swap_k = max(1, int(const.HIER_COLDSPOT_LOCAL_HARD_SWAP_K))
            local_soft_swap_k = max(1, int(const.HIER_COLDSPOT_LOCAL_SOFT_SWAP_K))
            if _additive_spare(ck_deadline):
                extra_k = max(0, int(const.HIER_ADDITIVE_SWAP_EXTRA_K))
                local_hard_swap_k += extra_k
                local_soft_swap_k += extra_k
            fields = (False, True)
            for use_density in fields:
                refined_h, refined_s, got, score, _stats = _region_bounded_swap_relief(
                    refined_h,
                    refined_s,
                    sizes[:n],
                    hw,
                    hh,
                    soft_hw,
                    soft_hh,
                    cw,
                    ch,
                    local_h_mask,
                    local_s_mask,
                    benchmark,
                    scorer,
                    score,
                    local_h_region,
                    local_s_region,
                    deadline=ck_deadline,
                    rounds=max(1, int(const.HIER_COLDSPOT_LOCAL_SWAP_ROUNDS)),
                    hard_k=local_hard_swap_k,
                    soft_k=local_soft_swap_k,
                    region_bias=bias,
                    escape_min=hard_escape,
                    min_gain=float(const.HIER_SWAP_MIN_GAIN),
                    soft_barrier_gain=hier_soft_barrier_gain,
                    min_field_relief=float(const.HIER_SWAP_MIN_FIELD_RELIEF),
                    enable_hh=True,
                    enable_hs=True,
                    enable_ss=False,
                    use_density=use_density,
                )
                stats_total["swap_accepts"] += got
                refined_h, refined_s, got, score, _stats = _region_bounded_swap_relief(
                    refined_h,
                    refined_s,
                    sizes[:n],
                    hw,
                    hh,
                    soft_hw,
                    soft_hh,
                    cw,
                    ch,
                    np.zeros(n, dtype=bool),
                    local_s_mask,
                    benchmark,
                    scorer,
                    score,
                    local_h_region,
                    local_s_region,
                    deadline=ck_deadline,
                    rounds=max(1, int(const.HIER_COLDSPOT_LOCAL_SWAP_ROUNDS)),
                    hard_k=local_hard_swap_k,
                    soft_k=local_soft_swap_k,
                    region_bias=bias,
                    escape_min=soft_escape,
                    min_gain=float(const.HIER_SWAP_MIN_GAIN),
                    soft_barrier_gain=hier_soft_barrier_gain,
                    min_field_relief=float(const.HIER_SWAP_MIN_FIELD_RELIEF),
                    enable_hh=False,
                    enable_hs=False,
                    enable_ss=True,
                    use_density=use_density,
                )
                stats_total["swap_accepts"] += got

            refined_h, got, score = _relocation_moves(
                refined_h,
                sizes[:n],
                hw,
                hh,
                cw,
                ch,
                local_h_mask,
                n,
                plc,
                benchmark,
                scorer,
                score,
                deadline=ck_deadline,
                top_hot=max(1, int(const.HIER_COLDSPOT_LOCAL_HARD_RELOC_TOP_K)),
                n_targets=max(1, int(const.HIER_COLDSPOT_LOCAL_RELOC_TARGETS)),
                use_density=False,
                region_bbox=local_h_region,
                region_bias=bias,
                region_escape_min=hard_escape,
                propose_accept_min_gain=float(const.HIER_RELOC_PROPOSE_MIN_GAIN),
                target_pool=local_target_pool,
                region_mask=local_region_mask,
            )
            stats_total["hard_reloc_accepts"] += got

            if local_s_mask.any():
                refined_s, got, score = _soft_relocation_moves(
                    refined_s,
                    soft_hw,
                    soft_hh,
                    cw,
                    ch,
                    n,
                    plc,
                    benchmark,
                    scorer,
                    score,
                    deadline=ck_deadline,
                    top_hot=max(1, int(const.HIER_COLDSPOT_LOCAL_SOFT_RELOC_TOP_K)),
                    n_targets=max(1, int(const.HIER_COLDSPOT_LOCAL_RELOC_TARGETS)),
                    soft_movable=local_s_mask,
                    use_density=False,
                    region_bbox=local_s_region,
                    region_bias=bias,
                    region_escape_min=soft_escape,
                    accept_min_gain=hier_soft_barrier_gain,
                    target_pool=local_target_pool,
                    region_mask=local_region_mask,
                )
                stats_total["soft_reloc_accepts"] += got

            stats_total.update(
                {
                    "local_refine_initial_proxy": float(start_score),
                    "local_refine_proxy": float(score),
                    "local_refine_proxy_delta": float(score) - float(start_score),
                    "local_refine_hard_count": int(np.count_nonzero(local_h_mask)),
                    "local_refine_soft_count": int(np.count_nonzero(local_s_mask)),
                    "local_refine_pad": float(local_stats.get("local_region_pad", 0.0)),
                    **local_stats,
                }
            )
            return refined_h, refined_s, score, stats_total

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
        ck_candidate_id = 0
        ck_candidate_pool_id = 0
        cold_memory = np.zeros((nr, nc), dtype=bool)
        for _ in range(ck_rounds):
            if ck_deadline is not None and time.monotonic() >= ck_deadline:
                break
            field = _congestion_field(ck_scorer, nr, nc)
            if field is None:
                break
            cold_memory = _remember_cold_cells(field)
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
            generated = _coldspot_cluster_kick_candidates(
                cur_h,
                sizes[:n],
                hw,
                hh,
                cw,
                ch,
                movable[:n],
                n,
                clusters,
                coldspot_candidate_softs,
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
                kick_count=ck_gnn_kicks,
                plc=plc,
                benchmark_name=benchmark.name,
            )
            if not generated:
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
            candidate_records = []
            pool_id = ck_candidate_pool_id
            ck_candidate_pool_id += 1
            include_noop = ck_gnn_select or ck_oracle
            if include_noop:
                noop_trace = dict(generated[0][2])
                noop_trace.update(
                    {
                        "candidate_rank": 0,
                        "source_field": float(noop_trace.get("cluster_heat", 0.0)),
                        "target_field": float(noop_trace.get("cluster_heat", 0.0)),
                        "score": 0.0,
                        "soft_moved": 0,
                        "hard_disp_mean": 0.0,
                        "hard_disp_max": 0.0,
                        "hard_dx_mean": 0.0,
                        "hard_dy_mean": 0.0,
                        "soft_disp_mean": 0.0,
                        "soft_disp_max": 0.0,
                        "cluster_cx_after": float(noop_trace.get("cluster_cx_before", 0.0)),
                        "cluster_cy_after": float(noop_trace.get("cluster_cy_before", 0.0)),
                        "cluster_bbox_after": noop_trace.get("cluster_bbox_before"),
                    }
                )
                candidate_records.append(
                    {
                        "candidate_id": ck_candidate_id,
                        "candidate_pool_id": pool_id,
                        "candidate_rank": 0,
                        "hard": cur_h,
                        "soft": cur_s,
                        "trace": noop_trace,
                        "is_noop": True,
                        "old_proxy": float(cur_proxy),
                        "hierarchy_quality_before": float(cur_quality),
                    }
                )
                ck_candidate_id += 1
            for rank, (kh, ks, ck_trace) in enumerate(generated, start=1 if include_noop else 0):
                ck_trace = dict(ck_trace)
                ck_trace["candidate_rank"] = int(rank)
                cand_soft = ks if ks is not None else cur_s
                kh, cand_soft, refined_proxy, refine_stats = _refine_coldspot_candidate(
                    kh,
                    cand_soft,
                    ck_trace,
                )
                ck_trace.update(refine_stats)
                candidate_records.append(
                    {
                        "candidate_id": ck_candidate_id,
                        "candidate_pool_id": pool_id,
                        "candidate_rank": int(rank),
                        "hard": kh,
                        "soft": cand_soft,
                        "candidate_proxy_precomputed": float(refined_proxy),
                        "trace": ck_trace,
                        "is_noop": False,
                        "old_proxy": float(cur_proxy),
                        "hierarchy_quality_before": float(cur_quality),
                    }
                )
                ck_candidate_id += 1
            ranked_records = (
                rank_coldspot_kick_candidates(candidate_records, benchmark_name=benchmark.name)
                if ck_gnn_select
                else _rank_graph_coldspot_candidates(candidate_records)
            )
            if ck_gnn_select:
                policy_records = ranked_records[: min(ck_gnn_top_k, len(ranked_records))]
            else:
                policy_records = [
                    cand for cand in ranked_records if not cand.get("is_noop", False)
                ][: min(ck_graph_top_k, len(ranked_records))]
            selected_ids = {id(c) for c in policy_records}
            accepted_record = None
            accepted_any = False
            committed_any = False
            for selector_rank, cand in enumerate(ranked_records):
                cand["selector_rank"] = int(selector_rank)
                selected = id(cand) in selected_ids
                should_score = bool(ck_oracle or selected)
                if not should_score:
                    cand["accepted"] = False
                    cand["rejection_reason"] = (
                        "not_selected_by_gnn" if ck_gnn_select else "not_selected_by_graph"
                    )
                    continue
                if committed_any and selected and not ck_oracle:
                    cand["accepted"] = False
                    cand["rejection_reason"] = "not_evaluated_after_accept"
                    continue
                if cand.get("is_noop", False):
                    cand["candidate_proxy"] = float(cur_proxy)
                    cand["proxy_delta"] = 0.0
                    cand["hierarchy_quality_after"] = float(cur_quality)
                    cand["hierarchy_quality_delta"] = 0.0
                    cand["accepted"] = False
                    cand["rejection_reason"] = "no_op"
                    cand["committed"] = False
                    continue
                kh = cand["hard"]
                ks = cand["soft"]
                if "candidate_proxy_precomputed" in cand:
                    kproxy = float(cand["candidate_proxy_precomputed"])
                else:
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
                cand["candidate_proxy"] = float(kproxy)
                cand["proxy_delta"] = float(kproxy) - float(cur_proxy)
                cand["hierarchy_quality_after"] = float(kquality)
                cand["hierarchy_quality_delta"] = float(kquality) - float(cur_quality)
                cand["accepted"] = bool(accepted)
                cand["rejection_reason"] = None if accepted else reason
                cand["committed"] = bool(accepted and selected and not committed_any)
                if cand["committed"]:
                    accepted_record = cand
                    accepted_any = True
                    committed_any = True
            for cand in candidate_records:
                log_gnn_event(
                    "hier_coldspot_candidate",
                    benchmark=benchmark.name,
                    operator="coldspot_tightening",
                    kind="coldspot_kick",
                    field="congestion",
                    candidate_id=int(cand["candidate_id"]),
                    candidate_pool_id=int(cand["candidate_pool_id"]),
                    candidate_pool_size=int(len(candidate_records)),
                    selector_enabled=bool(ck_gnn_select),
                    oracle_enabled=bool(ck_oracle),
                    selector_rank=cand.get("selector_rank"),
                    selector_top_k=int(ck_gnn_top_k if ck_gnn_select else ck_graph_top_k),
                    selected_by_gnn=bool(ck_gnn_select and id(cand) in selected_ids),
                    graph_selector_enabled=bool(not ck_gnn_select),
                    selected_by_graph=bool(not ck_gnn_select and id(cand) in selected_ids),
                    selected_by_policy=bool(id(cand) in selected_ids),
                    gnn_score=cand.get("gnn_score"),
                    gnn_rank_error=cand.get("gnn_rank_error"),
                    is_noop=bool(cand.get("is_noop", False)),
                    field_gap=float(field_gap),
                    min_field_gap=float(ck_min_field_gap),
                    old_proxy=float(cur_proxy),
                    candidate_proxy=cand.get("candidate_proxy"),
                    proxy_delta=cand.get("proxy_delta"),
                    hierarchy_quality_before=float(cur_quality),
                    hierarchy_quality_after=cand.get("hierarchy_quality_after"),
                    hierarchy_quality_delta=cand.get("hierarchy_quality_delta"),
                    accepted=bool(cand.get("accepted", False)),
                    committed=bool(cand.get("committed", False)),
                    rejection_reason=cand.get("rejection_reason"),
                    **cand["trace"],
                )
            if accepted_any and accepted_record is not None:
                cur_h = accepted_record["hard"]
                cur_s = accepted_record["soft"]
                cur_proxy = float(accepted_record["candidate_proxy"])
                cur_quality = float(accepted_record["hierarchy_quality_after"])
                ck_scorer = IncrementalScorer(
                    plc,
                    benchmark,
                    np.vstack([cur_h, cur_s]).astype(np.float64),
                )
                refreshed_field = _congestion_field(ck_scorer, nr, nc)
                if refreshed_field is not None:
                    cold_memory = _remember_cold_cells(refreshed_field)
                ck_acc += 1
        graph_fallback_acc = 0
        if ck_acc == 0 and (ck_deadline is None or time.monotonic() < ck_deadline):
            fallback_field = _congestion_field(ck_scorer, nr, nc)
            if fallback_field is not None:
                cold_memory = _remember_cold_cells(fallback_field)
            fallback_top_k = max(1, int(const.HIER_COLDSPOT_GRAPH_FALLBACK_TOP_K))
            fallback_records = _hot_cluster_fallback_candidates(
                fallback_field,
                cur_h,
                fallback_top_k,
            )
            accepted_fallback = None
            fallback_log_records = []
            for rank, fallback_trace in enumerate(fallback_records):
                if ck_deadline is not None and time.monotonic() >= ck_deadline:
                    break
                fallback_trace = dict(fallback_trace)
                fallback_trace["candidate_rank"] = int(rank)
                fallback_trace["graph_fallback"] = True
                fh, fs, fproxy, refine_stats = _refine_coldspot_candidate(
                    cur_h,
                    cur_s,
                    fallback_trace,
                )
                fallback_trace.update(refine_stats)
                fquality = hierarchy_quality_metric(fh, clusters)
                accepted = (
                    _hard_valid(fh)
                    and fquality <= cur_quality + ck_quality_budget
                    and fproxy <= cur_proxy + ck_budget
                    and fproxy <= base_proxy + ck_total
                    and fproxy < cur_proxy - ck_min_gain
                )
                if not _hard_valid(fh):
                    reason = "hard_legality_failed"
                elif fquality > cur_quality + ck_quality_budget:
                    reason = "hierarchy_quality_failed"
                elif fproxy > cur_proxy + ck_budget or fproxy > base_proxy + ck_total:
                    reason = "proxy_budget_failed"
                elif fproxy >= cur_proxy - ck_min_gain:
                    reason = "exact_proxy_failed"
                else:
                    reason = "accepted"

                candidate = {
                    "hard": fh,
                    "soft": fs,
                    "candidate_proxy": float(fproxy),
                    "hierarchy_quality_after": float(fquality),
                    "accepted": bool(accepted),
                    "candidate_id": int(ck_candidate_id),
                    "candidate_pool_id": int(ck_candidate_pool_id),
                    "candidate_pool_size": int(len(fallback_records)),
                    "selector_rank": int(rank),
                    "rejection_reason": None if accepted else reason,
                    "proxy_delta": float(fproxy) - float(cur_proxy),
                    "hierarchy_quality_delta": float(fquality) - float(cur_quality),
                    "trace": fallback_trace,
                }
                if accepted and (
                    accepted_fallback is None
                    or fproxy < float(accepted_fallback["candidate_proxy"])
                ):
                    accepted_fallback = candidate
                fallback_log_records.append(candidate)
                ck_candidate_id += 1

            committed_fallback_id = (
                int(accepted_fallback["candidate_id"]) if accepted_fallback is not None else None
            )
            for candidate in fallback_log_records:
                log_gnn_event(
                    "hier_coldspot_candidate",
                    benchmark=benchmark.name,
                    operator="coldspot_tightening",
                    kind="graph_local_fallback",
                    field="congestion",
                    candidate_id=int(candidate["candidate_id"]),
                    candidate_pool_id=int(candidate["candidate_pool_id"]),
                    candidate_pool_size=int(candidate["candidate_pool_size"]),
                    selector_enabled=False,
                    oracle_enabled=False,
                    selector_rank=int(candidate["selector_rank"]),
                    selector_top_k=int(fallback_top_k),
                    selected_by_gnn=False,
                    graph_selector_enabled=False,
                    selected_by_graph=False,
                    selected_by_policy=True,
                    is_noop=False,
                    field_gap=None,
                    min_field_gap=float(ck_min_field_gap),
                    old_proxy=float(cur_proxy),
                    candidate_proxy=float(candidate["candidate_proxy"]),
                    proxy_delta=float(candidate["proxy_delta"]),
                    hierarchy_quality_before=float(cur_quality),
                    hierarchy_quality_after=float(candidate["hierarchy_quality_after"]),
                    hierarchy_quality_delta=float(candidate["hierarchy_quality_delta"]),
                    accepted=bool(candidate["accepted"]),
                    committed=bool(candidate["candidate_id"] == committed_fallback_id),
                    rejection_reason=candidate["rejection_reason"],
                    **candidate["trace"],
                )

            ck_candidate_pool_id += 1
            if accepted_fallback is not None:
                cur_h = accepted_fallback["hard"]
                cur_s = accepted_fallback["soft"]
                cur_proxy = float(accepted_fallback["candidate_proxy"])
                cur_quality = float(accepted_fallback["hierarchy_quality_after"])
                ck_scorer = IncrementalScorer(
                    plc,
                    benchmark,
                    np.vstack([cur_h, cur_s]).astype(np.float64),
                )
                refreshed_field = _congestion_field(ck_scorer, nr, nc)
                if refreshed_field is not None:
                    cold_memory = _remember_cold_cells(refreshed_field)
                graph_fallback_acc = 1
                ck_acc += 1
                _log(
                    f"  [hier] graph-local coldspot fallback: 1 accept, "
                    f"proxy {base_proxy:.4f}->{cur_proxy:.4f}"
                )
        soft_only_acc = 0
        if (
            bool(const.HIER_COLDSPOT_SOFT_ONLY)
            and ck_acc == 0
            and n_soft
            and bool(np.any(soft_mov))
            and soft_region is not None
            and (ck_deadline is None or time.monotonic() < ck_deadline)
        ):
            soft_only_before = float(cur_proxy)
            soft_only_field = _congestion_field(ck_scorer, nr, nc)
            soft_only_target_cells = 0
            if soft_only_field is not None:
                cold_memory = _remember_cold_cells(soft_only_field)
                target_mask = cold_memory & ~_occupied_cells(cur_h, cur_s)
                target_pool = np.flatnonzero(target_mask.ravel()).astype(np.int64)
                soft_only_target_cells = int(target_pool.size)
                if target_pool.size:
                    cur_s, soft_only_acc, cur_proxy = _soft_relocation_moves(
                        cur_s,
                        soft_hw,
                        soft_hh,
                        cw,
                        ch,
                        n,
                        plc,
                        benchmark,
                        ck_scorer,
                        cur_proxy,
                        deadline=ck_deadline,
                        top_hot=max(1, int(const.HIER_COLDSPOT_SOFT_ONLY_TOP_K)),
                        n_targets=max(1, int(const.HIER_COLDSPOT_SOFT_ONLY_TARGETS)),
                        soft_movable=soft_mov,
                        use_density=False,
                        region_bbox=soft_region,
                        region_bias=bias,
                        region_escape_min=float(const.HIER_COLDSPOT_LOCAL_SOFT_ESCAPE_MIN),
                        accept_min_gain=max(
                            hier_soft_barrier_gain,
                            float(const.HIER_COLDSPOT_SOFT_ONLY_MIN_GAIN),
                        ),
                        target_pool=target_pool,
                        region_mask=target_mask,
                    )
                _trace_pass(
                    "coldspot_soft_only",
                    soft_only_before,
                    cur_proxy,
                    soft_only_acc,
                    quality=float(cur_quality),
                    target_cells=int(soft_only_target_cells),
                )
                _log(
                    f"  [hier] coldspot soft-only fallback: {soft_only_acc} accepts, "
                    f"targets={soft_only_target_cells}, "
                    f"proxy {soft_only_before:.4f}->{cur_proxy:.4f}"
                )
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
            graph_fallback_accepts=int(graph_fallback_acc),
            soft_only_accepts=int(soft_only_acc),
        )

        if not ck_skip_micro and region is not None and soft_region is not None:
            post_ck_micro_deadline = _deadline(
                float(const.HIER_POST_COLDSPOT_MICRO_SHIFT_BUDGET_S), ck_deadline
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
        elif ck_skip_micro:
            _log("  [hier] post-coldspot micro-shift replay: skipped by GNN selector")
            _trace_pass(
                "post_coldspot_micro_shift",
                cur_proxy,
                cur_proxy,
                0,
                quality=hierarchy_quality_metric(legal, clusters),
                skipped_by_gnn_coldspot_selector=True,
            )

        survivor_deadline = _deadline(float(const.HIER_SURVIVOR_BUDGET_S), rdeadline)
        pre_survivor_score = cur_proxy
        survivor_t0 = time.monotonic()
        legal, s_pos, survivor_acc, cur_proxy = _parallel_survivor_search(
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
            cur_proxy,
            clusters,
            cluster_softs=csofts,
            bridge_softs=bridge_softs,
            hard_region=region,
            soft_region=soft_region,
            deadline=survivor_deadline,
        )
        if not _hard_valid(legal):
            legal, s_pos, cur_proxy = best_h.copy(), best_s.copy(), best_score
            survivor_acc = 0
        survivor_stats = getattr(_parallel_survivor_search, "last_stats", {})
        _log(
            f"  [hier] survivor search: {survivor_acc} accepts, "
            f"proxy {pre_survivor_score:.4f}->{cur_proxy:.4f}"
        )
        _trace_pass(
            "survivor_search",
            pre_survivor_score,
            cur_proxy,
            survivor_acc,
            quality=hierarchy_quality_metric(legal, clusters),
            gpu_rank=bool(survivor_stats.get("gpu_rank", False)),
        )
        _record_plateau(
            "survivor_search",
            pre_survivor_score,
            cur_proxy,
            survivor_acc,
            time.monotonic() - survivor_t0,
            candidates=int(survivor_stats.get("candidates", 0)),
            legal=int(survivor_stats.get("legal", 0)),
            scored=int(survivor_stats.get("scored", 0)),
            quality=hierarchy_quality_metric(legal, clusters),
            gpu_rank=bool(survivor_stats.get("gpu_rank", False)),
        )

        state = PlacementState(
            legal.copy(),
            s_pos.copy(),
            float(_exact_proxy(_full_tensor(legal, s_pos), benchmark, plc)),
        )
        out = torch.tensor(state.full().astype(np.float32), dtype=torch.float32)
        proxy = float(state.proxy)
        hq_breakdown = hierarchy_quality_breakdown(legal, clusters)
        margin_eps = float(const.HIER_LEGALITY_MARGIN_EPS)
        legality_margin = _hard_legality_margin(legal, margin_eps)
        log_gnn_event(
            "hier_final",
            benchmark=benchmark.name,
            proxy=float(proxy),
            pre_relief_proxy=float(pre_relief),
            hierarchy_quality=float(hq_breakdown["quality"]),
            hierarchy_quality_radius=float(hq_breakdown["radius"]),
            hierarchy_quality_bbox=float(hq_breakdown["bbox"]),
            hierarchy_quality_crowd=float(hq_breakdown["crowd"]),
            clusters=int(len(clusters)),
            hierarchy_edges=int(len(hierarchy.edges)),
            hierarchy_oversize_split=True,
            hierarchy_split_parents=int(len(hierarchy.split_parents)),
            seed_portfolio=True,
            selected_seed=selected_seed_name,
            seed_candidates=int(len(seed_rows)),
            additive_candidate_pools=True,
            legality_margin_audit=True,
            legality_margin_eps=float(margin_eps),
            legality_pair_margin=float(legality_margin["pair_margin"]),
            legality_bounds_margin=float(legality_margin["bounds_margin"]),
            legality_min_margin=float(legality_margin["min_margin"]),
            hierarchy_confidence_mean=float(
                np.mean(list(hierarchy.cluster_confidence.values()))
                if hierarchy.cluster_confidence
                else 0.0
            ),
            group_weight=int(gw),
        )
        _log(
            f"  [hier] {len(clusters)} clusters, {len(hierarchy.edges)} edges, "
            f"oversize=1, "
            f"seed={selected_seed_name}, "
            f"additive=1, "
            f"margin={float(legality_margin['min_margin']):.3f}, "
            f"weight={gw}: proxy={proxy:.4f} "
            f"(pre-relief {pre_relief:.4f}; hierarchy-preserving NON-proxy mode)"
        )
        flush_plateau_events()
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
