"""Main macro-placement pipeline."""

import os
import random
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from macro_place.benchmark import Benchmark

from placer.config import _GPU_BACKEND, _GPU_DEVICE_NAME, _log
from placer.scoring.exact import _exact_proxy

_FALSE_ENV = {"", "0", "false", "FALSE", "no", "NO", "off", "OFF"}


class MacroPlacer:
    """Hierarchy-preserving macro placer."""

    def __init__(
        self,
        n_restarts: int = 50,
        noise_fracs: Optional[List[float]] = None,
        seed: int = 42,
        time_budget_s: float = 150.0,
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
        _env_budget = os.environ.get("V2_TIME_BUDGET")
        if _env_budget:
            try:
                self.time_budget_s = float(_env_budget)
            except ValueError:
                pass

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
        )
        from dreamplace_bridge.bookshelf_to_pb import read_dreamplace_positions_full
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
        from placer.local_search.hierarchy_swaps import _region_bounded_swap_relief
        from placer.local_search.region_expand import expand_regions_by_congestion
        from placer.local_search.relocation import (
            _relocation_moves,
            _soft_relocation_moves,
        )
        from placer.scoring.incremental import IncrementalScorer

        if not _dp_available():
            return None
        iccad = Path("external/MacroPlacement/Testcases/ICCAD04") / benchmark.name
        if not iccad.exists():
            return None

        def _auto_cuda_flag(name: str, default: str = "0") -> bool:
            raw = os.environ.get(name, default).strip()
            if raw.lower() == "auto":
                return _GPU_BACKEND == "cuda"
            return raw not in _FALSE_ENV

        hier_reloc_propose_all = _auto_cuda_flag("V2_HIER_RELOC_PROPOSE_ALL", "0")
        hier_soft_propose_all = _auto_cuda_flag("V2_HIER_SOFT_RELOC_PROPOSE_ALL", "0")
        hier_reloc_top_m_raw = os.environ.get("V2_HIER_RELOC_PROPOSE_TOP_M", "64").strip()
        hier_soft_top_m_raw = os.environ.get("V2_HIER_SOFT_RELOC_PROPOSE_TOP_M", "96").strip()
        hier_reloc_top_m = int(hier_reloc_top_m_raw) if hier_reloc_top_m_raw else None
        hier_soft_top_m = int(hier_soft_top_m_raw) if hier_soft_top_m_raw else None

        from placer.plc.loader import _load_plc

        plc = _load_plc(benchmark.name, benchmark)
        n = benchmark.num_hard_macros
        n_soft = benchmark.num_soft_macros
        cw, ch = float(benchmark.canvas_width), float(benchmark.canvas_height)
        sizes = benchmark.macro_sizes.numpy().astype(np.float64)
        hw, hh = sizes[:n, 0] / 2.0, sizes[:n, 1] / 2.0
        soft_hw, soft_hh = sizes[n : n + n_soft, 0] / 2.0, sizes[n : n + n_soft, 1] / 2.0
        movable = benchmark.get_movable_mask().numpy()
        gw = max(1, int(os.environ.get("V2_HIER_GROUP_WEIGHT", "8")))

        labels, clusters = derive_hard_clusters(
            plc,
            n,
            n_soft=n_soft,
            max_fanout=cluster_max_fanout(),
            min_edge=cluster_min_edge(),
        )
        bridge_ratio = float(os.environ.get("V2_HIER_BRIDGE_SOFT_RATIO", "0.6"))
        if os.environ.get("V2_HIER_BRIDGE_SOFTS", "1").strip() not in _FALSE_ENV:
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

        try:
            run_dreamplace(
                str(iccad),
                plc=plc,
                scratch_root="/tmp/dreamplace_v1_hier",
                iterations=300,
                num_threads=2,
                soft_macros_movable=True,
                cluster_groups=(groups or None),
                group_weight=gw,
            )
            hard, soft = read_dreamplace_positions_full(
                plc, f"/tmp/dreamplace_v1_hier/{benchmark.name}", benchmark.name
            )
        except Exception as exc:
            _log(f"  [hier] DREAMPlace failed: {type(exc).__name__}: {exc}")
            return None

        order = []
        for mem in sorted(clusters.values(), key=lambda m: -m.size):
            order += sorted((int(x) for x in mem), key=lambda i: -(sizes[i, 0] * sizes[i, 1]))
        order += sorted(
            [i for i in range(n) if labels[i] < 0], key=lambda i: -(sizes[i, 0] * sizes[i, 1])
        )
        legal = _will_legalize(
            hard.copy(),
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
        legal = _will_legalize(
            legal,
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

        pos = np.vstack([legal, soft]).astype(np.float64)
        s_pos = soft.copy()
        s_score = float(_exact_proxy(torch.tensor(pos, dtype=torch.float32), benchmark, plc))
        scorer = IncrementalScorer(plc, benchmark, pos.copy())
        soft_mov = movable[n : n + n_soft]
        for use_density in (False, True):
            s_pos, _, s_score = _soft_relocation_moves(
                s_pos,
                soft_hw,
                soft_hh,
                cw,
                ch,
                n,
                plc,
                benchmark,
                scorer,
                s_score,
                deadline=time.monotonic() + 30,
                top_hot=1024,
                n_targets=6,
                soft_movable=soft_mov,
                use_density=use_density,
                propose_all=hier_soft_propose_all,
                propose_top_m=hier_soft_top_m,
            )

        pre_relief = s_score
        if os.environ.get("V2_HIER_REGION_RELIEF", "1").strip() not in _FALSE_ENV:
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
            )
            if os.environ.get("V2_HIER_CONG_EXPAND_REGIONS", "1").strip() not in _FALSE_ENV:
                nr, nc = int(benchmark.grid_rows), int(benchmark.grid_cols)
                expand_field = _congestion_field(plc, nr, nc)
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
                    hot_percentile=float(os.environ.get("V2_HIER_REGION_EXPAND_HOT_PCT", "60")),
                    max_expand_frac=float(os.environ.get("V2_HIER_REGION_EXPAND_FRAC", "0.08")),
                    side_band=max(1, int(os.environ.get("V2_HIER_REGION_EXPAND_BAND", "3"))),
                )
                if n_expanded:
                    _log(f"  [hier] congestion-expanded regions: {n_expanded} clusters")
            bias = float(os.environ.get("V2_REGION_BIAS", "1.0"))
            escape_min = float(os.environ.get("V2_HIER_REGION_ESCAPE_MIN", "0.002"))
            rounds = max(1, int(os.environ.get("V2_HIER_REGION_ROUNDS", "2")))
            rdeadline = time.monotonic() + float(os.environ.get("V2_HIER_REGION_BUDGET_S", "40"))
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
                for use_density in (False, True):
                    h_pos, _, r_score = _relocation_moves(
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
                        propose_all=hier_reloc_propose_all,
                        propose_top_m=hier_reloc_top_m,
                        region_bbox=region,
                        region_bias=bias,
                        region_escape_min=escape_min,
                    )
                for use_density in (False, True):
                    s_pos, _, r_score = _soft_relocation_moves(
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
                        propose_all=hier_soft_propose_all,
                        propose_top_m=hier_soft_top_m,
                        region_bbox=soft_region,
                        region_bias=bias,
                        region_escape_min=escape_min,
                    )
            if _hard_valid(h_pos) and r_score < best_score - 1e-9:
                best_h, best_s, best_score = h_pos.copy(), s_pos.copy(), r_score
            if os.environ.get("V2_HIER_DECOMPRESS", "1").strip() not in _FALSE_ENV:
                pre_decomp_h, pre_decomp_s, pre_decomp_score = h_pos.copy(), s_pos.copy(), r_score
                d_deadline = min(
                    rdeadline,
                    time.monotonic() + float(os.environ.get("V2_HIER_DECOMPRESS_BUDGET_S", "18")),
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
                    rounds=max(1, int(os.environ.get("V2_HIER_DECOMPRESS_ROUNDS", "2"))),
                    hot_percentile=float(os.environ.get("V2_HIER_DECOMPRESS_HOT_PCT", "65")),
                    quality_budget=float(os.environ.get("V2_HIER_QUALITY_BUDGET", "0.03")),
                    min_proxy_gain=float(os.environ.get("V2_HIER_DECOMPRESS_MIN_GAIN", "0.0001")),
                )
                invalid_decomp = not _hard_valid(h_pos)
                weak_decomp = (
                    d_acc
                    and os.environ.get("V2_HIER_ROLLBACK_WEAK_DECOMP", "1").strip()
                    not in _FALSE_ENV
                    and r_score
                    > pre_decomp_score
                    - float(os.environ.get("V2_HIER_DECOMPRESS_MIN_GAIN", "0.0001"))
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
            if os.environ.get("V2_HIER_REGION_SWAPS", "1").strip() not in _FALSE_ENV:
                swap_rounds = max(1, int(os.environ.get("V2_HIER_REGION_SWAP_ROUNDS", "2")))
                swap_deadline = min(
                    rdeadline,
                    time.monotonic() + float(os.environ.get("V2_HIER_REGION_SWAP_BUDGET_S", "20")),
                )
                hard_k = max(1, int(os.environ.get("V2_HIER_HARD_SWAP_K", "16")))
                soft_k = max(1, int(os.environ.get("V2_HIER_SOFT_SWAP_K", "48")))
                swap_min_gain = float(os.environ.get("V2_HIER_SWAP_MIN_GAIN", "0.00001"))
                swap_min_field = float(os.environ.get("V2_HIER_SWAP_MIN_FIELD_RELIEF", "0.0"))
                enable_hh = os.environ.get("V2_HIER_SWAP_HH", "1").strip() not in _FALSE_ENV
                enable_hs = os.environ.get("V2_HIER_SWAP_HS", "1").strip() not in _FALSE_ENV
                enable_ss = os.environ.get("V2_HIER_SWAP_SS", "1").strip() not in _FALSE_ENV
                use_density_swaps = (
                    os.environ.get("V2_HIER_SWAP_DENSITY_FIELD", "1").strip() not in _FALSE_ENV
                )
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

        if os.environ.get("V2_HIER_COLDSPOT_KICK", "1").strip() not in _FALSE_ENV:
            from placer.local_search.lsmc_explore import _coldspot_cluster_kick

            ck_budget = float(os.environ.get("V2_HIER_COLDSPOT_BUDGET", "0.0"))
            ck_total = float(os.environ.get("V2_HIER_COLDSPOT_TOTAL", "0.0"))
            ck_min_gain = float(os.environ.get("V2_HIER_COLDSPOT_MIN_GAIN", "0.0001"))
            ck_quality_budget = float(os.environ.get("V2_HIER_COLDSPOT_QUALITY_BUDGET", "0.01"))
            ck_rounds = max(1, int(os.environ.get("V2_HIER_COLDSPOT_ROUNDS", "8")))
            ck_deadline = time.monotonic() + float(
                os.environ.get("V2_HIER_COLDSPOT_BUDGET_S", "30")
            )
            nr, nc = int(benchmark.grid_rows), int(benchmark.grid_cols)
            soft_mov = movable[n : n + n_soft]

            def _full(h, sft):
                return torch.tensor(np.vstack([h, sft]).astype(np.float32), dtype=torch.float32)

            cur_h, cur_s = legal.copy(), s_pos.copy()
            base_proxy = float(_exact_proxy(_full(cur_h, cur_s), benchmark, plc))
            cur_proxy, cur_quality = base_proxy, hierarchy_quality_metric(cur_h, clusters)
            ck_rng = np.random.default_rng(0)
            ck_acc = 0
            for _ in range(ck_rounds):
                if time.monotonic() >= ck_deadline:
                    break
                float(_exact_proxy(_full(cur_h, cur_s), benchmark, plc))
                field = _congestion_field(plc, nr, nc)
                if field is None:
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
                )
                if res is None:
                    continue
                kh, ks = res
                ks = ks if ks is not None else cur_s
                kproxy = float(_exact_proxy(_full(kh, ks), benchmark, plc))
                kquality = hierarchy_quality_metric(kh, clusters)
                if (
                    kquality <= cur_quality + ck_quality_budget
                    and kproxy <= cur_proxy + ck_budget
                    and kproxy <= base_proxy + ck_total
                    and kproxy < cur_proxy - ck_min_gain
                ):
                    cur_h, cur_s, cur_proxy, cur_quality = kh, ks, kproxy, kquality
                    ck_acc += 1
            legal, s_pos = cur_h, cur_s
            _log(
                f"  [hier] coldspot tightening: {ck_acc} accepts, "
                f"quality={cur_quality:.4f}, proxy {base_proxy:.4f}->{cur_proxy:.4f}"
            )

        out = torch.tensor(np.vstack([legal, s_pos]).astype(np.float32), dtype=torch.float32)
        proxy = float(_exact_proxy(out, benchmark, plc))
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
