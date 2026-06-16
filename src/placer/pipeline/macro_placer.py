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
            derive_cluster_softs,
            derive_hard_clusters,
            hier_region_density,
            hier_region_margin,
            hier_region_singleton,
        )
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
        csofts = derive_cluster_softs(plc, n, n_soft, labels)
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
            bias = float(os.environ.get("V2_REGION_BIAS", "1.0"))
            rounds = max(1, int(os.environ.get("V2_HIER_REGION_ROUNDS", "2")))
            rdeadline = time.monotonic() + float(os.environ.get("V2_HIER_REGION_BUDGET_S", "40"))
            h_pos = legal.copy()
            full = np.vstack([h_pos, s_pos]).astype(np.float64)
            r_score = float(_exact_proxy(torch.tensor(full, dtype=torch.float32), benchmark, plc))
            rscorer = IncrementalScorer(plc, benchmark, full.copy())
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
                        region_bbox=region,
                        region_bias=bias,
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
                    )
            legal = _will_legalize(
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

        if os.environ.get("V2_HIER_COLDSPOT_KICK", "1").strip() not in _FALSE_ENV:
            from placer.local_search.fields import _congestion_field
            from placer.local_search.lsmc_explore import _coldspot_cluster_kick

            ck_budget = float(os.environ.get("V2_HIER_COLDSPOT_BUDGET", "0.05"))
            ck_total = float(os.environ.get("V2_HIER_COLDSPOT_TOTAL", "0.15"))
            ck_rounds = max(1, int(os.environ.get("V2_HIER_COLDSPOT_ROUNDS", "8")))
            ck_deadline = time.monotonic() + float(
                os.environ.get("V2_HIER_COLDSPOT_BUDGET_S", "30")
            )
            nr, nc = int(benchmark.grid_rows), int(benchmark.grid_cols)
            soft_mov = movable[n : n + n_soft]

            def _intra_spread(hard):
                s = 0.0
                for mem in clusters.values():
                    p = hard[mem]
                    s += float(
                        np.hypot(p[:, 0].max() - p[:, 0].min(), p[:, 1].max() - p[:, 1].min())
                    )
                return s

            def _full(h, sft):
                return torch.tensor(np.vstack([h, sft]).astype(np.float32), dtype=torch.float32)

            cur_h, cur_s = legal.copy(), s_pos.copy()
            base_proxy = float(_exact_proxy(_full(cur_h, cur_s), benchmark, plc))
            cur_proxy, cur_spread = base_proxy, _intra_spread(cur_h)
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
                kspread = _intra_spread(kh)
                if (
                    kspread < cur_spread - 1e-9
                    and kproxy <= cur_proxy + ck_budget
                    and kproxy <= base_proxy + ck_total
                ):
                    cur_h, cur_s, cur_proxy, cur_spread = kh, ks, kproxy, kspread
                    ck_acc += 1
            legal, s_pos = cur_h, cur_s
            _log(
                f"  [hier] coldspot tightening: {ck_acc} accepts, "
                f"spread {cur_spread:.1f}, proxy {base_proxy:.4f}->{cur_proxy:.4f}"
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
