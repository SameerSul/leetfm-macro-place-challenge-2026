"""Macro placer entrypoint with hierarchy-only execution."""

import random
import time
from typing import List, Optional

import numpy as np
import torch
from macro_place.benchmark import Benchmark

from placer.pipeline.hierarchy_floorplan import run_hierarchy_floorplan
from placer.local_search.plateau_telemetry import log_plateau_event
from utils import constants as const
from utils.config import _GPU_BACKEND, _GPU_DEVICE_NAME, _log


class MacroPlacer:
    """Hierarchy-preserving macro placer."""

    def __init__(
        self,
        n_restarts: int = 50,
        noise_fracs: Optional[List[float]] = None,
        seed: int = 42,
        time_budget_s: float = const.TIME_BUDGET_S,
        event_sink=None,
        dreamplace_sample_every: int = 10,
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
            0.12,
            0.10,
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
        self.event_sink = event_sink
        self.dreamplace_sample_every = max(1, int(dreamplace_sample_every))

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

    def _hierarchy_floorplan(self, benchmark: Benchmark) -> torch.Tensor | None:
        return run_hierarchy_floorplan(
            benchmark,
            event_sink=self.event_sink,
            dreamplace_sample_every=self.dreamplace_sample_every,
        )

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        api_t0 = time.perf_counter()
        succeeded = False
        emitter = self.event_sink
        if emitter is not None:
            from visualizer.events import EventEmitter, emit_event

            if not isinstance(emitter, EventEmitter):
                emitter = EventEmitter(emitter)
            self.event_sink = emitter
            emit_event(
                emitter,
                "run_metadata",
                benchmark=str(benchmark.name),
                canvas=[float(benchmark.canvas_width), float(benchmark.canvas_height)],
                macro_names=list(benchmark.macro_names),
                macro_sizes=benchmark.macro_sizes.detach().cpu().tolist(),
                macro_fixed=benchmark.macro_fixed.detach().cpu().tolist(),
                num_hard_macros=int(benchmark.num_hard_macros),
                group_weight=int(const.HIER_GROUP_WEIGHT),
                net_nodes=[nodes.detach().cpu().tolist() for nodes in benchmark.net_nodes],
                net_weights=benchmark.net_weights.detach().cpu().tolist(),
                port_positions=benchmark.port_positions.detach().cpu().tolist(),
                macro_pin_offsets=[
                    pins.detach().cpu().tolist() for pins in benchmark.macro_pin_offsets
                ],
            )
            emit_event(
                emitter,
                "checkpoint",
                reason="initial_placement",
                positions=benchmark.macro_positions.detach().cpu().tolist(),
                metrics_stale=True,
            )
            setattr(benchmark, "_visualizer_event_sink", emitter)
            setattr(benchmark, "_visualizer_dreamplace_sample_every", self.dreamplace_sample_every)
        try:
            result = self._clamp_in_bounds(self._place_impl(benchmark), benchmark)
            succeeded = True
            if emitter is not None:
                from visualizer.events import emit_event

                if hasattr(emitter, "finish"):
                    emitter.finish()
                final_metrics = getattr(benchmark, "_visualizer_last_metrics", None)
                emit_event(
                    emitter,
                    "completion",
                    positions=result.detach().cpu().tolist(),
                    metrics=final_metrics,
                    metrics_stale=final_metrics is None,
                    elapsed_s=float(time.perf_counter() - api_t0),
                )
            return result
        except BaseException as exc:
            if emitter is not None:
                from visualizer.events import emit_event

                if hasattr(emitter, "finish"):
                    emitter.finish()
                emit_event(
                    emitter,
                    "error",
                    error=type(exc).__name__,
                    message=str(exc),
                    elapsed_s=float(time.perf_counter() - api_t0),
                )
            raise
        finally:
            log_plateau_event(
                "hier_stage_timing",
                benchmark=str(getattr(benchmark, "_hierarchy_trace_name", str(benchmark.name))),
                stage="placer_api_total",
                elapsed_s=float(time.perf_counter() - api_t0),
                succeeded=bool(succeeded),
            )
            for attr in (
                "_visualizer_event_sink",
                "_visualizer_dreamplace_sample_every",
                "_visualizer_hierarchy_metric",
                "_visualizer_last_metrics",
            ):
                if hasattr(benchmark, attr):
                    delattr(benchmark, attr)

    def _place_impl(self, benchmark: Benchmark) -> torch.Tensor:
        np.random.seed(self.seed)
        random.seed(self.seed)

        _log(f"[GPU] backend={_GPU_BACKEND} device={_GPU_DEVICE_NAME} | benchmark={benchmark.name}")

        hier = self._hierarchy_floorplan(benchmark)
        if hier is None:
            from dreamplace_bridge.run_bridge import availability_error

            detail = availability_error()
            suffix = f": {detail}" if detail else ""
            raise RuntimeError(
                "hierarchy floorplan path unavailable; proxy fallback has been removed" + suffix
            )
        return hier
