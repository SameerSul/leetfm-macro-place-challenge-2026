"""Shared orchestration objects for the hierarchy placement pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class PlacementState:
    """Mutable hard/soft placement state passed between hierarchy stages."""

    hard: np.ndarray
    soft: np.ndarray
    proxy: float

    def full(self) -> np.ndarray:
        return np.vstack([self.hard, self.soft]).astype(np.float64)


@dataclass(frozen=True)
class PassContext:
    """Immutable per-benchmark context shared by hierarchy passes."""

    benchmark_name: str
    canvas_width: float
    canvas_height: float
    num_hard: int
    num_soft: int
    diagnostic_no_deadlines: bool = False


@dataclass
class PlateauTelemetry:
    """Pass-level candidate yield record for deterministic scheduling."""

    name: str
    proxy_before: float
    proxy_after: float
    elapsed_s: float
    candidates: int = 0
    legal: int = 0
    scored: int = 0
    accepts: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def proxy_gain(self) -> float:
        return max(0.0, float(self.proxy_before) - float(self.proxy_after))

    @property
    def accept_rate(self) -> float:
        denom = max(int(self.scored), int(self.legal), int(self.candidates))
        if denom <= 0:
            return 0.0
        return float(self.accepts) / float(denom)

    def plateaued(self, min_accept_rate: float, min_proxy_gain: float) -> bool:
        return self.accept_rate < float(min_accept_rate) and self.proxy_gain < float(min_proxy_gain)

    def to_trace_kwargs(self) -> dict[str, Any]:
        out = {
            "plateau_pass": self.name,
            "proxy_before": float(self.proxy_before),
            "proxy_after": float(self.proxy_after),
            "proxy_gain": self.proxy_gain,
            "elapsed_s": float(self.elapsed_s),
            "candidates": int(self.candidates),
            "legal": int(self.legal),
            "scored": int(self.scored),
            "accepts": int(self.accepts),
            "accept_rate": self.accept_rate,
        }
        out.update(self.extra)
        return out
