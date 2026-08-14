"""Dependency-free schema-v1 placement event contract."""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping

SCHEMA_VERSION = 1
EVENT_TYPES = frozenset(
    {
        "run_metadata",
        "algorithm_start",
        "algorithm_end",
        "accepted_move",
        "checkpoint",
        "seed_status",
        "dreamplace_progress",
        "rollback",
        "completion",
        "error",
    }
)


def _deliver(sink: Any, event: Mapping[str, Any]) -> None:
    if sink is None:
        return
    if hasattr(sink, "emit"):
        sink.emit(event)
    else:
        sink(event)


def emit_event(sink: Any, event_type: str, **payload: Any) -> None:
    """Emit one schema-v1 event without requiring the visualizer dependencies."""
    if sink is None:
        return
    if event_type not in EVENT_TYPES:
        raise ValueError(f"unknown visualizer event type: {event_type}")
    event = {
        "schema": SCHEMA_VERSION,
        "type": event_type,
        "timestamp_ns": time.time_ns(),
        **payload,
    }
    _deliver(sink, event)


@dataclass
class EventEmitter:
    """Adds sequence numbers and the active algorithm context to events."""

    sink: Any
    sequence: int = 0
    _stages: list[dict[str, Any]] = field(default_factory=list)

    def activate(
        self,
        stage_id: str,
        label: str,
        *,
        round: int | None = None,
        lane: str | None = None,
    ) -> None:
        """Switch the inherited stage for long pipeline regions."""
        if self._stages:
            previous = self._stages.pop()
            emit_event(self, "algorithm_end", **previous, succeeded=True)
        stage = {"stage_id": stage_id, "label": label, "round": round, "lane": lane}
        self._stages.append(stage)
        emit_event(self, "algorithm_start", **stage)

    def finish(self) -> None:
        if self._stages:
            stage = self._stages.pop()
            emit_event(self, "algorithm_end", **stage, succeeded=True)

    def emit(self, event: Mapping[str, Any]) -> None:
        row = dict(event)
        row.setdefault("schema", SCHEMA_VERSION)
        row.setdefault("timestamp_ns", time.time_ns())
        row["sequence"] = self.sequence
        self.sequence += 1
        if self._stages:
            stage = self._stages[-1]
            row.setdefault("stage_id", stage["stage_id"])
            row.setdefault("algorithm", stage["label"])
            if stage.get("round") is not None:
                row.setdefault("round", stage["round"])
            if stage.get("lane") is not None:
                row.setdefault("lane", stage["lane"])
        _deliver(self.sink, row)

    @contextlib.contextmanager
    def algorithm(
        self,
        stage_id: str,
        label: str,
        *,
        round: int | None = None,
        lane: str | None = None,
    ) -> Iterator[None]:
        stage = {"stage_id": stage_id, "label": label, "round": round, "lane": lane}
        self._stages.append(stage)
        emit_event(self, "algorithm_start", **stage)
        started = time.perf_counter()
        try:
            yield
        except BaseException as exc:
            emit_event(
                self,
                "algorithm_end",
                **stage,
                elapsed_s=time.perf_counter() - started,
                succeeded=False,
                error=type(exc).__name__,
            )
            raise
        else:
            emit_event(
                self,
                "algorithm_end",
                **stage,
                elapsed_s=time.perf_counter() - started,
                succeeded=True,
            )
        finally:
            self._stages.pop()


@contextlib.contextmanager
def algorithm_scope(
    sink: Any,
    stage_id: str,
    label: str,
    *,
    round: int | None = None,
    lane: str | None = None,
) -> Iterator[None]:
    """Open a stage on an EventEmitter, or emit a compatible flat scope."""
    if sink is None:
        yield
        return
    if hasattr(sink, "algorithm"):
        with sink.algorithm(stage_id, label, round=round, lane=lane):
            yield
        return
    fields = {"stage_id": stage_id, "label": label, "round": round, "lane": lane}
    emit_event(sink, "algorithm_start", **fields)
    started = time.perf_counter()
    try:
        yield
    finally:
        emit_event(sink, "algorithm_end", **fields, elapsed_s=time.perf_counter() - started)
