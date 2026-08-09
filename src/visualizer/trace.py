"""Delta-encoded schema-v1 JSONL traces and tolerant replay."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np

from .events import SCHEMA_VERSION


class TraceWriter:
    """Write all events with stage-boundary and periodic position keyframes."""

    def __init__(self, path: str | Path, keyframe_every: int = 250):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("w", encoding="utf-8")
        self.keyframe_every = max(1, int(keyframe_every))
        self._positions: np.ndarray | None = None
        self._moves = 0

    def emit(self, event: Mapping[str, Any]) -> None:
        row = dict(event)
        if int(row.get("schema", -1)) != SCHEMA_VERSION:
            raise ValueError(f"unsupported trace schema: {row.get('schema')!r}")
        if "positions" in row:
            self._positions = np.asarray(row["positions"], dtype=np.float64).reshape((-1, 2))
        elif row.get("type") in {"accepted_move", "dreamplace_progress"}:
            indices = np.asarray(row.get("indices", ()), dtype=np.int64)
            positions = np.asarray(row.get("new_positions", ()), dtype=np.float64).reshape((-1, 2))
            if self._positions is not None and indices.size == positions.shape[0]:
                self._positions[indices] = positions
            if row.get("type") == "accepted_move":
                self._moves += 1
        self._write(row)
        if row.get("type") == "algorithm_start" and self._positions is not None:
            self._write_keyframe(row, "stage_boundary")
        if (
            row.get("type") == "accepted_move"
            and self._positions is not None
            and self._moves % self.keyframe_every == 0
        ):
            self._write_keyframe(row, "periodic_keyframe")

    def _write_keyframe(self, source: Mapping[str, Any], reason: str) -> None:
        self._write(
            {
                "schema": SCHEMA_VERSION,
                "type": "checkpoint",
                "timestamp_ns": source.get("timestamp_ns"),
                "sequence": source.get("sequence"),
                "stage_id": source.get("stage_id"),
                "algorithm": source.get("algorithm") or source.get("label"),
                "reason": reason,
                "positions": self._positions.tolist(),
                "metrics": source.get("metrics"),
                "metrics_stale": source.get("metrics_stale", source.get("metrics") is None),
            }
        )

    def _write(self, row: Mapping[str, Any]) -> None:
        self._stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
        self._stream.flush()

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.close()

    def __enter__(self) -> "TraceWriter":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


class TraceReader:
    """Read complete or partially-written traces and reconstruct positions."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def events(self) -> Iterator[dict[str, Any]]:
        with self.path.open("r", encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    # Only a trailing partial record is recoverable.
                    if not line.endswith("\n"):
                        break
                    raise ValueError(f"invalid JSONL record at line {line_no}")
                if int(row.get("schema", -1)) != SCHEMA_VERSION:
                    raise ValueError(
                        f"unsupported trace schema at line {line_no}: {row.get('schema')}"
                    )
                yield row

    def frames(self) -> Iterator[tuple[dict[str, Any], np.ndarray | None]]:
        positions: np.ndarray | None = None
        for event in self.events():
            if "positions" in event:
                positions = np.asarray(event["positions"], dtype=np.float64).reshape((-1, 2))
            elif event.get("type") in {"accepted_move", "dreamplace_progress"}:
                indices = np.asarray(event.get("indices", ()), dtype=np.int64)
                changed = np.asarray(event.get("new_positions", ()), dtype=np.float64).reshape(
                    (-1, 2)
                )
                if positions is not None and indices.size == changed.shape[0]:
                    positions = positions.copy()
                    positions[indices] = changed
            yield event, None if positions is None else positions.copy()


class FanoutSink:
    """Forward every event to multiple sinks in stable order."""

    def __init__(self, *sinks: Any):
        self.sinks = tuple(sink for sink in sinks if sink is not None)

    def emit(self, event: Mapping[str, Any]) -> None:
        for sink in self.sinks:
            if hasattr(sink, "emit"):
                sink.emit(event)
            else:
                sink(event)
