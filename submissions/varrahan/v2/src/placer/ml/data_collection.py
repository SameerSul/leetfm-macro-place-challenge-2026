"""Opt-in training-data collection for learned local-search candidate ranking."""

import atexit
import json
import os
import threading
import time
from pathlib import Path


class CandidateTrace:
    """Buffered JSONL writer with near-zero cost when tracing is disabled."""

    def __init__(self, path: str, flush_rows: int = 2048):
        self.path = Path(path)
        self.flush_rows = max(1, int(flush_rows))
        self._rows = []
        self._group_seq = 0
        self._lock = threading.Lock()

    def next_group_id(self, operator: str) -> str:
        with self._lock:
            self._group_seq += 1
            return f"{os.getpid()}:{operator}:{self._group_seq}"

    def record(
        self,
        *,
        benchmark: str,
        operator: str,
        field: str,
        group_id: str,
        state_score: float,
        trial_score: float,
        features: dict,
    ) -> None:
        row = {
            "schema_version": 1,
            "timestamp_ns": time.time_ns(),
            "benchmark": benchmark,
            "operator": operator,
            "field": field,
            "group_id": group_id,
            "state_score": float(state_score),
            "trial_score": float(trial_score),
            "score_gain": float(state_score - trial_score),
            "improves": bool(trial_score < state_score),
            "features": features,
        }
        with self._lock:
            self._rows.append(row)
            if len(self._rows) >= self.flush_rows:
                self._flush_locked()

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        if not self._rows:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in self._rows)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(payload)
        self._rows.clear()


_TRACE = None
_TRACE_INITIALIZED = False


def get_candidate_trace():
    """Return the process-wide trace writer, or None unless ML_TRACE_PATH is set."""
    global _TRACE, _TRACE_INITIALIZED
    if not _TRACE_INITIALIZED:
        _TRACE_INITIALIZED = True
        path = os.environ.get("ML_TRACE_PATH")
        if path:
            flush_rows = int(os.environ.get("ML_TRACE_FLUSH_ROWS", "2048"))
            _TRACE = CandidateTrace(path, flush_rows=flush_rows)
            atexit.register(_TRACE.flush)
    return _TRACE
