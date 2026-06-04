"""Opt-in training-data collection for learned local-search candidate ranking."""

import atexit
import gzip
import hashlib
import json
import math
import os
import socket
import threading
import time
import uuid
from pathlib import Path


def net_degree_features(incremental_scorer, module_idx: int, prefix: str = "") -> dict:
    """Cheap fixed-size connectivity features available before exact scoring."""
    degree = len(incremental_scorer.macro_to_nets.get(module_idx, ()))
    return {
        f"{prefix}net_degree": int(degree),
        f"{prefix}net_degree_log1p": float(math.log1p(degree)),
        f"{prefix}net_degree_norm": float(degree / max(incremental_scorer.n_nets, 1)),
    }


class CandidateTrace:
    """Buffered JSONL writer for candidate labels and search-policy events."""

    def __init__(self, path: str, flush_rows: int = 2048, run_id: str | None = None):
        self._path_template = path
        self.flush_rows = max(1, int(flush_rows))
        self.run_id = run_id or os.environ.get("ML_RUN_ID") or uuid.uuid4().hex
        self.run_started_ns = time.time_ns()
        self.host = socket.gethostname()
        self._owner_pid = os.getpid()
        self.path = self._resolve_path(child=False)
        self._rows = []
        self._group_seq = 0
        self._lock = threading.Lock()
        self._benchmark = {}
        self._benchmark_metadata = {}
        self._context = {}
        if hasattr(os, "register_at_fork"):
            os.register_at_fork(after_in_child=self._after_fork_child)

    def _resolve_path(self, child: bool) -> Path:
        resolved = self._path_template.replace("{run_id}", self.run_id).replace(
            "{pid}", str(os.getpid())
        )
        path = Path(resolved)
        if child and "{pid}" not in self._path_template:
            path = path.with_name(f"{path.stem}.pid-{os.getpid()}{path.suffix}")
        return path

    def _after_fork_child(self) -> None:
        """Discard inherited parent buffers and select a child-specific path."""
        self._owner_pid = os.getpid()
        self.path = self._resolve_path(child=True)
        self._rows = []
        self._group_seq = 0
        self._lock = threading.Lock()

    def next_group_id(self, operator: str) -> str:
        with self._lock:
            self._group_seq += 1
            return f"{self.run_id}:{os.getpid()}:{operator}:{self._group_seq}"

    def start_benchmark(
        self,
        *,
        benchmark,
        seed: int,
        config: dict,
        effective_budget_s: float,
        benchmark_index: int,
    ) -> None:
        config_json = json.dumps(config, sort_keys=True, separators=(",", ":"))
        self._benchmark_metadata = {
            "benchmark": benchmark.name,
            "benchmark_index": int(benchmark_index),
            "seed": int(seed),
            "effective_budget_s": float(effective_budget_s),
            "num_hard": int(benchmark.num_hard_macros),
            "num_soft": int(benchmark.num_soft_macros),
            "num_nets": int(benchmark.num_nets),
            "grid_rows": int(benchmark.grid_rows),
            "grid_cols": int(benchmark.grid_cols),
            "canvas_width": float(benchmark.canvas_width),
            "canvas_height": float(benchmark.canvas_height),
            "config": config,
            "config_hash": hashlib.sha256(config_json.encode("utf-8")).hexdigest()[:16],
        }
        self._benchmark = {
            key: value
            for key, value in self._benchmark_metadata.items()
            if key != "config"
        }
        self._context = {}
        self.event("benchmark_start", benchmark_metadata=self._benchmark_metadata)

    def set_context(self, **context) -> None:
        self._context = {key: value for key, value in context.items() if value is not None}

    def record(
        self,
        *,
        operator: str,
        field: str,
        group_id: str,
        state_score: float,
        trial_score: float,
        features: dict,
        candidate_rank: int | None = None,
        group_size: int | None = None,
        candidate_source: str | None = None,
    ) -> None:
        row = {
            **self._base_row("candidate"),
            "timestamp_ns": time.time_ns(),
            "operator": operator,
            "field": field,
            "group_id": group_id,
            "state_score": float(state_score),
            "trial_score": float(trial_score),
            "score_gain": float(state_score - trial_score),
            "improves": bool(trial_score < state_score),
            "candidate_rank": None if candidate_rank is None else int(candidate_rank),
            "group_size": None if group_size is None else int(group_size),
            "candidate_source": candidate_source,
            "features": features,
        }
        self._append(row)

    def event(self, event: str, **data) -> None:
        """Record non-training metadata such as prefilter/rejection counts."""
        self._append(
            {
                **self._base_row("event"),
                "timestamp_ns": time.time_ns(),
                "event": event,
                "data": data,
            }
        )

    def _base_row(self, row_type: str) -> dict:
        return {
            "schema_version": 2,
            "row_type": row_type,
            "run_id": self.run_id,
            "run_started_ns": self.run_started_ns,
            "host": self.host,
            "pid": os.getpid(),
            **self._benchmark,
            "context": dict(self._context),
        }

    def _append(self, row: dict) -> None:
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
        opener = gzip.open if self.path.suffix == ".gz" else open
        with opener(self.path, "at", encoding="utf-8") as handle:
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
            run_id = os.environ.get("ML_RUN_ID") or uuid.uuid4().hex
            _TRACE = CandidateTrace(path, flush_rows=flush_rows, run_id=run_id)
            atexit.register(_TRACE.flush)
    return _TRACE
