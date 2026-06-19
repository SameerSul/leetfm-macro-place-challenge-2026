"""Opt-in JSONL traces for future hierarchy-aware GNN training."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

TRACE_SCHEMA_VERSION = 1

_FALSE = {"0", "false", "False", "no", "NO", "off", ""}


def gnn_trace_enabled() -> bool:
    return os.environ.get("HIER_GNN_TRACE", "0").strip() not in _FALSE


def gnn_trace_limit(default: int = 512) -> int:
    return max(0, int(os.environ.get("HIER_GNN_TRACE_MAX_CANDIDATES", str(default))))


def _trace_path() -> Path:
    raw = os.environ.get("HIER_GNN_TRACE_PATH", "").strip()
    if raw:
        return Path(raw)
    root = Path(os.environ.get("HIER_GNN_TRACE_DIR", "ml_data/beyondppa_gnn"))
    run_id = os.environ.get("HIER_GNN_TRACE_RUN", "").strip()
    name = f"{run_id}.jsonl" if run_id else "trace.jsonl"
    return root / name


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def log_gnn_event(event: str, **payload: Any) -> None:
    """Append one trace event when `HIER_GNN_TRACE=1`."""
    if not gnn_trace_enabled():
        return
    path = _trace_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "time_s": time.time(),
        "event": event,
        **payload,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_jsonable(row), sort_keys=True, separators=(",", ":")) + "\n")
