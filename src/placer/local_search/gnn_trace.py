"""Opt-in JSONL traces for future hierarchy-aware GNN training."""

from __future__ import annotations

import json
import os
import subprocess
import time
import atexit
from pathlib import Path
from typing import Any

import numpy as np

TRACE_SCHEMA_VERSION = 1
PLATEAU_SCHEMA_VERSION = 2

_FALSE = {"0", "false", "False", "no", "NO", "off", ""}
_PLATEAU_BUFFER: list[str] = []
_PLATEAU_BUFFER_PATH: Path | None = None
_PROCESS_RUN_ID = os.environ.get("VIVAPLACE_RUN_ID", "").strip() or (
    f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}-pid{os.getpid()}"
)
_CODE_REVISION: str | None = None


def _code_revision() -> str:
    global _CODE_REVISION
    if _CODE_REVISION is not None:
        return _CODE_REVISION
    override = os.environ.get("VIVAPLACE_CODE_REVISION", "").strip()
    if override:
        _CODE_REVISION = override
        return _CODE_REVISION
    try:
        root = Path(__file__).resolve().parents[3]
        _CODE_REVISION = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=2.0,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        _CODE_REVISION = "unknown"
    return _CODE_REVISION


def _provenance(trace_kind: str) -> dict[str, Any]:
    specific = os.environ.get(f"HIER_{trace_kind}_TRACE_RUN", "").strip()
    return {
        "run_id": specific or _PROCESS_RUN_ID,
        "code_revision": _code_revision(),
        "pid": os.getpid(),
    }


def gnn_trace_enabled() -> bool:
    return os.environ.get("HIER_GNN_TRACE", "0").strip() not in _FALSE


def plateau_trace_enabled() -> bool:
    return os.environ.get("HIER_PLATEAU_TRACE", "1").strip() not in _FALSE


def plateau_trace_buffered() -> bool:
    raw = os.environ.get("HIER_PLATEAU_TRACE_BUFFERED", "1").strip()
    return raw not in _FALSE


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


def _plateau_trace_path() -> Path:
    raw = os.environ.get("HIER_PLATEAU_TRACE_PATH", "").strip()
    if raw:
        return Path(raw)
    root = Path(os.environ.get("HIER_PLATEAU_TRACE_DIR", "ml_data/beyondppa_gnn/plateau"))
    run_id = os.environ.get("HIER_PLATEAU_TRACE_RUN", "").strip()
    name = f"{run_id}.jsonl" if run_id else "plateau_telemetry.jsonl"
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


def flush_plateau_events() -> None:
    """Flush buffered plateau rows to JSONL."""
    if not _PLATEAU_BUFFER:
        return
    path = _PLATEAU_BUFFER_PATH or _plateau_trace_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(_PLATEAU_BUFFER)
    _PLATEAU_BUFFER.clear()
    with path.open("a", encoding="utf-8") as f:
        f.write("".join(rows))


atexit.register(flush_plateau_events)


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
        **_provenance("GNN"),
        **payload,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_jsonable(row), sort_keys=True, separators=(",", ":")) + "\n")


def log_plateau_event(event: str, **payload: Any) -> None:
    """Append one lightweight plateau telemetry row unless disabled."""
    global _PLATEAU_BUFFER_PATH
    if not plateau_trace_enabled():
        return
    path = _plateau_trace_path()
    row = {
        "schema_version": PLATEAU_SCHEMA_VERSION,
        "time_s": time.time(),
        "event": event,
        **_provenance("PLATEAU"),
        **payload,
    }
    line = json.dumps(_jsonable(row), sort_keys=True, separators=(",", ":")) + "\n"
    if plateau_trace_buffered():
        if _PLATEAU_BUFFER_PATH is not None and _PLATEAU_BUFFER_PATH != path:
            flush_plateau_events()
        _PLATEAU_BUFFER_PATH = path
        _PLATEAU_BUFFER.append(line)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
