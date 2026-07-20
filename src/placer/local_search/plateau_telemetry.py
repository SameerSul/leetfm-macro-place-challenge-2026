"""Buffered pass-level telemetry for deterministic schedule decisions."""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

PLATEAU_SCHEMA_VERSION = 2

_PLATEAU_BUFFER: list[str] = []
_PLATEAU_BUFFER_PATH: Path | None = None
_PROCESS_RUN_ID = os.environ.get("VIVAPLACE_RUN_ID", "").strip() or (
    f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}-pid{os.getpid()}"
)
_CODE_REVISION: str | None = None
_WORKTREE_PROVENANCE: dict[str, Any] | None = None


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


def _worktree_provenance() -> dict[str, Any]:
    """Fingerprint relevant tracked changes and untracked source files."""
    global _WORKTREE_PROVENANCE
    if _WORKTREE_PROVENANCE is not None:
        return _WORKTREE_PROVENANCE
    override = os.environ.get("VIVAPLACE_WORKTREE_FINGERPRINT", "").strip()
    if override:
        _WORKTREE_PROVENANCE = {
            "worktree_dirty": override != "clean",
            "worktree_fingerprint": override,
        }
        return _WORKTREE_PROVENANCE

    root = Path(__file__).resolve().parents[3]
    scope = (
        "src",
        "test",
        "scripts",
        "docs",
        "pyproject.toml",
        "requirements.txt",
        "uv.lock",
    )
    try:
        diff = subprocess.run(
            ["git", "diff", "--no-ext-diff", "--binary", "HEAD", "--", *scope],
            cwd=root,
            capture_output=True,
            timeout=5.0,
            check=True,
        ).stdout
        untracked_raw = subprocess.run(
            [
                "git",
                "ls-files",
                "--others",
                "--exclude-standard",
                "--",
                "src",
                "test",
                "scripts",
                "docs",
            ],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5.0,
            check=True,
        ).stdout
        untracked = sorted(path for path in untracked_raw.splitlines() if path)
        digest = hashlib.sha256()
        digest.update(diff)
        for relative in untracked:
            digest.update(relative.encode("utf-8", errors="surrogateescape"))
            path = root / relative
            if path.is_file():
                digest.update(path.read_bytes())
        dirty = bool(diff or untracked)
        _WORKTREE_PROVENANCE = {
            "worktree_dirty": dirty,
            "worktree_fingerprint": digest.hexdigest()[:16] if dirty else "clean",
        }
    except (OSError, subprocess.SubprocessError):
        _WORKTREE_PROVENANCE = {
            "worktree_dirty": None,
            "worktree_fingerprint": "unknown",
        }
    return _WORKTREE_PROVENANCE


def _provenance() -> dict[str, Any]:
    specific = os.environ.get("HIER_PLATEAU_TRACE_RUN", "").strip()
    return {
        "run_id": specific or _PROCESS_RUN_ID,
        "code_revision": _code_revision(),
        **_worktree_provenance(),
        "pid": os.getpid(),
    }


def _plateau_trace_path() -> Path:
    raw = os.environ.get("HIER_PLATEAU_TRACE_PATH", "").strip()
    if raw:
        return Path(raw)
    root = Path(os.environ.get("HIER_PLATEAU_TRACE_DIR", "ml_data/plateau_telemetry"))
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


def log_plateau_event(event: str, **payload: Any) -> None:
    """Buffer one lightweight plateau telemetry row."""
    global _PLATEAU_BUFFER_PATH
    path = _plateau_trace_path()
    row = {
        "schema_version": PLATEAU_SCHEMA_VERSION,
        "time_s": time.time(),
        "event": event,
        **_provenance(),
        **payload,
    }
    line = json.dumps(_jsonable(row), sort_keys=True, separators=(",", ":")) + "\n"
    if _PLATEAU_BUFFER_PATH is not None and _PLATEAU_BUFFER_PATH != path:
        flush_plateau_events()
    _PLATEAU_BUFFER_PATH = path
    _PLATEAU_BUFFER.append(line)
