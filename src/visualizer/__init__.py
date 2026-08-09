"""Opt-in live visualization and trace playback for VivaPlace."""

from .events import SCHEMA_VERSION, EventEmitter, algorithm_scope, emit_event
from .trace import TraceReader, TraceWriter

__all__ = [
    "SCHEMA_VERSION",
    "EventEmitter",
    "TraceReader",
    "TraceWriter",
    "algorithm_scope",
    "emit_event",
]
