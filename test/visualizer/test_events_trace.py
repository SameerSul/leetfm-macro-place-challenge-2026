import json

import numpy as np
import pytest
import torch

from macro_place.benchmark import Benchmark
from placer.pipeline.macro_placer import MacroPlacer

from visualizer.events import EventEmitter, SCHEMA_VERSION, emit_event
from visualizer.trace import TraceReader, TraceWriter


def test_event_stage_inheritance_and_sequence():
    rows = []
    emitter = EventEmitter(rows.append)
    emitter.activate("soft", "Region soft relocation", round=2, lane="congestion")
    emit_event(
        emitter,
        "accepted_move",
        indices=[3],
        new_positions=[[4.0, 5.0]],
        metrics_stale=True,
    )
    emitter.finish()
    assert [row["sequence"] for row in rows] == list(range(len(rows)))
    move = rows[1]
    assert move["stage_id"] == "soft"
    assert move["algorithm"] == "Region soft relocation"
    assert move["round"] == 2
    assert move["lane"] == "congestion"


def test_trace_delta_round_trip_and_periodic_keyframe(tmp_path):
    path = tmp_path / "trace.jsonl"
    with TraceWriter(path, keyframe_every=2) as writer:
        writer.emit(
            {
                "schema": SCHEMA_VERSION,
                "type": "checkpoint",
                "positions": [[0.0, 0.0], [1.0, 1.0]],
            }
        )
        for x in (2.0, 3.0):
            writer.emit(
                {
                    "schema": SCHEMA_VERSION,
                    "type": "accepted_move",
                    "indices": [1],
                    "new_positions": [[x, x]],
                }
            )
    rows = list(TraceReader(path).events())
    assert rows[-1]["reason"] == "periodic_keyframe"
    frames = list(TraceReader(path).frames())
    np.testing.assert_allclose(frames[-1][1], [[0.0, 0.0], [3.0, 3.0]])


def test_trace_adds_full_stage_boundary_keyframe(tmp_path):
    path = tmp_path / "stage.jsonl"
    with TraceWriter(path) as writer:
        writer.emit({"schema": SCHEMA_VERSION, "type": "checkpoint", "positions": [[1.0, 2.0]]})
        writer.emit(
            {
                "schema": SCHEMA_VERSION,
                "type": "algorithm_start",
                "stage_id": "swap",
                "label": "Region swaps",
            }
        )
    rows = list(TraceReader(path).events())
    assert rows[-1]["reason"] == "stage_boundary"
    assert rows[-1]["positions"] == [[1.0, 2.0]]


def test_trace_unknown_schema_rejected(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"schema": 99, "type": "checkpoint"}) + "\n")
    with pytest.raises(ValueError, match="unsupported trace schema"):
        list(TraceReader(path).events())


def test_truncated_final_line_is_ignored(tmp_path):
    path = tmp_path / "partial.jsonl"
    path.write_text(
        json.dumps({"schema": SCHEMA_VERSION, "type": "checkpoint", "positions": [[1, 2]]})
        + "\n"
        + '{"schema":1,"type":"accepted_move"'
    )
    rows = list(TraceReader(path).events())
    assert len(rows) == 1


def test_placer_disabled_instrumentation_equivalence_and_completion():
    benchmark = Benchmark(
        name="tiny",
        canvas_width=20.0,
        canvas_height=20.0,
        num_macros=2,
        num_hard_macros=2,
        num_soft_macros=0,
        macro_positions=torch.tensor([[5.0, 5.0], [15.0, 15.0]]),
        macro_sizes=torch.tensor([[2.0, 2.0], [2.0, 2.0]]),
        macro_fixed=torch.tensor([True, False]),
        macro_names=["fixed", "movable"],
        num_nets=0,
        net_nodes=[],
        net_weights=torch.zeros(0),
        grid_rows=2,
        grid_cols=2,
    )
    expected = torch.tensor([[5.0, 5.0], [12.0, 13.0]])
    plain = MacroPlacer(event_sink=None)
    plain._place_impl = lambda _benchmark: expected.clone()
    rows = []
    observed = MacroPlacer(event_sink=rows.append)
    observed._place_impl = lambda _benchmark: expected.clone()
    plain_result = plain.place(benchmark)
    observed_result = observed.place(benchmark)
    torch.testing.assert_close(plain_result, observed_result, rtol=0, atol=0)
    assert observed_result[0].tolist() == [5.0, 5.0]
    assert rows[-1]["type"] == "completion"
    assert rows[-1]["positions"] == observed_result.tolist()
