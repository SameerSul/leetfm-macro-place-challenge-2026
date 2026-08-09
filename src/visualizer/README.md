# VivaPlace live visualizer

This optional PyQtGraph/PySide6 dashboard shows the hierarchy placer while it
runs and replays its delta-encoded traces. It is separate from evaluator
`--vis`, which still creates only the evaluator's static final image.

## Launch

```bash
# IBM or NG45 design
uv run --extra visualizer python src/visualizer/main.py --benchmark ibm10
uv run --extra visualizer python src/visualizer/main.py --benchmark nvdla

# Any generated/source directory containing netlist.pb.txt; initial.plc is optional
uv run --extra visualizer python src/visualizer/main.py \
  --benchmark-dir test/benchmarks/generated/syn01

# Explicit output and DREAMPlace cadence
uv run --extra visualizer python src/visualizer/main.py --benchmark ibm10 \
  --trace ml_data/visualizer/manual.jsonl --dreamplace-sample-every 20

# Completed or partially-written trace
uv run --extra visualizer python src/visualizer/main.py --replay TRACE.jsonl
```

Live traces default to
`ml_data/visualizer/<benchmark>/<UTC-run-id>.jsonl`. Normal `uv sync` remains
Qt-free; only `--extra visualizer` installs PyQtGraph 0.14 and PySide6.
The selected upstream documentation is recorded in
[`docs/REFERENCES.md`](../../docs/REFERENCES.md#direct-python-and-build-dependencies).

## Display and controls

- Drag to pan, use the wheel to zoom, and use **Reset view** for the full die.
- Search by a substring of a macro name. Hover shows name, index, center, and
  size; clicking selects a macro and adds all its incident real nets.
- Hard macros use opaque leaf-hierarchy colours and solid borders. Owned soft
  macros use the related translucent colour and dashed borders. Bridge or
  unclustered softs are neutral. Fixed macros have a red double border.
- A moved macro glows temporarily and its old-to-new vector is drawn. Parent
  groups use translucent related outlines.
- Synthetic grouping nets are enabled by default and collapse the repeated
  DREAMPlace copies into one dashed centroid/spoke hyperedge. Real nets and
  hierarchy-graph centroid edges are independent toggles; the real-net slider
  controls the stable low-fanout/high-weight prefix (250 by default).
- **Pause** stops rendering only; placement and trace recording continue.
  **Live** jumps to the newest event. Arrow buttons and the timeline step or
  scrub recorded events. Replay speed ranges from 0.25× to 8×.

The sidebar shows the current algorithm/stage, exact or stale status, all five
lower-is-better metrics, signed change from the first exact state, and trends:

```text
proxy = wirelength + 0.5 * density + 0.5 * congestion
hierarchy = hierarchy_quality_vector(...)["composite"]
```

## Schema-v1 events

Records are newline-delimited JSON dictionaries with `schema`, `type`,
`timestamp_ns`, and (after the emitter is installed) `sequence`. Event types
are `run_metadata`, `algorithm_start`, `algorithm_end`, `accepted_move`,
`checkpoint`, `dreamplace_progress`, `rollback`, `completion`, and `error`.
Accepted moves contain output-tensor `indices`, `old_positions`,
`new_positions`, `move_kind`, and exact metrics. DREAMPlace frames contain
changed indices/centers and `metrics_stale=true`. Checkpoints contain full
positions at pipeline boundaries and every 250 accepted moves. Replay ignores
one truncated trailing JSON line but rejects unknown schema versions.

## Runtime and cache behavior

This is a research diagnostic and can be substantially slower. Exact metric
snapshots, hierarchy scoring, queue serialization, rendering, and trace I/O are
not valid placement-runtime measurements. Visualizer runs bypass DREAMPlace's
final-position cache so genuine optimizer motion is observable; ordinary
evaluator runs retain the existing cache and `subprocess.run` path. Event
instrumentation is guarded by `event_sink is not None` and does not change the
normal evaluator execution path or optional dependencies.
