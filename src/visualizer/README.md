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
  --benchmark-dir test/benchmarks/testcases/syn01_wide

# Explicit output and DREAMPlace cadence
uv run --extra visualizer python src/visualizer/main.py --benchmark ibm10 \
  --trace ml_data/visualizer/manual.jsonl --dreamplace-sample-every 20

# Record the production cached seed and subsequent accepted moves.
uv run --extra visualizer python src/visualizer/main.py --benchmark ibm10 \
  --use-dreamplace-cache --trace ml_data/visualizer/ibm10-production.jsonl

# Completed or partially-written trace file, or a directory to select its newest trace
uv run --extra visualizer python src/visualizer/main.py \
  --replay ml_data/visualizer/ibm10

# Headless-friendly H.264 MP4 export; 0.1 is 10× slower and 0.02 is 50× slower
QT_QPA_PLATFORM=offscreen uv run --extra visualizer python src/visualizer/main.py \
  --replay ml_data/visualizer/ibm10 --export-mp4 demo.mp4 --export-speed 0.1
```

Live traces default to
`ml_data/visualizer/<benchmark>/<UTC-run-id>.jsonl`. Normal `uv sync` remains
Qt-free; only `--extra visualizer` installs PyQtGraph 0.14, PySide6, ImageIO,
and its packaged FFmpeg encoder.
`--use-dreamplace-cache` produces evaluator-parity traces: it records the
cached production seed and later checkpoints/moves while omitting DREAMPlace
optimizer frames that did not execute.
Passing a directory to `--replay` selects its newest `.jsonl` trace by
modification time. A nonexistent path is rejected by argument validation with
a concise error instead of a Python traceback; `TRACE.jsonl` in examples means
an actual trace path, not a literal bundled file.
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
  DREAMPlace copies into one dashed centroid/spoke hyperedge; line thickness
  reflects `HIER_GROUP_WEIGHT`. Real nets and hierarchy-graph centroid edges
  are independent toggles. Real wiring uses macro pin offsets and fixed I/O
  endpoints where the input exposes them, and the slider controls the stable
  low-fanout/high-weight prefix (250 by default).
- **Play / Pause** starts or stops trace playback from the selected timeline
  position; starting at the end restarts the trace. **Live** jumps to the
  newest event. Arrow buttons and the timeline step or scrub recorded events.
  Pausing never stops placement or trace recording. Replay speed ranges from
  0.02× (50× slower) to 8×.
- **Export MP4…** saves a replay using the current window size, visible wiring
  layers, real-net limit, selection, and replay speed. Export is available only
  after opening a trace with `--replay`, so encoding cannot block a live
  placement's event queue. H.264 MP4 is used instead of GIF because long,
  line-heavy placement demos remain much smaller and retain full colour.

The command-line exporter accepts `--export-fps` (default 30) and
`--export-speed` values from 0.02× through 8×. It writes through a temporary
file and replaces the requested output only after encoding succeeds. Cancelling
an in-app export removes the partial temporary file. Slow exports encode each
recorded state once at a proportionally lower stream frame rate instead of
duplicating identical images; this preserves playback duration without
inflating the file. The CLI reports frame progress and an estimated time
remaining. Qt may print `This plugin does not support propagateSizeHints()` in
offscreen mode; that platform warning is harmless and encoding continues.

The sidebar shows the current algorithm, round, and lane; exact or stale
status; all five lower-is-better metrics; signed change from the exact initial
placement; and trends. After seed selection it also shows hard containment and
all six contract-component values beside their limits and signed headroom, so a
single composite cannot hide the component responsible for a rollback:

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
changed indices/centers and `metrics_stale=true`; the sidebar retains the most
recent exact values under its stale badge until the next exact checkpoint.
Checkpoints contain full positions at algorithm-stage boundaries, explicit
pipeline boundaries, and every 250 accepted moves. Replay ignores one
truncated trailing JSON line but rejects unknown schema versions.

## Runtime and cache behavior

This is a research diagnostic and can be substantially slower. Exact metric
snapshots, hierarchy scoring, queue serialization, rendering, and trace I/O are
not valid placement-runtime measurements. Visualizer runs bypass DREAMPlace's
final-position cache so genuine optimizer motion is observable; ordinary
evaluator runs retain the existing cache and `subprocess.run` path. Event
instrumentation is guarded by `event_sink is not None` and does not change the
normal evaluator execution path or optional dependencies.

The tracked bootstrap patch emits protocol-v1 compact base64 float32
lower-left coordinates at the requested optimizer cadence. The bridge converts
Bookshelf names, scales, and sizes back to VivaPlace output indices and center
coordinates. A malformed or unsupported progress record disables subsequent
DREAMPlace frames for that subprocess but does not invalidate an otherwise
successful placement. Bootstrap accepts the pinned upstream revision and both
known local revisions and applies the patch before building and after
installing. Verify both copies without changing them with:

```bash
uv run python scripts/dreamplace/apply_visualizer_patch.py --check
```

## First-version scope

The dashboard records and renders committed states, audit rollbacks, bulk
checkpoints, and raw DREAMPlace movement. Rejected candidate trials and live
congestion/density heatmap grids are intentionally outside this version. The
hierarchy value is the production six-component composite; it is not a new
acceptance objective.
