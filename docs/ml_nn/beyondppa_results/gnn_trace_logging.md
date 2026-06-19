# GNN Trace Logging

Implemented as opt-in logging for future hierarchy-aware GNN training.

## Controls

```bash
V2_HIER_GNN_TRACE=1
V2_HIER_GNN_TRACE_DIR=ml_data/beyondppa_gnn
V2_HIER_GNN_TRACE_RUN=my_run
V2_HIER_GNN_TRACE_MAX_CANDIDATES=512
```

Optional direct path:

```bash
V2_HIER_GNN_TRACE_PATH=/tmp/hier_gnn_trace.jsonl
```

## Events

- `hier_relocation_candidates`
  - hard propose-all candidate pool after proxy/structural candidate scoring
  - includes macro id, candidate rank, target location, field values, structural
    delta, and candidate score
- `hier_relocation_result`
  - accepted hard/soft relocation moves with proxy deltas
- `hier_pass_result`
  - pass-level summaries for micro-shift, decompression, swaps, post-swap
    relocation, coldspot tightening, and post-coldspot micro-shift
- `hier_final`
  - final proxy, pre-relief proxy, hierarchy quality, cluster count, and group
    weight

## Design Decision

Logging is attached to the existing hierarchy flow. It does not add a GNN model,
does not change candidate ranking by itself, and does not create a separate
placement path. The trace is intended to collect data for future learned
rankers or graph models.

For the full implementation roadmap, see
`docs/ml_nn/beyondppa_results/gnn_full_implementation_next_steps.md`.

## Verification

Smoke command:

```bash
V2_HIER_GNN_TRACE=1 \
V2_HIER_GNN_TRACE_PATH=/tmp/hier_gnn_trace_smoke.jsonl \
V2_HIER_GNN_TRACE_MAX_CANDIDATES=5 \
uv run evaluate src/main.py -b ibm01
```

Result:

```text
proxy=0.9435  (wl=0.082 den=0.640 cong=1.083)  VALID  [35.80s]
```

Trace file summary:

```text
24 events
hier_relocation_result: 13
hier_pass_result: 9
hier_relocation_candidates: 1
hier_final: 1
```
