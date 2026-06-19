# GNN Trace Logging

Implemented as opt-in logging for future hierarchy-aware GNN training.

## Environment Variables

These settings are intentionally runtime environment variables, not constants in
`src/utils/constants.py`. They can be exported directly in the shell or provided
through a `.env` workflow that exports them before running the placer.

```bash
HIER_GNN_TRACE=1
HIER_GNN_TRACE_DIR=ml_data/beyondppa_gnn
HIER_GNN_TRACE_RUN=my_run
HIER_GNN_TRACE_MAX_CANDIDATES=512
```

Optional direct path:

```bash
HIER_GNN_TRACE_PATH=/tmp/hier_gnn_trace.jsonl
```

## Events

All events include `schema_version`. The current schema is documented in
[`gnn_trace_schema.md`](gnn_trace_schema.md).

- `hier_relocation_candidates`
  - hard propose-all candidate pool after proxy/structural candidate scoring
  - includes macro id, candidate rank, target location, field values, structural
    delta, and candidate score
- `hier_relocation_result`
  - accepted hard/soft relocation moves with proxy deltas
- `hier_decompression_candidate`
  - cluster decompression candidate labels with expansion factor, axis scale,
    hierarchy-quality delta, exact proxy delta when scored, accepted flag, and
    rejection reason
- `hier_swap_candidates`
  - sampled hard/hard, hard/soft, and soft/soft region-swap candidate pools with
    legality, region, score, proxy delta, accepted flag, and rejection reason
- `hier_coldspot_candidate`
  - coldspot tightening candidates and skipped rounds with selected cluster,
    field gap, hierarchy-quality delta, exact proxy delta when scored, accepted
    flag, and rejection reason
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
HIER_GNN_TRACE=1 \
HIER_GNN_TRACE_PATH=/tmp/hier_gnn_trace_smoke.jsonl \
HIER_GNN_TRACE_MAX_CANDIDATES=5 \
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
