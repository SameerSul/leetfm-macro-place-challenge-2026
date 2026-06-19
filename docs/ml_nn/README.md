# ML And BeyondPPA Notes

This directory now contains the active BeyondPPA/GNN planning notes for the
hierarchy placer.

## Current Active Work

Use `beyondppa_results/` for the current implementation record:

- `beyondppa-structural-objectives.md` describes the deterministic structural
  objective and how it is integrated into hierarchy candidate ordering.
- `stage1_metrics.md` through `stage5_bounded_acceptance.md` record staged
  implementation decisions and verification.
- `gnn_trace_logging.md` documents the opt-in JSONL trace logger.
- `gnn_full_implementation_next_steps.md` is the roadmap from trace logging to
  a real hierarchy-aware GNN ranker.

## Current Status

No learned GNN model or DQN policy is active in production. The shipped
BeyondPPA-style pieces are deterministic structural metrics, default-off
hierarchy candidate ordering, and default-off GNN trace logging. Trace logging
is controlled by `HIER_GNN_TRACE*` runtime environment variables.

The integration rule is mandatory: structural and learned signals may rank
candidates inside existing hierarchy operators, but they must not bypass hard
legality, fixed macro immobility, bounds, hierarchy-region constraints,
hierarchy-quality gates, or exact-proxy gates.
