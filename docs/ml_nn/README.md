# ML And BeyondPPA Notes

This directory now contains the active BeyondPPA/GNN planning notes for the
hierarchy placer.

## Current Active Work

Use `gnn/` for the dedicated hierarchy-GNN project plan:

- `gnn/README.md` is the project entry point.
- `gnn/SKILL.md` is the stage-gated agent workflow for GNN implementation.
- `gnn/requirements.md` lists the required work from G3 through production.
- `gnn/data-plan.md` defines trace generation, dataset splits, and audits.
- `gnn/evaluation.md` defines offline and closed-loop gates.
- `gnn/artifacts.md` defines model artifact requirements.
- `gnn/expansion-plan.md` defines the expanded hierarchy-flow assistant roles.

Use `beyondppa_results/` for the current implementation record:

- `beyondppa-structural-objectives.md` describes the deterministic structural
  objective and how it is integrated into hierarchy candidate ordering.
- `stage1_metrics.md` through `stage5_bounded_acceptance.md` record staged
  implementation decisions and verification.
- `gnn_trace_logging.md` documents the opt-in schema-v1 JSONL trace logger.
- `gnn_dataset_schema.md` documents the schema-v1 trace-to-graph dataset.
- `gnn_full_implementation_next_steps.md` is the implementation record from
  trace logging to the hierarchy-aware GNN subsystem.

## Current Status

No learned GNN model or DQN policy is active in production. The shipped
BeyondPPA-style pieces are deterministic structural metrics, default-off
hierarchy candidate ordering, default-off schema-v1 GNN trace logging, and the
schema-v1 trace-to-graph dataset builder. Trace logging is controlled by
`HIER_GNN_TRACE*` runtime environment variables. The next implementation stage
is G3 baseline non-GNN rankers.

The integration rule is mandatory: structural and learned signals may rank,
propose, select, budget, and diagnose work inside existing hierarchy operators,
but they must not bypass hard legality, fixed macro immobility, bounds,
hierarchy-region constraints, hierarchy-quality gates, or exact-proxy gates.
