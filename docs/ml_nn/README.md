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
`HIER_GNN_TRACE*` runtime environment variables. Stage G3 has an accepted
default-off offline baseline artifact at
`ml_data/beyondppa_gnn/models/20260619_g3_candidate_baseline_min4/`. Stage G4
has an accepted default-off offline macro-net ranker artifact at
`ml_data/beyondppa_gnn/models/20260619_g4_macro_net_ranker_v1/`. Stage G5 has a
smoke-accepted default-off relocation-only inference hook. Stage G6 passed
legality but was not promoted because full-suite average proxy and runtime
regressed versus the accepted hierarchy baseline.

Post-G6 diagnostics found that smaller GNN top-k prefixes and the experimental
`HIER_GNN_PRESERVE_TOP_N` guarded-prefix mode were valid but worse on `ibm12`.
Keep learned ranking default-off until repeatable closed-loop diagnostics show a
real improvement.

`HIER_DIAGNOSTIC_NO_DEADLINES=1` is now available for repeatable GNN diagnostics
only. It made the `ibm12` GNN comparison repeatable and positive, but it is not
a production mode.

Default-off `HIER_GNN_EXTRA_TOP_K` supports additive GNN diagnostics that
preserve deterministic candidates and append a small learned tail. It is ready
for timed smoke, not promotion.

Default-off `HIER_GNN_COLDSPOT_SELECT` supports additive coldspot diagnostics.
It keeps heuristic hot-cluster/cold-window selection, generates multiple kicked
outcomes plus a no-op, ranks those outcomes with `HIER_GNN_COLDSPOT_MODEL`, and
then sends only the selected top slice through the existing hierarchy-quality
and exact-proxy gates. It is not a production mode.

Coldspot is not the primary learned-control target. The active GNN target is
regional relocation and regional hard-hard, hard-soft, and soft-soft swaps.
Default-off `HIER_GNN_OPERATORS=region_swaps` can rank regional swap candidates,
and `HIER_SOFT_BARRIER_GAIN=0.01` can be used in diagnostics as a soft-macro
barrier for soft relocation and soft-involving swaps. Closed-loop `ibm12` smoke
showed that full sequential swap reordering is valid but worse, so future work
should preserve deterministic swap order and add only budgeted GNN-ranked
supplemental candidates.

The integration rule is mandatory: structural and learned signals may rank,
propose, select, budget, and diagnose work inside existing hierarchy operators,
but they must not bypass hard legality, fixed macro immobility, bounds,
hierarchy-region constraints, hierarchy-quality gates, or exact-proxy gates.
