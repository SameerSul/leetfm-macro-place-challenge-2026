# Hierarchy GNN Project

This directory is the working project plan for the hierarchy-aware GNN subsystem.
The first production-safe role is candidate ranking, but the intended subsystem
can grow into a hierarchy-flow assistant that ranks, proposes, selects, budgets,
and diagnoses work inside the existing placer. It must not create a separate
placement path or bypass legality, fixed-macro, bounds, hierarchy-region,
hierarchy-quality, or exact-proxy gates.

## Current Status

Implemented:

- Stage G1 trace completeness for schema v1.
- Stage G2 JSONL-to-graph dataset builder.
- Dataset and trace schema documentation.
- Stage G3 offline baseline entrypoint:
  `scripts/gnn/train_gnn_baseline.py`.
- Stage G3 offline label-learnability gate accepted on the minimum
  4-benchmark split.
- Stage G4 dataset schema v2 macro-net graph extension.
- Stage G4 offline macro-net ranker accepted on the minimum 4-benchmark split.
- Stage G5 default-off relocation-only inference hook smoke-accepted on
  `ibm10`.
- Stage G6 closed-loop validation passed legality but was not promoted because
  full-suite AVG and runtime regressed.

Not implemented:

- Model artifact/version discipline.
- Closed-loop benchmark acceptance package.
- Promotion-quality closed-loop improvement.

In progress:

- GNN ranking diagnostics before any broader integration or promotion.

## Documentation

- [SKILL.md](SKILL.md): stage-gated agent workflow for production GNN
  implementation.
- [requirements.md](requirements.md): required next work from G3 through
  production promotion.
- [data-plan.md](data-plan.md): trace generation, split policy, and dataset
  audits.
- [evaluation.md](evaluation.md): offline and closed-loop metrics/gates.
- [artifacts.md](artifacts.md): required model and experiment files.
- [expansion-plan.md](expansion-plan.md): expanded GNN roles inside the
  hierarchy implementation after the first ranker is validated.
- [macrodiff_plus_notes.md](macrodiff_plus_notes.md): design ideas to borrow
  from MacroDiff+ without turning this subsystem into a diffusion placer.

## Reference Paper

Agents working on the G4 model should review
[Physics-Guided Geometric Diffusion for Macro Placement Generation](https://arxiv.org/pdf/2605.16451).
Use it for heterogeneous macro-net graph design, pin-offset edge features,
dynamic net HPWL features, and the topology-plus-geometry split. Do not use it
as a mandate to replace the hierarchy ranker with a full diffusion placer.

Related implementation docs:

- [`../beyondppa_results/gnn_trace_schema.md`](../beyondppa_results/gnn_trace_schema.md)
- [`../beyondppa_results/gnn_dataset_schema.md`](../beyondppa_results/gnn_dataset_schema.md)
- [`../beyondppa_results/gnn_full_implementation_next_steps.md`](../beyondppa_results/gnn_full_implementation_next_steps.md)

## Immediate Next Step

Diagnose why the default-off relocation hook regressed closed-loop average proxy
despite offline recall wins. Do not expand operators or promote anything
default-on until the regression is understood and fixed. Keep the expanded roles
in [expansion-plan.md](expansion-plan.md) as follow-on targets.

Current diagnostic focus:

- Compare ranking by accepted labels against ranking by exact proxy-gain labels.
- Measure top-k overlap between the G4 ranker and the deterministic trace order.
- Split failures by benchmark and operator before retraining.
- Collect closed-loop traces from `HIER_GNN_RANK=1` runs so the next model sees
  the states it creates, not only heuristic-generated states.
