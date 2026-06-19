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

Not implemented:

- Baseline non-GNN ranker.
- Learned GNN model.
- Train/eval scripts.
- Inference-time ranker integration.
- Model artifact/version discipline.
- Closed-loop benchmark acceptance package.

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

Start Stage G3. Train and evaluate baseline non-GNN rankers on the Stage-G2
dataset before implementing a graph model. Keep the expanded roles in
[expansion-plan.md](expansion-plan.md) as follow-on targets once the first
ranker is validated.
