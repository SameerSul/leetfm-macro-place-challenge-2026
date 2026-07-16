# Current Issues

Last revised: 2026-07-15.

This file tracks unresolved work in the hierarchy-only VivaPlace system. The
complete experiment history, including rejected proxy-path work, lives in
[`PROGRESS.md`](PROGRESS.md).

## Current State

`MacroPlacer.place()` requires grouped DREAMPlace and always runs the hierarchy
pipeline. The latest full IBM sweep is:

```text
uv run evaluate src/main.py --all
AVG 1.1199  17/17 VALID  0 overlaps  575.28s
```

All final hierarchy audits passed. The latest NG45 result is `AVG 0.7252`,
4/4 VALID, zero overlaps, all audits passed, in 232.41s.

No learned model is enabled in production. Structural candidate ordering and
GNN inference hooks remain default-off; exact proxy, hard legality, bounds,
fixed-macro immobility, hierarchy regions, and hierarchy-quality gates remain
authoritative.

## Open Work

### 1. Calibrate hierarchy-aware seed selection

Every seed now records hard-cluster compactness and worst spread, cluster
impurity, hierarchy-edge stretch, and owned/bridge soft distances. The
hierarchy-first selector is not ready for production: on `ibm10` it improved
the seed composite from `0.29168` to `0.16328` but regressed final proxy from
`1.1778` to `1.5281`.

Next step: treat the hierarchy vector as a feasibility or Pareto constraint
rather than a single weighted score. Any candidate policy needs multi-benchmark
validation and must preserve the final audit.

The exact-prescored seed portfolio now also contains a deterministic
constraint-graph legalization of `initial.plc`. This is a safe additive
baseline because the ordinary initial seed remains available and the graph
candidate advances only when its exact proxy is lower. It was selected on seven
benchmarks in the accepted sweep.

### 2. Use attributable telemetry for scheduling

Plateau telemetry schema v2 includes run id, code revision, and PID. Historical
data justified removing the broad survivor pool: 636 records across all 17 IBM
benchmarks produced no proxy gain and consumed 132.68 seconds. Some post-swap
passes also look low-yield in short runs, but historical traces contain real
gains, so there is not enough attributable full-suite evidence to remove them.

Next step: collect clean full-suite telemetry per revision and use
`scripts/analyze_plateau_telemetry.py` before changing another production
schedule.

### 3. Learned ranking has not cleared the production gate

The accepted Stage-G3 baseline and Stage-G4 macro-net ranker artifacts remain
available under `ml_data/beyondppa_gnn/models/`. Relocation-only inference is
implemented behind `HIER_GNN_RANK=1`, but the Stage-G6 full sweep regressed both
average proxy and runtime. Coldspot and regional-swap experiments also failed
to justify promotion.

Next step: favor additive, budgeted candidates that preserve the deterministic
prefix. Promote nothing without repeatable held-out ranking quality, closed-loop
proxy gain, unchanged legality/audit behavior, and acceptable runtime.

### 4. Exact scoring remains the runtime bottleneck

Large grids make exact validation expensive, and CPU contention can multiply
score time. The placement flow must keep a running maximum score estimate and
reserve enough time for the final score and audits. New operators should first
prove that their expected gain pays for their exact-score calls.

Hard-hard and hard-soft swap trials now share exact compiled batch kernels,
joining the existing batched soft relocation and soft-soft swap paths. Direct
scalar parity checks pass to floating-point roundoff without changing committed
scorer grids or caches. The remaining bottleneck is repeated full exact scoring,
especially the evaluator's final large-grid report and operators that cannot
share a fixed endpoint.

### 5. Portability coverage is still narrower than challenge coverage

The pinned DREAMPlace bootstrap and native-op preflight make the supported
CUDA 12.1/GCC 11/Python build reproducible. Other CUDA architectures and Python
ABIs still need an explicit rebuild. The EDA I/O path supports converted
LEF/DEF/Verilog inputs by attaching their generated source directory, but broad
real-design parser coverage remains a validation task rather than a claimed
guarantee.

## Maintenance Rules

- Keep the production path hierarchy-only.
- Do not restore deleted proxy-only operators or archived research scripts.
- Keep GNN/structural signals inside existing hierarchy operators and gates.
- Record accepted full-suite numbers in `PROGRESS.md`.
- Keep `ARCHITECTURE.md` and `DESIGN_FLOW.md` synchronized with active code.
