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
AVG 1.1205  17/17 VALID  0 overlaps  541.67s
```

All final hierarchy audits passed. The latest NG45 result is `AVG 0.7252`,
4/4 VALID, zero overlaps, all audits passed, in 232.41s.

No learned model is enabled in production. Structural candidate ordering and
GNN inference hooks remain default-off; exact proxy, hard legality, bounds,
fixed-macro immobility, hierarchy regions, and hierarchy-quality gates remain
authoritative.

## Open Work

### 1. Calibrate the production hierarchy contract

Every seed now records hard-cluster compactness and worst spread, cluster
impurity, hierarchy-edge stretch, and owned/bridge soft distances. Production
uses these as independent feasibility limits relative to legalized
`initial.plc`, selects the lowest-proxy eligible seed, and reapplies the same
contract relative to the selected seed at relief checkpoints and final
rollback. The first full sweep passed all component audits at `AVG 1.1205`.

Next step: calibrate the component-specific absolute/relative limits on more
commercial and synthetic designs. The scalar hierarchy-first selector remains
default-off because its focused proxy regression was too large.

The exact-prescored seed portfolio now also contains a deterministic
constraint-graph legalization of `initial.plc`. The ordinary initial seed
remains available and the graph candidate advances only when it passes every
component limit and its exact proxy is lower.

### 2. Use attributable telemetry for scheduling

Plateau telemetry schema v2 includes run id, code revision, and PID. Historical
data justified removing the broad survivor pool: 636 records across all 17 IBM
benchmarks produced no proxy gain and consumed 132.68 seconds. Some post-swap
passes also look low-yield in short runs, but historical traces contain real
gains, so there is not enough attributable full-suite evidence to remove them.

The ordinary post-swap soft pass is now skipped after two attributable full
suites produced zero gain in 34 runs. Its budget is reassigned to the
remaining deadline/final-audit reserve, with the skip recorded in schedule
telemetry. The direct plateau-breadth reinvestment was legal but regressed the
full suite and was rejected. Continue using
`scripts/analyze_plateau_telemetry.py` before changing another production
schedule.

The accepted skip-only sweep reproduced every prior benchmark proxy at
`AVG 1.1205` and reduced runtime from 544.94s to 541.67s. A direct
plateau-escape breadth reinvestment (`512` hot softs, `12` targets, `6.5s`)
was legal but regressed to `AVG 1.1213` in 546.13s and is not production.

The first compound related-soft sweep exact-scored 600 complete group states
in 9.20 seconds. One ibm11 move committed for a 0.000213 gain and survived the
final component audit; six candidates were rejected by that audit before exact
scoring. Ordinary post-swap soft relocation produced zero gain in the same run,
matching the preceding component-contract sweep and providing the clean
two-revision evidence needed for the next schedule change.

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
