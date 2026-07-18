# VivaPlace System Detriment Review

Last reviewed: 2026-07-17.

## Purpose

This document classifies components that currently waste work, obscure system
quality, contribute little, or have already been rejected. It separates a
runtime bottleneck from a detrimental component: an expensive pass that
materially improves proxy should be optimized, not removed.

The corresponding improvement sequence is in
[SYS_IMPROVEMENT.md](SYS_IMPROVEMENT.md). See
[ARCHITECTURE.md](ARCHITECTURE.md) for current behavior and
[PROGRESS.md](PROGRESS.md) for the full experimental record.

Status meanings:

- **Optimize**: productive but unnecessarily expensive or unstable.
- **Gate**: low or conditional yield; retain until rollback-aware evidence is
  sufficient for removal.
- **Remove**: confirmed current cost with no retained value.
- **Retain**: evidence supports the current component.
- **Already retired**: no production runtime cost; do not restore without a
  materially new hypothesis.

## Confirmed Active Detriments

### 1. Pass telemetry is recorded before audit rollback

**Status: Resolve.**

The historical issue has been addressed: telemetry now preserves proposal and
checkpoint-retained metrics in the same record, and rollback context is written
with the row.

In a focused `ibm10` review run, the two hard-relocation rounds reported a
combined `0.023976` gain, but the console showed an audit restore after both
rounds. This makes gain-per-second scheduling unsafe and can hide components
that repeatedly perform discarded work.

Impact: legacy telemetry no longer mixes retained and discarded work. This
now enables accurate pass ranking and explicit rollback reconstruction-cost
accounting.

Action:

- Pass records now retain proposal and checkpoint-retained proxy/accept
  fields.
- Rollback reason, violating vectors, and rebuild timing are emitted as
  telemetry metadata.
- `scripts/analyze_plateau_telemetry.py` now ranks by retained metrics
  (`retained_gain / scored` and `retained_gain / sec`) for better scheduling
  decisions.

### 2. Wall-clock budgets change placement behavior

**Status: Resolve.**

`wall-clock-only` gating could push the same deterministic proposal path to
different candidate counts under contention. This behavior was visible as
deterministic-order drift (`ibm18=1.3845` full-suite, `ibm18=1.3806`
isolated, `ibm10=1.0651` review-time versus `1.0641` accepted).

Impact: exact-gain trajectory changed without any algorithmic intent, so quality
attribution was contaminated by environment noise and late-pass ordering shifted
between identical operator sets.

Action: apply wall-lock behavior unconditionally so additive and late-pass
eligibility prioritize deterministic quotas by default. Deadlines remain as hard
safety caps for overrun protection only.

### 3. Audit-discarded local-search work

**Status: Gate/optimize, not remove.**

- Both hard-relocation rounds in the focused `ibm10` review crossed the legacy
  hard hierarchy limit and were restored.
- Eight of ten latest small-design passes set `audit_rollback=true`.
- `ibm11` small-design polish spent `1.96s`, recorded 45 accepts, and retained
  zero pass gain.
- `ibm11` post-swap micro-shift recorded 19 accepts and zero retained pass gain.

Most small-design passes still retained useful proxy improvement, so the whole
stage is not detrimental. The detriment is continuing through subpasses after
the current state has already exceeded a hierarchy limit.

Action: enforce the complete vector at completed subpass boundaries and stop or
restore before more exact-scored work is generated.

### 4. Hierarchy pass/fail does not communicate evidence coverage

**Status: Optimize accuracy reporting and inference.**

The hierarchy vector excludes unclustered hard macros from compactness/purity
and excludes unassigned soft macros from owned/bridge distance. Current
connectivity-derived hard coverage is only `2.2%` on `ibm06`, `24.4%` on
`ibm02`, and `43.1%` on `ibm04`. Soft-role coverage is `2.8%`, `14.9%`, and
`20.4%` respectively. All flat IBM designs activate zero high-confidence soft
bundles.

Impact: “all hierarchy audits passed” is a strong statement on high-coverage
designs but can be nearly vacuous for inferred hierarchy on low-coverage
designs. Singleton movement remains locally bounded, but it is not validated as
hierarchical clustering.

Action: report coverage and provenance with the existing audit; improve the
existing inference conservatively rather than weakening confidence thresholds.

### 5. Cluster confidence is partly self-confirming

**Status: Optimize.**

Hard clusters and their confidence both use the same low-fanout connectivity
graph. Confidence measures internal versus external weight after the partition,
but does not penalize transitive chaining, low total evidence, low coverage, or
an excessively large component. `ibm18`, for example, retains a 224-member
cluster out of 285 hard macros while mean reported cluster confidence remains
high.

Impact: the weak-cluster release mechanism may release small genuinely weak
clusters while overlooking a giant ambiguous component.

Action: retain one confidence value, but calibrate it with conductance, evidence
volume, size/area concentration, and independent synthetic truth.

### 6. Oversized-cluster split eligibility is design-global

**Status: Optimize conservatively.**

The current splitter requires a minimum number of bridge softs counted across
the entire flat design. This evidence is not local to the component being
split. It can block a useful split when the design has few inter-cluster bridge
softs, or enable unrelated components because the threshold is met elsewhere.

Impact: giant inferred components remain on designs such as `ibm18`, while the
historically unsafe full recursive splitter cannot be restored.

Action: use the existing split cut, balance, and area evidence per component;
retain all present minimum-size and maximum-cut safeguards.

### 7. Seed candidates pay for exact scoring before contract filtering

**Status: Optimize as a speed-only change.**

The portfolio legalizes and exact-scores alternatives, then computes their
complete hierarchy vectors and filters them relative to `initial.plc`.
Candidates with immutable hard-component violations can therefore consume soft
cleanup and full exact-score work even though they cannot be selected.

Impact: avoidable seed overhead, especially on large routing grids.

Action: build the initial reference first and prefilter immutable hard
violations before soft cleanup and exact scoring. Do not remove the
constraint-graph candidate or DREAMPlace seed.

## Active Pass Disposition

The table uses run `20260716-final-unused-code-cleanup-all`. It is now interpreted
with rollback-aware retained telemetry where noted.

| Component | Evidence | Disposition | Conclusion |
|---|---|---|---|
| Region swaps | `0.498495` gain, `150.16s`, 0/17 zero-gain | **Optimize** | Largest bottleneck and materially productive; do not remove or reorder casually |
| Region soft relocation | `1.554638` gain, `45.68s`, 0/34 zero-gain | **Retain** | Core proxy contributor |
| Region micro-shift | `1.678602` gain, `11.08s`, 0/34 zero-gain | **Retain** | Best measured return |
| Small-design polish | `0.113715` gain, `42.60s`; rollback on 8/10 | **Optimize** | Productive, but checkpoint earlier |
| Region hard relocation | `0.101256` reported gain, `4.67s`; focused gains rolled back | **Gate** | Cheap, but retained contribution is unknown |
| Interleaved soft repair | `0.049837` gain, `6.61s` | **Retain** | Good return for existing work |
| Plateau escape | `0.029077` gain, `9.53s` | **Retain** | Productive replacement for the dead ordinary soft pass |
| Strong soft repair | `0.025770` gain, `32.62s` | **Gate/optimize** | Positive but low efficiency; stop on no retained gain |
| Post-swap micro-shift | `0.002720` gain, `3.91s`, 47.1% zero-gain | **Gate** | Marginal; needs attributable retained evidence |
| Cluster decompression | Three runs, `0.000465` gain, `1.53s` | **Retain** | Narrow but cheap and positive |
| Compound soft relocation | `0.000251` gain, `1.52s`, 88.2% zero-gain | **Gate** | Negligible average contribution, but too cheap for immediate removal |
| Medium-soft continuation | Scheduled false on all latest 17 designs | **Gate/remove if persistently inert** | No current runtime cost or demonstrated contribution |
| Final hierarchy audit | Required contract; nearest-neighbor JIT is fast | **Retain** | Safety mechanism, not a performance target |

No currently executing production operator has enough clean retained-gain
evidence to justify immediate deletion. The ordinary post-swap soft pass and
broad survivor pool were already removed once that evidence existed.

## Maintenance-Only Drag

These items do not materially affect proxy or runtime but increase ambiguity:

- `MacroPlacer.__init__()` still accepts and stores `n_restarts` and the old
  `noise_fracs` portfolio even though hierarchy-only placement never consumes
  them. Keep only if external callers require constructor compatibility; mark
  them explicitly deprecated if the API can be cleaned later.
- Numerous default-off research hooks remain in active files: graph swap
  weights/masks, graph prefilter/rescue/ranking, ego-net and partial-frontier
  coldspot variants, soft-only coldspot fallback, weak-hot reshape, structural
  ordering, hierarchy-first seed selection, and diagnostic GPU isolation.
  They are not production runtime detriments when disabled, but they enlarge
  the verification and maintenance surface.
- The NG45 verifier chooses a prefix depth with a criterion close to production
  clustering. It validates locality of the selected tags, but is not a fully
  independent measure of hierarchy inference accuracy.

Action: do not delete compatibility or diagnostic surfaces solely for
tidiness. Remove them when they obstruct optimization work or after confirming
that no maintained verifier or user workflow depends on them.

## Already Retired or Rejected

These paths should not be restored without a materially different hypothesis
and an exact, attributable A/B.

| Retired/rejected idea | Evidence | Decision |
|---|---|---|
| Ordinary post-swap soft relocation | Zero gain in 34/34 attributable runs; skipping saved `3.27s` with identical proxy | **Already retired** |
| Broad survivor pool | 636 records, zero total gain, `132.68s` | **Already retired** |
| Learned GNN rankers and candidate logging | Leakage-free GNN lost to existing proposal order; trace I/O raised runtime to `594.48s` | **Already retired** |
| CUDA batch exact-score reductions | `AVG 1.1210 / 656.16s` versus CPU/Numba `1.1205 / 554.54s`; near-tie order changed | **Rejected and removed** |
| Broadened CUDA relocation-delta activation | `AVG 1.1569 / 876.08s`; consumed region-swap budgets | **Rejected and removed** |
| CUDA proposal-filter ranking | Preserved `1.1205` but was slower than the isolated CPU reference | **Rejected and removed** |
| CUDA overlap/bounds prefilter | Roughly 9% slower despite high candidate volume | **Diagnostic-only** |
| Graph-tension CUDA batch | Only 14–75 active edges; insufficient workload to amortize CUDA | **Diagnostic-only control** |
| Broadened plateau escape | `AVG 1.1213 / 546.13s` versus `1.1205 / 541.67s` | **Rejected** |
| Hierarchy-first seed selection | `ibm10` final proxy regressed from `1.1778` to `1.5281` | **Default-off; do not promote** |
| Connectivity-only soft bundle promotion | `ibm11` regressed from `1.0085` to `1.0087` | **Rejected** |
| Full recursive hard clustering | Helped some giant clusters but materially regressed other designs | **Replaced by guarded oversized splitting** |
| Broad weak/hot region reshape | Full-suite regression despite an `ibm04` win | **Default-off** |
| Graph prefilter/ranking/rescue variants | Focused or full-suite regressions, or no proxy benefit | **Default-off research only** |
| Gain-per-second swap scheduler | Shortened the isolated swap stage but regressed final proxy and total placement time | **Removed** |
| Zhang-Hager DREAMPlace line search | Worse seed quality and slower than short-BB on focused controls | **Rejected and removed** |

The same rule applies to the deleted proxy-only restarts, R2/2-opt, generic
LSMC, generic cluster kicks, swap/cycle path, and learned ranking stack: they
belong to historical research, not the hierarchy-only production system.

## Decision Rule for Future Removal

A component should be removed only when attributable, post-rollback telemetry
shows all of the following:

1. zero or negligible retained proxy gain across at least two clean full suites;
2. no hierarchy-component improvement that justifies its cost;
3. no downstream enabling effect when the following stages receive the same
   deterministic work quota;
4. measurable runtime or maintenance savings after removal;
5. unchanged legality, fixed-macro behavior, bounds, and final audits.

Until telemetry distinguishes proposed from retained work, label uncertain
passes as **Gate** rather than **Remove**.
