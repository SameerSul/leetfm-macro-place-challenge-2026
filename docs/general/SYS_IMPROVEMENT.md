# VivaPlace System Improvement Review

Last reviewed: 2026-07-17.

## Purpose

This document identifies the highest-value ways to improve the existing
hierarchy-only VivaPlace system. The priority is **hierarchy-safe proxy gain**:
an optimization is useful only when it preserves legality and does not weaken
the current hierarchy contract. The recommendations refine the existing
clustering, seed, scoring, scheduling, and local-search machinery; they do not
add a second placer, learned ranker, or replacement objective.

For the active flow, see [ARCHITECTURE.md](ARCHITECTURE.md) and
[DESIGN_FLOW.md](DESIGN_FLOW.md). Negative findings and retired experiments are
classified in [SYS_DETRIMENT.md](SYS_DETRIMENT.md). Historical measurements
come from [PROGRESS.md](PROGRESS.md).

## Current Reference

The stable accepted reference is:

```text
AVG 1.1205  17/17 VALID  0 overlaps  all hierarchy audits passed  554.54s
```

The latest unused-code-cleanup run reported `AVG 1.1204` in `492.98s` with the
same validity and audit result. Treat the rounded proxy and wall-time difference
as run-to-run evidence, not as a separately attributed algorithmic improvement.
The flow is deadline-sensitive: a review-time `ibm10` run finished at `1.0651`
instead of the accepted `1.0641` while remaining valid and audit-passing.

Latest attributable pass telemetry (`20260716-final-unused-code-cleanup-all`):

| Existing pass | Runs | Proxy gain | Time (s) | Zero-gain runs | Reading |
|---|---:|---:|---:|---:|---|
| Micro-shift | 34 | 1.678602 | 11.08 | 0.0% | Highest measured gain per second |
| Region soft relocation | 34 | 1.554638 | 45.68 | 0.0% | Major proxy contributor |
| Region swaps | 17 | 0.498495 | 150.16 | 0.0% | Largest measured runtime target |
| Small-design polish | 10 | 0.113715 | 42.60 | 10.0% | Productive, with rollback churn |
| Region hard relocation | 34 | 0.101256 | 4.67 | 17.6% | Reported gain is partly pre-rollback |
| Interleaved soft repair | 17 | 0.049837 | 6.61 | 29.4% | Productive and inexpensive |
| Plateau escape | 17 | 0.029077 | 9.53 | 17.6% | Productive late soft search |
| Strong soft repair | 17 | 0.025770 | 32.62 | 11.8% | Positive but comparatively inefficient |
| Post-swap micro-shift | 17 | 0.002720 | 3.91 | 47.1% | Marginal |
| Cluster decompression | 3 | 0.000465 | 1.53 | 0.0% | Narrow, cheap, positive |
| Compound soft relocation | 17 | 0.000251 | 1.52 | 88.2% | Marginal but inexpensive |

These gains are not all retained gains. Several pass records are written before
the hierarchy checkpoint can roll the placement back. On the focused `ibm10`
review run, both region-hard-relocation rounds were rolled back even though
telemetry attributed `0.023976` proxy gain to the pass. Correcting this
measurement is the first improvement because every later scheduling decision
depends on it.

## Priority 0: Make Optimization Evidence Trustworthy

### 1. Record retained work, not only attempted work

Extend the existing plateau telemetry rather than adding another trace system.
For every pass, record:

- entry, proposed-exit, and retained-exit proxy;
- proposed and retained accepts;
- whether a hard-quality or six-component vector check caused rollback;
- the violating hierarchy components and the discarded proxy gain;
- scorer reconstruction time after rollback;
- separate timings for DREAMPlace/cache lookup, seed creation and prescoring,
  coldspot work, full exact scores, and final audits.

Move the authoritative pass record to after checkpoint enforcement, while
keeping proposed values as diagnostic fields. Update the analyzer to rank work
by retained gain per scored candidate and retained gain per second.

Acceptance gate: telemetry must reconstruct the proxy at every retained pass
boundary, and a forced rollback test must report zero retained accepts/gain.

### 2. Make timed search reproducible

Keep wall-clock deadlines as safety caps, but give each existing operator a
deterministic candidate or exact-score quota derived from the accepted run.
Stop at the quota first and at the deadline only as an emergency guard. This
prevents CPU contention from changing which swap type, small-design round, or
late cleanup is reached.

Do not reintroduce the rejected gain-per-second scheduler. It changed search
basins because elapsed time was used as the primary decision signal. Candidate
quotas must preserve the current candidate order and commit order.

Acceptance gate: repeated isolated and `--all` runs must select the same
placement when the safety deadline is not hit; the safety cap must still keep
the suite below one hour.

## Priority 1: Recover Proxy and Runtime Inside the Existing Flow

### 3. Stop hierarchy-ineligible work earlier

Use the already-fast hierarchy vector at completed-state boundaries:

- check a hard-relocation or micro-shift round before starting the next pass;
- check the current winner after each swap sub-round, not every raw swap;
- in small-design polish, remember and restore the best vector-passing state
  after each subpass rather than after a long sequence of accepted moves;
- expose discarded gain in telemetry so a prefilter can be tightened only when
  rollback is repeated across attributable runs.

Eight of ten small-design passes in the latest suite set `audit_rollback=true`.
Most still retained useful net gain, so the pass should not be removed. The goal
is to avoid work after the state has already crossed a component limit.

Acceptance gate: final proxy and hierarchy vectors must match the control before
any saved time is reinvested. Reinvestment requires a separate full-suite A/B.

### 4. Accelerate region swaps while preserving exact CPU decisions

Region swaps consume `150.16s`, about 30% of the latest reported full-suite
runtime, and contribute `0.498495` proxy gain. They are the main optimization
target, not a removal target.

Refine the existing batched scorer by:

- constructing legality, region, blockage, routing, and density trial data in
  stable Numba batches for every candidate group with a common endpoint;
- reusing prepared routing/density buffers across hard-hard, hard-soft, and
  soft-soft groups where the committed scorer version is unchanged;
- batching small singleton groups together only when their original stable
  order and per-source commit boundary can be reproduced;
- keeping final float64 reduction and near-tie decisions in the accepted CPU
  order.

Do not move per-source batches to CUDA. Existing experiments showed that device
launch, copies, synchronization, and changed reduction order cost both proxy
and runtime.

Acceptance gate: batch/scalar deltas and committed grids must match the scalar
reference, candidate and winner order must be identical, and a full suite must
not lose any accepted proxy result.

### 5. Prune the seed portfolio before expensive scoring

Retain every current seed type, including the productive constraint-graph
legalization. Change only evaluation order:

1. Legalize `initial.plc` first and build its component limits.
2. Legalize each alternative and compute the hard-only hierarchy components.
3. Reject an alternative before soft cleanup or a full exact score when an
   immutable hard component already exceeds its limit.
4. Finish soft cleanup, compute the complete vector, and exact-score only the
   remaining eligible candidates.
5. Preserve the current `(score, name)` selection order among eligible seeds.

The constraint-graph seed must remain: it was selected on `ibm10`, `ibm12`, and
`ibm14` through `ibm18` in its promotion sweep.

Acceptance gate: the selected seed and final placement must be identical to the
control; the optimization is accepted initially as speed-only.

### 6. Tune late cleanup from retained yield

After rollback-aware telemetry is available, use conservative structural gates
inside the existing scheduler:

- keep both region micro-shift and region soft relocation unchanged;
- keep plateau escape and interleaved soft repair unless retained telemetry
  contradicts their current positive yield;
- stop strong soft repair after its first no-retained-gain lane or round;
- run post-swap micro-shift only when the preceding swap state changed relevant
  hot macros or the refreshed field exposes a new local opportunity;
- gate compound soft relocation by the presence of a qualifying group and a
  cheap field drop, while preserving the current exact final-state check.

Do not directly broaden another pass with saved time. The rejected `384/10/4s`
to `512/12/6.5s` plateau expansion regressed the suite by changing later search
basins.

## Priority 2: Improve Hierarchy Accuracy Without Replacing the Model

### 7. Qualify every hierarchy result by evidence coverage

The six-component contract is meaningful only for macros that received a hard
cluster or soft role. Current flat-netlist coverage varies widely:

| Design | Clustered hard | Hard coverage | Owned/bridge soft | Soft-role coverage | Largest hard cluster |
|---|---:|---:|---:|---:|---:|
| `ibm06` | 4/178 | 2.2% | 25/900 | 2.8% | 2 |
| `ibm02` | 66/271 | 24.4% | 160/1075 | 14.9% | 12 |
| `ibm04` | 127/295 | 43.1% | 221/1085 | 20.4% | 7 |
| `ibm10` | 730/786 | 92.9% | 784/1982 | 39.6% | 75 |
| `ibm18` | 236/285 | 82.8% | 48/1029 | 4.7% | 224 |

Add these coverage values and explicit-versus-inferred provenance to final
audit reporting. A placement can still pass, but the report must distinguish a
high-coverage hierarchy pass from a low-evidence pass. Use coverage to select
which benchmarks are suitable for clustering calibration; do not relax proxy
gates solely to increase coverage.

### 8. Make oversized splitting component-specific

The current split eligibility uses the total number of bridge softs in the
flat design. This can block a giant component such as `ibm18`'s 224-member
cluster, or enable unrelated components because bridge evidence exists
elsewhere.

Keep the existing balanced graph split, minimum leaf size, target size, and cut
ratio. Replace only the global eligibility decision with evidence local to the
component: internal cut ratio, balance, edge support, and area balance. Require
all existing safeguards and reject a split if any child lacks sufficient
internal support.

Acceptance gate: synthetic ground-truth accuracy must improve; correlated IBM
cases must improve or preserve proxy; NG45 path-derived clusters must be
unchanged.

### 9. Calibrate confidence independently from cluster construction

Current confidence is based on internal versus external low-fanout net weight
using the same graph that formed the cluster. It can therefore assign high
confidence to a component created by transitive chaining or to a giant cluster
with little useful separation.

Refine the existing confidence value with component conductance, normalized cut
weight, size/area concentration, and evidence coverage. Use the calibrated
value in the current weak-cluster release logic; do not add a second clustering
path.

Acceptance gate: confidence must correlate with synthetic cluster precision and
must not cause a previously protected explicit-tag group to be released.

### 10. Expand soft evidence conservatively

All flat IBM designs currently have zero active high-confidence soft bundles,
even though they expose many medium/low-confidence connectivity candidates.
The earlier direct promotion of inferred communities regressed `ibm11` from
`1.0085` to `1.0087`, so those communities must not be promoted wholesale.

Improve the existing owned/bridge classifier with net weights, repeated-net
support, and an explicit ambiguity margin. Use medium-confidence bundle
evidence first for audit coverage and proposal ordering only. Compound movement
remains restricted to explicit path bundles until an independent accuracy gate
is passed.

### 11. Validate clustering against independent truth

The synthetic generator already creates hard and soft cluster assignments but
does not preserve them as an accuracy contract. Store those labels in generated
metadata and evaluate inferred clusters with pairwise precision/recall and an
adjusted partition score. Add placement-level checks for within-cluster radius
and cross-cluster purity.

The NG45 verifier currently chooses a prefix depth with logic similar to the
production selector. Extend it to report preservation at every useful prefix
depth so it tests nested explicit hierarchy independently of the chosen flat
partition.

## Promotion Order

1. Rollback-aware telemetry and missing stage timings.
2. Deterministic work quotas and audit-churn measurement.
3. Seed prescore pruning as an exact-equivalent speed change.
4. Stable CPU/Numba swap throughput work.
5. Evidence-gated late-pass scheduling.
6. Synthetic/NG45 accuracy metrics.
7. Component-local split and confidence calibration.
8. Conservative soft-role calibration.

For every promotion, run focused `ibm10` plus at least one low-coverage and one
large-grid case before `--all`. A clustering change also requires the synthetic
ground-truth suite and NG45 hierarchy verification. Record accepted full-suite
numbers in [PROGRESS.md](PROGRESS.md); do not infer a win from deadline-induced
candidate-count changes alone.
