# VivaPlace System Improvement Review

Last reviewed: 2026-07-18.

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
AVG 1.1412  17/17 VALID  0 overlaps  all hierarchy audits passed  423.87s
```

The July 18 active-root work first established a corrected-reference control at
`AVG 1.1468 / 404.09s`, then accepted a hard-relocation containment prefilter at
`AVG 1.1412 / 404.01s`. Both sweeps were 17/17 valid with zero overlaps and all
final hierarchy audits passing. Deterministic exact-score ceilings then
preserved every placement and score at `AVG 1.1412 / 398.57s`.
The one-level hierarchy revision again preserves every IBM placement at `AVG
1.1412 / 423.87s`, while improving NG45 `AVG 0.7123 -> 0.7121` through one
retained explicit-hierarchy child move.

Latest attributable retained-yield telemetry
(`20260718-issue2-deterministic-quotas-all`):

| Existing pass | Runs | Proxy gain | Time (s) | Zero-gain runs | Reading |
|---|---:|---:|---:|---:|---|
| Micro-shift | 34 | 1.745321 | 12.42 | 2.9% | Highest measured gain per second |
| Region soft relocation | 34 | 1.597352 | 47.67 | 5.9% | Major proxy contributor |
| Region swaps | 17 | 0.476290 | 152.42 | 5.9% | Largest measured runtime target |
| Region hard relocation | 34 | 0.092785 | 14.57 | 11.8% | Rollback-free after prefilter |
| Interleaved soft repair | 17 | 0.055575 | 7.14 | 29.4% | Productive and inexpensive |
| Small-design polish | 10 | 0.047123 | 8.24 | 60.0% | Lower yield, still positive |
| Plateau escape | 16 | 0.021024 | 7.49 | 37.5% | Productive late soft search |
| Strong soft repair | 17 | 0.012716 | 23.30 | 41.2% | Positive but comparatively inefficient |
| Post-swap micro-shift | 17 | 0.002108 | 3.59 | 58.8% | Marginal |
| Cluster decompression | 2 | 0.000507 | 1.13 | 0.0% | Narrow, cheap, positive |
| Compound soft relocation | 17 | 0.000000 | 1.50 | 100.0% | Removal A/B still regressed two designs |

These are retained gains after checkpoint enforcement. Telemetry now separates
proposed and retained outcomes, includes rollback reasons and discarded gain,
and fingerprints dirty source trees. The accepted hard-relocation prefilter
rejected 654 candidates before exact scoring and left all 34 relocation pass
records rollback-free.

## Priority 0: Make Optimization Evidence Trustworthy

### 1. Implemented: record retained work, not only attempted work

Completed on 2026-07-18. Forced-rollback tests report zero retained
accepts/gain, and stage timing plus structured seed/final contract events share
the same attributable schema. The details below are retained as the contract
for future telemetry changes.

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

### 2. Implemented: make timed search reproducible

Completed on 2026-07-18. Exact-scored relocation, compound-relocation, and
regional-swap work now stops at deterministic per-pass ceilings before the
wall-clock safety guard. Repeated region rounds share their allowance, and all
three regional swap types share one 72,000-evaluation allowance. Telemetry
records usage/exhaustion and the analyzer exposes a `--quotas` cohort report.

Keep wall-clock deadlines as safety caps, but give each existing operator a
deterministic candidate or exact-score quota derived from the accepted run.
Stop at the quota first and at the deadline only as an emergency guard. This
places a stable upper bound on high-volume work before late cleanup and final
audits; severe contention may still trigger the safety deadline.

Do not reintroduce the rejected gain-per-second scheduler. It changed search
basins because elapsed time was used as the primary decision signal. Candidate
quotas must preserve the current candidate order and commit order.

Acceptance gate: repeated isolated and `--all` runs must select the same
placement when the safety deadline is not hit; the safety cap must still keep
the suite below one hour.

The full IBM and NG45 validations preserved every preceding score. An
aggressive binding control remained legal and reached later passes, but its
ibm11 regression rejected tighter production pruning.

## Priority 1: Recover Proxy and Runtime Inside the Existing Flow

### 3. Partly implemented: stop hierarchy-ineligible work earlier

The first high-confidence case is now production: hard-relocation candidates
above the selected seed's cheap hard-containment limit are rejected before
exact scoring. It improved the full suite without changing runtime and removed
the observed relocation rollback churn. Apply the remaining ideas only after
their own attributable rollback evidence.

Use the already-fast hierarchy vector at completed-state boundaries:

- consider the same early boundary check for micro-shift only if repeated
  attributable rollback remains after the current checkpoint;
- check the current winner after each swap sub-round, not every raw swap;
- in small-design polish, remember and restore the best vector-passing state
  after each subpass rather than after a long sequence of accepted moves;
- expose discarded gain in telemetry so a prefilter can be tightened only when
  rollback is repeated across attributable runs.

The latest accepted suite recorded no rollback for small-design polish and no
hard-relocation rollback. Region soft relocation and micro-shift still show
occasional rollback, so those boundaries are the next evidence-driven targets;
neither pass should be removed because retained yield remains high.

Acceptance gate: final proxy and hierarchy vectors must match the control before
any saved time is reinvested. Reinvestment requires a separate full-suite A/B.

### 4. Accelerate region swaps while preserving exact CPU decisions

Region swaps consume `153.22s`, about 38% of the latest reported full-suite
runtime, and contribute `0.476290` retained proxy gain. They are the main optimization
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

### 5. Implemented: prune the seed portfolio before expensive scoring

The legalized-reference correction and immutable hard-component prefilter are
now production. Mandatory stability candidates remain available, while
ordinary alternatives can be rejected before exact scoring. Structured events
make later limit calibration attributable.

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

Acceptance result: the corrected legalized reference intentionally changed
eligibility on designs where the former raw reference was inconsistent. The
full control improved and all 17 final contracts passed.

### 6. Tune late cleanup from retained yield

After rollback-aware telemetry is available, use conservative structural gates
inside the existing scheduler:

- keep both region micro-shift and region soft relocation unchanged;
- keep plateau escape and interleaved soft repair unless retained telemetry
  contradicts their current positive yield;
- stop strong soft repair after its first no-retained-gain lane or round;
- run post-swap micro-shift only when the preceding swap state changed relevant
  hot macros or the refreshed field exposes a new local opportunity;
- retain compound soft relocation. A full control/off A/B saved 6.66 seconds
  when disabled but regressed ibm12 and ibm16, so zero gain in one later trace
  is not sufficient removal evidence.

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

### 8. Implemented: make oversized splitting component-specific

Split eligibility now counts unique bridge softs per flat component, so evidence
from an unrelated component cannot authorize a split. A nearly all-covering
single component is the one exception because bridge evidence is impossible
before a partition exists. It uses shared low-fanout hard-to-soft affinity with
strict size/similarity gates, then falls back to a stricter partial hard-graph
cut. Synthetic truth improved to 10/10 passes and ibm10 preserved `1.1348`;
explicit NG45 path tags continue to bypass flat inference. Further expansion
still requires the same truth, IBM proxy, and explicit-tag preservation gates.

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

### 11. Implemented: validate clustering against independent truth

The synthetic generator now preserves hard and soft cluster assignments in
generated metadata. `run_synthetic.py` reports inference coverage, majority
purity, and pairwise precision/recall, then evaluates final placement with the
same six component forms over the generator's independent groups. It emits
`hierarchy_truth_audit` telemetry for the contract analyzer.

The single-component refinement now closes the important `syn03_sram` failure:
production inference recovers all four truth groups with pair recall, precision,
and purity of `1.0`, without loosening any numeric limit. The full rerun passes
10/10 truth audits and improves synthetic average proxy `1.4262 -> 1.4206`.
The refinement uses shared low-fanout hard-to-soft affinity only when nearly all
hard macros collapsed into one flat component; multi-component and explicit-tag
construction remain unchanged. Accuracy is still coarse on some passing cases,
so confidence calibration remains useful follow-up work rather than a reason to
broaden this gate.

The NG45 verifier currently chooses a prefix depth with logic similar to the
production selector. Extend it to report preservation at every useful prefix
depth so it tests nested explicit hierarchy independently of the chosen flat
partition.

### 12. Implemented: one non-recursive parent/child level

The hierarchy model now retains one additional structural level beside the
active DREAMPlace partition. The level comes from the nearest useful explicit
path ancestor, an original component above existing split leaves, or one strict
graph bisection of an eligible active cluster. It never recurses, so discovery
cost and confidence do not compound down an inferred tree.

The corresponding local pass runs after active-cluster macro relief. It tests
child translations and sibling slot swaps inside the parent region, carries
owned soft macros, and uses affected-only compaction/legalization when a rigid
state is blocked. Complete candidate states must pass active, child, and parent
contracts before exact mixed hard/soft scoring. The multilevel contract becomes
authoritative for later passes only when a child move is retained. This scope is
important: enabling the tighter contract from discovery alone changed otherwise
unrelated cleanup trajectories on the synthetic suite.

The pass has a shared 24-state exact quota, 4s guard, and 0.0001 gain floor. The
IBM sweep considered 418 candidates, exact-scored 55 in 2.26s, retained none,
and reproduced every reference score at `AVG 1.1412`. NG45 retained one
localized child move on `ariane136`, improving the suite to `AVG 0.7121` with
all audits passing. The 10-design synthetic suite remained 10/10 valid with
10/10 truth audits and improved slightly to `AVG 1.4204`.

## Promotion Order

1. Rollback-aware telemetry and missing stage timings (implemented).
2. Deterministic work quotas and audit-churn measurement (implemented).
3. Seed prescore pruning as an exact-equivalent speed change.
4. Stable CPU/Numba swap throughput work.
5. Evidence-gated late-pass scheduling.
6. Synthetic accuracy metrics (implemented); extend NG45 nested-prefix checks.
7. Single-component split (implemented); continue confidence calibration.
8. Conservative soft-role calibration.
9. One-level parent/child search (implemented); keep recursive inference out of
   scope until one-level evidence coverage is independently calibrated.

For every promotion, run focused `ibm10` plus at least one low-coverage and one
large-grid case before `--all`. A clustering change also requires the synthetic
ground-truth suite and NG45 hierarchy verification. Record accepted full-suite
numbers in [PROGRESS.md](PROGRESS.md); do not infer a win from deadline-induced
candidate-count changes alone.
