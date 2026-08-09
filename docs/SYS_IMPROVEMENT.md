# VivaPlace System Improvement Review

Last reviewed: 2026-07-19.

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
AVG 1.1404  17/17 VALID  0 overlaps  all hierarchy audits passed  318.55s
```

The July 18 active-root work first established a corrected-reference control at
`AVG 1.1468 / 404.09s`, then accepted a hard-relocation containment prefilter at
`AVG 1.1412 / 404.01s`. Both sweeps were 17/17 valid with zero overlaps and all
final hierarchy audits passing. Deterministic exact-score ceilings then
preserved every placement and score at `AVG 1.1412 / 398.57s`.
The current revision preserves 16 IBM scores and improves IBM09 `1.0122 ->
0.9978` through a tightly bounded contract repair, reaching `AVG 1.1404 /
318.55s`. Its fallback inference combines structural edges with initial
placement, density, and wire-pressure evidence. NG45 remains `AVG 0.7121`
through one retained explicit-hierarchy child move, now in 64.80s.
Region swaps read global topology directly, reduce congestion/density from
baseline plus touched cells, and share static hard separation geometry across
their complete schedule.

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

## 2026-07-19 Priority-Order Outcome

The requested priority sweep is complete:

1. Seed opportunity: a one-component repair retaining at least 95% of the
   failed source is now production. IBM09 retained 99.61% and improved its
   final proxy by 0.0144; broad IBM03/13 repairs were rejected.
2. Soft scoring: one exact compiled wirelength batch now filters targets before
   congestion/density work. It rejected 100,831 IBM proposals and reduced the
   four main soft phases by 20.7–22.3%. An exact sparse ordinary-move tail was
   measured and removed because IBM12/18 did not amortize it.
3. Swap throughput: a second stable 4/8/12 prefix preserves order and quotas,
   increases avoided exact work to 79,466, and reduces trace-compatible region
   time from 104.04s to 98.74s.
4. Attribution: five exclusive phases cover at least 99.86% of each API call.
   Hierarchy search is 202.38s of the 297.33s placer total; evaluator loading
   and final scoring add 21.22s outside the API.
5. Soft confidence: repeated support and ambiguity now calibrate role evidence.
   Flat roles remain medium/low and independent; only explicit path bundles may
   form compound moves.

The full gates are `AVG 1.1404` on IBM (17/17), `AVG 0.7121` on NG45 (4/4),
and `AVG 1.4192` on synthetic (10/10 truth), with all 94 project tests passing.

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

### 4. Implemented: stable-prefix region swaps preserve exact CPU decisions

Region swaps remain the largest local-search runtime and contribute `0.476290`
retained proxy gain. The accepted implementation preserves each source's stable
candidate order but scores only a four-candidate hard-hard prefix, an
eight-candidate hard-soft prefix, or a twelve-candidate soft-soft prefix first.
If the first prefix misses, it scores one second prefix of the same size. If
either contains the first acceptable move, the untouched suffix is irrelevant
and is not scored; otherwise the remainder runs as the original batch.

Refine the existing batched scorer by:

- constructing legality, region, blockage, routing, and density trial data in
  stable Numba batches for every candidate group with a common endpoint;
- consuming one global net/pin topology rather than constructing a topology
  object for every candidate pair;
- reducing exact congestion/density tails from changed cells plus the sorted
  unchanged baseline instead of materializing full result matrices;
- batching small singleton groups together only when their original stable
  order and per-source commit boundary can be reproduced;
- keeping final float64 reduction and near-tie decisions in the accepted CPU
  order.

Do not move per-source batches to CUDA. Existing experiments showed that device
launch, copies, synchronization, and changed reduction order cost both proxy
and runtime.

Literature provenance and scope are recorded in
[`REFERENCES.md`](REFERENCES.md#hierarchy-search-acceleration-literature),
entries 21–27. ABCDPlace and GPU-DPO motivated the batch/concurrency
investigation, but their published speedups are not VivaPlace forecasts. The
accepted CSR union, scratch reuse, and exact sparse reductions are independent
CPU implementations with VivaPlace's original sequential commit semantics.

Acceptance result: batch/scalar parity still passes; all 17 IBM scores are
bit-identical. The initial sweep avoided 58,820 exact evaluations versus full
batches and reduced attributed region-swap time `159.91s -> 150.68s`. Ranking
before hard legality, calibrating only the soft-soft prefix to 12, and skipping
disabled graph allocations then increased avoided work to 66,703 and reduced
the phase again to 148.29s. NG45 avoided 10,068 evaluations and synthetic
avoided 2,578. Reusable full-grid workspaces and a fused density reducer were
rejected because they did not improve isolated runtime.

A follow-up kept all candidate, score, and placement decisions identical while
removing two remaining allocation classes: congestion-tail reduction uses the
already-disposable batch grid as its partition workspace, and the complete
swap schedule shares one immutable pair of hard separation matrices. IBM
physical/avoided work remained 1,077,431 / 66,703 and attributed region time
fell `148.29s -> 146.98s`. Full evaluator time was flat at 416.87s, so the
accepted claim is the measured 1.32s local reduction, not an end-to-end speedup.

The direct-topology/touched-tail implementation completed that recommendation.
The compiled route loop packs selected pin cells from global net starts,
lengths, weights, pin references, and offsets while retaining the evaluator's
2-pin/3-pin/high-fanout order. Congestion recomputes only route-bbox strips and
changed hard-blockage cells; density applies only the four swap rectangles.
Both reducers merge candidate values with the sorted unchanged baseline for an
exact top tail. IBM physical/avoided work remained 1,077,431 / 66,703 while
attributed region time fell `146.98s -> 104.04s` and complete runtime fell
`416.87s -> 351.48s`. NG45 preserved AVG 0.7121 while total runtime fell
`79.43s -> 65.27s`.

The next accepted refinement caches the sorted congestion/density baseline,
density nonzero count, and density sum across rejected batches, invalidating
them only when a scorer commit changes placement state. IBM physical/avoided
work remained exactly 1,077,431 / 66,703 and attributed region time fell
`104.04s -> 102.68s`. Focused phase measurements were IBM04
`4.527s -> 4.183s`, IBM12 `6.811s -> 6.178s`, and IBM18
`9.008s -> 8.508s`. NG45 region time fell `15.406s -> 14.616s` with AVG
0.7121 unchanged. Complete sweep times (IBM 371.82s, NG45 67.27s) varied in
other stages, so they are verification totals rather than the optimization's
claimed gain.

The current follow-up exact-scores a second prefix of the same size before the
untouched remainder. It retains the first-acceptable winner and logical quota
semantics while moving IBM physical/avoided work to 1,066,186 / 79,466. The
trace-compatible region phase is 98.74s. The lower logical count (1,048,385)
comes from IBM09's changed seed basin, not from prefix truncation.

The latest follow-up prepares source-invariant work once when a schedule spans
multiple prefixes: candidate modules/coordinates, the position snapshot, and
the ragged incident-net union. Prefix calls use stable slices and rebased
offsets, so exact work and first-winner semantics remain unchanged. The full
IBM phase was non-regressive at `94.37s -> 94.29s` with 1,048,385 logical,
1,066,186 physical, and 79,466 avoided scores. A revision-scoped exact cache
was rejected despite 16,265 hits (`94.37s -> 95.51s`), and compact route/
blockage delta grids were rejected at 95.52s.

For soft relocation, cached Numba kernels now fuse grid-ID conversion,
clipping, symbolic-key construction, region-mask filtering, and stable stamp
deduplication. Candidate order, rich hierarchy callbacks, and exact workload
are unchanged. Across the five measured soft phases this reduced `74.039s ->
73.400s`; region-soft relocation reduced `38.431s -> 37.916s`. A 15,094-hit
exact cache regressed the five phases to 78.349s, and a fused scorer transaction
API regressed the accepted 73.400s reference to 75.117s; both were removed.

Two follow-ups were rejected and removed. Fusing hard-blockage batches into a
single reducer scratch changed IBM04/12/18 region times to 4.220s, 6.386s, and
8.428s, regressing two of three. Parallel congestion rows with Numba `prange`
were substantially worse: IBM18 reached the 20s guard after 32,498 candidates,
where the accepted serial reducer completes all 69,152 in 8.508s. The small,
repeated batches do not amortize a 22-thread launch or per-row scratch.

### 5. Implemented: prune the seed portfolio before expensive scoring

The legalized-reference correction and immutable hard-component prefilter are
now production. Mandatory stability candidates remain available, while
ordinary alternatives can be rejected before exact scoring. Structured events
make later limit calibration attributable.

A second accepted rule handles useful mandatory near misses without loosening
the contract. Candidates must have lower exact proxy than the selected seed and
violate exactly one component. Deterministic legalization/interpolation finds
the passing boundary, and only repairs retaining at least 95% of source
displacement are exact-scored. IBM09's accepted fraction is `0.99609375`;
broader IBM03 and IBM13 repairs are rejected.

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

Rollback-aware lane telemetry and conservative stopping are now implemented:

- keep both region micro-shift and region soft relocation unchanged;
- keep plateau escape and interleaved soft repair unless retained telemetry
  contradicts their current positive yield;
- stop strong/medium soft repair after audit restore or its first lane with
  retained gain no larger than `0.00005`;
- run post-swap micro-shift only when the preceding swap state changed relevant
  hot macros or the refreshed field exposes a new local opportunity;
- retain compound soft relocation. A full control/off A/B saved 6.66 seconds
  when disabled but regressed ibm12 and ibm16, so zero gain in one later trace
  is not sufficient removal evidence.

Do not directly broaden another pass with saved time. The rejected `384/10/4s`
to `512/12/6.5s` plateau expansion regressed the suite by changing later search
basins. Do not truncate the ordered sources inside a lane: 128-, 256-, and
384-source dry probes regressed IBM12 because a useful second-round congestion
move appears deeper in the tail.

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

### 10. Implemented: calibrate soft role evidence conservatively

All flat IBM designs currently have zero active high-confidence soft bundles,
even though they expose many medium/low-confidence connectivity candidates.
The earlier direct promotion of inferred communities regressed `ibm11` from
`1.0085` to `1.0087`, so those communities must not be promoted wholesale.

The existing owned/bridge classifier now records dominance share, runner-up
ambiguity, repeated support, score, confidence, and evidence source. Its role
mapping remains unchanged for individual regions and contracts. Flat evidence
is capped below high confidence and therefore cannot form a compound group;
compound movement remains restricted to explicit path bundles. The full IBM
cohort recorded 3,246 medium and 1,938 low flat roles, with no inferred high
roles. All proxy/legality gates passed, and compound stage time fell from 1.70s
to 0.16s because unsupported flat groups are no longer proposed.

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
graph bisection of an eligible active cluster. The fallback graph contains only
direct low-fanout hard relations and shared-soft support. Initial
hard/soft proximity, local macro-area density, and placed wire demand reinforce
those relations, but cannot invent a disconnected group. Raw structural cut,
compactness gain, and combined confidence gate the result. It never recurses,
so discovery cost and confidence do not compound down an inferred tree.

The corresponding local pass runs after active-cluster macro relief. It tests
child translations and sibling slot swaps inside the parent region, carries
owned soft macros, and uses affected-only compaction/legalization when a rigid
state is blocked. Complete candidate states must pass active, child, and parent
contracts before exact mixed hard/soft scoring. The multilevel contract becomes
authoritative for later passes only when a child move is retained. This scope is
important: enabling the tighter contract from discovery alone changed otherwise
unrelated cleanup trajectories on the synthetic suite.

The pass has a shared 24-state exact quota, 4s guard, and 0.0001 gain floor. The
spatial/structural IBM sweep inferred 23 fallback parents / 46 children on ten
designs, saw 38 eligible children, exact-scored 24 states, retained none, and
reproduced every reference score at `AVG 1.1412`. NG45 retained one localized
child move on `ariane136`, improving the suite to `AVG 0.7121` with all audits
passing. The 10-design synthetic suite remained 10/10 valid with 10/10 truth
audits at `AVG 1.4195`.

The deepest-child extension now turns that retained level into an internal
search boundary without inferring another partition. Each current child
footprint receives `0.01` base canvas-fraction margin plus up to `0.025` from a
congestion/density/graph-pressure blend. Hot boxes may grow another `0.01`
toward cold connected components, with graph corridors biasing direction, and
the final box is clipped to its parent. The operator searches individual hard
and owned-soft relocations plus same-child hard swaps, uses graph tension and
neighboring-child centroids for priority/anchors, and applies the full
multilevel contract before exact acceptance.

The accepted 48-score/3s pass exact-scored 528 IBM states in 2.93s and retained
none at its 0.0005 floor, preserving all reference placements at AVG 1.1412.
This zero retained yield is deliberate calibration, not a disabled operator:
the 0.0001 trial retained six moves with total immediate gains between roughly
0.0001 and 0.0004 per design, but the newly active downstream contract displaced
larger later gains and regressed AVG to 1.1453, including ibm18
`1.3773 -> 1.4361`. The higher floor restores every affected design exactly.
The independent synthetic sweep reached `AVG 1.4193`, 10/10 valid, zero
overlaps, and 10/10 truth-audit passes, confirming that the fixed boxes and
quotas also hold on non-square, fixed-macro, route-inverted, seedless, and
large-scale cases.

## Promotion Order

The preceding priority order is implemented through seed repair, stable CPU
swap throughput, exclusive attribution, conservative soft-role calibration,
compiled pair unions, reusable sparse-reducer scratch, stable integer soft
targets, prepared multi-prefix swap sources, compiled stable target filtering,
and fused reusable dense soft scoring. The exact-safe lower-bound
prototype was removed after only 1.2% IBM10 soft-soft rejection; speculative
source waves and net-optimal prefix ranking remain unpromoted.
The incremental-routing paper motivated the rejected bound; FastDP and CROP
motivated the unpromoted ranking experiment; Xplace supplied broad
operator-fusion systems context. Their precise scope and non-adoption status
are recorded as entries 23–27 in the
[acceleration literature index](REFERENCES.md#hierarchy-search-acceleration-literature).
The next order is:

1. Reduce the remaining hierarchy-search phase with dependency-safe batching
   or stronger exact-safe proposal bounds; measure region swaps and region-soft
   relocation separately, and require a clean-host attributable win.
2. Calibrate hard-cluster confidence against synthetic precision/conductance
   before changing weak-cluster release behavior.
3. Extend NG45 verification across every useful nested path-prefix depth.
4. Keep recursive inferred hierarchy out of scope until the existing child/deep
   operators show retained proxy yield.
5. Broaden DREAMPlace and EDA-I/O portability coverage without changing the
   accepted placement contract.

For every promotion, run focused `ibm10` plus at least one low-coverage and one
large-grid case before `--all`. A clustering change also requires the synthetic
ground-truth suite and NG45 hierarchy verification. Record accepted full-suite
numbers in [PROGRESS.md](PROGRESS.md); do not infer a win from deadline-induced
candidate-count changes alone.
