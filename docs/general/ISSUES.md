# Current Issues

Last revised: 2026-07-19.

This file tracks unresolved work in the hierarchy-only VivaPlace system. The
complete experiment history, including rejected proxy-path work, lives in
[`PROGRESS.md`](PROGRESS.md).

## Current State

`MacroPlacer.place()` requires grouped DREAMPlace and always runs the hierarchy
pipeline. The latest full IBM sweep is:

```text
uv run evaluate src/main.py --all
AVG 1.1412  17/17 VALID  0 overlaps  351.48s
```

All final hierarchy audits passed. Stable-prefix region-swap scoring,
ranked-only hard legality, and disabled-graph allocation removal reproduced
every per-design score from the 398.57s deterministic-quota reference exactly,
avoided 66,703 exact evaluations, and reduced attributed swap time from 159.91s
to 148.29s. In-place congestion-tail partitioning and schedule-scoped hard
separation geometry preserve the same physical/avoided score counts and reduce
the phase again to 146.98s. Direct global-topology swap routing plus exact
baseline/touched-cell congestion and density tails preserve those same counts
and reduce the phase to 104.04s. The latest NG45 result is `AVG 0.7121`, 4/4
VALID, zero overlaps, all audits passed, in 65.27s.

The learned-ranking stack has been removed. Candidate ordering is deterministic;
exact proxy, hard legality, bounds, fixed-macro immobility, hierarchy regions,
and hierarchy-quality gates remain authoritative.

## Open Work

### 1. Production hierarchy contract calibration (complete)

Resolved on 2026-07-18. Production legalizes `initial.plc` before building any
immutable component limit and exact-scores those same coordinates as the
ordinary initial candidate. Exact-scored seeds and the final placement emit
structured `hierarchy_contract_audit` rows containing the six component
values, limits, signed margins, violations, evidence coverage, and provenance.
NG45 rows now use the actual design names instead of the shared
`output_CT_Grouping` directory name.

`scripts/analyze_hierarchy_contract.py` replayed the active limits and
counterfactual profiles over 31 final placements: 17 IBM, four NG45, and ten
synthetic designs. All 31 inferred-contract finals pass. The cohort contains
16 high-, 11 partial-, and four low-coverage rows; three use explicit path
tags and 28 use inferred connectivity. The tightest observed final margins
are:

| Component | Minimum margin | Tightest design |
|---|---:|---|
| cluster compactness | 0.00133 | `ibm18` |
| worst-cluster spread | 0.00137 | `ibm18` |
| neighbor impurity | 0.00211 | `ibm11` |
| hierarchy-edge stretch | 0.00490 | `ibm04` |
| owned-soft distance | 0.00506 | `ibm03` |
| bridge-soft distance | 0.00124 | `ibm08` |

The production slacks remain unchanged. Lowering relative slack from 15% to
10% makes the accepted `ibm18` final fail worst-cluster spread and rejects the
selected `ibm07` seed on edge stretch. Keeping 15% relative slack while
reducing every absolute allowance by 20% makes the accepted `ibm08` and
`ibm11` finals fail and rejects NG45 `nvdla`'s selected seed. Loosening the
limits has no supporting failure and would weaken the contract.

The synthetic runner now preserves generator cluster labels and emits an
independent `hierarchy_truth_audit`. The single-component inference refinement
now passes all ten truth cases without changing the production slacks. The
previously failing `syn03_sram` case recovers all four truth groups exactly and
improves proxy `4.3964 -> 4.3257`. The scalar hierarchy-first selector remains
default-off because its focused proxy regression was too large.

The exact-prescored seed portfolio now also contains a deterministic
constraint-graph legalization of `initial.plc`. The ordinary initial seed
remains available and the graph candidate advances only when it passes every
component limit and its exact proxy is lower.

### 2. Use attributable telemetry for scheduling (complete)

Status on 2026-07-18: the attribution and rollback-accounting patches are
complete. Schema v2 distinguishes proposed from retained proxy/accepts, records
rollback reason, component violations, discarded gain and scorer rebuild time,
and attaches both the committed revision and a deterministic scoped dirty-
worktree fingerprint. Seed/cache/coldspot/exact-score/final-audit stages are
timed separately. The analyzer can filter by fingerprint, prints stage timing,
and reports gain-per-score as `n/a` when a pass made no exact-score calls.

A clean compound-move control/off A/B kept the production pass: control and off
both rounded to `AVG 1.1468`, and off saved 6.66 seconds, but it regressed
ibm12 `1.3060 -> 1.3065` and ibm16 `1.1637 -> 1.1641`. The pass therefore
remains enabled. Rollback evidence instead justified a cheap hard-relocation
containment prefilter. It rejected 654 hierarchy-ineligible candidates before
exact scoring, left all 34 region-hard-relocation pass records rollback-free,
and improved
the full result from `AVG 1.1468 / 404.09s` to
`AVG 1.1412 / 404.01s`, with 17/17 valid and all final audits passing.

Resolved on 2026-07-18. The high-volume region, interleaved, plateau,
compound, and strong/medium repair operators now receive deterministic
remaining-work quotas. Repeated region rounds share their pass quota, and
hard-hard, hard-soft, and soft-soft swaps share one regional-swap quota.
Candidate order and commit order are unchanged: the implementation truncates
the next exact-score batch at the quota and retains wall-clock deadlines as
emergency guards.

The accepted exact-score ceilings are:

| Pass | Limit |
|---|---:|
| region hard relocation | 2,600 |
| region soft relocation, shared across rounds | 24,000 |
| interleaved soft repair | 4,096 |
| region swaps, shared by all swap types | 72,000 |
| region-swap graph fallback | 100 |
| first/post plateau escape | 5,000 / 7,000 |
| compound soft relocation | 60 |
| strong/medium soft repair | 40,000 / 2,048 |

The full IBM validation preserved every score from the preceding accepted
reference at `AVG 1.1412`, 17/17 valid, zero overlaps, and all final audits
passing, while runtime moved from 404.01s to 398.57s. No late pass was skipped
for lack of quota. Only interleaved soft repair reached its 4,096 ceiling, on
ibm11 and ibm17; both already performed exactly 4,096 evaluations in the
uncapped attributable reference. NG45 likewise preserved every score at
`AVG 0.7123`, 4/4 valid, in 75.90s; no quota exhausted.

A deliberately aggressive profile proved that the limits actively bind and
that later operators still execute after exhaustion. It improved ibm10 from
1.1348 to 1.0990 but regressed ibm11 from 1.0122 to 1.0436, so it was rejected.
Production therefore uses measured maxima with modest headroom rather than
pruning accepted search work.

The ordinary post-swap soft pass remains skipped after two attributable full
suites produced zero gain in 34 runs. Its time remains deadline/final-audit
headroom. Continue using `scripts/analyze_plateau_telemetry.py --quotas` and
retained-yield evidence rather than scheduling from isolated runs.

### 3. Keep retired learned ranking out of production

The former relocation, regional-swap, and coldspot learned rankers failed to
clear offline and closed-loop gates and repeatedly increased runtime. Their
model loader, inference hooks, candidate logger, training scripts, diagnostics,
tests, active schemas, historical datasets, and model artifacts were removed on
2026-07-16.

Next step: improve deterministic proposal generation and exact-score efficiency.
Do not rebuild the learned-ranking stack without an explicit direction change
and evidence that a new target provides information beyond the existing
proposal score.

### 4. Expand inferred hierarchy coverage conservatively (partially resolved)

Resolved for the concrete `syn03_sram` failure on 2026-07-18. When flat
connectivity collapses at least 90% of hard macros into one component, the
model now partitions hard macros by strong cosine similarity of their shared
low-fanout soft affinity. Tiny fragments join their strongest positive-affinity
group; a strict partial hard-graph cut is the fallback when affinity is
inconclusive. This is still labeled inferred evidence and does not promote a
flat community to an explicit IP tag. Multi-component IBM graphs retain the
existing component-local bridge-soft rule, while explicit NG45 path tags still
take precedence.

Reference handling is topology-aware. A legal raw initial placement remains
the immutable reference only if legalization satisfies its raw limits, which
prevents double slack. If the raw placement is illegal, grouped DREAMPlace is
the reference instead; this preserves the seedless case's useful basin. Exact
candidate-level vector guards are limited to refined graphs of at most 64 hard
macros, while larger graphs use pass checkpoints and final rollback.

The latest synthetic rerun, with deepest-child boxes enabled, reached `AVG
1.4193`, 10/10 valid, zero overlaps, and 10/10 truth passes, versus the
attributed `AVG 1.4262` run with nine truth passes. `syn03_sram` moved from
purity `0.375` / pair precision `0.271` to `1.0 / 1.0`; `syn04_dense` also
recovers its six groups exactly. Remaining work is to improve the still-coarse
partitions on cases such as `syn01_wide` without weakening the exact contract
or regressing general proxy.

One shallow hierarchy level is now resolved for production. Explicit path
partitions retain their nearest useful ancestor; existing connectivity splits
retain the original component; otherwise an eligible active cluster receives
at most one strict graph bisection. That fallback now requires direct hard or
shared-soft structure, reinforced by initial macro proximity, local
macro-area density, and placed low-fanout wire demand. Geometry cannot create
an edge, and the split must clear raw-cut, compactness-gain, and confidence
gates. The active DREAMPlace groups do not change, and discovery never recurses.
A bounded pass relocates children or swaps sibling slots inside their parent,
co-moves owned soft macros, and can legalize only the affected child set when
rigid geometry is blocked. Child and parent contracts run before exact mixed-
group scoring and become authoritative downstream only after a retained child
move.

The deepest retained children now also receive fixed internal relief boxes:
current footprint plus a congestion/density/graph-pressure margin, optionally
expanded toward graph-favored cold components and clipped to the parent. Hard
and owned-soft relocations and same-child hard swaps stay inside those boxes.
The accepted 0.0005 floor retained no IBM deep moves while exact-scoring 528
states and preserved AVG 1.1412. A 0.0001 calibration retained six locally
improving states but regressed the final suite to 1.1453 by activating the
tighter downstream multilevel contract; it is rejected.

The final IBM sweep inferred 23 spatial parents / 46 children on ten designs,
retained no child moves, and reproduced all scores exactly; the pass exact-
scored 24 states. NG45 retained one localized child move on `ariane136`,
improving `0.7298 -> 0.7291` and the suite `0.7123 -> 0.7121`.
The rejected `0.00005` local-gain floor accepted a tiny ibm08 move and later
regressed that design by 0.0053 after activating the tighter contract. The
production 0.0001 whole-child floor rejects it. Further recursive hierarchy
inference and promotion of low-confidence flat communities remain intentionally
out of scope; the deepest-child boxes add search room without adding another
inferred partition level.

Explicit slash-separated soft instance paths now form high-confidence bundles
and take precedence in compound relocation. Flat IBM `Grp_*` names expose no
such paths, so production behavior is unchanged there. Conservative mutual-edge
soft connectivity communities are now derived diagnostically and scored against
common owned/bridge affinity. Only explicit high-confidence path evidence is
eligible to move as a compound bundle; flat-netlist inferred communities remain
medium or low confidence and unbundled. The first attempt to promote inferred
communities changed `ibm11` from 1.0085 to 1.0087, so it was rejected.

### 5. Exact scoring remains the runtime bottleneck

Large grids make exact validation expensive, and CPU contention can multiply
score time. The placement flow must keep a running maximum score estimate and
reserve enough time for the final score and audits. New operators should first
prove that their expected gain pays for their exact-score calls.

Hard-hard and hard-soft swap trials now share exact compiled batch kernels,
joining the existing batched soft relocation and soft-soft swap paths. Direct
scalar parity checks pass to floating-point roundoff without changing committed
scorer grids or caches. Region swaps now score four-candidate hard-hard,
eight-candidate hard-soft, or twelve-candidate soft-soft stable prefixes before
the untouched remainder; a prefix winner safely avoids the suffix. Hard
legality is evaluated only after the ranked 16/48-candidate cut, and disabled
graph paths avoid zero-array construction. Disposable congestion grids are now
partitioned in place, and static hard separation matrices are computed once
per swap schedule. Together these preserve all 66,703 avoided exact IBM swap
evaluations and reduce attributed swap time by 12.93s without changing any
winner. The
remaining bottleneck is repeated full exact scoring, especially the evaluator's
final large-grid report and operators that must choose the best candidate over
a complete batch rather than the first acceptable one.

The per-batch congestion tail reduction is exact in-place NumPy; the density
tail reduction and hierarchy vector's nearest-four impurity selection use
cached CPU Numba kernels. They preserve the
scalar and stable-sort references exactly, remove avoidable local Python and
N-by-N sort work, and do not replace the required final exact scorer.

### 6. Portability coverage is still narrower than challenge coverage

The pinned DREAMPlace bootstrap and native-op preflight make the supported
CUDA 12.1/GCC 11/Python build reproducible. Other CUDA architectures and Python
ABIs still need an explicit rebuild. The EDA I/O path supports converted
LEF/DEF/Verilog inputs by attaching their generated source directory, but broad
real-design parser coverage remains a validation task rather than a claimed
guarantee.

## Maintenance Rules

- Keep the production path hierarchy-only.
- Do not restore deleted proxy-only operators or archived research scripts.
- Keep deterministic structural signals inside existing hierarchy operators
  and gates. Do not restore learned ranking without explicit direction.
- Record accepted full-suite numbers in `PROGRESS.md`.
- Keep `ARCHITECTURE.md` and `DESIGN_FLOW.md` synchronized with active code.
