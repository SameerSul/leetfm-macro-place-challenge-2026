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
AVG 1.1404  17/17 VALID  0 overlaps  318.55s
```

All final hierarchy audits passed. This historical result included the
hierarchy-blind constraint-graph seed removed on 2026-08-11. Two stable swap prefixes preserve winner order and
logical quotas while raising avoided exact work from 66,703 to 79,466; the
trace-compatible region phase fell `104.04s -> 98.74s`. Batched exact soft
wirelength prefiltering rejected 100,831 proposals before field scoring and cut
the four main soft-relocation phases by 20.7–22.3%. Exclusive telemetry covers
at least 99.86% of each IBM API call: 297.33s was inside the placer and 21.22s
was evaluator loading/final scoring. NG45 remains `AVG 0.7121`, 4/4 VALID, zero
overlaps, and all audits passed in 64.80s. Synthetic validation is `AVG 1.4192`,
10/10 valid, with 10/10 truth passes.

The latest exact-equivalent scorer sweep compiles pair incident-net unions and
reuses sparse swap-reducer scratch, reducing the same-work region phase
`98.74s -> 94.37s`. Soft relocation now uses stable integer grid IDs and
capacity-grown dense field workspaces with fused in-place congestion reduction.
The verification sweep reproduced IBM `AVG 1.1404`, NG45 `AVG 0.7121`, and
synthetic `AVG 1.4192`; all 96 tests pass. Its 330.75s IBM wall time is not
treated as an end-to-end gain because broad host/runtime variance exceeded the
focused operator effect.

The next accepted same-work refinements prepare each multi-prefix swap source
once (`94.37s -> 94.29s`) and compile soft grid conversion, clipping, mask
filtering, symbolic keys, and stable stamp deduplication. The latter reduced
the five measured soft-relocation phases `74.039s -> 73.400s`, including
region-soft relocation `38.431s -> 37.916s`, with identical candidate and exact
score counts. Revision-scoped exact caches, compact swap delta grids, and a
fused soft prepare/score/revert API all regressed their attributable phases and
were removed. This leaves the same fundamental open issue: exact field scoring
still dominates after low-risk preparation overhead has been reduced.

The learned-ranking stack has been removed. Candidate ordering is deterministic;
exact proxy, hard legality, bounds, fixed-macro immobility, hierarchy regions,
and hierarchy-quality gates remain authoritative.

The 2026-08-09 video-driven candidate addresses IBM10's repeated whole-lane
rollback. Complete hierarchy checks now run on exact-improving micro-shift
winners, preserving the valid sequential prefix; direct soft-role fanout rises
from 8 to 16, proxy-banded seed selection prefers contract headroom, and weak
inferred child/deep search requires confidence 0.65. Focused IBM10 improved
`1.1348 -> 1.0590`, remained VALID, passed every audit with no rollback, and
raised soft coverage `39.56% -> 40.31%`. A bounded ordinary-leaf assembly pass
now jointly compacts 2-8-hard clusters with their owned movable softs and tests
whole-leaf slot exchanges. IBM10 retained one seed-stage compaction, improving
final hard hierarchy quality `6.11996 -> 4.84244` and composite `0.29896 ->
0.29555`; exact proxy changed `1.0590 -> 1.0600` (+0.096%). The post-relief
replay and slot lane retained no IBM10 moves, so richer connectivity-derived
slot proposals remain open work. A full IBM/NG45/synthetic sweep is
still required before replacing the established suite references above.

The follow-up colour-contiguity candidate promotes high-confidence leaves from
global-average audit terms to per-colour hierarchy islands. It freezes compact
post-assembly hard/owned-soft boxes, independently limits leaf spread, bounding
span, and local colour impurity, and selects the hierarchy-first eligible seed.
The full IBM sweep is `AVG 1.2951`, 17/17 VALID, zero overlaps, all audits
passing, in 456.05s. Compared with the proxy-first island control, fragmented
leaves fall `229 -> 194`, foreign intrusions `18,841 -> 9,130`, and mean hard
hierarchy quality `1.24789 -> 0.99651`; proxy worsens `1.1975 -> 1.2951`.
This confirms that the evaluator's density/congestion proxy conflicts with the
requested compact colour hierarchy. The accepted `AVG 1.1404` score remains the
comparison baseline rather than being replaced by this hierarchy-first result.

The current follow-up keeps those hierarchy levels fixed and targets congestion
inside each leaf. A net-derived internal floorplanner puts strongly
connected hard macros together, assigns externally connected macros to facing
leaf boundaries, leaves an inset routing channel, and pulls directly owned soft
macros toward their hard-affinity barycentres inside frozen island boxes.
The first pre-freeze schedule was rejected: despite `0.00639` summed local
gain, its full suite regressed `AVG 1.2951 -> 1.2977` because IBM02's small
`0.000415` win displaced larger swap and soft-repair gains, ending that design
`+0.1001` worse. The operator now runs as a final survivor inside frozen island
boxes. Focused checks restore IBM02 exactly to `1.2303` and improve IBM10
`1.7511 -> 1.7494`; both are VALID and audit-clean. The promoted late schedule
reaches `AVG 1.2949`, 17/17 VALID, zero overlaps, and all audits passing in
487.54s. Three retained layouts sum to `0.002001` attributable exact gain;
fragmented protected leaves improve `194 -> 193`, foreign intrusions
`9,130 -> 9,105`, and mean hierarchy composite is effectively flat
(`0.140344 -> 0.140363`). Congestion still contributes `0.8392` to the average
proxy, versus `0.3836` density and `0.0721` wirelength, so further work should
continue to target congestion inside the immutable hierarchy envelope.

The exact-tail diagnostic
`uv run python test/diagnostic/analyze_final_hotspots.py --all` reran all IBM
designs at AVG 1.2950, 17/17 valid, and localized the remaining cost. Across
the worst congestion component per design, exact implicated demand is 70.18%
unassigned-soft/IO, 25.03% internal to one inferred cluster, and 4.79%
cross-cluster. Worst density components are 83.61% soft area. Thus boundary
ports between hierarchy leaves are not the suite-wide primary blocker; the
largest gap is low soft-role coverage, especially IBM18 (4.66%) and IBM17
(16.97%). IBM08/03 remain strong internal-cluster cases, while IBM09/10/16 need
separate macro-blockage channel treatment. The six primary score targets
IBM10/12/14/16/17/18 account for 46.0% of suite congestion and 48.0% of suite
density. Detailed coordinates, nets, macros, and recommended lanes are in
`ml_data/hotspot_analysis/20260809T215011-findings.md`.

`test/diagnostic/analyze_soft_role_coverage.py` originally explained the low
direct coverage. Only 5,364/21,538 soft macros received a direct role. Of the
16,174 then-unassigned macros,
14,842 (91.76%) have soft/IO-only connectivity with no hard macro on any
multi-pin net, 1,291 (7.98%) connect only to unclustered hard macros, and just
41 (0.25%) reach clustered hard macros exclusively through nets above fanout
16. There are zero qualifying-but-unassigned cases. Raising the fanout cap is
therefore was not the remedy. The bounded confidence-tier implementation now
adds 8,044 hop-one and 5,222 hop-two roles for 86.50% anchored coverage. Eight
stable residual soft-only groups add 29 macros without promoting them to hard
ownership, for 86.63% total hierarchy coverage. The remaining 2,879 macros
still stay outside hierarchy; most lack the repeated, stable structural evidence
needed to distinguish an IP block from incidental flat-net connectivity.

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

The exact-prescored seed portfolio retains the hierarchy-aware ordinary initial
seed, whose legalization is cluster-consecutive and connectivity-pressure
ordered. Grouped DREAMPlace and its two recurrent hierarchy prototypes remain.
The hierarchy-blind constraint-graph seed and the four zero-selection blend,
radial-expansion, and synthetic-clearance candidates were deleted; post-hoc
contract eligibility is not considered hierarchy optimization, and candidates
with no measured selection yield do not retain production overhead.

The post-deletion IBM sweep reached `AVG 1.2151`, 17/17 VALID, zero overlaps,
and all final audits passing in `838.30s`. `initial` selected on 11 designs,
including IBM10/11/12/15 where the removed candidate had previously selected;
the two recursive seeds selected on four designs and grouped DREAMPlace on two.

An original Re²MaP-inspired hierarchy-leaf B*-tree seed was subsequently
tested and removed. It selected on 0/17 IBM designs, added 17.75s to the full
sweep, and provided no attributable final-score improvement. Its diagnostic
constructor, production hook, constant, and dedicated tests were pruned.

The active portfolio also contains two non-mandatory Re²MaP-inspired recurrent
prototype seeds. Before the hierarchy-blind constraint candidate was removed,
the full IBM sweep selected them on IBM01/02/07/08 and reached historical
`AVG 1.1625`, 17/17 VALID, zero overlaps, with every final audit passing in
`1065.11s`. Mean density/congestion were `0.630059 / 1.539235`. NG45 reached
`AVG 0.7233`, 4/4 VALID, zero overlaps, and its explicit-tag verifier passed;
`nvdla` selected recursive round two. The complete dirty-tree synthetic run was
10/10 VALID with zero overlaps but only 9/10 truth passes: `syn07_ports` missed
worst-spread and impurity limits after selecting the now-removed
`synthetic_clearance` seed. That result is retained as historical attribution;
the current single-component refinement passes all ten truth cases.

### 2. Use attributable telemetry for scheduling (complete)

Status on 2026-07-18: the attribution and rollback-accounting patches are
complete. Schema v2 distinguishes proposed from retained proxy/accepts, records
rollback reason, component violations, discarded gain and scorer rebuild time,
and attaches both the committed revision and a deterministic scoped dirty-
worktree fingerprint. Seed/cache/coldspot/exact-score/final-audit stages are
timed separately. The analyzer can filter by fingerprint, prints stage timing,
and reports gain-per-score as `n/a` when a pass made no exact-score calls.
The 2026-07-19 outer-boundary follow-up adds five mutually exclusive phases,
`hierarchy_floorplan_total`, and `placer_api_total`. Analyzer `--coverage`
reconciles at least 99.86% of every IBM API call; the accepted sweep attributes
202.38s to hierarchy search, 37.35s to setup, 27.22s to coldspot, 23.48s to seed
selection, and 6.46s after coldspot. The evaluator adds 21.22s outside the
297.33s submission API total, so missing end-to-end attribution is resolved.

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

The latest synthetic rerun reached `AVG 1.4192`, 10/10 valid, zero overlaps,
and 10/10 truth passes, versus the
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

Large-macro corridors are now used by a bounded final survivor pass rather
than being treated as generic cold windows. The pass found retained,
contract-safe whole-leaf moves on IBM10 (`1.7205 -> 1.7165`) and IBM13
(`1.1055 -> 1.1054`) in the 2026-08-10 full sweep. A locally improving IBM18
move (`1.7056 -> 1.7051`) was correctly removed by the independent final audit.
The follow-up now adds hard-clear canvas-edge pockets and standalone soft
occupants. Stable residual bundles keep their membership; additional movable
residual softs form pass-local low-fanout routing cohorts or singleton fallbacks
without acquiring hierarchy ownership. Exact rectangle occupancy replaces the
former mean-area estimate for these soft lanes, and layouts must avoid hard and
internal cohort overlaps. The full sweep reached `AVG 1.2832`, 17/17 VALID,
zero overlaps, and all hierarchy/island audits passing. Interior hard moves
retained on IBM10/13, edge hard moves on IBM08/11/18, and residual-soft fills on
IBM01/11/12/14, for 0.005968 stage-local exact gain. IBM04 remains unchanged at
`0.9964` after all 30 soft units fail the field/exact opportunity gates.

The focused visual follow-up addresses the remaining rigid-motion limitation.
For clusters owning a void boundary macro, it shifts only the hard/owned-soft
band aligned with that clear rectangle, backing off until output-precision hard
legality passes. Expansion wins receive first-winner priority before disjoint
rigid moves. IBM10 retained one left-edge expansion and its prior rigid leaf
move, improving `1.7205 -> 1.7157` with both audits passing. Visibility-aware
selection moves `a60088` and owned soft `Grp_613` outward by `1.504 µm`. Full-suite
promotion is still required.

Explicit slash-separated soft instance paths now form high-confidence bundles
and are the only source of compound relocation groups. Flat IBM `Grp_*` names
expose no such paths, so their softs stay independent. Conservative mutual-edge
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
scorer grids or caches. Region swaps now score two four-candidate hard-hard,
two eight-candidate hard-soft, or two twelve-candidate soft-soft stable prefixes
before the untouched remainder; a prefix winner safely avoids the suffix. Hard
legality is evaluated only after the ranked 16/48-candidate cut, and disabled
graph paths avoid zero-array construction. Disposable congestion grids are now
partitioned in place, and static hard separation matrices are computed once
per swap schedule. The current scheduler avoids 79,466 exact IBM swap
evaluations and completes the same-work phase in 94.37s without changing
any winner outside IBM09's earlier seed repair. Batched soft wirelength deltas
also remove Python scalar prefilter overhead before exact field work; stable
integer target IDs and reusable dense scorer workspaces remove additional
preparation and allocation overhead. The latest verification spent 218.61s in
hierarchy search and 317.63s in the placer API, but that run is retained as a
correctness/same-work result rather than a wall-time baseline. Region swaps
remain the dominant measured phase, followed by region-soft relocation and the
evaluator's final large-grid report outside the placer API.

Swap congestion/density tails and ordinary soft density tails use cached CPU
Numba scratch. Ordinary soft congestion overwrites disposable route grids with
exact smoothed values and reduces the tail in the same compiled call; hierarchy
nearest-four impurity is also cached CPU Numba. They preserve the
scalar and stable-sort references exactly, remove avoidable local Python and
N-by-N sort work, and do not replace the required final exact scorer.

The tested optimistic congestion lower bound is not an open production hook:
it rejected only 317/26,556 IBM10 soft-soft rows and was removed. Future
cross-source batching must prove dependency-safe reuse after sequential commits;
snapshot-only waves that alter first-winner behavior remain out of scope. The
tested revision-scoped caches are also closed experiments: 16,265 swap hits and
15,094 soft hits did not amortize Python key/value overhead on the measured
small batches.

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
