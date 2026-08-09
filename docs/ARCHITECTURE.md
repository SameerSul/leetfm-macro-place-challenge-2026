# VivaPlace v2 — Architecture

## Overview

`MacroPlacer.place()` always routes through `_hierarchy_floorplan()` in
`src/placer/pipeline/macro_placer.py`. There is no proxy-only fallback path:
the placer raises if grouped DREAMPlace is unavailable.

```text
benchmark input
  -> build HierarchyModel
       - infer hard-macro clusters from connectivity (or RTL instance-path
         prefixes when names provide useful coverage, e.g. NG45)
       - refine a nearly all-covering single flat component from shared
         low-fanout hard-to-soft affinity, with a strict graph-cut fallback
       - retain exactly one parent/child level without recursive discovery:
         use the nearest useful explicit path ancestor, retain an original
         connectivity component above existing split leaves, or bisect an
         eligible active cluster once from structural edges reinforced by
         initial placement proximity, local density, and placed wire pressure
       - classify soft macros as owned (one dominant cluster) or bridge and
         record independent repeated-support/ambiguity confidence
       - preserve explicit soft instance-path bundles when names expose them
       - record inter-cluster edge weights and confidence
  -> grouped DREAMPlace global placement (synthetic clique nets per cluster)
  -> cluster-consecutive hard legalization
  -> exact-proxy seed portfolio selection
       (grouped DREAMPlace, legalized initial.plc, DP/initial blends,
        radial expansion, synthetic-clearance push-apart, and a
        constraint-graph-legalized initial.plc; every candidate gets a complete
        hierarchy vector, the vector is constrained component-by-component
        relative to legalized initial.plc, except that a legal raw reference is
        retained for single-component affinity refinement and an illegal raw
        reference falls back to grouped DREAMPlace; a mandatory lower-proxy
        candidate that misses one component may be repaired toward the passing
        reference when at least 95% of its displacement survives; the lowest
        eligible exact proxy enters relief)
  -> congestion-expanded hard/soft hierarchy regions
  -> exact-gated local micro-shift polish
  -> region-locked hard relocation + soft cleanup
       - reject hard candidates that exceed the selected seed's inexpensive
         hard-containment limit before exact batch scoring
       - keep the complete six-component checkpoint authoritative after the pass
       - on refined graphs with at most 64 hard macros, reject individual hard
         and soft candidates that fail the complete contract before exact scoring
  -> parent-bounded child-cluster search
       - keep the production DREAMPlace/leaf partition unchanged
       - rigidly translate hot child groups toward cold space inside the parent
       - co-move every leaf-owned movable soft macro; bridge softs stay independent
       - test sibling slot swaps inside the same parent
       - when a rigid state is blocked, compact and legalize only the affected children
       - enforce child and parent hierarchy contracts before exact mixed-group scoring
       - activate the multilevel contract for later passes only after a move is retained
  -> deepest-child bounded internal relief
       - freeze a box around each current child footprint plus a per-child margin
       - blend congestion heat, density heat, and inter-child graph tension into the margin
       - expand hot boxes toward cold components favored by graph corridors, then clip to parent
       - relocate individual hard/owned-soft members and swap hard members only inside the box
       - use neighboring-child graph centroids for target guidance and exact-score at most 48 states
       - require 0.0005 local gain before activating the downstream multilevel contract
  -> exact-gated cluster decompression with composite hierarchy quality
       - large designs can order opportunities by hierarchy graph tension
  -> budget-aware interleaved soft repair
  -> region-bounded hard-hard / hard-soft / soft-soft swaps
       - hard-moving swap candidates must stay inside the hierarchy audit budget
       - exact-score two stable prefixes before the untouched suffix
       - optional micro-shift replay after each swap round
  -> post-swap micro-shift replay
  -> post-swap hard propose-all relocation with spare-budget additive candidates
  -> telemetry scheduler skips the duplicate ordinary post-swap soft pass
       - two clean attributable full suites produced 0 accepts in 34 runs
       - its time remains as deadline and final-audit headroom
  -> plateau-triggered compound related-soft relocation
       - form groups only from explicit high-confidence path bundles
       - keep every member inside its individual hierarchy region
       - preserve group-relative geometry while testing pair/quartet/full-group shifts
       - enforce the six-component hierarchy contract before scoring
       - exact-score and accept only the completed multi-soft state
  -> plateau- and component-aware strong soft repair when telemetry shows useful spare work
       - medium/large soft continuation runs only when structural shape and prior soft gain justify it
  -> coldspot tightening:
       - refresh current congestion field and cold-cell graph memory
       - generate coldspot kick candidates
       - optional default-off hard-only ego-net candidate groups move small graph neighbors
       - co-move owned/bridge soft macros
       - legalize candidate hard macros
       - expand local graph border through adjacent open cold cells
       - apply hard-core padding
       - run graph-bordered local swaps and hard/soft relocations
       - exact-rank refined outcomes with deterministic graph tie-breaks
       - exact proxy + hierarchy-quality gate before commit
       - large designs can rank hot clusters by hierarchy graph tension
  -> graph-local fallback when no coldspot kick commits:
       - select hottest eligible current clusters
       - reuse the same graph-expanded border
       - run the same swaps and relocations without a kick
       - exact proxy + hierarchy-quality gate before commit
  -> post-coldspot micro-shift replay
  -> structurally eligible small-design polish:
       - seed release candidates with weakest-k inferred hierarchy clusters
       - keep only clusters below the confidence threshold
       - release the hottest eligible weak clusters, capped by max clusters and weakest-k
       - no-release low-net small designs shift candidate breadth toward soft relocation and soft-involving swaps
       - build cold connected-component target pools
       - run bounded hard/soft relocation, hard swaps only after useful released hard relocation, soft-involving swaps, and micro-shift polish
       - restore the best audit-passing exact-scored state seen inside the small-design pass
       - exact proxy, hard legality, and hierarchy audit budget remain the commit gates
  -> gain-controlled passes: stop repeats when latest exact gain <= HIER_PLATEAU_PROXY_GAIN
  -> final scorer-compatible hard legality margin audit
  -> final hierarchy-quality audit against the selected hierarchy seed:
       - enforce both the legacy hard-cluster budget and independent limits for
         all six rich-vector components
       - roll back to the best saved audit-passing checkpoint when needed
  -> final legality and bounds checks
  -> return center coordinates for hard and soft macros
```

Passes advance on gain, not fixed repeat counts: each stage keeps running
while its most recent exact-proxy improvement exceeds
`HIER_PLATEAU_PROXY_GAIN`, then moves on.

There are no Boolean switches around promoted production behavior. BB and
DREAMPlace cache reads, component-aware expansion/decompression, decompression
feasibility and survivor handling, graph-mask fallback, adaptive gain control,
cold-component targets, structurally eligible small/medium soft polish, final
audit rollback, and plateau telemetry always run when their data, structural,
budget, and safety preconditions apply. Default-off research hooks remain
separate experiments. The former `HIER_DREAMPLACE_BB`,
`HIER_DREAMPLACE_CACHE`, `HIER_ADAPTIVE_PASSES`, `HIER_PLATEAU_TRACE`, and
`HIER_PLATEAU_TRACE_BUFFERED` feature switches are not read by production code;
legacy values cannot disable the selected behavior.

```text
proxy_cost = wirelength + 0.5 * density + 0.5 * congestion
```

The exact proxy is still the accept gate for every committed move, and still
drives evaluator reporting. But the system optimizes for **hierarchy
preservation** first: it deliberately keeps connected subsystems together
even when a flatter, more-spread placement would score lower proxy. The
structural reasoning behind this is in [OBJECTIVES.md](OBJECTIVES.md).

The structural objectives that drive the hierarchy flow are documented in
[OBJECTIVES.md](OBJECTIVES.md). The retired learned-GNN stack is preserved only
as historical experiment results in [PROGRESS.md](PROGRESS.md); its inference,
candidate tracing, offline training, artifacts, and dedicated documentation
are no longer part of the system.

Current verified full sweep with strict hierarchy-audit rollback,
audit-aware hard swap gating, component-aware region expansion/decompression,
large-design hierarchy graph-tension opportunity ordering, swap-round
micro-shift replay, stronger opportunity gates, component-aware scheduling,
post-coldspot small-design polish with subpass audit restore, no-release
low-net soft/SS breadth, medium/large soft-continuation scheduling, prepared
Numba routing/legalization kernels, exact batched hard-hard/hard-soft scoring,
batched soft relocation/swap scoring with direct global swap topology and exact
sparse congestion/density-tail reduction,
stable nearest-four hierarchy-audit selection, the guarded constraint-graph seed, and
the per-component seed/final hierarchy contract, legalized-reference seed
prefiltering, hierarchy-prefiltered hard relocation, and deterministic
exact-score work quotas, one-level child relocation/sibling swaps, and
graph/field-derived deepest-child relief boxes, two-stage stable-prefix region-
swap exact scoring, batched soft wirelength prefiltering, calibrated soft-role
confidence, and audited retained-gain late-lane stopping:

```text
uv run evaluate src/main.py --all
AVG 1.1404  17/17 VALID  0 overlaps  318.55s
```

The prior proxy-leaning hierarchy sweep reached `AVG 1.1627`, 17/17 VALID,
0 overlaps, 1116.90s, but final hierarchy audit was report-only and failed on
several designs after late proxy-improving relief. A strict final-rollback-only
audit sweep reached `AVG 1.1999`; the audit-preserving local-relief recovery
reached `AVG 1.1664`; the pre-optimization BB-on verification was
`AVG 1.1653` cold (`AVG 1.1652` with cache hits). The prior optimized
normal-cache sweep was `AVG 1.1575`; the constraint-graph/batched-swap sweep
was `AVG 1.1199`, the component-contract sweep was `AVG 1.1205` in `540.33s`,
and the compound-relocation sweep was `AVG 1.1205` in `544.94s`.
The learned-GNN removal sweep preserves every telemetry-scheduled production
score at `AVG 1.1205` in `542.58s`; its candidate rankers, selectors, tracing,
offline tooling, tests, and active schemas are gone.
The later conservative soft-bundle inference sweep again preserved every proxy
at `AVG 1.1205` in `547.05s`: only explicit soft instance paths change
compound-relocation groups, while flat-net connectivity and hard-affinity
communities remain diagnostic evidence.
The current batch-density/nearest-four-audit JIT validation again preserved
every proxy at `AVG 1.1205` in `554.54s`. That full-suite wall time is a
loaded-host observation, not a claimed end-to-end speedup; isolated kernels
are measured in [PROGRESS.md](PROGRESS.md).
Subsequent accepted 2026-07-18 scheduling, coldspot, incremental-scoring, and
region-expansion changes established an active-root reference of `AVG 1.1487`
in `409.72s`. Correcting the immutable contract reference to the actually
legalized `initial.plc` improved the control to `AVG 1.1468` in `404.09s`.
The hard-relocation containment prefilter then improved the active result to
`AVG 1.1412` in `404.01s`, with all 17 final component audits passing.
Deterministic exact-score quotas preserved every placement and score from that
reference while the observed runtime moved to `398.57s`.
The subsequent single-component hierarchy refinement is structurally dormant
on the audited multi-component IBM graphs. Its full IBM validation reproduced
all 17 accepted scores exactly at `AVG 1.1412`, with 17/17 VALID, zero overlaps,
and all audits passing in 413.49s; the preceding deterministic-quota reference
recorded the same placements in 398.57s. `ibm10` also reproduced `1.1348` VALID
in 26.57s (the preceding repeat was 25.13s). Full synthetic validation reached
`AVG 1.4206`, 10/10 VALID, zero
overlaps, and 10/10 independent truth audits in 338.6s, versus the attributed
`AVG 1.4262` reference with nine truth passes.
The one-level hierarchy pass then preserved all 17 IBM placements at `AVG
1.1412`, 17/17 VALID, zero overlaps, and all final audits passing in 423.87s.
It formed 418 child candidates, localized 150 blocked variants, exact-scored
55 complete states, and retained none; total operator time was 2.26s. NG45
improved `AVG 0.7123 -> 0.7121` in 76.85s by retaining one localized child
move on `ariane136` (`0.7298 -> 0.7291`), with 4/4 valid and all multilevel
audits passing. Final synthetic validation reached `AVG 1.4204`, 10/10 valid,
zero overlaps, and 10/10 truth audits.
The spatial/structural fallback then replaced topology-only bisection for
eligible active clusters. Direct low-fanout hard edges and shared-soft
support remain mandatory; initial macro proximity, local macro-area density,
and placed low-fanout wire demand only reinforce those relations. A split must
also satisfy the raw cut, compactness-gain, and combined-confidence floors.
The full IBM validation inferred 23 spatial parents / 46 children on 10
designs, exact-scored 24 child states, retained none, and reproduced every
score at `AVG 1.1412` in 430.06s. NG45 remained `AVG 0.7121` in 80.65s with
explicit path parents taking precedence. The synthetic suite reached `AVG
1.4195`, 10/10 valid, zero overlaps, and 10/10 truth audits.
The deepest-child internal phase then added immutable footprint-plus-margin
boxes after whole-child search. The margin is `0.01 + [0, 0.025]` of the larger
canvas dimension from a `0.45/0.35/0.20` congestion/density/graph-pressure
blend; directional cold-component expansion can add up to `0.01`, with graph
corridors biasing component choice. The accepted IBM sweep exact-scored 528
states over 11 participating designs in 2.93s, retained none at the 0.0005
floor, and reproduced every score at `AVG 1.1412` in 433.09s. An exploratory
0.0001 floor retained six locally improving moves but activated the tighter
downstream contract and regressed the final average to 1.1453, so that
calibration is rejected. The independent synthetic sweep reached `AVG 1.4193`,
10/10 valid, zero overlaps, and 10/10 truth-audit passes with the accepted
deepest-child pass enabled.
Region-swap exact scoring then split each stable candidate list into a short
prefix and its untouched remainder. Because this operator commits the first
acceptable candidate, a winner in the prefix makes the remainder irrelevant;
otherwise the remainder is scored in its original order. The IBM sweep avoided
58,820 exact evaluations relative to full per-source batches and reduced
region-swap time from 159.91s to 150.68s. A follow-up calibration increased the
soft-soft prefix from 8 to 12 while retaining 4/8 for hard-hard/hard-soft,
ranked candidates before evaluating hard legality, and removed zero-valued
graph-mask work when that path is disabled. It increased avoided exact work to
66,703 and reduced region-swap time again to 148.29s. All 17 exact scores were
reproduced at `AVG 1.1412`; complete evaluator time was 416.74s, effectively
flat under final-score timing noise. The batched congestion reducer now
partitions its disposable trial grids in place, and one cached pair of hard
separation matrices serves every field, round, and fallback in the region-swap
schedule. The next sweep preserved the same 1,077,431 physical exact scores
and 66,703 avoided scores while reducing attributed swap time to 146.98s;
complete evaluator time remained flat at 416.87s. Swap routing now reads the
global net/pin topology directly and packs only the selected pin cells inside
the compiled candidate loop, eliminating pair-specific topology construction
and flattening. Exact sparse reducers recompute the routing-changed H columns,
V rows, hard-blockage cells, and four density rectangles, then merge them with
the sorted unchanged baseline tail. The accepted sweep preserved all counts,
placements, and scores while reducing attributed region-swap time to 104.04s
and complete runtime to 351.48s. Strong and medium late-soft
lanes expose per-lane proposed/retained gain and stop the remaining rounds after
audit restore or a retained gain no larger than 0.00005. The synthetic suite
reached `AVG 1.4193`, 10/10 valid, zero overlaps, and 10/10 truth passes; one bridge-soft violation
was restored immediately. Prefix-truncating an individual late lane was
rejected because IBM12 finds useful moves beyond ordered source 384.
The exact swap-tail baseline is commit-scoped: congestion values and stable
descending order, density occupancy/order, nonzero count, and sum are built
once, reused across rejected batches, and invalidated by every scorer commit.
The accepted IBM verification preserved all physical/avoided counts and
reduced attributed region time from 104.04s to 102.68s. Focused IBM04/12/18
reductions were 7.6%, 9.3%, and 5.5%; NG45 region time fell 15.41s to 14.62s.
The full IBM wall time was 371.82s under broader run/compile variance, so only
the attributable phase reduction is claimed.
The current priority sweep accepts four further changes. First, a mandatory
lower-proxy seed with exactly one contract violation may be interpolated toward
its passing reference; repairs must remain legal and retain at least 95% of the
source displacement. IBM09 retained 99.61% of its constraint-graph candidate
and improved `1.0122 -> 0.9978`, while weaker IBM03/13 repairs were rejected.
Second, all eligible soft-relocation targets use one exact batched wirelength
prefilter. It rejected 100,831 IBM proposals before field scoring and reduced
the region/interleaved/plateau/strong-soft phases by 20.7–22.3%. Third, swap
scoring tests a second same-sized stable prefix before its untouched suffix,
raising avoided exact evaluations from 66,703 to 79,466 and reducing the
trace-compatible region phase `104.04s -> 98.74s`. Finally, flat owner/bridge
evidence is assigned medium/low confidence and no longer creates compound
groups; explicit path bundles remain the only high-confidence compound source.
The full IBM run reached `AVG 1.1404`, 17/17 valid, zero overlaps, and all
audits passing in 318.55s. NG45 remained `AVG 0.7121` in 64.80s, and synthetic
validation reached `AVG 1.4192`, 10/10 valid with 10/10 truth audits.
The next exact-equivalent scorer sweep compiles each swap pair's sorted
incident-net union directly from CSR arrays and reuses all grid-sized sparse
reducer scratch for the scorer lifetime. The IBM run preserved the same
1,048,385 logical, 1,066,186 physical, and 79,466 avoided scores while reducing
attributed region-swap time from 98.74s to 94.37s. Soft relocation retains
stable integer grid IDs through bounds, region-mask, hierarchy, and duplicate
filtering; only surviving IDs become coordinates. Its accepted dense scorer
reuses capacity-grown routing/density buffers, overwrites disposable H/V route
grids with exact smoothed congestion values, and performs the congestion
top-tail reduction in the same compiled call. Candidate order, scalar cost
semantics, scorer commits, and hierarchy gates are unchanged. The complete IBM
run remained `AVG 1.1404`; NG45 remained `AVG 0.7121`; synthetic remained `AVG
1.4192` with 10/10 truth audits.

The latest exact-equivalent follow-up prepares a multi-prefix swap source only
once. Candidate modules and coordinates, the position snapshot, and the ragged
incident-net union are sliced by the existing stable prefix boundaries instead
of rebuilt for each dispatch. The physical/logical/avoided work and all
placements remain unchanged; attributed region-swap time was `94.37s ->
94.29s`. Soft target filtering now runs grid-ID conversion, clipping, symbolic
keys, region masks, and stable stamp deduplication in cached Numba kernels while
leaving the ordered hierarchy callback unchanged. The identical five soft-
relocation phase workloads fell `74.039s -> 73.400s`; region-soft relocation
fell `38.431s -> 37.916s`. Exact-score caches, compact swap deltas, and a fused
soft transaction API were measured and removed after phase regressions.

The external algorithm lineage and the boundary between accepted code,
research inspiration, and rejected prototypes are documented in
[`REFERENCES.md`](REFERENCES.md#hierarchy-search-acceleration-literature),
entries 21–27. In particular, paper-reported ABCDPlace, GPU-DPO, and Xplace
speedups are not VivaPlace performance claims.

A conservative unchanged-cell congestion lower bound was removed after it
rejected only 1.2% of IBM10 soft-soft rows and added net overhead. Cross-source
speculative waves and net-optimal prefix ranking are likewise not production
paths: dependencies after a first-winner commit require recomputation, and no
exact-safe workload reduction cleared promotion.
The 2026-07-17 SYS_DETRIMENT checkpoint experiment is not part of this
reference: its vector-safe full sweep reached `AVG 1.1564 / 514.27s` and was
rejected for proxy regression despite the lower runtime.
The prior best same-path sweep was
`AVG 1.1657`. The
production path preserves the audit invariant earlier in local relief so fewer
proxy-improving states need to be discarded at finalization. Earlier Stage-6
audit sweeps are retained in `PROGRESS.md` as historical experiment records.

The graph-tension signal is advisory and applies to structurally eligible large designs. It
orders decompression/coldspot opportunities but does not change commit gates.
Direct graph-tension swap ordering remains available through
`HIER_GRAPH_TENSION_SWAP_WEIGHT`, but defaults to `0.0` after focused tests
regressed `ibm08` and `ibm10`.
Swap candidate ranking uses temporary graph-derived masks and soft mask
penalties whenever a graph mask is available:

```text
HIER_SWAP_GRAPH_MASK_MAX_EDGES=0
HIER_SWAP_GRAPH_MASK_PAD_CELLS=1
HIER_SWAP_GRAPH_MASK_PENALTY_WEIGHT=0.30
HIER_SWAP_GRAPH_DELTA_WEIGHT=0.0
HIER_SWAP_GRAPH_DELTA_SAMPLES=9
HIER_SWAP_GRAPH_FALLBACK_BUDGET_S=2.5
```

These controls are diagnostic/ranking only in default mode; no candidate is
rejected for violating the mask, and final commit still requires hard legality,
hierarchy quality, exact proxy gain, and the active audit checks.
Coldspot and decompression use graph-edge candidate deltas internally: weighted
edge stretch, corridor congestion change, weighted edge-length change, and a
combined graph delta. These are deterministic ranking features only; they do
not alter acceptance.
The default-off `HIER_COLDSPOT_GRAPH_DELTA_RANK` hook can use that combined
graph delta during exact coldspot candidate ordering by adding a small
proxy-equivalent penalty for graph-worsening moves before the normal graph-score
tie-break. Focused `ibm10`/`ibm12` tests were valid and audit-passing but did
not improve proxy, so the default weight remains `0.0`.
The default-off `HIER_REGION_GRAPH_COMPONENT_WEIGHT` hook uses hierarchy graph
edge corridors to bias which contiguous cold congestion component a hot region
expands toward. It changes only region construction; local relief still uses
the normal legality, exact-proxy, and hierarchy gates.
The default-off `HIER_COLDSPOT_GRAPH_ANCHOR_WEIGHT` hook keeps congestion as
the primary coldspot anchor signal, then uses the selected cluster's weighted
graph-neighbor centroid to break cold-window ties and near-ties. Candidate
acceptance is unchanged.
Decompression always estimates the proposed bbox's free area and neighbor
blockage before legalization and exact scoring, and logs `feasibility_blocked`
rejects.
The default-off `HIER_DECOMPRESS_GRAPH_RESCUE` hook uses the graph-edge delta
signal to rescue decompression candidates that improve graph geometry but fail
feasibility or hard-overlap legalization. It tries a bounded set of smaller or
cold-component-shifted variants, then returns to the normal hard legality,
hierarchy-quality, exact-proxy, and audit gates. Full-suite validation was
legal but not promoted because the average regressed to `1.1663`.
The graph-survivor path is narrower: for legal,
hierarchy-safe decompression candidates that miss exact proxy by a small amount
while improving graph-edge geometry, it exact-scores a tiny hard/soft local
polish pool around the moved cluster. It commits only if the final candidate
clears the normal exact-proxy gain and audit gates. The pre-optimization
cold-cache sweep was `AVG 1.1653`; the current optimized normal-cache sweep is
`AVG 1.1199`.
The default-off `HIER_GRAPH_PREFILTER` hook can reject low-tension
decompression/coldspot candidates before exact scoring when their cheap local
congestion estimate does not improve. It is trace-visible, but not promoted by
default because focused A/B found `ibm10` better with the filter disabled.
The default-off `HIER_COLDSPOT_EGONET` scaffold can synthesize temporary
small-neighbor hard-only coldspot candidate groups. These groups are candidate
generation inputs only; final acceptance still uses the original hierarchy
quality and audit gates, plus an ego-net-specific exact-gain floor
(`HIER_COLDSPOT_EGONET_MIN_GAIN`).

Current NG45 verification:

```text
uv run evaluate src/main.py --ng45
AVG 0.7121  4/4 VALID  0 overlaps  64.80s
```

Historical proxy-optimized scores in `PROGRESS.md` and `ISSUES.md` (avg
~1.12-1.18) describe a deleted code path — candidate restarts, R2/2-opt,
generic LSMC, ML candidate ranker — that did not preserve hierarchy. They are
not comparable to current numbers.

## Main Components

| Path | Role |
|---|---|
| `src/main.py` | Evaluator entrypoint. Exposes `MacroPlacer`; applies `SEED` only. |
| `src/utils/constants.py` | All tunable hierarchy constants (see appendix below). |
| `src/placer/pipeline/macro_placer.py` | Production flow entrypoint; raises if `_hierarchy_floorplan()` cannot run. |
| `src/placer/pipeline/hierarchy_floorplan.py` | The hierarchy pipeline itself: seed portfolio, region relief, swaps, and coldspot cleanup. |
| `src/placer/pipeline/hierarchy_context.py` | Shared `PlacementState`, `PassContext`, `PassResult`, `PlateauTelemetry` used across pipeline stages. |
| `src/placer/local_search/hierarchy_model.py` | Inferred hierarchy: active clusters, one parent/child level, soft roles, graphs, region builders. |
| `src/placer/local_search/soft_hierarchy.py` | Confidence-calibrated soft bundles: explicit instance paths can be active; flat-netlist connectivity and affinity remain diagnostic evidence. |
| `src/placer/local_search/hierarchy_quality.py` | Complete hierarchy vector, including cached-JIT stable nearest-four impurity selection. |
| `src/placer/local_search/clusters.py` | Hard-cluster derivation, oversized-cluster splitting, region-box primitives. |
| `src/placer/local_search/relocation.py` | Hard and soft relocation used by region-locked relief and post-swap polish. |
| `src/placer/local_search/subcluster_relocation.py` | Parent-bounded child relocation, deepest-child graph/field margin construction, bounded internal relocation, and hard swaps. |
| `src/placer/local_search/hierarchy_swaps.py` | Region-bounded hard-hard, hard-soft, soft-soft swap relief. |
| `src/placer/local_search/cluster_decompress.py` | Exact-gated decompression of hot hierarchy blobs. |
| `src/placer/local_search/region_expand.py` | Expands hot cluster regions toward colder congestion bands. |
| `src/placer/local_search/lsmc_explore.py` | Coldspot kick candidate generation. |
| `src/placer/local_search/fields.py` | Congestion/coldspot fields used by relocation and coldspot tightening. |
| `src/placer/local_search/plateau_telemetry.py` | Buffered schema-v2 pass-yield telemetry with run/revision provenance. |
| `src/placer/scoring/exact.py` | Exact TILOS proxy wrapper. |
| `src/placer/scoring/incremental.py` | Incremental scorer for relocation, swaps, and complete mixed hard/soft group moves, including cached-JIT bbox smoothing and batched density tails. |
| `src/placer/legalize/spiral.py` | Hard-macro legalization, with cluster-consecutive order support. |
| `src/placer/legalize/constraint_graph.py` | Deterministic horizontal/vertical separation-DAG projection for the guarded initial seed. |
| `src/dreamplace_bridge/` | ICCAD04 pb/plc → Bookshelf, cluster grouping injection, DREAMPlace launcher, read-back. |
| `scripts/dreamplace/` | Pinned source/toolchain bootstrap, CUDA-12 CUB patch, and native-extension preflight. |
| `scripts/analyze_plateau_telemetry.py` | Provenance-filtered pass-yield aggregation and conservative skip-candidate report. |
| `src/eda_io/` | Standard EDA file I/O; converts inputs to the same benchmark object. |

## Hierarchy Pipeline

### 1. Cluster Derivation

`HierarchyModel.build()` derives hard-macro clusters from RTL instance-path
prefixes when macro names carry useful coverage (e.g. NG45), otherwise from
low-fanout net connectivity. Oversized bridge-connected flat clusters are
selectively split toward a target leaf size. If flat connectivity instead
collapses nearly all hard macros into one component, bridge-soft evidence is
mathematically unavailable because every soft has only one current owner. That
narrow topology is refined by cosine-connected hard-to-soft affinity vectors
from shared low-fanout nets. Tiny fragments merge into their strongest
positive-affinity group; if the affinity partition is inconclusive, a stricter
partial hard-graph cut is the fallback. The result remains inferred evidence,
not an explicit IP tag, and multi-component designs retain the bridge-evidence
rule. Soft macros are classified as
**owned** (one cluster dominates their connectivity, so they move with that
cluster) or **bridge** (comparable affinity to multiple clusters, so they get
a region spanning the clusters they connect). When soft names themselves carry
useful slash-separated RTL paths, the deepest useful shared prefix creates a
high-confidence soft bundle. Explicit bundles take precedence over owner or
bridge-signature evidence during compound soft relocation; flat `Grp_*`
benchmarks do not form compounds from owner/bridge roles. A second diagnostic
layer builds a soft-only graph from repeated low-fanout shared nets and retains
only mutually strong components of at most 16 soft macros. These connectivity communities are
scored against common owned/bridge hard-cluster affinity. Combined evidence is
assigned deterministic `high` (≥0.90), `medium` (≥0.75), or `low` confidence.
Only explicit instance-path membership is high confidence and can form a
compound relocation group. Flat-netlist connectivity plus hard affinity is
limited to medium/low confidence: it continues to determine individual owned/
bridge regions and proposal evidence, but is not proof that the soft macros are
one IP.
Oversized split eligibility counts unique bridge softs per flat hard component;
evidence from another component cannot authorize a split.

The model separately retains one additional hierarchy level without changing
the active clusters sent to DREAMPlace. Path-tag designs keep the nearest
useful ancestor prefix above the selected leaf depth. If oversized-connectivity
refinement already split a flat component, that original component becomes the
parent of its active leaves. Otherwise, an active cluster with at least 12 hard
macros may be bisected exactly once when both children contain at least four
hard macros. The bisection graph starts with weighted low-fanout hard-hard
edges and hard connections through the same soft macro. Initial
hard/soft proximity reinforces those edges; a Gaussian macro-area neighborhood
estimates local density, while placed low-fanout net span and endpoint demand
estimate routing pressure. These physical signals cannot create an edge where
there is no structural connection. The resulting split is retained only when
its raw structural cut is at most 0.20, within-child mean distance improves by
at least 10%, and combined confidence is at least 0.54. No child is recursively
split. Parent and child layers each receive their own labels, hard/soft roles,
cluster graph, confidence, regions, reference vector, and six-component limits.

After ordinary active-cluster relocation, the child pass ranks hot eligible
children and tests rigid translations toward cold connected components and
available parent boundaries. It co-moves the child's owned movable soft macros.
If rigid geometry overlaps, only the affected child set is compacted and spiral
legalized against fixed outside macros. Siblings may also exchange slots; the
same affected-only legalization is available when a rigid exchange is blocked.
Every complete hard/soft state must remain in the parent regions and pass the
active, child, and parent contracts before exact incremental scoring. The pass
shares a 24-state quota, a 4s deadline guard, and requires a local exact gain of
0.0001. Multilevel limits become authoritative for downstream passes and final
rollback only after a child state is retained; pure discovery therefore cannot
perturb an otherwise unchanged placement trajectory.

After the whole-child pass, each deepest child receives a fixed internal box.
The initial outer box is the current child footprint plus a margin derived from
normalized congestion heat, density heat, and inter-child graph tension.
Congestion-aware region expansion can extend a hot side toward a nearby cold
connected component, while graph corridors bias which component is preferred;
the result is intersected with the retained parent box and then frozen.
Individual movable hard macros and owned soft macros may relocate within their
member-inset boxes. Hard-hard swaps are restricted to members of one child and
both swapped centers must remain in their own inset boxes. Weighted neighboring-
child centroids guide relocation targets, graph tension selects the children to
search, and active/child/parent contracts run before commit. The pass shares a
48-state exact quota and 3s guard. A retained state must improve local proxy by
at least 0.0005 before it can activate multilevel limits downstream.

### 2. Grouped DREAMPlace

`run_dreamplace()` accepts `cluster_groups` and `group_weight`; the bridge
adds synthetic clique nets among each cluster's hard and soft members so
DREAMPlace's global placement pulls each subsystem together. DREAMPlace is
required — there is no proxy-only fallback. Runtime availability is a real
subprocess import probe using the Python ABI that compiled DREAMPlace, including
representative native density, HPWL, and boundary ops plus the DREAMPlace 4.1
BB-Nesterov optimizer used by this stage. The bridge sets `macro_place_flag=1`
and `use_bb=1`. At each global-placement update, DREAMPlace uses the short
Barzilai-Borwein step
`alpha = (s^T y) / (y^T y)`, where `s` is the change in reference position and
`y` is the corresponding gradient change. This is a scalar inverse-Hessian
approximation that scales the Nesterov step from observed curvature without
forming or storing a Hessian. A non-positive BB step falls back to the predicted
Lipschitz step. A clean checkout can reproduce the local CUDA 12.1 build with
`scripts/dreamplace/bootstrap.sh all`; `scripts/dreamplace/bootstrap.sh preflight`
checks an existing install and now rejects builds without BB-Nesterov support.
BB and cache reads are fixed production behavior rather than runtime-gated
options. A bounded Zhang-Hager non-monotone Armijo trial was evaluated on
ibm04 and ibm10, regressed DREAMPlace seed quality on both, and was removed;
the focused numbers remain in `PROGRESS.md`.

### 3. Seed Portfolio Selection

Grouped DREAMPlace is one candidate seed among several: legalized
`initial.plc`, two DP/initial blends, a radial expansion from the DP basin,
and a synthetic-clearance push-apart from the DP basin. Production also adds a
constraint-graph legalization of `initial.plc`: overlapping pairs become
horizontal or vertical separation edges, both graphs stay acyclic under stable
seed-coordinate order, and longest-path earliest/latest bounds project each
movable macro toward its original coordinate. The ordinary initial candidate
remains in the same portfolio. The reference `initial.plc` is legalized before
any immutable limit is built; the same legalized coordinates are then
exact-scored as the ordinary initial candidate. Each scored candidate records a
richer hierarchy vector covering mean and worst hard-cluster spread,
nearest-neighbor cluster impurity, weighted inter-cluster edge stretch,
owned-soft distance, and bridge-soft corridor distance. Each vector component
must remain within its independent absolute-or-relative slack from the
legalized reference. Non-mandatory alternatives whose immutable hard
components already fail are rejected before exact scoring; the lowest-proxy
eligible scored seed enters hierarchy relief. The selected seed becomes
the reference for the same six-component contract at pass checkpoints and
final rollback. `HIER_SEED_HIERARCHY_SELECT=1` makes proxy the
secondary choice inside the best hierarchy-quality band. That policy remains
default-off: on the 2026-07-15 ibm10 experiment it improved seed composite
`0.29168 -> 0.16328` but regressed final proxy `1.1778 -> 1.5281`.
If no candidate satisfies the component contract, selection fails closed unless
the reference candidate itself passes; an invalid fallback is never promoted
to become the hierarchy baseline.
A mandatory candidate that has lower exact proxy than the selected seed and
misses exactly one component is a repair candidate. Production deterministically
interpolates it toward the authoritative passing reference, legalizes each
trial, and bisects the passing boundary. Only a repair that retains at least
`HIER_SEED_CONTRACT_REPAIR_MIN_FRACTION=0.95` of the source displacement is
exact-scored and returned to the ordinary selector. This guard rejected broad
IBM03/13 projections that changed the search basin; IBM09's accepted fraction
was `0.99609375`.
Single-component soft-affinity refinement adds one reference rule to avoid
compounding contract slack: when raw `initial.plc` is already in bounds and
overlap-free and its legalized form satisfies the raw limits, the raw vector
stays authoritative through final rollback. If the raw hard placement is
illegal, it cannot define trustworthy geometry; grouped DREAMPlace becomes the
reference for that topology. Seed telemetry records which reference was used.
Final reports classify evidence coverage as `high` (hard >= 0.75 and soft >=
0.25), `partial` (hard >= 0.25 and soft >= 0.10), or `low`; this is diagnostic
provenance and does not change proxy or hierarchy acceptance gates. Path-tag
clusters are reported as `explicit`, while flat-net connectivity is `inferred`.
The production 15% relative allowance and component absolute allowances were
retained after replay over 31 IBM, NG45, and synthetic final rows. A 10%
relative profile invalidated `ibm18`'s final state and `ibm07`'s selected seed;
a uniform 20% absolute reduction invalidated `ibm08`/`ibm11` finals and the
selected NG45 `nvdla` seed. The active profile therefore has measured real-
design support rather than being inferred from the aggregate composite score.

Neighbor impurity needs only the nearest four clustered hard macros. A cached
Numba kernel therefore keeps a four-entry insertion-ordered selection per
macro instead of materializing and stably sorting an N-by-N distance matrix.
It compares squared distances (the same ordering as Euclidean distance) and
uses the original clustered-row order for equal-distance ties, preserving the
previous stable-sort result.

Synthetic-clearance pair pushes are accumulated by a cached Numba kernel; the
seed update, clipping, legalization, scoring, and selection semantics are
unchanged.

### 4. Cluster-Consecutive Legalization

Hard macros legalize in an order that keeps cluster members adjacent
(largest clusters first, then connectivity-pressure × area within each
cluster, then unclustered macros), followed by a default-order safety pass
to guarantee legality. Each macro's expanding-ring search runs in a cached
Numba kernel with the original lexicographic candidate order, strict overlap
tests, and minimum-displacement tie behavior. Python retains the between-macro
deadline check, and the former vectorized conflict-matrix path remains the
diagnostic reference.

The constraint-graph candidate always runs the same default-order spiral safety
pass after projection. A dense or infeasible constraint graph therefore cannot
bypass legality, fixed-macro immobility, or bounds. On the accepted 17-design
sweep it was selected on ibm10, ibm12, and ibm14-18; the unchanged candidates
protected every other design from regression.

### 5. Soft Cleanup

`_soft_relocation_moves()` relocates soft macros by congestion and density.
Soft macros may overlap, so this pass has no legality constraint.

### 6. Region-Locked Relief

`HierarchyModel.hard_regions()` / `soft_regions()` build region boxes per
cluster, expanded toward nearby cold congestion components where available.
Hard and soft relocation inside these regions rank candidates by a
congestion-weighted proposal field plus density; moves that leave the
assigned region are accepted only when the exact-proxy gain clears
`HIER_REGION_ESCAPE_MIN`.

Before a region hard-relocation candidate reaches exact batch scoring, the
pipeline temporarily applies that one move and evaluates the inexpensive
legacy hard-cluster containment metric. Candidates above the selected seed's
hard audit limit are discarded and counted as `hierarchy_rejects`. The rich
six-component contract is still checked after the completed pass, so this
prefilter reduces known-ineligible work without weakening the authoritative
checkpoint. The accepted full sweep rejected 654 candidates this way,
exact-scored 18,637 remaining relocation candidates, and required no hard-
relocation rollback.

For refined single-component graphs with at most 64 hard macros, micro-shift
and hard/soft relocation candidates also pass the complete vector contract
before exact scoring. This retains useful work on the small SRAM-shaped case.
Larger refined graphs use the normal pass checkpoint and final rollback rather
than recomputing the rich vector for every proposal.

Relocation has a default-off structural ordering term
(`HIER_OBJECTIVE_STRUCTURAL_WEIGHT=0.0`) that combines edge clearance, grid
alignment, and local gap penalties. It only reorders proposals; legality,
region, hierarchy-quality, and exact-proxy gates are unaffected.

### 7. Cluster Decompression

`_cluster_decompression_relief()` expands hot clusters inside their regions
to open local routing channels, biasing toward nearby cold congestion
components when present. Candidates re-legalize hard macros, move owned softs
with their cluster, and are accepted only if exact proxy improves while a
composite hierarchy-quality metric (mean radius + bbox spread + crowding
penalty) stays within `HIER_QUALITY_BUDGET`.

### 8. Region-Bounded Swaps

`_region_bounded_swap_relief()` runs hard-hard, hard-soft, and soft-soft
swaps against the same congestion/density proposal field. In-region swaps use
the exact-proxy gate directly; out-of-region swaps must also clear the escape
threshold. Swap scoring uses the `IncrementalScorer` global topology arrays;
candidate ranking can use CUDA batch sorting when available, but exact
scoring remains the acceptance authority either way.

Each swap source preserves its ranked candidate list. Hard-hard batches first
score four candidates, hard-soft batches first score eight, and soft-soft
score twelve. If the first prefix has no acceptable candidate, the scheduler
scores a second prefix of the same size before the untouched remainder. If
either prefix contains the first candidate that clears exact proxy, region, and
hierarchy gates, the untouched suffix is skipped. Otherwise the remainder is
scored as one batch and the original first-winner order is unchanged. Logical score
counts continue to drive the accepted deterministic quota, while telemetry
separately records physical exact work, the skipped suffix count, and accepted
candidate-rank buckets. Hard-hard and hard-soft sources rank and truncate their
candidate lists before running the independent legality kernel, so candidates
that cannot reach exact scoring do not allocate overlap matrices. When graph
masks and deltas are disabled, the same loop also skips their zero-valued array
construction.

When a source schedule reaches beyond its first prefix, the scorer prepares the
source-invariant pair topology and coordinate snapshots once for the complete
stable candidate list. Each prefix receives a view of its candidate rows and a
rebased view of the ragged touched-net offsets. The untouched suffix is still
unprepared for exact field scoring until needed, and any mock/scalar scorer
without the prepared API follows the original batch calls.

Hard-macro widths and heights do not change during the schedule. The pipeline
therefore constructs their two pairwise center-separation matrices once before
the congestion/density rounds and passes the immutable tuple through ordinary
and graph-fallback swap calls. Standalone callers retain a local-computation
fallback.

The incremental scorer keeps raw and smoothed routing grids synchronized after
each trial move. A prepared routing structure caches pin-to-module references,
offsets, topology groups, weights, and scratch buffers. One cached Numba call
gathers current pin cells, applies two-, three-, and high-fanout routes, and
returns the touched bbox without Python buckets or sorting. Horizontal-column
and vertical-row bbox smoothing then uses cached Numba kernels with reusable
prefix buffers. The NumPy formulas remain the diagnostic oracle; both JIT
stages preserve their reference accumulation order and incremental deltas.

Soft-relocation grid IDs first pass through a compiled stable filter that
converts and clips target centers, constructs exact symbolic coordinate keys,
applies the optional region mask, and stamps first-winner duplicates. The
hierarchy callback remains scalar and ordered because it evaluates the rich
contract. Prepared target sets then use a compiled exact wirelength
delta batch. The existing threshold is applied in stable proposal order before
field scoring; this changes neither candidate order nor acceptance semantics.
Surviving target sets of size two or more use a true batched CPU path. Cached
JIT loops build per-target routing grids, touched bboxes,
wirelength, and density occupancy without mutating committed scorer state;
the congestion and density tail reductions then operate over the batch. The
congestion grids are disposable after reduction, so their float64 rows use
in-place `ndarray.partition` and avoid a second full batch copy while retaining
the evaluator's exact top-five-percent sum. The density reducer is itself a
cached Numba kernel that retains the scalar nonzero filtering, tiny-grid mean,
and top-tail partition semantics. Scalar trials remain the one-target path and
the parity oracle.

Soft-soft swap sets sharing one endpoint no longer build or flatten one routing
structure per endpoint pair. The JIT loop consumes sorted touched-net ids and
the scorer's global net starts, lengths, weights, pin references, and offsets.
It packs selected pin cells once per old/new state, applies two-pin, three-pin,
then high-fanout routes in evaluator order, and performs the exact
remove/swap/add sequence. The committed routing, smoothing, density, and
position caches remain read-only until the existing scalar commit method
accepts a winner.

Swap tail reduction is baseline-plus-delta rather than full-result-grid. For
congestion, a candidate recomputes only H columns and V rows in the route bbox
plus cells whose hard blockage changed; the reducer combines those values with
the highest unchanged baseline values before the exact top-five-percent sum.
For density, four rectangle deltas define every affected cell, and the same
merge recovers the scalar nonzero/top-ten-percent semantics. Raw routing trial
grids remain batched, but the former full congestion-value and density-result
matrices are gone. The sorted baseline values and density summary are cached
until a commit mutates routing, blockage, or occupancy state; every hard,
soft, swap, and multi-move commit invalidates that cache.

Two further memory/parallel experiments remain deliberately absent. Fusing
hard-blockage construction into one reducer scratch regressed IBM04 and IBM12,
and unbounded candidate-row `prange` launched 22 workers for thousands of
small batches, exhausting IBM18's 20s phase guard after 32,498 candidates
instead of completing 69,152 in 8.51s. Productive parallelism would require a
coarser schedule boundary, not parallel reduction inside each small batch.

Hard-hard and hard-soft sets sharing one hard endpoint also use exact batched
scoring. Candidate-specific hard-blockage grids reproduce the reference
remove/swap/add order, including top-row and right-column correction terms,
while prepared route, wirelength, and density kernels evaluate the full set.
Scalar scoring remains the one-candidate/fallback oracle. Cross-design parity
was exact to floating-point roundoff (maximum `2.22e-16`) and every committed
scorer cache remained unchanged during batch trials.

### 9. Post-Swap Polish

Hard propose-all relocation and soft relocation each run once more over the
swap-relieved state with a stronger exact-gain margin, followed by a
micro-shift replay. A budget-aware strong-soft-repair pass can spend any
remaining time on soft macros (without reopening hard legality) when recent
pass telemetry shows plateaued cleanup or a useful soft signal.

### 10. Coldspot Tightening

`_coldspot_cluster_kick()` selects a hot cluster, gathers it and its owned/
bridge soft macros into a low-congestion window, legalizes, and refines
locally (swaps + relocation) before accepting only if exact proxy improves
and the hierarchy-quality metric stays within budget. When no kick commits,
a graph-local fallback runs the same bordered refinement on the hottest
eligible cluster without a kick.

### 11. Final Audit

Production continues from post-coldspot replay directly to structurally
eligible small-design polish, then a hard-legality margin audit and final
hierarchy-quality audit against the selected seed. It rolls back to the best
saved audit-passing checkpoint if the final state fails. The former broad
survivor pool was removed after 636 telemetry records showed no proxy gain.

### 12. Plateau Telemetry

The hierarchy flow always buffers schema-v2 pass-level telemetry: proposed and
retained proxy/accepts, audit rollback and violations, discarded gain, scorer
rebuild time, elapsed time, accept rate, plateau flag, run id, code revision,
process id, dirty-worktree flag, and a deterministic scoped worktree
fingerprint. Seed creation/prescore, cache lookup, coldspot work, full exact
scores, and final audits are separate stage events. The outer orchestration also
records mutually exclusive setup, seed-portfolio, hierarchy-search, coldspot,
and post-coldspot phases plus `hierarchy_floorplan_total` and
`placer_api_total` boundaries. Exact-scored seed
candidates and the final placement also emit structured
`hierarchy_contract_audit` events containing vectors, limits, margins,
violations, coverage, and provenance. The default output is
`ml_data/plateau_telemetry/plateau_telemetry.jsonl`;
`HIER_PLATEAU_TRACE_PATH` can redirect it. Candidate-level trace logging and
learned ranking were removed because they added overhead and repeatedly failed
to improve placement. `scripts/analyze_plateau_telemetry.py` filters the
remaining scheduling telemetry by run, revision, worktree fingerprint, or
benchmark, reports aggregate retained yield and conservative skip candidates,
prints stage timing with `--stages`, and summarizes exact-score limits,
per-benchmark distributions, and exhaustion with `--quotas`. Every pass row
records its exact quota, usage before and after the invocation, and whether the
limit was exhausted. Passes with no exact-scored candidates
report gain-per-score as `n/a`, never as an artificial infinite yield.
`--coverage` reconciles the five exclusive phases to the floorplan and API
boundaries. On the accepted IBM sweep they cover at least 99.86% of every API
call: 297.33s inside `MacroPlacer.place()` versus 318.55s for the complete
evaluator command, exposing 21.22s of external load/final-score overhead.
`scripts/analyze_hierarchy_contract.py` separately aggregates per-component
headroom, relevant-row counts, allowance utilization, coverage/provenance
cohorts, and failures. It can replay alternative relative and per-component
absolute slacks without changing production. NG45 audit rows use their design
names (`ariane133`, `ariane136`, `mempool_tile`, and `nvdla`) even though their
source directories share the leaf name `output_CT_Grouping`.

The synthetic runner also emits `hierarchy_truth_audit` rows from cluster labels
preserved by `generate_benchmarks.py`. These are independent accuracy checks,
not production acceptance gates. The current ten-design sweep passes all ten.
The previously failing `syn03_sram` case now recovers its four truth groups
exactly (purity, pair precision, and recall all `1.0`) and improves proxy
`4.3964 -> 4.3257`; this is structural recovery rather than looser slack.

### 13. Deterministic Exact-Score Quotas

Wall-clock deadlines remain safety guards, but the high-volume region,
interleaved, plateau, compound, and strong/medium repair operators stop at
deterministic work ceilings first. Repeated region rounds consume one shared
pass allowance. Regional hard-hard, hard-soft, and soft-soft swaps likewise
share one allowance, so no swap type receives unbounded work. The next stable
candidate batch is sliced to the remaining allowance without changing
candidate or commit order.

| Pass | Exact-score ceiling |
|---|---:|
| region hard relocation | 2,600 |
| region soft relocation | 24,000 |
| parent-bounded child relocation / sibling swaps | 24 |
| deepest-child internal relocation / hard swaps | 48 |
| interleaved soft repair | 4,096 |
| regional swaps | 72,000 |
| regional-swap graph fallback | 100 |
| plateau escape, first / post | 5,000 / 7,000 |
| compound soft relocation | 60 |
| strong / medium soft repair | 40,000 / 2,048 |

These ceilings were derived from attributable accepted-run maxima with modest
headroom. The IBM and NG45 validation sweeps preserved every preceding score.
Only IBM interleaved soft repair reached a ceiling, on ibm11 and ibm17, where
the reference already scored exactly 4,096 candidates. An intentionally
aggressive binding profile regressed ibm11 and was not promoted.

## Scoring and Legality

- Fixed macros stay fixed.
- Hard macros must not overlap.
- All macro centers must be in bounds (`_clamp_in_bounds()` runs on every
  returned placement).
- Soft macros may overlap.

Exact proxy scoring drives evaluator reports, the initial hierarchy score
measurement, and every accept gate listed above (relocation, decompression,
swaps, coldspot tightening). `PlacementState` carries hard positions, soft
positions, and the current exact proxy through the pipeline; pass results
are returned as structured `PassResult` trace payloads.

## Verification

```bash
uv run python -m py_compile $(find src -type f -name "*.py")
uv run python test/verification/_verify_region_escape_gate.py
uv run python test/verification/_verify_score_region_swaps.py
uv run python test/verification/_verify_coldspot_kick.py ibm10
uv run pytest test/ -q
uv run evaluate src/main.py -b ibm10
uv run evaluate src/main.py --all
```

## Constants Reference (`src/utils/constants.py`)

Grouped by the pipeline stage that consumes them.

**Hierarchy / clustering**
```text
HIER_TAG_PREFIX_MAX_DEPTH=5            HIER_TAG_PREFIX_MIN_GROUP=2
HIER_TAG_PREFIX_MIN_COVERAGE=0.25      CLUSTER_MIN_EDGE=2
CLUSTER_MAX_FANOUT=8                   HIER_OVERSIZE_CLUSTER_START_FRAC=0.40
HIER_OVERSIZE_CLUSTER_TARGET_FRAC=0.15 HIER_OVERSIZE_CLUSTER_TARGET_TOL=1.10
HIER_OVERSIZE_CLUSTER_MIN_BRIDGE_SOFTS=5
HIER_OVERSIZE_CLUSTER_MIN_SIZE=6       HIER_OVERSIZE_CLUSTER_MAX_CUT_RATIO=0.45
HIER_SUBCLUSTER_MIN_PARENT_HARD=12     HIER_SUBCLUSTER_MIN_CHILD_HARD=4
HIER_SUBCLUSTER_MAX_CUT_RATIO=0.20     HIER_SUBCLUSTER_SHARED_SOFT_WEIGHT=0.75
HIER_SUBCLUSTER_SPATIAL_PROXIMITY_WEIGHT=1.0
HIER_SUBCLUSTER_SPATIAL_PRESSURE_WEIGHT=0.50
HIER_SUBCLUSTER_SPATIAL_NEIGHBORS=8    HIER_SUBCLUSTER_SPATIAL_MAX_SOFT_DEGREE=24
HIER_SUBCLUSTER_SPATIAL_MIN_COMPACTNESS_GAIN=0.10
HIER_SUBCLUSTER_SPATIAL_MIN_CONFIDENCE=0.54
HIER_SUBCLUSTER_RELOCATION_MAX_HARD=64
HIER_GROUP_WEIGHT=8
```

**Seed portfolio**
```text
HIER_SEED_BLEND_ALPHAS=0.35,0.65   HIER_SEED_EXPANSION_FRAC=0.06
HIER_SEED_CLEARANCE_FRAC=0.08      HIER_SEED_CLEARANCE_ITERS=3
HIER_SEED_CLEARANCE_AREA_PCT=97
HIER_VECTOR_CONTRACT_REL_SLACK=0.15
HIER_VECTOR_CONTRACT_ABS_SLACK={compactness:0.005,worst_spread:0.015,
  neighbor_impurity:0.05,edge_stretch:0.015,owned_soft:0.015,bridge_soft:0.015}
```

**Regions and relocation**
```text
HIER_REGION_DENSITY=0.65        REGION_BIAS=1.0
HIER_REGION_ROUNDS=2            HIER_REGION_BUDGET_S=40
HIER_REGION_ESCAPE_MIN=0.002
HIER_REGION_COMPONENT_COLD_PCT=45     HIER_REGION_COMPONENT_MIN_CELLS=4
HIER_SUBCLUSTER_RELOCATION_BUDGET_S=4 HIER_SUBCLUSTER_RELOCATION_MIN_SPARE_S=12
HIER_SUBCLUSTER_RELOCATION_TOP_CHILDREN=4 HIER_SUBCLUSTER_RELOCATION_TOP_SWAPS=4
HIER_SUBCLUSTER_RELOCATION_MIN_GAIN=0.0001
HIER_PROPOSAL_CONGESTION_WEIGHT=2.5   HIER_PROPOSAL_DENSITY_WEIGHT=1.0
HIER_PROPOSAL_OUTSIDE_RELIEF_MARGIN=0.08
HIER_RELOC_PROPOSE_MIN_GAIN=0.0005
HIER_COMPOUND_SOFT_BUDGET_S=4         HIER_COMPOUND_SOFT_MIN_SPARE_S=5
HIER_COMPOUND_SOFT_TOP_GROUPS=4       HIER_COMPOUND_SOFT_GROUP_SIZE=6
HIER_COMPOUND_SOFT_COLD_PCT=35        HIER_COMPOUND_SOFT_ANCHORS=2
HIER_COMPOUND_SOFT_SHIFT_FRACTIONS=0.25,0.5,1.0
HIER_COMPOUND_SOFT_MIN_FIELD_DROP=0.02 HIER_COMPOUND_SOFT_MIN_GAIN=0.00005
HIER_PLATEAU_ESCAPE_BUDGET_S=4        HIER_PLATEAU_ESCAPE_SOFT_TOP_K=384
HIER_PLATEAU_ESCAPE_SOFT_TARGETS=10
```

**Deterministic structural candidate ordering (opt-in)**
```text
HIER_OBJECTIVE_STRUCTURAL_WEIGHT=0.0   HIER_KEEP_OUT_WEIGHT=0.2
HIER_GRID_ALIGN_WEIGHT=0.2             HIER_NOTCH_WEIGHT=0.6
```

**Decompression**
```text
HIER_DECOMPRESS_ROUNDS=2          HIER_DECOMPRESS_BUDGET_S=18
HIER_QUALITY_BUDGET=0.03          HIER_QUALITY_RADIUS_WEIGHT=0.75
HIER_QUALITY_BBOX_WEIGHT=0.20     HIER_QUALITY_CROWD_WEIGHT=0.05
HIER_DECOMPRESS_LOCAL_SHIFT_FRAC=0.20
```

Rounds with no cheap hot-cluster to cold-window opportunity are skipped before
candidate generation and exact candidate scoring. The predictor blends
hot-to-cold field gap, open cold-cell capacity around the candidate window, and
source-to-window displacement. Default production tries the top two opportunity
clusters with five whole-cluster variants per cluster, then commits from
exact-proxy-ranked refined candidates rather than from a graph-ranked prefix.
Coldspot also stops after repeated generated pools fail to commit.
Weak-opportunity and dry-limit exits skip graph-local and soft-only coldspot
fallbacks too.

This is not the old generic LSMC path. It is a narrow hierarchy-tightening
helper. Candidate-local refinement runs hard-hard and hard-soft swaps with the
kicked hard cluster locked in the local box, plus soft-soft swaps and soft
relocation that may leave the local box only after a `0.0025` exact-proxy gain.
The local box includes owned/bridge soft macros, but its base pad is derived from
the kicked hard-core max dimension rather than the soft-inclusive bbox.
The phase tracks a current cold-cell grid from the active congestion field,
refreshes it after every finalized coldspot kick, masks out cells occupied by the
candidate, and expands the pre-margin local border through adjacent open cold
cells before applying the hard-core pad. This lets finalized cluster locations
use nearby coldspots for local relief while preserving swap and soft-locked
relocation room.
The graph supplies coldspot-local relocation target pools and gates relocation
targets by graph mask; default candidate commitment uses exact-proxy-ranked
refined outcomes.
`HIER_COLDSPOT_SOFT_ONLY=0` is a default-off fallback that runs only when hard
coldspot kicks and graph-local fallback commit no candidate. It keeps all hard
macros fixed, builds a target pool from remembered open cold cells, and invokes
the exact-gated soft relocation pass with hierarchy region boxes and the cold
cell mask still active.
Coldspot kick candidate generation augments each cluster's owned soft set with
movable bridge soft macros tied to the same hierarchy cluster. The default pool
now tries multiple opportunity-ranked clusters, with shape-preserving variants
for each instead of only repeating one random gather: multiple cold anchors,
compact original orientation, rotated orientation, source-facing border
compaction, and a lower-displacement centroid-blended candidate. The hard
cluster and those soft macros are placed into the cold window together, then the
existing legalization, local refinement, exact-proxy gate, and hierarchy-quality
gate accept or reject the resulting full candidate as one state.
`HIER_COLDSPOT_PARTIAL_FRONTIER=0` is a default-off experiment that can add one
capacity-aware partial frontier candidate to the same pool: it estimates the
connected cold area around the chosen anchor, selects a true subset of the hot
cluster nearest that anchor, biases the split by low-fanout net connectivity,
places cross-cut-heavy macros near the source-facing coldspot border, and then
uses the same legalization, local refinement, exact-proxy gate, and
hierarchy-quality gate as the normal kick. The prototype skips tiny source
clusters by default because far 2-of-3 splits can improve proxy while failing
the radius/bbox hierarchy-quality metric. It also runs a cheap pre-exact
split-shape predictor after partial hard legalization and rejects candidates
whose source cluster radius, bbox radius, or moved-vs-remaining separation
would grow beyond the configured ratios. Additional cheap gates reject majority
splits, splits that leave too few source macros behind, disconnected selected
subsets when low-fanout local edges are available, and high selected-vs-remaining
cut ratios. Majority/remaining-macro limits are applied during subset
construction, not only after selection, so the partial generator can try
smaller frontier groups before rejecting. When no coldspot kick commits, the
graph-local fallback runs the same bordered swaps and relocations on the
current placement for the hottest eligible clusters.
Production then reruns `_micro_shift_polish()` once more after coldspot
tightening; deterministic hot-cluster coldspot selection was tested and removed
after regressing the full sweep.

**Swap ranking**

The current CPU/Numba swap breadth is:

```text
HIER_HARD_SWAP_K=16          HIER_SOFT_SWAP_K=48
HIER_SWAP_MIN_GAIN=0.00001
```

Repeated batched swap evaluations reuse a bounded cache of static pair-topology
packing inside the incremental scorer; exact score values and acceptance gates
are unchanged.

Fallback congestion expansion now skips a hot cluster when no adjacent fallback
side is colder than the cluster; component-guided expansion remains unchanged.

**Seed alternatives**
```text
constraint-graph initial seed: always included; HIER_CONSTRAINT_GRAPH_MAX_ROUNDS=6
```

**Post-swap / plateau scheduling**
```text
HIER_POST_SWAP_MICRO_SHIFT_BUDGET_S=8   HIER_STRONG_SOFT_REPAIR_BUDGET_S=12
HIER_STRONG_SOFT_REPAIR_MIN_SPARE_S=2   HIER_STRONG_SOFT_REPAIR_ROUNDS=2
HIER_SOFT_QUOTA_REFERENCE_MACROS=1200   HIER_SOFT_QUOTA_MIN_SCALE=0.50
HIER_PLATEAU_ACCEPT_RATE=0.002          HIER_PLATEAU_PROXY_GAIN=0.00005
HIER_PLATEAU_ESCAPE_BUDGET_S=4
```

**Coldspot tightening**
```text
HIER_COLDSPOT_ROUNDS=8              HIER_COLDSPOT_BUDGET_S=30
HIER_COLDSPOT_MIN_GAIN=0.0001       HIER_COLDSPOT_QUALITY_BUDGET=0.01
HIER_COLDSPOT_MIN_FIELD_GAP=0.02    HIER_COLDSPOT_MAX_DRY_ROUNDS=2
HIER_COLDSPOT_OPPORTUNITY_TOP_CLUSTERS=1
HIER_COLDSPOT_WHOLE_VARIANTS=5      HIER_COLDSPOT_ANCHOR_VARIANTS=3
HIER_COLDSPOT_SOFT_ONLY=0           HIER_COLDSPOT_PARTIAL_FRONTIER=0
```

**Plateau telemetry (runtime env vars, not constants)**
```text
HIER_PLATEAU_TRACE_DIR=ml_data/plateau_telemetry
HIER_PLATEAU_TRACE_PATH=<optional output override>
VIVAPLACE_RUN_ID=<optional attributable run id>
VIVAPLACE_WORKTREE_FINGERPRINT=<optional provenance override for tests/tools>
HIER_<PASS_NAME>_MAX_EXACT=<optional positive exact-score ceiling override>
HIER_GPU_EXPERIMENT=<one isolated diagnostic CUDA hypothesis>
```

The gate is diagnostic-only: it does not enable a new production operator.
When set, it selects exactly one CUDA hypothesis and forces every other
optional CUDA route onto its CPU fallback; leave it unset for production.
Selections are `overlap_prefilter` and `graph_tension_batches`. The former
retains the fp64 overlap/bounds prefilter for diagnostics, while the latter is
an isolation control because its active designs have too few graph edges for a
viable batch kernel. Rejected exact-reduction, widened relocation-delta, and
proposal-filter experiment code was removed. Details are in `PROGRESS.md`.

Experiments that were tried and not promoted (full recursive bisection,
cluster-room/bridge-corridor modeling, broad weak-hot region reshape, early
strong-soft repair, early swap-lite, deterministic hot-cluster coldspot
selection, learned candidate reordering at full-suite scale) are recorded in
`ISSUES.md` and `PROGRESS.md`, not here — this document describes only the
active system.
