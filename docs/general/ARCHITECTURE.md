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
       - classify soft macros as owned (one dominant cluster) or bridge
       - record inter-cluster edge weights and confidence
  -> grouped DREAMPlace global placement (synthetic clique nets per cluster)
  -> cluster-consecutive hard legalization
  -> exact-proxy seed portfolio selection
       (grouped DREAMPlace, legalized initial.plc, DP/initial blends,
        radial expansion, synthetic-clearance push-apart, and a
        constraint-graph-legalized initial.plc; every candidate gets a complete
        hierarchy vector and lowest exact proxy enters relief)
  -> congestion-expanded hard/soft hierarchy regions
  -> exact-gated local micro-shift polish
  -> exact-gated cluster decompression with composite hierarchy quality
       - large designs can order opportunities by hierarchy graph tension
  -> budget-aware interleaved soft repair
  -> region-bounded hard-hard / hard-soft / soft-soft swaps
       - hard-moving swap candidates must stay inside the hierarchy audit budget
       - optional micro-shift replay after each swap round
  -> post-swap micro-shift replay
  -> post-swap hard propose-all relocation with spare-budget additive candidates
  -> post-swap soft relocation with spare-budget additive candidates
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
       - graph-rank generated outcomes when the GNN selector is off
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
[OBJECTIVES.md](OBJECTIVES.md). BeyondPPA-style deterministic structural
ranking and GNN trace logging are documented in
[../ml_nn/beyondppa_results/](../ml_nn/beyondppa_results/).

Current verified full sweep with strict hierarchy-audit rollback,
audit-aware hard swap gating, component-aware region expansion/decompression,
large-design hierarchy graph-tension opportunity ordering, swap-round
micro-shift replay, stronger opportunity gates, component-aware scheduling,
post-coldspot small-design polish with subpass audit restore, no-release
low-net soft/SS breadth, medium/large soft-continuation scheduling, prepared
Numba routing/legalization kernels, exact batched hard-hard/hard-soft scoring,
batched soft relocation/swap scoring, and the guarded constraint-graph seed:

```text
uv run evaluate src/main.py --all
AVG 1.1199  17/17 VALID  0 overlaps  575.28s
```

The prior proxy-leaning hierarchy sweep reached `AVG 1.1627`, 17/17 VALID,
0 overlaps, 1116.90s, but final hierarchy audit was report-only and failed on
several designs after late proxy-improving relief. A strict final-rollback-only
audit sweep reached `AVG 1.1999`; the audit-preserving local-relief recovery
reached `AVG 1.1664`; the pre-optimization BB-on verification was
`AVG 1.1653` cold (`AVG 1.1652` with cache hits). The prior optimized
normal-cache sweep was `AVG 1.1575`; the accepted constraint-graph/batched-swap
sweep is `AVG 1.1199`. The prior best same-path sweep was
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
Trace analysis for this signal lives in `scripts/gnn/analyze_graph_tension.py`.
Coldspot and decompression candidates also log graph-edge candidate deltas:
weighted edge stretch, corridor congestion change, weighted edge-length change,
and a combined graph delta. These are diagnostic/ranking features only; they do
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
AVG 0.7252  4/4 VALID  0 overlaps  232.41s
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
| `src/placer/local_search/hierarchy_model.py` | Inferred hierarchy: hard clusters, soft roles, cluster graph, region builders. |
| `src/placer/local_search/hierarchy_quality.py` | Complete hierarchy vector: compactness, worst spread, neighbor impurity, graph stretch, and owned/bridge soft-role distances. |
| `src/placer/local_search/clusters.py` | Hard-cluster derivation, oversized-cluster splitting, region-box primitives. |
| `src/placer/local_search/relocation.py` | Hard and soft relocation used by region-locked relief and post-swap polish. |
| `src/placer/local_search/hierarchy_swaps.py` | Region-bounded hard-hard, hard-soft, soft-soft swap relief. |
| `src/placer/local_search/cluster_decompress.py` | Exact-gated decompression of hot hierarchy blobs. |
| `src/placer/local_search/region_expand.py` | Expands hot cluster regions toward colder congestion bands. |
| `src/placer/local_search/lsmc_explore.py` | Coldspot kick candidate generation. |
| `src/placer/local_search/fields.py` | Congestion/coldspot fields used by relocation and coldspot tightening. |
| `src/placer/local_search/gnn_trace.py` | JSONL trace + plateau telemetry writers (diagnostic only). |
| `src/placer/scoring/exact.py` | Exact TILOS proxy wrapper. |
| `src/placer/scoring/incremental.py` | Incremental scorer for relocation and swap moves, including Numba-JIT bbox re-smoothing. |
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
selectively split toward a target leaf size. Soft macros are classified as
**owned** (one cluster dominates their connectivity, so they move with that
cluster) or **bridge** (comparable affinity to multiple clusters, so they get
a region spanning the clusters they connect).

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
remains in the same portfolio, so this alternative advances only when its exact
proxy is lower. All candidates are exact-scored; the lowest-proxy seed enters
hierarchy relief. Each seed also records a richer
hierarchy vector covering mean and worst hard-cluster spread, nearest-neighbor
cluster impurity, weighted inter-cluster edge stretch, owned-soft distance, and
bridge-soft corridor distance. `HIER_SEED_HIERARCHY_SELECT=1` makes proxy the
secondary choice inside the best hierarchy-quality band. That policy remains
default-off: on the 2026-07-15 ibm10 experiment it improved seed composite
`0.29168 -> 0.16328` but regressed final proxy `1.1778 -> 1.5281`.

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
threshold. Swap scoring uses cached `IncrementalScorer` routing structs;
candidate ranking can use CUDA batch sorting when available, but exact
scoring remains the acceptance authority either way.

The incremental scorer keeps raw and smoothed routing grids synchronized after
each trial move. A prepared routing structure caches pin-to-module references,
offsets, topology groups, weights, and scratch buffers. One cached Numba call
gathers current pin cells, applies two-, three-, and high-fanout routes, and
returns the touched bbox without Python buckets or sorting. Horizontal-column
and vertical-row bbox smoothing then uses cached Numba kernels with reusable
prefix buffers. The NumPy formulas remain the diagnostic oracle; both JIT
stages preserve their reference accumulation order and incremental deltas.

Prepared soft-relocation target sets of size two or more use a true batched
CPU path. Cached JIT loops build per-target routing grids, touched bboxes,
wirelength, and density occupancy without mutating committed scorer state;
the congestion and density tail reductions then operate over the batch. Scalar
trials remain the one-target path and the parity oracle.

Soft-soft swap sets sharing one endpoint use the same batch reductions. Pair
route structures are flattened from the existing topology cache, and a JIT
loop performs the exact remove/swap/add sequence for each second endpoint.
The committed routing, smoothing, density, and position caches remain
read-only until the existing scalar commit method accepts a winner.

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

### 12. Trace Logging and Plateau Telemetry

No learned ranker is active in production. `HIER_GNN_TRACE=1` writes
schema-v1 JSONL candidate traces (relocation, swaps, decompression, coldspot)
for offline GNN experiments; it does not change placement output. Candidate
schema v1 remains compatible with the existing dataset builder while adding
`run_id`, `code_revision`, and `pid` provenance fields.
The hierarchy flow always buffers schema-v2 pass-level telemetry (proxy
before/after, elapsed time, accept rate, plateau flag, and the same provenance).
`HIER_PLATEAU_TRACE_PATH` can redirect it. `scripts/analyze_plateau_telemetry.py`
filters by run, revision,
or benchmark and reports aggregate yield and conservative skip candidates. See
[`../ml_nn/beyondppa_results/`](../ml_nn/beyondppa_results/) for the current
state of GNN-assisted candidate ranking experiments — all are default-off.

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
HIER_GROUP_WEIGHT=8
```

**Seed portfolio**
```text
HIER_SEED_BLEND_ALPHAS=0.35,0.65   HIER_SEED_EXPANSION_FRAC=0.06
HIER_SEED_CLEARANCE_FRAC=0.08      HIER_SEED_CLEARANCE_ITERS=3
HIER_SEED_CLEARANCE_AREA_PCT=97
```

**Regions and relocation**
```text
HIER_REGION_DENSITY=0.65        REGION_BIAS=1.0
HIER_REGION_ROUNDS=2            HIER_REGION_BUDGET_S=40
HIER_REGION_ESCAPE_MIN=0.002
HIER_REGION_COMPONENT_COLD_PCT=45     HIER_REGION_COMPONENT_MIN_CELLS=4
HIER_PROPOSAL_CONGESTION_WEIGHT=2.5   HIER_PROPOSAL_DENSITY_WEIGHT=1.0
HIER_PROPOSAL_OUTSIDE_RELIEF_MARGIN=0.08
HIER_RELOC_PROPOSE_HOT_K=32           HIER_RELOC_PROPOSE_MIN_GAIN=0.0005
HIER_POST_RELOC_PROPOSE_ALL=auto      HIER_POST_RELOC_PROPOSE_TOP_M=16
HIER_POST_SOFT_RELOC_TOP_K=256        HIER_POST_SOFT_RELOC_MIN_GAIN=0.0005
```

**Structural candidate ordering (BeyondPPA, opt-in)**
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
`HIER_GNN_COLDSPOT_POLICY=1` can rank raw candidate proposals before this
refinement step and limit how many raw candidates are refined. `HIER_GNN_COLDSPOT_SELECT=1`
adds a second opt-in exact-score stage over the refined candidates.
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
cut ratios. When `HIER_GNN_TRACE=1`, generated-but-rejected partial candidates
emit `hier_coldspot_partial_reject` rows with the reject reason and shape/connectivity
stats. Majority/remaining-macro limits are applied during subset construction,
not only after selection, so the partial generator can try smaller frontier
groups before rejecting. When no coldspot kick commits, the
graph-local fallback runs the same bordered swaps and relocations on the
current placement for the hottest eligible clusters.
Production then reruns `_micro_shift_polish()` once more after coldspot
tightening; deterministic hot-cluster coldspot selection was tested and removed
after regressing the full sweep.

**Swap ranking**

The current swap breadth and optional GPU ranking controls are:

```text
HIER_HARD_SWAP_K=16          HIER_SOFT_SWAP_K=48
HIER_SWAP_MIN_GAIN=0.00001   HIER_GPU_RANK_SWAP_CANDIDATES=auto
HIER_GPU_RANK_MIN_CANDIDATES=512
```

**Seed alternatives**
```text
constraint-graph initial seed: always included; HIER_CONSTRAINT_GRAPH_MAX_ROUNDS=6
```

**Post-swap / plateau scheduling**
```text
HIER_POST_SWAP_MICRO_SHIFT_BUDGET_S=8   HIER_STRONG_SOFT_REPAIR_BUDGET_S=12
HIER_STRONG_SOFT_REPAIR_MIN_SPARE_S=2   HIER_STRONG_SOFT_REPAIR_ROUNDS=2
HIER_PLATEAU_ACCEPT_RATE=0.002          HIER_PLATEAU_PROXY_GAIN=0.00005
HIER_PLATEAU_ESCAPE_BUDGET_S=4
```

**Coldspot tightening**
```text
HIER_COLDSPOT_ROUNDS=8              HIER_COLDSPOT_BUDGET_S=30
HIER_COLDSPOT_MIN_GAIN=0.0001       HIER_COLDSPOT_QUALITY_BUDGET=0.01
HIER_COLDSPOT_MIN_FIELD_GAP=0.02    HIER_COLDSPOT_MAX_DRY_ROUNDS=2
HIER_COLDSPOT_WHOLE_VARIANTS=5      HIER_COLDSPOT_ANCHOR_VARIANTS=3
HIER_COLDSPOT_SOFT_ONLY=0           HIER_COLDSPOT_PARTIAL_FRONTIER=0
```

**Trace / telemetry (runtime env vars, not constants)**
```text
HIER_GNN_TRACE=0                 HIER_GNN_TRACE_DIR=ml_data/beyondppa_gnn
HIER_GNN_TRACE_MAX_CANDIDATES=512
HIER_PLATEAU_TRACE_DIR=ml_data/beyondppa_gnn/plateau
HIER_PLATEAU_TRACE_PATH=<optional output override>
VIVAPLACE_RUN_ID=<optional attributable run id>
```

Experiments that were tried and not promoted (full recursive bisection,
cluster-room/bridge-corridor modeling, broad weak-hot region reshape, early
strong-soft repair, early swap-lite, deterministic hot-cluster coldspot
selection, GNN candidate reordering at full-suite scale) are recorded in
`ISSUES.md` and `PROGRESS.md`, not here — this document describes only the
active system.
