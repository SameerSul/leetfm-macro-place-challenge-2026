# CongFlow v2 - Architecture

## Current Production System

As of 2026-06-22, v2 is a hierarchy-preserving placer with exact-prescored
seed portfolio selection. The active
`MacroPlacer.place()` path is:

```text
benchmark input
  -> build first-class HierarchyModel
       - selectively split oversized bridge-connected flat clusters
       - classify owned/bridge soft roles
       - record inter-cluster edge weights and confidence
  -> grouped DREAMPlace hierarchy floorplan
  -> cluster-consecutive hard legalization
  -> prescore seed portfolio:
       - grouped DREAMPlace seed
       - initial.plc seed
       - DREAMPlace/initial blend seeds
       - radial expansion seed
       - synthetic-clearance seed
       - exact proxy selects the seed that enters hierarchy relief
  -> build congestion-expanded hard and soft hierarchy regions
  -> exact-gated local micro-shift polish
  -> exact-gated cluster decompression with composite hierarchy quality
  -> budget-aware interleaved soft repair
  -> region-bounded hard-hard / hard-soft / soft-soft swaps
  -> post-swap micro-shift replay
  -> post-swap hard propose-all relocation with spare-budget additive candidates
  -> post-swap soft relocation with spare-budget additive candidates
  -> plateau-switched strong soft repair when telemetry shows useful spare work
  -> coldspot tightening:
       - refresh current congestion field and cold-cell graph memory
       - generate coldspot kick candidates
       - co-move owned/bridge soft macros
       - legalize candidate hard macros
       - expand local graph border through adjacent open cold cells
       - apply hard-core padding
       - run graph-bordered local swaps and hard/soft relocations
       - graph-rank generated outcomes when the GNN selector is off
       - exact proxy + hierarchy-quality gate before commit
  -> graph-local fallback when no coldspot kick commits:
       - select hottest eligible current clusters
       - reuse the same graph-expanded border
       - run the same swaps and relocations without a kick
       - exact proxy + hierarchy-quality gate before commit
  -> post-coldspot micro-shift replay
  -> final scorer-compatible hard legality margin audit
  -> final legality and bounds checks
  -> return center coordinates for hard and soft macros
```

The former proxy-optimized path has been removed from active code. This includes
candidate restarts, R2 interleaving, 2-opt, hard-soft swaps, soft swaps,
hard-soft-soft cycles, generic LSMC exploration, generic cluster LSMC kicks,
ML ranker defaults, and the proxy-only scorer methods used by those passes.

The exact proxy still matters for evaluation and for local accept gates:

```text
proxy_cost = wirelength + 0.5 * density + 0.5 * congestion
```

But the system now intentionally trades proxy for hierarchy. Historical
`--all` scores around 1.12 in `PROGRESS.md` describe the deleted proxy path, not
the current hierarchy output.

The structural objectives that drive the hierarchy flow are documented in
[OBJECTIVES.md](OBJECTIVES.md). BeyondPPA-style deterministic structural
ranking and GNN trace logging are documented in
[../ml_nn/beyondppa_results/](../ml_nn/beyondppa_results/).

Current verified full sweep after the six-stage hierarchy-aware revamp
through Stage 6 audit:

```text
uv run evaluate src/main.py --all
AVG 1.1817  17/17 VALID  0 overlaps  1383.28s
```

The same active behavior without the audit-only final logging swept at
`AVG 1.1796`, 17/17 VALID, 0 overlaps; the difference is run-to-run variance,
because the Stage 6 audit does not move macros.

Last lower-proxy accepted hierarchy full sweep before the graph-local and
six-stage architecture revamps:

```text
uv run evaluate src/main.py --all
AVG 1.3631  17/17 VALID  0 overlaps  602.76s
```

The graph-select / graph-target / graph-mask stack was also swept valid at
`AVG 1.3811`, but it was not promoted over the accepted hierarchy result.

## Main Components

| Path | Current role |
|---|---|
| `src/main.py` | Evaluator entrypoint. Exposes `MacroPlacer`; applies `SEED` only. |
| `src/utils/` | Runtime CUDA/numba config, logging shim, and accepted hierarchy constants. |
| `src/placer/pipeline/macro_placer.py` | Entire production flow. `_place_impl()` calls `_hierarchy_floorplan()` and raises if it cannot run. |
| `src/placer/pipeline/hierarchy_context.py` | Shared `PlacementState`, `PassContext`, `PassResult`, and `PlateauTelemetry` objects used by the hierarchy pipeline orchestration, scheduling, and trace payloads. |
| `src/dreamplace_bridge/` | Converts ICCAD04 pb/plc to Bookshelf, injects cluster grouping, launches DREAMPlace, reads hard/soft positions back. |
| `src/placer/local_search/hierarchy_model.py` | First-class inferred hierarchy model: hard clusters, soft roles, cluster graph, and reusable region builders. |
| `src/placer/local_search/clusters.py` | Low-level hard-cluster, oversized-cluster splitting, soft-role, and region-box primitives used by `HierarchyModel`. |
| `src/placer/legalize/spiral.py` | Legalizes hard macros, including cluster-consecutive order support. |
| `src/placer/local_search/relocation.py` | Hard and soft relocation used by region-locked relief. |
| `src/placer/local_search/structural_fields.py` | Deterministic BeyondPPA-style structural metrics used for diagnostics and opt-in hierarchy candidate ordering. |
| `src/placer/local_search/gnn_trace.py` | JSONL trace writers for optional GNN candidate traces and default plateau telemetry for future ML/DL scheduling work. |
| `src/placer/local_search/region_expand.py` | Expands hot cluster regions toward colder neighboring congestion bands. |
| `src/placer/local_search/cluster_decompress.py` | Exact-gated decompression of hot hierarchy blobs with hierarchy-quality checks. |
| `src/placer/local_search/hierarchy_swaps.py` | Region-bounded hard-hard, hard-soft, and soft-soft swap relief. |
| `src/placer/local_search/fields.py` | Congestion/coldspot fields used by relocation and coldspot tightening. |
| `src/placer/local_search/lsmc_explore.py` | Coldspot kick candidate generation remains. Generic LSMC was deleted. |
| `src/placer/scoring/exact.py` | Exact TILOS proxy wrapper. |
| `src/placer/scoring/incremental.py` | Incremental scorer for relocation and hierarchy-bounded swap moves. Proxy-only cycle APIs were deleted. |
| `src/eda_io/` | Standard EDA file I/O; converts inputs to the same benchmark object. |

Deleted active modules include `src/placer/ml/`, `local_search/two_opt.py`,
`local_search/soft_moves.py`, and `local_search/hard_soft.py`.

## Hierarchy Pipeline

### 1. Cluster Derivation

`HierarchyModel.build()` is the single entry point for hierarchy. When macro
names carry slash-separated RTL instance paths, as in NG45, it first derives
hard-macro clusters from useful path prefixes. Otherwise it falls back to
low-fanout net connectivity, selectively splits oversized bridge-connected flat
clusters, classifies soft macros, and records an inter-cluster weighted graph.
Full recursive weighted bisection was tested in the six-stage revamp and
removed from production code because it regressed full-suite proxy. The active
split rule is narrower: a flat cluster must exceed 40% of hard macros, the
design must expose at least five bridge softs, and the accepted split must
reduce leaves near the 15% hard-macro target. Because ICCAD04 netlists are flat
and direct hard-to-hard nets are sparse, the fallback cluster logic accounts for
hard/soft connectivity and maps carefully between placement-order indices and
`modules_w_pins` indices.

Constants in `src/utils/constants.py`:

```text
CLUSTER_MAX_FANOUT=8
CLUSTER_MIN_EDGE=2
HIER_OVERSIZE_CLUSTER_SPLIT=1
HIER_TAG_PREFIX_CLUSTERING=1
HIER_TAG_PREFIX_MAX_DEPTH=5
HIER_TAG_PREFIX_MIN_GROUP=2
HIER_TAG_PREFIX_MIN_COVERAGE=0.25
HIER_OVERSIZE_CLUSTER_START_FRAC=0.40
HIER_OVERSIZE_CLUSTER_TARGET_FRAC=0.15
HIER_OVERSIZE_CLUSTER_TARGET_TOL=1.10
HIER_OVERSIZE_CLUSTER_MIN_BRIDGE_SOFTS=5
HIER_OVERSIZE_CLUSTER_MIN_SIZE=6
HIER_OVERSIZE_CLUSTER_MAX_CUT_RATIO=0.45
```

`derive_soft_cluster_roles()` classifies soft macros as:

- **owned** when one hard cluster dominates the soft macro's connectivity;
- **bridge** when multiple clusters have comparable affinity.

Owned softs can move with their cluster. Bridge softs receive soft regions
spanning the clusters they connect. Cluster-room and bridge-corridor modeling
was tested in the six-stage revamp and then removed because it was too
restrictive on packed designs.

### 2. Grouped DREAMPlace

`run_dreamplace()` accepts `cluster_groups` and `group_weight`. The bridge
creates synthetic clique nets among each cluster's hard and soft members so
DREAMPlace pulls the subsystem together during global placement.

Control:

```text
HIER_GROUP_WEIGHT=8
```

The current production path requires DREAMPlace. If the bridge is unavailable,
there is no proxy fallback.

### 3. Seed Portfolio Prescoring

Grouped DREAMPlace remains one candidate seed, but the flow no longer assumes
it is the best proxy basin. `_hierarchy_floorplan()` exact-prescores a small
portfolio before region relief:

- grouped DREAMPlace;
- legalized `initial.plc`;
- two DREAMPlace/initial blend seeds;
- radial expansion from the DREAMPlace basin;
- synthetic-clearance push-apart from the DREAMPlace basin.

The lowest exact-proxy seed enters the normal hierarchy relief pipeline. This
is proxy-oriented: some selected initial seeds are less compact than grouped
DREAMPlace, but fixed macros, hard legality, hierarchy regions, hierarchy
quality checks, and exact-proxy gates still constrain later moves.

Constants in `src/utils/constants.py`:

```text
HIER_SEED_PORTFOLIO=1
HIER_SEED_BLEND_ALPHAS=0.35,0.65
HIER_SEED_EXPANSION_FRAC=0.06
HIER_SEED_SYNTHETIC_CLEARANCE=1
HIER_SEED_CLEARANCE_FRAC=0.08
HIER_SEED_CLEARANCE_ITERS=3
HIER_SEED_CLEARANCE_AREA_PCT=97
```

### 4. Cluster-Consecutive Legalization

Grouped DP output can overlap. The hard legalizer runs with an order that keeps
cluster members adjacent:

```text
largest clusters -> connectivity-pressure x area inside each cluster -> unclustered macros
```

Set `HIER_LEGALIZE_CONNECTIVITY_ORDER=0` to restore the prior larger-macro-first
ordering inside each cluster.

A default-order safety pass follows to guarantee hard legality.

### 5. Soft Cleanup

The path runs soft relocation by congestion and density using
`_soft_relocation_moves()`. Soft overlap is legal, so this phase optimizes
placement quality and soft positions without hard legality constraints.

### 6. Region-Locked Relief

`HierarchyModel.hard_regions()` and `HierarchyModel.soft_regions()` create the
active production region boxes. Hot cluster regions are expanded toward colder
neighboring congestion bands before relief. Hard and soft relocation then rank
candidates by a congestion-heavy blended proposal field and by density while
adding a penalty for leaving the assigned region.
Out-of-region moves are only accepted when the exact proxy gain clears
`HIER_REGION_ESCAPE_MIN`.

Constants in `src/utils/constants.py`:

```text
HIER_REGION_RELIEF=1
HIER_REGION_DENSITY=0.65
REGION_BIAS=1.0
HIER_REGION_ROUNDS=2
HIER_REGION_BUDGET_S=40
HIER_REGION_MARGIN=0
HIER_REGION_SINGLETON=0.05
HIER_REGION_ESCAPE_MIN=0.002
HIER_BRIDGE_SOFTS=1
HIER_CONG_EXPAND_REGIONS=1
HIER_CONGESTION_WEIGHTED_PROPOSALS=1
HIER_PROPOSAL_CONGESTION_WEIGHT=2.5
HIER_PROPOSAL_DENSITY_WEIGHT=1.0
HIER_PROPOSAL_HIERARCHY_AWARE=1
HIER_PROPOSAL_OUTSIDE_RELIEF_MARGIN=0.08
```

When no `region_bbox` is supplied, relocation remains the ordinary exact-gated
move primitive. The current production caller always uses it through the
hierarchy relief loop.

### 6a. BeyondPPA Structural Candidate Ordering

`src/placer/local_search/structural_fields.py` implements deterministic
structural penalties for edge keepout, grid alignment, and notch avoidance.
When enabled, `src/placer/local_search/relocation.py` adds the local structural
delta into existing hard and soft relocation candidate ordering.

Constants in `src/utils/constants.py`:

```text
HIER_OBJECTIVE_STRUCTURAL_WEIGHT=0.0
HIER_KEEP_OUT_WEIGHT=0.2
HIER_GRID_ALIGN_WEIGHT=0.2
HIER_NOTCH_WEIGHT=0.6
```

This is not a separate BeyondPPA placement path. The structural term only
changes proposal order. Fixed macros, bounds, hard legality, hierarchy-region
constraints, hierarchy-quality gates, and exact-proxy accept gates remain the
authority for every committed move. The production default is
`HIER_OBJECTIVE_STRUCTURAL_WEIGHT=0.0`, so default behavior is unchanged.

### 7. Cluster Decompression

`_cluster_decompression_relief()` expands hot clusters inside their expanded
regions to create local routing channels. Candidates are hard-legalized, owned
softs move with their clusters, bridge softs are nudged toward their corridor
centroid, and the move is accepted only if exact proxy improves while the
composite hierarchy-quality metric remains within budget. The quality metric
combines the old mean radius term with bounding-box spread and a small
nearest-cluster crowding penalty, keeping the scale near the prior gate.

Constants in `src/utils/constants.py`:

```text
HIER_DECOMPRESS=1
HIER_DECOMPRESS_ROUNDS=2
HIER_DECOMPRESS_BUDGET_S=18
HIER_QUALITY_BUDGET=0.03
HIER_QUALITY_RADIUS_WEIGHT=0.75
HIER_QUALITY_BBOX_WEIGHT=0.20
HIER_QUALITY_CROWD_WEIGHT=0.05
```

### 8. Region-Bounded Swaps

`_region_bounded_swap_relief()` runs hard-hard, hard-soft, and soft-soft swaps
against the congestion-heavy proposal field and live density field. In-region
swaps use the exact proxy accept gate; outside-region swaps must clear the
escape threshold. The current accepted system keeps a wider soft candidate list
because the largest remaining congestion cases are most sensitive to soft-soft
and mixed soft movement.

Constants in `src/utils/constants.py`:

```text
HIER_REGION_SWAPS=1
HIER_HARD_SWAP_K=16
HIER_SOFT_SWAP_K=48
HIER_SWAP_MIN_GAIN=0.00001
HIER_SWAP_DENSITY_FIELD=1
HIER_CONGESTION_WEIGHTED_PROPOSALS=1
HIER_PROPOSAL_HIERARCHY_AWARE=1
```

### 9. Post-Swap Polish

`_relocation_moves(..., propose_all=True)` runs once after swaps on CUDA
systems. Unlike the rejected pre-swap hard propose-all variants, this pass sees
the final swap-relieved state and uses a stronger exact-gain margin, so it only
accepts sparse cleanup moves.

`_soft_relocation_moves()` then runs as an ordinary post-swap soft polish pass
with a small exact-gain margin. This is not the rejected soft propose-all path;
it keeps sequential exact-gated soft relocation and only cleans up the final
swap-relieved state.

After the normal post-swap hard and soft polish passes, a budget-aware strong
soft repair may run. It uses larger soft target pools, two field orderings, and
the same exact-proxy accept gate to spend remaining local pass budget on soft
macros without reopening hard legality. The scheduler starts this pass only
when a small spare-time window remains and recent pass telemetry indicates
plateaued hard/soft cleanup or a useful soft-relocation signal.

The accepted Stage-3 flow also reruns `_micro_shift_polish()` after swaps. This
exact-gated replay is default-on and uses the same tiny one/two-cell moves as
the earlier in-region micro-shift pass.

When local pass budget remains, deterministic candidate prefixes are preserved
and a small additive tail is exact-checked for hard propose-all relocation and
swap-local refinement. This is additive exploration only: it does not replace
the deterministic ordering or weaken legality, region, hierarchy-quality, or
exact-proxy gates.

Constants in `src/utils/constants.py`:

```text
HIER_POST_SWAP_MICRO_SHIFT=1
HIER_POST_SWAP_MICRO_SHIFT_BUDGET_S=8
HIER_POST_RELOC_PROPOSE_ALL=auto
HIER_POST_RELOC_PROPOSE_TOP_M=16
HIER_RELOC_PROPOSE_MIN_GAIN=0.0005
HIER_POST_SOFT_RELOC=1
HIER_POST_SOFT_RELOC_TOP_K=256
HIER_POST_SOFT_RELOC_MIN_GAIN=0.0005
HIER_STRONG_SOFT_REPAIR=1
HIER_STRONG_SOFT_REPAIR_BUDGET_S=12
HIER_STRONG_SOFT_REPAIR_MIN_SPARE_S=2
HIER_STRONG_SOFT_REPAIR_ROUNDS=2
HIER_STRONG_SOFT_REPAIR_TOP_K=512
HIER_STRONG_SOFT_REPAIR_TARGETS=12
HIER_STRONG_SOFT_REPAIR_MIN_GAIN=0.00005
HIER_STRONG_SOFT_REPAIR_WL_PREFILTER=0.0005
HIER_PLATEAU_TELEMETRY=1
HIER_BUDGET_AWARE_SCHEDULING=1
HIER_PLATEAU_ACCEPT_RATE=0.002
HIER_PLATEAU_PROXY_GAIN=0.0005
HIER_ADDITIVE_CANDIDATE_POOLS=1
HIER_ADDITIVE_RELOC_EXTRA_TOP_K=8
HIER_ADDITIVE_SWAP_EXTRA_K=4
HIER_ADDITIVE_MIN_SPARE_S=2.0
```

### 10. Coldspot Tightening

`_coldspot_cluster_kick()` gathers a selected cluster into a low-congestion
window, co-moves connected soft macros, and legalizes the hard macros. The
hierarchy path then refines each kicked candidate inside a slightly expanded
local cluster box before accepting it only when exact proxy improves and the
hierarchy-quality metric remains within budget.

Constants in `src/utils/constants.py`:

```text
HIER_COLDSPOT_KICK=1
HIER_COLDSPOT_BUDGET=0.0
HIER_COLDSPOT_TOTAL=0.0
HIER_COLDSPOT_MIN_GAIN=0.0001
HIER_COLDSPOT_QUALITY_BUDGET=0.01
HIER_COLDSPOT_MIN_FIELD_GAP=0.02
HIER_COLDSPOT_ROUNDS=8
HIER_COLDSPOT_BUDGET_S=30
HIER_COLDSPOT_LOCAL_REFINE=1
HIER_COLDSPOT_LOCAL_HARD_PAD_FRAC=0.50
HIER_COLDSPOT_LOCAL_MIN_PAD_CELLS=1
HIER_COLDSPOT_LOCAL_MAX_PAD_FRAC=0.12
HIER_COLDSPOT_LOCAL_SOFT_ESCAPE_MIN=0.0025
HIER_COLDSPOT_GRAPH_SELECT=1
HIER_COLDSPOT_GRAPH_SELECT_CANDIDATES=4
HIER_COLDSPOT_GRAPH_SELECT_TOP_K=2
HIER_COLDSPOT_GRAPH_TARGET_POOL=1
HIER_COLDSPOT_GRAPH_MASK_GATING=1
HIER_COLDSPOT_GRAPH_FALLBACK=1
HIER_COLDSPOT_GRAPH_FALLBACK_TOP_K=3
HIER_COLDSPOT_ADAPTIVE_REGIONS=1
HIER_COLDSPOT_MEMORY_COLD_PCT=35
HIER_COLDSPOT_ADAPTIVE_MAX_CELLS=5
```

Rounds with no cheap hot-cluster to cold-window field gap are skipped before
candidate generation and exact candidate scoring.

This is not the old generic LSMC path. It is a narrow hierarchy-tightening
helper. Candidate-local refinement runs hard-hard and hard-soft swaps with the
kicked hard cluster locked in the local box, plus soft-soft swaps and soft
relocation that may leave the local box only after a `0.0025` exact-proxy gain.
The local box includes owned/bridge soft macros, but its base pad is derived
from the kicked hard-core max dimension rather than the soft-inclusive bbox.
The phase tracks a current cold-cell grid from the active congestion field,
refreshes it after every finalized coldspot kick, masks out cells occupied by
the candidate, and expands the pre-margin local border through adjacent open
cold cells before applying the hard-core pad. This lets finalized cluster
locations use nearby coldspots for local relief while preserving swap and
soft-locked relocation room. The graph also supplies coldspot-local relocation
target pools, gates relocation targets by graph mask, and ranks generated
outcomes before exact gating when the GNN selector is disabled. When no
coldspot kick commits, the graph-local fallback runs the same bordered swaps
and relocations on the current placement for the hottest eligible clusters.
Production then reruns `_micro_shift_polish()` once more with
`HIER_POST_COLDSPOT_MICRO_SHIFT=1`; deterministic hot-cluster coldspot
selection was tested and removed after regressing the full sweep.

### 11. Trace Logging And Plateau Telemetry

No GNN model is active in production. The current GNN-related implementation is
opt-in trace logging attached to the hierarchy flow. These are runtime
environment variables, not placement constants:

```text
HIER_GNN_TRACE=0
HIER_GNN_TRACE_DIR=ml_data/beyondppa_gnn
HIER_GNN_TRACE_RUN=<optional run id; default file is trace.jsonl>
HIER_GNN_TRACE_MAX_CANDIDATES=512
HIER_GNN_TRACE_PATH=<optional direct JSONL path>
```

When enabled, the logger records schema-v1 hierarchy candidate labels for
relocation, region swaps, cluster decompression, and coldspot tightening, plus
pass summaries and final placement summaries as JSONL. It does not change
candidate ordering or acceptance. A deterministic schema-v1 trace-to-graph
dataset builder now lives at `scripts/gnn/build_gnn_dataset.py`. The Stage-G3
offline baseline entrypoint is `scripts/gnn/train_gnn_baseline.py`; it trains and
evaluates candidate-feature-only rankers without changing placement behavior.

Plateau telemetry is separate and default-on. It writes lightweight pass-level
rows for pass name, proxy before/after, elapsed time, candidate/legal/scored
counts, accepts, accept rate, plateau flag, and scheduler decisions:

```text
HIER_PLATEAU_TRACE=1
HIER_PLATEAU_TRACE_DIR=ml_data/beyondppa_gnn/plateau
HIER_PLATEAU_TRACE_RUN=<optional run id; default file is plateau_telemetry.jsonl>
HIER_PLATEAU_TRACE_PATH=<optional direct JSONL path>
```

Stage G3 has an accepted default-off offline baseline artifact, Stage G4 has an
accepted default-off offline macro-net ranker artifact, and Stage G5 has a
smoke-accepted default-off relocation-only candidate-reordering hook. The
Stage G6 full-suite run was legal but not promoted because average proxy and
runtime regressed. The broader target is a default-off hierarchy-flow assistant
that can rank, propose, select, budget, and diagnose work inside existing
hierarchy operators while preserving all deterministic placement gates.

## Scoring And Legality

Hard requirements remain unchanged:

- Fixed macros stay fixed.
- Hard macros must not overlap.
- All macro centers must be in bounds.
- Soft macros may overlap.

The hierarchy path returns `torch.float32` center coordinates for all macros.
`_clamp_in_bounds()` runs on every returned placement.

The pipeline now carries hard positions, soft positions, and current exact proxy
through `PlacementState`. Pass summaries use `PassResult`, which keeps trace
payloads structured as the pass orchestration is split out of
`macro_placer.py`.

Exact proxy scoring is still used by:

- evaluator reports,
- initial hierarchy score measurements,
- soft and hard relocation accept gates,
- cluster decompression, region-bounded swaps, and coldspot tightening gates.

## Verification

Current focused checks:

```bash
uv run python -m py_compile $(find src -type f -name "*.py")
uv run python test/verification/_verify_region_escape_gate.py
uv run python test/verification/_verify_score_region_swaps.py
uv run python test/verification/_verify_coldspot_kick.py ibm10
uv run pytest test/verification/test_structural_fields.py -q
uv run evaluate src/main.py -b ibm10
uv run evaluate src/main.py --all
```

Historical verifiers for deleted proxy-only code were removed with that code.

## Historical Notes

The large proxy optimizer documented in older progress entries achieved strong
leaderboard proxy numbers, but the user-selected system is now the hierarchy
path. Keep historical measurements in `PROGRESS.md` for context, but do not
reintroduce proxy-only code unless explicitly asked to restore that path.
