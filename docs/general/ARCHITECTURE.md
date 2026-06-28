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
        radial expansion, synthetic-clearance push-apart; lowest exact
        proxy enters relief)
  -> congestion-expanded hard/soft hierarchy regions
  -> exact-gated local micro-shift polish
  -> exact-gated cluster decompression (hierarchy-quality budgeted)
  -> region-bounded hard-hard / hard-soft / soft-soft swaps
  -> post-swap hard/soft relocation polish + micro-shift replay
  -> coldspot tightening (congestion-driven local relief, graph-local fallback)
  -> bounded survivor-pool search
  -> final hard-legality, bounds, and hierarchy-quality audit
       (rolls back to the best saved audit-passing checkpoint if needed)
  -> return center coordinates for hard and soft macros
```

Passes advance on gain, not fixed repeat counts: each stage keeps running
while its most recent exact-proxy improvement exceeds
`HIER_PLATEAU_PROXY_GAIN`, then moves on.

```text
proxy_cost = wirelength + 0.5 * density + 0.5 * congestion
```

The exact proxy is still the accept gate for every committed move, and still
drives evaluator reporting. But the system optimizes for **hierarchy
preservation** first: it deliberately keeps connected subsystems together
even when a flatter, more-spread placement would score lower proxy. The
structural reasoning behind this is in [OBJECTIVES.md](OBJECTIVES.md).

Current full-suite result:

```text
uv run evaluate src/main.py --all
AVG 1.1999  17/17 VALID  0 overlaps  all hierarchy audits passed  1147.08s
```

NG45 (RTL instance-path hierarchy tags):

```text
uv run evaluate src/main.py --ng45
AVG 0.7320  4/4 VALID  0 overlaps
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
| `src/placer/pipeline/hierarchy_floorplan.py` | The hierarchy pipeline itself: seed portfolio, region relief, swaps, coldspot, survivor search. |
| `src/placer/pipeline/hierarchy_context.py` | Shared `PlacementState`, `PassContext`, `PassResult`, `PlateauTelemetry` used across pipeline stages. |
| `src/placer/local_search/hierarchy_model.py` | Inferred hierarchy: hard clusters, soft roles, cluster graph, region builders. |
| `src/placer/local_search/clusters.py` | Hard-cluster derivation, oversized-cluster splitting, region-box primitives. |
| `src/placer/local_search/relocation.py` | Hard and soft relocation used by region-locked relief and post-swap polish. |
| `src/placer/local_search/hierarchy_swaps.py` | Region-bounded hard-hard, hard-soft, soft-soft swap relief. |
| `src/placer/local_search/cluster_decompress.py` | Exact-gated decompression of hot hierarchy blobs. |
| `src/placer/local_search/region_expand.py` | Expands hot cluster regions toward colder congestion bands. |
| `src/placer/local_search/lsmc_explore.py` | Coldspot kick candidate generation. |
| `src/placer/local_search/survivor_search.py` | Bounded survivor-pool search over hierarchy-safe cluster moves; CUDA-rankable. |
| `src/placer/local_search/fields.py` | Congestion/coldspot fields used by relocation and coldspot tightening. |
| `src/placer/local_search/structural_fields.py` | Deterministic BeyondPPA-style edge-keepout/grid-alignment/notch metrics; opt-in candidate reordering only. |
| `src/placer/local_search/gnn_trace.py` | JSONL trace + plateau telemetry writers (diagnostic only). |
| `src/placer/scoring/exact.py` | Exact TILOS proxy wrapper. |
| `src/placer/scoring/incremental.py` | Incremental scorer for relocation and swap moves. |
| `src/placer/legalize/spiral.py` | Hard-macro legalization, with cluster-consecutive order support. |
| `src/dreamplace_bridge/` | ICCAD04 pb/plc → Bookshelf, cluster grouping injection, DREAMPlace launcher, read-back. |
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
required — there is no proxy-only fallback.

### 3. Seed Portfolio Selection

Grouped DREAMPlace is one candidate seed among several: legalized
`initial.plc`, two DP/initial blends, a radial expansion from the DP basin,
and a synthetic-clearance push-apart from the DP basin. All are exact-scored;
the lowest-proxy seed enters hierarchy relief.

### 4. Cluster-Consecutive Legalization

Hard macros legalize in an order that keeps cluster members adjacent
(largest clusters first, then connectivity-pressure × area within each
cluster, then unclustered macros), followed by a default-order safety pass
to guarantee legality.

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

Structural candidate ordering (`structural_fields.py`) can add edge-keepout,
grid-alignment, and notch-avoidance penalties into relocation candidate
ranking when `HIER_OBJECTIVE_STRUCTURAL_WEIGHT > 0` (default `0.0`). It only
reorders proposals — legality, region, hierarchy-quality, and exact-proxy
gates are unaffected.

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

### 11. Survivor Search and Final Audit

A bounded go-with-the-winners search generates hierarchy-safe cluster move
variants from multiple states, optionally GPU-ranks cheap candidate scores,
and exact-scores the top survivors. The pipeline finishes with a hard-legality
margin audit and a final hierarchy-quality audit against the selected seed,
rolling back to the best saved audit-passing checkpoint if the final state
fails.

### 12. Trace Logging and Plateau Telemetry

No learned ranker is active in production. `HIER_GNN_TRACE=1` writes
schema-v1 JSONL candidate traces (relocation, swaps, decompression, coldspot)
for offline GNN experiments; it does not change placement output.
`HIER_PLATEAU_TRACE` (default on) writes lightweight pass-level telemetry
(proxy before/after, elapsed time, accept rate, plateau flag) for future
ML/DL scheduling work. See
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
uv run pytest test/verification/test_structural_fields.py -q
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
HIER_REGION_ESCAPE_MIN=0.002    HIER_REGION_COMPONENT_EXPAND=True
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
HIER_DECOMPRESS_LOCAL_COMPONENT=True   HIER_DECOMPRESS_LOCAL_SHIFT_FRAC=0.20
```

**Swaps**
```text
HIER_HARD_SWAP_K=16          HIER_SOFT_SWAP_K=48
HIER_SWAP_MIN_GAIN=0.00001   HIER_GPU_RANK_SWAP_CANDIDATES=auto
HIER_GPU_RANK_MIN_CANDIDATES=512
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
HIER_PLATEAU_TRACE=1             HIER_PLATEAU_TRACE_DIR=ml_data/beyondppa_gnn/plateau
```

Experiments that were tried and not promoted (full recursive bisection,
cluster-room/bridge-corridor modeling, broad weak-hot region reshape, early
strong-soft repair, early swap-lite, deterministic hot-cluster coldspot
selection, GNN candidate reordering at full-suite scale) are recorded in
`ISSUES.md` and `PROGRESS.md`, not here — this document describes only the
active system.
