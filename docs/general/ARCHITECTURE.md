# CongFlow v2 - Architecture

## Current Production System

As of 2026-06-16, v2 is a hierarchy-preserving placer. The active
`MacroPlacer.place()` path is:

```text
derive clusters
  -> grouped DREAMPlace
  -> cluster-consecutive legalize
  -> soft cleanup
  -> congestion-expanded hard/soft regions
  -> region-locked hard/soft relief
  -> exact-gated cluster decompression
  -> region-bounded hard-hard / hard-soft / soft-soft swaps
  -> proxy-aware coldspot tightening
  -> clamp in bounds
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
[OBJECTIVES.md](OBJECTIVES.md).

Current verified smoke:

```text
uv run evaluate src/main.py -b ibm10
proxy=1.6759  VALID
```

Current verified full sweep:

```text
uv run evaluate src/main.py --all
AVG 1.4452  17/17 VALID  0 overlaps  520.08s
```

## Main Components

| Path | Current role |
|---|---|
| `src/main.py` | Evaluator entrypoint. Exposes `MacroPlacer`; applies `V2_SEED` only. |
| `src/placer/pipeline/macro_placer.py` | Entire production flow. `_place_impl()` calls `_hierarchy_floorplan()` and raises if it cannot run. |
| `src/dreamplace_bridge/` | Converts ICCAD04 pb/plc to Bookshelf, injects cluster grouping, launches DREAMPlace, reads hard/soft positions back. |
| `src/placer/local_search/clusters.py` | Derives hard clusters, owned/bridge soft memberships, and region boxes. |
| `src/placer/legalize/spiral.py` | Legalizes hard macros, including cluster-consecutive order support. |
| `src/placer/local_search/relocation.py` | Hard and soft relocation used by region-locked relief. |
| `src/placer/local_search/region_expand.py` | Expands hot cluster regions toward colder neighboring congestion bands. |
| `src/placer/local_search/cluster_decompress.py` | Exact-gated decompression of hot hierarchy blobs with hierarchy-quality checks. |
| `src/placer/local_search/hierarchy_swaps.py` | Region-bounded hard-hard, hard-soft, and soft-soft swap relief. |
| `src/placer/local_search/fields.py` | Congestion/coldspot fields used by relocation and coldspot tightening. |
| `src/placer/local_search/lsmc_explore.py` | Only `_coldspot_cluster_kick()` remains. Generic LSMC was deleted. |
| `src/placer/scoring/exact.py` | Exact TILOS proxy wrapper. |
| `src/placer/scoring/incremental.py` | Incremental scorer for relocation and hierarchy-bounded swap moves. Proxy-only cycle APIs were deleted. |
| `src/eda_io/` | Standard EDA file I/O; converts inputs to the same benchmark object. |

Deleted active modules include `src/placer/ml/`, `local_search/two_opt.py`,
`local_search/soft_moves.py`, and `local_search/hard_soft.py`.

## Hierarchy Pipeline

### 1. Cluster Derivation

`derive_hard_clusters()` builds hard-macro communities from low-fanout net
connectivity. Because ICCAD04 netlists are flat and direct hard-to-hard nets are
sparse, the cluster logic accounts for hard/soft connectivity and maps carefully
between placement-order indices and `modules_w_pins` indices.

Controls:

```text
V2_CLUSTER_MAX_FANOUT=8
V2_CLUSTER_MIN_EDGE=2
```

`derive_soft_cluster_roles()` classifies soft macros as:

- **owned** when one hard cluster dominates the soft macro's connectivity;
- **bridge** when multiple clusters have comparable affinity.

Owned softs can move with their cluster. Bridge softs receive corridor-style
regions spanning the clusters they connect.

### 2. Grouped DREAMPlace

`run_dreamplace()` accepts `cluster_groups` and `group_weight`. The bridge
creates synthetic clique nets among each cluster's hard and soft members so
DREAMPlace pulls the subsystem together during global placement.

Control:

```text
V2_HIER_GROUP_WEIGHT=8
```

The current production path requires DREAMPlace. If the bridge is unavailable,
there is no proxy fallback.

### 3. Cluster-Consecutive Legalization

Grouped DP output can overlap. The hard legalizer runs with an order that keeps
cluster members adjacent:

```text
largest clusters -> larger macros inside each cluster -> unclustered macros
```

A default-order safety pass follows to guarantee hard legality.

### 4. Soft Cleanup

The path runs soft relocation by congestion and density using
`_soft_relocation_moves()`. Soft overlap is legal, so this phase optimizes
placement quality and soft positions without hard legality constraints.

### 5. Region-Locked Relief

`compute_region_bbox()` creates a soft fence around each hard cluster, and
`compute_soft_region_bbox()` gives owned and bridge softs corresponding
regions. Hot cluster regions are expanded toward colder neighboring congestion
bands before relief. Hard and soft relocation then rank candidates by
congestion/density while adding a penalty for leaving the assigned region.
Out-of-region moves are only accepted when the exact proxy gain clears
`V2_HIER_REGION_ESCAPE_MIN`.

Controls:

```text
V2_HIER_REGION_RELIEF=1
V2_HIER_REGION_DENSITY=0.65
V2_REGION_BIAS=1.0
V2_HIER_REGION_ROUNDS=2
V2_HIER_REGION_BUDGET_S=40
V2_HIER_REGION_MARGIN=0
V2_HIER_REGION_SINGLETON=0.05
V2_HIER_REGION_ESCAPE_MIN=0.002
V2_HIER_BRIDGE_SOFTS=1
V2_HIER_CONG_EXPAND_REGIONS=1
```

When no `region_bbox` is supplied, relocation remains the ordinary exact-gated
move primitive. The current production caller always uses it through the
hierarchy relief loop.

### 6. Cluster Decompression

`_cluster_decompression_relief()` expands hot clusters inside their expanded
regions to create local routing channels. Candidates are hard-legalized, owned
softs move with their clusters, bridge softs are nudged toward their corridor
centroid, and the move is accepted only if exact proxy improves while the
hierarchy-quality metric remains within budget.

Controls:

```text
V2_HIER_DECOMPRESS=1
V2_HIER_DECOMPRESS_ROUNDS=2
V2_HIER_DECOMPRESS_BUDGET_S=18
V2_HIER_QUALITY_BUDGET=0.03
```

### 7. Region-Bounded Swaps

`_region_bounded_swap_relief()` runs hard-hard, hard-soft, and soft-soft swaps
against live congestion and density fields. In-region swaps use the exact proxy
accept gate; outside-region swaps must clear the escape threshold. The current
accepted system keeps a wider soft candidate list because the largest remaining
congestion cases are most sensitive to soft-soft and mixed soft movement.

Controls:

```text
V2_HIER_REGION_SWAPS=1
V2_HIER_HARD_SWAP_K=16
V2_HIER_SOFT_SWAP_K=48
V2_HIER_SWAP_MIN_GAIN=0.00001
V2_HIER_SWAP_DENSITY_FIELD=1
```

### 8. Coldspot Tightening

`_coldspot_cluster_kick()` gathers a selected cluster into a low-congestion
window, co-moves connected soft macros, and legalizes the hard macros. The
hierarchy path now accepts a kick only when exact proxy improves and the
hierarchy-quality metric remains within budget.

Controls:

```text
V2_HIER_COLDSPOT_KICK=1
V2_HIER_COLDSPOT_BUDGET=0.0
V2_HIER_COLDSPOT_TOTAL=0.0
V2_HIER_COLDSPOT_MIN_GAIN=0.0001
V2_HIER_COLDSPOT_QUALITY_BUDGET=0.01
V2_HIER_COLDSPOT_ROUNDS=8
V2_HIER_COLDSPOT_BUDGET_S=30
```

This is not the old generic LSMC path. It is a narrow hierarchy-tightening
helper.

## Scoring And Legality

Hard requirements remain unchanged:

- Fixed macros stay fixed.
- Hard macros must not overlap.
- All macro centers must be in bounds.
- Soft macros may overlap.

The hierarchy path returns `torch.float32` center coordinates for all macros.
`_clamp_in_bounds()` runs on every returned placement.

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
uv run evaluate src/main.py -b ibm10
uv run evaluate src/main.py --all
```

Historical verifiers for deleted proxy-only code were removed with that code.

## Historical Notes

The large proxy optimizer documented in older progress entries achieved strong
leaderboard proxy numbers, but the user-selected system is now the hierarchy
path. Keep historical measurements in `PROGRESS.md` for context, but do not
reintroduce proxy-only code unless explicitly asked to restore that path.
