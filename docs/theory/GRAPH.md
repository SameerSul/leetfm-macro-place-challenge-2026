# Graph Data Structure

This document describes the graph data structure currently used by the
hierarchy placer, the algorithms implemented on top of it, and why it exists.

## Purpose

The graph models usable cold space on the placement grid. Its first use is in
coldspot tightening, where a kicked cluster should be allowed to use adjacent
low-congestion empty space without turning the move into an unconstrained global
search.

The graph is not the circuit netlist. It is not the hierarchy graph. It is a
runtime spatial graph derived from the current congestion field and candidate
macro occupancy.

## Nodes And Edges

Each placement grid cell is a graph node:

```text
node = (row, col)
```

Edges are implicit 4-neighbor adjacency:

```text
(r, c) connected to (r - 1, c)
(r, c) connected to (r + 1, c)
(r, c) connected to (r, c - 1)
(r, c) connected to (r, c + 1)
```

Diagonal cells are not adjacent. This keeps expansion conservative and avoids
connecting through corners where there may not be meaningful placement room.

## Node Labels

The implementation stores node labels as boolean matrices:

```text
cold_memory[r, c]
occupied[r, c]
open_cold[r, c] = cold_memory[r, c] and not occupied[r, c]
```

`cold_memory` is generated from the current congestion field:

```python
threshold = percentile(field, HIER_COLDSPOT_MEMORY_COLD_PCT)
cold_memory = field <= threshold
```

`occupied` is generated per candidate by rasterizing hard and soft macro
footprints to the grid. A touched cell is marked occupied.

## Memory Semantics

Despite the name, `cold_memory` is not stale historical accumulation. It is a
current-state memory snapshot.

At the start of each coldspot round, it is replaced from the current congestion
field. After an accepted coldspot kick or graph-local fallback move, the
incremental scorer is rebuilt from the finalized placement and `cold_memory` is
refreshed from the new field.

This avoids using cold cells that were consumed or invalidated by a finalized
move.

## Algorithms

### Cold Cell Extraction

Function:

```text
_remember_cold_cells(field)
```

Algorithm:

1. Read `HIER_COLDSPOT_MEMORY_COLD_PCT`.
2. Compute that percentile of the congestion field.
3. Mark cells at or below that threshold as cold.

This is O(R * C) for an R by C grid plus percentile cost.

### Occupancy Rasterization

Function:

```text
_occupied_cells(hard_xy, soft_xy)
```

Algorithm:

1. For each macro, compute the grid span touched by its bbox.
2. Clamp the span to canvas grid bounds.
3. Mark every touched cell occupied.

This is conservative: a cell touched by any part of a macro is unavailable for
open-cold expansion.

### Adjacent Cold-Component Flood

Function:

```text
_expand_bbox_to_adjacent_cold(xlo, ylo, xhi, yhi, hard_xy, soft_xy)
```

Algorithm:

1. Build `open_cold = cold_memory & ~occupied`.
2. Convert the seed bbox to grid coordinates.
3. Seed a queue with open-cold cells adjacent to the seed bbox.
4. Flood through 4-neighbor open-cold cells up to
   `HIER_COLDSPOT_ADAPTIVE_MAX_CELLS`.
5. Convert the reached cells back into micron coordinates.
6. Expand the pre-margin local bbox to include reached cells.

The flood is bounded by `HIER_COLDSPOT_ADAPTIVE_MAX_CELLS`, so it cannot spread
across the full canvas unless that constant is intentionally raised.

### Graph Candidate Ranking

The coldspot phase generates multiple kicked outcomes and ranks them by graph
features before exact gating. This is a separate stage from the exact accept
rule.

The graph score currently includes:

```text
field relief
log(graph target cells)
log(adaptive cold cells)
log(graph region cells)
small displacement penalty
```

The selector chooses the top `HIER_COLDSPOT_GRAPH_SELECT_TOP_K` non-noop
outcomes for the normal exact-proxy and hierarchy-quality gates. It does not
yet enumerate multiple anchor windows; it ranks generated outcomes for the
selected cluster/window.

`HIER_COLDSPOT_PARTIAL_FRONTIER=False` is a default-off generator experiment
that can add one more outcome for that selected cluster/window. It estimates
connected cold capacity around the anchor, clamps the moved area so the outcome
remains a true subset kick, chooses frontier hard macros with a distance and
low-fanout connectivity heuristic, co-moves directly connected soft macros
when capacity remains, and places cross-cut-heavy macros nearest the border
between the source hotspot and the coldspot. Tiny source clusters are skipped
by default because far 2-of-3 splits are normally hierarchy-quality failures.
After partial hard legalization, a pre-exact split-shape predictor rejects
candidates whose full source-cluster radius, bbox radius, or moved-vs-remaining
centroid separation grows beyond the configured ratios. The graph selector and
exact gates treat surviving partial candidates like any other generated
coldspot outcome. Rejected partial attempts can emit
`hier_coldspot_partial_reject` trace rows, including selector, connectivity,
and split-shape reasons, so default-off tuning can inspect candidates that never
reach exact scoring.

### Graph Target Pools

Coldspot-local hard and soft relocation receive a flat grid-cell target pool
derived from the graph-expanded candidate region. The relocation helpers still
use exact scoring before any commit.

If the graph target pool is empty, relocation falls back to its normal low-field
target selection.

### Graph Mask Gating

Coldspot-local hard and soft relocation reject targets outside the graph-expanded
mask. This keeps relocation closer to the actual connected graph shape rather
than only the rectangle that encloses it.

Current limitation: swaps still use bbox-style region checks.

### Graph-Local Fallback

The coldspot phase does not stop when no kicked candidate commits. It selects
the hottest eligible clusters in the current placement, builds the same
graph-expanded local border, and runs the same local swaps and relocations
without moving the cluster to a new cold window first.

This fallback is still exact-gated. A move commits only if hard legality,
hierarchy quality, proxy budget, and minimum exact-proxy improvement all pass.

### Hard-Core Padding

After graph expansion, the local region gets an isotropic pad based on the
kicked hard cluster's core size:

```text
hard_core_span = max(hard_core_width, hard_core_height)
pad = HIER_COLDSPOT_LOCAL_HARD_PAD_FRAC * hard_core_span
```

The pad is clamped by:

```text
HIER_COLDSPOT_LOCAL_MIN_PAD_CELLS
HIER_COLDSPOT_LOCAL_MAX_PAD_FRAC
```

This avoids using a soft-stretched bbox to define excessive base margin while
still leaving room for swaps and relocations.

## Current Use In Placement

The graph is used in coldspot-local refinement:

1. A cluster is kicked into a cold window, or selected in place by the no-kick
   graph-local fallback.
2. Hard macros are legalized when a kick was generated.
3. A local seed bbox is built from the active hard cluster and its owned/bridge
   softs.
4. The graph expands that pre-margin border into adjacent open cold cells.
5. Hard-core padding is applied.
6. A graph-expanded region mask and graph-derived relocation target pool are
   built.
7. Local hard-hard, hard-soft, and soft-soft swaps are tried.
8. Local hard and soft relocations are tried, optionally using graph target
   pools and graph mask gating.
9. Graph candidate ranking selects which generated outcomes enter the exact
   gate when the GNN selector is off.
10. If no kicked candidate commits, the graph-local fallback repeats steps 3-8
   on the current placement for the hottest eligible clusters.
11. Exact proxy and hierarchy-quality gates decide whether the candidate is
   accepted.

## What The Graph Is Not Yet Used For

The graph does not yet rank multiple coldspot anchors. It does not guide
legalization. It is currently used for local-region shaping, local relocation
target generation, relocation mask gating, generated-outcome ranking, and
no-kick local fallback refinement.

The next likely improvement is graph-derived relocation target generation:

```text
target_pool = reached open-cold cells + small pad ring
```

That is now implemented for coldspot-local relocation. The next likely
improvement is anchor enumeration by connected open-cold components.

## Related Paper: Graph Placement Methodology

Mirhoseini et al., "A graph placement methodology for fast chip design"
(`Nature`, 2021, https://doi.org/10.1038/s41586-021-03544-w) is relevant, but
it should not be read as describing the graph currently implemented here.

The paper's graph is a circuit/netlist representation used by an Edge-GNN. Its
state includes netlist adjacency, node features such as type and dimensions,
edge features such as connection count, current macro identity, technology
metadata, and placement coordinates for placed nodes. The learned embedding is
then combined with a grid policy head and a feasibility mask over possible
macro locations.

Our current graph is different:

- it is a spatial graph over benchmark grid cells, not a netlist graph;
- nodes are cold/open cells, not macros or standard-cell clusters;
- edges are implicit 4-neighbor physical adjacency, not electrical
  connectivity;
- it runs only inside coldspot-local refinement and is exact-gated after
  candidate generation.

The useful lesson is architectural: future learned ranking should use both
graphs, not collapse one into the other. The spatial graph answers "where is
there connected cold room?" The netlist/hierarchy graph answers "which macros
and soft clusters are structurally related enough to move together?" A useful
next representation would expose both views to a ranker:

```text
candidate features =
    spatial_graph_features
  + hierarchy/netlist_graph_features
  + current placement geometry
  + exact-gated operator metadata
```

Concrete paper-aligned upgrades that fit this codebase:

- add netlist/hierarchy graph features to coldspot candidate traces:
  parent cluster id, child cluster ids, bridge-soft count, inter-cluster edge
  weights, fanout summaries, and source/target k-hop affinity;
- add placement-state node features similar in spirit to the paper:
  normalized width, height, macro/soft type, fixed flag, current x/y, local
  congestion, and density;
- add edge features beyond a raw count:
  normalized net weight, fanout bucket, pin-offset summaries when available,
  and whether the edge crosses a split parent boundary;
- use spatial masks the way the paper uses feasibility masks: generated
  learned candidates should be masked by hard legality, occupancy, hierarchy
  region, and graph-expanded cold-cell constraints before exact scoring;
- keep the current rule that no learned model can bypass hard legality,
  fixed-macro immobility, bounds, hierarchy quality, or exact-proxy accept
  gates.

Do not import the paper's full sequential RL policy into the production path
without a separate validated project. The current system is a deterministic
hierarchy placer with optional default-off rankers; the paper is most useful as
guidance for richer graph features and feasibility masking.

## Complexity

For a grid with R rows and C columns:

- Cold extraction: O(R * C)
- Occupancy rasterization: O(number of touched macro cells)
- Flood search: O(number of reached cells), bounded by max graph distance

The current IBM grids are small enough that this cost is negligible compared to
exact proxy scoring.
