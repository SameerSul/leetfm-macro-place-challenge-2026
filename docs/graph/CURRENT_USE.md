# Current Graph Use

This document describes the graph currently used by the hierarchy placer. It is
not a learned graph or netlist graph. It is a grid graph over placement cells,
used inside coldspot tightening to reason about open cold area near a kicked
cluster.

## Scope

The graph is active only in the hierarchy coldspot path in
`src/placer/pipeline/macro_placer.py`.

This is intentionally different from the netlist graph used by Mirhoseini et
al., "A graph placement methodology for fast chip design" (`Nature`, 2021,
https://doi.org/10.1038/s41586-021-03544-w). That paper's Edge-GNN graph
represents circuit connectivity and placement-state features for learned policy
and value networks. Our current graph represents physical cold-cell adjacency.
The paper is useful for future learned ranking features, but it does not change
the current graph's role as a spatial mask and local target generator.

The current use is local and candidate-scoped:

- Build a cold-cell map from the current congestion field.
- Build an occupied-cell mask from the candidate hard and soft macro footprints.
- Treat unoccupied cold cells as graph nodes.
- Flood through adjacent open cold cells near the active cluster's local seed.
- Convert the reached graph component into both an expanded bbox and a shape
  mask.
- Rank generated coldspot outcomes with graph features when the GNN selector is
  off.
- Use the graph-derived target pool for coldspot-local hard and soft
  relocation.
- Gate coldspot-local relocation targets by the graph-expanded mask.
- Run candidate-local swaps and relocations inside that box.
- If no coldspot kick commits, run the same graph-local swaps and relocations
  against the current placement for the hottest eligible clusters.
- Keep the normal exact-proxy and hierarchy-quality gates as the final accept
  rule.

## Data Model

The graph is represented by boolean arrays rather than an explicit adjacency
list.

```text
cold_memory[r, c] = cell is cold in the current congestion field
occupied[r, c]    = candidate macro footprint occupies the cell
open_cold[r, c]   = cold_memory[r, c] and not occupied[r, c]
```

Edges are implicit 4-neighbor grid adjacency:

```text
(r, c) -> (r - 1, c)
(r, c) -> (r + 1, c)
(r, c) -> (r, c - 1)
(r, c) -> (r, c + 1)
```

Diagonal adjacency is not used.

## Cold-Cell Refresh

The cold-cell map is a current-state snapshot, not historical accumulation.

At the start of each coldspot round:

```python
cold_memory = _remember_cold_cells(field)
```

After a coldspot kick or graph-local fallback move is accepted and finalized,
the incremental scorer is rebuilt from the committed placement and the
cold-cell map is refreshed from the new congestion field.

This is intentional. If a cluster consumes a coldspot, that area should not
remain available to later adaptive bounds.

## Candidate Occupancy

For each coldspot candidate, `_occupied_cells()` rasterizes hard and soft macro
footprints onto the benchmark grid. A cell is considered occupied if any part
of a macro footprint covers it.

The graph expansion uses:

```python
open_cold = cold_memory & ~occupied
```

This prevents adaptive expansion from treating cells under the candidate's
newly placed macros as usable cold space.

## Adaptive Border Expansion

The local seed bbox starts from:

- the kicked hard cluster, or the current hard cluster when the no-kick
  fallback is running, and
- owned/bridge soft macros associated with that cluster.

Before hard-core padding is applied, `_expand_bbox_to_adjacent_cold()` searches
for open cold cells adjacent to the seed bbox. It then flood-fills through
connected open cold cells up to `HIER_COLDSPOT_ADAPTIVE_MAX_CELLS` grid steps.

The reached cells expand the pre-margin local border. After that, the hard-core
pad is applied to provide room for swaps and relocations.

## Current Constants

```text
HIER_COLDSPOT_LOCAL_HARD_PAD_FRAC=0.50
HIER_COLDSPOT_LOCAL_MIN_PAD_CELLS=1
HIER_COLDSPOT_LOCAL_MAX_PAD_FRAC=0.12
HIER_COLDSPOT_LOCAL_SOFT_ESCAPE_MIN=0.0025
HIER_COLDSPOT_GRAPH_FALLBACK_TOP_K=3
HIER_COLDSPOT_MEMORY_COLD_PCT=35.0
HIER_COLDSPOT_ADAPTIVE_MAX_CELLS=5
```

Local refinement, graph target pools, graph mask gating, graph-local fallback,
and adaptive cold-cell expansion are now unconditional parts of the coldspot
path. Default non-GNN candidate commitment uses exact-proxy-ranked refined
outcomes; graph features remain in traces and as tie-breakers.

## Current Limitations

- Graph mask gating currently applies to coldspot-local hard and soft
  relocation targets. Swaps still use bbox-style region predicates.
- Graph candidate selection ranks generated outcomes for the selected
  cluster/window. It does not yet enumerate and rank multiple coldspot anchors.
- The no-kick graph-local fallback ran on the latest `ibm10` smoke but found no
  exact-gated improvement.
- The graph is refreshed only within coldspot tightening. Other hierarchy
  operators do not consume it yet.
