# Graph Issue Tracker

This is the issue tracker for graph-exclusive work. General hierarchy issues
belong in `docs/general/ISSUES.md`.

## Open

### G-001: Diagnose Graph-Derived Target Productivity

Graph-derived target pools and the no-kick graph-local fallback are now wired
into coldspot-local hard and soft relocation, but the latest `ibm10` smoke
still had `0` accepted coldspot moves.

Impact:

- We do not yet know whether the graph target pool is too sparse, too cold but
  wirelength-hostile, whether fallback-selected clusters are poor sources, or
  simply blocked by exact-proxy gates.

Proposed fix:

- Add a diagnostic script under `test/diagnostic/`.
- Report per candidate:
  - graph target cells
  - legal hard targets
  - soft targets skipped by WL prefilter
  - exact-scored targets
  - accepted targets
  - fallback candidates tried

### G-002: Swaps Still Use Bbox Region Predicates

The graph mask currently gates coldspot-local hard and soft relocation targets.
Hard-hard, hard-soft, and soft-soft swaps still use bbox-style region checks.

Impact:

- Swaps can consider positions inside the graph bbox but outside the actual
  connected graph shape.

Proposed fix:

- Add mask-aware swap region checks after relocation-target diagnostics prove
  the mask is useful.

### G-003: No Anchor Enumeration By Connected Open Area

Graph candidate selection ranks generated outcomes for the selected
cluster/window. It does not yet enumerate multiple coldspot anchors by connected
open area.

Impact:

- The selected anchor may still be cold but spatially unhelpful.

Proposed fix:

- Extend the coldspot generator to accept anchor candidates.
- Generate anchors from connected open-cold components.
- Rank anchors by coldness, connected open area, and source-cluster adjacency.

### G-004: Occupancy Is Conservative And Cell-Based

The occupancy mask marks every cell touched by a macro footprint.

Impact:

- Small overlaps with a cell can mark the entire cell occupied.
- This may undercount usable cold area on coarse grids.

Proposed fix:

- Keep the current conservative mask for safety.
- Add diagnostics comparing touched-cell occupancy to center-cell occupancy.
- Only relax occupancy if target generation becomes too sparse.

### G-005: No Diagnostics For Adaptive Expansion Effectiveness

Trace logs include `adaptive_cold_cells`, but there is no dedicated report that
summarizes graph usefulness across benchmarks.

Impact:

- It is hard to tell whether graph expansion is not firing or firing but not
  producing accepted moves.

Proposed fix:

- Add a diagnostic script under `test/diagnostic/`.
- Report per benchmark:
  - candidates refined
  - candidates with adaptive expansion
  - mean reached cells
  - local region area before and after expansion
  - accepted coldspot moves

## Closed

### G-000: Stale Cold Memory

Status: closed.

The first graph draft accumulated cold cells with an OR-style memory. That could
leave stale coldspots after a finalized kick consumed or changed a cold region.

Current behavior:

- The cold-cell map is replaced from the current congestion field at the start
  of each coldspot round.
- After any accepted coldspot kick or graph-local fallback move, the scorer is
  rebuilt and the cold-cell map is refreshed from the new field.

### G-009: Graph Refinement Required A Committed Coldspot Kick

Status: closed.

Coldspot-local swaps, hard relocation, soft relocation, graph borders, target
pools, and mask gating now also run through a no-kick fallback when no coldspot
kick commits. The fallback considers the hottest eligible clusters in the
current placement and still uses exact proxy, hard legality, hierarchy quality,
and proxy-budget gates before committing.

### G-006: Relocation Did Not Use Graph-Derived Targets

Status: closed.

Coldspot-local hard and soft relocation now accept a graph-derived flat cell
target pool. When `HIER_COLDSPOT_GRAPH_TARGET_POOL=True`, local relocation uses
targets from the graph-expanded candidate region.

### G-007: Relocation Targets Were Only Bbox-Gated

Status: closed for relocation, open for swaps under G-002.

Coldspot-local hard and soft relocation can now reject targets outside the
graph-expanded mask when `HIER_COLDSPOT_GRAPH_MASK_GATING=True`.

### G-008: No Graph-Aware Candidate Selection

Status: closed for generated outcomes, open for anchor enumeration under G-003.

Generated coldspot outcomes are now graph-ranked when
`HIER_COLDSPOT_GRAPH_SELECT=True` and the GNN selector is off. The graph selector
uses field relief, adaptive cold cells, graph region cells, and graph target
cells.
