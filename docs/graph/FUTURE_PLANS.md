# Future Graph Plans

The current graph only expands coldspot-local region bounds. The next useful
steps should make the graph directly steer candidate choice and target choice.

## 1. Graph-Scored Coldspot Candidate Selection

Status: partially implemented.

The current implementation ranks generated coldspot outcomes by graph features
when the GNN selector is off. It generates several kicked outcomes, scores their
available graph area, and exact-gates the top graph-ranked slice.

The kick generator still chooses the selected cluster/window primarily from
field values. That can select a low-congestion target that has little usable
empty area after the cluster lands.

Future selection should score coldspot anchors by:

```text
anchor_score =
    coldness
  + connected_open_area
  + adjacency_to_hot_cluster
  - occupied_boundary_penalty
```

This would prefer coldspots that are both cold and usable, not just cold in the
field.

Expected benefit:

- Fewer candidates rejected by exact proxy after the kick.
- Better use of coldspots that have room for local refinement.

## 2. Graph-Derived Relocation Target Pools

Status: implemented for coldspot-local hard and soft relocation.

Current behavior:

- The graph expands the local bbox and mask.
- Relocation target pools can be supplied from the graph-expanded region.
- Exact scoring remains the accept gate.

Remaining work:

- Add diagnostics showing whether graph target pools are sparse or productive.
- Compare graph-only target pools against graph-plus-generic fallback pools.

## 3. Masked Regions Instead Of Rectangular Regions

Status: implemented for coldspot-local relocation targets, not swaps.

The graph component now produces an allowed-cell mask. Hard and soft relocation
can reject targets outside that mask. Swaps still use bbox-style region checks.

Future swap region logic should support allowed-cell masks:

```text
allowed = reached_open_cold_component + hard_core_pad_ring
```

Relocation targets would be filtered by `allowed`. Swap candidates could use
the mask as a stronger region predicate than a bbox.

Expected benefit:

- Less accidental cross-cluster mixing.
- Better preservation of hierarchy while still using open cold cells.

## 4. Vacancy-Aware Legalization

After a coldspot kick, hard legalization can push macros away from the useful
part of the coldspot. The graph can guide legalization by preferring legal slots
inside or near the open-cold component.

Possible design:

- During overlap repair, rank candidate slots by distance to graph component.
- Prefer slots that keep the hard-core inside the connected cold region.
- Fall back to existing legalization when no graph-preferred legal slot exists.

## 5. Donor / Evacuation Moves

If a nearby cold component is blocked by unrelated softs or small movable hards,
the graph can identify blockers occupying that component.

Potential operators:

- Move blocking soft macros out of the component if exact proxy improves.
- Swap a cluster hard with a blocker only when the hard remains inside the
  local graph-expanded area.
- Use a larger gain threshold for blockers that are not structurally related to
  the kicked cluster.

## 6. Coldspot Persistence Metrics

A kick can consume a coldspot without reducing adjacent hot cells enough to be
worth it. Track whether the cold component remains useful after a tentative
candidate.

Possible metrics:

- open cold cells before vs after candidate
- hot-cell reduction near source cluster
- hot-cell reduction near target component
- ratio of consumed cold area to exact-proxy improvement

This can be used as a candidate-ranking feature before exact scoring.

## 7. Bridge-Soft Corridor Repair

Bridge soft macros connect multiple clusters. A graph search can find open cold
corridors between the relevant cluster regions.

Potential behavior:

- Identify bridge softs whose connected clusters are separated by hot cells.
- Search for open cold paths between those cluster boxes.
- Prefer soft relocation targets along those paths.

## 8. Spatial + Netlist Dual Graph Features

Status: research note added from Mirhoseini et al., "A graph placement
methodology for fast chip design" (`Nature`, 2021,
https://doi.org/10.1038/s41586-021-03544-w).

The paper is useful for future graph work, but it describes a different graph
from the one currently implemented here. Their Edge-GNN represents circuit
connectivity with node features, edge features, placement coordinates, metadata,
and a grid feasibility mask. Our graph is a spatial cold-cell graph used to
shape local coldspot refinement.

The next useful design is therefore a dual view:

```text
spatial graph: connected cold room, occupancy, masks, local target cells
netlist graph: macro/soft connectivity, hierarchy confidence, bridge roles
```

Candidate rankers should consume both views. The spatial graph should continue
to answer whether a target region has connected usable room. The netlist graph
should answer whether the source cluster, moved softs, bridge softs, and
neighbor clusters are structurally compatible with the move.

Potential features:

- source cluster size, area, heat, split-parent id, and child-cluster id;
- inter-cluster edge weights and k-hop structural affinity to nearby clusters;
- bridge-soft count and bridge-soft fanout between source and target regions;
- normalized macro width, height, fixed flag, current x/y, local congestion,
  and density;
- edge features such as connection count, net weight, fanout bucket, and
  pin-offset summaries when available;
- spatial mask features from the current graph: connected open-cold area,
  graph target cells, adaptive cold cells, and distance from source.

Acceptance constraints stay unchanged: learned graph features may rank or add
candidates only. They must not bypass hard legality, fixed-macro immobility,
bounds, hierarchy quality, or exact-proxy gates.

## Validation Strategy

Each graph feature should be tested in this order:

1. `uv run python -m py_compile $(find src -type f -name "*.py")`
2. `uv run evaluate src/main.py -b ibm10`
3. One additional non-ibm10 benchmark with different shape/soft profile.
4. Full `--all` only after a clear local win or a correctness-sensitive change.

Any accepted graph change should report:

- accepted coldspot moves
- exact proxy before and after coldspot
- runtime
- hard legality
- whether graph expansion increased local candidate counts
