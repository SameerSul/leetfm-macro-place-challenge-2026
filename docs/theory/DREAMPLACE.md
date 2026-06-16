# DREAMPlace

## Current Role

DREAMPlace is required by the current hierarchy-only production path. It is no
longer an async candidate generator feeding a proxy search. `_hierarchy_floorplan()`
calls grouped DREAMPlace once, reads back hard and soft macro positions, then
legalizes and refines that placement while preserving derived macro clusters.

If DREAMPlace is unavailable, `MacroPlacer.place()` raises because the old proxy
fallback has been removed.

## What The Algorithm Is

DREAMPlace is an analytical global placer. It optimizes continuous macro and
cell coordinates with a smooth wirelength model and a density penalty. That
objective is not the same as the TILOS proxy:

```text
proxy_cost = wirelength + 0.5 * density + 0.5 * congestion
```

The hierarchy path uses DREAMPlace primarily for structure, not for final proxy
optimality. Cluster grouping adds synthetic clique nets so connected subsystems
are pulled together during global placement.

## Bridge Flow

The bridge converts the ICCAD04 benchmark into Bookshelf files:

- `.nodes` for movable/fixed macros and soft macros
- `.nets` for net connectivity and pin offsets
- `.pl` for initial locations
- `.scl` for a row structure DREAMPlace can consume
- `.aux` as the Bookshelf entry file

The hierarchy path calls:

```python
run_dreamplace(
    iccad_dir,
    plc=plc,
    soft_macros_movable=True,
    cluster_groups=groups,
    group_weight=V2_HIER_GROUP_WEIGHT,
)
```

The bridge writes synthetic per-cluster clique nets into Bookshelf before
launching DREAMPlace. `read_dreamplace_positions_full()` then reads hard and
soft macro centers back into placement order.

## After DREAMPlace

Grouped DREAMPlace output is not returned directly. The hierarchy pipeline:

1. Legalizes hard macros in cluster-consecutive order.
2. Runs a default-order safety legalization pass.
3. Runs soft relocation cleanup.
4. Runs region-locked hard/soft relief.
5. Optionally runs bounded coldspot tightening.
6. Clamps movable macros in bounds.

The exact proxy is still used for evaluator reports and local gates, but
DREAMPlace's main job is to provide a hierarchy-aware global layout.
