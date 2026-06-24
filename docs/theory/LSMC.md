# LSMC Notes - Archived

This document is historical. The generic LSMC exploration engine was part of the
removed proxy-optimized path and is no longer active production code.

Deleted pieces include:

- generic `_lsmc_explore`,
- random hard-macro kick/descent/accept loop,
- generic `_cluster_kick`,
- `GPU_EXPLORE_*` defaults,
- `GPU_EXPLORE_CLUSTER_*` defaults,
- the `_verify_cluster_kick.py` verifier.

The only retained code in `src/placer/local_search/lsmc_explore.py` is
`_coldspot_cluster_kick()`, used by the hierarchy path as a bounded
cluster-tightening helper after region relief.

## Retained Concept

The retained coldspot kick is not a Markov-chain optimizer. It performs one
proposal:

1. Select a cluster.
2. Find a low-congestion window.
3. Gather the cluster near that window.
4. Co-move assigned soft macros.
5. Legalize hard macros.

`_hierarchy_floorplan()` accepts the proposal only if intra-cluster spread
drops and proxy stays within the configured budget:

```text
HIER_COLDSPOT_BUDGET=0.05
HIER_COLDSPOT_TOTAL=0.15
HIER_COLDSPOT_ROUNDS=8
HIER_COLDSPOT_BUDGET_S=30
```

## Historical Finding

Generic cluster-coherent LSMC kicks were tested against random kicks and later
removed. They were not a good fit for the selected hierarchy system because the
proxy objective rewards spreading macros, while hierarchy preservation compacts
connected subsystems. The selected system now uses grouped DREAMPlace,
cluster-consecutive legalization, region-locked relief, and bounded coldspot
tightening instead.
