# LSMC

## What The Algorithm Is

LSMC means Large-Step Markov Chain. The idea is to alternate between a large perturbation
that jumps to a different basin and a local descent that tries to settle that perturbed state
into a nearby local optimum. The final decision is made after descent, not immediately after
the kick.

In this system LSMC is used as the final exploration layer after R2 and post-R2 soft
relocation. It is a strict-improvement, zero-temperature variant: a descended candidate is
accepted only if its exact proxy score beats the current global incumbent.

## How It Works

Each LSMC iteration has four stages:

1. **Choose a seed.** The seed comes from the generic exact-scored pool: baseline,
   random-noise restarts, random-order legalization, pre-R2 best, or post-R2 best.
   DREAMPlace-specific and cong-grad-derived seed pools are intentionally excluded.
2. **Kick.** The algorithm applies a large step to the hard macro placement. Normal
   `src/main.py` runs enable cluster-coherent kicks by default:
   `V2_GPU_EXPLORE_CLUSTER_P=1.0` and `V2_GPU_EXPLORE_CLUSTER_MODE=both`.
   A cluster kick either gathers a derived connectivity cluster near one anchor or
   translates it rigidly. If no usable cluster exists, the code falls back to a random
   per-macro kick using `V2_GPU_EXPLORE_KICK`.
3. **Legalize and pre-screen.** Kicked hard placements are spiral-legalized. A small batch
   of kicked candidates can be exact-scored before descent, and only the best kick is
   descended.
4. **Descend and accept.** A fresh `IncrementalScorer` is built on the kicked state. The
   descent alternates hard relocation by congestion and density with soft relocation by
   congestion and density. The final candidate is exact-scored, and the global incumbent is
   updated only on strict improvement.

The "gradient" language in some notes refers to discrete hot-to-cold field descent, not
analytic gradient descent. The fields are congestion and density maps over the placement
grid. Moves are generated for hot macros, scored through the incremental proxy machinery,
and committed only when they reduce the proxy.

## Cluster Derivation

The cluster system infers macro communities from the flat netlist because the ICCAD04
benchmarks do not provide hierarchy. It does not force clusters to stay together during the
main flow. It only labels candidate groups that LSMC may use for a large-step kick.

The derivation runs in `placer/local_search/clusters.py`:

1. Build a weighted graph over movable hard macros.
2. Scan low-fanout nets only. Nets with fewer than two pins or more than
   `V2_CLUSTER_MAX_FANOUT` pins are skipped because high-fanout nets tend to be clocks,
   buses, or other weak grouping signals.
3. For each retained net, collect the hard-macro pins on that net and add clique edges
   among those hard macros.
4. Edge weight is the number of low-fanout nets shared by a hard-macro pair.
5. Union hard macros whose edge weight is at least `V2_CLUSTER_MIN_EDGE`.
6. Keep connected components with at least two hard macros as clusters.

There are two index spaces involved. The wirelength cache uses `modules_w_pins` indices:
ports first, then hard macros, then soft macros. LSMC uses placement-order indices:
hard macros are `[0, n)`, and soft macros are `[n, n + n_soft)`. The cluster builder maps
from `modules_w_pins` hard indices back into placement-order hard indices before storing the
cluster labels. The result is cached on the PLC object with the fanout and edge thresholds.

Soft macro membership is derived separately by `derive_cluster_softs()`. For every low-fanout
net, soft pins are attributed to the cluster ids of the hard pins on that net. Each soft macro
is assigned to the single hard cluster it co-occurs with most often. This gives LSMC an
optional soft co-move set for each hard cluster.

## Cluster-Coherent Kicks

Cluster-coherent kicks replace a random per-macro large step with a group move. They are
enabled in normal `src/main.py` runs by:

```text
V2_GPU_EXPLORE_CLUSTER_P=1.0
V2_GPU_EXPLORE_CLUSTER_MODE=both
```

Direct imports of `placer.pipeline.macro_placer` do not apply those wrapper defaults; in
that case `V2_GPU_EXPLORE_CLUSTER_P` defaults to `0.0` unless the caller sets it.

For each kick attempt, LSMC shuffles the derived cluster ids and picks the first usable
cluster. A usable cluster must have at least two movable hard macros and must not exceed
`V2_GPU_EXPLORE_CLUSTER_MAXSZ`. If no usable cluster is found, or the random draw misses
according to `V2_GPU_EXPLORE_CLUSTER_P`, LSMC falls back to the ordinary random hard-macro
kick.

There are two kick modes:

- **`translate`:** compute a feasible rigid `(dx, dy)` range that keeps every hard macro in
  the cluster inside the canvas, sample one translation, and move the whole cluster by that
  offset.
- **`gather`:** sample one legal anchor region, place the cluster members near that anchor
  with jitter, then let the spiral legalizer pack the hard macros into legal nearby sites.

`V2_GPU_EXPLORE_CLUSTER_MODE=both` randomly chooses `translate` or `gather` per cluster kick.

When `V2_GPU_EXPLORE_CLUSTER_SOFT` is enabled, connected soft macros assigned to the selected
cluster receive the same conceptual move. In translate mode they get the same `(dx, dy)`. In
gather mode they are placed near the same anchor with jitter. Soft macros are clipped to the
canvas but not overlap-legalized, because soft macro overlap is legal in this abstraction.
They are still included in the exact proxy score.

The important invariant is unchanged: a cluster kick is only a proposal mechanism. The kicked
hard placement is legalized, the hard and soft positions are descended with a fresh
`IncrementalScorer`, and the final placement is accepted only if the exact post-descent proxy
beats the incumbent. Cluster kicks can spend time on bad basins, but they cannot directly
commit a worse final placement.

## How We Use It

LSMC is deliberately placed after the main R2 finisher. R2 handles deterministic local
improvement; LSMC tests whether a large basin jump plus descent can find a better final
placement inside the remaining budget.

The active controls are:

```text
V2_GPU_EXPLORE=auto
V2_GPU_EXPLORE_TIME_S=30.0
V2_GPU_EXPLORE_PRESCREEN=8
V2_GPU_EXPLORE_MULTI_INCUMBENT=1
V2_GPU_EXPLORE_MAX_SEEDS=3
V2_GPU_EXPLORE_SEED_MARGIN=0.08
V2_GPU_EXPLORE_CLUSTER_P=1.0      # set by src/main.py unless overridden
V2_GPU_EXPLORE_CLUSTER_MODE=both
V2_GPU_EXPLORE_CLUSTER_SOFT=1
```

The invariant is simple: LSMC may spend time exploring worse kicked states, but it cannot
return a worse placement because every accepted result must pass the exact post-descent
proxy gate.
