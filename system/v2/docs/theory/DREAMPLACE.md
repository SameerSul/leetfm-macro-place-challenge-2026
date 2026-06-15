# DREAMPlace

## What The Algorithm Is

DREAMPlace is a GPU-accelerated analytical global placer. It formulates placement as a
continuous optimization problem over cell and macro coordinates. The objective combines a
smooth wirelength model with a density penalty, then optimizes that objective with a
Nesterov-style gradient method. In this repo we use DREAMPlace as an external subprocess,
not as the owner of the final placement flow.

The important distinction for this system is that DREAMPlace optimizes a different objective
from the challenge proxy. DREAMPlace is strong at wirelength and density spreading, while our
exact proxy is:

```text
wirelength + 0.5 * density + 0.5 * congestion
```

Congestion dominates the normalized proxy on the IBM benchmarks, so a DREAMPlace placement
is not automatically accepted just because DREAMPlace converged.

## How It Works

The bridge converts the TILOS/ICCAD04 benchmark into Bookshelf files:

- `.nodes` for movable/fixed macros and soft macros
- `.nets` for net connectivity and pin offsets
- `.pl` for initial locations
- `.scl` for a row structure DREAMPlace can consume
- `.aux` as the Bookshelf entry file

The subprocess then runs DREAMPlace global placement with:

- weighted-average wirelength
- Nesterov optimizer
- `macro_place_flag=1`
- `use_bb=1`
- configurable target density
- deterministic settings
- CPU thread caps to avoid oversubscribing the main placer

The active async variants are:

- `lo-fix`: target density `0.65`, soft macros fixed
- `hi-mov`: target density `0.85`, soft macros movable
- `hi-fix`: target density `0.85`, soft macros fixed

An optional grouped variant is gated by `V2_DP_GROUP`. It derives hard-macro clusters and
adds synthetic clique nets during Bookshelf conversion. Those nets bias DREAMPlace to keep
connected macro groups closer together. The grouped output is still just another candidate;
it does not alter the final accept rule.

After DREAMPlace exits, the bridge reads back hard and soft macro centers. Hard macros are
clipped to the canvas, legalized with the same spiral legalizer used by the main pipeline,
and then exact-scored with the PLC proxy. Soft positions from DREAMPlace are copied when
available and protected by the final in-bounds clamp.

## How We Use It

In `MacroPlacer.place()`, DREAMPlace is launched at the start of a benchmark call so it can
run concurrently with baseline legalization and early scoring. The main pipeline later polls
the async handles, legalizes any completed hard placements, exact-scores the full hard+soft
candidate, and updates `best_pl` only if the exact proxy strictly improves.

DREAMPlace is therefore a candidate generator, not a replacement for R2 or LSMC. Its role is
to provide independent basins that the hand-tuned initial placement, random restarts, and
random-order legalization may not reach. It is intentionally not used as a special LSMC seed
source. If a DREAMPlace candidate wins, it becomes the ordinary incumbent by exact score; if
it loses, it is discarded.

This keeps the system robust against objective mismatch. DREAMPlace can contribute when its
wirelength/density basin also has acceptable congestion, but the exact proxy gate prevents a
congestion-worse placement from being returned.
