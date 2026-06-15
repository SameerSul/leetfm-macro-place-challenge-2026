# CongFlow v2: LSMC Global Exploration And GPU Probe Notes

## Overview

This document records the final LSMC exploration phase and the GPU acceleration probes around
it. The active engine is a serial, exact-gated Large-Step Markov Chain: kick, legalize,
pre-screen, descend with a fresh `IncrementalScorer`, and accept only on the bit-exact
post-descent proxy. CUDA availability controls the default `V2_GPU_EXPLORE=auto` enablement,
and the existing CUDA relocation scorer remains available for hard-relocation proposal
ranking, but the active LSMC descent is not a batched multi-chain CUDA implementation.

**Hardware target for dormant GPU work: a single GPU.** The deployment hardware is fixed at
max 1 GPU, so any future batched-chain implementation must use a batch dimension on one
device, not a multi-GPU island model.

**Implementation status (2026-06-15):** Stages 0–2b are shipped (see §5). The single-chain
engine (`src/placer/local_search/lsmc_explore.py`), kick pre-screen, final-phase
multi-incumbent scheduling, and cluster-coherent kicks are active in normal
`src/main.py` runs when CUDA is available. Congestion-gradient phases have been deleted
from the active pipeline, so the LSMC seed pool is deliberately limited to generic local
placements: legalized baseline, random-noise restarts, random-order legalize trials,
pre-R2 best, and post-R2 best. LSMC must not depend on DREAMPlace/bridge outputs or
cong-grad state for future improvements.

## Revision notes (what changed from the previous draft and why)

The previous draft of this document proposed hand-written CUDA kernels scoring
`HPWL + fast grid density` in per-block shared memory, with the legacy pipeline phases
deleted up front. The design review against the two reference papers and the existing
codebase changed four things:

1. **Any GPU ranking cost must include congestion.** Congestion dominates proxy cost by
   ~30×, and HPWL+density-only candidates are often rejected by the exact gate. The
   `cuda_delta` scorer (see `CUDA-path.md`) computes HPWL, density, blockage, touched-net
   routing, and smoothed/top-k congestion deltas on GPU with ~1e-7 parity against the exact
   scorer. It is available to hard-relocation propose-all ranking; a future batched LSMC
   rewrite should reuse this full proxy shape rather than introduce a blinder cost model.
2. **Acceptance happens on post-descent cost, matching the LSMC paper.** GPU-DPO accepts
   a kicked state only after running descent on it and comparing the descended cost
   against the best-so-far. The old draft accepted on raw post-kick cost and only
   descended the final top-K, which can discard states whose basin contains the best
   optimum.
3. **Batched Torch remains the future GPU direction, not the active path.** Per-chain
   state is tiny (`[N, 2]` positions); the heavy static tensors (net incidence, grid) would
   be shared across all chains, matching the memory layout the chunked `cuda_delta` scorer
   already builds. IBM net arrays do not fit in the ~100 KB/block shared-memory limit, so a
   raw shared-memory CUDA design is not the right first step.
4. **Legacy phases are removed by subtraction-with-evidence, not up front.** The 3-step
   spine remains the target end state, but each deletion is a separate experiment gated
   on a paired multi-seed `--all` win (see Staged rollout).

Two citation fixes: the LSMC paper is ICCAD 2025, and its method is one sequential chain
with GPU-parallel descent — the many-parallel-chains structure here comes from cuGenOpt.

## 1. What this builds on

The CUDA hard-relocation path (`CUDA-path.md`, `src/placer/local_search/relocation.py`)
already provides an optional proposal-ranking path:

- `_score_relocation_proposals_cuda_delta_batch(...)`: batched Torch evaluation of the
  full proxy shape (`wl + 0.5·density + 0.5·congestion`) for pools of relocation
  proposals, with static tensors built once and reused across chunks.
- Memory-budgeted chunking (`V2_RELOC_PROPOSE_MAX_MB` / `_AUTO_MEM_FRAC`), sized for the
  available GPU memory.
- Parity verification (~1e-7 max delta on ibm01/ibm04) and CUDA-execution diagnostics
  (`test/diagnostic/_cuda_relocation_status.py`).

The active LSMC engine still exact-scores kicks and descends with `IncrementalScorer`; the
CUDA path is the verified building block for future batched descent.

## 2. The exploration engine

### 2.1 Chain structure (LSMC, faithful)

Each chain holds a private copy of the macro positions and runs:

1. **Kick move (large step).** Relocate a random subset of movable hard macros, or in
   normal `src/main.py` runs select a derived connectivity cluster and apply a
   gather/translate group kick. The kicked hard placement is spiral-legalized. Random
   kicks use `kick_ratio=0.02`; cluster kicks fall back to random kicks when no usable
   cluster exists.
2. **Greedy descent.** Build a fresh `IncrementalScorer` on the kicked state, then run hard
   relocation by congestion and density followed by soft relocation by congestion and
   density. Moves are applied one winner at a time from the current state, avoiding stale
   delta conflicts.
3. **Accept/reject on post-descent cost.** Zero-temperature to start (keep only strict
   improvement over the chain's best, as in GPU-DPO Algorithm 3), with per-chain failure
   counters and early exit after `F` consecutive failures. A low-temperature SA variant
   is a later experiment, not the default.

The shipped loop is serial. `V2_GPU_EXPLORE_CHAINS>1` splits the time slice across several
serial chains and keeps the best result; it is a probe, not the active default.

### 2.2 Soft macros

Descent includes hard relocation and soft relocation in the CPU `IncrementalScorer`
loop. Cluster kicks can also co-move connected soft macros before descent
(`V2_GPU_EXPLORE_CLUSTER_SOFT`, default on) and clip them to the canvas. The final exact
gate still scores hard and soft positions together.

### 2.3 Multi-incumbent scheduling (current)

The final LSMC phase now runs over a small exact-scored seed pool instead of only the
post-R2 incumbent. The pool intentionally excludes bridge/DREAMPlace-specific placements
and any congestion-gradient-derived placements. Current sources are:

- `baseline`: legalized `initial.plc`.
- `random noise`: ordinary Gaussian restart winners and near-best candidates.
- `random-order-legalize`: alternate legalizer tie-break outcomes.
- `pre-r2-best`: the best generic seed before the R2 finisher.
- `post-r2-best`: the fully refined incumbent and mandatory fallback seed.

`V2_GPU_EXPLORE_MAX_SEEDS` caps the number of seeds explored, sorted by exact proxy.
`V2_GPU_EXPLORE_SEED_MARGIN` keeps only near-best non-forced candidates. Each selected
seed receives an equal share of the final LSMC time slice, and the global incumbent is
updated only by exact strict improvement. This keeps the final gate unchanged while
giving LSMC basin diversity after cong-grad deletion.

### 2.4 Multi-chain on one GPU (future, dormant)

The single-chain engine descends one kick at a time; the pre-screen scores several kicks but
still descends one selected kick serially. A future Stage 2c could batch the descent itself:
K chains advance together as a leading batch dimension, so one `cuda_delta` call scores all
chains' relocation pools at once over the shared static tensors. The accept/reject would be
per-chain, and the handoff would take the best chain's state.

That would be the cuGenOpt "P independent solutions" structure realized on a single device — no
islands, no inter-device migration, no worker-per-GPU. Diversity comes from independent
per-chain kick RNG and from the generic incumbent pool above. The hard constraint is the
6 GB budget: chain count is chosen so `K × proposal-pool dynamic bytes + shared static`
fits, reusing the existing memory accounting (`_relocation_*_bytes` in `relocation.py`).
The serial `V2_GPU_EXPLORE_CHAINS` probe found a narrow payoff and is left dormant; a
batched rewrite needs a fresh paired gate before it becomes active work.

Degradation: no GPU → `V2_GPU_EXPLORE=auto` skips LSMC exploration; `V2_GPU_EXPLORE=1`
forces it.

### 2.5 Next LSMC improvement methods

These are the current improvement directions for LSMC. They preserve the final exact
accept gate and avoid bridge/cong-grad coupling.

1. **Seed-pool calibration.** Tune `MAX_SEEDS`, `SEED_MARGIN`, and time allocation across
   baseline/random/P9/pre-R2/post-R2 seeds. Equal time is simple; a better scheduler may
   give the post-R2 incumbent a fixed floor and distribute the remainder by seed diversity
   or score gap.
2. **Generic kick families.** Keep kicks random-field independent, but vary their shape:
   area-weighted macro picks, displacement-window kicks, edge-biased legal regions,
   group kicks for nearby macro clusters, and lower kick ratios for already-tight seeds.
   These use only placement geometry, not routed congestion fields.
3. **Soft-aware large steps.** After a hard kick, reposition a bounded subset of soft
   macros using existing soft relocation or cheap geometric anchors before descent. This
   addresses stale soft placement after larger hard moves without calling external tools.
4. **Adaptive pre-screen.** Let `PRESCREEN` depend on measured exact-score time and seed
   type. Cheap benchmarks can score more kicks; slow grids should preserve time for
   descent and final exact restoration.
5. **R2-as-descent experiment.** For one or two seeds only, treat a shortened R2 finisher
   as the descent part of the chain, then exact-accept after the final score. This tests
   whether LSMC can find basins that normal post-R2 kicks cannot reach, without moving
   the accept gate earlier.
6. **Acceptance schedule, final-gated.** Low-temperature or late-acceptance hill climbing
   can be tested inside a chain, but returned placements still must beat the global
   incumbent on exact proxy. This keeps downside bounded to wasted time.

## 3. CPU handoff

The exact proxy is the accept score. For any future GPU-ranked candidate path, the handoff per
benchmark remains:

1. Take the best chain's candidate(s).
2. **Legalize** each with the existing spiral legalizer — kicks create hard-macro
   overlaps, and R2 is a refinement loop, not a legalizer. Legalization time is charged
   to the exploration budget, not R2's.
3. Exact-score through a **fresh `IncrementalScorer` initialization** or the PLC exact proxy;
   never patch the pre-candidate scorer state with an approximate bulk delta.
4. Keep the winner as the final placement only if it strictly beats the incumbent on true
   proxy. The accept-on-true-proxy guarantee is unchanged.

**Adaptive K.** Exact scoring costs ~160 s on ibm15 and ~220 s on ibm18, so a flat
top-K=5 handoff is unaffordable there. K is sized per benchmark from the running-max
`t_one_score` and remaining budget: K=3–5 on cheap benchmarks, K=1 on the large grids,
K=0 (skip exploration entirely) when the floor-reservation allocator says scoring one
extra candidate would breach `PER_BENCH_FLOOR_S`.

## 4. Configuration

Shipped (default-on under CUDA):

```env
V2_GPU_EXPLORE=auto         # auto: run when CUDA visible (default); 1: force; 0: off
V2_GPU_EXPLORE_KICK=0.02    # kick ratio (fraction of movable hard macros per kick)
V2_GPU_EXPLORE_FAILS=5      # per-chain early-exit failure tolerance (F)
V2_GPU_EXPLORE_TIME_S=30.0  # wall ceiling per benchmark for the exploration loop
V2_GPU_EXPLORE_PRESCREEN=8  # kicks scored per iteration; descend only the best (2b)
V2_GPU_EXPLORE_MULTI_INCUMBENT=1     # final phase explores generic local seeds
V2_GPU_EXPLORE_MAX_SEEDS=3           # top distinct incumbents by exact score
V2_GPU_EXPLORE_SEED_MARGIN=0.08      # keep non-best seed candidates within this gap
V2_GPU_EXPLORE_CLUSTER_P=1.0         # set by src/main.py unless caller overrides it
V2_GPU_EXPLORE_CLUSTER_MODE=both     # gather or translate selected per kick
V2_GPU_EXPLORE_CLUSTER_MAXSZ=32      # skip oversized inferred clusters
V2_GPU_EXPLORE_CLUSTER_SOFT=1        # co-move connected soft macros on cluster kicks
```

Dormant probe:

```env
V2_GPU_EXPLORE_CHAINS=1     # serial chain split; >1 is exploratory, not default
```

Budget: the exploration phase runs after R2 within the per-benchmark budget; the
floor-reservation allocator shrinks its time slice as `t_one_score` rises so it never
breaches `PER_BENCH_FLOOR_S`. Measured cost at the shipped 30 s slice is ~0–60 s/benchmark
(`--all` on-arm ~51 min, under the 1 h cap).

## 5. Staged rollout

Each stage gates the next. Quality comparisons are paired multi-seed `--all` runs —
never cross-day single runs.

- **Stage 0 — re-baseline (DONE 2026-06-11).** avg 1.1243, CUDA diagnostic parity
  1.5e-07, DREAMPlace sm_89 build healthy, numba present.
- **Stage 1 — validate the existing CUDA propose-all path (DONE 2026-06-12).** Paired
  gate WASH (mean +0.0020, 2/3 seeds worse) → `V2_RELOC_PROPOSE_ALL` stays opt-in.
- **Stage 2a — single-chain exploration engine (SHIPPED 2026-06-12).** Post-R2 LSMC
  kick/descent/accept. Paired gate 2/2, mean −0.0042, best 1.1194.
- **Stage 2b — kick pre-screen (SHIPPED 2026-06-13).** Score a batch of kicks, descend
  the best. Paired gate 2/2, mean −0.0020, best 1.1176.
- **Stage 2c — multi-chain on one GPU (PROBED, DORMANT).** Serial chain-split diversity
  produced too little signal to justify a batched-descent rewrite at the current score.
- **Stage 2d — multi-incumbent final scheduling (SHIPPED).** Explore a bounded generic
  seed pool after R2: baseline/random/P9/pre-R2/post-R2 only. Gate: paired multi-seed
  `--all` win over single-incumbent 2b.
- **Stage 2e — cluster-coherent kicks (ACTIVE).** `src/main.py` enables derived-cluster
  gather/translate kicks by default, with random fallback and exact post-descent accept.
- **Stage 3 — LSMC-only improvements.** Test the methods in §2.5 one at a time. Do not
  add bridge-specific seeds or cong-grad-derived kicks/seeds.

The target spine for any future batched GPU work is:

   generic local seeds
        → R2 finisher (bit-exact CPU)
        → LSMC exploration (kick / descent / post-descent accept)
        → final exact gate

## 6. Verification

1. **Parity:** `_verify_relocation_cuda_delta_scores.py` must stay green for the optional
   CUDA proposal scorer; add an equivalent check before any batched LSMC descent is enabled.
2. **Legality:** every handoff candidate passes the overlap/bounds/fixed-macro checks
   after spiral legalization, before exact scoring.
3. **Fresh-scorer drift check:** exact score of a handoff candidate computed from a
   fresh `IncrementalScorer` must match a from-scratch scoring of the same positions.
4. **No-regression:** `test/verification/_stress_verify.py` with `V2_GPU_EXPLORE=1`; the
   exact gate must catch any ranking-vs-true-proxy discrepancy.
5. **Memory:** static tensors and chain state must be freed or reused between the 17
   sequential benchmark calls in `--all`.

## 7. References

- **LSMC (GPU-DPO):** Kahng, A. B., Liang, J., & Wang, Z. (2025). *LSMC Meets GPU
  Acceleration: Scalable and High-Quality Multi-Row Detailed Placement.* ICCAD 2025.
  [PDF via UCSD VLSI CAD Lab](https://vlsicad.ucsd.edu/Publications/Conferences/425/c425.pdf)
  - Source of the chain structure: kick move (random legal swaps, ratio 0.10) →
    GPU-parallel greedy descent → accept on post-descent cost, early exit after F=5
    failures ("zero-temperature annealing in the neighborhood induced by the large
    step"). Also the evidence that kick size is the critical tuning knob, and that
    parallel-evaluated moves need sequential conflict resolution before commit.
  - Caveat: their domain is standard-cell detailed placement with legal sites; the
    descent operators do not transfer, only the chain structure does.

- **cuGenOpt:** Liu, Y. (2026). *cuGenOpt: A GPU-Accelerated General-Purpose
  Metaheuristic Framework for Combinatorial Optimization.*
  [arXiv:2603.19163](https://arxiv.org/abs/2603.19163)
  - Source of the many-independent-chains architecture (one block/chain evolves one
    solution; evaluate → reduce → apply-one within a chain) and the memory-hierarchy
    lesson: performance regimes are set by where the working set lives (shared / L2 /
    DRAM), with diminishing returns past n ≈ 300 in the DRAM regime — the reason this
    design batches through Torch over shared static tensors instead of replicating
    layout state into per-block shared memory.
