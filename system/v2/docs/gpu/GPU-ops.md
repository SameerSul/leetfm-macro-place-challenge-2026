# CongFlow v2: GPU-Accelerated Global Exploration (Revised Design)

## Overview

This document specifies a GPU-accelerated global exploration phase that runs as the final
quality phase after R2. The engine runs many independent Markov chains, each following the
LSMC structure (kick move, greedy descent, accept on post-descent cost), with the cost
evaluated by the existing `cuda_delta` scorer / bit-exact `IncrementalScorer` gate.

**Hardware target: a single GPU, always** (the deployment hardware is fixed at max 1 GPU —
the 6 GB RTX 4050 this was developed on is representative, not a placeholder). There is no
multi-GPU island model; "many chains" means a batch dimension on one GPU, not sharding
across devices.

**Implementation status (2026-06-13):** Stages 0–2b are shipped (see §5). The single-chain
engine (`src/placer/local_search/lsmc_explore.py`) and the kick pre-screen are default-on
under CUDA. Stage 2c (multi-chain batched on one GPU) is the next build. `V2_GPU_EXPLORE_*`
variables marked "proposed" below are not yet wired.

## Revision notes (what changed from the previous draft and why)

The previous draft of this document proposed hand-written CUDA kernels scoring
`HPWL + fast grid density` in per-block shared memory, with the legacy pipeline phases
deleted up front. The design review against the two reference papers and the existing
codebase changed four things:

1. **The GPU cost now includes congestion.** Congestion dominates proxy cost by ~30×, and
   the phases the old draft deleted were exactly the congestion-aware ones. An
   HPWL+density-only explorer risks generating candidates the exact gate systematically
   rejects — the same anti-correlation failure already documented for the `density_score`
   fallback. The `cuda_delta` scorer (see `CUDA-path.md`) already computes HPWL, density,
   blockage, touched-net routing, and smoothed/top-k congestion deltas on GPU with ~1e-7
   parity against the exact scorer. The exploration engine reuses it instead of
   introducing a second, blinder cost model.
2. **Acceptance happens on post-descent cost, matching the LSMC paper.** GPU-DPO accepts
   a kicked state only after running descent on it and comparing the descended cost
   against the best-so-far. The old draft accepted on raw post-kick cost and only
   descended the final top-K, which can discard states whose basin contains the best
   optimum.
3. **Batched Torch first, raw CUDA kernels only if profiling demands it.** Per-chain
   state is tiny (`[N, 2]` positions); the heavy static tensors (net incidence, grid) are
   shared across all chains, which is the memory layout the chunked `cuda_delta` scorer
   already builds. This sidesteps the shared-memory budget problem entirely: IBM net
   arrays do not fit in the ~100 KB/block shared-memory limit, so the old draft's
   "layout state in SM shared memory" plan would have landed in cuGenOpt's L2/DRAM
   regime anyway, where their own data shows diminishing returns for n > 300.
4. **Legacy phases are removed by subtraction-with-evidence, not up front.** The 3-step
   spine remains the target end state, but each deletion is a separate experiment gated
   on a paired multi-seed `--all` win (see Staged rollout).

Two citation fixes: the LSMC paper is ICCAD 2025, and its method is one sequential chain
with GPU-parallel descent — the many-parallel-chains structure here comes from cuGenOpt.

## 1. What this builds on

The CUDA hard-relocation path (`CUDA-path.md`, `src/placer/local_search/relocation.py`)
already provides:

- `_score_relocation_proposals_cuda_delta_batch(...)`: batched Torch evaluation of the
  full proxy shape (`wl + 0.5·density + 0.5·congestion`) for pools of relocation
  proposals, with static tensors built once and reused across chunks.
- Memory-budgeted chunking (`V2_RELOC_PROPOSE_MAX_MB` / `_AUTO_MEM_FRAC`), sized for the
  6 GB GPU — the permanent VRAM budget, so chain count × proposal-pool size must fit here.
- Parity verification (~1e-7 max delta on ibm01/ibm04) and CUDA-execution diagnostics
  (`test/diagnostic/_cuda_relocation_status.py`).

The exploration engine is a control loop around this machinery, not a new scorer.

## 2. The exploration engine

### 2.1 Chain structure (LSMC, faithful)

Each chain holds a private copy of the macro positions and runs:

1. **Kick move (large step).** Relocate a random subset of movable hard macros, then
   spiral-legalize. The GPU-DPO starting point was `kick_ratio ≈ 0.10`; tuning on IBM
   found smaller is better (shipped **0.02**; 0.02 > 0.05 > 0.10) because the post-R2
   incumbent is already well-refined and large kicks cannot be recovered in one descent.
2. **Greedy descent.** A few rounds of propose-all relocation restricted to the chain's
   own state: generate candidate targets for hot macros, score the pool with
   `cuda_delta`, apply the best non-conflicting improvement per round. Within a chain,
   moves are applied one winner at a time (evaluate → reduce → apply), the cuGenOpt
   pattern; this avoids the stale-delta conflict problem of applying a precomputed swap
   matrix.
3. **Accept/reject on post-descent cost.** Zero-temperature to start (keep only strict
   improvement over the chain's best, as in GPU-DPO Algorithm 3), with per-chain failure
   counters and early exit after `F` consecutive failures. A low-temperature SA variant
   is a later experiment, not the default.

Chains are independent: a batch dimension over chains on the one GPU. Static netlist
tensors are shared across chains; only positions, costs, and counters are per-chain. Chain
count is bounded by the 6 GB budget (chains × proposal-pool dynamic bytes + shared static),
not by device count.

### 2.2 Soft macros

Descent moves hard macros only at first (the scope `cuda_delta` already covers). Soft
relocation stays in R2/post-R2 on CPU. Extending `cuda_delta` to soft-macro moves is a
separate follow-up — it matters because soft macros left behind after large hard-macro
displacement is a documented failure mode.

### 2.3 Multi-chain on one GPU (Stage 2c)

The single-chain engine (2a) descends one kick at a time; the pre-screen (2b) batches the
*kick selection* but still descends serially. Stage 2c batches the *descent* itself: K
chains advance together as a leading batch dimension, so one `cuda_delta` call scores all
chains' relocation pools at once over the shared static tensors. The accept/reject is
per-chain (each keeps its own best + failure counter); the handoff (§3) takes the best
chain's state.

This is the cuGenOpt "P independent solutions" structure realized on a single device — no
islands, no inter-device migration, no worker-per-GPU. Diversity comes from independent
per-chain kick RNG and, optionally, seeding different chains from the different available
incumbents (baseline legalization, each DREAMPlace basin). The hard constraint is the 6 GB
budget: chain count is chosen so `K × proposal-pool dynamic bytes + shared static` fits,
reusing the existing memory accounting (`_relocation_*_bytes` in `relocation.py`).

Degradation: no GPU → exploration skipped, pipeline unchanged.

## 3. CPU handoff

The GPU cost is a ranking score, not the true proxy. The handoff per benchmark:

1. Take the best chain's candidate(s).
2. **Legalize** each with the existing spiral legalizer — kicks create hard-macro
   overlaps, and R2 is a refinement loop, not a legalizer. Legalization time is charged
   to the exploration budget, not R2's.
3. Exact-score through a **fresh `IncrementalScorer` initialization** (never patch the
   pre-GPU scorer state with the bulk GPU delta).
4. Feed the winner into R2 only if it strictly beats the incumbent on true proxy. The
   accept-on-true-proxy guarantee is unchanged.

**Adaptive K.** Exact scoring costs ~160 s on ibm15 and ~220 s on ibm18, so a flat
top-K=5 handoff is unaffordable there. K is sized per benchmark from the running-max
`t_one_score` and remaining budget: K=3–5 on cheap benchmarks, K=1 on the large grids,
K=0 (skip exploration entirely) when the floor-reservation allocator says scoring one
extra candidate would breach `PER_BENCH_FLOOR_S`.

## 4. Configuration

Shipped (default-on under CUDA):
```
V2_GPU_EXPLORE=auto         # auto: run when CUDA visible (default); 1: force; 0: off
V2_GPU_EXPLORE_KICK=0.02    # kick ratio (fraction of movable hard macros per kick)
V2_GPU_EXPLORE_FAILS=5      # per-chain early-exit failure tolerance (F)
V2_GPU_EXPLORE_TIME_S=30.0  # wall ceiling per benchmark for the exploration loop
V2_GPU_EXPLORE_PRESCREEN=8  # kicks scored per iteration; descend only the best (2b)
```
Proposed for Stage 2c:
```
V2_GPU_EXPLORE_CHAINS=auto  # parallel chains; auto = max that fits the 6 GB budget
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
- **Stage 2c — multi-chain on one GPU (NEXT).** Batch the descent across K chains (§2.3).
  Gate: paired multi-seed `--all` win over 2b.
- **Stage 4 — prune legacy phases, one at a time.** Start with multi-seed 2-opt, then
  the cong-grad phases (1–3, 5b/c, 7, 8). Each deletion is its own paired comparison;
  a deletion that loses gets reverted, not bandaged.

The end state, if every gate passes, is the simplified spine:

    seeds (baseline + async DREAMPlace)
        → R2 finisher (bit-exact CPU)
        → GPU multi-chain exploration (kick / batched descent / post-descent accept)
        → final exact gate

## 6. Verification

1. **Parity:** `_verify_relocation_cuda_delta_scores.py` must stay green; add an
   equivalent check for the descent loop's scoring path.
2. **Legality:** every handoff candidate passes the overlap/bounds/fixed-macro checks
   after spiral legalization, before exact scoring.
3. **Fresh-scorer drift check:** exact score of a handoff candidate computed from a
   fresh `IncrementalScorer` must match a from-scratch scoring of the same positions.
4. **No-regression:** `test/verification/_stress_verify.py` with `V2_GPU_EXPLORE=1`;
   the exact gate must catch any GPU-vs-true-proxy discrepancy.
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
