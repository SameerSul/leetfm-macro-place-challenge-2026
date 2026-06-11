# CongFlow v2: GPU-Accelerated Global Exploration & Pipeline Refactor

## Overview
This module introduces a GPU-accelerated global exploration phase (using principles from LSMC and cuGenOpt) to bridge the gap between initial seed generation and strict greedy refinement. 

Unlike the previously retired `S2` attempt (which batched candidate evaluation *per macro* and bottlenecked on CPU strip-generation), this architecture uses **cross-macro parallel sampling**. By serializing the entire layout state and evaluating thousands of multi-macro perturbations simultaneously on the GPU, we bypass the CPU `_trial_at` bottleneck entirely. This allows the placer to escape deep local minima before handing the best candidate states back to the CPU's bit-exact `IncrementalScorer`.

Furthermore, fully adopting this parallel sampling architecture allows us to gut several legacy CPU heuristics (kNN restrictions, TOP-K limits, and iterative finite-difference gradients), collapsing the 9-phase pipeline into a streamlined 3-step engine.

---

## 1. Core Mechanics

### 1.1 Data Serialization (AoS to SoA)
To achieve memory bandwidth saturation on the GPU, the highly object-oriented CPU state (`benchmark._cached_plc`, `incremental_scorer`) is flattened into a **Structure of Arrays (SoA)**. 
* Macro metadata (X, Y, Area, Fixed Status) and Net Bounding Boxes are converted to contiguous 1D arrays.
* This transfer occurs **once** per benchmark execution to completely eliminate PCIe latency during the trial loops.

### 1.2 Independent Parallel Evaluation (Overcoming the S2 Bottleneck)
In `S2`, the CPU generated routing strips for a single macro's targets and handed them to the GPU, causing a sync-lock. 
In the new architecture, we distribute identical copies of the layout state to the shared memory of multiple Streaming Multiprocessors (SMs). Each CUDA block independently:
1. Executes a randomized structural perturbation (a "Large Step").
2. Calculates an *approximate* hardware-friendly proxy cost (HPWL + fast grid density).
3. Evaluates if the new state should be kept based on an annealing temperature or aggressive accept-degrading logic.

Because the Markov chains are independent across blocks, the GPU evaluates completely disjoint regions of the search space without locking.

### 1.3 The CPU Handoff Guarantee
The GPU evaluation phase is **not** bit-exact to `proxy_cost`. It does not execute the heavy `_apply_3pin_routing_vec` logic. Instead, it serves purely as a highly correlated topological generator. 
The top $K$ layouts returned by the GPU are fed directly into the existing `_exact_proxy` gate in the CPU's R2 loop. **The strict accept-on-true-proxy guarantee remains intact.** If the GPU generates a layout that looks good but actually increases congestion, the CPU's exact gate simply rejects it.

---

## 2. Configuration & Budget

Because the GPU can sample thousands of candidates in milliseconds, this phase consumes a minimal, fixed slice of `effective_budget_s`.

**Environment Variables:**
* `USE_GPU_EXPLORATION=1`: Enables GPU acceleration (Requires CUDA toolkit and compatible GPU).
* `GPU_EXPLORATION_K=5`: The number of top candidate layouts returned to the CPU for R2 refinement (Default: 5).
* `GPU_TIME_SLICE_S=10.0`: The strict time ceiling allocated to the CUDA kernel before forcing a reduction and return.

**Budget Impact:**
By offloading global exploration to the GPU, we free up CPU cycles in the `R2` phase. We can safely increase `R2_HOT` or `n_targets` for hard relocation without breaching the `PER_BENCH_FLOOR_S` guarantees calculated by the floor-reservation allocator.

---

## 3. Verification

To ensure this integration does not violate CongFlow's strict non-regressing guarantees, verify the following after enabling:

1. **Memory Leak Check:** Ensure the SoA buffers are properly freed or reused between the 17 sequential benchmark calls in the `--all` harness.
2. **Drift Verification:** The coordinates returned by the GPU must pass through a fresh initialization of `IncrementalScorer` before entering R2. Do not attempt to incrementally patch the pre-GPU CPU scorer state with the massive GPU delta.
3. **No-Regression Run:** Run `test/verification/_stress_verify.py` with `USE_GPU_EXPLORATION=1` to guarantee that the CPU exact gate successfully catches any discrepancy between the GPU's approximate cost and the true proxy.

---

## 4. Pipeline Simplification: Replacing Legacy Heuristics

Transitioning to a cuGenOpt/LSMC GPU architecture allows us to delete large swaths of heuristic "glue" code that only exist to bandage the limitations of a single-threaded CPU. 

### 4.1 Deleting the "Cong-Grad" Spine (Phases 1-3, 5b/c, 7, 8)
* **Legacy:** Four distinct phases dedicated to finite-difference congestion gradient descent, requiring heuristics like `frac=0.08` wide-steps and `TOP-K` restrictions to escape local minima traps.
* **GPU Replacement (Single Stochastic Flow Kernel):** We replace all of these with a single GPU block. Thousands of threads probabilistically sample multi-directional macro shifts simultaneously. The gradient descent emerges naturally globally without explicit iterative math or heuristic gates. 

### 4.2 Deleting Multi-Seed 2-Opt & kNN Restrictions
* **Legacy:** A multi-seed 2-opt that artificially restricts swaps to a spatial kNN ($k=20$) and uses the `S9` cold-region teleport hack because scoring all $O(N^2)$ pairs sequentially is too slow.
* **GPU Replacement (Unrestricted Swap Matrix):** We drop the kNN restriction entirely. The GPU evaluates the approximate proxy delta for every possible pair combination on the chip in parallel, naturally finding optimal long-range structural swaps that the old kNN was physically blind to.

### 4.3 Replacing R3 & R5 (Sequential Soft Relocation)
* **Legacy:** A strict sequential loop where soft macros move one by one, updating the global density/congestion field after every single commit.
* **GPU Replacement (Parallel Markov Chain Co-Adaptation):** Moving soft macros is purely a density spreading problem. We map different soft-macro clusters to different SMs, allowing the GPU to explore how thousands of different move *orderings* affect the final congestion map, breaking the CPU sync-lock.

### 4.4 The New Radically Simplified Spine
By ripping out these heuristics, the sprawling 9-phase pipeline collapses into a highly streamlined 3-step engine:

    ┌────────────────────────────────────────────────────────────────┐
    │ Step 1: Phase 0 & 5                                            │
    │ Initialize Baseline + Asynchronous DREAMPlace Seeds            │
    └────────────────────────────────────────────────────────────────┘
             │
             ▼
    ╔════════════════════════════════════════════════════════════════════╗
    ║ Step 2: The GPU Mega-Phase (Cross-Macro Parallel Sampling)         ║
    ║ Replaces Legacy Phases 1-3, 5b/c, 7, 8, 9, and Multi-seed 2-Opt.   ║
    ║                                                                    ║
    ║ 1. Serialize layout state to VRAM.                                 ║
    ║ 2. Massively parallelize structural swaps, relocations, and        ║
    ║    stochastic congestion spreading without kNN/TOP-K restrictions. ║
    ║ 3. Return the top 5 global minima coordinates via parallel reduce. ║
    ╚════════════════════════════════════════════════════════════════════╝
             │
             ▼ (Top 5 new starting points)
    ┌────────────────────────────────────────────────────────────────┐
    │ Step 3: The R2 CPU Finisher (Reduced Scope)                    │
    │ Feed GPU coordinates into the bit-exact CPU IncrementalScorer. │
    │ Acts purely as a rapid, greedy CPU "snap" to perfectly legalize│
    │ and finalize the true proxy score.                             │
    └────────────────────────────────────────────────────────────────┘

---

## 5. References & Literature

The architectural paradigm of this GPU refactor is built upon the following recent advancements in GPU-accelerated EDA and combinatorial optimization:

* **cuGenOpt:** Liu, Y. (2026). *cuGenOpt: A GPU-Accelerated General-Purpose Metaheuristic Framework for Combinatorial Optimization*. [arXiv:2603.19163](https://arxiv.org/abs/2603.19163)
    * *Application in CongFlow:* Informs the "one block evolves one solution" massive parallel sampling architecture, overcoming traditional sequential bottlenecks by evaluating hundreds of macro swaps simultaneously in shared memory.

* **LSMC (GPU-DPO):** Kahng, A. B., Liang, J., & Wang, Z. (2026). *LSMC Meets GPU Acceleration: Scalable and High-Quality Multi-Row Detailed Placement*. [PDF via UCSD VLSI CAD Lab](https://vlsicad.ucsd.edu/Publications/Conferences/425/c425.pdf)
    * *Application in CongFlow:* Validates the "Large-Step Markov Chain" approach for VLSI placement, proving that aggressive topological perturbation followed by immediate greedy refinement effectively escapes deep local minima while adhering to physical constraints./goa