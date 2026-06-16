# CongFlow v2 -- Architecture & Experiment Log

> **Reference `--all`: 1.1252** (2026-06-11, all 17 VALID / 0 overlaps,
> **2337s ~39min**) — **S10 ML hard-relocation ranker enabled as the production
> default.** `src/main.py` now enables the shipped XGBoost ranker when no `ML_*`
> env var is preset and the model artifact + `xgboost` are available, widening
> hard relocation to a 32-candidate pool and exact-scoring the ranked top 16.
> The strict true-proxy accept gate is unchanged; missing deps or any preset
> `ML_*` var falls back to the prior pure-heuristic path. The prior reference was
> **1.1272** (2026-06-10, **2645s ~44min**) from **S16: fixed a silent DREAMPlace
> ABI break.** The bridge launched DP
> with the repo `.venv` (upgraded to Python 3.14 for numba at S13), but DP's compiled
> extensions are cpython-310, so `import place_io_cpp` died ~4s after launch and the
> bridge masked it as a benign "not ready" → **DP produced ZERO seeds on every
> benchmark since S13.** So the prior 1.1379 @2117s (S14) was a DP-OFF run. Pointing
> `VENV_PYTHON` at the DP build env (`dpenv` 3.10) restored all 3 basins: `--all`
> **1.1379 → 1.1272 (−0.0107)**, 51/51 DP launches ready / 0 failures, DP basins
> contributing on all 17. (+528s is the DP candidate-scoring + DP-basin 2-opt work;
> still well under the 3300s soft cap.) See `ISSUES.md` S16.
> Trajectory: 1.1782 → 1.1500 → 1.1423 (S11) → 1.1403 (S12) → 1.1380 (S13) →
> 1.1379 (S14, DP-off) → 1.1272 (S16, DP restored) → **1.1252 (ML filter default)**.
> The per-benchmark
> decomposition table below is the 2026-05-31 1.1782 snapshot (kept as the detailed
> breakdown); see `ISSUES.md` and `PROGRESS.md` for the current headline.
>
> **NB — LAHC disproven (2026-06-10).** Late-Acceptance Hill Climbing on the
> 2-opt-on-winner was tested (env-gated) and **reverted**: strictly worse on
> ibm12/17/18, tightening the history length only recovers greedy and never beats
> it. The deadline-bound 2-opt converges fast to a strong basin minimum, so
> non-monotonic acceptance just wastes budget wandering (consistent with the S1
> basin-hopping disproof). The accept gate stays strict-improvement greedy.
>
> **Active design note (2026-06-14):** the congestion-gradient spine has been
> pruned from the current code. The active pipeline is baseline + async DP
> candidate scoring + random restarts + Phase 9 random-order legalization + R2 +
> post-R2 soft relocation + generic multi-incumbent LSMC. See `PROGRESS.md` for
> post-pruning measurements.

## Reference Result (2026-06-11)

**Average proxy cost: 1.1252** across all 17 IBM ICCAD04 benchmarks, with the
ML hard-relocation filter enabled by default. The filter's gain is confirmed by
a same-day paired multi-seed `--all` (3 seeds × ON/OFF, sequential): Δ(ON−OFF)
= −0.0051 / −0.0044 / −0.0029, mean **−0.0041**, filter wins 3/3, all 6 runs
17/17 VALID. ON mean 1.1245 vs OFF mean 1.1286 (seed 44's ON run hit 1.1235 —
seed variance, not a selectable config; the harness runs the default seed).
See ISSUES.md S10 and PROGRESS.md 2026-06-11.

| Benchmark | Hard macros | v2 score | vs RePlAce (1.4578) |
|-----------|-------------|----------|---------------------|
| ibm01 | 246 | **0.9146** | -37.3% |
| ibm02 | 271 | **1.1621** | -20.3% |
| ibm03 | 290 | **0.9896** | -32.1% |
| ibm04 | 295 | **1.0137** | -30.5% |
| ibm06 | 178 | **1.2059** | -17.3% |
| ibm07 | 291 | **1.1819** | -18.9% |
| ibm08 | 301 | **1.1543** | -20.8% |
| ibm09 | 253 | **0.8409** | -42.3% |
| ibm10 | 786 | **1.0945** | -24.9% |
| ibm11 | 373 | **0.9354** | -35.8% |
| ibm12 | 651 | **1.3100** | -10.1% |
| ibm13 | 424 | **0.9988** | -31.5% |
| ibm14 | 614 | **1.2133** | -16.8% |
| ibm15 | 393 | **1.2130** | -16.8% |
| ibm16 | 458 | **1.1608** | -20.4% |
| ibm17 | 760 | **1.3502** | -7.4% |
| ibm18 | 285 | **1.3885** | -4.8% |
| **AVG** | | **1.1252** | **-22.8%** |

Comparisons:
- vs RePlAce (1.4578): **-22.8%** (-0.3326 points)
- vs UT Austin GPU/DREAMPlace leader (1.4076): **-20.1%** (-0.2824 points)
- vs sameer_v1 baseline (1.4860): **-24.3%**

Total `--all` time: 2337s (~39 min).

---

## Architecture Overview

### proxy_cost formula

    proxy_cost = 1.0 x wirelength + 0.5 x density + 0.5 x congestion

After normalization: WL ~0.06-0.11, congestion ~1.0-2.1.
Congestion dominates by ~30x. Optimizing WL alone reliably increases proxy cost.

### Phase pipeline (per benchmark call)

1. Load seed -- read `initial.plc`, legalize hard-macro overlaps.
2. DP variants -- async DREAMPlace candidates are launched when the bridge is available; they are scored as ordinary candidates only.
3. Random restarts -- Gaussian perturbations of initial hard positions, legalize, exact-score.
4. Phase 9 -- 3 random-order legalization trials.
5. R2 -- multi-round local refinement:
   - reloc: single-macro repositioning to reduce cong/density/combined fields
   - soft-reloc[cong]: gradient-driven cong reduction
   - soft-reloc[density]: gradient-driven density spreading
   - soft-soft, hard-soft, and hard-soft-soft swaps/cycles
   - 2-opt: pairwise swap accepting proxy improvements
6. Post-R2 soft relocation -- leftover soft congestion/density cleanup.
7. LSMC final exploration -- random hard-macro kick, legalize, descend with hard/soft moves, accept only exact post-descent improvement.

The congestion-gradient spine, DP-rescue chains, TOP-K cong-grad, and multi-seed
2-opt phase are no longer part of the active pipeline. LSMC seed collection is
generic: baseline, random-noise restarts, random-order legalize trials, pre-R2
best, and post-R2 best.

### Budget allocation (L-change + M1, 2026-05-31)

    HARNESS_TOTAL_BUDGET_S = 3300.0   # total PLACE time (excludes scoring overhead)
    HARNESS_TOTAL_BENCHMARKS = 17
    BUDGET_OVERRUN_S = 83.0           # each benchmark can run 83s past soft cap
    PER_BENCH_FLOOR_S = 110.0         # every benchmark guaranteed >= 110s
    HARD_CAP_SAFE_S = 3540.0          # wall-clock safety guard

Key change (L-change): Budget tracks _total_place_time_s (actual place() execution
time) instead of wall-clock monotonic. This eliminates ~821s of harness overhead
(scoring, I/O, startup) that was eating into later benchmarks' budgets.

M1 change: ibm01 hardcoded budget reduced 200s to 150s. Saves 50s place time with
negligible quality impact (0.0002 proxy difference on ibm01).

Worst-case reserved place time in `--all` mode:
- ibm01: 150+83 = 233s place time (first benchmark soft budget + overrun)
- ibm02 to ibm18: floor-bound at 110+83 = 193s each
- Reserved place time: 233 + 16x193 = 3321s

Actual runs can finish earlier when later phases converge or skip. The 2026-06-11
headline run finished in 2337s (~39 min).

---

## Change History

### eda_io plug-and-play layer (2026-06-09) -- no placer change

New `src/eda_io/` package + `src/place_design.py` CLI make the placer usable in
any physical-design flow. Inputs: LEF, DEF, structural Verilog, SDC, Liberty
(mix freely; minimum = one geometry source + one instance source). Outputs:
updated DEF (input DEF patched in place, byte-identical outside placement
clauses), ICC2/Innovus Tcl, QoR `.rpt`, visualization PNG. Strategy: every
input combo is merged into one neutral `Design` (`eda_io/design.py`) and
converted to the ICCAD04 `netlist.pb.txt` + `initial.plc` pair (`eda_io/
build.py`), then loaded through the standard `load_benchmark` with the plc
attached via `benchmark._cached_plc` — so external designs get the exact TILOS
scorer and identical placer behavior, and no second native format exists.
Conversion semantics: LEF CLASS BLOCK → hard macro (fallback: >10× median
area); std cells clustered ~50/cluster (by location if placed, else union-find
connectivity) into square soft macros; DEF FIXED stays fixed; placement
BLOCKAGES become fixed dummy macros; non-zero DEF origins shifted to a
(0,0) canvas and shifted back on output; SDC → net weights (clock 0.0,
critical 2.0, false path 0.25), Liberty scales by sink capacitance. 15 tests
(`test/eda_io/`, incl. DEF round-trip + subprocess e2e) all pass; fixture
chiptop run: proxy 0.8737 → 0.8177, VALID. Full docs: `src/eda_io/README.md`.

### Readability refactor (2026-06-04) -- no algorithm change, `--all` avg 1.1500

Pure code-simplification pass; move generation, scoring, and RNG untouched, so the
1.1500 / 17-VALID / 0-overlap `--all` reflects a favorable full-budget run within
normal timing variance, not an algorithmic gain. (a) ML-trace per-candidate
congestion/density feature lookups consolidated into a `TraceFields` helper in
`ml/data_collection.py` — data collection stays byte-identical, pinned by
`test/verification/test_trace_fields_equivalence.py`; the local-search files shed
~110 lines of trace boilerplate. (b) The 7×-duplicated pairwise separation matrices
deduped into `geometry.separation_matrices`. (c) `place()`'s budget math and
DREAMPlace-launch block extracted into the `_effective_budget()` and
`_launch_dreamplace_seeds()` methods. pyflakes now clean across `src`. NB:
ibm01's single-bench score is timing-sensitive
(0.9084–0.9094 by wall-time, deadline-gated R2), so non-degradation was confirmed at
the `--all` aggregate, not by single-benchmark bit-identity.

### L-change (2026-05-31) -- avg 1.2593 -> 1.1782

Replaces wall-clock cumulative tracking with _total_place_time_s. Prior K-change
run had ~821s of harness/scoring overhead inflating the cumulative timer, starving
ibm15-18 to only 24-46s budget each.

With L-change, ibm15-18 each get the 110s PER_BENCH_FLOOR_S guarantee:

## 4. The pipeline

The placer is invoked once per benchmark via `MacroPlacer.place(benchmark) →
torch.Tensor[num_macros, 2]`. Internally, `place()` runs the following pipeline:

> **Note — opt-in alternate path.** When `V2_HIER_FLOORPLAN=1` or
> `V2_REGION_LOCK=1` is set, `place()` short-circuits at entry into the
> non-proxy regional pipeline (§5.9) and **none of the proxy pipeline below
> runs**. The diagram below is the default leaderboard/proxy path.

The pipeline is **a single sequential spine** with one genuinely concurrent
branch: DREAMPlace is launched as up to 3 subprocesses at `place()` entry.
DREAMPlace outputs are harvested before random restarts and scored as ordinary
candidates; they are not retained as later LSMC seed basins. Everything else runs
in order on the main thread, except Phase 9 legalization trials, which may run in
a small thread pool before sequential exact scoring.

```
   ┌──────────────────────────────────────────────────────────────────────┐
   │  initial.plc   (hand-tuned spread)                                   │
   └──────────────────────────────────────────────────────────────────────┘
            │
            │ ─── async side-channel: launch at place() entry ───────────┐
            │                                                            ▼
            │                          ┌───────────────────────────────────────┐
            │                          │  DREAMPlace ×3  (subprocess,          │
            │                          │  runs concurrently with the spine)    │
            │                          │  lo-fix / hi-mov / hi-fix configs     │
            │                          │  legalize + exact-score as candidates │
            │                          └───────────────────────────────────────┘
            ▼                                                           │
   ┌────────────────────────┐                                           │
   │ Phase 0 — Baseline     │                                           │
   │ _will_legalize         │                                           │
   │ (vectorized rings)     │                                           │
   └────────────────────────┘                                           │
            │                                                           │
            ▼                                                           │
   ┌────────────────────────┐   harvest DP results ◄────────────────────┘
   │ Score completed DP     │   best_pl may update; no DP-specific later seed
   │ candidates             │   dependency in LSMC
   └────────────────────────┘
            │
            ▼
   ┌────────────────────────┐
   │ Noise restarts         │   random Gaussian perturbations of initial.plc,
   │ (inline, sequential)   │   ~24 fracs spanning 1%–25%; accept-on-true-proxy
   └────────────────────────┘
            │
            ▼
   ┌────────────────────────┐
   │ Phase 9 random-order   │
   │ legalize variants      │
   └────────────────────────┘
            │
            ▼
   ╔════════════════════════════════════════════════════════════════════╗
   ║  R2 interleave loop (≤20 budget-gated rounds, accept-on-true-proxy)║
   ║  ┌──────────────────────────────────────────────────────────────┐  ║
   ║  │  Round r:                                                    │  ║
   ║  │  ┌─────────────────────────────────────────────────────┐     │  ║
   ║  │  │  Hard relocation (R1/R2/R2b)                        │     │  ║
   ║  │  │    top_hot=48 by max(H,V), n_targets=16             │     │  ║
   ║  │  │    accept on true-proxy drop                        │     │  ║
   ║  │  └─────────────────────────────────────────────────────┘     │  ║
   ║  │                       ▼                                      │  ║
   ║  │  ┌─────────────────────────────────────────────────────┐     │  ║
   ║  │  │  Soft cong relocation (R3) — IF r ≤ 3 (A: hard cap) │     │  ║
   ║  │  │    top_hot=128, n_targets=24                        │     │  ║
   ║  │  │    field = plc routing max(H,V)                     │     │  ║
   ║  │  └─────────────────────────────────────────────────────┘     │  ║
   ║  │                       ▼                                      │  ║
   ║  │  ┌─────────────────────────────────────────────────────┐     │  ║
   ║  │  │  Soft density relocation (R5)                       │     │  ║
   ║  │  │    top_hot = 128 (r ≤ 3) or 192 (r > 3, C: boost)   │     │  ║
   ║  │  │    field = grid_occupied / dens_grid_area           │     │  ║
   ║  │  └─────────────────────────────────────────────────────┘     │  ║
   ║  │                       ▼                                      │  ║
   ║  │  ┌─────────────────────────────────────────────────────┐     │  ║
   ║  │  │  2-opt cleanup (8s budget slice)                    │     │  ║
   ║  │  │    k=16 spatial kNN + S9 cold-teleport (S11)        │     │  ║
   ║  │  └─────────────────────────────────────────────────────┘     │  ║
   ║  │                       ▼                                      │  ║
   ║  │       round_improved? — yes → next round                     │  ║
   ║  │                       — no  → terminate                      │  ║
   ║  └──────────────────────────────────────────────────────────────┘  ║
   ╚════════════════════════════════════════════════════════════════════╝
                │
                ▼
        ┌──────────────────────────┐
        │ Post-R2 soft relocation  │
        │ congestion then density  │
        └──────────────────────────┘
                │
                ▼
        ┌──────────────────────────┐
        │ LSMC final exploration   │
        │ generic seeds only:      │
        │ baseline / random / P9 / │
        │ pre-R2 / post-R2 best    │
        └──────────────────────────┘
                │
                ▼
        ┌──────────────────────────┐
        │  best placement returned │
        │(centers, [num_macros, 2])│
        └──────────────────────────┘
```

Everything above runs inside a single per-benchmark budget
(`effective_budget_s`) allocated by the **floor-reservation allocator** (§4.2).

### 4.1 Pipeline phase reference

| Phase | What it does | Why |
|---|---|---|
| **0 Baseline** | Legalize from `initial.plc` via `_will_legalize` (vectorized greedy spiral) | Establish a valid baseline; preserve the hand-tuned spread |
| **DREAMPlace ×3 (async)** | Launch DP at `place()` entry in 3 configs (lo-fix, hi-mov, hi-fix); legalize and exact-score completed outputs | Add independent candidate placements without blocking the main spine |
| **Noise restarts** | Inline (sequential) random Gaussian perturbations of `initial.plc`, ~24 fracs spanning 1%–25%, accept-on-true-proxy | Broad basin-hopping around the hand-tuned seed; fills budget between the directed phases |
| **9 random-order legalize** | N=3 trials with randomized secondary-sort key in `_will_legalize` | Different legalization arrangements from the same starting positions |
| **R2 interleave (≤20 budget-gated rounds)** | Hard reloc ⇄ soft-cong reloc ⇄ soft-density reloc ⇄ 2-opt cleanup plus soft/hard-soft swaps and cycles | The dominant lever — see § 2.3 |
| **Post-R2 soft relocation** | Short leftover-budget soft relocation by congestion then density | Clean up soft macro placement after the main R2 loop saturates |
| **LSMC final exploration** | For generic seed pool, random hard kick → legalize → hard/soft descent → exact accept | Basin exploration without bridge/cong-grad coupling; final exact gate prevents accepted regressions |

> **Historical phase numbers.** Some older notes refer to phases 1-8 and
> multi-seed 2-opt. Those were experimental labels, not a required sequence.
> Cong-grad phases, DP-rescue chains, TOP-K cong-grad, and multi-seed 2-opt are
> retired from the active spine. The current numbered survivor is Phase 9
> random-order legalization; R2 and LSMC are the main refinement/exploration
> layers.

### 4.2 Budget allocation (floor-reservation)

The placer is invoked 17 times by `evaluate --all` on the same instance, and
must keep total wall-clock under the harness's 3600s hard cap. Each call
computes `effective_budget_s` from the cumulative monotonic elapsed:

```python
remaining_total = HARNESS_TOTAL_BUDGET_S - cumulative_elapsed       # 3300s internal cap
remaining_benchmarks = max(1, HARNESS_TOTAL_BENCHMARKS - benchmarks_done)
# Reserve (floor + overrun) for every OTHER remaining benchmark + own overrun
reserve_others = (PER_BENCH_FLOOR_S + BUDGET_OVERRUN_S) * (remaining_benchmarks - 1)
this_cap = remaining_total - reserve_others - BUDGET_OVERRUN_S
effective_budget_s = min(time_budget_s, max(PER_BENCH_FLOOR_S, this_cap))
# Hard-cap safety: never push past the 3540s headroom under the 3600s harness cap
effective_budget_s = min(effective_budget_s, HARD_CAP_SAFE_S - cumulative_elapsed - BUDGET_OVERRUN_S)
```

Constants: `PER_BENCH_FLOOR_S=110`, `BUDGET_OVERRUN_S=83`,
`HARNESS_TOTAL_BUDGET_S=3300`, `HARD_CAP_SAFE_S=3540`.

Worst-case simulation (first call uses `time_budget_s=150` per M1; every
benchmark overruns its soft budget by `BUDGET_OVERRUN_S=83s`):

```
b01      cum=     0   eff=150   actual=233   (first call: time_budget_s, M1)
b02      cum=   233   eff=110   actual=193   ← floor binds immediately (reserve_others dominates)
b03–b17  cum stepping by 193    eff=110   actual=193
final cum = 233 + 16×193 = 3321             ← matches the per-benchmark breakdown above
```

Even in the worst case, **every benchmark gets ≥110 s** and the total lands at
~3321 s — just over the 3300 s soft target but comfortably under the 3540 s
hard-cap headroom and the 3600 s harness cap. The
pre-floor-reservation allocator (`adaptive_cap = remaining/remaining_benchmarks·0.9`
plus a blunt `cumulative > 95% × cap → baseline` guard) starved ibm18 in one
real `--all` run; the floor-reservation fix makes that structurally
impossible.

### M1-change (2026-05-31) -- ibm01 budget 200s -> 150s

ibm01 R2 rounds 10-11 improved proxy by only 0.0002 total (0.9403 -> 0.9402).
Reducing to 150s saves 50s of place time and prevents ibm18's HARD_CAP_SAFE_S
guard from clamping its budget.

### K-change (2026-05-30) -- avg ~1.42 -> 1.2593

## 5. Algorithm explanations

### 5.1 Vectorized greedy spiral legalize (`_will_legalize`)

The base legalizer. Sort macros by largest area first; for each macro, search
expanding rings of cells around its starting position for the nearest
unoccupied legal slot. The naive scalar implementation is a Python loop over
ring radii and over candidates within each ring. The vectorized version
generates all `8r` candidates per ring at once via `_ring_offsets`, runs a
single `[K, P]` conflict matrix against placed macros, and picks the argmin
distance. ~12× faster than the scalar reference.

**Critical correctness detail** (one of the harder bugs we fixed): the
original scalar code computes `d² = (cx − pos[idx, 0])²` where `cx` is a
Python float and `pos[idx, 0]` is `float32`. NumPy demotes the Python float to
float32 for the subtraction (the Python-scalar-meets-numpy-scalar rule), so
ties between symmetric ring candidates are broken at float32 precision. The
vectorized version must cast its candidate arrays to `float32` **before** the
subtraction — otherwise the float64 ties pick a different ring direction, and
ibm04 (where the tie-break historically affected the retired cong-grad
trajectory) landed at 1.3364 instead of 1.3316.

### 5.2 LSMC final exploration (`_lsmc_explore`)

The active global exploration layer is LSMC, not congestion-gradient
perturbation. LSMC runs after R2 and post-R2 soft relocation, when the placement
is already locally refined:

1. Select one of the generic seed placements: baseline, random-noise restart,
   random-order legalize, pre-R2 best, or post-R2 best.
2. Apply a large-step random kick to a small fraction of movable hard macros.
   (A "cluster-coherent" kick variant — move a derived connectivity cluster as a
   unit — was built and tested but **removed** as a dead-end: paired multi-seed
   `--all` showed it was within run-to-run noise, since the late, proxy-gated kick
   only changes the result when a clustered config beats the refined incumbent,
   which clustering rarely does. See ISSUES.md S20.)
3. Legalize the kicked hard placement.
4. Build a fresh `IncrementalScorer` on the kicked state.
5. Descend through hard relocation by congestion/density and soft relocation by
   congestion/density.
6. Exact-score the descended placement and accept only strict true-proxy
   improvement.

This is not analytic gradient descent. The descent step is a discrete
hot-to-cold field search: choose macros sitting in high congestion/density cells,
try legal colder cells, and commit only proxy-improving moves. The exact
post-descent gate is the invariant that lets LSMC explore without accepting a
worse final placement.

The old `_routing_congestion_perturb` cong-grad spine was deleted from the
active pipeline. Older phase labels 1/2/3/5b/5c/7/8 in historical notes refer to
that retired design.

### 5.3 Vectorized routing fill (`_apply_3pin_routing_vec`, `_apply_h/v_strips_batch`)

The hot path of the incremental scorer's congestion update (cProfile shows
67% of a per-move trial). For a touched net, the routing demand is "filled"
into the H/V flats along L-shaped or steiner paths between the net's pins.

The vectorized fill:

1. **Classify** all touched nets by length (2-pin, 3-pin, ≥4-pin steiner).
2. **For each class**, compute all the L/steiner segments in one batched numpy
   expression. The 3-pin path is the trickiest (4 sub-cases depending on
   collinearity); we mirror the scalar reference exactly via two sorts
   (one by `(col, row)` for cases 1–3, one by `(row, col)` for the T-route
   case 4).
3. **Apply** the segments to the H/V flats via a **difference-array + cumsum
   trick**: for an H-strip on row `r` from column `lo` to `hi`, store
   `+weight` at `(r, lo)` and `−weight` at `(r, hi)` in a `(grid_row, grid_col+1)`
   events array, then `cumsum(events, axis=1)[:, :grid_col]` recovers the
   filled values. Multiple overlapping strips on the same row accumulate via
   `np.add.at` on the endpoints (unbuffered scatter for correctness on
   duplicates).

**Idea #1 optimization (subset-cumsum strip-batch):** the
`(grid_row, grid_col+1)` diff array is wasteful when only a few rows are
touched. `np.unique` the touched row indices and cumsum only those rows.
The cumsum is per-row independent, so the result is **bit-identical** to the
full version — duplicate accumulation order is preserved because `np.add.at`
processes entries in the original index order, and the unique remap doesn't
change which entries land in which compact row.

**Idea #2 optimization (topology-struct cache):** the gather indices, length
classification, and the ≥4-pin sink-index layout are placement-independent.
Split `_apply_net_routing_subset` into `_build_net_routing_struct` (cacheable
per macro) + `_apply_net_routing_struct` (does only the gcell extraction +
dispatch + fill given a precomputed struct). The scorer caches the struct
per module index; single-macro paths reuse it across every candidate target
(within a relocation pass) and across the −1 / +1 within each move.

**S14 explicit-loop JITs (2026-06-07).** numba JITs the per-net strip fill above
(`_apply_h/v_strips_batch_jit`, `_apply_3pin_routing_vec_jit`), but once numba was
installed (S13), cProfile found three more vectorized-numpy scoring functions with
**no JIT path** dominating per-move cost. Each got an explicit-loop numba version
that matches numpy's accumulation order (bit-exact), dispatched behind
`if HAS_NUMBA:` with the numpy path retained as fallback:
- `_apply_macro_routing` → `_apply_macro_routing_scatter_jit` — the per-cell
  hard-macro routing scatter (was `np.add.at`/`subtract.at` over macro footprints,
  the biggest tottime).
- `_macro_occ` → `_macro_occ_jit` — the per-macro density footprint
  (cell enumeration + overlap area).
- `_compute_per_net_hpwl_subset` → `_hpwl_subset_jit` — per-net HPWL bounding box
  (min/max are order-independent → bit-identical).

Verified by the existing scorer verifiers run WITH numba (stress Hcong/Vcong ~1e-15,
density Δ=0, swap Δ=0). `--all` 2563s → 2117s (~17 % faster; ~39 % vs the no-numba
3486s), score unchanged (pure speed).

### 5.4 Bit-exact incremental congestion cost (`_compute_cong_cost`)

Cost computation from the maintained H/V flats:

1. Normalize H/V routing flats by routes-per-cell.
2. Add the macro-routing blockage (Hm/Vm flats).
3. Smooth H/V via a 1D box filter (`_smooth_routing_cong_vec`, separable per
   axis).
4. Concatenate H_total + V_total, take the **top-5% of all cells**, return
   `mean(top)`.

Steps 3 + 4 cost ~17% of a per-move trial. The smoother is a fixed-kernel
convolution (kernel width `2·smooth_range + 1`, with `smooth_range` read from
`plc.smooth_range` — 2 under the TILOS evaluator), and crucially, it's
**separable**:

- H is smoothed *along rows-within-a-column* (each column independent).
- V is smoothed *along cols-within-a-row* (each row independent).

So changing the raw flats in a small region only affects a localized set of
smoothed values in the same row/column.

**The incremental cong cost:** cache `H_smoothed` and `V_smoothed` as
`(grid_row, grid_col)` 2D state. On a move:

- The routing apply (`_apply_net_routing_struct`) returns the bounding box
  of touched-net pin gcells.
- Re-smooth only the affected **columns** for H (since H smooths per column)
  and affected **rows** for V (per row), using the bbox.
- Crucially, **recompute from the raw flats**, not from accumulated deltas.
  Each value in the cache always equals exactly what a full re-smooth would
  produce → **no drift, no float-accumulation error**.
- `_compute_cong_cost` becomes: V_total = cached V_smoothed + Vm,
  H_total = cached H_smoothed + Hm, concat, top-5% partition. No re-smoothing.

The trial path (`score_swap`/`score_move`/`score_move_soft`) snapshots the
affected columns/rows of the smoothed cache before applying, and restores on
revert; the commit path persists the re-smoothed values. All six move paths
wire through a shared `_resmooth_bbox` helper.

This is the **bit-exact incremental cong cost** lever (1.2799 → 1.2767 in
isolation, fully verified). The "from raw, not deltas" detail is what makes
it bit-exact and drift-free — a delta-accumulating version would lose
~1e-16 / commit and slowly diverge.

### 5.5 R2 proxy-driven hard 2-opt (`_two_opt_proxy_swap`)

Local search via macro pair-swaps, scored by the incremental proxy. Naive
2-opt is O(N²) candidates per iteration; we restrict to a **spatial kNN**
(`k=20` nearest macros) per macro, sorted by distance. Each candidate swap
is scored via `score_swap` (~1.4 ms incremental); accept on strict
true-proxy drop.

**S9 augmentations** (cong-aware candidate selection):

- **Hot-first ordering.** Sort the outer loop by descending local cong
  (`macro_cong[i]`), so the deadline-bound search spends its time on the
  routing peaks that dominate the proxy.
- **Cold-region teleport.** Spatial kNN can only swap nearby macros — a
  routing-heavy macro can never relocate across the chip via local swaps
  alone. For the 20 hottest macros, append the 8 *coldest* as extra
  candidates — a long-range edge that expands the reachable placement set.
  Size-incompatible teleports fail the free conflict check before scoring.

The proxy gate validates every swap, so candidate selection only changes
*which* swaps are tried, never *accepts* a worse placement.

The active invocation is the R2 cleanup pass with `k=16`, chosen to free scoring
time for the more productive soft relocation/swap passes. A WL-delta prefilter
(`wl_delta_swap`) exists but is **off** for hard 2-opt — calibration showed hard
spatial-kNN swaps have near-zero WL spread, so it can't discriminate and would
only add cost. Hard 2-opt still scores every kNN candidate it reaches within its
budget slice.

The older final multi-seed 2-opt from DP basins is retired from the active
pipeline. Final basin exploration is now handled by LSMC's generic seed pool and
exact post-descent accept gate.

### 5.6 Relocation moves (`_relocation_moves`, `_soft_relocation_moves`)

The move type 2-opt **can't** make. 2-opt only *exchanges* two macros'
positions — it can never relocate a routing-heavy macro into an empty
low-congestion gap (a swap would dump some other macro into the vacated hot
spot). Relocation adds exactly that missing move:

For each of the hottest movable macros:

1. Read its current cell's local cong (or density) → `local_cong[i]`.
2. From the global pool of low-cong (or low-density) candidate cells, keep
   only those *strictly colder* than `local_cong[i]`.
3. Sort the candidates by Euclidean distance to the macro's current position
   (nearest first) — local moves cost less wirelength, more likely to pass
   the proxy gate.
4. For each candidate (up to `n_targets`), clip to keep in-bounds, then
   `score_move(i, target)`. Hard moves include an overlap check vs other
   hard macros; soft moves don't (softs may overlap).
5. Track the best target that strictly lowers the true incremental proxy.
   If found, `commit_move(i, best)` persists the move and `best_score`
   ratchets down.

**S11 WL-delta prefilter (soft relocation).** Step 4 for soft relocation first
computes a cheap WL-only delta (`wl_delta_move_soft`, bit-exact, no routing/density
work) and **skips the full `_trial_at_soft`** when the WL increase alone exceeds
`1e-4` (the routing-apply is ~73% of a trial's cost). Soft relocation commits only
the best target per macro, so skipping non-best candidates is free; this drops ~37%
of soft-relocation trials (~10% of total scoring time) with no quality loss. The
exact gate still validates every survivor. Tunable via `SOFT_RELOC_WL_PREFILTER`;
the soft-soft 2-opt pass has the analogous prefilter at `3e-4`.

**The hot/cold fields:**

| Move type | Field source | Reads from |
|---|---|---|
| Hard reloc | `max(H, V)` routing cong | `plc.get_*_routing_congestion()` |
| Soft cong (R3) | `max(H, V)` routing cong | `plc.get_*_routing_congestion()` |
| Soft density (R5) | occupancy `grid_occupied / dens_grid_area` | `incremental_scorer.grid_occupied` |

The soft passes use `top_hot=128` and `n_targets=24` (top ~6–14% of softs
per round on IBM). R5 boosts to `top_hot=192` on rounds 4–6 (where cong is
skipped by the A+C optimization) to spend the freed budget on more density
attempts. Hard relocation uses `top_hot=R2_HOT=48` with, by default since
2026-06-11, the S10 ML filter: the candidate pool widens to `n_targets=32`
and the shipped XGBoost ranker
(`ml_data/models/clean-wide32-holdout-ibm13-001`) picks the 16 candidates to
exact-score — same scoring budget as the old heuristic `n_targets=16` path.
`src/main.py` sets these defaults only when no `ML_*` env var is present;
otherwise (or if the model / `xgboost` is missing) hard relocation falls back
to the pure-heuristic `n_targets=R2_TGT=16`. The exact accept gate is
unchanged either way. See ISSUES.md S10.

**Mean-field coupling.** Softs don't relocate "against each other" pairwise
or "against hard macros" specifically. The fields are *aggregates* of all
contributions (hard + soft) on the shared grid, and each move is decided by
its effect on the global proxy. Hard macros are *fixed* during a soft pass;
they contribute to the field but don't move. Softs co-adapt to each other
*through the shared grid* — a later soft in the same pass sees the field
updated by earlier commits.

**Optional region soft-lock (`region_bbox` / `region_bias`).** `_relocation_moves`
(and its propose-all variant) accept an optional per-macro center-feasible box
`region_bbox[n,4]` plus a `region_bias` weight. When set, step 3's distance
ranking adds a penalty `region_bias · max(cw,ch)² ` to every candidate cell
*outside* macro `i`'s box, pushing in-region cells to the front so they fill
`n_targets` first; out-of-region cells are reached only when in-region options
run out. It is a **soft** lock — a macro can still exit its box on a large
proxy win — and is **bit-identical when `region_bbox is None`**, which the
leaderboard pipeline always leaves it. Only the hierarchy-floorplan region-relief
pass (§5.9) currently sets it. See ISSUES.md S20.

### 5.7 IncrementalScorer internals (B3 phases + P3)

The scorer is built in `IncrementalScorer.__init__` from the current
placement (~25–48 ms one-time per benchmark per pass). It maintains:

- `per_net_hpwl`, `total_wl_raw` — for B3p2 (per-net incremental WL).
- `H_flat`, `V_flat`, `H_macro_flat`, `V_macro_flat` — for B3p4
  (per-net incremental routing).
- `H_smoothed`, `V_smoothed` — 2D smoothed cache for incremental cong cost
  (§5.4).
- `grid_occupied` — for P3 (incremental density).
- `committed_hard_pos`, `committed_soft_pos` — running state.
- `_route_struct_cache: dict[module → struct]` — for idea #2 topology cache.
- `macro_to_nets` — gathered from `wl_cache`'s `ref_idx`/`pin_to_net` via
  vectorized stable-sort + boundary-partition (built once).

Move methods:

| Method | Inputs | What it does |
|---|---|---|
| `score_swap(i, xy_i, j, xy_j)` | Two hard macros + new positions | Trial-score the swap; snapshot/apply/compute/revert; bit-exact vs `_exact_proxy` |
| `commit_swap(i, xy_i, j, xy_j)` | Same | Persist the swap (no revert) |
| `score_move(i, xy)` | One hard macro + new position | Single-macro relocation trial |
| `commit_move(i, xy)` | Same | Persist |
| `score_move_soft(k, xy)` | One soft macro + new position | Soft trial (no macro-routing blockage) |
| `commit_move_soft(k, xy)` | Same | Persist |

### 5.8 Verification regime (the foundation)

The accept-on-true-proxy guarantee is only as good as the incremental
scorer's correctness. Every move-path is verified bit-exact against
`_exact_proxy`:

| Verifier | Path | Tolerance | Drift over commits |
|---|---|---|---|
| `_verify_incremental_scorer.py` | swap | Δ ≤ 4.4e-16 (machine eps) | 0 |
| `_verify_score_move.py` | hard move | Δ ≤ 1.8e-9 | stable |
| `_verify_score_move_soft.py` | soft move | Δ ≤ 5e-10 | stable |
| `_verify_subset_routing.py` | `_apply_net_routing_subset` / `_apply_macro_routing_subset` vs full routing | bit-exact | — |
| `_verify_congestion.py` | vectorized `_patch_plc_congestion` vs scalar `plc.get_congestion_cost` | bit-exact | — |
| `_verify_density.py` | vectorized `_patch_plc_density` vs scalar `plc.get_density_cost` | bit-exact | — |
| `_stress_verify.py` | Many sequential commits, observe drift | none over 1000s of moves | |

Every speedup added to the scoring path must pass these verifiers before
shipping. This is the discipline that lets us add five mutually compounding
speedups without ever introducing a regression.

### 5.9 Hierarchy-floorplan mode + region-locked relief (non-proxy, opt-in)

The main pipeline (§4) optimizes the congestion-dominated proxy, which **rewards
spread** — connected subsystems get scattered across the chip. The
hierarchy-floorplan mode is a separate, opt-in production path
(`_hierarchy_floorplan` in `pipeline/macro_placer.py`) that instead keeps
connected subsystems together, **trading proxy for hierarchy by design**. It is
off by default and never affects the leaderboard path. Enable with
`V2_HIER_FLOORPLAN=1` (explicit non-proxy deliverable) or `V2_REGION_LOCK=1`
(production output should respect regions) — both route to the same path.

**The regional pipeline:**

1. **Derive clusters** — connectivity "subsystems" from the netlist. Hard macros
   are grouped by a weighted hard-hard graph (each low-fanout net forms a clique
   among its hard pins; merge edges with shared-net count ≥ `V2_CLUSTER_MIN_EDGE`,
   default 2). `derive_cluster_softs` then attaches each soft macro to the cluster
   it shares the most nets with. NB: macro index space A (placement order) vs
   space B (`modules_w_pins`, ports-first) — clustering runs in B and maps back to
   A; see ISSUES.md S20.
2. **Grouped global placement** — run DREAMPlace with synthetic clique nets per
   cluster (the Bookshelf grouping in `pb_to_bookshelf.py`, weight
   `V2_HIER_GROUP_WEIGHT`, default 8), which pulls each subsystem together in the
   analytic placement.
3. **Cluster-consecutive legalization** — legalize with the cluster members
   placed back-to-back (the legalizer's `order` parameter), so each subsystem
   claims its region before others invade. This recovers ~42% of the grouping
   that the default global-largest-first order scatters at legalize time.
4. **Region-locked congestion relief** (`V2_HIER_REGION_RELIEF`, default on in
   this mode) — the key lever. The dense per-cluster packing has bad local
   congestion, and the legalized hard macros can't be moved by soft-only cleanup.
   This pass moves **hard macros to colder cells within their own cluster region**:
   - `compute_region_bbox` (clusters.py) sizes a per-cluster box, area =
     member-area / `V2_HIER_REGION_DENSITY` (default 0.65; higher = tighter lock /
     less relief), centered on the footprint midpoint, never smaller than the
     current footprint, always containing each macro's current spot.
   - It then runs region-biased hard `_relocation_moves` (§5.6 soft-lock) +
     soft relocation, interleaved and true-proxy-gated, finishing with a
     cluster-consecutive safety legalize. Knobs: `V2_REGION_BIAS` (1.0),
     `V2_HIER_REGION_ROUNDS` (2), `V2_HIER_REGION_BUDGET_S` (40).

   **Result:** proxy drops while hard↔hard and intra-cluster closeness stay
   ~unchanged — ibm01 1.0194→0.9469 (−0.073, closeness Δ≈0), ibm10
   1.8215→1.6809 (−0.14, closeness loosens modestly, tunable). Congestion is
   relieved locally without sending any macro far from its cluster.

**Why region-locking is confined to this dedicated path.** Region-biasing only
the *main* pipeline's R2 relocation is ineffective: its other phases (DP
candidates, noise restarts, LSMC kicks) generate spread placements that win on
proxy and override the bias, and anchoring regions to the already-spread
`initial.plc` makes the boxes too loose (measured no-op: ibm10 1.0834→1.0818).
So `V2_REGION_LOCK` runs the dedicated regional pipeline above rather than
trying to constrain the spread-oriented main flow. Validity is asserted by
`test/verification/_verify_region_relief.py`; the proxy/closeness tradeoff is
measured by `test/diagnostic/_hier_region_relief.py`. Full investigation:
ISSUES.md S20.

---

## Timing Notes

The current 2026-06-11 `--all` re-baseline completed in **2337s (~39 min)**,
all 17 VALID / 0 overlaps. The floor-reservation allocator still reserves a
worst-case 3321s of place time, but the present move set often converges or
skips saturated passes before consuming that reserve.

Historical note: the L-change + M1 snapshot was 3662s (61.0 min) in WSL and
marginally over the 60-minute harness limit because it consumed the full
floor-reserved budget plus host overhead. The overage breakdown was:
- ibm01: 304.7s wall (233s place + 71.7s cold Python startup + scoring)
- ibm02-18: ~210.8s each (193s place + ~17.8s warm scoring)

### Why M-change (HARNESS_TOTAL_BUDGET_S 3300->3500) does NOT help ibm15-18

With 3500 total budget and current constants:
- ibm01: still hardcoded 150s -> 233s place
- ibm02: formula cap = (3500-233) - 193x15 - 83 = 289s (gets extra 179s vs 110s floor)
- ibm03: cumulative = 233+372=605s -> cap = (3500-605) - 193x14 - 83 = 110s (floor)
- ibm15: cumulative ~2728s -> cap = (3500-2728) - 193x3 - 83 = 110s (still floor)

M-change only benefits ibm02. Total wall-clock rises to ~3841s (~64 min) -- worse.

To give ibm15-18 more budget the correct lever is reducing BUDGET_OVERRUN_S:
- BUDGET_OVERRUN_S 83->65: saves ~306s wall-clock (13s/benchmark x ~17+ibm01 delta)
- Impact: ~2-3 fewer R2 rounds per benchmark; negligible score change (~0.005)
- New total wall-clock: ~3356s = 56 min (well under 60-min limit)

---

## 6. Planned speedups and GPU notes

The per-move scorer is CPU + numba (~0.5 ms/move with numba; the routing-apply
fill is ~74% of it). The score is **coupled to throughput** — accept-on-true-proxy
is strictly non-regressing, so within a fixed budget more moves ⇒ lower proxy.
Current parallelism is intentionally narrow: DREAMPlace subprocesses run
asynchronously when available, and Phase 9 legalization trials can run in a
small thread pool. The retired multi-seed 2-opt multiprocessing path is no
longer part of the active design.

### 6.1 S2 — GPU-batched candidate evaluation (explored, NOT a win — removed)

The relocation / 2-opt passes try `n_targets` (16–24) candidate positions per hot
macro **sequentially**, each a full `_trial_at`. They are independent trials from
a shared base (the macro removed), so in principle they batch into one GPU op. This
was built end-to-end (three variants), verified bit-exact against the sequential
`_trial_at` loop, measured — and then **removed**, because none beat the numba
sequential path on the IBM grids. The keystone that made it sound: box-filter
smoothing is **linear**, so a candidate's congestion decomposes into a shared base
(macro removed) plus a localized, batchable delta — verified bit-exact (Δ ≤ 6.7e-16).

**Root cause it lost (measured).** The GPU compute ceiling is ~12×, but it's
unreachable at this granularity: per-candidate **strip generation** (routing
bucketing, looped K on CPU) is **~4.9 ms/macro — ~73% of the whole sequential
cost** — and dominates regardless of how fast the GPU reductions run. The IBM
grids are small (~2000 cells) and the per-macro batch (K≈24) is too small to
amortize kernel-launch overhead, while the numba sequential fill is already
well-optimized. Best variant landed at 0.97× (CPU fill → GPU smooth/topk); the
GPU-resident variant was 0.67× (the CPU strip loop became the bottleneck).

**What could still win (not pursued).** Only **cross-macro batching** — evaluate
*all* hot macros' candidates (top_hot × n_targets ≈ 1000+) in one GPU batch to
amortize launch overhead and vectorize strip-gen — and that requires restructuring
relocation from sequential prep→trial→commit into evaluate-all-then-commit, which
changes accept semantics (a different algorithm). The approach would also pay off
on **much larger grids** (industrial / NG45) where per-batch work amortizes GPU
overhead; IBM is simply too small. The implementation + verifiers were deleted
(zero production impact) — this section records the finding so it isn't re-run.

---

## Known Issues / Next Ideas

- DREAMPlace is now built and functional. The earlier Python/ABI break is
  resolved by launching the subprocess with the DREAMPlace build env; its
  outputs are scored as ordinary candidates.
- **numba** must be installed for full speed (JITs the routing-apply, ~half the
  runtime; soft import with a numpy fallback). **CRITICAL (S13):** numba is in
  `v2/requirements.txt` but NOT `pyproject.toml`, so `uv sync` alone does **not**
  install it and the placer falls back to numpy **silently** (~25 % slower in the
  S13 measurements, enough to lose deadline-bound refinement). `config.py` now warns
  when it's missing; **the eval env must install requirements.txt**. See ISSUES S13.
- **GPU batching — see §6.** S2 GPU candidate batching was implemented, verified bit-exact, and
  measured a net loss on IBM grids; the code + verifiers were removed (the §6.1
  finding is kept so it isn't re-run).
- **At the IBM floor for this move set (S15):** budget is dead (benchmarks
  converge) and width tuning is noise-level. ibm12 (1.31), ibm17 (1.36), ibm18
  (1.39) are the remaining highest-proxy benchmarks; further gains need generic
  LSMC seed/kick/descent improvements or new move types.
- WireMask-BBO greedy evaluator: highest-leverage non-GPU idea not yet implemented.
- Current `--all` timing has comfortable headroom (~39 min in the 2026-06-11
  run). Reducing `BUDGET_OVERRUN_S` remains the safe knob if a slower machine or
  fallback dependency set pushes runtime toward the 1h limit.
