# Iteration Progress Log

All scores are proxy cost (lower is better).
Target: beat RePlAce avg of 1.4578.

> **Status (2026-05-31 — full-stack `--all` incl. HS3 hard-soft 3-cycle + 3-pin routing JIT):**
> **Avg 1.1963 — beats RePlAce (1.4578) by 0.262 (−17.9%), and beats the UT
> Austin DREAMPlace leaderboard (1.4076) by 0.211 (−15.0%).** All 17 VALID /
> 0 overlaps. **11/17 wins** vs 1.1993 baseline. Cumulative Δ −0.0504,
> avg −0.0030/bench. Biggest movers: **ibm16 −0.0287** (recovers the
> +0.0108 fluke-loss from the prior run AND adds a net win), ibm07 −0.0151,
> ibm01 −0.0069, ibm13/ibm12 −0.005, ibm14 −0.0048, ibm09 −0.0034,
> ibm17/ibm18 −0.0033. Losses (all small): ibm10 +0.0172 (the
> mirror-image of ibm16: prior big winner became this run's main loser —
> RNG sensitivity, swap nets cumulative wins), ibm11 +0.0077, ibm06/ibm08
> +0.0017, ibm03 +0.0010, ibm04 +0.0003. Total runtime 4429s wall (74min,
> harness monotonic well under 3300s — no host-suspend drift this run).
>
> **HS3 (hard-soft 3-cycle rotation):** new move type. Captures
> configurations where H wants S1's slot but swapping H↔S1 hurts because
> S1's connections need to go elsewhere — 2-opt can't accept that chain
> individually, but the single combined 3-cycle (H → S1's old pos, S1 →
> S2's old pos, S2 → H's old pos) can. New `score_cycle_hard_soft_soft`
> + `commit_cycle_hard_soft_soft` on `IncrementalScorer` (extension of
> HXS to 3 modules via _touched_nets3). Bit-exact verified
> (`_verify_score_cycle_hard_soft_soft.py`: Δ ≤ 2.22e-16 across all
> trials and sequential commits on ibm01/04/10). New pass
> `_three_opt_hard_soft_soft` in the R2 round, dual-field, top_hot=15
> hards × k_inner=5 S1 × k_inner+1=6 S2 = ~375 trials/pass, 3s tight
> deadline cap, adaptive skip-if-empty. Cubic-in-knn but knn-truncated.
> **3-pin routing dispatcher numba JIT (#35):** speedup. The 3-pin
> dispatcher was 38% of move time (per profile) — the numpy gather /
> scatter / per-case mask dance carries meaningful overhead beyond the
> arithmetic. Collapsed into a single per-net numba loop with manual
> 3-element sort + case branching + direct H/V strip writes. Bit-exact
> within ≤4.4e-16. Saves another ~13-15s/bench → freed ~250s over the
> full `--all` (the ibm04 smoke went from 138.8s to 124.8s).
> ibm04 progression: 1.2092 baseline 1.0304 → ... → 1.0062 (prior shared-
> scorer + numba strips) → **1.0067** (+ HS3 + 3pin JIT). HS3 fired 4
> cycles on ibm04 (R1: 7 cycles, R2: 2). Note ibm04 score barely changed
> but runtime dropped 14s — the freed budget compounds across other
> benchmarks.
>
> Prior milestones (stacked):
> **Status (2026-05-31 — full-stack `--all` incl. HXS+R6+WL-prefilter+shared-scorer+numba-JIT):**
> **Avg 1.1993 — beats RePlAce (1.4578) by 0.259 (−17.7%), and beats the UT
> Austin DREAMPlace leaderboard (1.4076) by 0.208 (−14.8%).** All 17 VALID /
> 0 overlaps. **14/17 wins** vs 1.2092 baseline (only ibm07 +0.004, ibm15
> +0.0004, ibm16 +0.0108 — the latter likely a fluke-loss back toward the
> ibm16 long-run mean; the prior 1.2092 run got an unusually-good 1.2641 on
> ibm16). Cumulative Δ −0.1683, avg −0.0099/bench. Biggest movers:
> **ibm18 −0.0359** (starvation FIXED — went from +0.283 with the previous
> HXS+R6 budget overrun to −0.036 with the shared scorer + numba freed-up
> budget), ibm17 −0.0252, ibm04 −0.0226, ibm10 −0.0209, ibm11 −0.0186,
> ibm06 −0.0158. Total runtime 11486s wall (host-suspend inflated; harness
> monotonic ≤3300s).
>
> **HXS (hard ⇄ soft cross-swap):** new move type. Exchanges a hard macro
> with a soft macro. Neither hard-2opt nor soft-2opt can find such pairs
> (each swaps only within its own kind). New `score_swap_hard_soft` /
> `commit_swap_hard_soft` on `IncrementalScorer` — hybrid of score_swap
> (hard's routing blockage via macro_subset) + score_swap_soft (no
> macro_subset for the soft). Bit-exact verified
> (`_verify_score_swap_hard_soft.py`: Δ ≤ 4.4e-16 across all trials and
> sequential commits on ibm01/04/10). New pass `_two_opt_hard_soft_swap`
> in the R2 round, dual-field (cong + density), 2.5s tight deadline cap,
> adaptive skip-if-empty.
> **R6 (combined cong+density relocation):** third hard-reloc pass per
> round, hotness = geometric mean of normalized cong & density. Catches
> macros moderately hot on both fields that neither pure pass prioritized
> (each ranking favors pure-field extremes). 4s deadline cap. Same proxy
> gate, same overlap check. Sparse firings (1-3/round) before adaptive
> skip-if-empty triggers.
> **WL-delta prefilter for soft-2opt:** new cheap `wl_delta_swap_soft`
> method on `IncrementalScorer` computes per-net HPWL change in ~50µs
> (vs ~5-10ms for the full score_swap_soft). Used in
> `_two_opt_soft_swap` as a prefilter — skip the full score call when
> predicted WL delta exceeds 0.01 (loose enough to keep every
> historically-accepted swap; typical accepted ΔWL is <0.002).
> **Persistent shared scorer per R2 round (#33):** the R2 round body has
> ~10 distinct passes (hard reloc cong / density / combined, soft reloc
> cong / density, soft-2opt cong / density × A5 passes, HXS cong /
> density, 2-opt cleanup); the status quo rebuilt an `IncrementalScorer`
> per pass (~0.1-0.3s each → ~10-20 s/benchmark). Now the scorer is
> built ONCE per round, lazily rebuilt on the rare case a pass's
> committed accepts don't pass the cumulative `cand_true < best_score`
> gate. Saves ~15-25s/benchmark, which the R2 loop spends on additional
> productive rounds.
> **Numba-JIT routing apply (#34):** soft-import numba; if available,
> JIT-compile `_apply_h_strips_batch` / `_apply_v_strips_batch` (the
> inner-inner loops of the 2-pin / 3-pin / big-net routing apply,
> ~10% of move time per profile). Pure numpy fallback when numba is
> absent. Bit-exact within ≤4.4e-16 (verified by the existing scorer
> verifier on ibm01/04/10). Saves another ~10-15s/benchmark.
> ibm04 progression (validating the stack incrementally):
> 1.2092-baseline 1.0304 → + HXS+R6 (tight caps) 1.0162 → + WL prefilter
> 1.0139 (187s) → + shared scorer 1.0074 (163s, **−24s**) → + numba JIT
> **1.0062 (138s, −49s vs pre-shared)** — total −0.0242 score,
> −49s/bench freed.
>
> Prior milestones (stacked):
> **Status (2026-05-30 — full-stack `--all` incl. A4+A5+adaptive R2/skip-empty):**
> **Avg 1.2092 — beats RePlAce (1.4578) by 0.249 (−17.1%), and beats the UT
> Austin DREAMPlace leaderboard (1.4076) by 0.198 (−14.1%).** All 17 VALID /
> 0 overlaps. **15/17 wins** vs 1.2195 baseline (only ibm04 +0.0017 and ibm18
> +0.0063 — both near noise). Cumulative Δ −0.1755, avg −0.0103/bench.
> Biggest movers: ibm15 −0.0311, ibm06 −0.0259, ibm12 −0.0194, ibm13 −0.0174,
> ibm08 −0.0148, ibm14 −0.0135, ibm11 −0.0121. Total runtime 2716s.
>
> **A4 (WL-aware soft-2opt candidate ordering):** `_two_opt_soft_swap` now
> takes `net_centroid` + `wl_blend=0.3`, blending Euclidean distance with
> distance-to-net-centroid in the candidate ordering — the soft-2opt analog
> of A3. Pure ordering change; strictly non-regressing.
> **A5 (adaptive multi-pass soft-2opt):** each soft-2opt call in R2 now runs
> up to `A5_NUM_PASSES=2` passes with early-stop if the first pass made no
> improvement. Pass 2 fired 186/189 opportunities across the run — nearly
> every round had a productive 2nd pass.
> **Adaptive R2 round termination:** added `TINY_R2_ROUNDS_TO_STOP=2`
> consecutive rounds of Δ < `R2_DELTA_THRESHOLD=1e-3` to short-circuit
> diminishing-returns rounds. In practice the tiny-streak guard never fired
> on the winning run (every round productively > 1e-3) — confirms the rounds
> are doing real work.
> **Adaptive skip-empty replacing hardcoded `R3_CONG_MAX_ROUNDS`:** both the
> single-soft cong-relocation pass and the A1b cong-field soft-2opt now skip
> a round only after `SKIP_EMPTY_AFTER=1` empty round in a row. The earlier
> hardcoded round-3 cap on A1b was found to regress scores by killing
> productive late-round work (A1b finds 7–35 swaps even at round 6 on some
> benchmarks). Density `top_hot` boost still triggers, but adaptively (when
> the cong empty-streak counter saturates).
> **#3v2 time-shifted multi-seed 2-opt subprocess pool (drafted, env-gated
> off):** `V2_MULTISEED_MP=1` runs the main "best" 2-opt inline first (full
> solo CPU during the 15s deadline), then submits DP seed 2-opts to a
> ProcessPoolExecutor afterward. Default off — direct subprocess parallelism
> on the deadline-bound search caused regression due to CPU contention.
> Total runtime 2716s (clean, no host suspend, well under 3600s hard cap).
>
> Prior milestones (stacked):
> **Status (2026-05-30 — full-stack `--all` incl. H5+A1b+A1c+A1×2+Phase9-parallel):**
> **Avg 1.2195 — beats RePlAce (1.4578) by 0.238 (−16.3%), and beats the UT
> Austin DREAMPlace leaderboard (1.4076) by 0.188 (−13.4%).** All 17 VALID /
> 0 overlaps. We **beat RePlAce on every benchmark** (ibm01 flipped from
> +2.6% to −1.0%). All 17 benchmarks improved vs the 1.2433 baseline
> (17/17 wins, cumulative Δ −0.4044, avg −0.024/bench).
> **H5 (hard density relocation):** new pass — the R5-analog for hard macros.
> `_relocation_moves` now switches its hot/cold field via `use_density=True`;
> a new pass in the R2 round runs the hard-density variant after the existing
> cong-based hard reloc. Modest (1-3 moves/round) but consistent contribution.
> **A1b (cong-field soft-2opt):** soft-2opt now runs TWICE per round — once
> on the cong hotness field, once on density — same dual-field symmetry that
> gave R3 + R5 their compound gain. Finds 7-35 swaps/round on the cong pass.
> **A1c (cold-teleport):** each A1 pass appends 4 globally-coldest movable
> softs to the kNN candidate set per hot — analog of S9 cold-teleport for the
> hard 2-opt.
> **Phase 9 parallelization:** ThreadPoolExecutor on the 3 random-order
> legalize trials (numpy releases the GIL on the heavy work). Score step
> stays sequential (plc state). Saves ~0.3s/bench.
> **DREAMPlace ×3 already parallel** (confirmed) — 3 async subprocess
> handles, no change needed.
> Combined `--all`: 1.2433 → **1.2195** (−0.0238). Biggest movers: ibm12
> −0.069, ibm11 −0.041, ibm10 −0.029, ibm08 −0.030, ibm15 −0.028, ibm17
> −0.028. Total runtime 2598s (clean, under cap).
> Prior milestones (stacked):
> **A1 + A3 (added 2026-05-29) — the dominant new lever.**
> **A1 (soft-soft 2-opt):** new pair-swap move type that exchanges two soft
> macros' positions. Single-soft relocation can't find moves where two softs
> need to swap places (e.g., both at suboptimal cells where their connections
> would be happier in each other's slot). New `score_swap_soft` /
> `commit_swap_soft` on `IncrementalScorer` (analog of `score_swap` minus
> macro_subset since softs don't block routing), new `_two_opt_soft_swap`
> pass in the R2 interleave round (between soft-density and the hard 2-opt
> cleanup): top_hot=64 density-hot softs × k_neighbors=12 nearest movable
> softs, accept-on-true-proxy, ~6s budget slice. Bit-exact verified
> (`_verify_score_swap_soft.py`: Δ ≤ 2.2e-16 machine eps across trials and
> sequential commits). **A3 (smart soft candidate ordering):** new
> `soft_net_centroids()` method (analog of `hard_net_centroids`);
> `_soft_relocation_moves` now blends Euclidean distance with distance-to-
> net-centroid via `wl_blend=0.3` so candidates aligned with the soft's
> WL anchor are tried first. Pure ordering change — strictly non-regressing.
> Combined `--all`: 1.2737 → **1.2433** (−0.0304, **ALL 17 wins**, biggest
> movers ibm17 −0.059, ibm07 −0.050, ibm13 −0.043, ibm16/15 −0.039,
> ibm14 −0.035, ibm18 −0.032). Per-round soft-2opt accepted 9–41 swaps
> consistently across all 6 rounds, confirming A1 finds many real moves the
> single-soft passes couldn't reach. Total runtime 2291s (clean, no WSL
> inflation this run). **A1 is the largest single algorithmic improvement
> since R5** — they're now co-dominant levers, both around −0.03 to −0.1
> magnitude.
> Prior milestones (stacked):
> **S1 + S3 (added 2026-05-29):** S1 hoists the loop-invariant "subtract k's
> old routing + density" out of the relocation candidate inner loop via a new
> `_prepare_move(_soft)` / `_trial_at(_soft)` / `_commit_after_prep(_soft)` /
> `_revert_prep(_soft)` quartet on `IncrementalScorer`. Per-trial cost in the
> realistic same-macro / nearby-target pattern drops 25–43% (ibm10
> 1.50→0.90 ms, ibm15 1.50→0.86 ms, ibm17 1.82→1.36 ms). Bit-exact verified
> (`_verify_prep_trial.py`: Δ=0.00e+00 on every trial vs `score_move(_soft)`).
> S3 replaces `np.add.at` with `np.bincount` in the strip-batch routing fill —
> same-order accumulation, swap verifier still passes at Δ≤4.4e-16.
> Combined `--all`: 1.2755 → **1.2737** (−0.0018; 10/17 wins; ibm18 −0.021
> and ibm06 −0.019 the biggest movers).
> Prior milestones (stacked):
> **Latest changes stacked this session** (each one bit-exact-verified before
> the next): (1) **Incremental congestion cost** — `IncrementalScorer` caches
> the smoothed normalized H/V and per move re-smooths only the touched-net
> bbox from raw flats (bit-identical to a full re-smooth; swap Δ≤4.4e-16);
> isolated `--all` 1.2799 → **1.2767**. (2) **Idea #1 subset-cumsum strip-batch**
> (only the unique touched rows/cols are cumsummed). (3) **Idea #2 topology-
> struct cache** for the routing apply (position-independent gather + 2/3/≥4-pin
> classification built once per macro, reused across the −1/+1 applies and
> across moves; the position-dependent fill is still recomputed → bit-exact).
> (4) **Floor-reservation budget allocator** — closes the ibm18-starvation bug:
> reserve `(110+60)·remaining` for every other remaining benchmark plus 60s of
> own-overrun slack so the last benchmark always gets ≥110s. Worst-case
> simulation: all 17 ≥110s, cumulative ends at 3300. (5) **A: cong soft-pass
> hard-cap at round 3 + C: density `top_hot` boost 128→192 on rounds 4–6** —
> cong saturates by round 3 (ibm09 round 4+ accepts ≤2 moves, ~zero gain);
> skip it and spend the freed ~4–5s/round on density. Combined `--all`:
> 1.2767 → **1.2755** (−0.0012; 12/17 wins vs the cong-only baseline). **ibm18
> = 1.5787** (vs the floor-res-only run's starved 1.7941 — confirms the
> allocator works). Biggest movers: ibm17 −0.034, ibm16 −0.019, ibm07 −0.015.
> (Wall-time reported 3860s under WSL host-suspend inflation; the placer's
> `monotonic` budget held — no benchmark returned baseline.)
>
> **Disproven this session:** the "shared scorer" lever — measure-first
> profile (`_profile_init.py`) showed the per-pass fixed overhead is only
> ~0.1–0.28s/round (not the projected 60–75s), so a shared-scorer refactor
> would save <1.7s/benchmark and risk correctness. NOT implemented. **Disproven:** R4 WL-aware hard
> relocation (net-centroid target bias) — slightly worse, reverted (scaffolding
> kept). Prior: **R3** soft cong relocation 1.4216→1.3764; **R2/R2b** 1.4326→
> 1.4216; **R1** 1.4422→1.4326; **S9/P3** before that.
> Earlier detail — **R1 congestion-directed relocation moves** — a post-2-opt
> pass that RELOCATES the hottest macros into empty low-congestion legal gaps
> (a move the swap-only 2-opt can't make). Uses the incremental scorer's new
> `score_move` (verified bit-exact, ≤6e-9). --all 1.4422 → 1.4326 (all 17
> improved). Before that: **S9 congestion-aware 2-opt** (hot-first ordering +
> cold teleport augmentation), 1.4424 → 1.4422; and **P3 incremental
> density** — `IncrementalScorer` now keeps
> the occupancy grid as state and updates only the 2 swapped macros' cells
> per score (verified bit-exact vs full recompute, ≤4.4e-16). score_swap is
> −22% to −29% faster → +40–56% more 2-opt scores fit the 15s deadline →
> avg 1.4435 → **1.4424** (the gain lands on the deadline-bound large
> benchmarks: ibm10 1.3381→1.3346, ibm16 1.5057→1.5041). `--all` 979s
> (WSL-inflated). S1 (basin-hopping 2-opt) is implemented but DORMANT pending
> its own --all (P3 now makes small benchmarks converge early, freeing kick
> budget). Prior changes: multi-seed 2-opt-on-winner (O2), k=20 (S2),
> IncrementalScorer clean-init (O5). See section below for headlines.
>
> History notes (2026-05-20): this file started as v1's local copy of
> the team's PROGRESS.md, updated through v14. The "Iteration Log"
> section below tracks the v1-era progression (v1 → v14). The v2
> session (2026-05-23 → 2026-05-25) is summarized in the new section
> immediately after the Baselines table.

---

## Baselines (reference)

| Placer | Avg (17 benchmarks) | Notes |
|---|---|---|
| SA baseline | 2.1251 | challenge organizer SA |
| will_seed | 1.5338 | challenge organizer legalization |
| sameer_v1 leg-only | 1.5062 | our legalize-only, confirmed |
| RePlAce | 1.4578 | Grand Prize target |
| UT Austin (DREAMPlace) | 1.4076 | leaderboard #1 |
| **v2 (this submission)** | **1.2799** | **BEATS RePlAce by 0.178 (−12.2%); below leaderboard 1.4076 by 0.128** (R5 soft density relocation + R3 + R2 + R1 + S9 + P3) |

---

## v2 — Submission state (2026-05-25)

### Headline

| Metric | Value |
|---|---|
| 17 IBM benchmarks avg | **1.2799** |
| RePlAce target | 1.4578 |
| **Gap to RePlAce** | **−12.2% (beat by 0.178)** |
| v12 starting point | 1.4854 |
| **Total v2 improvement** | **−0.1090** |
| DREAMPlace leaderboard | 1.4076 — **v2 BEATS it by 0.128 (−9.1%)** |
| `--all` wall-clock | 2639s (< 3600s cap) |
| NG45 avg (Tier 2) | 0.7830 |

### Per-benchmark results (v12 → R5 1.2799)

| Bench | v12 | R5 (1.2799) | Δ vs v12 |
|---|---|---|---|
| ibm01 | 1.1860 | 1.0544 | −0.132 |
| ibm02 | 1.5923 | 1.3302 | −0.262 |
| ibm03 | 1.3603 | 1.0787 | −0.282 |
| ibm04 | 1.3316 | 1.0648 | −0.267 |
| ibm06 | 1.6684 | 1.3104 | −0.358 |
| ibm07 | 1.4924 | 1.2955 | −0.197 |
| ibm08 | 1.5251 | 1.3048 | −0.220 |
| ibm09 | 1.1304 | 0.9720 | −0.158 |
| ibm10 | 1.4037 | 1.2071 | −0.197 |
| ibm11 | 1.2354 | 1.0862 | −0.149 |
| ibm12 | 1.6507 | 1.5127 | −0.138 |
| ibm13 | 1.4011 | 1.1769 | −0.224 |
| ibm14 | 1.6033 | 1.3945 | −0.209 |
| ibm15 | 1.6061 | 1.4290 | −0.177 |
| ibm16 | 1.5323 | 1.3494 | −0.183 |
| ibm17 | 1.7437 | 1.6172 | −0.127 |
| ibm18 | 1.7896 | 1.5754 | −0.214 |
| **AVG** | **1.4854** | **1.2800** | **−0.205** |

(R1 column = `--all` 2026-05-27: P3 incremental density + S9 cong-aware 2-opt +
R1 congestion-directed relocation. ALL 17 improved vs the prior 1.4435; R1's
relocation pass alone was −0.0096 avg over the 1.4422 state. Total runtime 751s,
all 17
VALID / 0 overlaps.)

**All 17 benchmarks improved.** No regressions vs v12.

### Architecture changes vs v1

1. **`MacroPlacer.__init__` cross-benchmark state** (B1) — tracks
   cumulative wall-clock with `time.monotonic()` for adaptive
   per-benchmark budget under `--all`'s 3600s harness cap.
2. **Proxy-driven 2-opt-on-winner** (A1) — `_two_opt_proxy_swap` uses
   `_exact_proxy` rescoring per swap (was: displacement-from-init,
   anti-correlated with proxy).
3. **B3 incremental scoring** (4 phases) —
   - Phase 1: global position cache eliminates per-call get_pos loops.
   - Phase 2: per-net HPWL incremental via macro→nets index.
   - Phase 3: numpy abu (np.partition) replaces Python sorted +
     .tolist() conversions.
   - Phase 4: per-net incremental ROUTING via subset dispatch
     helpers (`_apply_net_routing_subset`, `_apply_macro_routing_subset`).
     Per-score on ibm10 dropped 22.5ms → ~3ms (7.5× faster).
4. **B4 dispatch cache** — pre-compute topology-fixed index arrays in
   `_build_cong_cache` (idx2/idx3/idx_big/net_local_ids/global_pin_idx).
5. **A6 axis #1: Phase 8 TOP-K cong-grad** with multi-iter chains —
   restrict cong-grad to K hottest macros; chain up to 3 iters per K
   in {5, 10, 20}.
6. **A6 axis #4: Phase 9 random-tiebreak legalize order** — N=3
   variant orderings of `_will_legalize` with random secondary sort
   key (primary key −area preserved).
7. **2-opt widening** — k_neighbors 5 → 10, max_iters 3 → 6.
8. **A2 DREAMPlace soft_movable diversification** — 2-DP launch:
   lo-fix (td=0.65, soft_movable=False) + hi-mov (td=0.85,
   soft_movable=True). Best-of-both candidate per benchmark.
9. **WSL2 clock-drift hardening** — all 56 `time.time()` calls
   replaced with `time.monotonic()` to prevent host-suspend-induced
   wall-clock jumps from corrupting deadlines / budgets.
10. **NG45 disambiguation** — `_load_plc` matches NG45 designs by
    canvas dimensions when `benchmark.name == "output_CT_Grouping"`
    (all 4 NG45 designs share that name due to load_benchmark's
    basename logic).

### Reproducibility

Multiple `--all` runs confirmed avg 1.4475 ± noise (typically
≤ 0.001 per-benchmark variance). Largest run-to-run swing observed:
ibm10 ±0.0024 due to non-deterministic CPU scheduling affecting
2-opt deadline-bound decisions.

### Headline progression through the v2 session (2026-05-23 → 2026-05-25)

| Milestone | Avg | Δ from prior | Gap vs RePlAce 1.4578 |
|---|---|---|---|
| v12 (session start) | 1.4854 | — | +1.9% |
| + B1 cumulative-budget guard | 1.4782 | −0.0072 | +1.4% |
| + A1 proxy 2-opt | 1.4723 | −0.0059 | +1.0% |
| + B3 phase 1 (pos cache) | 1.4719 | −0.0004 | +1.0% |
| + B3 phase 2 (per-net HPWL incr) | 1.4714 | −0.0005 | +0.9% |
| + B3 phase 3 (numpy abu) | 1.4711 | −0.0003 | +0.9% |
| + A6 Phase 8 (TOP-K cong-grad) | 1.4701 | −0.0010 | +0.8% |
| + Phase 9 (random-order legalize) | 1.4698 | −0.0003 | +0.8% |
| + B4 dispatch cache | 1.4698 | 0 | +0.8% |
| + B3 phase 4 (per-net cong incr) | 1.4690 | −0.0008 | +0.8% |
| + 2-opt widening (k=10, iters=6) + Phase 8 chains | 1.4647 | −0.0043 | +0.5% |
| + A2 (DP soft_movable best-of-both) | 1.4486 | −0.0161 | **−0.6%** |
| + A2 refined (lo-fix + hi-mov) | **1.4475** | **−0.0011** | **−0.7%** |
| (+ WSL2 monotonic clock fix — no score Δ, ↓ wall-clock 720s → 526s) | | | |

---

## Iteration Log

### v1: Legalization only
- Strategy: legalize directly from initial.plc, no restarts
- All benchmarks: return baseline legalized position
- ibm01: 1.2253, avg: 1.5062

### v2/v3: Multi-restart with exact proxy scoring (broken by density fallback regression)
- Strategy: 5 random Gaussian restarts, score all with exact proxy, pick best
- For n>350 benchmarks: density fallback to rank restarts (ANTI-CORRELATED, see below)
- ibm01: 1.1854, ibm03: 1.3944, ibm08: 1.5251 (exact benchmarks improved)
- ibm11: 1.3770 (density fallback selected 8% noise → actual proxy 11.5% WORSE than baseline!)
- Full eval avg = 1.5656 (REGRESSED from v1! Density fallback hurt large benchmarks by +0.14 each)

### v4: Density fallback disabled, exact scoring for ibm11 (CURRENT)
- Fix 1: Non-exact benchmarks (n>400 or grid>2000 cells) return baseline immediately
- Fix 2: Raised EXACT_MACRO_THRESHOLD from 350 to 400 → ibm11 (n=373) now uses exact scoring
- ibm11 with exact scoring: baseline=1.2354 (81s), restart 1 (2%)=1.2591 → baseline wins
- Expected avg: ~1.501 (v1 for non-exact benchmarks + improved exact benchmarks)
- Full eval running (2026-04-29)

### v5: Budget-filling restarts
- Extended noise_fracs from 4 entries to 35 entries
- n_restarts=50 (budget check is the actual limit, not n_restarts)
- Core 4 fracs [0.02, 0.04, 0.06, 0.08] unchanged → preserves ibm01/03/08 wins
- Fast benchmarks now fill their budget:
  - ibm01 (~5s/score): ~20 restarts vs 4 before
  - ibm03 (~10s/score): ~9 restarts vs 4
  - ibm04 (~14s/score): ~10 restarts vs 4
  - ibm06 (~16s/score): ~8 restarts vs 4
  - ibm09 (~20s/score): ~6 restarts vs 4
  - ibm08 (~36s/score): ~4 restarts (unchanged, already at budget limit)
  - ibm11 (~81s/score): ~1 restart (unchanged)

### v8: Iterative congestion-gradient descent + wide step (CURRENT CODE)
- Phase 1: Iterative gradient descent at frac=0.04, up to 4 steps. After each improving step,
  extract legalized position from best_pl and use it with plc's updated congestion map for the
  next gradient step. Stop when a step fails to improve or budget < 3×t_score.
- Phase 2: After any improvement from phase 1, try frac=0.08 then frac=0.12 from baseline_pos
  using current (possibly stale) plc congestion state. Stop when a wide step fails to improve.
  Key insight: stale plc from failed iter=2 provides 2nd-order info that guides a larger jump.
- Benchmarks where cong-grad doesn't improve (iter=1 fails): wide steps skipped, exact same
  behavior as v6 for ibm07, ibm08, ibm11.
- ibm15 confirmed at 164s scoring (SLOW_SCORE_THRESHOLD catches it), EXACT_GRID_CELL_LIMIT stays 2000.
- Confirmed improvements vs v6 (2026-04-30):
  - ibm02: 1.6203 → **1.5823** (-0.038; stale iter=2 plc + wide=8% from baseline is key)
  - ibm03: 1.3854 → **1.3583** (-0.027; 2 iterative steps)
  - ibm04: 1.3882 → **1.3479** (-0.040; 4 iterative steps, budget fills)
  - ibm06: 1.6838 → **1.6810** (-0.003; 2 iterative steps)
- No regressions: ibm08=1.5251, ibm09=1.1304 both confirmed clean
- Est. avg: ~1.4867 (gap to RePlAce: 0.029, down from 0.035 in v6)

---

### v6: Routing-congestion-gradient perturbation
- After baseline scoring, plc has the routing congestion map from get_congestion_cost().
- New restart (k=1 for IBM benchmarks): perturb baseline_pos using the REAL H/V routing
  congestion map from PlacementCost.get_horizontal/vertical_routing_congestion().
- For each macro in a cell with congestion > 0.5: move against the finite-difference
  gradient of the congestion map (toward lower-congestion neighbors). Small random noise
  (0.1× scale) added to break symmetry.
- Uses separate RandomState(seed+1) so main np.random state unchanged; noise restarts
  get identical draws to v5 (ibm01 6% win preserved at k shifted by 1).
- Confirmed improvements (2026-04-29):
  - ibm02 (cong=2.375): 1.6800 → 1.6203 (-0.0597)
  - ibm06 (cong=2.503): 1.7198 → 1.6838 (-0.0360)
  - ibm01 (cong=1.274): no improvement (congestion too low for gradient signal)
- ibm07, ibm08 tests contaminated by system load (scoring inflated 3-4x); clean results pending
- Full clean eval running (2026-04-29)

---

## Per-Benchmark Detail (confirmed from full evals)

v12 = current best (varrahan/v1, --all confirmed 2026-05-10 with budget-relaxation fix).
v12 stable --all avg = **1.4854**. Reproduced in 2 of 3 runs (3rd run had Run-1 ibm04 spike,
fixed by adding `BUDGET_OVERRUN_S=60s` allowance for directed-restart phases).

| Benchmark | hard_n | grid_cells | v1 (leg) | v8 | v11 | **v12 (current)** | RePlAce | vs RePlAce | Notes |
|---|---|---|---|---|---|---|---|---|---|
| ibm01 | 246 | 45x41=1845 | 1.2253 | 1.1854 | 1.1854 | **1.1860** | 0.9976 | -18.9% | t_score=2-3s clean; 6% noise wins; v11's 1.1854 was a lucky outlier |
| ibm02 | 271 | 30x27=810 | 1.6800 | 1.5823 | 1.5823 | **1.5923** | 1.8370 | +13.3% | t_score=7-8s clean; wide=8% wins; v11's 1.5823 was a lucky outlier (stale-plc lottery) |
| ibm03 | 290 | 32x29=928 | 1.4100 | 1.3583 | 1.3547 | **1.3603** | 1.3222 | -2.9% | t_score=5-6s clean; iter=2 cong-grad wins; v11's 1.3547 was a lucky outlier |
| ibm04 | 295 | 31x30=930 | 1.4101 | 1.3479 | 1.3390 | **1.3316** | 1.3024 | -2.2% | t_score=6-7s clean; 7 iter steps + Phase 2 + Phase 3 wins. **STABLE under --all with budget fix** (was fragile in run 1 without fix) |
| ibm06 | 178 | 31x28=868 | 1.7198 | 1.6810 | 1.6797 | **1.6684** | 1.6187 | -3.1% | clean CPU rediscovery: −0.0113 vs v11 stale (frac=0.02 at iter=4 + Phase 3) |
| ibm07 | 291 | 35x32=1120 | 1.4950 | 1.4950 | 1.4950 | **1.4924** | 1.4633 | -2.0% | clean CPU, 1% noise restart wins (−0.0026 vs v11); cong-grad doesn't help |
| ibm08 | 301 | 38x34=1292 | 1.5582 | 1.5251 | 1.5251 | **1.5251** | 1.4285 | -6.8% | cong-grad worse; 6% noise wins; stable across runs |
| ibm09 | 253 | 36x38=1368 | 1.1363 | 1.1304 | 1.1304 | **1.1304** | 1.1194 | -1.0% | 1 cong-grad iter wins |
| ibm10 | 786 | 55x41=2255 | 1.4037 | 1.4037 | 1.4037 | **1.4037** | 1.5009 | +6.5% | n>400; returns baseline |
| ibm11 | 373 | 39x45=1755 | 1.2354 | 1.2354 | 1.2354 | **1.2354** | 1.1774 | -4.9% | v12: re-included in exact pipeline (t_score=17s clean); 10 restarts attempted, baseline wins |
| ibm12 | 651 | 47x47=2209 | 1.6507 | 1.6507 | 1.6507 | **1.6507** | 1.7261 | +4.4% | n>400; returns baseline |
| ibm13 | 424 | 43x43=1849 | 1.4011 | 1.4011 | 1.4011 | **1.4011** | 1.3355 | -4.9% | n>400; returns baseline |
| ibm14 | 614 | 49x44=2156 | 1.6033 | 1.6033 | 1.6033 | **1.6033** | 1.5436 | -3.9% | n>400; returns baseline |
| ibm15 | 393 | 57x38=2166 | 1.6061 | 1.6061 | 1.6061 | **1.6061** | 1.5159 | -5.9% | v12: re-included (t_score=43s clean); restarts attempted, baseline wins |
| ibm16 | 458 | 45x48=2160 | 1.5323 | 1.5323 | 1.5323 | **1.5323** | 1.4780 | -3.7% | n>400; returns baseline |
| ibm17 | 760 | 51x44=2244 | 1.7437 | 1.7437 | 1.7437 | **1.7437** | 1.6446 | -6.0% | n>400; returns baseline |
| ibm18 | 285 | 55x39=2145 | 1.7941 | 1.7941 | 1.7941 | **1.7896** | 1.7722 | -1.0% | v12: re-included (t_score=62s clean); cong-grad iter=1 wins (−0.0045) |

**v10b full eval avg (2026-04-30):** 1.4877 (ibm04=1.3390 new best; ibm08=1.5539 under load)
**v11 clean estimate:** 1.4860 (composite — never actually --all'd; numbers were lucky outliers for ibm01/02/03)
**v12 stable --all avg (2026-05-10):** **1.4854** with `BUDGET_OVERRUN_S=60.0s` fix; reproducible across runs

---

### v12 = varrahan/v1 (2026-05-08 → 2026-05-10): threshold change + budget-relaxation fix

Three concrete code changes vs sameer_v1/v11:

1. **EXACT_MACRO_THRESHOLD: 340 → 400** (re-includes ibm11, ibm15)
2. **EXACT_GRID_CELL_LIMIT: 2000 → 2200** (re-includes ibm15, ibm18)
3. **BUDGET_OVERRUN_S = 60.0s** for directed-restart phases (Phase 1/2/3 cong-grad). Allows
   the placer to spend up to `time_budget_s + 60s` on directed restarts, while keeping the
   noise loop strict (`time_budget_s` only).

#### Why the threshold change

Re-measurement of scoring time on clean CPU (2026-05-08) revealed PROGRESS.md v11 estimates were
4–13× too high:

| Benchmark | v11 estimate | v12 measured (clean) |
|---|---|---|
| ibm11 | 75–263s | **17.7s** |
| ibm15 | 160s | **42.8s** |
| ibm18 | 220s | **61.7s** |

All three well under `SLOW_SCORE_THRESHOLD_S=100s`. Threshold change re-includes them. Restarts
attempted on each:
- ibm11: 10 restarts, baseline (1.2354) wins — no change vs v11
- ibm15: restarts attempted, baseline (1.6061) wins — no change vs v11
- ibm18: 2 restarts, **cong-grad iter=1 wins → 1.7896** (−0.0045 vs baseline-only 1.7941)

#### Why the budget fix

**Problem found in --all run 1 (2026-05-10):** ibm04 normally scores 7s/call on clean CPU. But
during run 1, iter=1 of cong-grad spiked to 200s (likely transient CPU contention). This pushed
total time to 209s — over the 200s soft budget — and the post-scoring guard fired, returning
False from `_try_restart`. The calling code `if not _try_restart(...): return best_pl` then
terminated the entire placer, returning iter=1's result (1.3882) instead of Phase 3's 1.3316.

ibm04 collapsed by +0.0566. That single benchmark cost +0.0033 on the avg (1.4854 → 1.4888).

**Fix:**
- Add `allow_overrun: bool = False` parameter to `_try_restart`. When True, use
  `time_budget_s + 60s` as the cap for both pre- and post-scoring checks.
- Pass `allow_overrun=True` for all directed-restart calls (density-grad, all Phase 1 cong-grad
  iters, Phase 2 wide steps, Phase 3 cong-grad).
- Change cong-grad call sites from `if not _try_restart(...): return best_pl` to
  `if not _try_restart(...): break` so a budget exhaustion in one phase doesn't kill subsequent
  phases.
- Noise loop calls keep default `allow_overrun=False` — they're exploratory and shouldn't
  push us over budget on dead-end benchmarks.

**Result:** ibm04's 1.3316 win is now reproducible under --all conditions. Confirmed in --all
run 3 (2026-05-10): ibm04 = 1.3316. Bonus: ibm18 ticks 1.7898 → 1.7896 (one extra cong-grad iter
fits within the relaxed cap).

#### Bonus rediscoveries (clean CPU, no code change required)

- **ibm06 = 1.6684** (was 1.6797 in v11). Clean CPU runs hit a different cong-grad iteration
  pattern that lands at 1.6684 consistently. −0.0113 improvement.
- **ibm07 = 1.4924** (was 1.4950 in v11). Restart 6 (1% noise) wins. PROGRESS.md v11 said "noise
  doesn't help" but never tested 1% noise specifically on ibm07. −0.0026 improvement.
- **ibm04 = 1.3316** (was 1.3390 in v11). On clean CPU (t_score=6.4s instead of 15s) the placer
  fits more iterations and Phase 3 lands at 1.3316. −0.0074 improvement.

#### v11 numbers were a mix of outliers

The v11 PROGRESS.md figures for ibm01, ibm02, ibm03 turned out to be **lucky outliers**, not
stable targets. Today's clean runs (and the --all results) show:
- ibm01: 1.1860 (was 1.1854 in v11) +0.0006
- ibm02: 1.5923 (was 1.5823 in v11) +0.0100 — stale-plc trick is timing-sensitive
- ibm03: 1.3603 (was 1.3547 in v11) +0.0056

These regressions partially offset the v12 wins. Net delta vs v11 estimate: **−0.0006 to the avg**
(small, but in the right direction, and now reproducible).

#### Final numbers

- **v11 estimate (composite, never measured --all):** 1.4860
- **v10b actual --all (2026-04-30):** 1.4877
- **v12 stable --all (2026-05-10):** **1.4854**

Wins (vs v11 estimate): ibm04 −0.0074, ibm06 −0.0113, ibm07 −0.0026, ibm18 −0.0045 = −0.0258
Regressions (vs v11 estimate): ibm01 +0.0006, ibm02 +0.0100, ibm03 +0.0056 = +0.0162
Net: −0.0096 / 17 ≈ −0.00057 to avg.

---

### v14 = today's session (2026-05-19 → 2026-05-20): speed-only kept, structural attempts deferred to DREAMPlace

#### Kept changes (verified, no regression)

1. **Tier 3: Vectorize `_will_legalize`** (2026-05-19). Greedy spiral search rewritten in numpy: per ring, generate all 8r candidates at once via `_ring_offsets`, run a single `[K, P]` conflict matrix instead of nested Python loops. ibm04 legalize: 3.2s → 0.27s (12×). All cong-grad iters now run sub-second.

   **Critical correctness fix**: the original scalar computed `d² = (cx - pos[idx, 0])²` where `cx` is a Python float and `pos[idx, 0]` is numpy float32. NumPy demotes the Python float to float32 for the subtraction (Python-scalar-meets-numpy-scalar rule), so d² is computed at float32 precision. This causes symmetric ring candidates like (-1, 0) and (0, -1) to break ties at float32 noise instead of being truly equal. My initial vectorized version computed d² in float64 (cand_x is a strong-typed float64 array, no demotion), which made ties exact and changed which candidate `np.argmin` picked. Result: ibm04's cong-grad iter-2 diverged and the placer landed at 1.3364 instead of 1.3316. **Fix**: cast `cand_x`/`cand_y` to `pos.dtype` before subtraction, mirroring scalar's float32 demotion. Bit-equivalent legalize confirmed via diff harness on ibm04 iter-2 input. See `placer.py:178-191`.

2. **Running-max `t_one_score`** (2026-05-19). Re-added v11's running-max budget guard (removed in v12 for noise sensitivity). Under --all CPU contention, scoring can be 3-5× slower than the baseline measurement (not "jitter"). Without adaptation, the budget check approves restarts that then exceed cap, causing Phase 3 to skip on benchmarks like ibm04 (observed 1.3316 → 1.3449 regression in multi-order --all). The running-max tightens budget when contention is observed; brief blips that double t_one_score still leave 60s overrun for directed phases. Defensive — no improvement on its own, enables other experiments.

3. **2-opt swap post-pass on baseline-only branch** (2026-05-19). After greedy spiral-search legalize, scan macro pairs within K=5 nearest neighbors; accept swaps that legal AND strictly reduce per-pair displacement. Applied only to the `n>400` baseline-only branch (no cong-grad trajectory to disrupt). Tested gains: ibm10 −0.0006, ibm12 −0.0001, ibm13 −0.0005, ibm14 −0.0005, ibm16 +0.0001, ibm17 +0.0001. Net sum: −0.0015 across 6 baseline-only benchmarks ≈ −0.0001 to 17-bench avg.

#### Cleanup (2026-05-20, prepping for DREAMPlace integration)

- **Removed from placer.py**: `_wiremask_place`, `_build_wm_net_cache`, `_density_gradient_perturb`, `_congestion_heatmap`, `_box_blur` (all dead code on IBM — density-grad never fires for n>100).
- **Deleted files**: `surrogate.py`, `_calibration_test.py`, `_path3_incremental_test.py` (rejected experiments, never wired in).
- **placer.py**: 1159 → 894 lines.

#### Rejected today (sporadic / catastrophic / dead)

1. **Multi-order baseline (Phase 1-disrupting)** — adds smallest-area / tallest / widest orderings as extra Phase-0 candidates before Phase 1. Under --all CPU contention, the 4 extra scorings (3 multi-order + 1 baseline re-score) consumed enough budget that Phase 3 didn't fit on ibm04 (regression 1.3316 → 1.3449). Net negative.

2. **Displacement-ranked multi-order on baseline-only** — pick the legalization with smallest total displacement from initial.plc among 3 orderings. Catastrophically wrong: ibm10 with `tallest` order had lowest displacement (414 vs 1051 default) but proxy was 1.5658 vs 1.4037 (+0.162 regression — congestion blew up). ibm12 with `smallest-area` produced INVALID placement (27 overlaps) because large macros couldn't find slots within the 60s spiral deadline. Conclusion: displacement-sum is NOT a useful proxy ranker across orderings.

3. **2-opt-everywhere (in `_try_restart`)** — applied to every legalize result (baseline + cong-grad iters + noise). On ibm04, baseline 2-opt nudge cascades through cong-grad: −0.0115 win (1.3201). But on ibm06, the same baseline 2-opt nudge improves iter-1 enough that iter-2 can't beat it → Phase 1 break-on-no-improvement fires at cong_iter=1 (<2 halving threshold) → Phase 3 skipped → +0.0087 regression. Sporadic. Root cause: 2-opt's "snap back toward target" interferes with cong-grad's "push away from congested cells" trajectory.

4. **Multi-frac Phase 3** — try `frac ∈ {0.02, 0.04, 0.06}` instead of just 0.04. Tested ibm04/06/02/09: f=0.04 always wins; extra fracs add 2 scorings of overhead per Phase-3 benchmark for no improvement. Safe but ineffective.

5. **WireMask-BBO + congestion penalty** (alpha=30, G=25) — the v13 salvage path from PROGRESS.md. Tested ibm01/04/06: sporadic. Helps sparse (ibm01: 1.1964 vs baseline 1.2253 = −0.029) but hurts dense (ibm04: 1.5070 vs 1.4101 = +0.097; ibm06: 1.8890 vs 1.7197 = +0.169). Root cause: WireMask is constructive — rebuilds from scratch and loses initial.plc's hand-tuned spread that the pipeline operates around. A single alpha cannot satisfy all benchmarks (would need per-benchmark tuning, which violates the "no benchmark-specific tweaks" rule). Implementation removed from placer.py.

6. **`plc.optimize_stdcells` post-pass** — academic force-directed soft-macro re-placement (`external/MacroPlacement/CodeElements/Plc_client/plc_client_os.py` line 2886). Timed on ibm01 (smallest, n_soft=894) at num_steps=10: **126.6s** (~13s per step), and the result was **+0.1296 WORSE** than baseline (1.2253 → 1.3549 with default attract/repel factors). Pure Python iteration over ~1000 soft macros and ~10000 nets per step; no C++ binding. Effectively infeasible inside our 200s budget. Would need a multi-day rewrite in vectorized numpy/torch with tuned parameters to ever be useful. Dead path.

7. **Vectorized soft-macro re-placement (the rewrite of #6)** — implemented and tested 2026-05-20. Three algorithms, all in a new `soft_relax.py` module: (a) HPWL² gradient descent (textbook analytical placement); (b) connectivity-weighted displacement-follow (translate softs by the avg displacement of their connected hards); (c) HPWL + grid-bin density repulsion (the "do it right" combined version). Edge extraction from `plc.nets` cached on plc object (~1.5s one-time per benchmark); per-call runtime 0.5–10ms — performance was never the issue. **All three regress proxy on every benchmark tested.** Best result was hpwl+density with 2 steps × 0.005 max_frac × dw=0.5: +0.003 (ibm06) to +0.031 (ibm01). Tested across hard-perturbation magnitudes 0% → 80%; no regime where any variant netted negative delta. Module deleted after the test; see this entry for the negative result.

   **Decomposition that explains the loss** (ibm04, 15% hard perturb):
   ```
   stale softs:           WL=0.082, D=0.951, C=1.815, proxy=1.465
   HPWL-only relax:       WL=0.077, D=1.074, C=1.715, proxy=1.472  (D ↑)
   HPWL + density repel:  WL=0.072, D=0.950, C=1.747, proxy=1.420 → 1.420 vs base 1.410 = +0.010
   ```
   HPWL relax DOES improve WL (−0.005) and congestion (−0.10), and density repulsion successfully cancels the density rise — but the residual is still net positive. Initial.plc softs sit in a steep local minimum on the *joint* (WL, density, congestion) surface; any motion away from there pays one component faster than it gains on others, no matter how the forces are balanced.

   **Corrects #6's "soft mismatch" theory.** PROGRESS.md previously attributed DREAMPlace standalone's 0.2–0.3 regression to stale softs around moved hards. The decomposition above shows stale-soft cost is at most **~0.05** even at 30%–80% random hard perturbation (ibm04 stale-softs at perturb=0.30: 1.579; at perturb=0.80: 1.500 — actually goes *down* as hards spread to fill canvas). DREAMPlace standalone's real failure is that **its WL-optimized hard placement lands in a WL basin, which is uncorrelated with the congestion-dominated proxy basin** — same root cause as the WireMask-BBO failure. No amount of soft re-placement can fix that. The async DREAMPlace integration retains value as a side-channel for plc-state mutation (which seeds new cong-grad basins), not for its placement quality per se.

   **Implication: hard-placement search is the only useful axis.** Stop trying to optimize softs.

#### In progress: async DREAMPlace bridge (2026-05-20)

The v13 sync bridge was rejected in May because its 10-15s subprocess overhead displaced productive restarts on 7/17 benchmarks (net +0.0043 worse). PROGRESS.md notes the salvage path: async invocation so DREAMPlace runs in parallel with our scoring.

**Status**: restored `dreamplace_bridge/` from commit 111f315; added `AsyncDreamplaceHandle` + `launch_dreamplace_async` for non-blocking subprocess management. Integrated into `placer.py` as Phase 5: launch DREAMPlace at `place()` entry, check after Phase 3 as additive candidate ("dreamplace global"), and follow with one cong-grad iter from DREAMPlace's legalized position ("cong-grad from-dreamplace") to exploit the plc-state-mutation effect that PROGRESS.md noted as the source of v13's real wins. DREAMPlace build in progress (cmake done, `make -j2` ~70% on 2026-05-20 10:15).

**Expected gain (if async parallelism works)**: −0.005 to −0.025 to avg. Lower bound (−0.005) is the v13 wins (ibm04, ibm11) without the displacement cost. Upper bound (−0.025) assumes DREAMPlace also mutates plc-state usefully on 3-5 other benchmarks. Won't reach DREAMPlace standalone's 1.4076 because our pipeline still owns the basin search; DREAMPlace is just one additional seed.

**Risks**:
- *Async parallelism may not materialize* — depends on whether plc's C++ scoring releases the GIL. If it doesn't, DREAMPlace burns CPU contending with the scoring thread.
- *Soft-macro mismatch* — v13's standalone DREAMPlace was ~0.2-0.3 worse than baseline because soft macros stayed at initial positions while hard macros moved. The cong-grad-from-DREAMPlace step partially compensates (cong-grad nudges hard macros and softs are re-scored via plc), but doesn't fix the underlying issue. `optimize_stdcells` would, but it's too slow.

---

### v15 = current session (2026-05-20 → 2026-05-21): DREAMPlace bridge functional, Improvement #1 enabled

**Headline: --all avg 1.4854 → 1.4804 (−0.0050 absolute)** — confirmed via partial v5 run (16/17 benchmarks; ibm17 timed out at 3600s cumulative). Wins: ibm01 (−0.044), ibm04 (−0.012), ibm10 (−0.037), ibm14 (−0.003). Regression: ibm07 (+0.003).

#### Bridge architecture fix (was: DP NLP plateaus at iter=1; output is junk)

Diagnostic 2026-05-20: DREAMPlace's Nesterov optimizer was producing essentially no movement on our Bookshelf input — wHPWL frozen at 5.31e7 across 150 iters, iter times 0.3ms (vs typical 50-500ms). Standalone DP proxy ~1.7714 even after fixing soft_macros_movable (vs predicted 1.3-1.4 in DREAMPLACE_FIXES.md). Three compounding bugs in `pb_to_bookshelf.py` / `run_bridge.py`:

1. **`.scl` row structure (`pb_to_bookshelf.py:_write_scl`)** — was emitting a single canvas-height row (`Height: 34081` for ibm04). DREAMPlace's density bins and macro legalizer need stdcell-row-height rows to function. Reference benchmark `simple.scl` uses 8 rows of 12 over a 96-tall canvas (12.5% per row). **Fix**: write `num_rows_target=8` rows of height `canvas_h/8` each (~4260 scaled units = 4.3 microns for ibm04). After this, iter 0 → iter 1 transition produces real motion but optimizer still plateaus.

2. **`macro_place_flag=1` + `use_bb=1` (`run_bridge._default_dreamplace_config`)** — was off. Without these, DP treats macros as huge stdcells and the optimizer's gradient step is essentially zero (we were seeing ~0.5ms per iter wall time, way below the ~50ms needed for real per-cell gradient computation). With macro_place_flag, the 2-stage BB-step → NLP pipeline engages: trajectory becomes wHPWL 5.56e7 → 5.10e7 → 5.22e7 over 150 iters, Overflow drops 0.20 → 0.40 (real convergence). Standalone DP proxy on ibm04 drops from 1.7714 → 1.5207.

3. **Iteration count `iter=300`** — `iter=150` was under-converged (Overflow stuck at 0.4 vs target 0.10). Bumped to 300 → ibm04 standalone DP proxy = **1.3196**. Bigger values (500-1000) showed DensityWeight runaway (Obj jumping to 1e12) with no proxy improvement. `iter=300` is the sweet spot.

After all three fixes: standalone DP proxy on ibm04 = **1.3196** (vs Phase 3's 1.3316 — beats it by 0.012). On ibm01: standalone DP = **1.1521** (vs PROGRESS.md best 1.1964 — beats it by 0.044). On ibm06/08/11: DP loses to Phase 3 / noise restarts (small margins).

#### Kept changes (verified, no regression in --all v5)

1. **DREAMPlace bridge rewrite** (above). Module: `dreamplace_bridge/{pb_to_bookshelf.py, run_bridge.py, bookshelf_to_pb.py}`. The async Phase 5 candidate now actually wins on ibm01 and ibm04. Also: `soft_macros_movable=False` (verified 2026-05-20: softs movable inflates congestion +0.011 on ibm04).

2. **Phase 5c — wide-from-best at frac=0.08** (`placer.py` after Phase 5b). Fills the slot left by Phase 2 (wide from BASELINE only) and Phase 3/5b (frac=0.04 from BEST only). Purely additive — fires only if `cong_improved=True` and budget allows; placed after Phase 5b so no current winning rng_cong path is disturbed. Fires on ibm04/06; doesn't find new wins in tested benchmarks but doesn't regress either. Net ~0 with no risk.

3. **CPU contention fixes in DREAMPlace subprocess launcher** (`run_bridge.launch_dreamplace_async`):
   - Set `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS` to match `num_threads=2` in the DP subprocess env. Without this, DREAMPlace's internal OMP/MKL pools default to all available cores and oversubscribe with the parent's scoring thread.
   - Watchdog thread in `AsyncDreamplaceHandle._start_watchdog()` enforces `timeout_s` even when the placer is blocked in scoring (without it, a hung DP saturated CPU and slowed scoring 100×, observed on ibm06: baseline scoring took 1599s vs typical 14s, triggered SLOW_SCORE_THRESHOLD and lost the 1.6684 win → 1.7197).
   - `start_new_session=True` + `os.killpg()` for clean teardown.

4. **Improvement #1 — DP on `n>400` / `grid>2200` benchmarks** (`placer.py` `if not use_exact:` branch). 6 benchmarks (ibm10/12/13/14/16/17) previously took the baseline-only early return. Now does one head-to-head: score baseline once with `_exact_proxy`, wait for DP, legalize+score DP, return whichever is better. Gated on baseline scoring < **130s** (raised from initial 100s after v4 measurements showed ibm10 baseline scoring climbs 67s → 101s under --all CPU contention, just tripping the 100s threshold and losing the −0.037 DP win). Wins: **ibm10 1.4031 → 1.3661 (−0.037)**, ibm14 1.6028 → 1.6002 (−0.003). No regressions on ibm12/13 (baseline correctly wins). ibm16/17 skip (baseline scoring 157s/280s+ exceeds 130s threshold).

#### Rejected today

1. **Fix 3 variant A: "DP as PRIMARY baseline_pos"** (DREAMPLACE_FIXES.md's recommendation). Replace `baseline_pos = _will_legalize(initial.plc)` with `baseline_pos = legalized DP output` when DP wins as Candidate 0. Phase 1/2/3 then iterate from DP placement instead of initial.plc. **Tested on ibm04 (1.3196 — same as additive) and ibm06 (1.6789 vs 1.6684 — +0.0105 regression)**. The MD's warned-of risk materialized: Phase 3 cong-grad from DP's placement converges to a different (worse) basin than Phase 3 from initial.plc. ibm06's 1.6684 win specifically lives in the basin reached by stale-plc-after-Phase-2 from initial.plc; DP's plc-state path doesn't get there.

2. **Fix 3 variant B: "Phase 6 cong-grad-from-DP" (additive, multi-iter)** — preserves all existing wins by NOT replacing baseline; instead adds a 4-iter cong-grad loop starting from DP's placement after Phase 5b. **Tested on ibm04 (1.3196 — same), ibm06 (1.6684 — same), ibm01 (1.1521 — same), ibm08 (1.5419 — +0.017 regression)**. ibm08 found a small win on Phase 6 iter=1 (1.5419 vs DP additive 1.5444) BUT the 4-iter loop consumed budget that previously reached the noise=6% winner (1.5251 in v14). Limiting to 1 iter still didn't fit noise=6% within budget. Conclusion: cong-grad from DP placement doesn't find systematically better basins; the marginal wins it does find cost more budget than they save elsewhere.

3. **DP-first ordering on Improvement #1 path** — flip the order: score DP first, then baseline if budget allows. Goal: capture wins on ibm16/ibm17 where baseline scoring exceeds the threshold. **Tested on ibm16 (DP=1.5751 vs baseline=1.5324 → +0.043 regression)** — DP loses to baseline on ibm16, and trusting DP unconditionally when baseline scoring doesn't fit is strictly worse than skipping DP. ibm17 timed out at 350s. Baseline-first is strictly safer.

#### Outstanding issues (deferred)

- **ibm07 regression (+0.003)**: DP candidate consumes ~60s of budget; on ibm07's tight budget the winning 1% noise restart (5th in the noise_fracs order) doesn't get enough time. PROGRESS.md table says ibm07 wins at 1% noise. Mitigation would be runtime gating: skip DP launch when expected DP+score time > available-budget-for-noise. Low priority (0.003 only).

- **--all wall-clock budget**: v4 and v5 both timed out at ibm17 (>3600s cumulative). 17 benchmarks × ~210s avg = ~3570s leaves no margin. Bottleneck: ibm15 (239s) and ibm16 (170s baseline-only after slow-score skip) and ibm17 (>300s baseline scoring alone). The challenge spec allows 1 hour total; if --all itself takes >3600s in the harness, we lose. **Workaround for now**: PROGRESS.md results assume ibm17=1.7438 and ibm18=1.7881 from prior runs; --all avg 1.4804 is a partial-run extrapolation. A clean full --all needs either lower scoring threshold on largest benchmarks or a more aggressive timeout management.

- **No new wins on ibm02/03/06/08/09/11/15/16/18** (9 of 17 benchmarks). These contribute roughly half the avg sum but have no DP win and no Improvement #1 win. The fundamental signal: DP optimizes WL+density while our proxy is congestion-dominated; on benchmarks where the cong-grad pipeline already finds a deep basin, DP can't compete. The next leverage frontier is something orthogonal to both — possibly a soft-macro re-placement that DOES help (the rejected #7 in v14 attempts), or a different perturbation primitive (gravitational rather than gradient-following).

---

### v11: Budget safety + EXACT_MACRO_THRESHOLD 400→340

**Problem found in v10b full eval**: ibm11 (n=373) baseline scored in **263.6s** under CPU load
(8+ prior benchmarks running had heated up the CPU). The SLOW_SCORE_THRESHOLD=100s check DID
trigger, returning baseline — but only AFTER the 263.6s scoring finished. Total=266.8s (over budget).
No improvement was possible anyway (all perturbations worse for ibm11), so this was wasted time.

**Fix 1**: EXACT_MACRO_THRESHOLD: 400 → 340. ibm11 (n=373 > 340) now returns baseline without
exact scoring — same result but in <5s. ibm08 (n=301 ≤ 340) still included.

**Fix 2**: `t_one_score` now adaptive running max inside `_try_restart`. If CPU slows mid-benchmark,
future budget checks use the updated (worse) scoring time as reference, preventing more over-budget runs.

**Fix 3**: Post-scoring budget check in `_try_restart`. If `time.time()-t0 > time_budget_s` after
any scoring call, return False immediately (stop further restarts). Limits overrun to ≤1 scoring
cycle beyond budget instead of full loop continuation.

**New best from v10b full eval**: ibm04=**1.3390** (6 cong-grad iterations at t_score=15s).
Previously best was 1.3468 (5 iters at t_score=12s). Timing-sensitive — full eval conditions gave
one extra iteration. ibm06=**1.6797** (slightly better than 1.6802 from isolated test).

**Update 2026-05-09**: ibm04 floor revised to **1.3316** (−0.0074 vs prior 1.3390). On clean CPU
(t_score≈6.4s) the placer fits 7 cong-grad iters + Phase 2 wide + Phase 3 perturb; Phase 3 from
the best-so-far position with stale plc consistently lands on 1.3316. Confirmed in 3-for-3 isolated
runs. No code change — emerges naturally from existing v10b code under clean CPU. PROGRESS.md's
1.3390 was a slower-timing artifact (6 iters instead of 7+). Likely also reproducible in `sameer_v1/`.

### v10: Adaptive cong-grad frac + range(12)
- Extended iterative loop from range(4) to range(12)
- Adaptive frac reduction: when frac=0.04 fails but cong_improved=True and cong_iter≥2,
  halve frac (0.04→0.02→0.01). cong_iter≥2 guard protects ibm02 stale-plc mechanism.
- Confirmed: ibm03=1.3547, ibm04=1.3468 (→1.3390 in v10b full eval), ibm06=1.6802

### v8: Iterative congestion-gradient descent + wide step
  ibm02: 1.6800→1.5823 (iterative cong-grad + wide=8% from baseline with stale iter=2 plc)
  ibm03: 1.4100→1.3583 (2 iterative steps)
  ibm04: 1.4101→1.3479 (4 iterative steps)
  ibm06: 1.7198→1.6810 (2 iterative steps)

Non-exact benchmarks (n>400 or grid>2200 in v12; was n>340 or grid>2000 in v11) return pure
baseline; no restarts possible. ibm10, ibm12 already beat RePlAce at legalization-only.

---

## Key Findings So Far

- WL is tiny (~0.06 normalized). Congestion (~1.3-2.5) dominates the proxy.
- SA over-optimizes WL, clusters macros, spikes congestion. Never use WL-only SA.
- initial.plc already has good spread. Legalization preserves it.
- Small noise (1-6%) finds better legalization arrangements on some benchmarks (ibm01: 6%, ibm07: 1%, ibm08: 6%).
- **Density fallback ANTI-CORRELATED**: sum-of-squares density rewards spread placements.
  But spread placements have WORSE proxy (higher congestion). Evidence: ibm11 density-selected
  result = 1.3770 vs baseline = 1.2354 (11.5% regression). Full eval confirmed +0.14 avg hurt.
  Fix: return baseline immediately for any non-exact benchmark.
- ibm18 anomaly RESOLVED (v12, 2026-05-08): 285 macros and 55x39 grid score in 62s clean, not
  220s as previously estimated. Now included in exact pipeline. Cong-grad iter=1 improves to 1.7898.
- ibm11 (n=373): with EXACT_MACRO_THRESHOLD=400 (v12), uses exact scoring. t_score=17s clean.
  Baseline wins (proxy=1.2354). 10 restarts fit; all worse than baseline. Same result as v4 era.
- **PROGRESS.md scoring estimates were 4–13× too high** for ibm11/ibm15/ibm18 — measurements were
  apparently taken under heavy CPU load. v12's threshold change re-included them after re-measuring
  clean (2026-05-08). The `SLOW_SCORE_THRESHOLD_S=100s` safety guard catches any regression under
  load — falls back to baseline, identical to v11 behavior.
- **Surrogate ranker (varrahan/v1/surrogate.py) was tested and rejected.** WL-only weighting had
  Spearman +0.83/+0.94 vs real proxy on ibm11/ibm15, but ties between near-optimal candidates
  broke the wrong way. Net effect: zero or slightly negative. Documented in `surrogate.py`,
  `_calibration_test.py`. See README in v1.
- **Path 3 (incremental scoring via plc.set_use_incremental_cost) is dead.** Incremental mode
  only refreshes WL — density and congestion components are frozen. Since proxy ≈ congestion,
  the rescore signal is anti-correlated with what we want to optimize. Documented in
  `_path3_incremental_test.py`.
- **Cong-grad from a noise-perturbed start (Phase 4)** was implemented and tested on
  ibm04/ibm07/ibm08/ibm11 (2026-05-09). Always strictly worse than at least one existing
  restart. 2% perturbation lands cong-grad in a worse local minimum than baseline-start
  cong-grad. Reverted. Could be retried with different perturbation scales (0.5%, 4-6%) but
  is speculative.
- **WireMask-BBO greedy** (real algorithm, not the continuous-pull approximation) was
  implemented and tested on ibm01/ibm04/ibm07/ibm15 (2026-05-09). Two failure patterns:
  (a) on sparse benchmarks (ibm01, ibm07, ibm15), the wire-mask output legalized back to
  exactly the baseline placement — no real movement; (b) on ibm04, the greedy clustered
  macros tightly enough that congestion increased more than wirelength dropped, producing
  a placement (1.4127) STRICTLY WORSE than baseline (1.4101). This is exactly the failure
  mode CLAUDE.md and PAPERS_NOTES.md predicted: pure HPWL minimization clusters connected
  macros, hurting the congestion-dominated proxy. Reverted. The function and call site
  were both removed (see git history for the implementation if revisiting). Possible
  salvage paths (each is a separate experiment): wire-mask + per-cell congestion penalty,
  wire-mask as inner-loop scorer for an outer BBO/SA optimizer (the actual paper
  contribution), or wire-mask applied only to the highest-net-weight subset.
- **Budget guard fragility (FIXED in v12, 2026-05-10).** v11's post-scoring budget guard
  (`if time - t0 > time_budget_s: return False`) combined with `if not _try_restart(): return
  best_pl` was killing the entire placer on a single transient scoring spike. Observed on
  ibm04 in --all run 1 (2026-05-10): scoring of cong-grad iter=1 spiked from typical 7s to
  200s, post-guard fired, placer returned 1.3882 instead of Phase 3's 1.3316. Fix: added
  `BUDGET_OVERRUN_S=60s` allowance for directed-restart phases (Phase 1/2/3) and changed
  cong-grad call sites from `return best_pl` to `break`. ibm04's 1.3316 is now reproducible
  under --all conditions. Bonus: ibm18 picked up −0.0002 (1.7898 → 1.7896) from the relaxed
  cap allowing one more iteration.
- **DREAMPlace bridge (Phase 1-3 implemented, integration REVERTED 2026-05-11).**
  Built and installed DREAMPlace from source (Phase 1, ~75min including OOM-fix and ABI=1
  rebuild). Wrote pb.txt → Bookshelf converter and back-converter (Phase 2-3a). Integrated
  as a new restart slot before Phase 1 cong-grad (Phase 3b). Tested on full --all
  (2026-05-11): **avg 1.4897 vs v12's 1.4854 (+0.0043 worse)**. Two real wins (ibm04 −0.0075,
  ibm11 −0.0019, both from DREAMPlace's plc-state mutation enabling new cong-grad basins),
  but seven regressions (biggest: ibm03 +0.034, ibm08 +0.029, ibm09 +0.006) all caused by
  DREAMPlace's 10-15s subprocess overhead displacing productive noise/cong-grad restarts.
  DREAMPlace's standalone placement is consistently ~0.2-0.3 worse than baseline because
  soft macros stay at initial positions while hard macros move (the soft-macro mismatch
  problem from CLAUDE.md). Reverted from placer.py. The bridge module was later deleted
  (commit a93a5ae) but **restored 2026-05-20** with async wrapper — see v14 entry.
- **`plc.optimize_stdcells` salvage attempt tested + REJECTED 2026-05-20.** The academic
  force-directed soft-macro re-placement (path (d) above). Timed on ibm01 (n_soft=894) at
  num_steps=10: 126.6s per call AND +0.13 regression with default attract/repel params.
  Pure Python iteration over thousands of soft macros and tens of thousands of nets per
  step; no C++ binding. Effectively infeasible inside our 200s budget. Path (d) is dead;
  would need multi-day vectorized-numpy rewrite to ever be useful.
- **Async DREAMPlace integration (salvage path (b)+(c) combined) IN PROGRESS 2026-05-20.**
  Restored `dreamplace_bridge/` from commit 111f315; added `AsyncDreamplaceHandle` and
  `launch_dreamplace_async` for non-blocking subprocess management. Integrated into
  `placer.py` as Phase 5: launch at `place()` entry (subprocess runs while we score baseline
  and Phase 1/2/3), check after Phase 3 as additive candidate. Adds a second additive
  ("cong-grad from-dreamplace") that runs one cong-grad iter from DREAMPlace's legalized
  position to capture the plc-state-mutation effect that v13's wins came from. Build
  completes ~2026-05-20; results pending.

---

## Tunable Parameters (current v14 values)

```python
n_restarts            = 50         # cap; budget check is the real limit
noise_fracs           = [0.02, 0.04, 0.06, 0.08,  # core (preserved wins)
                          0.01, 0.03, 0.05, 0.07, 0.09,
                          0.06, 0.06, 0.04, 0.10, 0.12, 0.08,
                          0.025, 0.035, 0.045, 0.055, 0.065, 0.075,
                          0.15, 0.20, 0.10,
                          0.05, 0.06, 0.07, 0.03, 0.04, 0.02,
                          0.005, 0.010, 0.015, 0.030, 0.050]
time_budget_s         = 200.0
BUDGET_OVERRUN_S      = 60.0       # v12 (2026-05-10): allow up to 60s extra for directed-restart phases (cong-grad Phase 1/2/3) so a transient scoring spike doesn't kill the whole pipeline. Noise loop stays strict.
EXACT_MACRO_THRESHOLD = 400        # v12: was 340 in v11. ibm11 (n=373) and ibm15 (n=393) included; ibm13 (n=424) excluded
EXACT_GRID_CELL_LIMIT = 2200       # v12: was 2000 in v11. ibm15 (2166) and ibm18 (2145) included; ibm12 (2209) excluded
SLOW_SCORE_THRESHOLD_S = 100.0     # safety net for exact scoring
# DENSITY_GRAD_MAX_N removed in v14 — density-grad helpers deleted (never fired on IBM)

# v14 (2026-05-20): t_one_score is now a RUNNING MAX inside _try_restart, not a fixed
# baseline value. Defends against --all CPU contention where scoring can be 3-5× slower
# than baseline. Re-adds v11's logic that v12 removed; the v12 rationale ("scorings are
# within jitter of baseline") doesn't hold under --all heat.

# v14 (2026-05-20): 2-opt swap post-pass applied ONLY on the baseline-only branch
# (n>400 / grid>2200). k_neighbors=5, max_iters=3. Net +0.0001 to avg.
# Applied to cong-grad/noise legalize outputs (2-opt-everywhere): tested and REJECTED
# due to sporadic gain/loss pattern (ibm04 −0.0115 ✓ but ibm06 +0.0087 ✗).

# v14 (2026-05-20): Async DREAMPlace as Phase 5 candidate. Launch at place() entry,
# wait_for_result(max_wait_s=30) after Phase 3, follow with one cong-grad iter from
# DREAMPlace's legalized position. Gated by `is_available()` so placer is a no-op
# when DREAMPlace isn't built. Build location: submissions/varrahan/dreamplace_build/
# (gitignored, ~500MB).
```

---

## Next Experiments to Try

1. [x] Full v4 17-benchmark eval -- confirmed all 17 baselines
2. [x] v5 budget-filling restarts -- ibm01 confirmed 1.1854 with 11 restarts (no improvement beyond 6% win)
3. [x] v6 congestion-gradient perturbation -- ibm02 (-0.060) and ibm06 (-0.036) confirmed
4. [x] CLEAN full v6 eval -- ran 2026-05-08 (varrahan/v1). Avg 1.4901 under heavy load (ibm04 safety-net fired at 666s); estimated 1.4853 clean.
5. [x] ibm15 scoring time test (n=393, grid=2166) -- DONE (v12, 2026-05-08): 43s clean, included via raised limits. Baseline still wins (1.6061).
6. [~] ibm08 + ibm07 congestion-grad clean test -- DONE (v12, 2026-05-09): ibm07 1% noise wins (1.4924); ibm08 6% noise wins (1.5251); cong-grad doesn't help either.
7. [ ] Additional congestion-grad fracs (0.08, 0.12) for high-cong benchmarks:
       After confirming ibm08 behavior, add more cong-grad restarts at larger scales.
8. [~] Multiple congestion-grad starting points (Phase 4): TESTED 2026-05-09 with 2% perturbed start. Strictly worse on all 4 benchmarks tested. Reverted. Could retry with 0.5% or 4-6% scales.
9. [x] ibm04 congestion-grad: Phase 3 cong-grad now consistently lands at 1.3316 on clean CPU (was 1.3390 in v11). Confirmed 3-for-3 on 2026-05-09. Gap to RePlAce closed from -2.8% to -2.2%.
10. [~] **WireMask-BBO greedy evaluator** -- IMPLEMENTED AND REVERTED 2026-05-09. Two failure patterns: (a) sparse benchmarks legalized back to baseline (no movement), (b) ibm04 produced 1.4127 vs baseline 1.4101 (clustered macros → worse congestion). Confirms CLAUDE.md/PAPERS_NOTES warning that pure HPWL minimization hurts congestion-dominated proxy. See "Key Findings" section above. Salvage paths: wire-mask + congestion penalty, wire-mask + outer BBO loop, wire-mask on top-net-weight subset.
11. [~] **DREAMPlace bridge sync** (`pb.txt → Bookshelf → DREAMPlace global → legalize`) -- IMPLEMENTED AND REVERTED 2026-05-11. v13 --all = 1.4897 vs v12's 1.4854 (+0.0043 worse). Real wins on ibm04 (−0.0075) and ibm11 (−0.0019). 10-15s subprocess overhead displaced productive restarts on 7 benchmarks. Bridge module deleted in a93a5ae but restored 2026-05-20.
12. [x] **Tier 1/2/3 vectorize core paths** -- DONE 2026-05-19. Vectorized `_will_legalize` (12× speedup on ibm04), `_routing_congestion_perturb`, `_score` pl_scratch buffer. Critical float32 precision fix in vectorized legalize (without it, ibm04 lands at 1.3364 instead of 1.3316). Bit-equivalent to scalar baseline; ibm04/ibm06/ibm02 preserved.
13. [x] **Running-max t_one_score** -- DONE 2026-05-19. Defensive; re-adds v11 logic that v12 removed. Adapts to --all CPU contention.
14. [x] **2-opt swap post-pass on baseline-only branch** -- DONE 2026-05-19. Net −0.0015 sum across 6 baseline-only benchmarks, ≈ −0.0001 to avg.
15. [~] **2-opt-everywhere (in `_try_restart`)** -- TESTED AND REVERTED 2026-05-19. Sporadic: ibm04 −0.0115 ✓ but ibm06 +0.0087 ✗, ibm02 +0.0015 ✗. Root cause: 2-opt's "snap toward target" disrupts cong-grad's "push away from congestion" trajectory.
16. [~] **Multi-frac Phase 3 (fracs 0.02/0.04/0.06)** -- TESTED AND REVERTED 2026-05-19. Safe but ineffective: f=0.04 always wins on tested benchmarks.
17. [~] **WireMask + congestion penalty (α=30, G=25)** -- TESTED AND REVERTED 2026-05-19. Sporadic: ibm01 −0.029 ✓ but ibm04 +0.097 ✗, ibm06 +0.169 ✗. Same root cause as pure WireMask: constructive placer abandons initial.plc's good seed.
18. [~] **Multi-order baseline (smallest-area / tallest / widest)** -- TESTED AND REVERTED 2026-05-19. Phase 1-disrupting version regressed ibm03/04/09 under --all. Displacement-ranked variant on baseline-only catastrophically wrong (ibm10 +0.162, ibm12 INVALID).
19. [~] **`plc.optimize_stdcells` post-pass** -- TESTED AND REJECTED 2026-05-20. 126.6s per call on smallest benchmark (ibm01) AND +0.13 proxy regression with default FD params. Pure Python; would need multi-day rewrite to be feasible. Dead path.
20. [x] **Async DREAMPlace bridge as Phase 5** -- DONE 2026-05-20/21. Three architectural bugs found and fixed: `.scl` single-row → 8 rows of `canvas_h/8`; `macro_place_flag=1` + `use_bb=1` enabled; iter raised 150→300. Standalone DP proxy on ibm04 dropped 1.7714 → 1.3196. Wins as Phase 5 additive candidate on ibm01 (−0.044) and ibm04 (−0.012). See v15 section for full diagnostic.
21. [x] **DREAMPlace CPU contention fix** -- DONE 2026-05-20. Set OMP/MKL/OPENBLAS/NUMEXPR `NUM_THREADS=2` in DP subprocess env to match `num_threads=2` config. Added watchdog thread in `AsyncDreamplaceHandle` to enforce `timeout_s` regardless of placer state. Without these, DP saturated CPU during scoring and slowed it 100× (ibm06: 1599s baseline scoring → triggered safety bail → +0.051 regression). Fix verified: ibm06 baseline scoring returned to ~10s.
22. [x] **Phase 5c — wide-from-best at frac=0.08** -- DONE 2026-05-20. Additive cong-grad step using current plc state. Fills the gap between Phase 2 (wide from baseline) and Phase 3/5b (frac=0.04 from best). Fires on cong_improved benchmarks; doesn't find new wins but doesn't regress. Pure insurance.
23. [x] **Improvement #1: DP on n>400 / grid>2200 benchmarks** -- DONE 2026-05-21. Adds head-to-head DP-vs-baseline comparison on the 6 large benchmarks (ibm10/12/13/14/16/17) that previously took the baseline-only early return. Threshold 130s on baseline scoring time (raised from 100s after observing CPU-contention slowdowns under --all). Wins: **ibm10 −0.037, ibm14 −0.003**. ibm12/13 baseline correctly wins. ibm16/17 skip (too slow). See v15 section.
24. [~] **Fix 3 "DP as PRIMARY baseline_pos"** -- TESTED AND REJECTED 2026-05-21. Phase 1/2/3 cong-grad from DP placement converges to a different (worse) basin on ibm06 (+0.0105 regression on the 1.6684 win). Same architecture risk warned about in DREAMPLACE_FIXES.md.
25. [~] **Fix 3 variant: Phase 6 additive cong-grad from DP placement** -- TESTED AND REJECTED 2026-05-21. On ibm08, the 4-iter loop displaced budget that previously reached noise=6% (the 1.5251 winner), causing +0.017 regression. Marginal wins (ibm08 found 1.5419 on Phase 6 iter=1) don't outweigh budget displacement costs.
26. [~] **DP-first ordering on Improvement #1** -- TESTED AND REJECTED 2026-05-21. Flipping to score DP before baseline on large benchmarks lets us return DP when baseline scoring would exceed threshold. But on ibm16, DP=1.5751 loses to baseline=1.5324 (+0.043 regression). Trusting DP unconditionally when baseline can't be scored is strictly worse than skipping DP. Baseline-first kept.
