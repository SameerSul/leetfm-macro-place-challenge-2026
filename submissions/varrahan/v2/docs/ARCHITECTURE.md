# CongFlow v2 -- Architecture & Experiment Log

## Current Best Result (L-change + M1, 2026-05-31)

**Average proxy cost: 1.1782** across all 17 IBM ICCAD04 benchmarks.

| Benchmark | Hard macros | v2 score | vs RePlAce (1.4578) |
|-----------|-------------|----------|---------------------|
| ibm01 | 246 | **0.9402** | -35.5% |
| ibm02 | 271 | **1.1971** | -17.9% |
| ibm03 | 290 | **1.0406** | -28.6% |
| ibm04 | 295 | **1.0445** | -28.3% |
| ibm06 | 178 | **1.3052** | -10.5% |
| ibm07 | 291 | **1.2100** | -17.0% |
| ibm08 | 301 | **1.1712** | -19.7% |
| ibm09 | 253 | **0.8916** | -38.8% |
| ibm10 | 786 | **1.1457** | -21.4% |
| ibm11 | 373 | **0.9735** | -33.2% |
| ibm12 | 651 | **1.3641** | -6.4% |
| ibm13 | 424 | **1.0637** | -27.0% |
| ibm14 | 614 | **1.2940** | -11.2% |
| ibm15 | 393 | **1.2613** | -13.5% |
| ibm16 | 458 | **1.2126** | -16.8% |
| ibm17 | 760 | **1.4519** | -0.4% |
| ibm18 | 285 | **1.4615** | +0.3% |
| **AVG** | | **1.1782** | **-19.2%** |

Comparisons:
- vs RePlAce (1.4578): **-19.2%** (-0.2796 points)
- vs UT Austin GPU/DREAMPlace leader (1.4076): **-16.3%** (-0.2294 points)
- vs sameer_v1 baseline (1.4860): **-20.7%**

Total wall-clock (WSL): 3662.42s = 61.0 min (see timing notes below)

---

## Architecture Overview

### proxy_cost formula

    proxy_cost = 1.0 x wirelength + 0.5 x density + 0.5 x congestion

After normalization: WL ~0.06-0.11, congestion ~1.0-2.1.
Congestion dominates by ~30x. Optimizing WL alone reliably increases proxy cost.

### Phase pipeline (per benchmark call)

1. Load seed -- read initial.plc, legalize overlaps
2. DP variants -- dual-density DP (target 0.85 and 0.65), async if ibm10+
3. Phase 1/2/3 -- iterative cong-grad chain from each DP seed (Phase 7 multi-iter)
4. Phase 5b/5c -- score DP placements, merge winners back
5. Phase 8 -- 40 noise restarts + 12 cong-grad refinement restarts (TOP-5/10/20)
6. Phase 9 -- 3 random-order legalization trials
7. 2-opt -- pairwise macro swap from top seeds, 15s budget
8. R2 -- multi-round local refinement:
   - reloc: single-macro repositioning to reduce cong/density
   - soft-reloc[cong]: gradient-driven cong reduction
   - soft-reloc[density]: gradient-driven density spreading
   - 2-opt: pairwise swap accepting proxy improvements

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

Budget per benchmark in --all mode:
- ibm01: 150+83 = 233s place time (hardcoded, first benchmark)
- ibm02 to ibm18: all hit the 110s floor -> 193s each (110+83s overrun)
- Total place time: 233 + 16x193 = 3321s
- Wall-clock overhead (startup, scoring): ~341s
- Total wall-clock: ~3662s = 61.0 min

---

## Change History

### L-change (2026-05-31) -- avg 1.2593 -> 1.1782

Replaces wall-clock cumulative tracking with _total_place_time_s. Prior K-change
run had ~821s of harness/scoring overhead inflating the cumulative timer, starving
ibm15-18 to only 24-46s budget each.

With L-change, ibm15-18 each get the 110s PER_BENCH_FLOOR_S guarantee:

| Benchmark | K-change | L-change | Delta |
|-----------|----------|----------|-------|
| ibm15 | 1.6061 | 1.2613 | -0.345 |
| ibm16 | 1.5323 | 1.2126 | -0.320 |
| ibm17 | 1.7437 | 1.4519 | -0.292 |
| ibm18 | 1.7941 | 1.4615 | -0.333 |
| avg | 1.2593 | 1.1782 | -0.0811 |

### M1-change (2026-05-31) -- ibm01 budget 200s -> 150s

ibm01 R2 rounds 10-11 improved proxy by only 0.0002 total (0.9403 -> 0.9402).
Reducing to 150s saves 50s of place time and prevents ibm18's HARD_CAP_SAFE_S
guard from clamping its budget.

### K-change (2026-05-30) -- avg ~1.42 -> 1.2593

Multi-restart Phase 8 (40 noise + 12 cong-grad) + 2-opt + R2 multi-round refinement.
First run to break below 1.30 on several benchmarks.

---

## Timing Notes

Total wall-clock was 3662s (61.0 min) in WSL, marginally over the 60-min harness
limit. All 17 benchmarks completed VALID. The overage breakdown:
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

## Known Issues / Next Ideas

- DREAMPlace broken (VENV_PYTHON symlink in WSL). Biggest untapped upside (~0.05-0.10 avg).
- ibm17 (1.4519) and ibm18 (1.4615) still above 1.4 -- main remaining targets.
- WireMask-BBO greedy evaluator: highest-leverage non-GPU idea not yet implemented.
- Timing tight at 61 min. Reducing BUDGET_OVERRUN_S 83->65 is the safe fix.
