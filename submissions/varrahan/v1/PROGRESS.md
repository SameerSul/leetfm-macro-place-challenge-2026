# Iteration Progress Log

All scores are proxy cost (lower is better).
Target: beat RePlAce avg of 1.4578.

> **Note (2026-05-09):** This is varrahan/v1's local copy of the team's `PROGRESS.md`,
> updated with v12 findings. The team copy at `/PROGRESS.md` is read-only for this
> submission slot — apply the v12 edits there manually if you want them in the team log.

---

## Baselines (reference)

| Placer | Avg (17 benchmarks) | Notes |
|---|---|---|
| SA baseline | 2.1251 | challenge organizer SA |
| will_seed | 1.5338 | challenge organizer legalization |
| sameer_v1 leg-only | 1.5062 | our legalize-only, confirmed |
| RePlAce | 1.4578 | Grand Prize target |
| UT Austin (DREAMPlace) | 1.4076 | leaderboard #1 |

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

---

## Tunable Parameters (current v12 values)

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
DENSITY_GRAD_MAX_N    = 100        # never fires for IBM benchmarks (all n>100)
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
11. [ ] **DREAMPlace bridge** (`pb.txt → Bookshelf → DREAMPlace global → legalize`) -- leaderboard #1 (1.4076) does this. Multi-day build. Bridge converter `scripts/pb_to_bookshelf.py` doesn't exist; would need to live inside `submissions/varrahan/` per file-scope restriction.
