# Iteration Progress Log

All scores are proxy cost (lower is better).
Target: beat RePlAce avg of 1.4578.

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

v12 = current best. Full v12 eval confirmed 2026-05-01: avg=**1.4877**.

| Benchmark | hard_n | grid_cells | v1 (leg) | v8 | **v12 (current)** | RePlAce | vs RePlAce | Notes |
|---|---|---|---|---|---|---|---|---|
| ibm01 | 246 | 45x41=1845 | 1.2253 | 1.1854 | **1.1854** | 0.9976 | -18.8% | t_score=7.8s; 6% noise wins; cong-grad worse |
| ibm02 | 271 | 30x27=810 | 1.6800 | 1.5823 | **1.5823** | 1.8370 | +14.0% | t_score=13-16s; iter+wide=8% wins; stale-plc trick; gap CLOSED |
| ibm03 | 290 | 32x29=928 | 1.4100 | 1.3583 | **1.3547** | 1.3222 | -2.5% | t_score=9-12s; adaptive frac=0.01 at iter=7-8 wins |
| ibm04 | 295 | 31x30=930 | 1.4101 | 1.3479 | **1.3390** | 1.3024 | -2.8% | t_score=16s clean; 6 cong-grad iters → new best-ever in full eval |
| ibm06 | 178 | 31x28=868 | 1.7198 | 1.6810 | **1.6797** | 1.6187 | -3.8% | t_score=19s; 3-4 iters + Phase 3 gives 1.6797 |
| ibm07 | 291 | 35x32=1120 | 1.4950 | 1.4950 | **1.4950** | 1.4633 | -2.2% | cong-grad doesn't help; noise doesn't help; baseline wins |
| ibm08 | 301 | 38x34=1292 | 1.5582 | 1.5251 | **1.5251** | 1.4285 | -6.8% | cong-grad worse; 6% noise wins (clean); 1.5539 under load |
| ibm09 | 253 | 36x38=1368 | 1.1363 | 1.1304 | **1.1304** | 1.1194 | -1.0% | 1 cong-grad iter wins |
| ibm10 | 786 | 55x41=2255 | 1.4037 | 1.4037 | **1.4037** | 1.5009 | +6.5% | n>430; returns baseline |
| ibm11 | 373 | 39x45=1755 | 1.2354 | 1.2354 | **1.2354** | 1.1774 | -4.9% | n≤430 (v13: exact scoring); clean t_score~81s; SLOW_SCORE guard |
| ibm12 | 651 | 47x47=2209 | 1.6507 | 1.6507 | **1.6507** | 1.7261 | +4.4% | n>430; returns baseline |
| ibm13 | 424 | 43x43=1849 | 1.4011 | 1.4011 | **1.4011** | 1.3355 | -4.9% | n≤430 (v13: exact scoring); clean t_score unknown |
| ibm14 | 614 | 49x44=2156 | 1.6033 | 1.6033 | **1.6033** | 1.5436 | -3.9% | n>430; returns baseline |
| ibm15 | 393 | 57x38=2166 | 1.6061 | 1.6061 | **1.6061** | 1.5159 | -5.9% | grid>2000; returns baseline |
| ibm16 | 458 | 45x48=2160 | 1.5323 | 1.5323 | **1.5323** | 1.4780 | -3.7% | n>430 AND grid>2000; returns baseline |
| ibm17 | 760 | 51x44=2244 | 1.7437 | 1.7437 | **1.7437** | 1.6446 | -6.0% | n>430; returns baseline |
| ibm18 | 285 | 55x39=2145 | 1.7941 | 1.7941 | **1.7941** | 1.7722 | -1.2% | grid>2000; returns baseline |

**v12 full eval avg (2026-05-01): 1.4877** (ibm08=1.5539 under load; ibm04=1.3390 best-ever in full eval!)
**Best-ever clean estimate:** ~1.4860 (ibm04=1.3390 confirmed, ibm08=1.5251 clean)
**Gap to RePlAce:** 1.4877 - 1.4578 = 0.030 (2.0%)

---

### v12: Phase 3 + clean eval confirms ibm04 best-ever (CURRENT)

**Full eval 2026-05-01: avg=1.4877** (improved from v11 1.4882)

Key result: clean t_score for ibm04 = **16.4s** (vs 38s under load in v11). This allowed 6 cong-grad
iterations → **ibm04=1.3390**, first time best-ever has appeared in a FULL eval (not just isolated test).

**Phase 3 confirmed**: ibm06 improved 1.6802 → **1.6797** via Phase 3 (cong-grad from best_pl using
stale plc map). Phase 3 also fired for ibm02/ibm03/ibm04 but didn't improve (best already found by
Phase 1 in those cases).

**ibm08**: 1.5582 → 1.5539 (slight improvement under load; still worse than clean 1.5251).

**Code change**: EXACT_MACRO_THRESHOLD raised **340 → 430** for v13. This will include ibm11 (n=373)
and ibm13 (n=424) in exact scoring. SLOW_SCORE_THRESHOLD=100s guards against load overrun.
ibm11 clean scoring ~81s → gets 1 cong-grad restart. ibm13 scoring time unknown (isolation test pending).

### v11: Budget safety + EXACT_MACRO_THRESHOLD 400→340 + Phase 3 (CURRENT)

**Problem found in v10b full eval**: ibm11 (n=373) baseline scored in **263.6s** under CPU load
(8+ prior benchmarks running had heated up the CPU). The SLOW_SCORE_THRESHOLD=100s check DID
trigger, returning baseline — but only AFTER the 263.6s scoring finished. Total=266.8s (over budget).
No improvement was possible anyway (all perturbations worse for ibm11), so this was wasted time.

**Fix 1**: EXACT_MACRO_THRESHOLD: 400 → 340. ibm11 (n=373 > 340) now returns baseline without
exact scoring — same result but in <5s. ibm08 (n=301 ≤ 340) still included.

**Fix 2**: Post-scoring budget check in `_try_restart`. If `time.time()-t0 > time_budget_s` after
any scoring call, return False immediately (stop further restarts). Limits overrun to ≤1 scoring
cycle beyond budget instead of full loop continuation.

**NOTE**: Adaptive `t_one_score` (running max inside _try_restart) was implemented and then
REVERTED. It caused ibm04 regression: baseline=19.9s updated to 22s → budget check blocked
iter=5, giving 1.3479 instead of 1.3468. Static baseline measurement kept throughout.

**Phase 3 (experimental)**: After Phase 2 wide steps, if cong_improved and budget ≥ 1.3×t_score,
run one more cong-grad from best_pl using the current (stale) plc map. Phase 3 never fired in
v11 full eval — budget checks too conservative for all benchmarks.

**v11 full eval avg: 1.4882** (2026-05-01). ibm04=1.3479 (load-limited to 4 iters vs 6 in v10b).
ibm08=1.5539 (6% noise skipped — scoring took 38s, only 14s left after 4 restarts).

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

Non-exact benchmarks (n>340 or grid>2000) return pure baseline; no restarts possible.
ibm10, ibm12 already beat RePlAce at legalization-only.

---

## Key Findings So Far

- WL is tiny (~0.06 normalized). Congestion (~1.3-2.5) dominates the proxy.
- SA over-optimizes WL, clusters macros, spikes congestion. Never use WL-only SA.
- initial.plc already has good spread. Legalization preserves it.
- Small noise (2-6%) finds better legalization arrangements on some benchmarks.
- **Density fallback ANTI-CORRELATED**: sum-of-squares density rewards spread placements.
  But spread placements have WORSE proxy (higher congestion). Evidence: ibm11 density-selected
  result = 1.3770 vs baseline = 1.2354 (11.5% regression). Full eval confirmed +0.14 avg hurt.
  Fix: return baseline immediately for any non-exact benchmark.
- ibm18 anomaly: 285 macros but 55x39 grid → exact scoring takes ~220s (whole budget).
  Detection: EXACT_GRID_CELL_LIMIT=2000 grid cells. Returns baseline without restarts.
- ibm11 (n=373): with EXACT_MACRO_THRESHOLD=400, uses exact scoring. t_score=75-81s.
  Baseline wins (proxy=1.2354). 1 restart fits: restart 1 (2%) = 1.2591 (worse).

---

## Tunable Parameters (current v5 values)

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
EXACT_MACRO_THRESHOLD = 400        # ibm11 (n=373) included; ibm13 (n=424) excluded
EXACT_GRID_CELL_LIMIT = 2000       # ibm18 (2145) and ibm10-17 excluded
SLOW_SCORE_THRESHOLD_S = 100.0     # safety net for exact scoring
DENSITY_GRAD_MAX_N    = 100        # never fires for IBM benchmarks (all n>100)
```

---

## Next Experiments to Try

1. [x] Full v4 17-benchmark eval -- confirmed all 17 baselines
2. [x] v5 budget-filling restarts -- ibm01 confirmed 1.1854 with 11 restarts (no improvement beyond 6% win)
3. [x] v6 congestion-gradient perturbation -- ibm02 (-0.060) and ibm06 (-0.036) confirmed
4. [ ] CLEAN full v6 eval -- running. Need uncontaminated ibm07, ibm08, ibm09 results.
       Expected avg: ~1.495-1.498 (ibm02, ibm06 improved; ibm08 uncertain)
5. [ ] ibm15 scoring time test (n=393, grid=2166):
       ibm15 is ONLY excluded by EXACT_GRID_CELL_LIMIT=2000. n=393 < 400 threshold.
       Actual scoring time estimated ~68s (vs conservative formula saying 216s).
       If scoring < 100s: raise EXACT_GRID_CELL_LIMIT to 2200. Gap vs RePlAce = 0.090 (biggest of non-exact group).
6. [ ] ibm08 + ibm07 congestion-grad clean test:
       All prior tests were under load (ibm08 scored 95-131s instead of 31-36s).
       Need clean single-benchmark test to know if cong-grad helps these.
       ibm08 tension: cong-grad at k=1 may cut budget for 6% noise (which gives 1.5251 clean).
7. [ ] Additional congestion-grad fracs (0.08, 0.12) for high-cong benchmarks:
       After confirming ibm08 behavior, add more cong-grad restarts at larger scales.
8. [ ] Multiple congestion-grad starting points:
       Use noise-perturbed position as INPUT to cong-grad instead of always starting from
       baseline_pos. Might find different local minima.
9. [ ] ibm04 congestion-grad: 1.4101 vs RePlAce 1.3024 (gap=0.108); cong=1.783 might benefit.
