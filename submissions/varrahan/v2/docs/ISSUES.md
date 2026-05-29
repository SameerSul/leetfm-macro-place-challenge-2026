# Open issues — v2 placer (last revised 2026-05-25)

This is a **clean rewrite**. All issues that have been resolved or
rejected have been removed; their findings are captured in commit
messages and in PROGRESS.md. This file now tracks **only what's
open**: known gaps in the current placer, speculative score ideas that
haven't been tried, and follow-up work that's been queued but not
started.

---

## Current state (headline)

| Metric | Value |
|---|---|
| Best `--all` avg | **1.2799** (P3 + S9 + R1 + R2/R2b + R3 + R5 relocation family) |
| RePlAce target | 1.4578 |
| **Gap to RePlAce** | **−12.2% (beat by 0.178)** |
| DREAMPlace leaderboard | 1.4076 (UT Austin) |
| **Gap to leaderboard** | **−9.1% (BEATS by 0.128)** |
| NG45 (Tier 2) avg | 0.7830 |
| `--all` wall-clock | 2639s (< 3600s cap; see budget note) |

All 17 IBM benchmarks improved vs v12 baseline. The **relocation family** is the
dominant lever of the effort: R1 −0.0096, R2 −0.0083, R2b −0.0027, R3 (soft cong
relocation) −0.0452, **R5 (soft density relocation) −0.0965** → 1.2799, well below
the leaderboard. The throughline: softs are the bulk of BOTH the congestion and
density terms and were frozen at initial.plc by every prior placer; relocating
them (cong-targeted then density-targeted, interleaved with hard reloc + 2-opt)
is where the win lives.

**Open: budget margin.** R5 fits at 2639s on a clean machine, but ibm09 overshot
the 200s soft per-bench limit (307s); under official-eval CPU contention (3–5×
scoring slowdown) it could threaten the 3600s hard cap. A speedup pass is queued
to buy that margin — and a profile (`_profile_move.py`, 2026-05-29) **re-pointed
it**: the per-trial-move cost is NOT dominated by `_compute_cong_cost` (only
~18–21% of a move; density ~0.7%), so the original "incremental smoothing"
target is low-leverage. The real fixed overhead is the **~24 redundant full
`IncrementalScorer` re-inits + per-pass base re-scores per benchmark** (each does
a full routing build + density scatter + full WL) → the **shared scorer** is the
lever. See P5. Budget guard stays as the last stand.

---

## Open issues

### R1. Congestion-directed relocation moves (SHIPPED 2026-05-27 — 1.4422 → 1.4326)

The single biggest lever of the session. The 2-opt search only EXCHANGES two
macros' positions — it can never relocate a routing-heavy macro into an empty
low-congestion gap (a swap would dump some other macro into the vacated hot
spot). R1 adds that missing move: a post-2-opt pass (`_relocation_moves`) that,
for the hottest macros (by live `max(H,V)` congestion), tries moving each into
the nearest lower-congestion legal cell centers, accepting only on a strict
true-proxy drop via the incremental scorer's new `score_move` (single-macro
analogue of `score_swap`; verified bit-exact ≤6e-9, no drift, in
`_verify_score_move.py`). Legality = in-bounds + no overlap with other HARD
macros (softs may overlap). The proxy gate filters far moves that spike WL.

**Result:** --all 1.4422 → **1.4326** (−0.0096), **ALL 17 improved** (ibm04
−0.034, ibm02 −0.026, ibm01 −0.018, ibm15 −0.016, ibm10/13 −0.011), gain in the
congestion term as designed, at ~0.1–0.2s/benchmark (~288 incremental score_move
calls). Strictly non-regressing by construction (best_pl only updates on a true
re-score improvement). RELOC_PROBE (env-gated) reproduces the per-benchmark
measurement.

**Why it worked where DP1 didn't:** R1 relieves congestion with a DIRECT,
proxy-gated move on the placement we already have, rather than trying to fix
DREAMPlace's congestion-blind global placement (which trades away its wl/den edge,
DP1) or refine via swaps only (2-opt).

### R5. Soft DENSITY relocation (SHIPPED 2026-05-29 — 1.3764 → 1.2799, the dominant lever)

R3 relocated softs by the **congestion** field. R5 adds a second soft pass per
interleave round that relocates by the **density** field (softs in the densest
cells → low-density cells). Softs are the bulk of BOTH terms, and — since softs
may overlap — the cong pass can pile them into low-cong cells without relieving
density. `DENS_SOFT_PROBE` proved the headroom: on the (cong-converged) best_pl
the cong field finds **0** more moves but the density field finds **22–68**, for
−0.011 to −0.020, all in the density term. Implemented by adding `use_density` to
`_soft_relocation_moves` (build the hot/cold field from the scorer's occupancy
grid instead of the routing map) + `score_move_soft` already handles it; the R2
soft pass became a two-field loop (`cong` then `density`).

**Result:** --all 1.3764 → **1.2799** (−0.0965), ALL 17 improved (ibm13/02/08
each −0.122, ibm06 −0.120, ibm18 −0.214), all VALID / 0 overlaps, 2639s. The
interleave compounds it (single density pass −0.011/−0.020 on best_pl → −0.03 in
the loop → −0.097 across the full pipeline). Also folds in **R3b** (soft top_hot
48→128). Beats RePlAce by 12.2%, leaderboard by 9.1%.

**Open follow-up — budget margin / speedup** (see headline note + P5): fits at
2639s clean but ibm09 = 307s; the queued speedup is the **shared scorer** across
interleave passes (the `_profile_move.py` profile retired the earlier
"incremental `_compute_cong_cost`" plan — smoothing is only ~20% of a move).

### R4. WL-aware HARD relocation targeting (DISPROVEN 2026-05-29)

Probe of biasing hard-relocation targets toward each macro's net centroid (`wl_blend`
of distance-to-current vs distance-to-centroid) so cong relief costs less WL.
Post-hoc on best_pl was a no-op (hard relocation already converged → 0–2 moves);
the in-loop production A/B (`WL_AWARE=0.5`) was **slightly worse** (ibm03 +0.0015,
ibm07 +0.0025) — the centroid bias steers the greedy interleave to a worse local
min, no upside. Reverted the production gate. Kept inert: `hard_net_centroids()`,
the `wl_blend` option (default 0), and the `WLAWARE_PROBE` diagnostic. (Consistent
with O3's finding that the WL-centroid blend on *softs* gave ~0 — things sit near
their centroids already.)

### R3. Soft-macro relocation (SHIPPED 2026-05-28 — 1.4216 → 1.3764, BEATS leaderboard)

**The dominant lever of the whole effort — −0.0452, all 17 improved, and it put
v2 below the UT Austin DREAMPlace leaderboard (1.3764 < 1.4076).** Soft macros
(std-cell-cluster stand-ins) are the **bulk of the routing demand**, and every
prior placer froze them at `initial.plc`. R3 applies the R1 relocation move to
SOFT macros: relocate the hottest soft clusters (by live `max(H,V)`) into low-
congestion cells, accept-on-true-proxy via the scorer's new `score_move_soft`
(softs touch WL + net-routing congestion + density, NOT macro blockage — no
legality check since softs may overlap; verified bit-exact in
`_verify_score_move_soft.py`). Wired as a **third move type in the R2 interleave
loop** (hard reloc ⇄ soft reloc ⇄ 2-opt), so it compounds round-over-round.

**Result:** --all 1.4216 → **1.3764** (−0.0452), ALL 17 improved, gain in the
congestion term: ibm06 −0.102, ibm07 −0.080, ibm03 −0.067, ibm12/14 −0.062,
ibm17 −0.061. All VALID / 0 overlaps (softs are movable — 0 fixed softs on IBM,
confirmed; a `soft_movable` guard defends NG45/other inputs). 2350s. The
interleave makes the gain 2–4× a single soft pass (each soft move opens new
hard/2-opt moves). `SOFT_RELOC_PROBE` (env-gated) reproduces the single-pass
measurement.

**This corrects O3.** O3 closed soft-repositioning, but only tested *bulk* moves
(WL-centroid blends, gradient spreads). Discrete, proxy-gated, R1-style soft
relocation is a different operator and it is the biggest win we found.

**Follow-ups:** confirmation re-run (the −0.0452 jump is large; all 17 improved
far above the noise floor so it's robust, but a second --all is cheap insurance);
tune soft `top_hot`/`n_targets`; soft 2-opt swaps (exchange two soft clusters).

### R2. Interleaved relocation ⇄ 2-opt (SHIPPED 2026-05-27 — 1.4326 → 1.4243 → 1.4216)

R1 ran relocation once, after 2-opt. R2 ALTERNATES a relocation pass and a 2-opt
cleanup pass (≤6 rounds, budget-gated, break on no-improvement): each relocation
opens new swap opportunities and vice versa, so they compound. Both accept-on-
true-proxy → strictly non-regressing. Relocation runs first each round (the
multi-seed block already 2-opt-converged best_pl). --all 1.4326 → **1.4243**
(−0.0083), ALL 17 improved (ibm04 −0.043, ibm10 −0.022, ibm02 −0.015, ibm12
−0.011); per-benchmark the interleave roughly doubled R1's single-pass gain on
the high-leverage benchmarks (ibm04 walked 1.27→1.19 over 4–6 rounds, monotonic).

**R2b — widened relocation candidate set (top_hot 24→48, n_targets 12→16).**
"Squeeze R2" follow-up: raising the round cap (6→10) was tapped (rounds 7+ below
the noise floor), but the binding limit on large benchmarks was top_hot per
round — at 24 it covered only ~3% of ibm10's 786 macros/round. Widening to 48/16
relieves more hot macros per round (and converges in fewer rounds, so it's also
*faster* on the large benchmarks). --all 1.4243 → **1.4216** (−0.0027), all
improved-or-flat; broader than expected (ibm12 −0.008, ibm16 −0.0065, ibm11
−0.006, not just the largest). 1518s.

**Leverage analysis (`_reloc_leverage.py`):** gain correlates with hard-macro
utilization (canvas fraction occupied by hard macros) gated by congestion
headroom — NOT with macro dominance or open space (both hypotheses refuted:
ibm15 has the most-dominant macro but small gain; ibm18 has the most open space
but the smallest gain). Big gainers (ibm04/10/02/12) have util 0.42–0.60 + cong
above the floor; the two lowest-congestion benchmarks (ibm01/09) and the lowest-
hard-util ones (ibm17/18) barely move.

**Follow-ups:** soft-macro relocation (for the soft/net-dominated low-hard-util
benchmarks ibm17/18 — see option 2 in the alternatives menu); more rounds /
larger 2-opt slice on benchmarks that hit the round cap with budget to spare.

### DP1. Congestion-aware DREAMPlace — the leaderboard gap is pure congestion (CLOSED 2026-05-27 — routopt can't move the proxy)

**Diagnosis (DP_DIAG, env-gated logging in `place()`).** Our DREAMPlace (DP)
candidates lose to the cong-grad "best" seed 15/17. Decomposing why, on the
congestion-heavy benchmarks, shows the loss is **entirely congestion** —
DREAMPlace is *better* on wirelength and density (it optimizes those) and only
loses on the term it can't see:

| | wl | den | cong | proxy |
|---|----|----|------|-------|
| ibm10 raw dp[hi-fix] | 0.0574 | 0.3774 | **0.9543** | 1.3891 |
| ibm10 final best | 0.0636 | 0.3804 | **0.8904** | 1.3344 |
| Δ (dp − best) | −0.006 | −0.003 | **+0.064** | +0.055 |
| ibm12 raw dp[hi-fix] | 0.0626 | 0.3968 | **1.2497** | 1.7090 |
| ibm12 final best | 0.0608 | 0.4017 | **1.1749** | 1.6375 |
| Δ (dp − best) | +0.002 | −0.005 | **+0.075** | +0.071 |

**Post-hoc repair ruled out (mostly).** DP_PROBE (env-gated ceiling test) ran a
generous ungated cong-grad descent + 2-opt on the raw DP basin. ibm10 *did*
recover below best (1.3279 vs 1.3337) — but the production realization (Phase 7b)
was REVERTED: the descent is budget-hungry (~30s/bench), high-variance, and not
even reproducible at fixed seed (plc-state-dependent on pipeline position — seed
777 gave 1.3639 post-pipeline vs 1.3730 mid-pipeline). Captured zero net gain
in-pipeline. Relieving DP's congestion by moving macros *afterward* trades away
its wl/den edge as fast as it gains — the trade-off must be resolved *inside* the
global placement, not after.

**The lever: enable DREAMPlace's built-in routability optimization.** DREAMPlace
has `routability_opt_flag` + `adjust_rudy_area_flag` (params.json) — it computes
a RUDY/RISA routing-congestion map mid-placement and inflates node areas in
congested regions (≤`max_num_area_adjust`=3 times), so the density penalty
spreads cells out of routing hotspots. This is congestion *in the global
objective*. Our bridge currently leaves it OFF (`_default_dreamplace_config`
defaults `routability_opt_flag=0`).

**Result: routopt CANNOT move the TILOS proxy congestion — CLOSED.** Enabling
DREAMPlace's `routability_opt_flag` required two fixes first: a dead-code bug in
`_default_dreamplace_config` (the routability keys were appended after `return`),
and a crash in `PlaceObj.build_nctugr_congestion_map` (it needs per-layer
`unit_horizontal_capacities`, which are None for Bookshelf inputs — patched both
`dreamplace_src` and `dreamplace_build/install` to build the NCTUgr map only when
`adjust_nctugr_area_flag` is set; RUDY is used otherwise, so safe). With routopt
genuinely firing, on ibm10 (`_routopt_poc.py`, `_routopt_calib.py`):

| config | proxy | cong |
|---|---|---|
| routopt OFF | 1.3891 | 0.9543 |
| ON, bins=64, default caps | 1.4109 | 0.9658 (worse) |
| ON, bins=grid(55×41), caps physical×{1,4,16,64} | 1.3891 (all) | 0.9543 (no effect) |

Across a 64× capacity sweep (both directions) + grid-matched route bins, routopt
is either a **no-op or a regression** — it never lowers the proxy congestion. Why:
routopt spreads *movable* cells out of RUDY hotspots, but with
`soft_macros_movable=False` the only movable objects are the hard macros (few,
large, density-dominated) so area inflation barely moves them; and when it does
engage (bins=64) RUDY relieves cells that aren't the TILOS proxy's hotspots
(RUDY ≠ TILOS congestion), with a density headwind. The 0.064 congestion gap to
best is **not closable** via the built-in routability opt.

**Kept (gated off, no pipeline change):** the bridge `routability_opt` knob +
calibration params (default off), the NCTUgr-guard source patch (genuine bug
fix), and the diagnostics (`_routopt_poc`, `_routopt_calib`, `DP_DIAG`/`DP_PROBE`).
v2 stays at **1.4422**.

**Not pursued (low EV / big build):** `soft_macros_movable=True` + routopt (the
`hi-mov` base is already 1.92, far above best); a custom congestion penalty map
fed from the *TILOS* field rather than RUDY (higher ceiling, substantial
DREAMPlace-source build with a per-iteration feedback loop).


### O1. ibm09 / ibm13 small regressions vs the v2-combined baseline (RESOLVED 2026-05-25 — kept 3-DP)

**Status: 3-DP shipped.** `--all` avg 1.4475 → 1.4471 (−0.0004).
Adding `hi-fix` as a 3rd DP (target_density=0.85, soft_movable=False)
recovered the ibm09/ibm13 regressions:

| Bench | 2-DP | 3-DP | Δ |
|---|---|---|---|
| ibm09 | 1.1116 | **1.1035** | **−0.0081** ✓ |
| ibm13 | 1.3890 | **1.3828** | **−0.0062** ✓ |
| ibm08 | 1.5076 | 1.5019 | −0.0057 ✓ (bonus) |
| ibm17 | 1.7372 | 1.7359 | −0.0013 ✓ |
| ibm04 | 1.2797 | 1.2899 | **+0.0102** ⚠ (see O2) |
| ibm10 | 1.3378 | 1.3416 | +0.0038 |
| ibm15/ibm16 | (same) | +0.0007 each | small |

Net cumulative across 17: −0.0070, avg delta −0.0004. Wall-clock
+102s (526s → 628s) for the third DP. The +0.010 ibm04 regression
is a path-dependency issue tracked separately in O2.

Phase 7 RNG isolation (in commit adaf693) was a prerequisite for this
fix — the original 3-DP attempt 2026-05-24 had to be reverted because
adding a third Phase 7 chain caused rng_cong drift that regressed
ibm10 +0.036. Now with isolation, ibm10 only sees +0.004.

### O2. ibm04 path-dependency under multi-DP (RESOLVED 2026-05-25 — candidate #2 shipped)

**Status: multi-seed 2-opt shipped.** ibm04 1.2899 → **1.2797**
(−0.0102, fully recovering the 3-DP regression). `--all` avg
1.4471 → **1.4464**.

The fix is candidate #2 below (run the final 2-opt from each DP basin,
keep the global minimum). Two corrections to the original analysis,
both established empirically this session:

  - **The tags were muddled.** Real ibm04 DP proxies are lo-fix 1.3588,
    hi-mov 1.3210, hi-fix 1.3188. The 2-DP winner was **hi-mov** (1.3210),
    and the 3-DP hijacker was **hi-fix** (1.3188), beating hi-mov by only
    0.0022.
  - **Fix candidate #1 (margin gate) was DISPROVEN.** Adding a 0.005
    acceptance margin so hi-fix can't displace hi-mov as best_pl gave
    1.2913 — *worse* than 3-DP's 1.2899, and nowhere near 1.2797. The
    1.2797 was never a property of the best_pl seed; it was a property
    of the whole 2-DP configuration. Even with hi-mov kept as best_pl,
    the mere presence of the hi-fix candidate perturbs plc state (Phase
    5b uses the last-scored plc state) and adds a Phase 7 chain. So a
    best_pl gate alone (== S7) cannot reproduce the 2-DP trajectory.

**What shipped (candidate #2):** the final 2-opt now runs from `best_pl`
PLUS each DP candidate basin in `dp_placements`, keeping the lowest
result. hi-mov's basin 2-opts to 1.2797 even though it lost the best_pl
race. The win generalizes — ibm09 also improved (1.1035 → 1.1026, via
the dp[hi-fix] basin). Implementation notes:

  - **Selection is by a fresh `_exact_proxy`, never the
    IncrementalScorer's `final_score`.** The incremental WL drifts
    seed-dependently (ibm01 dp[lo-fix]: internal 1.1309 vs true 1.1506).
    A first cut that compared internal scores picked a phantom winner
    and regressed ibm01 1.1317 → 1.1506. Re-scoring each finalist
    exactly fixed it (and incidentally cleaned up the cross-seed plc
    state leakage). The change is strictly additive: the `best` seed
    reproduces the committed single-seed 2-opt, and a seed is kept only
    if its true proxy beats the true-scored incumbent.
  - **Pruning (`DP_SEED_2OPT_WINDOW = 0.02`):** a DP seed whose raw
    proxy is > 0.02 above best_score can't catch up (max observed 2-opt
    gain ~0.04; both wins sit at +0.011 / +0.002), so it's skipped. This
    is provably score-neutral and cut `--all` wall-clock from ~1198s
    (no prune) back to ~722s (committed 3-DP was ~628s). 35 seeds pruned
    across the suite.

**Remaining (not pursued):** candidate #3 (full per-DP plc-state +
best_pl isolation) would let the pipeline reproduce each DP's standalone
trajectory, possibly squeezing a bit more, but it's much more invasive
and the cheap candidate #2 already recovered the regression.

### O3. Soft-macro repositioning (CLOSED 2026-05-26 — confirmed dead lever)

**Status: closed, no headroom.** Soft macros stay at `initial.plc`
throughout the non-DP pipeline; the earlier estimate was ~0.01-0.02 of
recoverable proxy. A measure-first investigation
(`test/diagnostic/_soft_headroom.py`) closed it: `initial.plc` soft
positions sit at a robust local proxy optimum, and every repositioning
method tested makes proxy equal-or-worse.

| Method (probe) | targets | result on stale-soft benches |
|---|---|---|
| WL net-centroid blend (a sweep) | wirelength | best ~−0.002 (a≈0.05), often 0 |
| congestion-gradient bulk soft move | congestion | strictly worse |
| density-spread bulk soft move | density | strictly worse |

Why: wirelength is only ~5% of proxy and the entire soft-WL swing is
~0.005; the dominant density+congestion terms are driven by HARD
placement + net routing, not soft positions. Clustering softs (WL min)
spikes density; spreading them spikes WL + congestion; moving them down
the congestion gradient just relocates congestion. The `initial.plc`
spread (from the prior EDA flow) already balances all three.

Seed analysis (`--all` run4): 15/17 win via the `best` seed and 4 large
benches (ibm08/10/12/16) have NO DP candidate → their softs are
definitely `initial.plc` — yet even those showed zero headroom. So this
isn't a "softs happen to be good on DP benches" artifact; it's structural.

**Do not revisit** without a fundamentally different objective (e.g. a
soft model that DREAMPlace's density-aware NLP optimizes jointly with
hard — but that's the DP path we already have, and DP only wins 2/17).

### O4. The pre-flight skip guard occasionally fires on benign WSL2 clock drift

`time.monotonic()` covers the inside-the-placer paths but the
harness's own `time.time()` reporting (in `evaluate.py`) still
occasionally shows wall-clocks of 36000+ seconds. The harness's
3600s cap uses host wall-clock, so a single Windows-host suspend
during a real submission run could blow the cap.

**Mitigation options:**
- Wrap the harness call with a wrapper that uses a Linux clock that
  pauses during suspend (e.g., `CLOCK_MONOTONIC` not just for the
  placer but for the cap timer).
- File a bug / patch against the harness.
- Run inside a container that has reliable wall-clock under suspend.

Not blocking; submission should run on a non-WSL Linux box where this
doesn't manifest.

### O5. IncrementalScorer relies on clean plc state at init (RESOLVED 2026-05-26)

**Status: fixed.** `IncrementalScorer.__init__` now sets
`plc._last_pos_cache = None` before `_fast_set_placement`, forcing a full
re-set of every macro. After the fix every seed's internal `final_score`
equals the true `_exact_proxy` (`incr==true` across the spot set), so the
seed-dependent drift is structurally gone, not just worked around. The
multi-seed path's true-rescore selection is retained as defence-in-depth.
Root cause and original analysis below.

**Surfaced 2026-05-25 during O2 candidate #2.** `IncrementalScorer.__init__`
calls `_fast_set_placement(plc, current_placement_np)`, which is
"idempotent if positions match `last_pos_cache`". When a second scorer is
built right after a prior 2-opt has mutated plc (the multi-seed case),
the idempotency cache can skip setting some positions, so the WL baseline
(`_compute_per_net_hpwl_full`) is computed against a mismatched plc state.
Result: the scorer's `final_score` drifts from the true `_exact_proxy`
(ibm01 dp[lo-fix]: internal 1.1309 vs true 1.1506).

**Currently mitigated, not fixed.** O2's multi-seed path works around it by
calling `_exact_proxy` (a clean full set) between seeds and selecting on
the true proxy, never the internal score. The single-seed path was never
affected (one scorer, built from a clean-enough state — ibm01 matched).

**Robust fix (~5 lines, defensive):** force a full placement set in
`IncrementalScorer.__init__` (bypass / invalidate the idempotency cache)
so the scorer's baseline is always self-consistent regardless of prior plc
state. Removes the implicit "caller must hand me a clean plc" contract and
unblocks any future code that builds multiple scorers.

**Risk:** low. Worst case is one redundant full set (~ms) at init.

---

## Speculative score improvements (not started)

### S1. Basin-hopping 2-opt — cong-grad kick between passes (DISPROVEN 2026-05-26 — kept dormant)

**Result:** enabling sliced basin-hopping (5s passes + cong-grad kick,
`S1_MAX_KICKS=2`) on top of P3 regressed `--all`: 6/7 benchmarks worse, 1 tie,
0 better before the run was stopped (ibm01 1.1269→1.1306, ibm04 1.2686→1.2777,
ibm08 1.4978→1.5023; cumulative +0.025 over 7). **Slicing the 15s into 5s
passes starves the productive deadline-bound 2-opt search**, and the kicks
perturb away from the optimum without recovering. The "more accepts" signal
that looked promising on a single ibm04 run (671→1072) was misleading — the
extra accepts were repairing kick damage, not net-improving; and the one
ibm04=1.2293 run was a lucky noise draw (ibm04 swings ~0.05 run-to-run).
Even ibm01, which converges early (where S1 *should* help), regressed.
**Kept dormant** (`S1_MAX_KICKS=0` = single full-15s pass); code retained for
reference. A gentler non-sliced variant (full-deadline pass, kick only with
leftover budget after early convergence) is low-EV: it fires only on small
benchmarks with ~1-2s to spare and never on the large average-movers.

**Original idea (for the record):**

**Idea:** 2-opt only PERMUTES existing macro slots — it can never reach a
position no macro occupies. After a pass converges to a swap-only local
min, inject a `_routing_congestion_perturb` KICK (continuous move of the
hottest macros against the live congestion field), legalize, and run 2-opt
again to clean up. Accept-on-true-proxy, keeping the running best across
passes. Per seed: up to `S1_MAX_KICKS+1` passes of `S1_PASS_BUDGET`=5s each
within the same 15s/seed envelope.

**Implemented** in the multi-seed 2-opt loop (basin-hop while-loop, RNG-
isolated via a local RandomState). Currently `S1_MAX_KICKS=0` (DORMANT) =
single full-15s pass = byte-identical to the committed single-pass code, so
P3 can be measured without S1 confounding.

**Key finding (2026-05-26):** the original "kick only on early convergence"
trigger never fired — at k=20/iters=6 the 2-opt is deadline-bound (uses the
full 15s without converging) on *every* benchmark, even small ones. So a
full-deadline pass leaves no budget to kick. Two consequences:
  1. Slicing (5s passes) is required to make kicks fire. An early sliced test
     on ibm04 showed the kick genuinely surfaces NEW improving swaps (accepts
     671→1072), but single-benchmark proxy is too noisy to judge (ibm04 swings
     ~0.05 run-to-run because the deadline-bound greedy path is CPU-load
     sensitive — 1.2293 vs 1.2846 on identical-algorithm reruns).
  2. **P3 changes the regime:** with ~25% faster scoring, small/mid benchmarks
     now converge before 15s (ibm04 12.8–14.3s), freeing budget for kicks.
     So S1 should be re-enabled (`S1_MAX_KICKS=2`) and --all-tested *after*
     P3 lands — it's P3 that makes S1 viable.

**Cost:** ~60 lines (shipped). RNG-isolated.
**Expected gain:** unknown until tested on top of P3; the accept-count jump
is suggestive but noise-dominated at the single-benchmark level.

### S2. Wider 2-opt k_neighbors (SHIPPED 2026-05-26 — k=20)

k_neighbors 10 → 15 → 20 in the multi-seed 2-opt-on-winner.
  - k=10 → 15: all 17 improved, avg 1.4464 → 1.4443 (−0.0021).
  - k=15 → 20: avg 1.4443 → **1.4435** (−0.0008), 15/17 improved. The two
    regressions (ibm13 +0.0004, ibm14 +0.0003) are deadline-bound — wider
    k means fewer total passes fit the 15s budget on large benchmarks — but
    noise-level and outweighed by broad small gains (ibm04 −0.0040).
Wall-clock ~826s. k=25+ not pursued: the deadline-bound regime is
expanding, so further widening likely hurts large benchmarks more than it
helps small ones. An adaptive-k (wider on fast benchmarks) is the next
lever if this is revisited.

### S3. Phase 8 with extended TOP-K set ({3, 5, 7, 10, 15, 20, 30, 50})

Currently k ∈ {5, 10, 20}. Some benchmarks may benefit from finer
gradations.

**Cost:** ~10 lines (extend the for-loop).
**Risk:** budget displacement.

### S4. 2-opt from multiple seed placements (PARTIALLY SHIPPED — see O2)

The multi-seed 2-opt framework shipped 2026-05-25 (O2 candidate #2):
2-opt now runs from best_pl + each DP candidate basin, with true-proxy
selection and window-0.02 pruning. Remaining cheap extensions, now that
the framework + `twoopt_seeds` list exist (each is ~1-2 lines):
  - **`baseline_pos` as a seed** — catches benchmarks where the refined
    best_pl landed in a worse basin than the raw legalized baseline.
  - **top-K noise restarts as seeds** — requires tracking the best few
    noise placements (more state); defer unless baseline_pos pays off.

**Cost:** baseline_pos ~2 lines; noise restarts moderate.
**Expected gain:** −0.001 to −0.010 (speculative); pruning keeps the
cost near-zero on benchmarks where these can't win.

### S5. Cong-grad with adaptive frac per cell

Currently `frac=0.04` is a single global parameter. Per-cell
adaptive perturbation magnitude based on local congestion ratio
could provide more targeted moves.

**Cost:** modify `_routing_congestion_perturb`.
**Expected gain:** small, depends on whether the simple linear
`move_scale = scale * local_cong` (already present) captures most
of the benefit.

### S6. Phase 7 starting from best_pl alternatives

Currently Phase 7 chains start from each DP placement. Phase 8 chains
start from best_pl. Could try Phase 7 starting from:
  - Each DP candidate (current).
  - Each noise restart in the top-K by score.
  - baseline_pos with cong-grad applied N times beforehand.

**Cost:** moderate.
**Risk:** budget displacement.

### S7. Acceptance criterion for DP candidates (DISPROVEN 2026-05-25 — see O2)

Tested: a 0.005 best_pl acceptance margin on ibm04 gave 1.2913, *worse*
than 3-DP's 1.2899. The path-dependency isn't in the seed choice (plc
state + Phase 7 chain count also shift), so a best_pl gate can't help.
Superseded by O2's candidate #2 (multi-seed 2-opt), which shipped.

### S8. Phase 9 random-order: increase trial count

Currently N=3 random-tiebreak trials. With B3p4 + 2-opt widening,
budget is freed; more trials might find better legalizations on
benchmarks with many same-area macros.

**Cost:** 1 line.
**Expected gain:** small.

### S9. Congestion-aware 2-opt candidate selection (SHIPPED 2026-05-26 — 1.4424 → 1.4422)

Two layered changes inside `_two_opt_proxy_swap`, gated on a `macro_cong`
(per-macro local `max(H,V)` snapshot taken at seed time):
  - **Variant 1 — hot-first outer ordering.** Iterate macros by descending
    local congestion instead of by index. On deadline-bound benchmarks the
    swaps evaluated before the budget expires are then the hotspot ones —
    the dominant proxy term. Pure budget reallocation (can't beat the
    deadline-free convergence point).
  - **Variant 2 — cold-region teleport augmentation.** Spatial kNN can only
    swap nearby macros, so a routing-heavy macro can never relocate across
    the chip (intermediate local swaps all reject). For the `cong_hot_k`=20
    hottest macros, append the `cong_cold_k`=8 coldest as extra candidates —
    a long-range edge that expands the reachable placement set. Size-
    incompatible teleports fail the free conflict check before scoring.

The proxy gate validates every swap, so this only changes WHICH candidates
are tried, never accepts a worse placement. `macro_cong=None` reproduces the
prior index-order / spatial-only behavior exactly.

**Result:** --all 1.4424 → **1.4422** (−0.0002). 12/17 improved, 5 slightly
worse, cumulative −0.0042 (ibm06 −0.0023, ibm14 −0.0011 the standouts; ibm01
+0.0015 the worst). 12/17 same-direction is ≈7% by chance, so likely-real but
**marginal — edge-of-noise.** All 17 VALID / 0 overlaps; teleports confirmed
firing (ibm10 accepts 1168→1327). Theoretically the higher-ceiling of the two
candidate-selection variants (expands reachability vs reorders a fixed set);
kept because it's net-positive, consistent-direction, and correctness-safe.

**Theory note (vs S1):** unlike S1 (which sliced the budget and starved the
search → regressed), S9 keeps the full pass and only changes candidate choice
— every accepted teleport strictly lowers proxy, so no budget-waste damage.

---

## Speculative performance ideas (not started)

### P1. B5 GIL-aware parallel scoring

**Untested.** If `plc.get_*_cost` C++ paths release the GIL,
ThreadPoolExecutor with 2-4 workers could double or quadruple score
throughput. Currently single-threaded.

**Verification first:** spawn 2 threads each calling `_exact_proxy`
on the same plc. If wall-clock ≈ 50% of single-thread, GIL releases.

**Implementation if it works:** 2-opt could try 2-4 swap candidates
in parallel, score each, accept the best improvement. Requires
careful synchronization since accept changes shared state.

**Expected gain:** doubles or quadruples 2-opt accept rate, may
translate to small score improvement on candidate-bound benchmarks.

### P2. B6 batched `_fast_set_placement`

Currently Python loop calls `set_pos` per-macro. With ~1500 macros
on large benchmarks, iteration overhead is ~2ms per call. Multiple
candidates per score → potentially 5-10ms per benchmark per `--all`.

**Status:** plc may not have a batched API. Worst case requires a
binding contribution.
**Expected gain:** small wall-clock save.

### P3. Per-net incremental DENSITY (IMPLEMENTED 2026-05-26 — verifying via --all)

B3 phase 4 made congestion routing incremental. Density was the last
full-recompute in `score_swap` (`plc.get_density_cost` scatters ALL
soft+hard macros into the occupancy grid each call). On a 2-opt swap
only macros i, j move, so the occupancy delta is a handful of cells.

`IncrementalScorer` now maintains `grid_occupied` as state:
`score_swap` subtracts i,j's OLD footprints + adds NEW (via
`_macro_occ`, an exact per-macro replica of the full overlap math),
takes top-10% over the grid, then reverts the touched cells;
`commit_swap` persists the delta. `_compute_density_cost` mirrors
`get_density_cost` (0.5 × mean of top floor(0.1·n_cells) nonzero cells).

**Verified:** `_verify_incremental_scorer.py` — score_swap (incl. density)
matches `_exact_proxy` to ≤4.4e-16 (machine eps) on ibm01/04/10, both
trial swaps and sequential commits (no drift).
**Measured speedup** (`_profile_density.py`): score_swap −22% to −29%
(ibm01 1.77→1.36ms, ibm04 1.46→1.14ms, ibm10 2.03→1.44ms, ibm16
1.99→1.47ms). Translates to ~40–56% more 2-opt scores in the 15s
deadline (ibm04 7914→11058; ibm10 4784→7482).

**Why it matters:** the 2-opt is deadline-bound on large benchmarks
(ibm10/12/16 use the full 15s), so more-scores-per-second converts
directly to more accepts → lower proxy on the congestion-heavy
benchmarks that dominate the --all average. On small/mid benchmarks
(ibm01/04) the speedup lets 2-opt *converge* before 15s — which is the
budget S1 needs to fire (see S1).

### P4. Skip `_routing_congestion_perturb` on Phase 9 trials

Phase 9 (random-order legalize) doesn't use congestion gradient —
it just legalizes from init_pos with a shuffled order. The
`_routing_congestion_perturb` calls in Phase 1/2/3/5b/5c/7/8 are
needed for cong-grad. Phase 9 trials currently don't call
_routing_congestion_perturb (correct), but verify no redundant
state computation.

**Status:** likely a no-op fix; flagged for completeness.

### P5. Interleave speedup — shared scorer (PROFILED 2026-05-29, not yet implemented)

The R5 interleave (hard reloc ⇄ soft-cong ⇄ soft-density ⇄ 2-opt, ≤6 rounds)
fits in 2639s clean but ibm09 hit 307s — thin under official-eval CPU contention.
A speedup pass is queued; `_profile_move.py` (env-free, run on ibm15/17/10)
**measured where the time goes and corrected the plan**:

| bench | grid | `score_move_soft` | `_compute_cong_cost` | smoothing | `_compute_density_cost` |
|---|---|---|---|---|---|
| ibm15 | 2166 | 1.54 ms | 0.30 ms (20%) | 18% | 0.01 ms (0.7%) |
| ibm17 | 2244 | 1.35 ms | 0.27 ms (20%) | 21% | 0.01 ms (0.7%) |
| ibm10 | 2255 | 1.27 ms | 0.28 ms (22%) | 20% | 0.01 ms (0.7%) |

**Finding:** the per-move congestion cost (re-smooth + top-5% partition) is only
~20% of a trial, and density is ~0.7%. So **incremental `_compute_cong_cost`
(the original speedup target) is low-leverage** — it could shave ≤0.3 ms of
1.4 ms, and per-move cost is only part of the interleave time. The other ~78% of
a move is the touched-net routing apply + WL subset + flat snapshot/restore
(already incremental).

**Revised target — the shared scorer.** The dominant *fixed* overhead is that the
interleave rebuilds a fresh `IncrementalScorer` ~24×/benchmark (each = full
routing build + full density scatter + full WL, ~1–3 s each) plus a per-pass
`_exact_proxy` base re-score. Keeping **one** scorer for the whole interleave —
committing moves to it across passes and deriving `best_pl` from its committed
state — eliminates that, est. ~60–75 s/benchmark (ibm09 ~307 s → ~180 s).

**Key enabler / the one subtlety:** the relocation passes currently read their
hot/cold field from `plc.get_*_routing_congestion()`, which is only fresh because
each per-pass rebuild calls `plc.get_congestion_cost()`. A shared scorer must
instead build that field from the scorer's **maintained `H_flat`/`V_flat`** (the
same flats `_compute_cong_cost` already uses — current after every commit, no
recompute). That removes the only thing forcing a rebuild between passes.

**Risk:** real refactor of the core interleave loop (cross-pass move-state
sharing + field-source change). Backstopped by the verified bit-exact incremental
proxy (no drift over commits) and a confirming `--all` to prove 1.2799 holds.
Budget guard stays as the last stand regardless.

---

## Maintenance items

### M1. Stale failing test in `test/test_varrahan_v2_congestion.py`

The test asserts H+V combined-perturb behavior, but
`_routing_congestion_perturb` uses `max(H, V)` per a documented A/B
test. The actual code is correct; the test was authored when the
code briefly used H+V and never re-synced.

**Constraint:** `test/` is read-only per CLAUDE.md. Requires user to
update the test or grant write permission.

### M2. Harness `time.time()` exposure (see O4)

Cosmetic but confusing — the harness's "Total runtime" output
sometimes shows 36000+ seconds while actual elapsed is <600s. Caused
by WSL2 wall-clock drift in the harness's own timing (which we don't
control).

---

## What's NOT in this list (resolved or rejected — see commits)

The session 2026-05-23 → 2026-05-25 closed the following. They're
documented in the commit messages; no need to track them here.

- B1 — `--all` wall-clock timeout (cumulative-budget guard).
- A1 — 2-opt-on-winner uses displacement, not proxy (proxy 2-opt shipped).
- B3 phases 1-4 — incremental scoring (position cache, per-net HPWL,
  numpy abu, per-net routing).
- B4 — `_vectorized_get_routing` dispatch cache.
- A3 — DREAMPlace diagnostic re-run.
- A6 axis #1 (TOP-K cong-grad / Phase 8) and axis #4 (random-order /
  Phase 9).
- 2-opt widening (k_neighbors=10, max_iters=6, Phase 8 multi-iter chains).
- A2 — DP soft_macros_movable diversification (lo-fix + hi-mov).
- WSL2 clock-drift hardening (`time.monotonic()` throughout).
- NG45 design disambiguation in `_load_plc`.
- A5 — Phase 7 retro-eval (gate + RNG isolation, bit-stable).
- Rejected: A4 (DP gating), A6 axes #2 (drop lo) and #3 (fine-noise
  from best), B7 (score cache), B8 (adaptive max_iters), B9 (smart
  ordering), 2-opt cache memoization, stale soft-resnap helpers (only
  in v1).
