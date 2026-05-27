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
| Best `--all` avg | **1.4435** (3-DP + multi-seed 2-opt + k=20) |
| RePlAce target | 1.4578 |
| **Gap to RePlAce** | **−1.0% (beat by 0.0143)** |
| DREAMPlace leaderboard | 1.4076 (UT Austin) |
| **Gap to leaderboard** | **+2.5%** (~0.036 absolute) |
| NG45 (Tier 2) avg | 0.7830 |
| `--all` wall-clock | ~826s (3-DP + multi-seed k=20, window=0.02) |

All 17 IBM benchmarks improved vs v12 baseline. The remaining headroom
(~0.040 to leaderboard) is the focus of the open work below.

---

## Open issues

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

### S1. Basin-hopping 2-opt — cong-grad kick between passes (IMPLEMENTED, DORMANT pending P3)

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
