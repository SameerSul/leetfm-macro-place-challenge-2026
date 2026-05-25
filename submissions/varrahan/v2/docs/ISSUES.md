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
| Best `--all` avg | **1.4471** (3-DP) |
| RePlAce target | 1.4578 |
| **Gap to RePlAce** | **−0.7% (beat by 0.0107)** |
| DREAMPlace leaderboard | 1.4076 (UT Austin) |
| **Gap to leaderboard** | **+2.7%** (~0.040 absolute) |
| NG45 (Tier 2) avg | 0.7830 |
| `--all` wall-clock | ~628s (3-DP) / ~525s (2-DP) |

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

### O2. ibm04 path-dependency under multi-DP

**New issue surfaced 2026-05-25.** ibm04's 2-opt starting point is
chosen via best_pl propagation through Phase 5b / Phase 7 / Phase 8.
Adding a new DP candidate that happens to score better than the
existing best at that point HIJACKS the trajectory.

For ibm04 specifically:
  - 2-DP (lo-fix + hi-mov): best DP candidate = lo-fix at 1.3X (some
    value where 2-opt then reaches 1.2797).
  - 3-DP (+hi-fix): hi-fix DP scores 1.3210 — becomes new best_pl
    earlier, displacing whatever trajectory got 2-opt to 1.2797.
  - 2-opt then reaches only 1.2899 from the hi-fix-derived starting
    point.

**Fix candidates:**
  1. Cherry-pick: only let a DP candidate become best_pl if it improves
     by more than some threshold over previous best.
  2. Track multiple "best" placements (top-K), run 2-opt from each.
  3. Per-DP Phase 7 chain isolation (rng_cong is isolated; this would
     also isolate best_pl propagation — much more invasive).

### O3. Soft macros are still pinned during non-DP candidates

**Partial resolution via A2:** DP-launched-with-soft_movable=True gives
DP-optimized softs on the DP candidate. But the rest of the pipeline
(noise restarts, cong-grad, 2-opt, Phase 7/8) operates on pl_scratch
which has softs from initial.plc throughout.

**Estimated remaining cost:** ~0.01-0.02 of the gap to leaderboard
(per the 2026-05-20 decomposition: soft mismatch ~0.05 at large hard
displacement, but our actual displacement is smaller after refinement).

**Implementation options (multi-hour each):**
- **Force-directed soft placement** with explicit soft-soft repulsion.
  Run as a post-2-opt pass: project softs to new positions that
  minimize per-net spring force + density-bin repulsion.
- **Quadratic placement** via `scipy.sparse.linalg.spsolve` treating
  softs as variables in `min Σ w_n · ||soft − centroid_n||²`. Closed
  form, preserves spread via L2 objective. Needs density
  regularization.
- **Vectorized `PlacementCost.optimize_stdcells` rewrite** — the
  academic FD method. Tested 2026-05-20: 126s per call in Python with
  +0.13 regression at default params. Multi-day vectorized rewrite.

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

---

## Speculative score improvements (not started)

### S1. Multi-pass 2-opt with budget partitioning

**Idea:** instead of one 15s 2-opt-on-winner at the end, run 3 passes
of 5s each, interleaved with one cong-grad iter between them. The
cong-grad in the middle can escape local minima that 2-opt is stuck
in.

**Cost:** ~30 lines. RNG-isolated.
**Expected gain:** −0.001 to −0.005 (speculative).

### S2. Wider 2-opt k_neighbors (15 or 20)

Current k=10. With B3p4's ~3ms scoring, we still aren't always
candidate-bound on small benchmarks. k=15 doubles candidate pool with
small extra cost.

**Cost:** 1-line change.
**Expected gain:** small, depends on which benchmarks are
candidate-bound.

### S3. Phase 8 with extended TOP-K set ({3, 5, 7, 10, 15, 20, 30, 50})

Currently k ∈ {5, 10, 20}. Some benchmarks may benefit from finer
gradations.

**Cost:** ~10 lines (extend the for-loop).
**Risk:** budget displacement.

### S4. 2-opt from multiple seed placements

Currently 2-opt runs once from best_pl. With B3p4's faster scoring,
we could:
  - Run 2-opt from best_pl (current).
  - ALSO run 2-opt from baseline_pos.
  - ALSO run 2-opt from each DP candidate.
  - Pick the lowest-proxy result.

This adds 3-5× the 2-opt budget but only if those paths actually find
different basins.

**Cost:** moderate (~50 lines).
**Expected gain:** −0.001 to −0.010 (highly speculative).

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

### S7. Acceptance criterion for DP candidates

Per O2's analysis, DP candidates blindly become best_pl when they
beat current best. Could require a more conservative margin (e.g.,
"DP candidate becomes best_pl only if it beats current best by more
than 0.005") to prevent shifting downstream trajectories on
small-margin wins.

**Cost:** ~5 lines.
**Risk:** lose legitimate DP wins.

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

### P3. Per-net incremental DENSITY (parallel to B3p4 for cong)

B3 phase 4 made congestion routing incremental. Density is still
recomputed in full each score (via `plc.get_density_cost`). On a
2-opt swap, only 2 macros' cells change density — could be
incremental.

**Cost:** moderate (similar in scope to B3p4 but for density).
**Expected gain:** ~1-2ms per score on top of B3p4.

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
