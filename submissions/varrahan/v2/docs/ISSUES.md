# Open issues — v2 placer (audits 2026-05-22, 2026-05-23)

This file lists glaring, actionable issues. Split 2026-05-23 into two
parts:

- **Part A — Score improvement issues** (directly affect proxy cost).
- **Part B — Performance / speedup issues** (affect wall-clock and
  throughput; speedups also unlock more score work, e.g. more 2-opt
  candidates per budget).

History audit dates: original 1-5 from 2026-05-22 vectorization sweep;
6-9 from 2026-05-23 end-to-end review; B3-B9 added 2026-05-23 after
issue #1 proxy 2-opt result reframed speedups as score-enabling.

Cross-reference to old numbering: A1=#1, A2=#2, A3=#3, A4=#7, A5=#8,
A6=#9 (score); B1=#6, B2=#4 (perf); C1=#5 (maint).

---

## Priority order (2026-05-23)

### Tier 1 — CRITICAL (blocks every other gain)

- **B1 — `--all` wall-clock timeout** (medium work, existential impact;
  IN PROGRESS 2026-05-23). Without this, no other improvement can be
  reproducibly measured.

### Tier 2 — HIGH IMPACT, LOW WORK (immediate score wins)

- **A1 — 2-opt-on-winner uses displacement, not proxy** (**RESOLVED
  2026-05-23**). `--all` avg 1.4782 → 1.4723 (−0.0059); all 17 benchmarks
  improved. See A1 section for per-benchmark deltas.
- **A4 — DP launch displaces noise winner on ibm07** (~20 lines).
  Note: A1 changed ibm07 from 1.4924 to 1.4866, so the +0.003 vs v15
  gap is now closed. A4 may have less to give post-A1 — re-measure
  before implementing.
- **A5 — Phase 7 (DP-rescue chain) is unvalidated** (~10 lines instrument).

### Tier 3 — HIGH LEVERAGE PERFORMANCE (unlocks more score work)

- **B3 — Incremental scoring for 2-opt** (highest leverage). A 2-opt
  swap perturbs only 2 macros — most nets/cells unchanged. Current
  `_exact_proxy` rescores everything. Estimated 5-20× speedup; would
  quadruple 2-opt's win rate.
- **B4 — `_vectorized_get_routing` dispatch overhead** (17ms, half the
  per-score cost). Profile first; fix the bottleneck.

### Tier 4 — MEDIUM IMPACT, LOW WORK (diagnostic, unlocks Tier 5)

- **A3 — Re-run `_dp_diagnostic.py` with the fixed bridge**.

### Tier 5 — HIGHEST CEILING, HIGH WORK (the score floor)

- **A6 — 9/17 benchmarks have no improvement over the v12 floor**.

### Tier 6 — INVESTIGATED / BLOCKED / RESOLVED (no further action)

- **A2 — Soft macros pinned at initial positions**. Investigated; no
  cheap path.
- **C1 — Stale failing test in `test/`**. `test/` is read-only.
- **B2 — `_smooth_routing_cong_vec` Python loop** (fixed 2026-05-22).

### Speculative / supporting performance ideas (B5-B9)

Listed in Part B below. Lower-priority than B3/B4 but cheap to try.

---

# Part A — Score improvement issues

## A1. 2-opt accepts swaps on the wrong cost function (RESOLVED 2026-05-23)

**Status: RESOLVED — `--all` validation passed.**

**`--all` avg: 1.4782 → 1.4723 (−0.0059). All 17 benchmarks improved.**
Gap to RePlAce shrunk from +1.4% to +1.0%.

| Benchmark | v2+#6 | v2+#6+#1 | Δ |
|---|---|---|---|
| ibm01 | 1.1505 | 1.1352 | −0.0153 |
| ibm02 | 1.5800 | 1.5713 | −0.0087 |
| ibm03 | 1.3593 | 1.3532 | −0.0061 |
| ibm04 | 1.3079 | 1.2971 | −0.0108 |
| ibm06 | 1.6790 | 1.6744 | −0.0046 |
| ibm07 | 1.4924 | 1.4866 | −0.0058 |
| ibm08 | 1.5189 | 1.5142 | −0.0047 |
| ibm09 | 1.1139 | 1.1037 | −0.0102 |
| ibm10 | 1.3876 | 1.3821 | −0.0055 |
| ibm11 | 1.2326 | 1.2292 | −0.0034 |
| ibm12 | 1.6506 | 1.6482 | −0.0024 |
| ibm13 | 1.3955 | 1.3907 | −0.0048 |
| ibm14 | 1.5956 | 1.5906 | −0.0050 |
| ibm15 | 1.6060 | 1.6045 | −0.0015 |
| ibm16 | 1.5225 | 1.5181 | −0.0044 |
| ibm17 | 1.7437 | 1.7425 | −0.0012 |
| ibm18 | 1.7941 | 1.7870 | −0.0071 |
| **AVG** | **1.4782** | **1.4723** | **−0.0059** |

Wall-clock cost: 437s → 543s placer time (~+105s of 2-opt scoring,
runs within budget on every benchmark, no timeouts).

**Where:** `placer.py` `_two_opt_proxy_swap` and call site at the end of
`place()`.

**What was wrong:** `_two_opt_swap` accepted a swap iff per-pair
displacement-from-init decreased
(`d_i_new + d_j_new < disp_sq[i] + disp_sq[j]`). That criterion has no
direct relation to proxy cost, which is congestion-dominated. As a
result, 2-opt "improved" the swap count but routinely made proxy worse:

| Benchmark | Best proxy | 2-opt result (displacement) | Δ |
|---|---|---|---|
| ibm01 | 1.1505 | 1.1603 | +0.0098 (worse) |
| ibm04 | 1.3159 | 1.3210 | +0.0051 (worse) |
| ibm10 | 1.3866 | 1.3945 | +0.0079 (worse) |
| ibm12 | 1.6506 | 1.6506 | tied |

The placer guarded with `if opt_score < best_score:`, so this wasn't a
correctness bug — but the 15s budget was wasted.

**Why it persisted:** displacement-from-init was a fast surrogate when
`_exact_proxy` cost 450ms/call. With ~5-50ms scoring, real proxy-driven
2-opt is feasible.

**Fix:** `_two_opt_proxy_swap` scores each candidate swap via
`_exact_proxy`. Cheap bounds + conflict checks remain as a free filter.
Apply swap tentatively, score, keep if proxy improves else revert.

---

## A2. Soft macros are pinned at their initial positions (INVESTIGATED 2026-05-22 — no cheap win)

**Where:** `placer.py:1438` (soft_indices handling) and downstream
candidate construction.

**What's wrong:** CLAUDE.md flags this explicitly:

> Soft macros must be repositioned when hard macros move significantly.
> The current placers leave soft macros at their initial positions —
> acceptable for small perturbations, problematic for large
> displacements (e.g., DREAMPlace-style global re-placement).

When DREAMPlace's NLP moves hard macros far from initial, the soft
macros (stdcell clusters in the proxy model) stay rooted at original
locations → phantom wirelength + density spikes around stranded
clusters.

**Investigation 2026-05-22 — results:**

| Approach | ibm04 | ibm10 | Conclusion |
|---|---|---|---|
| Initial soft (baseline) | 1.3079 | 1.3866 | reference |
| `soft_macros_movable=True` in DP bridge | 1.3209 (+0.013) | (n/a) | DP's NLP softs misalign with subsequent cong-grad-from-DP |
| Unconditional analytic re-snap | 1.6465 (+0.34) | n/a | naive centroid clusters softs; congestion 1.62 → 2.21 |
| DP-only +resnap candidate, blend=1.0 | 1.6506 (+0.34) | (n/a) | same clustering |
| DP-only +resnap candidate, blend=0.2 | 1.3656 (+0.05) | (n/a) | still net-negative |
| DP-only +resnap candidate, blend=0.05 | 1.3220 (+0.014) | 1.3906 (+0.002) | net-negative + burns 15s budget |

Cheap analytic re-snap **consistently regresses** at every blend factor
on every benchmark. Root cause: initial.plc's hand-tuned spread is more
valuable for congestion than connection alignment.

**What's kept:** `_build_soft_resnap_cache` and `_resnap_soft_macros`
remain in `placer.py` (~200 lines) for future exploration, NOT wired
into the pipeline.

**Real paths if revisited:**
- Force-directed with explicit soft-soft repulsion.
- Solver-based quadratic placement (scipy.sparse.linalg).
- Vectorized rewrite of `PlacementCost.optimize_stdcells`.

None fit on remaining timeline relative to A6 leverage axes.

---

## A3. DREAMPlace's actual contribution to best-of is unclear

**Where:** async DREAMPlace launch and result merge in `place()`.

**What's wrong:** UT Austin's DREAMPlace pipeline holds leaderboard #1
at 1.4076. v2 imports the v1 bridge, launches 2 target_density
candidates (0.85 and 0.65). Yet v2 lands around 1.478 avg — a 0.07 gap.
DREAMPlace's NLP solution is either (a) not actually winning many
benchmarks, or (b) being destroyed by the post-NLP legalizer.

**How to apply:** re-run `_dp_diagnostic.py` (in
`tests/dreamplace/_dp_diagnostic.py`) with the bridge architecture fix
applied. Log raw-DP proxy vs post-legalize proxy vs cong-grad proxy on
a few benchmarks. If raw-DP is close to leaderboard and post-legalize
is far, the legalizer is the culprit (A2 contributes). If raw-DP is
already far, DREAMPlace isn't the win path it appears to be.

---

## A4. DREAMPlace launch unconditionally displaces noise winners on tight budgets (formerly #7)

**Where:** async DP launch and Phase 5 DP candidate scoring in
`place()`.

**What's wrong:** the two async DREAMPlace handles launch at every
`place()` entry, and Phase 5 waits up to ~30s per handle then spends
~60s legalizing+scoring each. On benchmarks where DP loses cleanly
(e.g., ibm07), this consumes budget that would have reached the
winning 1% noise restart. Net effect: documented +0.003 regression on
ibm07.

**How to apply:**
- Generic gate: skip Phase 5 wait if `cong_improved=True` AND
  `noise_fracs[5:]` haven't been tried yet AND `remaining < 3 *
  t_one_score`.
- Cap Phase 5 DP wait at `min(remaining * 0.3, 30s)` instead of 30s.

**Estimated reclaim:** −0.003 on ibm07 directly; possibly more silent
helps on ibm09/ibm11.

---

## A5. Phase 7 (DP-rescue cong-grad chain) contribution is unvalidated (formerly #8)

**Where:** `placer.py` Phase 7 loop (`MAX_P7_ITERS=3`).

**What's wrong:** added 2026-05-21 to chain up to 3 cong-grad iters
from each DP placement after the noise loop. No per-benchmark
accounting demonstrates Phase 7 actually wins anywhere. The greedy
break-on-no-improvement guard limits damage, but each iter costs one
legalize + one score (~10-60s combined per benchmark), and the chain
runs per DP handle.

**How to apply:**
- Add a per-Phase-7 log accumulator: count wins (Phase 7 candidate
  beats `best_score` at chain entry) vs losses across a full `--all`
  run. If wins ≤ 1, delete the phase (~45 lines).
- Alternatively, run `_dp_diagnostic.py` with Phase 7 enabled/disabled.

**Risk if left in:** budget bleed on benchmarks where it produces
nothing.

---

## A6. 9/17 benchmarks have no improvement over the v12 floor (formerly #9)

**Where:** systemic; affects the score ceiling.

**What's wrong:** v15 wins on 8 benchmarks vs v12 and ties on 9
(ibm02, ibm03, ibm08, ibm09, ibm13, ibm15, ibm16, ibm17, etc.). Those
9 contribute ~half the avg sum. The empirical pattern: DP loses on
these benchmarks; cong-grad already converged to a local minimum that
noise can't escape; A1 (proxy 2-opt) may rescue some of these — TBD
from `--all` validation.

**Why it persisted:** every individual fix idea has been tested and
either regressed or netted zero. The remaining axis isn't an
incremental tweak — it's an orthogonal search primitive.

**Concrete unexplored leverage axes:**
1. **Multi-restart from `best_pl` instead of `baseline_pos`**: Phase 1
   restarts from baseline. A "fine-noise from best" tail (frac=0.5-2%,
   5-10 candidates) costs little.
2. **Per-macro selective perturbation**: move only the TOP-K
   most-congested macros (k=5-20) per call.
3. **Order-randomization in `_will_legalize`**: random shuffles within
   same-area buckets.
4. **`_dp_diagnostic.py` re-run** (A3): tells us which 9 have headroom.

**How to apply:** pick one axis (recommendation: #1 — fine-noise from
best — lowest risk, easiest to A/B). Should be attacked after A3.

---

# Part B — Performance / speedup issues

## B1. `--all` wall-clock timeout makes the 1.4804 headline unverified (CRITICAL — IN PROGRESS 2026-05-23)

**Status update 2026-05-23:** Three defensive changes applied; `--all`
validation completed successfully (run before A1):

- avg = **1.4782** (vs v15 partial 1.4804, −0.0022).
- All 17 benchmarks VALID, 0 overlaps.
- Wall-clock ~3360s under harness 3600s cap.
- ibm18 returned baseline (cumulative=3352s triggered pre-flight guard).

**Changes applied:**
1. **Cross-benchmark cumulative tracking** (`MacroPlacer.__init__`,
   `_first_place_call_time` + `_benchmarks_done`). Computes
   `adaptive_cap = (3300 - cumulative) / (17 - done) * 0.9`, uses
   `effective_budget_s = min(time_budget_s, max(30s, adaptive_cap))`
   throughout the 12 budget-check sites in `place()`.
2. **Pre-baseline-score skip guard** — if `effective_budget_s < 60s`
   OR `cumulative > 0.95 * HARNESS_TOTAL_BUDGET_S`, return baseline
   immediately.
3. **`SLOW_SCORE_THRESHOLD_S` tightened 100s → 80s**.

**Tested and reverted:**
- `n>500` DP launch gate. A/B on ibm10: 1-DP=1.3891 vs 2-DP=1.3876
  (−0.0015 score, no wall-clock saving). Reverted.

**Original problem statement** (kept for reference):
> v4 and v5 `--all` runs both timed out at ibm17 at the 3600s
> cumulative cap. The 1.4804 avg was a partial-run extrapolation, not
> a measured headline.

| Benchmark | Time | Source |
|---|---|---|
| ibm15 | 239s | baseline scoring + cong-grad |
| ibm16 | 170s | baseline-only after slow-score skip |
| ibm17 | >300s | baseline scoring alone |

The cumulative-tracking fix means even when the harness adds large
overhead between benchmarks (observed 2270s gap between ibm16 and
ibm17 in the validation run), the placer adapts and returns baseline
for late benchmarks rather than timing out the whole run.

---

## B2. `_smooth_routing_cong_vec` had a per-row/col Python loop (FIXED 2026-05-22)

**Where:** `placer.py` (`_smooth_routing_cong_vec`).

**What was wrong:** the smoothing function used a Python `for r in
range(grid_row)` loop that did a numpy add per row. For ibm10
(grid_row=41, grid_col=55) that was ~96 iterations × small numpy op —
roughly 1–3ms per `get_routing()` call.

**Fix:** replaced with `np.add.at` on flattened/broadcast indices.
For `axis_h=True` the call becomes `np.add.at(events, lp, weighted)`.
For `axis_h=False`, uses tuple-of-arrays advanced indexing.

**Result:** 1–3ms → 0.2ms per get_routing call (~10× speedup on
smoothing). End-to-end ibm04 dropped 1.3159 → 1.3079; ibm01 unchanged.
Bit-equivalent vs scalar verified.

---

## B3. Incremental scoring for 2-opt (NEW 2026-05-23, highest leverage perf)

**Where:** would touch `_exact_proxy` + `_score` callers; new code path.

**Why it matters:** A1's proxy-2-opt found 137 swaps in 1706 scores on
ibm01 (12s), 274 swaps in 1714 scores on ibm04 (15s), 202 swaps in 549
scores on ibm10 (15s — score-bound). **Speedup directly buys more
2-opt accepts.** ibm10 found ~37% accept rate but ran out of budget at
549 scores; with 2× faster scoring, ~1100 scores → ~400 accepts → much
larger −Δproxy.

**What's wrong:** a 2-opt swap moves only 2 macros, but `_exact_proxy`
recomputes:
- All net HPWLs (most don't touch macros i,j).
- All grid-cell densities (most cells don't contain macros i,j).
- All grid-cell routing congestion (most cells don't change).

The bulk of the work is rescoring the *same* state.

**How to apply:**
- Build an inverse index: net → macros, macro → nets (cached once per
  benchmark).
- For a candidate swap (i, j): compute the set of touched nets =
  `nets_of(i) ∪ nets_of(j)`. Rescore only those nets for HPWL.
- For density: macros i and j occupy old + new cells. Subtract old
  contribution, add new. Re-sort top-10% only when the changed set
  affects the top-10% boundary (rare).
- For congestion: harder — the smoothing kernel makes per-cell changes
  ripple. Best path: cache the routing field's per-net contributions,
  recompute only touched nets' contributions, re-smooth.

**Expected gain:** 5-20× speedup on the 2-opt path. End-to-end:
ibm10's 2-opt could go from −0.0035 → ~−0.012 (more candidates fit).

**Risk:** correctness drift from incremental updates accumulating
numerical error. Mitigate with periodic full rescore (every 50 swaps)
to catch drift.

---

## B4. `_vectorized_get_routing` dispatch overhead (17ms/call, ~half of per-score cost)

**Where:** `placer.py` `_vectorized_get_routing` (~line 1362).

**What's wrong:** the per-call score-cost breakdown attributes 17ms to
`_vectorized_get_routing`, with the comment "dispatch + lexsort
overhead dominates". This means the math is already vectorized, but
the Python-side glue (calls into vectorized helpers, lexsort to
order pin events, etc.) eats 17ms.

**How to apply:**
- Profile with cProfile / line_profiler to identify the specific lines.
- Common Python overhead sources to check:
  - `np.lexsort` on large arrays (lexsort is O(n log n) and allocates).
  - Repeated `np.asarray` calls (each one validates and may copy).
  - Dictionary lookups in hot loops.
- Possible fix patterns:
  - Cache the lexsort result if pin positions are net-stable.
  - Pre-sort pins at cache-build time so per-call order is fixed.
  - Replace lexsort with a single integer composite key.

**Expected gain:** 17ms → ~5ms (3× speedup on this stage, ~25%
total-score speedup).

---

## B5. GIL-aware score parallelism (speculative)

**Where:** would wrap the score function in a `ThreadPoolExecutor` or
multiprocess pool.

**What's wrong (or could be improved):** scoring is sequential. If
`plc`'s C++ scoring path releases the GIL on long ops (the v15
DREAMPlace async architecture implicitly assumed this), Python
threads can score multiple candidates concurrently.

**How to verify:** before implementing, write a benchmark: spawn two
threads each calling `_exact_proxy` repeatedly on the same plc. If
wall-clock for 100 scores per thread ≈ 100 × single-thread score time,
plc holds the GIL (parallel won't help). If wall-clock ≈ 50% of that,
the GIL releases (parallel will help).

**Risk:** plc may hold internal state that's not thread-safe (caches,
last-scored-placement). Need separate `plc` instances per thread, which
costs memory but doesn't change correctness.

**Expected gain:** 2× score throughput → applies to all paths
(2-opt, noise restarts, cong-grad). Highest impact on 2-opt (A1) but
helps Phase 1/3 cong-grad iters too.

---

## B6. Batch `_fast_set_placement` (5ms/call)

**Where:** `placer.py` `_fast_set_placement` (~line 1631).

**What's wrong:** per-macro `set_pos` is the cost. For n=760 macros,
that's 760 Python→C++ calls per score (5ms total).

**How to apply:**
- Check if `plc` has a batched API (e.g., `set_positions` taking an
  N×2 array). If yes, switch.
- If no, check whether `plc_client_os.py` exposes the underlying data
  array directly (some C++ bindings let you write into a buffer).
- Worst case: contribute a batched API to the bridge.

**Expected gain:** 5ms → ~0.5ms (10× speedup on this stage, ~14%
total-score speedup).

---

## B7. Cache score results during 2-opt (cheap insurance)

**Where:** A1's `_two_opt_proxy_swap`.

**What's wrong (or could be improved):** during 2-opt iteration, if
the search loops back to a previously seen placement (e.g., a swap
reverts a prior accept), we rescore. Likely rare but free to guard.

**How to apply:**
- Add an `lru_cache`-style dict keyed by
  `(min(i,j), max(i,j), pos[i].tobytes(), pos[j].tobytes())`.
- Cache stores the proxy result for that specific swap configuration.

**Expected gain:** small (~1-5% of score calls likely repeat) but
deterministic when it does fire.

---

## B8. Adaptive `max_iters` in 2-opt

**Where:** A1's `_two_opt_proxy_swap` (currently `max_iters=3`).

**What's wrong (or could be improved):** if outer iter 1 finds many
accepts (e.g., 100+ on ibm04), iter 2 likely also has high yield
(positions changed substantially). If iter 1 finds few accepts (e.g.,
5 on ibm15), iter 2 will likely find even fewer — wasted time.

**How to apply:**
- After iter 1, measure `accepts_per_score = accept_count /
  score_calls`. If < 0.05, stop. If > 0.15, allow 4-5 outer iters
  (current cap is 3).

**Expected gain:** small. Frees budget on low-yield benchmarks for
other work; allows more accepts on high-yield ones.

---

## B9. Smarter candidate ordering in 2-opt

**Where:** A1's `_two_opt_proxy_swap` (currently lex-ordered by i,j
via kNN order).

**What's wrong (or could be improved):** all candidates are
score-tested in the same order regardless of likely impact. Swaps with
large displacement (likely to change WL/cong significantly) are
intermixed with near-trivial micro-swaps. Higher-impact swaps tested
first → more accepts in fixed budget.

**How to apply:**
- Sort candidate (i, j) pairs by `dx_ij² + dy_ij²` descending before
  scoring.
- Or by `displacement_change_estimate(i, j)` — analytic estimate of
  how much WL changes (cheap to compute given net adjacency).

**Expected gain:** small but free (no extra work, just different
order). Likely improves accept rate by 5-15%.

---

# Part C — Maintenance / blocked

## C1. Pre-existing failing congestion-perturb test (formerly #5)

**Where:** `test/test_varrahan_v2_congestion.py:65`.

**What's wrong:** the test asserts H+V combined perturb behavior, but
`_routing_congestion_perturb` uses `max(H, V)` per a documented A/B
test on ibm03/ibm07. The test failure is unrelated to any recent
change; it was authored when the code briefly used H+V and never
re-synced after the switch.

**Constraint:** `test/` is read-only per CLAUDE.md. Requires user
permission to fix.

**How to apply:** either update the test to construct a placement
where max(H, V) makes a definite-direction prediction, or surface to
the user. The actual code is correct.

---

## Per-call score-cost breakdown (ibm10, after vectorization sweep)

For reference when picking the next speedup target:

| Stage | Cost | Notes |
|---|---|---|
| `_vectorized_get_routing` (congestion) | 17ms | dispatch + lexsort overhead dominates (B4) |
| `get_density_cost` | 4ms | top-10% sort, fully vectorized |
| `_vectorized_get_grid_cells_density` | 4ms | batched bincount |
| `_apply_3pin_routing_vec` | 4ms | batched 4-branch dispatch |
| `_apply_h/v_strips_batch` | 3ms each | difference-array prefix-sum |
| `_apply_2pin_routing` | 1ms | thin wrapper over strip batch |
| `_apply_macro_routing` | 1ms | batched rectangle expansion |
| `_vectorized_wirelength` | <1ms | reduceat per-net HPWL |
| `_fast_set_placement` | 5ms | per-macro `set_pos` (B6) |

Total per-call: ~34ms (from 450ms baseline before vectorization).

After B3 + B4 + B6 (the three concrete speedup targets), expected per-
call: ~10ms (3× total improvement, dominantly from incremental scoring
which targets the 2-opt path where it matters most).

---

## What is NOT in this list (intentionally)

- **Density-score fallback anti-correlation** — known, CLAUDE.md flags
  it, current thresholds handle it correctly.
- **CPU contention slowdown** — environmental, not a code issue.
- **PARTIAL_OVERLAP correction in vectorized macro routing** —
  bit-equivalence verified.
- **`MaskRegulate` comment in PAPERS_NOTES.md** — code is right, doc
  is wrong; PAPERS_NOTES.md is read-only.
- **Dead soft-resnap code** (`_build_soft_resnap_cache`,
  `_resnap_soft_macros`) — ~200 lines, not wired in. Kept per A2 "for
  future exploration"; not a score issue but worth removing if no
  force-directed / quadratic variant is planned.
