# GNN routing-fill surrogate — implementation roadmap

## 0. What this is

A step-by-step plan to take the "GNN as an ultra-fast surrogate for routing fill"
idea from concept to either a shipped feature or a justified shelving. It records
every component to **add** and every existing piece to **restructure**, plus the
decision gates that stop the work early if the premise fails.

Read [`01-candidate-ranker-design.md`](01-candidate-ranker-design.md) and
[`02-why-it-can-improve.md`](02-why-it-can-improve.md) first — this builds on the
same "exact gate stays the oracle" principle. The bottleneck numbers come from
`../general/ARCHITECTURE.md` §5.3 (vectorized routing fill) and §6.2 (S2, the GPU-batch
attempt that lost).

## 1. The premise, in one paragraph

Per-candidate **strip generation** (routing bucketing, looped-K on CPU) is
**~4.9 ms/macro ≈ 73 %** of a trial's cost (`ARCHITECTURE.md` §6.2). The
congestion/density math on top is ~2 %. S2 tried to keep the *exact* strip-gen and
move the reductions to GPU; it lost (0.67–0.97×) because the CPU strip loop
dominates and IBM grids are too small to amortize launch overhead. This idea
*replaces* strip-gen with a learned Δ-congestion predictor cheap enough to batch
1000+ candidates, flags the top few, and lets the exact scorer verify only those —
keeping the non-regression guarantee. The doc's own "what could still win" is the
skeleton: cross-macro batching of `top_hot × n_targets ≈ 1000+` candidates.

## 2. The constraints that shape every step

These are non-negotiable; any step that violates one is wrong.

1. **The exact gate stays the arbiter.** The surrogate only chooses what gets
   exact-scored; every *accept* is still a strict true-proxy drop. The only failure
   mode is under-improvement (missed move), never regression.
2. **Inference must clear the CPU bar or run batched on GPU.** A single exact trial
   is ~0.4 ms. A per-candidate CPU GNN forward pass loses; the only viable form is
   one batched GPU op over a large candidate set, per pass (not per candidate).
3. **No per-benchmark branching** (`benchmark.name`), no external proprietary
   tools, `< 1 h` for all 17 IBM, zero hard-macro overlaps.
4. **Do not modify the tracer's hot path to change scores.** Enabling tracing
   already perturbs timing-gated control flow (see the ml-trace gotcha in
   `../general/ISSUES.md`). Add fields behind the existing `if trace is not None` guards
   only; verify any change with the arithmetic-equivalence test, not end-to-end
   score identity.
5. **Write scope:** everything lives under the root active submission tree
   (`src/placer/ml/`, `src/placer/local_search/`, `test/diagnostic/`, `ml_data/`).
6. **Do not overshadow the shipped S11 scoring-cost cuts.** The validated default
   (`--all` **1.1423**) is the *sequential* prep→trial path carrying the WL-delta
   prefilters (`soft_relocation` 1e-4 via `wl_delta_move_soft`, `soft_2opt` 3e-4,
   `hard_2opt` k=16). The Phase-C *propose-all* / CUDA-batch path replaces that loop
   and so **bypasses these prefilters**. Therefore:
   - The propose-all / GPU relocation path must stay **opt-in** (`V2_RELOC_PROPOSE_ALL`,
     default off) until it is proven to **beat the prefiltered CPU default on the
     deadline-bound IBM benchmarks** — multi-seed, per-benchmark non-regression vs
     1.1423. "Faster in isolation" is not enough; S2 showed GPU batching *loses* on
     the small IBM grids, so the CPU-prefiltered path is expected to remain default
     for the IBM submission, with the GPU path winning only on NG45/large grids.
   - When propose-all is generalized to **soft** relocation (the score MVP), the
     batched path must reproduce the prefilter's *selection effect* or re-validate
     the headline — do not let a batched soft path silently drop the soft prefilter.
   - Currently safe: `_soft_relocation_moves` has **no** propose-all branch (the
     CUDA work is hard-relocation only), so the soft prefilter is always active on
     the default path. Keep it that way unless the above is satisfied.

## 3. Decision gates (stop here if a gate fails)

Build in this order; a failed gate kills or reroutes the work **before** the
expensive parts.

- **Gate A — recall survives width.** A cheap surrogate must keep the true-best
  improving move inside a small top-K as the pool widens. If recall@K collapses
  with width, a 1000-wide prefilter is dead. *(In progress — Phase A.)*
- **Gate B — the pool actually gets wide.** If the *legal* pool on IBM saturates at
  tens (early data: median 14–16, max ~49 even at `N_TARGETS=64`), then 1000-wide
  is only reachable **cross-macro**, which forces the algorithm restructuring in
  Phase C. This decides whether the whole thing is an IBM play or an NG45 play.
- **Gate C — the exact kernel can't just be vectorized.** Before approximating,
  confirm the exact strip-gen loop genuinely can't be batched to throughput
  (Phase B). If it can, do that instead — exact, no recall risk, no training.
- **Gate D — online non-regression.** Even with good offline recall, the restructured
  search must not regress per-benchmark under the same wall-clock budget.

## 4. Phase A — Feasibility: recall@K vs pool width *(in progress)*

**Goal:** the kill-switch curve, offline, before any model is built.

**Already added:**

- `test/diagnostic/_recall_at_width.py` — per (benchmark, width): legal-pool-width
  distribution + `improving_recall@K` + `regret@K`, using the deployed ranker.
- Wide-pool data generation: re-run with `ML_HARD_RELOCATION_N_TARGETS={64,128,256}`,
  tracer on, filter off, into `ml_data/recall_study/` (so every legal candidate is
  exact-labelled with `score_gain`).

**To finish Phase A:**

1. Run the analyzer across all widths/benchmarks; read the recall@K-vs-width curve.
2. Add a **fresh-trained** surrogate with whole-benchmark holdout (not just the
   deployed 32-wide model) so recall isn't optimistic — reuse `placer.ml.train`
   with `--test-benchmarks`. This bounds what an XGBoost surrogate can do; a GNN
   must beat it to be worth the cost.
3. Record the curve + the legal-width distribution in `../general/ISSUES.md` S10.

**Exit criteria:** a recall@K(width) table + the legal-width distribution. Decides
Gate A and Gate B.

### 4.1 Phase A — Results (2026-06-05, ibm13 honest holdout)

Data: `ml_data/recall_study/{ibm10,ibm13}_w{64,128,256}.jsonl.gz`. Surrogates: the
deployed 32-wide ranker **and** a fresh ranker trained *including* wide pools
(`ml_data/models/wide-surrogate-holdout-ibm13`). Analyzer:
`test/diagnostic/_recall_at_width.py`.

**Gate B — FAILED for within-macro widening.** The *legal* pool saturates far below
the request: at `N_TARGETS=256`, improving-group width is median **94 / max 168**
(ibm13) and **44 / 145** (ibm10). Overlap rejection + cold-cell availability cap it.
**1000-wide is unreachable within a macro → only cross-macro (Phase C).** The cheap
"just widen `N_TARGETS`" path tops out at low hundreds.

**Gate A — recall collapses with width, and it is fundamental, not OOD.**
`improving_recall@5` on held-out ibm13: w64 (pool 18) **0.78** → w128 (45) **0.67**
→ w256 (94) **0.36**. The fresh wide-pool-trained surrogate is **identical** to the
32-wide model (0.36 vs 0.33 at width-94) — *training on wide pools does not recover
recall.* So a cheap tabular surrogate cannot reliably put the literal best move in a
small top-K once the pool is wide. The "evaluate 1000, verify top-5" architecture
needs recall@5 ≈ 1.0; we get 0.36.

**But recall@K is the wrong metric — gain-regret is benign.** Gains are densely
near-tied, so missing the literal #1 costs little. As a fraction of mean achievable
gain, the fresh surrogate's **regret@10 is 1.1 % / 2.4 % / 5.3 %** at width
18/45/94, and regret@5 is 2.1 % / 6.6 % / 15.5 %. So a surrogate scoring **top-10**
captures ≥95 % of achievable hard-relocation gain at every width, cutting exact
scores up to 9.4× (94→10).

**And the operator is low-leverage.** Hard-relocation gains are tiny (mean
**1.3e-4**/group, median 3.5e-5) — a small slice of the total proxy.

**Verdict.** A cheap **XGBoost** surrogate already triages wide pools at acceptable
*gain-regret* (top-10 → ~95 %), so **ranking quality is not the bottleneck** — the
GNN's only edge (better recall@5 → a smaller K) targets a metric that barely
matters here, on a low-gain operator. Therefore **do not build the GNN for
IBM/hard-relocation.** The 73 % strip-gen cost is better attacked by **Phase B
(vectorize the exact kernel cross-macro)** — exact, no recall risk, and it applies
to *all* operators (incl. the dominant soft passes), not just this one. The GNN
earns its keep only if **all** of: Phase B fails to vectorize, the Phase C
restructuring lands, on **NG45/large grids**, generalized to soft operators. Until
then it is shelved in favour of the exact-kernel route. (Re-run on NG45 before
fully closing — the width/gain story may differ on large industrial grids.)

### 4.2 Phase A — NG45 re-check (2026-06-05)

Re-ran the study on the four NG45 designs (ariane133/136, mempool_tile, nvdla —
**more macros** 581–915, **smaller/denser grids** 504–1404 cells, ~3× the nets/cell
of IBM). Data: `ml_data/recall_study_ng45/*_w{64,256}.jsonl.gz`; surrogate = the
IBM-trained deployed ranker (cross-design transfer test).

**Two decisive findings, both reinforcing the shelve verdict:**

1. **NG45 is not deadline-bound — it converges with ~40 % budget to spare**
   (budget 150 s, elapsed 90–97 s, `reason=completed`). This is the regime
   [`02-why-it-can-improve.md`](02-why-it-can-improve.md) calls downside-only: with
   spare budget the filter has nothing to reclaim and can only under-improve. The
   whole "free budget → more rounds" premise is void here.
2. **hard-relocation is near-idle on NG45** — only **1.9–3.0 %** of groups have any
   improving move (vs 20–25 % on IBM). The OpenROAD initial placement already seats
   the hard macros. The productive operators are **soft_2opt (34 %)** and
   soft_relocation (13 %); hard_2opt 8.8 %.

**Width is mixed and moot.** Some designs reach wide within-macro pools
(mempool_tile median 116 / **max 203**, wider than IBM); the densest (ariane133,
915 macros / 504 cells) stays narrow (11/12). Recall numbers are statistically
unreliable (6–13 improving hard-reloc groups per design), so no recall conclusion
is drawn — findings #1 and #2 settle it regardless.

**Caveat on the "real designs" argument.** This challenge's NG45 tier does **not**
reach the large-grid, deadline-bound regime where a learned routing-fill surrogate
would pay off. These are real netlists but *coarse macro-placement grids*
(504–1404 cells) the CPU search handles trivially. A genuinely industrial block
(thousands × thousands of gcells, exact score in seconds, search deadline-bound)
would still flip the economics — neither tier of this contest exercises that scale.
**Useful side-finding:** if learned effort goes anywhere on NG45, it is **soft_2opt**,
not hard relocation.

## 5. Phase B — Exact-kernel race (the no-ML competitor)

**Goal:** prove the GNN is *necessary*, not just sufficient.

**To add:**

1. A profiling spike under `test/diagnostic/` that isolates the strip-gen loop
   (`_apply_3pin_routing_vec` / `_apply_h_v_strips_batch` per `ARCHITECTURE.md`
   §5.3) and measures whether net classification + L/Steiner segment construction
   vectorizes across **all hot macros' candidates at once** (the cross-macro batch),
   producing bit-identical strips.
2. If it vectorizes: a batched exact strip-gen is the win — exact, no model. Pursue
   that and **stop the GNN track** (Gate C fails for the GNN).

**Why first:** the strip-gen is "routing bucketing, looped K on CPU" — the *fill*
(difference-array + cumsum) is already vectorized; only the segment construction
loops. If that loop flattens, you get the 1000-batch win with zero recall risk.

## 6. Phase C — Cross-macro evaluate-all-then-commit restructuring

This is the real cost of the idea, not the GNN. Required because 1000-wide is
inherently cross-macro (Gate B).

**To restructure (`src/placer/local_search/relocation.py`, `_relocation_moves`):**

- Current shape is **sequential greedy**: for each hot macro, prep → trial its
  targets → commit the best improving one → the commit mutates the state the next
  macro sees.
- New shape is **propose-all → verify-and-commit-serially**:
  1. Freeze a base state; gather candidates for **all** `top_hot` macros against it.
  2. One batched surrogate call ranks the whole pool (1000+).
  3. Take the global top-M proposals; **exact-verify them sequentially**, each
     re-scored against the *current* (post-commit) state, committing only strict
     improvements. Conflicts (two macros want the same cell; A's move changes B's
     Δ) are resolved by the serial re-verify — a proposal ranked against the stale
     base that no longer improves is simply dropped.
- **Keep the legacy sequential path** behind a flag; the restructured path is a
  *different algorithm* and must be A/B'd, not assumed equivalent.

**To add:**

- A bit-exact verifier (mirror the S2 verifiers): the serial-commit result must
  equal what the sequential greedy loop would have produced **when the surrogate is
  replaced by the exact scorer** (i.e. M = full pool). This isolates "did the
  restructuring change the math" from "did the surrogate change the choices".

**Risk:** this changes accept semantics (`ARCHITECTURE.md` §6.2 calls it out
explicitly). Re-validate every benchmark from scratch; do not trust the 1.15 carry
over.

## 7. Phase D — The surrogate model

**Prediction target.** Not raw congestion — the **localized Δ** of a move. The S2
work proved congestion = shared base (macro removed) + a localized, batchable delta
(linear box-filter smoothing, verified bit-exact Δ ≤ 6.7e-16). Predict that delta,
or directly `score_gain`, per candidate.

**Inputs (must condition on current fill, not just the netlist):**

- Netlist-local subgraph around the macro (nets as hyperedges, pins, macro sizes).
- **The current local H/V congestion patch at the source and target** — without it
  the model can't tell a cold destination from an already-saturated one. This is
  the single most important feature and the reason a pure static-graph GNN is
  insufficient.
- Cheap pre-score features already traced (`TraceFields`): displacement, hot/cold
  rank, net degree, source/target congestion & density.

**To add:**

- `src/placer/ml/data_collection.py`: optionally log the **H/V congestion delta**
  per candidate (behind the `if trace is not None` guard; verify via the
  arithmetic-equivalence test, never end-to-end). Only needed if predicting the
  congestion delta rather than `score_gain`.
- A DAgger loop: train v0 on exhaustive/wide traces → collect traces from the states
  the restructured search induces → retrain on the union (the propose-all path
  visits a different state distribution than the sequential one).
- Model code: extend `src/placer/ml/modeling.py` with a GNN backend behind the same
  `ModelSpec` / `CandidateRanker` interface so the integration surface
  (`filter_candidate_indices`) is unchanged. Export to a flat numpy/treelite/ONNX
  evaluator so `torch`/GNN libs aren't on the submission's inference import path.

**Recall bar:** the surrogate must keep the true-best inside a small top-M out of
1000 — a far harder bar than today's recall@16-of-7. Measure with the Phase A
harness on wide pools before wiring anything live.

## 8. Phase E — Integration + validation

**To restructure (`src/placer/ml/shadow.py`):**

- Generalize `filter_candidate_indices` from per-group (per-macro) to a
  **cross-macro batch** entry point that takes the whole pool, returns the global
  top-M indices. Keep the per-operator manifest, the missing-model fallback, and the
  calibration/guard rails.

**To add (`src/placer/pipeline/macro_placer.py`):**

- A `_USE_GPU`-gated branch in the R2 hard-relocation step that calls the
  restructured propose-all path when a model + GPU are present, else the legacy
  sequential path. Reuse `_ml_r2_context` to stamp trace context.

**Validation (Gate D):**

1. Offline: recall@M / regret@M on held-out benchmarks (IBM **and** NG45).
2. Online: `--all` with the model, no tracing, **per-benchmark non-regression**,
   under the same wall-clock budget. Multi-seed — single-benchmark scores swing
   ±0.005–0.01 from timing alone (ml-trace gotcha), so a one-seed delta is noise.
3. Confirm the freed strip-gen budget actually converts to more rounds / wider
   pools on the budget-bound benchmarks (the only place it can win).

## 9. Inventory — add vs restructure

**Add:**

- `test/diagnostic/_recall_at_width.py` *(done)* — recall vs width.
- `test/diagnostic/_strip_batch_vectorize_spike.py` — Phase B exact-kernel race.
- `test/verification/_test_propose_all_bitexact.py` — Phase C equivalence verifier.
- GNN backend in `src/placer/ml/modeling.py` + flat-evaluator export.
- Optional H/V-delta logging in `src/placer/ml/data_collection.py`.
- DAgger collection/retrain scripts (extend `placer.ml.train`).

**Restructure:**

- `relocation.py::_relocation_moves` — sequential greedy → propose-all /
  serial-verify-commit (flagged, legacy path retained).
- `shadow.py::filter_candidate_indices` — per-group → cross-macro batch entry.
- `macro_placer.py` R2 hard-relocation step — `_USE_GPU`-gated propose-all branch.

## 10. Scope note — this is likely an NG45 play

`ARCHITECTURE.md` §6.2 is explicit: IBM grids (~2000 cells, K≈24/macro) are too
small to amortize GPU launch; the batch win lands on **much larger grids
(industrial / NG45)**. Combined with Gate B (IBM legal pools saturate at tens),
expect this to **not move the 1.15 IBM average** and to be the lever for the Tier-2
commercial designs instead. Validate on NG45 first; treat IBM as the
non-regression guard, not the target.

## 11. Honest kill criteria

Shelve the GNN (and prefer the exact-kernel route or nothing) if **any** hold:

- Phase A: recall@M-of-1000 from a trained surrogate is materially below 1.0 on the
  held-out benchmark (drops real moves the gate never sees).
- Phase B: the exact strip-gen vectorizes to the cross-macro batch (do that — exact
  beats approximate).
- Phase C: the propose-all restructuring regresses per-benchmark even with the exact
  scorer as M=full (the algorithm change itself hurts).
- Phase E: the freed budget doesn't convert to a measurable, multi-seed per-benchmark
  win on the budget-bound / NG45 designs.
