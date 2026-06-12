# Candidate-ranker design

## Goal

The R2 local search is **deadline-bound** on the large benchmarks
(ibm12/14/15/17/18): exact scoring is slow, so the search gets few rounds and
evaluates only a fraction of the candidate space before the budget runs out. The
ranker's job is to spend that fixed scoring budget on the candidates most likely
to improve the proxy — without changing the accept decision.

## What XGBoost can and can't replace

XGBoost is a scalar predictor over tabular features. It **cannot** emit
placements, generate moves, or run the legalizer — so it can't replace the
loop's machinery. It can only replace the **decision logic** inside the loop, and
there's a fork in how:

1. **Ranker / filter in front of the exact gate (chosen).** The model predicts,
   from cheap pre-score features, which candidates are worth exact-scoring. We
   score only those; the exact accept-on-true-proxy gate still decides. The
   search stays **strictly non-regressing** — the model only reorders/prunes what
   gets scored, it never accepts a move.
2. **Surrogate scorer replacing the gate (rejected).** Accept on the model's
   predicted delta, skipping the exact score. Faster per move, but every false
   positive commits a real regression — it breaks the non-regression guarantee
   the whole score depends on. Not pursued.
3. **Meta-controller (smaller, separate).** Predict round-level "stop?" or
   operator selection instead of per-candidate decisions. Lower leverage.

We build **(1)**. The exact scorer remains the oracle; the model is cheap triage.

## Per-operator models

Separate models for the three dominant operators — **hard relocation**, **soft
relocation**, **hard 2-opt** (relocations + swaps). The cong/density `field` is a
**feature**, not a separate model (the candidate structure is identical across
fields, so sharing triples the data per model). The remaining passes (HXS, HS3,
soft-2opt) keep the exhaustive path for now.

Two heads per operator, both trainable from the same trace:

- **Group gater** (across-group): `binary:logistic` on "did this macro's group
  contain any improving move?" → which macros to attempt at all.
- **Target ranker** (within-group): LambdaMART (`rank:pairwise` / `rank:ndcg`),
  query groups = `group_id`, labels = `dataset.add_group_relevance` → which
  targets / swap-partners to score for a chosen macro.

## Features and labels

- **Features must be pre-score and cheap** — computable before the expensive
  evaluation, or the model needs the very thing it replaces. `CandidateTrace`
  already records the right ones: net degree, source/target congestion &
  density, displacement, hot/cold rank, macro size, position.
- **Labels are the exact `score_gain` / `improves`** — ground truth from the
  proxy itself, which is why "trained on our own runs" is fine (see
  [`02-why-it-can-improve.md`](02-why-it-can-improve.md)).
- **Do not feed `benchmark.name`** — leakage, and the rules forbid per-benchmark
  branching. Shape features (`num_hard`, `grid_*`, util) are how it generalizes.

## Integration sketch (when wired)

Per pass, with the model present: build a **vectorized** feature matrix for all
candidates (one matrix, not the per-candidate dicts the tracer uses — those are
offline-only and too slow for inference), one batched `predict`, gate groups +
take top-K targets, exact-score only the survivors through the existing
`prep` / `_trial_at` / commit path, accept on the exact gate. Model absent → the
current exhaustive path (fallback). Export the boosters to a numpy / treelite
evaluator so `xgboost` isn't imported on the submission's inference path, and so
predict latency stays well under one incremental score.

## Validation

- Train per operator with **whole-benchmark holdout** (never random rows —
  within-group rows are correlated and random splits overstate accuracy).
- Pick K from an offline **recall@K of the true-best target per group** curve
  *before* any `--all`.
- Confirm online: `--all` with the model, no tracing, **per-benchmark
  non-regression**, watching that freed budget beats predict overhead on the
  large benchmarks. Stratify the holdout across IBM **and** NG45.
