# What the model actually selects

## Predict, don't evaluate

A precise distinction: the model does **not** evaluate (score) candidates. It
**predicts, from cheap features, which candidates are worth scoring** — without
touching the exact or incremental scorer. The actual evaluation (the real proxy
score) runs only on the ones it picks, and the exact gate still decides
accept/reject. The model is triage; the scorer is the judge.

## Two levels of selection

There are two ranking decisions, not one — and they have different leverage:

1. **Which macros to attempt** (across-group). Pick the N most promising *source*
   macros to even try moving / swapping. **Higher leverage**, because attempting
   a macro costs the expensive `_prepare_move` (the routing-apply) regardless of
   how many targets are then tried. Gating this gates the dominant per-group cost.
2. **Which targets / partners per macro** (within-group). For a chosen macro,
   score only the top-K candidate *destinations* (relocation cells) or *swap
   partners* (2-opt neighbours), instead of all of them. Lower leverage, because
   each trial is already cheap once `prep` is paid — but free to add.

A learned ranker can drive both (the group gater and the target ranker in
[`01-candidate-ranker-design.md`](01-candidate-ranker-design.md)).

## It replaces a top-N heuristic that already exists

The placer **already** keeps "only the N best": `top_hot` (e.g. 48 hottest
macros) × `n_targets` (e.g. 16 nearest cells). It already discards most
candidates. But today "best" means a **hand-coded heuristic** — sort by local
congestion, pick the nearest cold cells. The model doesn't introduce the top-N
idea; it replaces the *heuristic that chooses the N* with a *learned* ranking
trained on which candidates actually improved the proxy. So either:

- **same N, smarter N** — the N you keep are the genuinely productive ones (more
  accepts per round), or
- **smaller N, same accepts** — fewer exact-scores to find the same wins (freed
  budget → more rounds, see [`02-why-it-can-improve.md`](02-why-it-can-improve.md)).

## Per operator

It's per-operator: separate rankers for **hard relocation**, **soft relocation**,
and **hard 2-opt** — relocations and swaps, each with its own model.

## In one sentence

The model predicts which macros (and which targets / partners) are worth
exact-scoring, so we score only the top-ranked few instead of the heuristic's
top-N — with the exact gate still making every accept decision.
