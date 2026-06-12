# ML notes — learned candidate ranker

Conceptual / "why it works" notes for the per-operator XGBoost candidate ranker.
This is the explanatory companion to:

- **`../general/ISSUES.md` S10** — the terse experiment-tracker entry (status, plan,
  dataset counts).
- **top-level `README.md` → "ML candidate-ranker data collection"** — how to run
  the collection script.

**Status (2026-06-05): `hard_relocation` filter wired (opt-in) and compared at
equal budget — comparable-or-better than the exhaustive interleave (net ~−0.008
over 10 benchmarks, no robust regression). Production default unchanged.** Key
finding: the ranker is not the bottleneck (`best_recall@16` ≈ 1.0), so retraining
has no headroom; the leverage is integration policy and the routing-fill cost. See
`../general/ISSUES.md` S10 for the numbers.

## Notes

| File | Question it answers |
|---|---|
| [`01-candidate-ranker-design.md`](01-candidate-ranker-design.md) | What does XGBoost replace, and how is it wired so the search stays non-regressing? |
| [`02-why-it-can-improve.md`](02-why-it-can-improve.md) | If it's trained on our own placer, how can its placements beat our placer? |
| [`03-selection-mechanism.md`](03-selection-mechanism.md) | Concretely, what does the model select, and how does that differ from what the placer does today? |
| [`04-gnn-routing-fill-surrogate.md`](04-gnn-routing-fill-surrogate.md) | Roadmap to replace the ~73% strip-gen cost with a learned Δ-congestion prefilter: what to add, what to restructure, and the decision gates. |
