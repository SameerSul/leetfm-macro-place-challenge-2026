# ML notes — learned candidate ranker

Conceptual / "why it works" notes for the per-operator XGBoost candidate ranker.
This is the explanatory companion to:

- **`docs/ISSUES.md` S10** — the terse experiment-tracker entry (status, plan,
  dataset counts).
- **top-level `README.md` → "ML candidate-ranker data collection"** — how to run
  the collection script.

**Status (2026-06-04): data collected, ranker not yet wired into the placer.**
The R2 local search still exact-scores every candidate. These notes describe the
design and the reasoning behind it, not shipped behavior.

## Notes

| File | Question it answers |
|---|---|
| [`01-candidate-ranker-design.md`](01-candidate-ranker-design.md) | What does XGBoost replace, and how is it wired so the search stays non-regressing? |
| [`02-why-it-can-improve.md`](02-why-it-can-improve.md) | If it's trained on our own placer, how can its placements beat our placer? |
| [`03-selection-mechanism.md`](03-selection-mechanism.md) | Concretely, what does the model select, and how does that differ from what the placer does today? |
