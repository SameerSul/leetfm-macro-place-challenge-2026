# BeyondPPA-Style Structural Objectives

This note defines a deterministic, staged path for adding BeyondPPA-style
structural guidance to the active hierarchy placer. It intentionally does not
start with a GNN or DQN implementation. The current production system is a
hierarchy-preserving flow, and every new structural objective must first prove
that it adds useful signal without disturbing legality, fixed macros, region
constraints, or exact proxy gates.

## Repo-Specific Summary

BeyondPPA uses learned structural feedback to guide macro placement toward
layouts that are easier to route and optimize. In this repository, the useful
near-term part is not the model architecture. It is the structural objective:
prefer placements with reasonable edge spacing, regular grid alignment, and
fewer small unusable notches between large blocks.

The active placer already has much of the substrate needed for this:

- Grouped DREAMPlace creates a hierarchy-aware initial floorplan.
- Hard clustering and soft ownership keep connected subsystems together.
- Density and congestion fields identify local pressure.
- Region-bounded swaps, relocation, decompression, coldspot tightening, and
  micro-shift passes already keep legality and exact proxy gates mandatory.

The first implementation therefore adds deterministic metrics and opt-in
ranking hooks. Learned policies can be added later only if these metrics show
measurable value.

## Already Present

- `src/placer/pipeline/macro_placer.py` always uses the hierarchy floorplan path.
- Grouped DREAMPlace is required; the deleted proxy fallback is not part of this
  work.
- `src/placer/local_search/fields.py` exposes congestion and density fields used
  by hard and soft relocation.
- `src/placer/local_search/relocation.py` has exact-gated hard relocation, soft
  relocation, and micro-shift polishing.
- `src/placer/local_search/hierarchy_swaps.py` has region-bounded hard/hard,
  hard/soft, and soft/soft swaps.
- `src/placer/local_search/cluster_decompress.py` tracks hierarchy quality for
  exact-gated cluster relief.

## Deferred

- No learned GNN model is active in this staged work. The only GNN-related code
  currently shipped is opt-in trace logging for future training data.
- No DQN or reinforcement learning loop is added.
- No macro rotation support is added. The competition API and current legalizers
  assume the provided macro sizes.
- No production default changes are made until full benchmark evidence justifies
  promotion.

## Staged Rollout

### Stage 1: Documentation And Metrics

Add pure metric helpers in `src/placer/local_search/structural_fields.py`:

- `edge_keepout_penalty`
- `grid_alignment_penalty`
- `notch_penalty`
- `combined_structural_penalty`

These helpers are deterministic and independent of the production pipeline.
Unit tests use synthetic placements with clear better/worse examples.

Acceptance gate: focused unit tests pass, `py_compile` passes, and existing
smoke tests remain green.

### Stage 2: Diagnostic Reporting

Add `test/diagnostic/_structural_metrics.py` to report structural scores for a
final placement. The diagnostic can run the current placer or report the initial
placement, but it must not alter placement output.

Initial calibration targets:

```bash
uv run python test/diagnostic/_structural_metrics.py ibm10
uv run python test/diagnostic/_structural_metrics.py ibm01
uv run python test/diagnostic/_structural_metrics.py ibm17
```

Acceptance gate: metrics are stable, interpretable, and cheap enough for local
diagnostics.

### Stage 3: Hierarchy Candidate Ranking Integration

Add a BeyondPPA-style structural term inside the existing hard and soft
hierarchy relocation candidate ordering. This is not a second placement path.
The structural score affects proposal ordering only. Existing legality checks,
fixed-macro rules, region constraints, and exact proxy accept gates remain the
authority.

Constants in `src/utils/constants.py`:

```bash
HIER_OBJECTIVE_STRUCTURAL_WEIGHT=0.0
HIER_KEEP_OUT_WEIGHT=0.2
HIER_GRID_ALIGN_WEIGHT=0.2
HIER_NOTCH_WEIGHT=0.6
```

Acceptance gate: `HIER_OBJECTIVE_STRUCTURAL_WEIGHT=0.0` is behaviorally
unchanged. Opt-in runs must remain valid on `ibm10` before broader testing.

### Stage 4: Exact-Gated Hierarchy Polish Integration

Do not add a separate structural polish pass. The exact-gated hierarchy polish
already exists as micro-shift, relocation, swaps, decompression, and coldspot
tightening. BeyondPPA-style structure integrates by influencing candidate
ordering before those existing exact gates.

Acceptance gate: opt-in polish keeps hard legality, fixed macro immobility,
bounds, and hierarchy-region constraints mandatory.

### Stage 5: Bounded Structural Acceptance

Bounded structural acceptance is deferred. It would change the hierarchy flow's
acceptance semantics by allowing exact proxy regressions, so it should be added
only after candidate ordering shows value across multiple benchmarks. If added
later, it must be part of the hierarchy accept gate itself, not a separate
post-processing path.

## Verification Plan

After each code stage:

```bash
uv run python -m py_compile $(find src -type f -name "*.py")
uv run pytest test/
```

Current caveat: the full `uv run pytest test/` suite has an unrelated eda_io
CLI failure because that fixture still exercises the removed proxy fallback.
Use focused structural tests plus placement smokes until the eda_io fixture is
updated for hierarchy-only DREAMPlace requirements.

Focused Stage 1 check:

```bash
uv run pytest test/verification/test_structural_fields.py
```

Stage 2 diagnostics:

```bash
uv run python test/diagnostic/_structural_metrics.py ibm10
uv run python test/diagnostic/_structural_metrics.py ibm01
uv run python test/diagnostic/_structural_metrics.py ibm17
```

Stage 3 and later:

```bash
uv run evaluate src/main.py -b ibm10
HIER_OBJECTIVE_STRUCTURAL_WEIGHT=1 uv run evaluate src/main.py -b ibm10
```

Only run `--all` after `ibm01`, `ibm10`, and `ibm17` justify the extra runtime.
Record accepted benchmark results in `docs/general/PROGRESS.md`.

## GNN Trace Logging

Future GNN work should use opt-in hierarchy traces rather than changing the
placement path first:

```bash
HIER_GNN_TRACE=1 HIER_GNN_TRACE_RUN=ibm10_smoke uv run evaluate src/main.py -b ibm10
```

Trace files are JSONL under `ml_data/beyondppa_gnn/` by default. They include
hierarchy relocation candidates, accepted move labels, pass summaries, and final
benchmark summaries.

The full GNN implementation roadmap is documented in
`docs/ml_nn/beyondppa_results/gnn_full_implementation_next_steps.md`.
