---
name: hierarchy-gnn-production
description: Use when implementing, extending, validating, or reviewing the hierarchy-aware GNN subsystem for the macro placement challenge. Enforces stage-gated completion from trace/data through baseline rankers, hetero macro-net GNNs, candidate ranking, expanded hierarchy-flow roles, inference integration, artifacts, and production benchmark promotion.
---

# Hierarchy GNN Production Skill

Use this skill for all work on the hierarchy-aware GNN subsystem.

The GNN must remain inside the existing hierarchy placer. It may rank, propose,
select, budget, and diagnose hierarchy work, but it must not create a separate
placer or bypass legality, fixed-macro immobility, bounds, hierarchy-region,
hierarchy-quality, or exact-proxy gates.

## Required References

Read only what is needed for the current stage:

- Project overview: `docs/ml_nn/gnn/README.md`
- Stage requirements: `docs/ml_nn/gnn/requirements.md`
- Data requirements: `docs/ml_nn/gnn/data-plan.md`
- Evaluation gates: `docs/ml_nn/gnn/evaluation.md`
- Artifact contract: `docs/ml_nn/gnn/artifacts.md`
- Expanded roles: `docs/ml_nn/gnn/expansion-plan.md`
- MacroDiff+ adaptation notes: `docs/ml_nn/gnn/macrodiff_plus_notes.md`
- Trace schema: `docs/ml_nn/beyondppa_results/gnn_trace_schema.md`
- Dataset schema: `docs/ml_nn/beyondppa_results/gnn_dataset_schema.md`

## Non-Negotiable Integration Rules

- Default-off until explicitly promoted.
- GNN outputs are advisory unless the current stage explicitly allows use.
- Existing exact gates remain authoritative.
- No fixed macro may move.
- No hard macro overlap may be accepted.
- Bounds checks remain mandatory.
- Hierarchy-region constraints remain mandatory.
- Hierarchy-quality gates remain mandatory.
- Exact-proxy gates remain mandatory for accepted placement changes.
- Do not reintroduce the deleted proxy-only placement path.
- Do not implement full diffusion coordinate generation as the production path.

## Stage Gate Policy

Do not start a later stage until the current stage has:

- implementation complete;
- focused tests or verifier complete;
- smoke run complete where applicable;
- documentation updated;
- artifact or schema contract updated where applicable;
- results recorded in `docs/general/PROGRESS.md` when behavior or accepted
  subsystem status changes.

If a stage fails its gate, stop and fix that stage. Do not hide failures by
moving complexity into the next stage.

## Stage G1: Trace Completeness

Status: implemented for schema v1.

Completion requirements for future edits:

- Candidate-level traces exist for relocation, swaps, decompression, and
  coldspot tightening.
- Candidate records include accepted flag and rejection reason.
- Trace rows include `schema_version`.
- Trace logging does not change placement output.

Validation:

```bash
uv run python -m py_compile $(find src -type f -name "*.py")
HIER_GNN_TRACE=1 HIER_GNN_TRACE_PATH=/tmp/hier_gnn_trace_smoke.jsonl HIER_GNN_TRACE_MAX_CANDIDATES=5 uv run evaluate src/main.py -b ibm01
```

Check event counts include candidate event families and all rows have the
expected schema version.

## Stage G2: Graph Dataset Builder

Status: implemented for schema v1.

Completion requirements for future edits:

- Builder reads schema-v1 JSONL traces.
- Output contains benchmark graphs and candidate examples.
- Repeated builds from the same trace are tensor-identical.
- Feature schema is written and versioned.

Validation:

```bash
uv run python scripts/gnn/build_gnn_dataset.py --trace-path /tmp/hier_gnn_trace_smoke.jsonl --out /tmp/hier_gnn_dataset.pt --benchmark ibm01
uv run python test/verification/_verify_gnn_dataset_builder.py
```

## Stage G3: Baseline Non-GNN Ranker

Goal: prove the labels are learnable before graph-model work.

Required implementation:

- Train/eval script for candidate-feature-only baselines.
- At least one simple classifier/ranker.
- Benchmark-level train/validation split.
- Metrics by operator and benchmark.

Required metrics:

- top-k accepted-candidate recall;
- mean reciprocal rank;
- accept/reject quality when class balance allows;
- proxy-delta correlation for known deltas;
- class balance and accepted-rate per operator.

Gate:

- Baseline beats heuristic ordering on held-out traces for at least one
  operator.
- If not, improve traces, features, or labels before G4.

## Stage G4: Hetero Macro-Net GNN

Goal: score candidates using graph context plus candidate features.

Required design:

- Use MacroDiff+ only as graph-design inspiration.
- Extend graph data toward macro nodes, net nodes, macro-net edges, pin offsets,
  net degree, and dynamic net HPWL/wirelength pressure.
- Keep model small: 2-3 message-passing layers, hidden size 32 or 64.
- CPU inference must be viable.

Required outputs:

- ranking score;
- proxy-delta estimate when labels exist;
- accept probability;
- hierarchy-quality delta estimate when labels exist;
- rejection-reason probabilities when labels exist.

Gate:

- Offline top-k recall improves over G3.
- Runtime on `ibm10` candidate pools fits the hierarchy pass budget.

## Stage G5: Default-Off Inference Integration

Goal: use the model inside existing operators.

Required controls:

```bash
HIER_GNN_RANK=0
HIER_GNN_MODEL=ml_data/beyondppa_gnn/model.pt
HIER_GNN_TOP_K=32
HIER_GNN_OPERATORS=relocation
```

Initial target:

- Hard post-swap propose-all relocation only.

Gate:

```bash
uv run evaluate src/main.py -b ibm10
HIER_GNN_RANK=1 uv run evaluate src/main.py -b ibm10
```

Both runs must be valid. The GNN run must not materially slow or regress the
baseline before expanding to more benchmarks.

## Expanded Roles After Candidate Ranking

Only start these after G5 candidate ranking is validated.

Recommended order:

1. Candidate pool expansion.
2. Operator selection in advisory/logging mode.
3. Acceptance-risk surrogate for scoring prioritization.
4. Region guidance.
5. Soft macro role guidance.
6. Budget allocation.
7. Diagnostics and benchmark triage.

Each role must be:

- independently default-off;
- separately traced;
- separately evaluated;
- closed-loop validated before promotion.

## Stage G6: Closed-Loop Validation

Required sequence:

```bash
HIER_GNN_RANK=1 uv run evaluate src/main.py -b ibm10
HIER_GNN_RANK=1 uv run evaluate src/main.py -b ibm01
HIER_GNN_RANK=1 uv run evaluate src/main.py -b ibm12
HIER_GNN_RANK=1 uv run evaluate src/main.py -b ibm17
HIER_GNN_RANK=1 uv run evaluate src/main.py --all
```

Promotion requires:

- 17/17 VALID;
- 0 overlaps;
- runtime comfortably below 1 hour total;
- average proxy improvement or documented hierarchy-quality improvement;
- no severe single-benchmark regression without a documented reason.

## Stage G7: Artifacts

Every trained model or promoted baseline needs:

```text
feature_schema.json
train_config.json
splits.json
trace_manifest.json
metrics.json
model.pt
README.md
```

Store under:

```text
ml_data/beyondppa_gnn/models/<model_id>/
```

The artifact must include trace IDs, benchmark split, feature schema version,
model config, offline metrics, inference benchmark results, and code/worktree
fingerprint.

## Documentation Updates

When changing GNN behavior, update relevant docs:

- `docs/ml_nn/gnn/*`
- `docs/ml_nn/beyondppa_results/gnn_*`
- `docs/general/PROGRESS.md`
- `docs/general/ARCHITECTURE.md` if runtime architecture changes
- `docs/general/DESIGN_FLOW.md` if hierarchy flow changes
- `docs/general/ISSUES.md` if status or risks change

## Final Review Checklist

Before marking work complete:

- Stage gate satisfied.
- Tests/verifiers run and results reported.
- No production default changed without benchmark evidence.
- No placement gates bypassed.
- No stale docs contradict current code.
- Worktree changes are scoped to allowed files.
