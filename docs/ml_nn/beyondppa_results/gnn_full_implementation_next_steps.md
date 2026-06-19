# Full GNN Implementation Next Steps

This implementation record describes how to move from opt-in GNN trace logging
to a hierarchy-aware GNN subsystem. The first production role is candidate
ranking, but the broader target is a default-off hierarchy-flow assistant that
can rank, propose, select, budget, and diagnose work inside existing hierarchy
operators. It must not create a second placement path or bypass legality,
fixed-macro, bounds, hierarchy-region, hierarchy-quality, or exact-proxy gates.

The dedicated project plan lives in `docs/ml_nn/gnn/`.

## Current State

Already implemented:

- Deterministic structural metrics:
  - edge keepout
  - grid alignment
  - notch penalty
  - combined structural penalty
- Hierarchy-integrated structural candidate ordering controlled by
  `HIER_OBJECTIVE_STRUCTURAL_WEIGHT`.
- Opt-in JSONL trace logging controlled by the `HIER_GNN_TRACE*` runtime
  environment variables.
- Trace events for relocation candidates, relocation results, hierarchy pass
  summaries, and final placement summaries.
- Schema v1 candidate labels for region swaps, cluster decompression, and
  coldspot tightening.
- Schema-v1 trace-to-graph dataset builder in `scripts/build_gnn_dataset.py`.
- Dedicated GNN project docs and stage-gated skill in `docs/ml_nn/gnn/`.

Not implemented:

- Learned model.
- Train/eval scripts.
- Inference-time ranker integration.
- Model artifact versioning.
- Full benchmark acceptance policy.

## Stage G1: Trace Completeness

Status: implemented for schema v1. The schema is documented in
[`gnn_trace_schema.md`](gnn_trace_schema.md).

Goal: collect enough supervised data to train a candidate ranker.

Add trace coverage for:

- Region swap candidates:
  - hard/hard
  - hard/soft
  - soft/soft
- Cluster decompression candidates:
  - cluster id
  - expansion factor
  - axis scale
  - hierarchy quality delta
  - exact proxy delta when scored
- Coldspot tightening candidates:
  - selected cluster
  - field gap
  - quality delta
  - proxy delta
- Candidate rejection reasons:
  - illegal overlap
  - out of bounds
  - out of hierarchy region
  - exact proxy failed
  - hierarchy quality failed

Acceptance gate:

```bash
HIER_GNN_TRACE=1 HIER_GNN_TRACE_RUN=ibm10_trace uv run evaluate src/main.py -b ibm10
```

The trace should include candidate and label events from all active hierarchy
operators without changing placement output.

## Stage G2: Graph Dataset Builder

Status: implemented for schema v1. The dataset payload is documented in
[`gnn_dataset_schema.md`](gnn_dataset_schema.md).

Goal: convert JSONL traces into graph examples.

The current builder is:

```bash
uv run python scripts/build_gnn_dataset.py \
  --trace-dir ml_data/beyondppa_gnn \
  --out ml_data/beyondppa_gnn/dataset.pt
```

Current graph nodes:

- Hard macros.
- Soft macros.
- Hierarchy clusters.

Current node features:

- Macro type: hard, soft, fixed.
- Width, height, area.
- Current normalized center coordinates.
- Cluster id or cluster embedding id.
- Movable flag.
- Distance to canvas edge.
- Grid-alignment offset.

Current edges:

- Netlist connectivity edges, with net weight and fanout features.
- Macro-to-cluster membership edges.
- Spatial neighbor edges.

Current labels:

- Regression target: exact proxy delta.
- Optional auxiliary targets:
  - hierarchy quality delta
- Classification target:
  - accepted by exact gate
  - rejected by exact gate

Planned G4 graph extension before the first learned GNN:

- net nodes;
- macro-net edges;
- pin-offset edge features where available;
- net degree/fanout;
- dynamic net HPWL or wirelength-pressure features where practical.

Acceptance gate:

- Dataset builder is deterministic.
- Rebuilding from the same traces produces identical tensor shapes and labels.
- A small `ibm01`/`ibm10` trace builds without requiring CUDA.

## Stage G3: Baseline Non-GNN Ranker

Goal: prove the dataset labels are learnable before adding graph complexity.

Train simple baselines:

- Logistic regression for accept/reject.
- Gradient boosted trees or random forest if available.
- Small MLP on candidate features only.

Metrics:

- Top-k recall of accepted moves.
- Mean reciprocal rank for accepted candidates.
- Proxy-delta regression correlation.
- Per-operator recall: relocation, swap, decompression, coldspot.

Acceptance gate:

- The baseline beats existing heuristic ordering on held-out traces for at least
  one operator.
- If not, improve labels/features before building the GNN.

## Stage G4: GNN Model

Goal: train a graph model that scores candidate moves.

Recommended first model:

- Small GraphSAGE, GAT-style, or relational hetero-GNN encoder.
- MacroDiff+-inspired macro-net graph extension before training:
  - net nodes;
  - macro-net edges;
  - pin-offset edge features where available;
  - net degree/fanout;
  - dynamic net HPWL or wirelength-pressure features where practical.
- Candidate scorer MLP over:
  - source macro embedding
  - target macro/bin embedding
  - cluster embedding
  - net-context embedding when present
  - candidate scalar features

Keep the model small:

- 2-3 graph layers.
- Hidden size 32 or 64.
- CPU inference must be acceptable for IBM benchmarks.

Outputs:

- Candidate score for ranking.
- Optional proxy-delta estimate.
- Optional accept probability.
- Optional hierarchy-quality delta estimate.
- Optional rejection-reason probabilities.

Loss:

- Pairwise ranking loss within each candidate pool.
- Auxiliary regression loss for exact proxy delta.
- Optional binary cross entropy for accepted/rejected label.
- Optional hierarchy-quality regression loss.
- Optional rejection-reason classification loss.

Acceptance gate:

- Offline top-k recall improves over the non-GNN baseline.
- Model inference on `ibm10` candidate pools is below the pass budget.

## Stage G5: Inference Integration

Goal: start by using the GNN only as a ranker inside existing hierarchy
operators. After ranking is validated, expand one default-off role at a time.

Add controls:

```bash
HIER_GNN_RANK=0
HIER_GNN_MODEL=ml_data/beyondppa_gnn/model.pt
HIER_GNN_TOP_K=32
HIER_GNN_OPERATORS=relocation
```

Rules:

- Initial GNN integration may only reorder candidates.
- Expanded roles may propose, select, budget, or diagnose only after separate
  traces, offline metrics, and closed-loop validation.
- Existing legality checks remain mandatory.
- Fixed macros remain immobile.
- Bounds checks remain mandatory.
- Region constraints remain mandatory.
- Exact proxy gates remain mandatory.
- Hierarchy quality gates remain mandatory.
- Default stays disabled until full benchmark evidence justifies promotion.

Initial integration target:

- Hard post-swap propose-all relocation only.

Do not integrate everywhere at once. Add operators one at a time.

Expanded role order after candidate ranking:

1. Candidate pool expansion.
2. Operator selection in advisory/logging mode.
3. Acceptance-risk surrogate for scoring prioritization.
4. Region guidance.
5. Soft macro role guidance.
6. Budget allocation.

Acceptance gate:

```bash
uv run evaluate src/main.py -b ibm10
HIER_GNN_RANK=1 uv run evaluate src/main.py -b ibm10
```

The GNN run must be valid and not materially slower before testing additional
benchmarks.

## Stage G6: Multi-Benchmark Validation

Required sequence:

```bash
HIER_GNN_RANK=1 uv run evaluate src/main.py -b ibm01
HIER_GNN_RANK=1 uv run evaluate src/main.py -b ibm10
HIER_GNN_RANK=1 uv run evaluate src/main.py -b ibm12
HIER_GNN_RANK=1 uv run evaluate src/main.py -b ibm17
```

Only then run:

```bash
HIER_GNN_RANK=1 uv run evaluate src/main.py --all
```

Promotion requirements:

- 17/17 VALID.
- 0 overlaps.
- Runtime remains comfortably below the 1 hour total limit.
- Average proxy improves or hierarchy-quality metrics improve without
  unacceptable proxy regression.
- No single benchmark has a severe regression without an explicit documented
  reason.

## Stage G7: Artifact And Reproducibility

Add metadata for every trained model:

- Training trace IDs.
- Benchmark split.
- Feature schema version.
- Model architecture.
- Git commit or worktree fingerprint.
- Offline validation metrics.
- Inference benchmark results.

Store model bundles under:

```text
ml_data/beyondppa_gnn/models/<model_id>/
```

Required files:

- `feature_schema.json`
- `train_config.json`
- `splits.json`
- `trace_manifest.json`
- `metrics.json`
- `model.pt`
- `README.md`

## Risks

- Existing exact proxy often rewards spread more than hierarchy. A GNN trained
  only on proxy labels may learn proxy-spread behavior that damages hierarchy.
- Candidate labels are biased by the operators that generated them.
- Large traces can become expensive quickly; keep trace sampling configurable.
- A model that improves offline ranking can still regress closed-loop placement.

## Recommended Immediate Next Step

Start Stage G3: train and evaluate baseline non-GNN rankers on schema-v1
datasets. Do not build the GNN model until the baseline proves the labels are
learnable on held-out benchmark traces. Keep the expanded hierarchy-flow roles
documented in `docs/ml_nn/gnn/expansion-plan.md` as follow-on work after
candidate ranking is validated.
