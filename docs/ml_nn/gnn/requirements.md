# GNN Production Requirements

These are the required steps to complete the GNN subsystem for production.
The initial implementation is a candidate ranker. The broader target is the
hierarchy-flow assistant described in [expansion-plan.md](expansion-plan.md).

## G3: Baseline Non-GNN Ranker

Purpose: prove the schema-v1 labels are learnable before adding graph model
complexity.

Required work:

- Add train/eval scripts for candidate-feature-only models.
- Train at least:
  - logistic regression for accept/reject;
  - a small MLP or tree/GBM ranker if dependencies allow.
- Evaluate by operator:
  - relocation;
  - region swaps;
  - cluster decompression;
  - coldspot tightening.

Required metrics:

- top-k recall of accepted candidates;
- mean reciprocal rank;
- proxy-delta regression correlation for examples with known proxy delta;
- accepted-rate and class-balance reporting per operator;
- per-benchmark validation metrics.

Promotion gate:

- A baseline must beat the current heuristic ordering on held-out traces for at
  least one operator.
- If it does not, improve data coverage, features, or labels before building a
  GNN.

## G4: GNN Model

Purpose: score candidate moves using graph context plus candidate scalar
features.

Paper-informed design requirement:

- Use [MacroDiff+](macrodiff_plus_notes.md) as the graph-design reference for
  topology-aware modeling.
- Extend the Stage-G2 graph toward a heterogeneous macro-net graph before
  training the first GNN ranker.
- Preserve pin-offset information on macro-net edges where the source data
  exposes it.
- Add dynamic net features such as normalized HPWL or wirelength pressure when
  practical.

Required model shape:

- Small GraphSAGE, GAT-style, or relational hetero-GNN encoder.
- 2-3 graph layers.
- Hidden size 32 or 64.
- Candidate scorer MLP over:
  - source node embedding;
  - target node embedding when present;
  - cluster node embedding when present;
  - net-context embedding when present;
  - candidate scalar features.

Required outputs:

- ranking score;
- optional proxy-delta estimate;
- optional accept probability;
- optional hierarchy-quality delta estimate;
- optional rejection-reason probabilities.

Required losses:

- pairwise ranking loss inside each candidate pool;
- auxiliary regression loss for exact proxy delta when known;
- optional binary cross entropy for accepted/rejected labels;
- optional hierarchy-quality regression loss;
- optional rejection-reason classification loss.

Promotion gate:

- Offline top-k recall improves over the G3 baseline.
- CPU inference on `ibm10` candidate pools fits inside the hierarchy pass
  budget.

## G5: Inference Integration

Purpose: use the learned scorer only to reorder candidates inside existing
hierarchy operators.

Required runtime controls:

```bash
HIER_GNN_RANK=0
HIER_GNN_MODEL=ml_data/beyondppa_gnn/model.pt
HIER_GNN_TOP_K=32
HIER_GNN_OPERATORS=relocation
```

Required rules:

- The GNN may only reorder candidates.
- MacroDiff+-style diffusion sampling is out of scope for production
  integration.
- Existing legality checks remain mandatory.
- Fixed macros remain immobile.
- Bounds checks remain mandatory.
- Hierarchy-region constraints remain mandatory.
- Hierarchy-quality gates remain mandatory.
- Exact-proxy gates remain mandatory.
- Default remains disabled until full benchmark evidence justifies promotion.

Initial integration target:

- Hard post-swap propose-all relocation only.

Do not integrate all operators at once. Add one operator, validate it, then move
to the next.

After candidate ranking validates, add expanded roles in this order:

1. Candidate pool expansion.
2. Operator selection in advisory/logging mode.
3. Acceptance-risk surrogate for scoring prioritization.
4. Region guidance.
5. Soft macro role guidance.
6. Budget allocation.

Each role must be independently default-off and benchmark-gated.

## G6: Closed-Loop Validation

Required sequence:

```bash
HIER_GNN_RANK=1 uv run evaluate src/main.py -b ibm10
HIER_GNN_RANK=1 uv run evaluate src/main.py -b ibm01
HIER_GNN_RANK=1 uv run evaluate src/main.py -b ibm12
HIER_GNN_RANK=1 uv run evaluate src/main.py -b ibm17
HIER_GNN_RANK=1 uv run evaluate src/main.py --all
```

Promotion requirements:

- 17/17 VALID.
- 0 overlaps.
- Runtime comfortably below the 1 hour total limit.
- Average proxy improves, or hierarchy-quality metrics improve without
  unacceptable proxy regression.
- No benchmark has a severe regression without a documented reason.

## G7: Artifact And Reproducibility

Required before any default-on promotion:

- versioned feature schema;
- training trace IDs;
- benchmark split metadata;
- model config;
- model artifact;
- offline metrics;
- closed-loop benchmark results;
- git commit or worktree fingerprint.
