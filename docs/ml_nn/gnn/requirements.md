# GNN Production Requirements

These are the required steps to complete the GNN subsystem for production.
The initial implementation is a candidate ranker. The broader target is the
hierarchy-flow assistant described in [expansion-plan.md](expansion-plan.md).

## G3: Baseline Non-GNN Ranker

Purpose: prove the schema-v1 labels are learnable before adding graph model
complexity.

Status: accepted offline on the minimum 4-benchmark split. The accepted
artifact is
`ml_data/beyondppa_gnn/models/20260619_g3_candidate_baseline_min4/`.
`scripts/gnn/train_gnn_baseline.py` trains logistic and MLP candidate-feature
baselines, reports trace-order and existing-score heuristic comparisons, and
can write the required baseline artifact files.

Required work:

- Collect benchmark-level train/validation traces.
- Build a Stage-G2 dataset from those traces.
- Run train/eval scripts for candidate-feature-only models.
- Train at least:
  - logistic regression for accept/reject;
  - a small MLP;
  - a tree/GBM ranker later if dependencies justify it.
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

Accepted result, 2026-06-19:

- Train: `ibm01`, `ibm17`.
- Validation: `ibm10`, `ibm12`.
- Dataset: 4 graphs, 183,452 examples, 1,082 accepted.
- Overall validation top-4 recall: trace order `0.3268`, logistic `0.5758`,
  MLP `0.5368`.
- Region-swap top-4 recall: trace order `0.3210`, logistic `0.5721`, MLP
  `0.5328`.
- Promotion decision: `default_off`; this artifact remains an offline baseline
  reference, not a production model.

Current command shape:

```bash
uv run python scripts/gnn/train_gnn_baseline.py \
  --dataset ml_data/beyondppa_gnn/dataset.pt \
  --train-benchmark ibm02 \
  --train-benchmark ibm03 \
  --val-benchmark ibm10 \
  --val-benchmark ibm12 \
  --out-dir ml_data/beyondppa_gnn/models/<model_id>
```

## G4: GNN Model

Purpose: score candidate moves using graph context plus candidate scalar
features.

Status: accepted offline on the minimum 4-benchmark split. The accepted
artifact is `ml_data/beyondppa_gnn/models/20260619_g4_macro_net_ranker_v1/`.

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

Accepted result, 2026-06-19:

- Dataset schema v2 adds net nodes, macro-net incidence edges, pin-offset edge
  features, net degree/fanout, and normalized net HPWL pressure.
- Train: `ibm01`, `ibm17`.
- Validation: `ibm10`, `ibm12`.
- Overall top-4 recall: accepted G3 MLP `0.5368`, same-split retrained G3 MLP
  `0.5216`, G4 macro-net ranker `0.7922`.
- Region-swap top-4 recall: accepted G3 MLP `0.5328`, same-split retrained G3
  MLP `0.5175`, G4 macro-net ranker `0.7904`.
- `ibm10` CPU scoring smoke: 37,414 candidates in `0.0336s` to `0.0500s`
  after warmup.
- Promotion decision: `default_off`; G5 may use this artifact only as a gated
  candidate ranker.

## G5: Inference Integration

Purpose: use the learned scorer only to reorder candidates inside existing
hierarchy operators.

Status: smoke-accepted for default-off relocation-only hard propose-all
candidate reordering. The runtime hook is
`src/placer/local_search/gnn_ranker.py`.

Required runtime controls:

```bash
HIER_GNN_RANK=0
HIER_GNN_MODEL=ml_data/beyondppa_gnn/model.pt
HIER_GNN_TOP_K=32
HIER_GNN_OPERATORS=relocation
HIER_GNN_PRESERVE_TOP_N=0
HIER_GNN_EXTRA_TOP_K=0
```

`HIER_GNN_PRESERVE_TOP_N` is diagnostic-only. It keeps the first N
deterministic proposals before appending GNN-ranked candidates. The tested
`ibm12` setting `HIER_GNN_PRESERVE_TOP_N=12` with `HIER_GNN_TOP_K=4` was
rejected.

`HIER_GNN_EXTRA_TOP_K` is also default-off. It allows additive diagnostics that
preserve deterministic proposals and append extra GNN-ranked proposals before
exact checking. It is not promoted.

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

Smoke result, 2026-06-19:

```bash
uv run evaluate src/main.py -b ibm10
HIER_GNN_RANK=1 \
HIER_GNN_MODEL=ml_data/beyondppa_gnn/models/20260619_g4_macro_net_ranker_v1/model.pt \
HIER_GNN_OPERATORS=relocation \
HIER_GNN_TOP_K=32 \
uv run evaluate src/main.py -b ibm10
```

Result:

- default-off baseline: VALID, `proxy=1.6192`, `45.80s`;
- GNN-ranked relocation-only smoke: VALID, `proxy=1.6180`, `50.16s`.

This only smoke-accepts the default-off hook. It is not promoted default-on.
G6 later passed legality but failed promotion, so broader use is blocked on
post-G6 diagnostics and a closed-loop improvement.

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

Status: legal but not promoted. The default-off relocation hook passed the
required sequence and full `--all`, but full-suite average proxy and runtime
regressed versus the accepted hierarchy baseline.

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

Result, 2026-06-19:

- Required sequence: `ibm10=1.6177`, `ibm01=0.9434`, `ibm12=2.1718`,
  `ibm17=2.1007`; all VALID.
- Full `--all`: AVG `1.3676`, 17/17 VALID, 0 overlaps, 786.25s.
- Accepted hierarchy baseline: AVG `1.3631`, 17/17 VALID, 0 overlaps, 602.76s.
- Decision: do not promote; keep `HIER_GNN_RANK=0` in production.

## Post-G6: Ranking Diagnostics And Improvement

Status: started. The first diagnostic compares trace order, G3 MLP, and G4
macro-net rankings against exact proxy-gain labels inside candidate pools.

Required command:

```bash
uv run python scripts/gnn/diagnose_gnn_ranking.py \
  --dataset ml_data/beyondppa_gnn/dataset.pt \
  --g3-model ml_data/beyondppa_gnn/models/20260619_g3_candidate_baseline_min4/model.pt \
  --g4-model ml_data/beyondppa_gnn/models/20260619_g4_macro_net_ranker_v1/model.pt \
  --top-k 4 \
  --out ml_data/beyondppa_gnn/diagnostics/<run_id>.json
```

First result, 2026-06-19:

- trace-order exact-gain recall@4: `0.2513`;
- G3 exact-gain recall@4: `0.5239`;
- G4 exact-gain recall@4: `0.6318`;
- G4 top-4 overlap with trace order: `0.1889`.

Paired trace result, 2026-06-19:

- `scripts/gnn/compare_gnn_trace_pairs.py` added for heuristic-vs-GNN JSONL trace
  comparison.
- Paired validation set final deltas:
  - `ibm01=-0.0001`;
  - `ibm10=-0.0001`;
  - `ibm12=+0.0001`;
  - `ibm17=+0.0002`.
- Hard propose-all accepts changed:
  - `ibm01: 0->0`;
  - `ibm10: 1->0`;
  - `ibm12: 4->1`;
  - `ibm17: 0->0`.
- All GNN paired traces had scored relocation samples and no ranker errors.

Rejected `ibm12` follow-ups:

- `HIER_GNN_TOP_K=8`: VALID, final delta `+0.0106`, hard propose-all accepts
  `4->3`;
- `HIER_GNN_TOP_K=16`: VALID, final delta `+0.0106`, hard propose-all accepts
  `4->3`;
- `HIER_GNN_PRESERVE_TOP_N=12` plus `HIER_GNN_TOP_K=4`: VALID, final delta
  `+0.0126`, hard propose-all accepts `4->3`;
- `PYTHONHASHSEED=0` did not fully stabilize upstream region-swap counts.

Improvement requirements:

- compare pass-level accepted move count, exact-score count, runtime, and final
  proxy;
- decide whether the next fix is value-weighted retraining, smaller
  `HIER_GNN_TOP_K`, or a runtime budget guard;
- rerun G6 before any operator expansion or default-on promotion.

Next required experiment:

- Extend `HIER_DIAGNOSTIC_NO_DEADLINES=1` comparisons to `ibm01`, `ibm10`, and
  `ibm17` before changing the model. The first controlled `ibm12` result is
  positive, so the next fix should focus on production budget/timing
  interaction, not immediate retraining.

Accepted diagnostic result, 2026-06-20:

- `HIER_DIAGNOSTIC_NO_DEADLINES=1` made `ibm12` heuristic and GNN traces
  repeatable across two runs each.
- Heuristic: final `2.1719`, region swaps `391`, hard propose-all accepts `3`.
- GNN top-k 32: final `2.1707`, region swaps `391`, hard propose-all accepts
  `5`.
- Controlled delta: `-0.0012` proxy, hard propose-all accepts `3->5`.

Controlled additive result, 2026-06-20:

- Mode: `HIER_GNN_PRESERVE_TOP_N=16`, `HIER_GNN_TOP_K=8`,
  `HIER_GNN_EXTRA_TOP_K=8`, with `HIER_DIAGNOSTIC_NO_DEADLINES=1`.
- Four-benchmark controlled deltas: `ibm01=+0.0000`, `ibm10=+0.0000`,
  `ibm12=-0.0024`, `ibm17=+0.0000`; total `-0.0024`.
- The mode removes the pure-GNN `ibm10` displacement (`1->0` hard propose-all
  accepts) and improves `ibm12` hard propose-all accepts `3->6`.
- Next gate: timed production-mode smoke on `ibm10` and `ibm12`; do not
  promote before timed G6 validation.

Timed additive smoke, 2026-06-20:

- `ibm10`: VALID, current timed heuristic `1.6184`, additive `1.6192`, delta
  `+0.0008`, hard propose-all accepts `1->1`, exact checks `15->23`.
- `ibm12`: VALID, current timed heuristic `2.1857`, additive `2.1703`, delta
  `-0.0154`, hard propose-all accepts `4->4`, exact checks `4->4`.
- Decision: mixed; keep default-off. Do not run full G6 promotion until repeated
  timed smoke or a stricter production budget guard removes the `ibm10` risk.

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
