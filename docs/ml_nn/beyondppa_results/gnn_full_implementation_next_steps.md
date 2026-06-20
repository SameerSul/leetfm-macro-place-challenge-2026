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
- Schema-v1 trace-to-graph dataset builder in `scripts/gnn/build_gnn_dataset.py`.
- Dedicated GNN project docs and stage-gated skill in `docs/ml_nn/gnn/`.
- Stage-G3 offline baseline script in `scripts/gnn/train_gnn_baseline.py`.
- Accepted Stage-G3 offline baseline artifact:
  `ml_data/beyondppa_gnn/models/20260619_g3_candidate_baseline_min4/`.
- Stage-G4 dataset schema v2 macro-net graph extension in
  `scripts/gnn/build_gnn_dataset.py`.
- Stage-G4 offline graph ranker script in `scripts/gnn/train_gnn_ranker.py`.
- Accepted Stage-G4 offline macro-net ranker artifact:
  `ml_data/beyondppa_gnn/models/20260619_g4_macro_net_ranker_v1/`.
- Stage-G5 default-off relocation-only inference hook in
  `src/placer/local_search/gnn_ranker.py`.

Not implemented:

- Promotion-quality closed-loop improvement.
- Full artifact/reproducibility discipline for future promoted models.

## Stage G1: Trace Completeness

Status: implemented and smoke-verified for schema v1. The schema is documented in
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

Latest smoke verification, 2026-06-19:

```bash
HIER_GNN_TRACE=1 \
HIER_GNN_TRACE_PATH=/tmp/hier_gnn_trace_smoke_1781894380.jsonl \
HIER_GNN_TRACE_MAX_CANDIDATES=5 \
uv run evaluate src/main.py -b ibm01
```

Result: VALID, proxy `0.9435`, runtime `35.85s`. The trace contained 1,535
schema-v1 rows and no schema-version failures:

- `hier_relocation_candidates`: 1
- `hier_relocation_result`: 13
- `hier_swap_candidates`: 1,498
- `hier_decompression_candidate`: 5
- `hier_coldspot_candidate`: 8
- `hier_pass_result`: 9
- `hier_final`: 1

## Stage G2: Graph Dataset Builder

Status: implemented and smoke-verified for schema v1. The dataset payload is documented in
[`gnn_dataset_schema.md`](gnn_dataset_schema.md).

Goal: convert JSONL traces into graph examples.

The current builder is:

```bash
uv run python scripts/gnn/build_gnn_dataset.py \
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

Latest smoke verification, 2026-06-19:

```bash
uv run python test/verification/_verify_gnn_dataset_builder.py
uv run python scripts/gnn/build_gnn_dataset.py \
  --trace-path /tmp/hier_gnn_trace_smoke_1781894380.jsonl \
  --out /tmp/hier_gnn_dataset.pt \
  --benchmark ibm01
```

Result: builder verifier passed; smoke dataset wrote 1 graph, 7,508 examples,
and 108 accepted examples.

## Stage G3: Baseline Non-GNN Ranker

Status: accepted offline on the minimum 4-benchmark split. The baseline
entrypoint is `scripts/gnn/train_gnn_baseline.py`; focused verification is
`test/verification/_verify_gnn_baseline.py`.

Goal: prove the dataset labels are learnable before adding graph complexity.

Train simple baselines:

- Logistic regression for accept/reject.
- Small MLP on candidate features only.
- Gradient boosted trees or random forest later if dependencies justify it.

Metrics:

- Top-k recall of accepted moves.
- Mean reciprocal rank for accepted candidates.
- Proxy-delta regression correlation.
- Per-operator recall: relocation, swap, decompression, coldspot.

Acceptance gate:

- The baseline beats existing heuristic ordering on held-out traces for at least
  one operator.
- If not, improve labels/features before building the GNN.

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

The script reports validation metrics for trace-order heuristic, existing-score
heuristic, logistic regression, and MLP. It can write the required G7 baseline
artifact files. Future baseline artifacts still need benchmark-level held-out
metrics before they can replace this accepted G3 reference.

Accepted result, 2026-06-19:

```bash
uv run python scripts/gnn/train_gnn_baseline.py \
  --dataset ml_data/beyondppa_gnn/dataset.pt \
  --train-benchmark ibm01 \
  --train-benchmark ibm17 \
  --val-benchmark ibm10 \
  --val-benchmark ibm12 \
  --out-dir ml_data/beyondppa_gnn/models/20260619_g3_candidate_baseline_min4
```

Dataset:

- traces: `ibm01`, `ibm10`, `ibm12`, `ibm17`;
- graphs: 4;
- examples: 183,452;
- accepted labels: 1,082.

Validation summary:

- overall top-4 recall: trace order `0.3268`, existing-score heuristic
  `0.3268`, logistic `0.5758`, MLP `0.5368`;
- overall MRR: trace order `0.2611`, logistic `0.4772`, MLP `0.4437`;
- region-swap top-4 recall: trace order `0.3210`, logistic `0.5721`, MLP
  `0.5328`;
- region-swap MRR: trace order `0.2557`, MLP `0.4389`.

The G3 acceptance gate is met because the baseline beats heuristic ordering on
the held-out region-swap operator. Promotion decision remains `default_off`:
this artifact proves offline label learnability only and does not change
placement.

## Stage G4: GNN Model

Status: accepted offline on the minimum 4-benchmark split. The accepted
artifact is `ml_data/beyondppa_gnn/models/20260619_g4_macro_net_ranker_v1/`.

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

Accepted result, 2026-06-19:

```bash
uv run python scripts/gnn/train_gnn_ranker.py \
  --dataset ml_data/beyondppa_gnn/dataset.pt \
  --g3-model ml_data/beyondppa_gnn/models/20260619_g3_candidate_baseline_min4/model.pt \
  --train-benchmark ibm01 \
  --train-benchmark ibm17 \
  --val-benchmark ibm10 \
  --val-benchmark ibm12 \
  --out-dir ml_data/beyondppa_gnn/models/20260619_g4_macro_net_ranker_v1 \
  --epochs 80 \
  --hidden-size 32 \
  --layers 2 \
  --seed 11
```

Dataset schema v2 adds:

- net-node features: degree, macro degree, net weight, normalized HPWL x/y, and
  weighted HPWL pressure;
- macro-net incidence edges;
- macro-net edge features: normalized pin offsets, net weight, fanout, and
  driver-pin flag.

Validation summary:

- overall top-4 recall: trace order `0.3268`, accepted G3 MLP artifact
  `0.5368`, same-split G3 MLP `0.5216`, G4 macro-net ranker `0.7922`;
- overall MRR: trace order `0.2611`, accepted G3 MLP artifact `0.4437`,
  same-split G3 MLP `0.4199`, G4 macro-net ranker `0.6412`;
- region-swap top-4 recall: trace order `0.3210`, accepted G3 MLP artifact
  `0.5328`, same-split G3 MLP `0.5175`, G4 macro-net ranker `0.7904`.

Runtime smoke:

- `ibm10` validation candidate count: 37,414;
- CPU scoring time after warmup: `0.0336s` to `0.0500s`.

The G4 acceptance gate is met because offline top-k recall improves over G3 and
CPU scoring is comfortably below the hierarchy pass budget. Promotion decision
remains `default_off`: no inference integration or placement behavior changes.
The model's proxy-delta correlation is weaker than G3, so exact-proxy gates
must remain authoritative in G5.

## Stage G5: Inference Integration

Status: smoke-accepted for default-off relocation-only hard propose-all
candidate reordering on `ibm10`.

Goal: start by using the GNN only as a ranker inside existing hierarchy
operators. After ranking is validated, expand one default-off role at a time.

Add controls:

```bash
HIER_GNN_RANK=0
HIER_GNN_MODEL=ml_data/beyondppa_gnn/model.pt
HIER_GNN_TOP_K=32
HIER_GNN_OPERATORS=relocation
HIER_GNN_PRESERVE_TOP_N=0
```

`HIER_GNN_PRESERVE_TOP_N` is a diagnostic-only guard that preserves the first N
deterministic proposals before appending GNN-ranked candidates. It is not an
accepted improvement; the `ibm12` `preserve=12, top_k=4` test regressed.

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

The hook only reorders the existing hard propose-all relocation candidate list.
All downstream hard legality, bounds, fixed-macro, hierarchy-region,
hierarchy-quality, and exact-proxy gates remain unchanged. The hook remains
default-off and is not promoted until G6 multi-benchmark validation passes.

## Stage G6: Multi-Benchmark Validation

Status: valid but not promoted. The default-off relocation hook passed legality
on the required sequence and full `--all`, but regressed average proxy and
runtime versus the accepted hierarchy baseline.

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

Result, 2026-06-19:

- Required sequence:
  - `ibm10=1.6177`, VALID, 50.00s;
  - `ibm01=0.9434`, VALID, 35.03s;
  - `ibm12=2.1718`, VALID, 46.84s;
  - `ibm17=2.1007`, VALID, 49.24s.
- Full `--all`: AVG `1.3676`, 17/17 VALID, 0 overlaps, 786.25s.
- Accepted hierarchy baseline for comparison: AVG `1.3631`, 17/17 VALID,
  0 overlaps, 602.76s.

Decision: do not promote. Keep `HIER_GNN_RANK=0` in production. The next GNN
work should diagnose candidate ordering quality by benchmark and operator,
especially where the full-suite average regressed despite offline recall wins.

## Post-G6 Diagnostic Stage

Status: started. The first task is to explain the G6 gap before expanding the
GNN to more operators.

Diagnostic command:

```bash
uv run python scripts/gnn/diagnose_gnn_ranking.py \
  --dataset ml_data/beyondppa_gnn/dataset.pt \
  --g3-model ml_data/beyondppa_gnn/models/20260619_g3_candidate_baseline_min4/model.pt \
  --g4-model ml_data/beyondppa_gnn/models/20260619_g4_macro_net_ranker_v1/model.pt \
  --top-k 4 \
  --out ml_data/beyondppa_gnn/diagnostics/20260619_gnn_ranking_value_top4.json
```

First result, 2026-06-19:

- trace-order exact-gain recall@4: `0.2513`;
- G3 exact-gain recall@4: `0.5239`;
- G4 exact-gain recall@4: `0.6318`;
- G4 top-4 overlap with trace order: `0.1889`.

Interpretation: G4 does identify more exact proxy-gain candidates offline, so
the closed-loop regression is not explained by a weak local ranker alone. The
next work should isolate distribution shift from model-ranked states,
downstream pass interaction, and runtime budget displacement.

Paired trace comparison started with
`scripts/gnn/compare_gnn_trace_pairs.py`. Four-benchmark result, 2026-06-19:

- `ibm01`: heuristic `0.9435`, GNN `0.9434`, delta `-0.0001`, hard
  propose-all accepts `0->0`;
- `ibm10`: heuristic `1.6173`, GNN `1.6172`, delta `-0.0001`, hard
  propose-all accepts `1->0`;
- `ibm12`: heuristic `2.1772`, GNN `2.1772`, delta `+0.0001`, hard
  propose-all accepts `4->1`;
- `ibm17`: heuristic `2.1005`, GNN `2.1007`, delta `+0.0002`, hard
  propose-all accepts `0->0`.

All paired GNN traces had scored relocation samples and no ranker errors. The
current failure mode is candidate-order interaction, not model-load failure.
`ibm12` is the first targeted benchmark because GNN ranking suppresses three
hard propose-all accepts there and downstream coldspot/micro-shift work recovers
most, but not all, of the local loss.

Rejected `ibm12` sensitivity tests, 2026-06-19:

- `HIER_GNN_TOP_K=8`: VALID, final delta `+0.0106`, hard propose-all accepts
  `4->3`;
- `HIER_GNN_TOP_K=16`: VALID, final delta `+0.0106`, hard propose-all accepts
  `4->3`;
- `HIER_GNN_PRESERVE_TOP_N=12` plus `HIER_GNN_TOP_K=4`: VALID, final delta
  `+0.0126`, hard propose-all accepts `4->3`.

A controlled rerun with `PYTHONHASHSEED=0` still changed upstream region-swap
counts between heuristic and GNN processes, so the paired traces are not clean
causal A/Bs. They are useful diagnostics, but the next improvement needs either
a more repeatable diagnostic mode or repeated-run variance estimates.

Repeatable diagnostic mode added, 2026-06-20:

```bash
HIER_DIAGNOSTIC_NO_DEADLINES=1 \
HIER_GNN_TRACE=1 \
uv run evaluate src/main.py -b ibm12
```

This disables local hierarchy relief deadlines for diagnostics only. It is not
an accepted production mode, but it makes `ibm12` causal GNN comparisons
repeatable:

- heuristic A/B: VALID, final `2.1719`, region swaps `391`, hard propose-all
  accepts `3`;
- GNN top-k 32 A/B: VALID, final `2.1707`, region swaps `391`, hard propose-all
  accepts `5`;
- controlled GNN delta: `-0.0012` proxy with upstream region swaps unchanged.

This reverses the earlier uncontrolled `ibm12` read: once deadline noise is
removed, the GNN ranker improves the post-swap hard propose-all hook. The
remaining production problem is likely that the learned hook spends or shifts
local time budget enough to change later search behavior.

Additive controlled mode, 2026-06-20:

```bash
HIER_DIAGNOSTIC_NO_DEADLINES=1 \
HIER_GNN_RANK=1 \
HIER_GNN_TOP_K=8 \
HIER_GNN_PRESERVE_TOP_N=16 \
HIER_GNN_EXTRA_TOP_K=8 \
uv run evaluate src/main.py -b <benchmark>
```

This preserves the deterministic top-16 hard propose-all candidates and appends
8 GNN-ranked candidates. Controlled no-deadline result:

- `ibm01`: delta `+0.0000`, hard propose-all accepts `0->0`;
- `ibm10`: delta `+0.0000`, hard propose-all accepts `1->1`;
- `ibm12`: delta `-0.0024`, hard propose-all accepts `3->6`;
- `ibm17`: delta `+0.0000`, hard propose-all accepts `0->0`;
- total delta `-0.0024`.

The additive mode is the first controlled GNN shape that does not displace
deterministic wins on the four-benchmark diagnostic set. It increases exact
checks from 16 to 24 for no-gain cases, so it must be tested under production
deadlines before any promotion.

Timed production smoke, 2026-06-20:

- `ibm10`: current timed heuristic `1.6184` VALID; additive `1.6192` VALID;
  delta `+0.0008`; hard propose-all accepts `1->1`; exact checks `15->23`.
- `ibm12`: current timed heuristic `2.1857` VALID; additive `2.1703` VALID;
  delta `-0.0154`; hard propose-all accepts `4->4`; exact checks `4->4`.

Decision: do not promote. The additive mode has a controlled positive signal
and a strong timed `ibm12` result, but it still shows `ibm10` timed risk and
extra exact checks. The next implementation should add a production guard for
unused exact-check budget or repeat timed smoke before running G6.

Immediate next steps:

1. Add a production guard so additive GNN only uses spare post-swap
   hard-propose budget, or run repeated timed smoke to quantify the `ibm10`
   risk.
2. Retest timed `ibm10` and `ibm12` after the guard.
3. If timed smoke is non-regressive, rerun the four-benchmark G6 sequence with
   additive mode.
4. Keep broader operator integration blocked until production G6 improves.
5. Keep production default-off until the closed-loop gate beats the accepted
   hierarchy baseline.

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

Continue by adding a production-safe additive budget guard or by repeating timed
`ibm10`/`ibm12` smoke. The no-deadline additive shape is controlled-positive,
but timed smoke is mixed, so keep it default-off. Keep expanded hierarchy-flow
roles documented in `docs/ml_nn/gnn/expansion-plan.md` as follow-on work after
candidate ranking is closed-loop validated.
