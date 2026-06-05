# ML Candidate Ranking

## Recommendation

Do not initially replace the full R2 interleave with an end-to-end placement
model. R2 is a sequential, constrained local search: accepted moves alter the
next state, hard-macro legality must remain exact, and the useful output is a
small set of promising actions rather than an unconstrained placement.

The first production model should be a **separate gradient-boosted tree candidate
ranker for each move family**:

- hard relocation
- soft relocation
- hard 2-opt
- soft 2-opt
- hard-soft swap
- hard-soft-soft cycle

Start with XGBoost: compare a regressor that predicts exact proxy gain against a
LambdaMART ranker trained on within-group ordinal relevance labels. For every
decision group (one source macro/state and its legal candidate destinations or
partners), rank candidates by predicted gain. Then run the existing incremental
scorer and exact acceptance gate only on the top-K ranked candidates. This
preserves correctness and monotonicity while targeting the expensive
candidate-evaluation loop.

Gradient-boosted trees are the right first model because the features are small,
tabular, nonlinear, and heterogeneous; data volume is initially modest; CPU
inference is cheap; and feature importance makes failures inspectable. A GNN or
RL policy may eventually outperform it, but requires substantially more data and
creates a much harder correctness and generalization problem.

## Collecting data

Tracing is opt-in and does not change placement decisions:

```bash
ML_TRACE_PATH=/tmp/ibm_moves.jsonl.gz \
  uv run evaluate submissions/varrahan/v2/src/main.py -b ibm01
```

For any run that enables subprocess-based local search, use placeholders so
each process writes its own valid JSONL stream:

```bash
ML_TRACE_PATH='/tmp/traces/{run_id}-{pid}.jsonl.gz' \
  uv run evaluate submissions/varrahan/v2/src/main.py --all
```

For broader data, run all public benchmarks across multiple seeds/configurations
and write each run to a separate JSONL file. Each row contains:

- `row_type`: `candidate` for model rows or `event` for metadata/summaries
- `run_id`, seed, configuration, configuration hash, and benchmark dimensions
- search context: phase, R2 round, pass, elapsed time, and remaining budget
- `group_id`: candidates competing at the same decision state
- `operator`, `field`, heuristic rank, group size, and candidate source
- normalized, inference-cheap candidate features
- `score_gain = state_score - trial_score`, the supervised regression label
- `improves`, a convenient binary label
- group-summary events with generated/scored/rejected candidate counts

Use a `.gz` suffix for full runs. Compression is transparent to the collector
and dataset loader and substantially reduces the large candidate-level traces.
Full configuration metadata is stored once in each `benchmark_start` event;
candidate rows retain the stable run/config hash and numeric benchmark context.

Training loaders must select only `row_type == "candidate"`. Event rows are for
data-quality analysis, especially measuring legality and cheap-prefilter
selection bias.

`placer.ml.dataset` provides dependency-free helpers to load/flatten candidates,
derive within-group LambdaMART relevance, and summarize trace quality:

```python
from placer.ml.dataset import add_group_relevance, load_candidates, trace_summary

paths = ["/tmp/traces/run-a.jsonl", "/tmp/traces/run-b.jsonl"]
rows = add_group_relevance(load_candidates(paths, operator="hard_relocation"))
print(trace_summary(paths))
```

Split train/validation/test **by benchmark and run**, never by row. Row-level
splits leak nearly identical states and overstate generalization. Hold out at
least 3 IBM benchmarks and all NG45 designs from model selection.

## Model framework

`placer.ml.modeling` defines the inactive integration surface. It is not wired
into the placer pipeline yet.

- `OPERATORS` names the six independent model families:
  `hard_relocation`, `soft_relocation`, `hard_2opt`, `soft_2opt`,
  `hard_soft_swap`, and `hard_soft_soft_cycle`.
- `feature_names_for(operator)` returns the stable feature order expected by
  that operator's model.
- `ModelSpec` describes one model artifact.
- `CandidateRanker` vectorizes candidate rows, predicts scores, ranks
  candidates, and returns top-K indices for later exact scoring.
- `ModelBank` loads a manifest containing multiple independent operator models.
- `build_training_matrix(rows, operator, label=...)` converts flattened trace
  rows into `X`, `y`, and grouped row counts suitable for gain regression or
  LambdaMART.

The future pipeline integration should call a ranker immediately after cheap
candidate generation and before exact incremental scoring. It should still keep
the exact scorer and exact accept gate as the source of truth.

Example manifest shape:

```json
{
  "models": [
    {
      "operator": "soft_relocation",
      "backend": "xgboost_json",
      "feature_names": ["accepted_in_pass", "source_hot_rank_norm"],
      "model_path": "soft_relocation.xgb.json",
      "top_k_default": 16,
      "keep_heuristic_first": 2,
      "random_exploration_fraction": 0.05
    }
  ]
}
```

`xgboost_json` loads `xgboost.Booster` lazily, so XGBoost is only required when
real model artifacts are used. Tests use the `linear_json` backend as a tiny
dependency-free stand-in; it is intended for integration tests and smoke checks,
not as the production model.

## Training and rollout

1. Collect unfiltered traces from the current policy. Preserve rejected moves;
   they are essential negatives.
2. Train one model per operator. Compare gain regression against LambdaMART using
   within-group ordinal or quantized relevance derived from `score_gain`.
3. Evaluate ranking quality using `Recall@K` of improving moves and regret:
   best actual gain minus best actual gain among predicted top-K.
4. Shadow the model first: log predictions but keep scoring every candidate.
5. Enable top-K filtering only after `Recall@K` is high on held-out benchmarks.
   Keep 5-10% random exploration and always retain the heuristic's first few
   candidates to limit distribution shift.
6. Measure the actual objective: final proxy score under the same wall-clock
   budget. Per-row RMSE is secondary.

The model should initially reduce exact evaluations, not remove the interleave
or its exact scorer. With the saved time, R2 can inspect wider candidate pools or
run additional rounds.

## Offline training CLI

`placer.ml.train` trains and evaluates the per-operator XGBoost artifacts from
trace files. It is offline-only and is not imported by the placer runtime.

Small smoke run:

```bash
PYTHONPATH=submissions/varrahan/v2/src \
uv run python -m placer.ml.train \
  submissions/varrahan/v2/ml_data/traces/s42_20260604_181419.jsonl.gz \
  --output-dir /tmp/v2_ml_models \
  --operators soft_relocation,soft_2opt \
  --objective ranker \
  --max-rows-per-operator 50000 \
  --rounds 20
```

Full run shape:

```bash
PYTHONPATH=submissions/varrahan/v2/src \
uv run python -m placer.ml.train \
  submissions/varrahan/v2/ml_data/traces/*.jsonl.gz \
  --output-dir submissions/varrahan/v2/ml_data/models/latest \
  --objective ranker \
  --rounds 200
```

Outputs:

- `manifest.json`: `ModelBank`-compatible list of model specs.
- `metrics.json`: row counts, group counts, RMSE, `best_recall@K`,
  `improving_recall@K`, and `mean_regret@K`.
- one `*.xgb.json` model per trained operator.

Splitting is by `(benchmark, run_id)`, never by row. By default, benchmarks whose
name starts with `ng` are held out as test data and the remaining runs are split
into train/validation.
