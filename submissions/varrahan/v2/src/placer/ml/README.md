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
ML_TRACE_PATH=/tmp/ibm_moves.jsonl \
  uv run evaluate submissions/varrahan/v2/src/main.py -b ibm01
```

For broader data, run all public benchmarks across multiple seeds/configurations
and write each run to a separate JSONL file. Each row contains:

- `group_id`: candidates competing at the same decision state
- `operator` and `field`
- normalized, inference-cheap candidate features
- `score_gain = state_score - trial_score`, the supervised regression label
- `improves`, a convenient binary label

Split train/validation/test **by benchmark and run**, never by row. Row-level
splits leak nearly identical states and overstate generalization. Hold out at
least 3 IBM benchmarks and all NG45 designs from model selection.

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
