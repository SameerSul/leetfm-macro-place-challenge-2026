# GNN Evaluation Gates

Evaluation has two layers: offline ranking quality and closed-loop placer
behavior. Offline wins are necessary but not sufficient.

## Offline Metrics

Compute metrics by benchmark and by operator.

Required:

- top-1, top-4, top-8, top-16 recall of accepted candidates;
- mean reciprocal rank;
- accepted-vs-rejected ROC AUC when class balance allows it;
- proxy-delta Spearman or Pearson correlation for known-delta examples;
- hierarchy-quality delta correlation when available;
- rejection-reason classification accuracy when trained;
- mean predicted rank of accepted candidates;
- oracle top-k recall for each candidate pool.

Report aggregate metrics and per-operator metrics for:

- relocation;
- hard/hard swaps;
- hard/soft swaps;
- soft/soft swaps;
- cluster decompression;
- coldspot tightening.

## Baseline Comparison

Every learned scorer must be compared against:

- current heuristic trace order;
- simple score/rank fields already present in the trace;
- the G3 non-GNN baseline.

Do not move to G4 unless the non-GNN baseline shows the labels are learnable.

For the MacroDiff+-inspired graph upgrade, also compare:

- macro/cluster graph only;
- heterogeneous macro-net graph without pin offsets;
- heterogeneous macro-net graph with pin offsets;
- heterogeneous macro-net graph with dynamic net HPWL features.

Keep the smallest graph representation that wins offline and remains cheap
enough for CPU inference.

## Expanded-Role Metrics

For the roles in [expansion-plan.md](expansion-plan.md), evaluate:

- candidate pool expansion: accepted-move recall added per extra candidate and
  exact-score overhead;
- operator selection: pass value prediction accuracy and missed-improvement
  rate;
- region guidance: hierarchy-quality delta, proxy delta, and downstream
  accepted-move count;
- soft-role guidance: agreement with deterministic roles and closed-loop proxy /
  hierarchy impact;
- acceptance-risk surrogate: exact scores saved, false-negative rate for good
  moves, and runtime impact;
- budget allocation: proxy/runtime Pareto change and full-suite timeout risk.

## Runtime Metrics

Measure:

- feature extraction time;
- graph encoding time;
- candidate scoring time;
- total per-pass inference time;
- memory usage for `ibm10` and one large benchmark such as `ibm17` or `ibm18`.

CPU inference must be acceptable because production cannot require a GPU for
the GNN ranker.

## Closed-Loop Smoke

First closed-loop command:

```bash
uv run evaluate src/main.py -b ibm10
HIER_GNN_RANK=1 uv run evaluate src/main.py -b ibm10
```

Required before expanding:

- both runs are VALID;
- no hard overlaps;
- runtime change is small enough for the full-suite budget;
- GNN run does not materially regress proxy or hierarchy quality.

## Multi-Benchmark Gate

After `ibm10`:

```bash
HIER_GNN_RANK=1 uv run evaluate src/main.py -b ibm01
HIER_GNN_RANK=1 uv run evaluate src/main.py -b ibm12
HIER_GNN_RANK=1 uv run evaluate src/main.py -b ibm17
```

Then:

```bash
HIER_GNN_RANK=1 uv run evaluate src/main.py --all
```

Required for promotion:

- 17/17 VALID;
- 0 overlaps;
- runtime below the 1 hour challenge limit with margin;
- average proxy improvement or documented hierarchy-quality improvement;
- no severe single-benchmark regression without a documented reason.

## Regression Policy

If a model improves offline ranking but regresses closed-loop placement, the
closed-loop result wins. Keep the model default-off and either:

- improve labels/features;
- restrict the model to the operator where it helps;
- reduce `HIER_GNN_TOP_K`;
- retrain on traces generated from the model-induced states.
