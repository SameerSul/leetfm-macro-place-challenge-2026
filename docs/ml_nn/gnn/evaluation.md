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

For coldspot selector traces, use the dedicated pool-level diagnostic:

```bash
uv run python scripts/gnn/diagnose_coldspot_selector.py \
  --dataset ml_data/beyondppa_gnn/dataset.pt \
  --model ml_data/beyondppa_gnn/models/path/to/model.pt \
  --top-k 1 --top-k 4
```

This compares trace order, cheap field-delta order, stable random order,
exact-proxy oracle order, and the optional GNN model using
`candidate_pool_id` groups.

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

## Post-G6 Diagnostics

Stage G6 showed that the relocation-only G4 hook is legal but not promotable:
offline accepted-label recall improved, while full-suite average proxy and
runtime regressed. Before training or integrating a broader model, run the
ranking-value diagnostic:

```bash
uv run python scripts/gnn/diagnose_gnn_ranking.py \
  --dataset ml_data/beyondppa_gnn/dataset.pt \
  --g3-model ml_data/beyondppa_gnn/models/20260619_g3_candidate_baseline_min4/model.pt \
  --g4-model ml_data/beyondppa_gnn/models/20260619_g4_macro_net_ranker_v1/model.pt \
  --top-k 4 \
  --out ml_data/beyondppa_gnn/diagnostics/<run_id>.json
```

Required report fields:

- accepted-label top-k recall for trace order, G3, and G4;
- exact proxy-gain top-k recall for trace order, G3, and G4;
- mean rank of the best exact-gain candidate;
- cumulative top-k exact proxy gain;
- top-k overlap between G4 and trace order;
- all metrics split by benchmark and operator.

First report, 2026-06-19:

- `trace_gain@4=0.2513`;
- `g3_gain@4=0.5239`;
- `g4_gain@4=0.6318`;
- `g4_overlap_with_trace@4=0.1889`.

This means G4 is finding more exact-gain candidates offline, but it strongly
disagrees with the deterministic order. The closed-loop regression is therefore
likely caused by distribution shift, downstream pass interaction, or runtime
budget displacement rather than a simple inability to identify locally good
candidates.

Paired closed-loop trace comparison:

```bash
uv run python scripts/gnn/compare_gnn_trace_pairs.py \
  --heuristic ml_data/beyondppa_gnn/traces/20260619_postg6_ibm10_heuristic.jsonl \
  --gnn ml_data/beyondppa_gnn/traces/20260619_postg6_ibm10_gnn.jsonl \
  --out ml_data/beyondppa_gnn/diagnostics/20260619_postg6_ibm10_trace_pair.json
```

First four-benchmark paired result, 2026-06-19:

- `ibm01`: final delta `-0.0001`, hard propose-all accepts `0->0`;
- `ibm10`: final delta `-0.0001`, hard propose-all accepts `1->0`;
- `ibm12`: final delta `+0.0001`, hard propose-all accepts `4->1`;
- `ibm17`: final delta `+0.0002`, hard propose-all accepts `0->0`.

All GNN-ranked paired runs were VALID and had non-null `gnn_score` samples with
zero `gnn_rank_error` samples. The main observed issue is not inference failure;
it is that GNN top-k reordering can suppress deterministic hard propose-all
accepts, especially on `ibm12`, while later coldspot and micro-shift passes may
partially recover the lost proxy.

Follow-up `ibm12` sensitivity tests, 2026-06-19:

- `HIER_GNN_TOP_K=8`: VALID, final delta `+0.0106`, hard propose-all accepts
  `4->3`;
- `HIER_GNN_TOP_K=16`: VALID, final delta `+0.0106`, hard propose-all accepts
  `4->3`;
- `HIER_GNN_PRESERVE_TOP_N=12` with `HIER_GNN_TOP_K=4`: VALID, final delta
  `+0.0126`, hard propose-all accepts `4->3`;
- `PYTHONHASHSEED=0` did not fully stabilize upstream region-swap counts, so
  paired trace deltas must be interpreted as diagnostic signals, not clean
  causal A/B measurements.

Conclusion: smaller GNN prefixes and a preserved deterministic prefix are not
accepted improvements. The next improvement should first make diagnostic runs
more repeatable, or train/evaluate against closed-loop traces with repeated
seeds so the model is judged against run-to-run variance.

Repeatable diagnostic mode, 2026-06-20:

Set `HIER_DIAGNOSTIC_NO_DEADLINES=1` to disable local hierarchy relief
deadlines while keeping production defaults unchanged. This is diagnostic-only;
it is not challenge-safe for promotion because runtime can increase.

`ibm12` repeatability result:

- heuristic run A: VALID, final `2.1719`, region swaps `391`, hard propose-all
  accepts `3`;
- heuristic run B: VALID, final `2.1719`, region swaps `391`, hard propose-all
  accepts `3`;
- GNN run A: VALID, final `2.1707`, region swaps `391`, hard propose-all
  accepts `5`;
- GNN run B: VALID, final `2.1707`, region swaps `391`, hard propose-all
  accepts `5`.

Controlled conclusion: after removing local deadline noise, the GNN ranker has
a real positive `ibm12` signal for the hard propose-all hook: final delta
`-0.0012`, hard propose-all accepts `3->5`, with upstream region swaps
unchanged. The production G6 regression is therefore likely a budget/timing
interaction, not a pure ranking-quality failure.

Controlled four-benchmark no-deadline result:

- pure GNN top-k 32:
  - `ibm01`: delta `+0.0000`, hard propose-all accepts `0->0`;
  - `ibm10`: delta `+0.0019`, hard propose-all accepts `1->0`;
  - `ibm12`: delta `-0.0012`, hard propose-all accepts `3->5`;
  - `ibm17`: delta `+0.0000`, hard propose-all accepts `0->0`;
  - total delta `+0.0007`.
- additive GNN mode, `HIER_GNN_PRESERVE_TOP_N=16`,
  `HIER_GNN_TOP_K=8`, `HIER_GNN_EXTRA_TOP_K=8`:
  - `ibm01`: delta `+0.0000`, hard propose-all accepts `0->0`;
  - `ibm10`: delta `+0.0000`, hard propose-all accepts `1->1`;
  - `ibm12`: delta `-0.0024`, hard propose-all accepts `3->6`;
  - `ibm17`: delta `+0.0000`, hard propose-all accepts `0->0`;
  - total delta `-0.0024`.

The additive mode is the first controlled GNN integration shape that does not
displace deterministic accepts on this four-benchmark set. It costs extra exact
checks in the post-swap hard propose-all pass, so the next gate is a
production-mode timed smoke, not promotion.

Timed production-mode additive smoke, 2026-06-20:

- Mode: `HIER_GNN_PRESERVE_TOP_N=16`, `HIER_GNN_TOP_K=8`,
  `HIER_GNN_EXTRA_TOP_K=8`, with normal local deadlines.
- `ibm10`: VALID, current timed heuristic `1.6184`, additive `1.6192`, delta
  `+0.0008`; hard propose-all accepts `1->1`, exact checks `15->23`.
- `ibm12`: VALID, current timed heuristic `2.1857`, additive `2.1703`, delta
  `-0.0154`; hard propose-all accepts `4->4`, exact checks `4->4`.

Decision: mixed timed smoke. Keep additive mode default-off and do not run G6
promotion yet. The next useful test is repeated timed smoke or a cheaper
production guard that only enables additive GNN when the deterministic pass has
unused exact-check budget.
