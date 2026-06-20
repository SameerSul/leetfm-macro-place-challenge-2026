# GNN Artifact Contract

Every trained GNN or baseline ranker must have a reproducible artifact bundle.

Store bundles under:

```text
ml_data/beyondppa_gnn/models/<model_id>/
```

Use model ids that encode date, stage, and short description:

```text
20260619_g3_mlp_candidate_v1
20260620_g4_graphsage_reloc_v1
```

## Required Files

```text
feature_schema.json
train_config.json
splits.json
trace_manifest.json
metrics.json
model.pt
README.md
```

## `feature_schema.json`

Must match the dataset builder output. Include:

- dataset schema version;
- trace schema version;
- node feature names;
- edge feature names;
- candidate feature names;
- categorical id maps.

## `train_config.json`

Include:

- model type;
- hidden size;
- graph layer count if applicable;
- losses and loss weights;
- optimizer;
- learning rate;
- batch size;
- epochs;
- random seed;
- training device;
- early stopping rule.

## `splits.json`

Include benchmark-level split membership:

- train benchmarks;
- validation benchmarks;
- holdout benchmarks.

## `trace_manifest.json`

Include:

- trace file paths;
- trace run ids;
- trace generation environment variables;
- code fingerprint or git commit;
- dataset build command;
- dataset artifact path.

## `metrics.json`

Include:

- offline metrics by operator;
- offline metrics by benchmark;
- runtime metrics;
- closed-loop smoke results when available;
- promotion decision: `default_off`, `candidate`, or `promoted`.

## `README.md`

Human-readable summary:

- what the model ranks;
- what traces it trained on;
- where it helps;
- where it regresses;
- exact commands used for offline eval and closed-loop eval.

## Promotion Rule

No model can become default-on without:

- artifact bundle complete;
- closed-loop `--all` result recorded;
- docs updated in `docs/general/PROGRESS.md`;
- explicit promotion decision recorded in the model README.
