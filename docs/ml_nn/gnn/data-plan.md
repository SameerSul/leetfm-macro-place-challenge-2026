# GNN Data Plan

The GNN is only as useful as its traces and labels. Data work must come before
model complexity.

## Trace Generation

Generate schema-v1 traces with:

```bash
HIER_GNN_TRACE=1 \
HIER_GNN_TRACE_RUN=<run_id> \
HIER_GNN_TRACE_MAX_CANDIDATES=512 \
uv run evaluate src/main.py -b <benchmark>
```

Use a fresh `HIER_GNN_TRACE_RUN` or `HIER_GNN_TRACE_PATH` for each collection
run. The trace writer appends to the selected JSONL path, and the dataset
builder validates every row in that file.

Minimum benchmark set before G3:

- `ibm01`
- `ibm10`
- `ibm12`
- `ibm17`

Then expand to all 17 IBM benchmarks.

Store traces under:

```text
ml_data/beyondppa_gnn/traces/
```

Use run ids that encode date, benchmark, and config, for example:

```text
20260619_ibm10_schema1_default
```

## Dataset Build

Build datasets with:

```bash
uv run python scripts/gnn/build_gnn_dataset.py \
  --trace-dir ml_data/beyondppa_gnn/traces \
  --out ml_data/beyondppa_gnn/dataset.pt
```

Single-trace smoke:

```bash
uv run python scripts/gnn/build_gnn_dataset.py \
  --trace-path /tmp/hier_gnn_trace_smoke.jsonl \
  --out /tmp/hier_gnn_dataset.pt \
  --benchmark ibm01
```

Verifier:

```bash
uv run python test/verification/_verify_gnn_dataset_builder.py
```

Stage-G3 baseline smoke:

```bash
uv run python test/verification/_verify_gnn_baseline.py
```

Held-out baseline run:

```bash
uv run python scripts/gnn/train_gnn_baseline.py \
  --dataset ml_data/beyondppa_gnn/dataset.pt \
  --train-benchmark ibm02 \
  --train-benchmark ibm03 \
  --val-benchmark ibm10 \
  --val-benchmark ibm12 \
  --out-dir ml_data/beyondppa_gnn/models/<model_id>
```

## MacroDiff+-Inspired Feature Work

Before G4, extend the dataset builder beyond the current macro/cluster graph:

- Add net nodes.
- Add macro-net edges.
- Add edge features:
  - pin offset x/y;
  - normalized net weight;
  - optional pin role if available.
- Add net-node features:
  - degree / fanout;
  - normalized HPWL for the current placement state;
  - optional congestion or density pressure.
- Keep existing macro, cluster, spatial-neighbor, and candidate scalar features.

This work is inspired by
[MacroDiff+](macrodiff_plus_notes.md), but the labels remain candidate-ranking
labels from our hierarchy operators, not full-placement denoising targets.

## Split Policy

Split by benchmark, not by random candidate rows. Random row splits leak
benchmark-specific candidate pools and pass states into validation.

Initial split:

- Train: most IBM benchmarks with complete traces.
- Validation: `ibm10`, `ibm12`.
- Smoke/holdout: `ibm01`, `ibm17`.

Final split must be recorded in:

```text
ml_data/beyondppa_gnn/splits.json
```

## Required Audits

Before training:

- Count examples per benchmark.
- Count examples per operator.
- Count accepted examples per operator.
- Count examples with known proxy delta.
- Report rejection reason distribution.
- Report candidate-pool sizes.
- Report top-k oracle recall by operator.
- Report pass-level runtime and exact-score counts when available.

If accepted examples are too sparse for an operator, do not train a model for
that operator yet. Collect more traces or keep that operator disabled for GNN
ranking.

## Expanded-Role Data

The expanded roles in [expansion-plan.md](expansion-plan.md) need additional
labels beyond candidate accept/reject.

Add trace fields for:

- operator state before and after each pass;
- operator skipped/running labels;
- per-pass runtime;
- exact-score count;
- candidate count before and after GNN expansion;
- region boxes and region deltas;
- deterministic soft ownership and bridge roles;
- hierarchy-quality delta for accepted and rejected candidates when practical;
- saved-score estimates for risk and budget decisions.

Do not block G3 on these fields. Add them before training operator-selection,
region-guidance, soft-role, or budget-allocation heads.

## Post-G6 Improvement Data

The next ranker needs data that explains why offline gains did not translate
into a promoted closed-loop result.

Collect and preserve:

- heuristic traces and `HIER_GNN_RANK=1` traces for the same benchmark set;
- `gnn_score` on sampled relocation candidates when ranking is enabled;
- `gnn_rank_error` on sampled relocation candidates when ranker inference
  fails and the deterministic order is preserved;
- exact `proxy_delta` for accepted and rejected scored candidates;
- trace provenance sufficient to compute the rank of the best exact-gain
  candidate inside each pool;
- per-pass accepted move count before and after GNN reordering;
- pass-level runtime and exact-score count before and after GNN reordering;
- final benchmark proxy, validity, overlap count, and runtime;
- top-k overlap between deterministic trace order and GNN order.

Use these labels to separate three failure modes:

- model miss: the best exact-gain candidate is ranked below the tested prefix;
- integration miss: the model ranks a locally good candidate, but downstream
  passes get worse;
- budget miss: inference or changed ordering displaces later useful work.

Do not train a replacement model only on heuristic-generated traces if it is
intended to run after `HIER_GNN_RANK=1` changes the state distribution.

## Leakage Rules

Do not:

- mix candidate rows from the same benchmark into both train and validation;
- train on traces generated by a model and compare against heuristic-only traces
  without recording that distinction;
- overwrite trace files used by a model artifact.

Do:

- keep trace IDs immutable once referenced by a model;
- record environment variables used to generate traces;
- record the code fingerprint used to generate traces.
