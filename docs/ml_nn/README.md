# ML And GNN Support

Learned models are offline or diagnostic only. No GNN or DQN policy is enabled
in the production placer.

## Active Files

- [`beyondppa_results/gnn_trace_schema.md`](beyondppa_results/gnn_trace_schema.md)
  defines candidate traces and plateau telemetry.
- [`beyondppa_results/gnn_dataset_schema.md`](beyondppa_results/gnn_dataset_schema.md)
  defines the graph dataset produced from candidate traces.
- `scripts/gnn/build_gnn_dataset.py` builds datasets.
- `scripts/gnn/train_gnn_baseline.py` trains the Stage-G3 non-GNN baseline.
- `scripts/gnn/train_gnn_ranker.py` trains the Stage-G4 macro-net ranker.
- The remaining `scripts/gnn/` tools analyze graph tension and compare or
  diagnose trace/model rankings.

Accepted default-off artifacts live under:

```text
ml_data/beyondppa_gnn/models/20260619_g3_candidate_baseline_min4/
ml_data/beyondppa_gnn/models/20260619_g4_macro_net_ranker_v1/
```

## Production Boundary

Candidate logging is enabled with `HIER_GNN_TRACE=1`. Plateau telemetry always
records buffered pass-level rows. Both streams include run/revision/PID
provenance; `HIER_PLATEAU_TRACE_PATH` redirects the plateau stream.

`HIER_GNN_RANK=1` enables the experimental relocation ranker, but it is not a
production setting: Stage-G6 was legal and audit-safe but regressed average
proxy and runtime. Any future learned signal must remain inside existing
hierarchy operators and may only rank or add bounded candidates. It must not
bypass legality, bounds, fixed macros, hierarchy regions, hierarchy-quality
checks, or exact-proxy acceptance.
