# Hierarchy GNN Dataset Schema

`scripts/gnn/build_gnn_dataset.py` converts schema-v1 JSONL traces into a
framework-neutral PyTorch payload. The current dataset schema is v2.

```bash
uv run python scripts/gnn/build_gnn_dataset.py \
  --trace-dir ml_data/beyondppa_gnn \
  --out ml_data/beyondppa_gnn/dataset.pt
```

For a single smoke trace:

```bash
uv run python scripts/gnn/build_gnn_dataset.py \
  --trace-path /tmp/hier_gnn_trace_smoke.jsonl \
  --out /tmp/hier_gnn_dataset.pt \
  --benchmark ibm01
```

The builder also writes `feature_schema.json` next to `--out` unless
`--schema-out` is supplied.

## Payload

The `.pt` file is a dictionary:

- `metadata`: dataset version, trace version, graph schema, trace files,
  benchmark names, graph count, example count, and accepted-example count.
- `feature_schema`: node, edge, and candidate feature names plus categorical id
  maps.
- `graphs`: one graph dictionary per benchmark.
- `examples`: stacked candidate examples that reference graphs by `graph_id`.

## Graphs

Each graph contains:

- `node_features`: `[num_nodes, 18]` float tensor.
- `edge_index`: `[2, num_edges]` long tensor.
- `edge_features`: `[num_edges, 6]` float tensor.
- `net_node_features`: `[num_nets, 6]` float tensor.
- `macro_net_edge_index`: `[2, num_macro_net_edges]` long tensor where row 0 is
  macro node id and row 1 is net node id.
- `macro_net_edge_features`: `[num_macro_net_edges, 7]` float tensor.
- `macro_cluster`: `[num_macros]` long tensor, `-1` for unclustered macros.
- `cluster_node`: `[num_clusters]` long tensor mapping cluster id to graph node.
- `bridge_softs`: mapping from soft macro id to bridge cluster ids.
- benchmark metadata: canvas, macro counts, cluster count, names.

Graph nodes include hard macros, soft macros, and one node per inferred hard
cluster. Graph edges include:

- netlist clique edges from the existing wirelength cache;
- bidirectional macro-cluster membership edges;
- local spatial-neighbor edges.

Schema v2 also includes the MacroDiff+-inspired hetero macro-net view used by
Stage G4:

- net nodes, one per wirelength-cache net;
- macro-net incidence edges;
- pin-offset edge features normalized by canvas max dimension;
- edge net weight, fanout, and driver-pin flag;
- net-node degree, macro degree, net weight, normalized HPWL x/y, and weighted
  HPWL pressure.

## Examples

`examples` contains:

- `graph_id`: graph index for each candidate.
- `source_node`: source macro or cluster node.
- `target_node`: target macro node when applicable, otherwise `-1`.
- `features`: `[num_examples, 27]` candidate feature tensor.
- `accepted`: boolean label.
- `proxy_delta`: exact proxy delta when known, otherwise `0`.
- `proxy_delta_known`: boolean mask for `proxy_delta`.
- `rejection_id`: categorical rejection reason.
- string sidecars: `operator`, `kind`, `benchmark`, `trace_file`.
- `trace_line`: source JSONL line number.

## Determinism

For the same trace files and benchmark roots, the builder produces identical
tensor shapes and tensor values. The verifier is:

```bash
uv run python test/verification/_verify_gnn_dataset_builder.py
```

## Scope

This dataset is training input for baseline rankers, GNN rankers, and expanded
hierarchy-flow assistant roles. It does not enable inference or change placement
behavior.
