#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-ml_data/beyondppa_gnn/models/20260620_g4_ibm_ng45_macro_net_v1/model.pt}"
TOP_K="${HIER_GNN_TOP_K:-32}"

if [[ "$#" -gt 0 ]]; then
  BENCHMARKS=("$@")
else
  BENCHMARKS=(ibm10 ibm12 ibm16 ibm18)
fi

for bench in "${BENCHMARKS[@]}"; do
  HIER_GNN_RANK=1 \
  HIER_GNN_MODEL="${MODEL}" \
  HIER_GNN_OPERATORS=relocation \
  HIER_GNN_TOP_K="${TOP_K}" \
  uv run evaluate src/main.py -b "${bench}"
done
