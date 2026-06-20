#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-ml_data/beyondppa_gnn/datasets/20260620_ibm_ng45_heuristic.pt}"
G3_MODEL="${G3_MODEL:-ml_data/beyondppa_gnn/models/20260620_g3_ibm_ng45_v1/model.pt}"
G4_MODEL="${G4_MODEL:-ml_data/beyondppa_gnn/models/20260620_g4_ibm_ng45_macro_net_v1/model.pt}"
OUT="${OUT:-ml_data/beyondppa_gnn/diagnostics/20260620_ibm_ng45_g4_value_top4.json}"

uv run python scripts/gnn/diagnose_gnn_ranking.py \
  --dataset "${DATASET}" \
  --g3-model "${G3_MODEL}" \
  --g4-model "${G4_MODEL}" \
  --top-k "${TOP_K:-4}" \
  --out "${OUT}"
