#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-ml_data/beyondppa_gnn/datasets/20260620_ibm_ng45_heuristic.pt}"
G3_MODEL="${G3_MODEL:-ml_data/beyondppa_gnn/models/20260620_g3_ibm_ng45_v1/model.pt}"
OUT_DIR="${OUT_DIR:-ml_data/beyondppa_gnn/models/20260620_g4_ibm_ng45_macro_net_v1}"

uv run python scripts/gnn/train_gnn_ranker.py \
  --dataset "${DATASET}" \
  --g3-model "${G3_MODEL}" \
  --train-benchmark ibm01 \
  --train-benchmark ibm02 \
  --train-benchmark ibm03 \
  --train-benchmark ibm04 \
  --train-benchmark ibm06 \
  --train-benchmark ibm07 \
  --train-benchmark ibm08 \
  --train-benchmark ibm09 \
  --train-benchmark ibm11 \
  --train-benchmark ibm13 \
  --train-benchmark ibm14 \
  --train-benchmark ibm15 \
  ${EXTRA_TRAIN_BENCHMARK_ARGS:-} \
  --val-benchmark ibm10 \
  --val-benchmark ibm12 \
  --val-benchmark ibm17 \
  --epochs "${EPOCHS:-80}" \
  --hidden-size "${HIDDEN_SIZE:-32}" \
  --layers "${LAYERS:-2}" \
  --seed "${SEED:-11}" \
  --out-dir "${OUT_DIR}"
