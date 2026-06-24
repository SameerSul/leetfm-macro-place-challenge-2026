#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-ml_data/beyondppa_gnn/datasets/20260620_ibm_ng45_heuristic.pt}"
G3_MODEL="${G3_MODEL:-ml_data/beyondppa_gnn/models/20260620_g3_ibm_ng45_v1/model.pt}"
OUT_DIR="${OUT_DIR:-ml_data/beyondppa_gnn/models/20260620_g4_ibm_ng45_macro_net_v1}"
TRAIN_ARGS="${TRAIN_BENCHMARK_ARGS:-}"
if [[ -z "${TRAIN_ARGS}" ]]; then
  TRAIN_ARGS="$(uv run python scripts/gnn/train_benchmark_args.py "${DATASET}")"
fi

uv run python scripts/gnn/train_gnn_ranker.py \
  --dataset "${DATASET}" \
  --g3-model "${G3_MODEL}" \
  ${TRAIN_ARGS} \
  ${EXTRA_TRAIN_BENCHMARK_ARGS:-} \
  --val-benchmark ibm10 \
  --val-benchmark ibm12 \
  --val-benchmark ibm17 \
  --epochs "${EPOCHS:-80}" \
  --hidden-size "${HIDDEN_SIZE:-32}" \
  --layers "${LAYERS:-2}" \
  --seed "${SEED:-11}" \
  --out-dir "${OUT_DIR}"
