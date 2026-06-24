#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-ml_data/beyondppa_gnn/datasets/20260620_ibm_ng45_heuristic.pt}"
OUT_DIR="${OUT_DIR:-ml_data/beyondppa_gnn/models/20260620_g3_ibm_ng45_v1}"
TRAIN_ARGS="${TRAIN_BENCHMARK_ARGS:-}"
if [[ -z "${TRAIN_ARGS}" ]]; then
  TRAIN_ARGS="$(uv run python scripts/gnn/train_benchmark_args.py "${DATASET}")"
fi

uv run python scripts/gnn/train_gnn_baseline.py \
  --dataset "${DATASET}" \
  ${TRAIN_ARGS} \
  ${EXTRA_TRAIN_BENCHMARK_ARGS:-} \
  --val-benchmark ibm10 \
  --val-benchmark ibm12 \
  --val-benchmark ibm17 \
  --holdout-benchmark ibm16 \
  --holdout-benchmark ibm18 \
  --out-dir "${OUT_DIR}"
