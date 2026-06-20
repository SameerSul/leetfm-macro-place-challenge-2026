#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-ml_data/beyondppa_gnn/datasets/20260620_ibm_ng45_heuristic.pt}"
OUT_DIR="${OUT_DIR:-ml_data/beyondppa_gnn/models/20260620_g3_ibm_ng45_v1}"

uv run python scripts/gnn/train_gnn_baseline.py \
  --dataset "${DATASET}" \
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
  --holdout-benchmark ibm16 \
  --holdout-benchmark ibm18 \
  --out-dir "${OUT_DIR}"
