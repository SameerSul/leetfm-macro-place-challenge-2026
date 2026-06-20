#!/usr/bin/env bash
set -euo pipefail

IBM_RUN_ID="${IBM_RUN_ID:-20260620_fullsuite_heuristic_train}"
NG45_RUN_ID="${NG45_RUN_ID:-20260620_ng45_heuristic_train}"
TRACE_DIR="${TRACE_DIR:-ml_data/beyondppa_gnn/training_traces/20260620_ibm_ng45}"
OUT="${OUT:-ml_data/beyondppa_gnn/datasets/20260620_ibm_ng45_heuristic.pt}"

mkdir -p "${TRACE_DIR}" "$(dirname "${OUT}")"
rm -f "${TRACE_DIR}"/*.jsonl
cp "ml_data/beyondppa_gnn/${IBM_RUN_ID}.jsonl" "${TRACE_DIR}/"
uv run python scripts/gnn/normalize_ng45_trace.py \
  --in "ml_data/beyondppa_gnn/${NG45_RUN_ID}.jsonl" \
  --out "${TRACE_DIR}/${NG45_RUN_ID}_normalized.jsonl"

uv run python scripts/gnn/build_gnn_dataset.py \
  --trace-dir "${TRACE_DIR}" \
  --benchmark-root external/MacroPlacement/Testcases/ICCAD04 \
  --benchmark-root external/MacroPlacement/Flows/NanGate45 \
  --out "${OUT}"
