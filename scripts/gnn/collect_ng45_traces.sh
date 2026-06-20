#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-20260620_ng45_heuristic_train}"
TRACE_MAX="${HIER_GNN_TRACE_MAX_CANDIDATES:-512}"

HIER_GNN_TRACE=1 \
HIER_GNN_TRACE_RUN="${RUN_ID}" \
HIER_GNN_TRACE_MAX_CANDIDATES="${TRACE_MAX}" \
uv run evaluate src/main.py --ng45
