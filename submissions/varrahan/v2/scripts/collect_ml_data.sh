#!/usr/bin/env bash
#
# Collect candidate-ranking training data for the per-operator XGBoost models.
#
# Runs the placer with ML_TRACE_PATH set, so every local-search candidate trial
# (hard relocation, soft relocation, hard 2-opt, and the other R2 passes) is
# written as a JSONL row with its pre-score features + score_gain/improves
# labels + group_id. See src/placer/ml/data_collection.py for the schema and
# src/placer/ml/dataset.py for the loaders.
#
# Data diversity comes from running --all (17 distinct benchmark trajectories)
# across several seeds (V2_SEED), each a different legalization/perturbation path.
# A single --all already emits millions of candidate rows, so a small seed sweep
# is plenty; seeds give cross-trajectory variety, not just volume.
#
# IMPORTANT (see memory: ml-trace-verification-gotcha): tracing perturbs timing,
# which perturbs deadline-gated passes, which perturbs scores. These runs are for
# DATA ONLY -- never read a placement score off a traced run.
#
# Usage (run from the repo root):
#   submissions/varrahan/v2/scripts/collect_ml_data.sh [--all|--ng45] [seed ...]
#   submissions/varrahan/v2/scripts/collect_ml_data.sh 42 43 44          # IBM (default)
#   submissions/varrahan/v2/scripts/collect_ml_data.sh --ng45 42 43 44   # NG45 (Tier 2)
#
# --ng45 traces are written with an "ng45_" filename prefix so they sit alongside
# the IBM traces in the same directory and the trainer can include or hold them
# out by name. Run both sets when you want cross-design variability (the NG45
# designs are structurally different from IBM and guard against overfitting).
#
set -euo pipefail

# Resolve repo root from this script's location (scripts/ -> v2 -> varrahan ->
# submissions -> root) so the relative paths the evaluator needs resolve.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"

PLACER="submissions/varrahan/v2/src/main.py"
OUT_DIR="submissions/varrahan/v2/ml_data/traces"
LOG_DIR="submissions/varrahan/v2/ml_data/logs"
mkdir -p "$OUT_DIR" "$LOG_DIR"

# Optional leading mode flag selects the benchmark set; default is IBM (--all).
MODE="--all"
case "${1:-}" in
  --all|--ng45) MODE="$1"; shift ;;
esac
TAG=""
[ "$MODE" = "--ng45" ] && TAG="ng45_"

# Seeds: remaining CLI args override the default sweep.
SEEDS=("$@")
if [ "${#SEEDS[@]}" -eq 0 ]; then
  SEEDS=(42 43 44)
fi

# Fewer, larger flushes than the 2048 default: millions of rows means the gzip
# append (one member per flush) is cheaper amortized at this size.
export ML_TRACE_FLUSH_ROWS="${ML_TRACE_FLUSH_ROWS:-20000}"

echo "=================================================================="
echo " ML data collection"
echo "   placer : $PLACER"
echo "   mode   : $MODE  (tag='$TAG')"
echo "   seeds  : ${SEEDS[*]}"
echo "   out    : $OUT_DIR"
echo "   logs   : $LOG_DIR"
echo "=================================================================="

START_TS=$(date +%s)
for seed in "${SEEDS[@]}"; do
  run_id="${TAG}s${seed}_$(date +%Y%m%d_%H%M%S)"
  trace_path="$OUT_DIR/${run_id}.jsonl.gz"
  log_path="$LOG_DIR/${run_id}.log"

  echo ""
  echo ">>> seed=$seed  run_id=$run_id"
  echo "    trace: $trace_path"
  echo "    log  : $log_path"

  # V2_SEED is read by submissions/varrahan/v2/src/main.py (default-preserving).
  # ML_RUN_ID feeds the {run_id} substitution in ML_TRACE_PATH.
  V2_SEED="$seed" \
  ML_RUN_ID="$run_id" \
  ML_TRACE_PATH="$OUT_DIR/{run_id}.jsonl.gz" \
    uv run evaluate "$PLACER" "$MODE" > "$log_path" 2>&1 \
    || echo "    WARN: seed=$seed run exited non-zero (see $log_path); continuing"

  # Per-run volume report (multi-member gz is fine for zcat).
  if [ -f "$trace_path" ]; then
    rows=$(zcat "$trace_path" 2>/dev/null | wc -l || echo "?")
    size=$(du -h "$trace_path" | cut -f1)
    echo "    done: $rows rows, $size"
  else
    echo "    WARN: no trace file produced at $trace_path"
  fi
done

ELAPSED=$(( $(date +%s) - START_TS ))
echo ""
echo "=================================================================="
echo " Collection complete in ${ELAPSED}s"
echo " Traces:"
du -ch "$OUT_DIR"/*.jsonl.gz 2>/dev/null || echo "   (none)"
echo "=================================================================="
