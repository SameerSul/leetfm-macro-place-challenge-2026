#!/bin/bash
# Batch runner: ibm02, ibm03, ibm04, ibm06, ibm07, ibm09 (3300s each)
# Run from project root: bash scripts/run_batch_v16_remaining.sh
# Expected total time: ~6 * 3300s = ~5.5 hours
# Note: ibm17 has n=760 > threshold=430 -> returns baseline instantly (skip full run)
# Launch AFTER ibm13/15/18 batch (PID 769) completes (~05:45 on 2026-05-03)

set -e
cd "$(dirname "$0")/.."

echo "[batch] Starting v16 remaining benchmarks: ibm02 ibm03 ibm04 ibm06 ibm07 ibm09"
echo "[batch] Total estimated time: ~5.5 hours"
date

echo "[batch] --- ibm02 ---"
python scripts/test_v16_ibm02.py > /tmp/ibm02_v16.txt 2>&1
echo "[batch] ibm02 done: $(tail -5 /tmp/ibm02_v16.txt)"

echo "[batch] --- ibm03 ---"
python scripts/test_v16_ibm03.py > /tmp/ibm03_v16.txt 2>&1
echo "[batch] ibm03 done: $(tail -5 /tmp/ibm03_v16.txt)"

echo "[batch] --- ibm04 ---"
python scripts/test_v16_ibm04.py > /tmp/ibm04_v16.txt 2>&1
echo "[batch] ibm04 done: $(tail -5 /tmp/ibm04_v16.txt)"

echo "[batch] --- ibm06 ---"
python scripts/test_v16_ibm06.py > /tmp/ibm06_v16.txt 2>&1
echo "[batch] ibm06 done: $(tail -5 /tmp/ibm06_v16.txt)"

echo "[batch] --- ibm07 ---"
python scripts/test_v16_ibm07.py > /tmp/ibm07_v16.txt 2>&1
echo "[batch] ibm07 done: $(tail -5 /tmp/ibm07_v16.txt)"

echo "[batch] --- ibm09 ---"
python scripts/test_v16_ibm09.py > /tmp/ibm09_v16.txt 2>&1
echo "[batch] ibm09 done: $(tail -5 /tmp/ibm09_v16.txt)"

echo "[batch] All done!"
date
echo "[batch] Results:"
for b in ibm02 ibm03 ibm04 ibm06 ibm07 ibm09; do
    grep "Final proxy=" /tmp/${b}_v16.txt 2>/dev/null | tail -1 | sed "s/^/  ${b}: /"
done
