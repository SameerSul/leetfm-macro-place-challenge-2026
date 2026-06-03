#!/bin/bash
# Usage: bash run_eval.sh ibm01 [ibm04 ...]
# Runs evaluate for one or more benchmarks via the project venv.
cd "/mnt/c/Users/Sameer/Desktop/Claude Code Projects/macro-place-challenge-2026"
UV=/home/sameersul/.local/bin/uv
for bm in "$@"; do
    $UV run evaluate submissions/varrahan/v2/src/submit.py -b "$bm" 2>&1
done
