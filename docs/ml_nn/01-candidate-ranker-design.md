# Candidate Ranker - Archived

This document is historical. The proxy-path ML ranker and all `src/placer/ml/`
code have been deleted from active code.

## What It Was

The ranker predicted which proxy-path local-search candidates were worth
exact-scoring. It never accepted placements directly; the exact proxy gate made
the final decision.

The deleted design targeted hard relocation, soft relocation, hard 2-opt,
group-level gating, target-level ranking, and trace-driven XGBoost/LambdaMART
training.

## Why It Was Removed

The selected system is now hierarchy-only. It does not run the old R2 candidate
search, so the ranker has no active integration point. Historical validation and
design rationale remain in `docs/general/PROGRESS.md` and
`docs/general/ISSUES.md`.
