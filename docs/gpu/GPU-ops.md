# GPU And LSMC Notes - Archived

This document records removed proxy-path GPU/LSMC work. It is not the current
production flow.

As of 2026-06-16:

- generic `V2_GPU_EXPLORE` LSMC is deleted,
- generic cluster kicks are deleted,
- ML ranker defaults are deleted,
- the active `MacroPlacer.place()` path is hierarchy-only,
- `_coldspot_cluster_kick()` remains only as a hierarchy-tightening helper.

The current production GPU-sensitive component is DREAMPlace itself, launched
through `src/dreamplace_bridge/`. The Python placement pipeline does not run a
batched CUDA LSMC or CUDA proposal-ranking loop.

## Current Relevant Checks

```bash
uv run python -m py_compile $(find src -type f -name "*.py")
uv run python test/verification/_verify_coldspot_kick.py ibm10
uv run evaluate src/main.py -b ibm10
```

## Historical Summary

The removed proxy path explored:

- CUDA hard-relocation proposal ranking,
- multi-incumbent final LSMC,
- kick pre-screening,
- generic cluster-coherent gather/translate kicks,
- serial multi-chain probes.

The useful conclusion was negative for the current target: proxy-gated
exploration favors spread placements, while the selected system is a
hierarchy-preserving floorplan. Future GPU work should start from the current
hierarchy pipeline instead of restoring the old proxy LSMC stack by default.
