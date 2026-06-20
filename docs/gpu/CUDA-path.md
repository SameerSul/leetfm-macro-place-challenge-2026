# CUDA Path - Archived

This document is historical. It described an opt-in CUDA proposal scorer for the
deleted proxy optimizer's hard-relocation loop.

The current production placer is hierarchy-only and does not pass
`RELOC_PROPOSE_ALL` through a main R2 loop. The active GPU-sensitive component
is DREAMPlace, launched through `src/dreamplace_bridge/`.

## Historical Finding

The CUDA proposal scorer proved that batched Torch scoring could run and match
the CPU proxy shape closely, but the search-policy A/B did not justify making it
production for IBM. That work should not be revived by default. If GPU work is
reopened, start from the current hierarchy pipeline and its cluster/region
objective.
