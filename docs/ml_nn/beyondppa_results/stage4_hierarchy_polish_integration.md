# Stage 4: Exact-Gated Hierarchy Polish Integration

Implemented by removing the separate structural polish path.

## Changes

- Removed the standalone post-pipeline structural polish hook.
- Kept hierarchy's existing exact-gated polish mechanisms as the only polish
  path: micro-shift, relocation, swaps, decompression, and coldspot tightening.
- BeyondPPA-style structure participates through hierarchy candidate ordering
  before existing exact gates.

## Verification

No separate polish command remains. Verification is covered by the default-off
hierarchy smoke and the weighted candidate-ordering smoke in Stage 3.

## Decision

Accepted. There is no distinct BeyondPPA polish path.
