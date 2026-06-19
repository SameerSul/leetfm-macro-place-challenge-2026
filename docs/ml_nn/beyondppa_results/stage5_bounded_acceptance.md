# Stage 5: Bounded Structural Acceptance

Deferred.

## Changes

- Removed the standalone bounded structural acceptance hook.
- The hierarchy flow still accepts moves through existing exact-gated operators.
- A future bounded structural accept mode must be implemented inside those
  hierarchy accept gates, not as a post-processing pass.

## Verification

No bounded structural acceptance mode is active.

## Decision

Do not implement bounded proxy-regression acceptance until the hierarchy
candidate-ordering term has useful multi-benchmark evidence.
