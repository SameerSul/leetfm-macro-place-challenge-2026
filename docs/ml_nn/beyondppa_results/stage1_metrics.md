# Stage 1: Documentation And Metrics

Implemented.

## Changes

- Added the structural objective note now kept at
  `docs/ml_nn/beyondppa_results/beyondppa-structural-objectives.md`.
- Added `src/placer/local_search/structural_fields.py`.
- Added pure deterministic helpers:
  - `edge_keepout_penalty`
  - `grid_alignment_penalty`
  - `notch_penalty`
  - `combined_structural_penalty`
- Added focused synthetic tests in `test/verification/test_structural_fields.py`.

## Verification

```bash
uv run pytest test/verification/test_structural_fields.py -q
```

Result: 4 passed.

```bash
uv run python -m py_compile $(find src -type f -name "*.py")
```

Result: passed.

## Decision

Accepted as deterministic metric infrastructure. The metrics are inputs to
hierarchy objective decisions and diagnostics, not a separate placer.
