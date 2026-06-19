# Stage 2: Diagnostic Reporting

Implemented.

## Changes

- Added `test/diagnostic/_structural_metrics.py`.
- The diagnostic reports exact proxy plus structural components:
  - edge keepout
  - grid alignment
  - notch
  - weighted combined score
- By default it runs the active placer and measures the final placement.
- `RUN_PLACER=0` measures the benchmark initial placement without changing any
  placement output.

## Calibration Runs

```bash
RUN_PLACER=0 uv run python test/diagnostic/_structural_metrics.py ibm10
```

Result:

```text
ibm10: proxy=1.3397 structural=10.195095
  edge_keepout     50.008303
  grid_alignment   0.329084
  notch            0.212697
```

Final-placement diagnostics:

```bash
uv run python test/diagnostic/_structural_metrics.py ibm01
uv run python test/diagnostic/_structural_metrics.py ibm10
uv run python test/diagnostic/_structural_metrics.py ibm17
```

Results:

```text
ibm01: proxy=0.9403 structural=0.204403
  edge_keepout     0.028358
  grid_alignment   0.200181
  notch            0.264492

ibm10: proxy=1.6144 structural=0.217257
  edge_keepout     0.001181
  grid_alignment   0.249084
  notch            0.278673

ibm17: proxy=2.0967 structural=0.202782
  edge_keepout     0.000266
  grid_alignment   0.238188
  notch            0.258486
```

## Decision

Accepted as diagnostic-only. These scores are not calibrated enough to promote
any production default.
