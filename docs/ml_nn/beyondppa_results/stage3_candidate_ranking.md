# Stage 3: Hierarchy Candidate Ranking Integration

Implemented as a hierarchy objective term and left weight-zero by default.

## Changes

- Added hard and soft hierarchy relocation candidate ordering via
  `HIER_OBJECTIVE_STRUCTURAL_WEIGHT`.
- Added weight controls:
  - `HIER_KEEP_OUT_WEIGHT=0.2`
  - `HIER_GRID_ALIGN_WEIGHT=0.2`
  - `HIER_NOTCH_WEIGHT=0.6`
- Existing legality, fixed-macro, region, and exact proxy accept gates remain
  mandatory.
- Structural ranking affects proposal order only and does not create a second
  placement path.

## Verification

Default-off smoke:

```bash
uv run evaluate src/main.py -b ibm10
```

Result:

```text
proxy=1.6209  (wl=0.057 den=1.088 cong=2.040)  VALID  [47.81s]
```

Opt-in smoke:

```bash
HIER_OBJECTIVE_STRUCTURAL_WEIGHT=1 uv run evaluate src/main.py -b ibm10
```

Result:

```text
proxy=1.7706  (wl=0.049 den=1.257 cong=2.185)  VALID  [108.65s]
```

## Decision

Implemented but rejected for nonzero default weight. The hierarchy-integrated
term is valid, but weight `1.0` regressed `ibm10` proxy and runtime. Keep
`HIER_OBJECTIVE_STRUCTURAL_WEIGHT=0.0`.
