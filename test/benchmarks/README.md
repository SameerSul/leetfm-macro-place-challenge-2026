# Synthetic anti-overfitting benchmark suite

The 17 IBM ICCAD04 benchmarks the leaderboard scores against are homogeneous:
square canvases, zero fixed macros, identical routing capacities (H=66 /
V=107 routes/um), ~0.8 total utilization, hundreds of small lognormal-ish
macros, and a hand-tuned spread seed from a prior EDA flow. A placer tuned
only on them can silently overfit to those properties and fall over on
realistic designs. This suite generates testcases that vary exactly those
axes, one at a time, in the same ICCAD04 protobuf format - so they load and
score through the unmodified TILOS `PlacementCost` evaluator and the standard
`macro_place` loader.

## The benchmarks

| name | axis probed | hard | soft | nets | canvas (um) | grid | notes |
|---|---|---|---|---|---|---|---|
| syn01_wide | non-square canvas (2.6:1) | 280 | 1000 | 14k | 65x25 | 24x62 | IBM canvases are all square |
| syn02_fixed | pre-placed fixed macros | 266 | 950 | 13k | 42x42 | 42x42 | 6 large fixed blocks (corners + center); IBM has zero fixed macros |
| syn03_sram | few large uniform macros + low routing capacity | 40 | 550 | 16k | 55x55 | 36x36 | commercial/bp_quad-like; routes 12.5/13.5 |
| syn04_dense | high utilization (~0.90 total) | 300 | 1100 | 15k | 38x38 | 40x40 | little whitespace for legalization |
| syn05_sparse | low utilization (~0.33 total) | 190 | 700 | 11k | 62x62 | 40x40 | congestion driven by topology, not packing |
| syn06_cluster | strong Rent-style clustering | 320 | 1200 | 16k | 45x45 | 43x43 | 9 communities, 92% intra-cluster nets |
| syn07_ports | one-sided I/O pull | 250 | 900 | 13k | 40x40 | 41x41 | 320 ports, 70% on LEFT edge, 25% port-driven nets |
| syn08_routes | inverted routing capacity | 280 | 1000 | 14k | 45x45 | 43x43 | H=107 > V=66 (IBM is always V-dominant) |
| syn09_seedless | random initial seed | 280 | 1000 | 14k | 45x45 | 43x43 | no good seed to lean on; netlist locality kept coherent |
| syn10_xl | scale stress | 820 | 2000 | 26k | 80x80 | 50x50 | beyond ibm17 in macro count and grid cells |

Seed placements are shelf-packed spreads with jitter (IBM-seed-like; small
overlap counts are normal - ibm01's own seed has 69) except `syn09_seedless`,
which scrambles positions after the netlist is built so locality structure
survives. Net topology uses gaussian sink locality around each driver plus a
cluster preference, calibrated so seed costs land in the IBM seed regime
(syn01 seed: wl 0.072 / den 0.885 / cong 1.42 vs ibm01 seed: 0.064 / 0.818 /
1.14).

## Usage

```bash
# regenerate the suite (deterministic per-benchmark RNG seeds)
uv run python test/benchmarks/generate_benchmarks.py

# evaluate v2 on the whole suite + render visualizations
uv run python test/benchmarks/run_synthetic.py

# single benchmark, custom placer or budget
uv run python .../run_synthetic.py -b syn02_fixed
uv run python .../run_synthetic.py --placer src/main.py --budget 60

# just look at the benchmarks themselves (seed scoring + vis, no placer)
uv run python .../run_synthetic.py --initial-only

# same runner on the 17 IBM benchmarks (writes results_ibm.json; slow - one
# placer run per benchmark)
uv run python .../run_synthetic.py --ibm

# per-metric impact analysis: which of wirelength/density/congestion is
# responsible for the remaining cost, per benchmark and overall. Toggle the
# benchmark groups in/out to compare.
uv run python test/benchmarks/analyze_impact.py
uv run python .../analyze_impact.py --no-synthetic   # IBM only
uv run python .../analyze_impact.py --no-ibm         # synthetic only
```

Outputs:

- `vis/<name>_initial.png` - the benchmark itself: seed placement, density
  heatmap, congestion heatmap (the standard 3-panel `visualize_placement`)
- `vis/<name>_placed.png` - the placer's result on the same 3 panels
- `results.json` - per-benchmark cost breakdown (initial vs placed, deltas,
  validity, out-of-bounds details)
- `results_ibm.json` - same schema for `--ibm` runs

These outputs are generated on demand and are intentionally not kept in the
working tree after cleanup.

Note: these scores are **not comparable to the IBM leaderboard numbers** -
they are for relative comparison between placer versions and for spotting
hard failures (invalid placements, score regressions on a specific axis).

## How the runner wires into v2

`benchmark.name` for synthetic cases doesn't resolve under
`external/MacroPlacement/Testcases/ICCAD04/`, so v2's `_load_plc` would
return None and bail to baseline. The runner sets `benchmark._cached_plc`
(the cache attribute `_load_plc` already honors) so v2 gets exact scoring.
DREAMPlace seeding is skipped automatically (its ICCAD04 path check fails),
which mirrors how v2 would behave on any unseen benchmark.

## Findings log

The first entries below refer to proxy-path modules that were later deleted
(`soft_moves.py`, `two_opt.py`, `hard_soft.py`). They are retained as historical
overfitting findings; the current hierarchy-only placer no longer contains those
passes.

- **2026-06-09, v2 @ 90s budget, first full run:** 9 of 10 benchmarks INVALID
  with small out-of-bounds overhangs (0.15-0.52um, 1-15 macros each; only
  `syn05_sparse` survived). Initial hypothesis was a square-canvas assumption
  (`syn01_wide` failed first), but square benchmarks failed too. Probing
  `syn03_sram` showed every OOB macro was **soft**. Root cause: the soft-2opt
  swap (`src/placer/local_search/soft_moves.py`) exchanged positions with no
  bounds check, so a larger soft macro inheriting a smaller macro's edge-flush
  slot overhung the canvas by `hw_large - hw_small`. IBM never exposes this
  because its hand-tuned seeds keep soft macros off the canvas edge - a
  textbook overfit to the seed style. Fixed by clamping swap targets per-macro
  (soft_moves.py) and tightening the EPS=0.05 overhang tolerance in the hard
  bounds checks (two_opt.py, relocation.py, legalize/swap.py) to strict, since
  `validate_placement` has zero tolerance. Verified: syn03 VALID with a
  slightly better proxy (4.3063 vs 4.3113); ibm01 unaffected (0.9111 VALID).
- **2026-06-09, second leak, same shape:** rerun after the fix left only
  `syn07_ports` INVALID (1 macro, 0.253um vertical). `hard_soft.py` (HXS
  cross-swap + HS3 3-cycle) bounds-checked only the HARD macro's destination -
  the soft macros inheriting slots in the exchange were never checked. Fixed
  identically (clamp each soft's inherited slot by its own half-size, strict
  hard bounds). All 10 synthetics now VALID; ibm01 still 0.9111 VALID.
- **2026-06-09, budget overrun at scale:** `syn10_xl` (820 hard macros) ran
  **504s against a 90s budget** and `syn09_seedless` ran 173s - the per-phase
  deadline logic does not bound whatever dominates at this scale. The IBM
  suite (max 786 macros, and DREAMPlace handles the big ones) never shows
  this; an `--all`-style 17-benchmark run made of syn10-sized cases would
  blow the 1-hour harness cap.
- **2026-06-09, seed dependence:** `syn09_seedless` (random seed, identical
  netlist style to syn08) lands at proxy 3.27 vs ~1.14 when a coherent seed
  exists. v2 recovers overlaps (240 -> 0) but cannot rebuild global structure
  from a bad seed - consistent with CLAUDE.md's "initial.plc is already a
  good seed" note, and a real risk on any benchmark without a curated seed.

## Files

- `generate_benchmarks.py` - suite generator (configs in `SUITE` at the top)
- `run_synthetic.py` - evaluation + visualization runner (synthetic or `--ibm`)
- `analyze_impact.py` - per-metric impact analysis across results JSONs, with
  `--no-synthetic` / `--no-ibm` group toggles
- `sample_initial.png` / `sample_placed.png` - one committed example of the
  visualization output (syn02_fixed); the full `vis/` set is gitignored
- Generated by `generate_benchmarks.py` and gitignored:
  - `testcases/<name>/` - netlist.pb.txt + initial.plc (ICCAD04 format)
  - `processed/<name>.pt` - Benchmark tensor mirrors (same format as
    `benchmarks/processed/public/*.pt`, including port positions)
  - `metadata/<name>.json` - generation config + axis description per benchmark
- Generated by `run_synthetic.py` and gitignored:
  - `vis/<name>_initial.png` / `vis/<name>_placed.png`
  - `results.json` / `results_ibm.json`

`testcases/`, `processed/`, `metadata/`, `vis/`, and result JSONs are **not
committed** (~140MB total; see the root `.gitignore`). Regenerate them with the
commands at the top of Usage - generation is deterministic, so the rebuilt suite
is identical.
