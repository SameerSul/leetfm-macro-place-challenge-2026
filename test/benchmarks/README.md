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

Synthetic names do not resolve under the challenge benchmark tree. The runner
therefore attaches both `benchmark._source_dir` for grouped DREAMPlace and
`benchmark._cached_plc` for exact scoring. This suite is diagnostic coverage;
it does not replace the IBM acceptance sequence.

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
