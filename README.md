# v2 - Varrahan's Submission

Active placer for the Partcl/HRT Macro Placement Challenge.

**Current production mode (2026-06-16): hierarchy-only.** `MacroPlacer.place()`
always routes through `_hierarchy_floorplan()` in
`src/placer/pipeline/macro_placer.py`. The previous proxy-optimized production
path has been deleted: random candidate restarts, R2/2-opt/swap/cycle search,
generic LSMC exploration, generic cluster LSMC kicks, and the ML ranker package
are no longer part of active code.

The placer now deliberately preserves connected macro groups. It uses grouped
DREAMPlace to form a hierarchical global placement, legalizes hard macros in
cluster-consecutive order, performs bounded region-locked hard/soft relief, and
optionally applies coldspot tightening to compact clusters further. The exact
proxy is still used for evaluation and local gates, but it is no longer the
primary design objective.

The placement objective note is in [docs/general/OBJECTIVES.md](docs/general/OBJECTIVES.md).

Current smoke reference:

```text
uv run evaluate src/main.py -b ibm10
proxy=1.7076  VALID  [~12s locally]
```

Historical proxy leaderboard numbers remain in `docs/general/PROGRESS.md` and
`docs/general/ISSUES.md`; they describe the removed proxy path and should not be
treated as the current system's output.

## Setup

```bash
git submodule update --init external/MacroPlacement
uv sync
uv pip install -r requirements.txt
```

`requirements.txt` installs `numba`, which keeps routing/scoring helpers fast.
The hierarchy path is much smaller than the retired proxy optimizer, but the
shared scoring and relocation code still benefits from it.

## Main Commands

```bash
# Single benchmark smoke
uv run evaluate src/main.py -b ibm10

# Full IBM run
uv run evaluate src/main.py --all

# Visualize a placement
uv run evaluate src/main.py -b ibm10 --vis

# Verify coldspot cluster kick helper used by hierarchy tightening
uv run python test/verification/_verify_coldspot_kick.py ibm10

# Bytecode sanity
uv run python -m py_compile $(find src -type f -name "*.py")
```

## Pipeline

```text
initial.plc / benchmark
  -> derive hard clusters through low-fanout hard-soft connectivity
  -> attach connected soft macros to clusters
  -> grouped DREAMPlace with synthetic cluster clique nets
  -> cluster-consecutive hard legalization
  -> soft relocation cleanup
  -> region-locked hard relocation + soft relocation relief
  -> optional coldspot cluster tightening
  -> final movable-macro in-bounds clamp
  -> return macro centers
```

Key environment knobs:

```text
V2_HIER_GROUP_WEIGHT=8
V2_CLUSTER_MIN_EDGE=2
V2_CLUSTER_MAX_FANOUT=8
V2_HIER_REGION_RELIEF=1
V2_HIER_REGION_DENSITY=0.65
V2_REGION_BIAS=1.0
V2_HIER_REGION_ROUNDS=2
V2_HIER_REGION_BUDGET_S=40
V2_HIER_COLDSPOT_KICK=1
V2_HIER_COLDSPOT_BUDGET=0.05
V2_HIER_COLDSPOT_TOTAL=0.15
V2_HIER_COLDSPOT_ROUNDS=8
```

`V2_SEED` is still accepted by `src/main.py` for reproducible runs.

## Source Layout

```text
src/main.py                    evaluator-facing entrypoint
src/placer/pipeline/           hierarchy orchestration
src/placer/local_search/       cluster fields, relocation, coldspot kick helper
src/placer/scoring/            exact and incremental proxy scoring
src/placer/routing/            routing demand and congestion helpers
src/placer/legalize/           hard-macro legalization
src/dreamplace_bridge/         pb.txt <-> Bookshelf bridge and DP launcher
src/eda_io/                    LEF/DEF/Verilog/SDC/Liberty I/O layer
test/verification/             focused correctness checks
docs/general/                  current architecture, issues, progress history
docs/gpu/, docs/ml_nn/         archived notes for removed proxy/ML/GPU paths
```

Deleted active subsystems:

- `src/placer/ml/`
- `src/placer/local_search/two_opt.py`
- `src/placer/local_search/soft_moves.py`
- `src/placer/local_search/hard_soft.py`
- generic `_lsmc_explore` / `_cluster_kick`
- swap/cycle scorer APIs once used only by the proxy path

## eda_io

The placer can still run from standard EDA inputs through `src/place_design.py`.
Inputs are converted to the same ICCAD04-style benchmark object, then the
hierarchy path runs unchanged.

```bash
uv run python src/place_design.py \
  --lef tech.lef --lef macros.lef --def floorplan.def \
  --out-def placed.def --out-tcl place_macros.tcl --report qor.rpt
```

See `src/eda_io/README.md` for parser and output details.

## Documentation Map

- `docs/general/DESIGN_FLOW.md` - current hierarchy flow.
- `docs/general/ARCHITECTURE.md` - current architecture and live modules.
- `docs/general/ISSUES.md` - current findings plus recent dead ends.
- `docs/general/PROGRESS.md` - chronological experiment history; older proxy
  scores are historical.
- `docs/theory/LSMC.md`, `docs/gpu/`, `docs/ml_nn/` - archived references for
  deleted proxy-era work.
