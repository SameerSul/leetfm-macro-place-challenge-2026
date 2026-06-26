# v2 - Sameer + Varrahan's Submission

Active placer for the Partcl/HRT Macro Placement Challenge.

**Current production mode (2026-06-24): hierarchy-only.** `MacroPlacer.place()`
always routes through `_hierarchy_floorplan()` in
`src/placer/pipeline/macro_placer.py`. The previous proxy-optimized production
path has been deleted: random candidate restarts, R2/2-opt/swap/cycle search,
generic LSMC exploration, generic cluster LSMC kicks, and the ML ranker package
are no longer part of active code.

The placer now deliberately preserves connected macro groups. It uses grouped
DREAMPlace to form a hierarchical global placement, legalizes hard macros in
cluster-consecutive order, classifies soft macros as owned or bridge, expands
hot cluster regions by congestion, runs bounded hard/soft relief, applies
exact-gated cluster decompression, and finishes with region-bounded swaps plus
round-level and post-swap micro-shift replay, post-swap hard and soft polish
passes, component-aware late cleanup scheduling, proxy-aware coldspot
tightening, and bounded survivor search. The exact proxy is still used for
evaluation and local gates, but it is no longer the primary design objective.

The placement objective note is in [docs/general/OBJECTIVES.md](docs/general/OBJECTIVES.md).
The hierarchy-integrated BeyondPPA structural objective notes and GNN trace
roadmap are in [docs/ml_nn/beyondppa_results/](docs/ml_nn/beyondppa_results/).

Current smoke reference:

```text
uv run evaluate src/main.py -b ibm10
proxy=1.1534  VALID  audit=pass  [~78s locally]
```

Current full IBM reference:

```text
uv run evaluate src/main.py --all
AVG 1.1999  17/17 VALID  0 overlaps  all hierarchy audits passed  [1147.08s locally]
```

Pass progression is now adaptive: the pipeline advances to the next stage when the
most recent exact proxy improvement is no longer better than
`HIER_PLATEAU_PROXY_GAIN`.

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
  -> classify soft macros as owned or bridge
  -> grouped DREAMPlace with synthetic cluster clique nets
  -> cluster-consecutive hard legalization
  -> soft relocation cleanup
  -> congestion-expanded hard/soft regions
  -> region-locked hard relocation + soft relocation relief
  -> exact-gated in-region micro-shift polish
  -> exact-gated cluster decompression
  -> region-bounded hard-hard / hard-soft / soft-soft swaps
  -> optional swap-round micro-shift replay
  -> post-swap micro-shift replay
  -> post-swap hard propose-all polish
  -> post-swap soft relocation polish
  -> component-aware strong soft repair scheduling
  -> proxy-aware coldspot tightening
  -> post-coldspot micro-shift replay
  -> bounded survivor-pool search
  -> final movable-macro in-bounds clamp
  -> return macro centers
```

Accepted hierarchy constants live in `src/utils/constants.py`:

```text
HIER_GROUP_WEIGHT=8
CLUSTER_MIN_EDGE=2
CLUSTER_MAX_FANOUT=8
HIER_RELOC_PROPOSE_HOT_K=32
HIER_POST_RELOC_PROPOSE_ALL=auto
HIER_POST_RELOC_PROPOSE_TOP_M=16
HIER_RELOC_PROPOSE_MIN_GAIN=0.0005
HIER_POST_SOFT_RELOC_TOP_K=256
HIER_POST_SOFT_RELOC_MIN_GAIN=0.0005
HIER_REGION_DENSITY=0.65
REGION_BIAS=1.0
HIER_REGION_ROUNDS=2
HIER_REGION_BUDGET_S=40
HIER_REGION_ESCAPE_MIN=0.002
HIER_SOFT_SWAP_K=48
HIER_COLDSPOT_BUDGET=0.0
HIER_COLDSPOT_TOTAL=0.0
HIER_PLATEAU_PROXY_GAIN=0.00005
HIER_COLDSPOT_MIN_FIELD_GAP=0.02
HIER_COLDSPOT_ROUNDS=8
HIER_OBJECTIVE_STRUCTURAL_WEIGHT=0.0
```

Default-on hierarchy behavior is no longer represented as constants; those
operators run unconditionally unless an explicit default-off experiment or
runtime environment variable says otherwise.

Runtime environment variables are intentionally limited. `SEED` is accepted by
`src/main.py` for reproducible runs, and `HIER_GNN_TRACE*` controls optional
JSONL trace output.

`HIER_OBJECTIVE_STRUCTURAL_WEIGHT` is the BeyondPPA-style structural ranking
constant inside the existing hierarchy relocation operators. It only reorders
candidates; legality, fixed-macro, region, hierarchy-quality, and exact-proxy
gates still decide accepted moves. `HIER_GNN_TRACE=1` writes JSONL traces for
future hierarchy-aware GNN training without changing placement output. Offline
Stage-G3 candidate baselines can be trained with
`scripts/train_gnn_baseline.py`; the accepted G3 artifact is default-off and is
not used at placement time. Offline Stage-G4 macro-net rankers can be trained
with `scripts/train_gnn_ranker.py`; the accepted G4 artifact is also
default-off. `HIER_GNN_RANK=1` enables the experimental Stage-G5
relocation-only candidate reordering hook; it is default-off and not promoted
after valid but regressive Stage-G6 closed-loop validation.
Post-G6 GNN diagnostics are ongoing. Smaller top-k and guarded-prefix variants
are valid but not accepted improvements.
`HIER_DIAGNOSTIC_NO_DEADLINES=1` is available for repeatable GNN diagnostics
only; it is not a production mode.
`HIER_GNN_EXTRA_TOP_K` is also default-off and experimental; it is used for
additive GNN diagnostics before timed smoke.

## Source Layout

```text
src/main.py                    evaluator-facing entrypoint
src/placer/pipeline/           hierarchy orchestration
src/placer/local_search/       cluster fields, relocation, coldspot kick helper
src/placer/scoring/            exact and incremental proxy scoring
src/placer/routing/            routing demand and congestion helpers
src/placer/legalize/           hard-macro legalization
src/utils/                     runtime config and accepted placement constants
src/dreamplace_bridge/         pb.txt <-> Bookshelf bridge and DP launcher
src/eda_io/                    LEF/DEF/Verilog/SDC/Liberty I/O layer
test/verification/             focused correctness checks
docs/general/                  current architecture, issues, progress history
docs/gpu/                      archived notes for removed proxy/GPU paths
docs/ml_nn/beyondppa_results/  active BeyondPPA/GNN trace notes and stage logs
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
- `docs/ml_nn/beyondppa_results/` - current deterministic BeyondPPA structural
  integration notes, stage results, GNN trace logging, and GNN roadmap.
- `docs/theory/LSMC.md`, `docs/gpu/` - archived references for deleted
  proxy-era work.
