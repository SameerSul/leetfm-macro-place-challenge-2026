# AGENTS.md

This file gives coding agents the context to work productively in this repository on the first try. It is a living document - when something here turns out to be wrong or missing, fix it immediately rather than working around it.

## What this repo is

Submission to the **Partcl/HRT Macro Placement Challenge** (deadline May 21, 2026, $20K grand prize). The historical objective was to minimize proxy cost (lower is better; RePlAce avg **1.4578**), but the current selected system, **VivaPlace**, is a hierarchy-preserving macro placer.

The active submission now lives at the repository root: `src/`, `docs/`,
`test/`, `scripts/`, and `ml_data/`. The prior `system/v1/` checkpoint may be
absent after the root-layout migration; if present, treat it as frozen /
read-only.

**Current production mode (2026-07-15): hierarchy-only.** `MacroPlacer.place()`
always routes through `_hierarchy_floorplan()` in
`src/placer/pipeline/macro_placer.py` and raises if grouped DREAMPlace is not
available. The old proxy path has been deleted: candidate restarts, R2/2-opt,
hard-soft/soft swap and cycle passes, generic LSMC, generic cluster kicks, ML
ranker defaults, and their proxy-only verifiers are not active code.

Current verified result with normal BB/cache behavior:
`uv run evaluate src/main.py --all` = **AVG 1.1205**, 17/17 VALID, 0
overlaps, all final hierarchy audits passed, **541.67s**. The exact-scored seed
portfolio filters candidates through an independent six-component hierarchy
contract relative to legalized `initial.plc`, and the same contract is enforced
against the selected seed throughout relief and final rollback. The portfolio
includes a default-on constraint-graph legalization alternative for
`initial.plc`, and hard-hard / hard-soft swap sets use exact batched scoring.
Plateaued late soft cleanup also tests a bounded compound related-soft move:
every member stays in-region, the complete state must pass the rich hierarchy
contract, and exact incremental scoring occurs only after the group is formed.
The ordinary post-swap soft relocation pass is skipped after two attributable
full suites produced zero gain in 34 runs; its time remains as deadline and
final-audit headroom.
Passes advance on
gain (`HIER_PLATEAU_PROXY_GAIN=0.00005`) rather
than fixed repeat counts, and a final hierarchy-quality audit rolls back to
the best saved audit-passing checkpoint if the post-search state drifts too
far from the selected hierarchy seed. See `docs/general/ARCHITECTURE.md` for
the full pipeline, `docs/general/ISSUES.md` for current gaps, and
`docs/general/PROGRESS.md` for rejected or superseded experiments.

NG45 explicit hierarchy-tag check: `uv run evaluate src/main.py --ng45` =
**AVG 0.7252**, 4/4 VALID, 0 overlaps, all hierarchy audits passed, 232.41s;
`uv run python
test/verification/_verify_ng45_hierarchy_tags.py` passes. The hierarchy model
uses slash-separated instance-path prefixes when macro names provide useful
coverage, then falls back to inferred connectivity on flat-name benchmarks.

BeyondPPA-style work is integrated into the hierarchy path, not as a separate
placer. The current shipped piece is default-off hierarchy relocation candidate ordering
(`HIER_OBJECTIVE_STRUCTURAL_WEIGHT=0.0` constant). The trace logger is
default-off through the `HIER_GNN_TRACE=0` runtime environment variable.
Pass-level plateau telemetry always records buffered, attributable schema-v2
rows for future ML/DL scheduling work. It writes to
`ml_data/beyondppa_gnn/plateau/plateau_telemetry.jsonl` unless
`HIER_PLATEAU_TRACE_PATH` is supplied.
Stage-G3 offline baseline tooling lives in `scripts/gnn/train_gnn_baseline.py`, and
the accepted default-off baseline artifact is
`ml_data/beyondppa_gnn/models/20260619_g3_candidate_baseline_min4/`. Stage-G4
offline macro-net ranker tooling lives in `scripts/gnn/train_gnn_ranker.py`, and
the accepted default-off graph-ranker artifact is
`ml_data/beyondppa_gnn/models/20260619_g4_macro_net_ranker_v1/`. A default-off
Stage-G5 relocation-only candidate reordering hook exists behind
`HIER_GNN_RANK=1`; Stage G6 full-suite validation was legal but not promoted
because average proxy and runtime regressed. No inference ranker is default-on,
and no DQN policy is active in placement.

For the full problem statement see [`README.md`](README.md). For the API contract see [`SETUP.md`](SETUP.md). For experiment history and known-good numbers see [`PROGRESS.md`](docs/general/PROGRESS.md). For the placement objectives that should guide the hierarchy flow, see [`OBJECTIVES.md`](docs/general/OBJECTIVES.md). Do not duplicate that content here.

## Common commands

```bash
# Setup (run once - submodule is required, no-op evaluator otherwise)
git submodule update --init external/MacroPlacement
uv sync
scripts/dreamplace/bootstrap.sh all
# Existing install/ABI check without rebuilding:
scripts/dreamplace/bootstrap.sh preflight
# Optional mirror install if the environment was not created by uv sync. Numba is
# a first-class pyproject dependency; missing numba now raises unless
# ALLOW_NUMBA_FALLBACK=1 is set for slow diagnostic-only runs.
uv pip install -r requirements.txt

# Single benchmark - fastest feedback loop, use this while iterating
uv run evaluate src/main.py -b ibm10

# All 17 IBM benchmarks - current hierarchy system, not the old proxy headline
uv run evaluate src/main.py --all

# NG45 commercial designs (Tier 2, OpenROAD inputs)
uv run evaluate src/main.py --ng45

# Visualize a placement
uv run evaluate src/main.py -b ibm01 --vis

# Compare active placer against the v1 checkpoint, if system/v1 is present
uv run python scripts/compare_placers.py system/v1/placer.py src/main.py

# Compare two placers head-to-head
uv run python scripts/compare_placers.py path/to/placer_a.py path/to/placer_b.py

# Smoke tests (project-level)
uv run pytest test/

# Run a diagnostic or verification script
uv run python test/verification/_verify_coldspot_kick.py ibm10

# Bytecode sanity after edits
uv run python -m py_compile $(find src -type f -name "*.py")

# Synthetic anti-overfitting suite (generate once, then run / analyze)
uv run python test/benchmarks/generate_benchmarks.py
uv run python test/benchmarks/run_synthetic.py          # synthetic designs
uv run python test/benchmarks/run_synthetic.py --ibm    # IBM cross-check
uv run python test/benchmarks/analyze_impact.py         # cost-term breakdown

# eda_io: run the placer on standard EDA inputs (LEF/DEF/Verilog/SDC/Liberty)
uv run python src/place_design.py \
    --lef tech.lef --def floorplan.def --out-def placed.def --out-tcl place.tcl --report qor.rpt

# eda_io tests (pytest is not in the project venv - use --with)
uv run --with pytest python -m pytest test/eda_io/ -v
```

If `uv` is not on PATH, fall back to `pip install -e .` and replace `uv run` with `python -m`.

## File modification scope

**IMPORTANT - active code now lives at the repository root.** Keep normal code
work inside `src/**`, `test/**`, `scripts/**`, `ml_data/**`,
`dreamplace_build/**`, and `dreamplace_src/**`. Root-level documentation,
package-management, and tool-configuration files are also writable.

Writable:

- `src/**` - evaluator entrypoint, placer package, eda_io, DREAMPlace bridge
- `docs/**` - active documentation and experiment notes
- `test/**` - diagnostics, verification scripts, synthetic benchmark tools
- `scripts/**` - active helper scripts
- `ml_data/**` - historical traces/models/logs and generated comparison data
- `dreamplace_build/**` - DREAMPlace install tree (rebuilds / patches allowed)
- `dreamplace_src/**` - DREAMPlace source (custom forks / modifications allowed)
- Root documentation: `*.md`, including `AGENTS.md`, `CLAUDE.md`, `README.md`,
  and other root-level docs.
- Root package/config/ignore files: `pyproject.toml`, `requirements*.txt`,
  `uv.lock`, `.gitignore`, `.python-version`, and similar root-level files for
  dependency management, formatting, linting, typing, tests, or tool settings.

Read-only (agents may read but must not edit, create, move, or delete):

- **`system/v1/**`**, if present - frozen v17 checkpoint, kept for comparison.
  Treat as if it lived under `external/`.
- Framework, benchmark, and challenge files outside the active submission:
  `macro_place/`, `external/`, `benchmarks/`, generated benchmark inputs, and
  challenge/evaluator assets. Root-level documentation and package/config files
  are writable under the rules above.

If a task seems to require modifying a read-only file (e.g. fixing a bug in
`macro_place/`, correcting challenge metadata, or porting/tweaking something
from `v1/`), stop and surface the proposed change to the user instead of
editing.

This rule is documented here so agents follow it. If local tool settings are needed, keep them at the repository root; do not add per-subtree agent configuration directories under `src/`.

## Submission contract (don't break these)

A placer is a Python file exposing a class with `place(benchmark) -> torch.Tensor` of shape `[num_macros, 2]`, returning **center coordinates** (not corners) for both hard and soft macros. The class name does not need to be `MacroPlacer` - the harness instantiates the first placer-shaped class it finds - but callers in this repo may import by name, so prefer `MacroPlacer`.

Hard requirements enforced by the evaluator:

- **Zero hard-macro overlaps.** Soft macros may overlap; they are stand-ins for standard-cell clusters.
- **Fixed macros stay put** (`benchmark.macro_fixed`). Do not move them.
- **All macros within canvas bounds.**
- **<1 hour total** for all 17 IBM benchmarks combined (hard timeout in the harness).

Forbidden by the rules:

- Modifying the TILOS evaluator (`external/MacroPlacement/`).
- Hardcoding per-benchmark solutions or branching on `benchmark.name` to apply benchmark-specific tweaks.
- Calling external proprietary placement tools.

## What's actually being optimized

The current production path optimizes for **hierarchy preservation**, not the
lowest proxy score. It keeps connected subsystems together using grouped
DREAMPlace, cluster-consecutive legalization, owned/bridge soft roles,
congestion-expanded regions, region-locked hard/soft relief, exact-gated cluster
decompression, region-bounded swaps, and proxy-aware coldspot tightening. The
exact proxy is still used for evaluator reports and local accept gates, but the
old spread-oriented proxy optimizer is gone.

Historical proxy objective:

```python
proxy_cost = 1.0 × wirelength + 0.5 × density + 0.5 × congestion
```

After normalization, **wirelength ≈ 0.06**, **congestion ≈ 1.3–2.7**. Congestion dominates by ~30×. This is why the proxy path preferred spread placements and why compact hierarchy-preserving placements cost more proxy by design.

Historical `--all` scores in `docs/general/PROGRESS.md` are retained as
experiment history for the deleted proxy path. Do not treat them as the current
hierarchy output.

## Repo layout

```md
src/main.py         Evaluator-facing entrypoint - exposes MacroPlacer.
src/placer/        Active hierarchy placer package: pipeline, scoring, routing, legalize, local_search.
src/utils/         Runtime config, logging shim, and accepted placement constants.
src/dreamplace_bridge/  pb.txt <-> Bookshelf converters + DREAMPlace launcher.
src/eda_io/        Plug-and-play EDA I/O: LEF/DEF/Verilog/SDC/Liberty in, DEF/Tcl/QoR-report out.
src/place_design.py CLI tying eda_io together - see src/eda_io/README.md.
docs/general/      Current architecture/flow/issues plus the PROGRESS experiment ledger.
docs/ml_nn/        Current BeyondPPA/GNN schemas and status.
test/benchmarks/   Synthetic anti-overfitting suite: generator, runner, impact analyzer.
test/diagnostic/   Maintained smoke tests plus current profiling/recall probes.
test/eda_io/       eda_io pytest suite + LEF/DEF/Verilog/SDC/Liberty fixture design.
test/verification/ Correctness checks vs scalar references.
system/v1/         Frozen v17 checkpoint if present. READ-ONLY.
external/MacroPlacement/  TILOS submodule - evaluator + ICCAD04 testcases. Read-only.
benchmarks/processed/     Pre-processed .pt files for fast loading.
scripts/                  Comparison + benchmark-conversion utilities.
```

## Things that have already burned us (read before debugging)

- **Hierarchy and proxy are opposed.** The exact proxy usually rewards spreading connected macros apart; the current system intentionally keeps subsystems together and accepts the proxy cost.
- **DREAMPlace is required for the current production path.** `_place_impl()` raises if `_hierarchy_floorplan()` cannot run; the old proxy fallback has been deleted.
- **DREAMPlace curvature scaling is already enabled.** The grouped stage sets
  `macro_place_flag=1` and `use_bb=1`, selecting DREAMPlace 4.1's short
  Barzilai-Borwein Nesterov update. A paper-faithful Zhang-Hager Armijo trial
  regressed DREAMPlace seed quality on ibm04 and ibm10 and was removed. Do not
  restore it without a materially different integration hypothesis and a new
  cache-separated A/B.
  BB and DREAMPlace cache reads are fixed production behavior; do not re-add
  runtime feature switches for them. Legacy `HIER_DREAMPLACE_BB` and
  `HIER_DREAMPLACE_CACHE` values have no effect.
- **Exact scoring is slow on large grids.** ibm15 (n=393, grid=2166) takes ~160s; ibm18 (grid=2145) takes ~220s. Always factor scoring time into a per-benchmark time budget. The harness has a 200s/benchmark soft limit and post-scoring budget guard.
- **CPU contention slows scoring 3–5×.** ibm08 scores in 31s clean but 95–131s under load; ibm11 scored 263s under heat. Use a running-max `t_one_score` for budget estimation, not the baseline-only measurement.
- **`initial.plc` is already a good seed.** It comes from a prior EDA flow with hand-tuned spread. The job of legalization is to resolve overlaps without destroying that spread. Restart from random or grid layouts has consistently lost to restarting from `initial.plc + small perturbation`.
- **Soft macros must move with hierarchy.** The current path classifies soft macros as owned or bridge, gives them region boxes, lets grouped DREAMPlace place them, and uses soft relocation plus soft-heavy region swaps after hard legalization/relief. The accepted `HIER_SOFT_SWAP_K=48` default is intentional; `24` was worse on ibm12/15/17, while `64` regressed ibm17.
- **BeyondPPA/GNN must stay inside hierarchy.** Structural metrics and future
  learned rankers may reorder candidates inside existing hierarchy operators,
  but accepted moves still need hard legality, fixed macro immobility, bounds,
  hierarchy-region constraints, hierarchy-quality gates, and exact-proxy gates.
  Do not add a separate structural polish or acceptance path unless the user
  explicitly changes that direction.

## Code style

- `black` line length 100 (configured in `pyproject.toml`).
- Numpy `float64` for placement math; convert to `torch.float32` only at the API boundary.
- Position arrays are `[N, 2]` with `(x, y)` in canvas microns. Centers, not corners.
- Avoid premature abstraction - submissions are short-lived experiments. Inline beats refactor here.
- When writing comments and docstrings, ensure simplicity with all descriptions for functions and code blocks.
- Simplicity of code first. Our code must be human readable, and as such, prioritize the simplicity of code and program structure while ensuring that simplicity does not dampen performance.

## Workflow

- Iterate on one benchmark (`-b ibm10` is the current hierarchy smoke) until the change is sound; run `--all` only when you need a full benchmark sweep.
- When a change alters hierarchy quality or proxy cost, verify it on more than one benchmark before treating it as a system improvement.
- Record concrete numbers in `docs/general/PROGRESS.md` when a change becomes a new accepted system result - that file is the source of truth for "what works", not commit messages.
- Documentation updates are part of every system modification. If a change alters placement flow, operator order, acceptance gates, constants, default behavior, diagnostics, graph/GNN hooks, verification status, or user-facing commands, update `docs/general/ARCHITECTURE.md`, `docs/general/DESIGN_FLOW.md`, and all other relevant docs in the same turn. Relevant docs may include `README.md`, `docs/general/ISSUES.md`, `docs/general/PROGRESS.md`, `docs/ml_nn/**`, or test/diagnostic READMEs. If no documentation needs an update, explicitly note why in the final response.
- Once a change has been accepted and verified as a new system result, record concrete numbers in `docs/general/PROGRESS.md` and make sure `docs/general/ARCHITECTURE.md`, `docs/general/DESIGN_FLOW.md`, and any related subsystem docs describe the accepted behavior instead of stale experiment behavior.
- **All v2-specific tests, diagnostics, and probes live under `test/`** (current subdirs: `benchmarks/`, `diagnostic/`, `eda_io/`, `verification/`). Never create v2 test files in the repo-root `test/` directory (that's read-only per the file-modification-scope rule above and is reserved for the project-level smoke tests). When the user asks an agent to write a verification script, perf probe, or one-off diagnostic for v2 work, put it inside `test/` under the matching subdirectory - and when executing tests for v2 code, point pytest / direct script invocations at that path, not `test/`. The repo-root `test/` exists for the smoke tests only; the v2 slot owns its own test tree.
- Never commit unless asked.
- Do not push, force-push, or create PRs unless asked.

## When in doubt

- For current work, start with `docs/general/DESIGN_FLOW.md` and `docs/general/ARCHITECTURE.md`; they describe the hierarchy system.
- `docs/ml_nn/beyondppa_results/` contains the active hierarchy-integrated GNN schemas. Deleted proxy-path research must not be reintroduced unless the user explicitly changes direction.
- Do not reintroduce deleted proxy-only code unless the user explicitly asks to restore the proxy path.
