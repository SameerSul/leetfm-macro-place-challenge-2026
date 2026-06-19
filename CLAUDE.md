# CLAUDE.md

This file gives Claude Code the context to work productively in this repository on the first try. It is a living document - when something here turns out to be wrong or missing, fix it immediately rather than working around it.

## What this repo is

Submission to the **Partcl/HRT Macro Placement Challenge** (deadline May 21, 2026, $20K grand prize). The historical objective was to minimize proxy cost (lower is better; RePlAce avg **1.4578**), but the current selected system is a hierarchy-preserving macro placer.

The active submission now lives at the repository root: `src/`, `docs/`,
`test/`, `scripts/`, and `ml_data/`. The prior `system/v1/` checkpoint may be
absent after the root-layout migration; if present, treat it as frozen /
read-only.

**Current production mode (2026-06-16): hierarchy-only.** `MacroPlacer.place()`
always routes through `_hierarchy_floorplan()` in
`src/placer/pipeline/macro_placer.py` and raises if grouped DREAMPlace is not
available. The old proxy path has been deleted: candidate restarts, R2/2-opt,
hard-soft/soft swap and cycle passes, generic LSMC, generic cluster kicks, ML
ranker defaults, and their proxy-only verifiers are not active code.
Current accepted hierarchy result: `uv run evaluate src/main.py --all` =
**AVG 1.4452**, 17/17 VALID, 0 overlaps, 520.08s; `ibm10` smoke is
`proxy=1.6759`, VALID.

For the full problem statement see [`README.md`](README.md). For the API contract see [`SETUP.md`](SETUP.md). For the team's research notes see [`PAPERS_NOTES.md`](docs/general/PAPERS_NOTES.md). For experiment history and known-good numbers see [`PROGRESS.md`](docs/general/PROGRESS.md). For the placement objectives that should guide the hierarchy flow, see [`OBJECTIVES.md`](docs/general/OBJECTIVES.md). Do not duplicate that content here.

## Common commands

```bash
# Setup (run once - submodule is required, no-op evaluator otherwise)
git submodule update --init external/MacroPlacement
uv sync
# REQUIRED for full speed: numba JITs the routing-apply (~half the runtime). It's
# in requirements.txt but NOT pyproject.toml, so `uv sync` alone does NOT install
# it — the placer then silently falls back to numpy (~25% slower, --all ~58min near
# the 1h cap, avg 1.1403 vs 1.1380 with JIT). See docs/general/ISSUES.md S13.
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

Read-only (Claude may read but must not edit, create, move, or delete):

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

This rule is documented here so Claude follows it. If local tool settings are
needed, keep them at the repository root; do not add per-subtree `.claude/`
directories under `src/`.

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
congestion-expanded regions, region-locked hard/soft relief, exact-gated
cluster decompression, region-bounded swaps, and proxy-aware coldspot
tightening. The exact proxy is still used for evaluator reports and local
accept gates, but the old spread-oriented proxy optimizer is gone.

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
src/dreamplace_bridge/  pb.txt <-> Bookshelf converters + DREAMPlace launcher.
src/eda_io/        Plug-and-play EDA I/O: LEF/DEF/Verilog/SDC/Liberty in, DEF/Tcl/QoR-report out.
src/place_design.py CLI tying eda_io together - see src/eda_io/README.md.
docs/general/      ARCHITECTURE.md / ISSUES.md / PROGRESS.md / DESIGN_FLOW.md.
docs/gpu/          Archived CUDA/GPU proxy-path notes.
docs/ml_nn/        Archived learned-ranker and GNN-surrogate notes.
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
- **Exact scoring is slow on large grids.** ibm15 (n=393, grid=2166) takes ~160s; ibm18 (grid=2145) takes ~220s. Always factor scoring time into a per-benchmark time budget. The harness has a 200s/benchmark soft limit and post-scoring budget guard.
- **CPU contention slows scoring 3–5×.** ibm08 scores in 31s clean but 95–131s under load; ibm11 scored 263s under heat. Use a running-max `t_one_score` for budget estimation, not the baseline-only measurement.
- **`docs/general/PAPERS_NOTES.md` describes the MaskRegulate regularity mask incorrectly.** The actual paper formula `min(x, X_max-x) + min(y, Y_max-y)` rewards placing macros near canvas *edges*. The notes describe distance-to-center, which is the opposite. The implementation in `_density_gradient_perturb` does neither - it is a pure occupancy-spreading gradient. If you see comments referencing "MaskRegulate centering", the comments are wrong, not the code.
- **`initial.plc` is already a good seed.** It comes from a prior EDA flow with hand-tuned spread. The job of legalization is to resolve overlaps without destroying that spread. Restart from random or grid layouts has consistently lost to restarting from `initial.plc + small perturbation`.
- **Soft macros must move with hierarchy.** The current path classifies soft macros as owned or bridge, gives them region boxes, lets grouped DREAMPlace place them, and uses soft relocation plus soft-heavy region swaps after hard legalization/relief. The accepted `V2_HIER_SOFT_SWAP_K=48` default is intentional; `24` was worse on ibm12/15/17, while `64` regressed ibm17.

## Code style

- `black` line length 100 (configured in `pyproject.toml`).
- Numpy `float64` for placement math; convert to `torch.float32` only at the API boundary.
- Position arrays are `[N, 2]` with `(x, y)` in canvas microns. Centers, not corners.
- Avoid premature abstraction - submissions are short-lived experiments. Inline beats refactor here.
- When writing comments and docstrings, ensure simplicity with all descriptions for functions and code blocks
- Simplicity of code first. Our code must be human readable, and as such, we should prioritize the simplicity of our code and program structure, while ensuring that simplicity does not dampen our programs performance

## Workflow

- Iterate on one benchmark (`-b ibm10` is the current hierarchy smoke) until the change is sound; run `--all` only when you need a full benchmark sweep.
- When a change alters hierarchy quality or proxy cost, verify it on more than one benchmark before treating it as a system improvement.
- Record concrete numbers in `docs/general/PROGRESS.md` when a change becomes a new accepted system result - that file is the source of truth for "what works", not commit messages.
- Once a change has been accepted and verified, ensure that all relevant documentation, such as `README.md`, `docs/general/ARCHITECTURE.md`, `docs/general/ISSUES.md`, `docs/general/PROGRESS.md`, and `docs/general/DESIGN_FLOW.md`, has been updated with the latest changes to avoid stale documentation.
- **All v2-specific tests, diagnostics, and probes live under `test/`** (current subdirs: `benchmarks/`, `diagnostic/`, `eda_io/`, `verification/`). Never create v2 test files in the repo-root `test/` directory (that's read-only per the file-modification-scope rule above and is reserved for the project-level smoke tests). When the user asks Claude to write a verification script, perf probe, or one-off diagnostic for v2 work, put it inside `test/` under the matching subdirectory - and when executing tests for v2 code, point pytest / direct script invocations at that path, not `test/`. The repo-root `test/` exists for the smoke tests only; the v2 slot owns its own test tree.
- Never commit unless asked.
- Do not push, force-push, or create PRs unless asked.

## When in doubt

- For current work, start with `docs/general/DESIGN_FLOW.md` and `docs/general/ARCHITECTURE.md`; they describe the hierarchy system.
- `docs/gpu/`, `docs/ml_nn/`, and `docs/theory/LSMC.md` are archived proxy-path notes unless explicitly revived by the user.
- Do not reintroduce deleted proxy-only code unless the user explicitly asks to restore the proxy path.
