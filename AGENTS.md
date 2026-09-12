# AGENTS.md

This file gives coding agents the context to work productively in this repository on the first try. It is a living document - when something here turns out to be wrong or missing, fix it immediately rather than working around it.

## What this repo is

VivaPlace is the active hierarchy-preserving submission to the Partcl/HRT
Macro Placement Challenge. Active code is at the repository root (`src/`,
`docs/`, `test/`, `scripts/`, `ml_data/`). Treat `system/v1/`, if present, as a
frozen, read-only checkpoint.

**Production objective: minimize exact proxy subject to legality and the
existing hierarchy contract.** Lower proxy is better. `MacroPlacer.place()`
always calls `_hierarchy_floorplan()` and raises if grouped DREAMPlace is
unavailable. Select the lowest exact-proxy contract-passing seed. Do not spend
proxy quality on extra hierarchy compactness or headroom after the constraints
pass. Terminal envelope isolation and its dependent polishes are removed.

Current behavior and validation:

- Keep grouped DREAMPlace, cluster-consecutive legalization, confidence-based
  leaf safeguards, and complete final active/child/parent hierarchy audits.
  The legalizer reserves every fixed obstacle before placing movable macros,
  including temporary recurrent anchors.
- Soft hierarchy is confidence-calibrated. Explicit shared instance paths may
  form rigid compound bundles. Repeated flat-net support and hard affinity are
  evidence, not confirmed IP boundaries. Retain bounded one/two-hop propagation
  and stable residual groups; routing-only cohorts never claim ownership.
- Retain exactly one additional parent/child level. Geometry may reinforce
  structural evidence but cannot create hierarchy edges. Whole-child relocation
  and sibling-slot swaps remain; their contracts become active after a retained
  child move. The separate deepest-child internal pass was removed after a
  31-design isolated ablation preserved every coordinate, score, and hierarchy
  metric. Consolidation stays because its removal regressed IBM02.
- Late passes keep spectral leaf floorplanning and void/boundary relocation.
  Final free-soft density relief freezes all hard/fixed macros, hierarchy roles,
  bundles, and direct hard-connected softs. Final cluster-tile search uses
  complete incident-net pressure and the canonical location graph to rearrange
  2–4 units within one leaf/child partition. Ownership, confidence, fixed and
  bridge macros stay immutable; complete explicit path bundles move rigidly.
  At most four patches test 16 exact candidates each, borrowing up to three
  seconds of unused internal-floorplan/void allowance. Retained output must
  pass fresh clamped float32 scoring, hard legality, and the complete contract.
- The validated tile replay improves 4/31 designs and leaves 27 unchanged.
  All outputs pass legality, complete contracts, four NG45 tag checks, and ten
  synthetic truth checks. The ordinary-guard run returns exactly the same
  coordinates. See `ml_data/algorithm_review/20260909/results.md`; these are
  paired-input gains, not an end-to-end speed claim.
- The shared `_last_pos_cache` checkpoint-restore defect remains open. Its
  global repair was reverted after per-design regressions. Final density/tile
  transactions build fresh scoring state for their own baseline and winner;
  do not describe that as a global repair.
- Preserve deterministic candidate order, exact-score quotas, and retained-gain
  plateau scheduling. Do not truncate an ordered hot-source tail: later sources
  can contain the retained improvements. Ordinary time guards still change
  scheduling under contention; use fixed-work or same-input comparisons for
  attribution, alongside ordinary-guard validation.
- Telemetry records proposed versus retained work, source revision/fingerprint,
  stage timing, and complete seed/final contracts. It writes schema-v2 rows to
  `ml_data/plateau_telemetry/plateau_telemetry.jsonl`, or
  the `utils.constants.HIER_PLATEAU_TRACE_PATH` constant. Use `scripts/analyze_plateau_telemetry.py --quotas`
  and `scripts/analyze_hierarchy_contract.py`; do not relax contract limits to
  improve a score.
- Do not restore the removed proxy restart/LSMC path or learned GNN ranking
  without an explicit user change in direction. Deleted coldspot variants,
  graph prefilter/rescue, and weak/hot region reshaping have rejected or inert
  controls in the experiment ledger. Rejected CUDA overlap filtering, graph
  construction/affinity kernels, graph-corridor region bias, and RUDY seed wrappers
  are removed. Retain the promising CPU split diagnostic and exact-gated GPU
  soft-refinement replay; neither is a production promotion.
- Constructor options are keyword-only: `seed`, `event_sink`, and
  `dreamplace_sample_every`. Ignored `n_restarts`, `noise_fracs`, and
  `time_budget_s` options are removed. Runtime is controlled by existing
  per-pass limits and quotas, not an overall constructor budget.

Historical scores and rejected experiments belong in `docs/PROGRESS.md`, not
this onboarding guide. Its first status entry is current; later entries are
historical. Current validation and limitations are in the first status entry.

For the full problem statement see [`README.md`](README.md). For the API contract see [`SETUP.md`](SETUP.md). For experiment history and known-good numbers see [`PROGRESS.md`](docs/PROGRESS.md). For the placement objectives that should guide the hierarchy flow, see [`OBJECTIVES.md`](docs/OBJECTIVES.md). Do not duplicate that content here.

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
# utils.constants.ALLOW_NUMBA_FALLBACK is True for slow diagnostic-only runs.
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

# Maintained correctness and interface tests
uv run --with pytest python -m pytest test/verification/ test/eda_io/ test/visualizer/

# Run a diagnostic or verification script
uv run python test/verification/_verify_coldspot_kick.py ibm10

# Bytecode sanity after edits
uv run python -m compileall -q src

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

The evaluator requires a placer class defined in the entrypoint module
(`cls.__module__ == path.stem`); keep the local `MacroPlacer` adapter in
`src/main.py` rather than replacing it with a plain import alias.

A placer is a Python file exposing a class with `place(benchmark) -> torch.Tensor` of shape `[num_macros, 2]`, returning **center coordinates** (not corners) for both hard and soft macros. The class name does not need to be `MacroPlacer` - the harness instantiates the first placer-shaped class it finds - but callers in this repo may import by name, so prefer `MacroPlacer`.

Hard requirements enforced by the evaluator:

- **Zero hard-macro overlaps.** Soft macros may overlap; they are stand-ins for standard-cell clusters.
- **Fixed macros stay put** (`benchmark.macro_fixed`). Do not move them. The
  shared legalizer reserves all fixed obstacles before processing movable
  members, even when the supplied order lists a fixed macro later.
- **All macros within canvas bounds.**
- **<1 hour total** for all 17 IBM benchmarks combined (hard timeout in the harness).

Forbidden by the rules:

- Modifying the TILOS evaluator (`external/MacroPlacement/`).
- Hardcoding per-benchmark solutions or branching on `benchmark.name` to apply benchmark-specific tweaks.
- Calling external proprietary placement tools.

## What's actually being optimized

The current production path minimizes **exact proxy subject to hierarchy and
legality constraints**. It keeps connected subsystems together using grouped
DREAMPlace, cluster-consecutive legalization, owned/bridge soft roles,
congestion-expanded regions, region-locked hard/soft relief, exact-gated cluster
decompression, transactional graph-adjacent ownership transfer with an exact
density/congestion gate, post-transfer same-leaf repair, region-bounded swaps,
and proxy-aware coldspot tightening. The
exact proxy ranks contract-passing seeds and gates local improvements. The
old standalone spread-oriented proxy optimizer remains removed.

`HierarchyModel.edges` and `LocationAwareGraph.active_edges` share one canonical
list. Do not add per-cluster adjacency copies or rebuild hierarchy edges from
normalized macro adjacency. Hard ownership changes must use cached eligible
nets and the full-net `_cluster_graph` weighting; soft ownership changes must
not alter the hard hierarchy graph. Retained child and parent edges are distinct
hierarchy projections and should remain separate.

Historical proxy objective:

```python
proxy_cost = 1.0 × wirelength + 0.5 × density + 0.5 × congestion
```

After normalization, **wirelength ≈ 0.06**, **congestion ≈ 1.3–2.7**. Congestion dominates by ~30×. This is why the proxy path preferred spread placements and why compact hierarchy-preserving placements cost more proxy by design.

Historical `--all` scores in `docs/PROGRESS.md` are retained as
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
docs/              Current architecture/flow plus the PROGRESS experiment ledger.
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

- **Hierarchy constrains proxy optimization.** Preserve connected subsystems,
  but reject extra hierarchy boosting that worsens proxy after the contract
  already passes. The isolation-first tradeoff is superseded.
- **DREAMPlace is required for the current production path.** `_place_impl()` raises if `_hierarchy_floorplan()` cannot run; the old proxy fallback has been deleted.
- **DREAMPlace backend selection is explicit.** Set the source constant
  `utils.constants.DREAMPLACE_GPU = True` for CUDA; the default `False` uses CPU even when the main process detects a GPU. CUDA cache
  entries are separate; existing CPU cache keys remain unchanged. See
  `ml_data/dreamplace_cuda/20260909/` for the focused backend comparison.
  The requested normal-guard CUDA IBM sweep is AVG 1.1806, 17/17 VALID, zero
  overlaps, and all final audits passing in 980.93s placer time (55 fresh CUDA
  seed calls). This is not a paired CPU comparison; the default remains CPU.
- **DREAMPlace curvature scaling is already enabled.** The grouped stage sets
  `macro_place_flag=1` and `use_bb=1`, selecting DREAMPlace 4.1's short
  Barzilai-Borwein Nesterov update. A paper-faithful Zhang-Hager Armijo trial
  regressed DREAMPlace seed quality on ibm04 and ibm10 and was removed. Do not
  restore it without a materially different integration hypothesis and a new
  cache-separated A/B.
  BB and DREAMPlace cache reads are fixed production behavior; do not re-add
  runtime feature switches for them. Legacy `HIER_DREAMPLACE_BB` and
  `HIER_DREAMPLACE_CACHE` values have no effect.
- **Scoring and deadlines need controlled measurements.** CPU contention,
  compilation, and DREAMPlace cache state can change the work completed before
  a safety guard. Compare identical inputs or deterministic work counts and
  fresh external scores; do not infer a speedup from unmatched wall times.
- **`initial.plc` is already a good seed.** It comes from a prior EDA flow with hand-tuned spread. The job of legalization is to resolve overlaps without destroying that spread. Restart from random or grid layouts has consistently lost to restarting from `initial.plc + small perturbation`.
- **Soft macros must move with hierarchy.** The current path classifies soft macros as owned or bridge, gives them region boxes, lets grouped DREAMPlace place them, and uses soft relocation plus soft-heavy region swaps after hard legalization/relief. The accepted `HIER_SOFT_SWAP_K=48` default is intentional; `24` was worse on ibm12/15/17, while `64` regressed ibm17.
- **The learned-GNN stack is retired.** It repeatedly regressed proxy and
  runtime and was removed from runtime, tooling, tests, data, and active docs.
  Do not restore learned ranking unless the user explicitly changes direction.
  Deterministic structural signals remain advisory and cannot bypass legality,
  bounds, fixed macros, hierarchy regions, hierarchy-quality gates, or exact
  proxy acceptance.

## Code style

- Placement and diagnostic settings use source constants in `src/utils/constants.py`.
  Do not add environment-variable overrides. Diagnostic runners assign constants
  directly; native library launch paths/thread pools and fixed cuBLAS workspace
  setup are separate from placement configuration.

- `black` line length 100 (configured in `pyproject.toml`).
- Numpy `float64` for placement math; convert to `torch.float32` only at the API boundary.
- Position arrays are `[N, 2]` with `(x, y)` in canvas microns. Centers, not corners.
- Avoid premature abstraction - submissions are short-lived experiments. Inline beats refactor here.
- When writing comments and docstrings, ensure simplicity with all descriptions for functions and code blocks.
- Simplicity of code first. Our code must be human readable, and as such, prioritize the simplicity of code and program structure while ensuring that simplicity does not dampen performance.

## Workflow

- Iterate on one benchmark (`-b ibm10` is the current hierarchy smoke) until the change is sound; run `--all` only when you need a full benchmark sweep.
- When a change alters hierarchy quality or proxy cost, verify it on more than one benchmark before treating it as a system improvement.
- Record concrete numbers in `docs/PROGRESS.md` when a change becomes a new accepted system result - that file is the source of truth for "what works", not commit messages.
- When a paper, technical article, or external method informs implementation or
  an experiment, add or update its numbered entry in
  `docs/REFERENCES.md`. Verify title, authors, venue/year, pages, and DOI
  or primary-author/publisher link; state whether the method is production,
  independently adapted, research-only, rejected, or future work; and link the
  relevant design/experiment document back to that entry. Keep results reported
  by the source explicitly separate from VivaPlace measurements and forecasts.
- Documentation updates are part of every system modification. If a change alters placement flow, operator order, acceptance gates, constants, default behavior, diagnostics, structural hooks, verification status, or user-facing commands, update `docs/ARCHITECTURE.md`, `docs/DESIGN_FLOW.md`, and all other relevant docs in the same turn. Relevant docs may include `README.md`, `docs/ISSUES.md`, `docs/PROGRESS.md`, or test/diagnostic READMEs. If no documentation needs an update, explicitly note why in the final response.
- Once a change has been accepted and verified as a new system result, record concrete numbers in `docs/PROGRESS.md` and make sure `docs/ARCHITECTURE.md`, `docs/DESIGN_FLOW.md`, and any related subsystem docs describe the accepted behavior instead of stale experiment behavior.
- Keep tests, diagnostics, and probes in the matching `test/` subdirectory
  (`verification/`, `diagnostic/`, `benchmarks/`, `eda_io/`, `visualizer/`).
  Target those suites directly. The former nested v2 test-tree convention is
  obsolete; the active test tree is at the repository root.
- Never commit unless asked.
- Do not push, force-push, or create PRs unless asked.

## When in doubt

- For current work, start with `docs/DESIGN_FLOW.md` and `docs/ARCHITECTURE.md`; they describe the hierarchy system.
- The deleted learned-GNN and proxy-path research must not be reintroduced
  unless the user explicitly changes direction.
- Do not reintroduce deleted proxy-only code unless the user explicitly asks to restore the proxy path.
