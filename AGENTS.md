# AGENTS.md

This file gives coding agents the context to work productively in this repository on the first try. It is a living document - when something here turns out to be wrong or missing, fix it immediately rather than working around it.

## What this repo is

Submission to the **Partcl/HRT Macro Placement Challenge** (deadline May 21, 2026, $20K grand prize). The historical objective was to minimize proxy cost (lower is better; RePlAce avg **1.4578**), but the current selected system, **VivaPlace**, is a hierarchy-preserving macro placer.

The active submission now lives at the repository root: `src/`, `docs/`,
`test/`, `scripts/`, and `ml_data/`. The prior `system/v1/` checkpoint may be
absent after the root-layout migration; if present, treat it as frozen /
read-only.

**Current production mode (2026-07-19): hierarchy-only.** `MacroPlacer.place()`
always routes through `_hierarchy_floorplan()` in
`src/placer/pipeline/macro_placer.py` and raises if grouped DREAMPlace is not
available. The old proxy path has been deleted: candidate restarts, R2/2-opt,
hard-soft/soft swap and cycle passes, generic LSMC, generic cluster kicks, ML
ranker defaults, and their proxy-only verifiers are not active code.

Current verified result with normal BB/cache behavior:
`uv run evaluate src/main.py --all` = **AVG 1.1404**, 17/17 VALID, 0
overlaps, all final hierarchy audits passed, **318.55s**. Sixteen per-design
scores are unchanged; a contract-preserving repair of the IBM09 constraint-
graph seed improved that design from **1.0122 to 0.9978**.
Region swaps now exact-score short stable prefixes before their untouched
suffix. When a prefix contains the first acceptable candidate, the suffix is
provably irrelevant and skipped. Hard-hard and hard-soft prefixes remain 4 and
8; the calibrated soft-soft prefix is 12. Hard-involving legality is evaluated
only after ranking, on the 16/48 candidates that can reach exact scoring, and
disabled graph paths no longer build zero-valued masks/penalties. The accepted
follow-up sweep increased avoided exact swap evaluations from 58,820 to 66,703
and reduced attributed region-swap time from 150.68s to 148.29s without
changing candidate order, logical quotas, placements, or scores. Its complete
evaluator runtime was 416.74s, effectively flat under final-score noise. The
scorer first reduced its disposable batched congestion grids with in-place
top-tail partitioning, and each region-swap schedule computes the static hard
separation matrices once for all fields, rounds, and graph-fallback work. That
sweep preserved the same 1,077,431 physical and 66,703 avoided exact scores
while reducing attributed region-swap time from 148.29s to 146.98s.
The current region-swap scorer no longer builds or flattens a routing topology
for every candidate pair. One compiled kernel packs the selected pins directly
from the scorer's global net/pin arrays and preserves the evaluator's 2-pin,
3-pin, then high-fanout accumulation order. Congestion top-tail scoring
recomputes only routing-changed H columns, V rows, and hard-blockage cells, then
merges those values with the sorted unchanged baseline. Density scoring applies
only the four changed occupancy rectangles and merges their values with the
baseline tail. The accepted sweep retained the same physical/avoided counts and
reduced attributed region-swap time **146.98s -> 104.04s** (29.2%) and complete
runtime **416.87s -> 351.48s** (15.7%).
The congestion/density baseline arrays, stable descending orders, density
nonzero count, and density sum are now cached across rejected swap batches and
invalidated after every committed hard, soft, swap, or compound move. The
follow-up IBM sweep preserved the same 1,077,431 physical / 66,703 avoided
scores and reduced attributed region-swap time again **104.04s -> 102.68s**;
focused IBM04/12/18 reductions were 7.6%, 9.3%, and 5.5%. The complete sweep
took 371.82s under broader run/compile variance, so this is accepted as an
attributable region-swap improvement, not an end-to-end runtime claim. A fused
single-candidate hard-blockage scratch and Numba `prange` candidate-row
reduction were measured and removed: the former regressed two of three focused
designs, while the latter hit the 20s IBM18 swap guard after less than half the
normal candidates.
The current sweep adds a second same-sized stable prefix before the untouched
swap suffix. It preserves candidate order, first-winner semantics, and logical
quotas while increasing avoided exact swap evaluations to **79,466**; the
trace-compatible IBM region-swap phase fell **104.04s -> 98.74s**. Soft
relocation now batches its exact wirelength prefilter, rejecting **100,831**
proposals before congestion/density scoring. Full-suite region, interleaved,
plateau, and strong-soft times fell respectively **44.09 -> 34.97s**, **6.24 ->
4.93s**, **7.43 -> 5.87s**, and **23.41 -> 18.18s**. Mutually exclusive
placer timing phases account for at least 99.86% of every IBM API call: the
full sweep spent 297.33s in `MacroPlacer.place()` and 318.55s in the evaluator,
leaving 21.22s in evaluator loading/final scoring outside the submission API.
Swap pair-net unions are now merged from sorted incident-net CSR rows in one
compiled kernel, and the sparse exact reducers reuse scorer-owned grid scratch.
The same 1,048,385 logical / 1,066,186 physical / 79,466 avoided IBM work fell
from 98.74s to 94.37s. Soft relocation retains stable integer grid IDs through
deduplication and uses capacity-grown dense workspaces with fused in-place
congestion smoothing/tail reduction. The verification sweep preserved IBM
AVG 1.1404, NG45 0.7121, and synthetic 1.4192 with all audits/truth checks; its
330.75s IBM wall time is treated as run variance, not an end-to-end claim. The
tested optimistic congestion lower bound rejected only 1.2% of IBM10
soft-soft rows and was removed; speculative source waves and net-optimal prefix
ranking are not production paths.
Deterministic per-pass exact-score quotas cap work before the wall-clock safety
guards and preserve every placement and score from the preceding 404.01s
reference. The seed
portfolio filters candidates through an independent six-component hierarchy
contract relative to an `initial.plc` reference that is legalized before its
limits are built; immutable-hard failures are removed before exact scoring where
the candidate is not mandatory. A mandatory lower-proxy seed that misses exactly
one component may be deterministically interpolated toward the passing reference;
only a legal repair retaining at least 95% of the source displacement is exact-
scored. The same contract is enforced against the selected seed throughout
relief and final rollback. The portfolio
includes a default-on constraint-graph legalization alternative for
`initial.plc`, and hard-hard / hard-soft swap sets use exact batched scoring.
Swap congestion/density tails now use exact baseline-plus-touched-cell Numba
reducers; ordinary disposable congestion batches still partition in place.
Nearest-neighbor hierarchy-audit selection also uses a cached Numba kernel.
All preserve the scalar and stable-sort reference semantics.
Region hard relocation rejects candidates above the selected seed's cheap hard-
containment limit before exact batch scoring. The full six-component checkpoint
remains authoritative after the pass.
Plateaued late soft cleanup also tests bounded compound moves for explicit
high-confidence path bundles. Flat owner/bridge evidence remains useful for
individual hierarchy regions but cannot form a compound group. Every compound
member stays in-region, the complete state must pass the rich hierarchy
contract, and exact incremental scoring occurs only after the group is formed.
The ordinary post-swap soft relocation pass is skipped after two attributable
full suites produced zero gain in 34 runs; its time remains as deadline and
final-audit headroom.
Passes advance on
gain (`HIER_PLATEAU_PROXY_GAIN=0.00005`) rather
than fixed repeat counts. Strong/medium late-soft scheduling records every
congestion/density lane and stops the entire remaining pass after an audited
lane has no retained gain. Do not truncate a lane's ordered hot-source tail:
IBM12 improvements begin beyond source 384. A final hierarchy-quality audit rolls back to
the best saved audit-passing checkpoint if the post-search state drifts too
far from the selected hierarchy seed. See `docs/general/ARCHITECTURE.md` for
the full pipeline, `docs/general/ISSUES.md` for current gaps, and
`docs/general/PROGRESS.md` for rejected or superseded experiments.

Soft hierarchy inference is deliberately confidence-calibrated. Useful shared
slash-separated soft instance paths are high-confidence bundles and can move as
a compound; repeated flat-net connectivity and shared hard-cluster affinity are
recorded as medium/low-confidence evidence only. Do not treat a flat-netlist
community as a confirmed IP without an explicit structural tag.

Flat hard-cluster inference has one conservative single-component refinement.
When at least 90% of hard macros collapse into one connectivity component, the
model may partition the hard macros from shared low-fanout soft affinity, with
a strict hard-graph-cut fallback. The result remains inferred, does not create
an explicit compound soft bundle, and is dormant on audited multi-component IBM
graphs. A legal raw seed may anchor the refined contract to avoid double slack;
an illegal raw seed uses grouped DREAMPlace as the reference. The 2026-07-18
synthetic sweep reached AVG 1.4204, 10/10 VALID, zero overlaps, and 10/10 truth
passes; `syn03_sram` recovered its four truth groups exactly. `ibm10` reproduced
the accepted 1.1348 score. The subsequent full IBM sweep reproduced all 17
accepted scores exactly at AVG 1.1412, with 17/17 valid, zero overlaps, and all
audits passing in 423.87s.

The hierarchy model retains exactly one additional parent/child level; it does
not recursively discover an arbitrary tree. Explicit slash-separated paths keep
their nearest useful ancestor above the active leaf partition. Existing
oversized connectivity splits keep the original flat component as their parent;
otherwise an eligible active cluster may receive one strict graph bisection.
That fallback begins with direct low-fanout hard edges and shared-soft
support, then reinforces only those structural relations using initial hard/soft
proximity, local macro-area density, and placed low-fanout wire demand. Geometry
alone cannot create a hierarchy. Production requires raw cut ratio `<=0.20`,
within-child compactness gain `>=0.10`, and combined confidence `>=0.54`.
After ordinary leaf-local relief, a bounded pass rigidly relocates child groups
or swaps sibling slots inside the parent region, co-moving leaf-owned soft
macros. Blocked candidates may compact and legalize only the affected child set.
Every candidate passes the child and parent hierarchy contracts before exact
mixed hard/soft scoring. A retained child move activates those multilevel
contracts for later passes and final rollback. The pass has a shared 24-score
quota, a 4s guard, and a `0.0001` minimum local gain; it is non-recursive and
leaves the active DREAMPlace grouping unchanged.

Every deepest retained child also receives an immutable internal relief box.
The box starts at the child's current hard-macro footprint and adds a per-child
canvas-fraction margin from congestion heat, density heat, and normalized
inter-child graph tension; hot boxes can expand directionally toward cold
components favored by graph corridors. The result is clipped to the retained
parent region. Individual hard and owned-soft relocations and hard-hard swaps
may then search inside the box, with neighboring-child graph centroids guiding
targets and the complete active/child/parent contract checked before commit.
The pass has a 48-score ceiling, a 3s guard, and a `0.0005` gain floor because
any retained deep move activates the stricter downstream multilevel contract.
The accepted IBM sweep exact-scored 528 states in 2.93s, retained none, and
preserved every score at AVG 1.1412. A `0.0001` trial retained six shallow local
moves but regressed the final average to 1.1453, so that floor is rejected.
The independent synthetic sweep reached AVG 1.4193, 10/10 VALID, zero overlaps,
and 10/10 truth-audit passes with the accepted deep pass enabled.

NG45 explicit hierarchy-tag check: `uv run evaluate src/main.py --ng45` =
**AVG 0.7121**, 4/4 VALID, 0 overlaps, all hierarchy audits passed. The latest
validation observed 64.80s; explicit path parents bypassed
the fallback inference.
`uv run python
test/verification/_verify_ng45_hierarchy_tags.py` passes. The hierarchy model
uses slash-separated instance-path prefixes when macro names provide useful
coverage, then falls back to inferred connectivity on flat-name benchmarks.

The learned-GNN stack was removed on 2026-07-16 after repeated offline and
closed-loop regressions. Production has no model loader, learned candidate
reordering, candidate-trace logger, GNN training scripts, or GNN verification
surface. Do not restore that stack unless the user explicitly changes
direction. The deterministic structural ordering term remains default-off at
`HIER_OBJECTIVE_STRUCTURAL_WEIGHT=0.0` and is not a learned model.
Pass-level plateau telemetry remains because it drove a productive schedule
change and does not affect candidate ordering. It writes buffered,
attributable schema-v2 rows to
`ml_data/plateau_telemetry/plateau_telemetry.jsonl` unless
`HIER_PLATEAU_TRACE_PATH` is supplied. Rows distinguish proposed from retained
work and include both the committed revision and a scoped dirty-worktree
fingerprint; exact-scored seeds and final placements also emit structured
hierarchy-contract audit events. NG45 audit rows use the parent design name,
not their shared `output_CT_Grouping` leaf. Use
`scripts/analyze_plateau_telemetry.py --quotas` for per-pass exact-score
utilization and exhaustion, and use
`scripts/analyze_hierarchy_contract.py` for component headroom and offline
slack replay; current production limits were retained after calibration on 31
IBM, NG45, and synthetic final rows.

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
- **The learned-GNN stack is retired.** It repeatedly regressed proxy and
  runtime and was removed from runtime, tooling, tests, data, and active docs.
  Do not restore learned ranking unless the user explicitly changes direction.
  Deterministic structural signals remain advisory and cannot bypass legality,
  bounds, fixed macros, hierarchy regions, hierarchy-quality gates, or exact
  proxy acceptance.

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
- When a paper, technical article, or external method informs implementation or
  an experiment, add or update its numbered entry in
  `docs/general/REFERENCES.md`. Verify title, authors, venue/year, pages, and DOI
  or primary-author/publisher link; state whether the method is production,
  independently adapted, research-only, rejected, or future work; and link the
  relevant design/experiment document back to that entry. Keep results reported
  by the source explicitly separate from VivaPlace measurements and forecasts.
- Documentation updates are part of every system modification. If a change alters placement flow, operator order, acceptance gates, constants, default behavior, diagnostics, structural hooks, verification status, or user-facing commands, update `docs/general/ARCHITECTURE.md`, `docs/general/DESIGN_FLOW.md`, and all other relevant docs in the same turn. Relevant docs may include `README.md`, `docs/general/ISSUES.md`, `docs/general/PROGRESS.md`, or test/diagnostic READMEs. If no documentation needs an update, explicitly note why in the final response.
- Once a change has been accepted and verified as a new system result, record concrete numbers in `docs/general/PROGRESS.md` and make sure `docs/general/ARCHITECTURE.md`, `docs/general/DESIGN_FLOW.md`, and any related subsystem docs describe the accepted behavior instead of stale experiment behavior.
- **All v2-specific tests, diagnostics, and probes live under `test/`** (current subdirs: `benchmarks/`, `diagnostic/`, `eda_io/`, `verification/`). Never create v2 test files in the repo-root `test/` directory (that's read-only per the file-modification-scope rule above and is reserved for the project-level smoke tests). When the user asks an agent to write a verification script, perf probe, or one-off diagnostic for v2 work, put it inside `test/` under the matching subdirectory - and when executing tests for v2 code, point pytest / direct script invocations at that path, not `test/`. The repo-root `test/` exists for the smoke tests only; the v2 slot owns its own test tree.
- Never commit unless asked.
- Do not push, force-push, or create PRs unless asked.

## When in doubt

- For current work, start with `docs/general/DESIGN_FLOW.md` and `docs/general/ARCHITECTURE.md`; they describe the hierarchy system.
- The deleted learned-GNN and proxy-path research must not be reintroduced
  unless the user explicitly changes direction.
- Do not reintroduce deleted proxy-only code unless the user explicitly asks to restore the proxy path.
