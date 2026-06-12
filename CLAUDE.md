# CLAUDE.md

This file gives Claude Code the context to work productively in this repository on the first try. It is a living document - when something here turns out to be wrong or missing, fix it immediately rather than working around it.

## What this repo is

Submission to the **Partcl/HRT Macro Placement Challenge** (deadline May 21, 2026, $20K grand prize). Goal: write a Python `MacroPlacer` that beats the RePlAce baseline (avg proxy cost **1.4578** across 17 IBM ICCAD04 benchmarks). Lower is better.

Per-team active system slot: `system/v2/`. The prior slot `system/v1/` is **frozen / read-only** - it captures the v17 placer (multi-DP, multi-iter Phase 7, 2-opt-on-winner) as a checkpoint to compare against. All new work goes in `system/v2/`.

For the full problem statement see [`README.md`](README.md). For the API contract see [`SETUP.md`](SETUP.md). For the team's research notes see [`PAPERS_NOTES.md`](system/v2/docs/general/PAPERS_NOTES.md). For experiment history and known-good numbers see [`PROGRESS.md`](system/v2/docs/general/PROGRESS.md). Do not duplicate that content here.

## Common commands

```bash
# Setup (run once - submodule is required, no-op evaluator otherwise)
git submodule update --init external/MacroPlacement
uv sync
# REQUIRED for full speed: numba JITs the routing-apply (~half the runtime). It's
# in v2/requirements.txt but NOT pyproject.toml, so `uv sync` alone does NOT install
# it — the placer then silently falls back to numpy (~25% slower, --all ~58min near
# the 1h cap, avg 1.1403 vs 1.1380 with JIT). See system/v2/docs/general/ISSUES.md S13.
uv pip install -r system/v2/requirements.txt

# Single benchmark - fastest feedback loop, use this while iterating
uv run evaluate system/v2/src/main.py -b ibm01

# All 17 IBM benchmarks - the headline score (~30 min on sameer_v1)
uv run evaluate system/v2/src/main.py --all

# NG45 commercial designs (Tier 2, OpenROAD inputs)
uv run evaluate system/v2/src/main.py --ng45

# Visualize a placement
uv run evaluate system/v2/src/main.py -b ibm01 --vis

# Compare v2 against the v1 checkpoint
uv run python scripts/compare_placers.py system/v1/placer.py system/v2/src/main.py

# Compare two placers head-to-head
uv run python scripts/compare_placers.py path/to/placer_a.py path/to/placer_b.py

# Smoke tests (project-level)
uv run pytest test/

# Run a v2-specific diagnostic or verification script (note the v2/test/ path,
# not the repo-root test/ path)
uv run python system/v2/test/diagnostic/_profile_score.py
uv run python system/v2/test/verification/_stress_verify.py

# Synthetic anti-overfitting suite (generate once, then run / analyze)
uv run python system/v2/test/benchmarks/generate_benchmarks.py
uv run python system/v2/test/benchmarks/run_synthetic.py          # synthetic designs
uv run python system/v2/test/benchmarks/run_synthetic.py --ibm    # IBM cross-check
uv run python system/v2/test/benchmarks/analyze_impact.py         # cost-term breakdown

# eda_io: run the v2 placer on standard EDA inputs (LEF/DEF/Verilog/SDC/Liberty)
uv run python system/v2/src/place_design.py \
    --lef tech.lef --def floorplan.def --out-def placed.def --out-tcl place.tcl --report qor.rpt

# eda_io tests (pytest is not in the project venv - use --with)
uv run --with pytest python -m pytest system/v2/test/eda_io/ -v
```

If `uv` is not on PATH, fall back to `pip install -e .` and replace `uv run` with `python -m`.

## File modification scope

**IMPORTANT - write scope is restricted to `system/v2/**` plus root `CLAUDE.md`.** Anything outside that is read-only, including the prior system slot `system/v1/**`.

Writable:
- `system/v2/**` - the active system slot (entrypoint `src/main.py`, the `src/placer/` package, any new files Claude creates here)
- `system/dreamplace_build/**` - DREAMPlace install tree (rebuilds / patches allowed)
- `system/dreamplace_src/**` - DREAMPlace source (custom forks / modifications allowed)
- `CLAUDE.md` - this file

Read-only (Claude may read but must not edit, create, move, or delete):
- **`system/v1/**`** - frozen v17 checkpoint, kept for comparison. Treat as if it lived under `external/`.
- Everything outside `system/` - `macro_place/`, `external/`, `scripts/`, `benchmarks/`, `pyproject.toml`, `README.md`, `SETUP.md`, `TEAM_GUIDE.md`, `LICENSE.md`, etc.

If a task seems to require modifying a read-only file (e.g. fixing a bug in `macro_place/`, adding a script under `scripts/`, correcting an error outside `system/v2/`, or porting/tweaking something from `v1/`), stop and surface the proposed change to the user instead of editing. They will lift the restriction explicitly when appropriate - typically by asking Claude to copy the v1 file into v2 first, then modify the v2 copy.

This rule is documented here so Claude follows it. If local tool settings are
needed, keep them at the repository root; do not add per-subtree `.claude/`
directories under `system/v2/src/`.

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

```
proxy_cost = 1.0 × wirelength + 0.5 × density + 0.5 × congestion
```

After normalization, **wirelength ≈ 0.06**, **congestion ≈ 1.3–2.7**. Congestion dominates by ~30×. **Optimizing for wirelength alone reliably makes proxy cost worse** because clustering connected macros spikes density and congestion. This was tested exhaustively (see `system/v2/docs/general/PROGRESS.md`); do not retry it without a specific reason.

The floor v2 must clear is **the frozen v17 placer at `system/v1/placer.py`** - multi-DP at target_density 0.85/0.65 + multi-iter Phase 7 cong-grad chain from each DP + 2-opt-on-winner. 6-benchmark spot check vs v15 was −0.0258 cumulative (notable: ibm02 −0.0194, ibm04 −0.0025, ibm07 −0.0026). Headline `--all` number not yet measured at the freeze point. Earlier reference (`sameer_v1`, avg 1.486) reaches its score by legalizing from `initial.plc` then running multi-restart with congestion-gradient perturbations.

## Repo layout

```
macro_place/        Core framework - benchmark loader, evaluator wrapper, utilities. Don't modify lightly.
system/             Varrahan system implementations and local DREAMPlace build.
  v0/               Reference/simple early placers.
  v1/               Frozen v17 checkpoint - multi-DP + multi-iter Phase 7 + 2-opt-on-winner. READ-ONLY.
  v2/               Active system slot - writable.
    src/main.py         Evaluator-facing entrypoint - exposes MacroPlacer (imports from src/placer/).
    src/placer/           The placer package: pipeline/, scoring/, routing/, plc/, legalize/, local_search/, perturb/.
    src/dreamplace_bridge/  pb.txt ↔ Bookshelf converters + async launcher.
    src/eda_io/           Plug-and-play EDA I/O: LEF/DEF/Verilog/SDC/Liberty in, DEF/Tcl/QoR-report out
                          (converts to ICCAD04 pb+plc, so the placer + exact scorer run unchanged).
    src/place_design.py   CLI tying eda_io together - see src/eda_io/README.md.
    docs/general/         ARCHITECTURE.md / ISSUES.md / PROGRESS.md / DESIGN_FLOW.md.
    docs/gpu/             CUDA and GPU exploration notes.
    docs/ml_nn/           Learned candidate-ranker and GNN-surrogate notes.
    test/                 v2-specific tests / diagnostics / probes - put ALL new v2 test files here.
      benchmarks/         Synthetic anti-overfitting suite: generator, runner, impact analyzer.
      diagnostic/         Maintained smoke tests plus current profiling/recall probes.
      eda_io/             eda_io pytest suite + LEF/DEF/Verilog/SDC/Liberty fixture design.
      verification/       Correctness checks vs scalar references.
external/MacroPlacement/  TILOS submodule - evaluator + ICCAD04 testcases. Read-only.
benchmarks/processed/     Pre-processed .pt files for fast loading.
scripts/                  Comparison + benchmark-conversion utilities.
test/                     Project-level pytest smoke tests. READ-ONLY for v2 work - do not add v2 tests here.
```

## Things that have already burned us (read before debugging)

- **`density_score` fallback is ANTI-CORRELATED with proxy cost.** Sum-of-squares occupancy rewards spread placements, but spread placements have *worse* proxy because they hurt congestion. For any benchmark that cannot use exact scoring (`n > 340` or `grid_cells > 2000`), return the baseline legalization. See the legacy threshold notes in `system/v1/placer.py`.
- **Exact scoring is slow on large grids.** ibm15 (n=393, grid=2166) takes ~160s; ibm18 (grid=2145) takes ~220s. Always factor scoring time into a per-benchmark time budget. The harness has a 200s/benchmark soft limit and post-scoring budget guard.
- **CPU contention slows scoring 3–5×.** ibm08 scores in 31s clean but 95–131s under load; ibm11 scored 263s under heat. Use a running-max `t_one_score` for budget estimation, not the baseline-only measurement.
- **`system/v2/docs/general/PAPERS_NOTES.md` describes the MaskRegulate regularity mask incorrectly.** The actual paper formula `min(x, X_max-x) + min(y, Y_max-y)` rewards placing macros near canvas *edges*. The notes describe distance-to-center, which is the opposite. The implementation in `_density_gradient_perturb` does neither - it is a pure occupancy-spreading gradient. If you see comments referencing "MaskRegulate centering", the comments are wrong, not the code.
- **`initial.plc` is already a good seed.** It comes from a prior EDA flow with hand-tuned spread. The job of legalization is to resolve overlaps without destroying that spread. Restart from random or grid layouts has consistently lost to restarting from `initial.plc + small perturbation`.
- **Soft macros must be repositioned when hard macros move significantly.** The `PlacementCost.optimize_stdcells` API does this but takes minutes per call in Python. The current placers leave soft macros at their initial positions - acceptable for small perturbations, problematic for large displacements (e.g., DREAMPlace-style global re-placement).

## Code style

- `black` line length 100 (configured in `pyproject.toml`).
- Numpy `float64` for placement math; convert to `torch.float32` only at the API boundary.
- Position arrays are `[N, 2]` with `(x, y)` in canvas microns. Centers, not corners.
- Avoid premature abstraction - submissions are short-lived experiments. Inline beats refactor here.
- When writing comments and docstrings, ensure simplicity with all descriptions for functions and code blocks
- Simplicity of code first. Our code must be human readable, and as such, we should prioritize the simplicity of our code and program structure, while ensuring that simplicity does not dampen our programs performance

## Workflow

- Iterate on one benchmark (`-b ibm01` or `-b ibm04`) until the change is sound; run `--all` only when you want a full leaderboard number. A `--all` run takes ~30 minutes, so it is not a substitute for unit-style debugging.
- When a change improves one benchmark, verify it does not regress others before committing. The repo's history (`git log`) shows several "win on ibm04, lose on ibm09" reverts.
- Record concrete numbers in `system/v2/docs/general/PROGRESS.md` when a change becomes the new best - that file is the source of truth for "what works", not commit messages.
- Once a change has been accepted and verified, ensure that all relevant documentation, such as `system/v2/README.md`, `system/v2/docs/general/ARCHITECTURE.md`, `system/v2/docs/general/ISSUES.md`, `system/v2/docs/general/PROGRESS.md`, and `system/v2/docs/general/DESIGN_FLOW.md`, has been updated with the latest changes to avoid stale documentation.
- **All v2-specific tests, diagnostics, and probes live under `system/v2/test/`** (current subdirs: `benchmarks/`, `diagnostic/`, `eda_io/`, `verification/`). Never create v2 test files in the repo-root `test/` directory (that's read-only per the file-modification-scope rule above and is reserved for the project-level smoke tests). When the user asks Claude to write a verification script, perf probe, or one-off diagnostic for v2 work, put it inside `system/v2/test/` under the matching subdirectory - and when executing tests for v2 code, point pytest / direct script invocations at that path, not `test/`. The repo-root `test/` exists for the smoke tests only; the v2 slot owns its own test tree.
- Never commit unless asked.
- Do not push, force-push, or create PRs unless asked.

## When in doubt

- The leaderboard #1 entry (UT Austin DREAMPlace, 1.4076) uses `pb.txt → Bookshelf → DREAMPlace global placement → legalize`. v1's bridge (`system/v1/dreamplace_bridge/`) implements this path - v2 can import or copy it forward. The remaining gap (~0.05 from v1 to the leaderboard) is mostly congestion-aware optimization that DREAMPlace's NLP doesn't see; see v1's `_dp_diagnostic.py` for the empirical decomposition.
- WireMask-BBO's greedy evaluator is the highest-leverage *non-GPU* unimplemented idea (avg ~27M HPWL on mixed-size IBM, no training needed). The current `_compute_wire_pull` is a continuous approximation, not the real greedy mask.
- For anything ML-heavy (ChiPFormer-style DT, MaskPlace-style RL, diffusion), the cost/benefit ratio is poor on the remaining timeline - read `system/v2/docs/general/PAPERS_NOTES.md` for the team's reasoning before starting one.
