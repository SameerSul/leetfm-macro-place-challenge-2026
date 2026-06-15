# v2 — Varrahan's Submission

Active placer for the Partcl/HRT Macro Placement Challenge. A multi-restart
legalization placer with async DREAMPlace seed candidates, generic random/local
restart diversity, a **fully-incremental proxy scorer**, deep **R2 move-based
local search**, and a final **LSMC kick/descent/accept** exploration layer.

**Reference headline (`--all`, 2026-06-11): avg `1.1252`** (17/17 VALID, 0 overlaps,
**2337s ~39min**) — beats the RePlAce target (`1.4578`) by **22.8%** and the
#1 leaderboard (UT Austin DREAMPlace, `1.4076`) by **0.282 (−20.1%)**, on every
single benchmark. Trajectory: 1.1782 → 1.1500 → 1.1423 (S11 prefilters) →
1.1403 (S12 wider soft pool) → 1.1380 (S13 numba re-enabled) → 1.1379 (S14
hand-JIT scoring hot paths, DP-off) → 1.1272 (S16 DREAMPlace ABI fix, DP basins
restored) → **1.1252** (S10 ML hard-relocation filter enabled by default).

The active 2026-06-14 design has since pruned the congestion-gradient spine. LSMC
is now deliberately generic: its seed pool comes from the legalized baseline,
random-noise restarts, random-order legalization, pre-R2 best, and post-R2 best.
DREAMPlace still contributes ordinary scored candidates to `best_pl`, but LSMC
does not depend on DREAMPlace/bridge-specific seed pools or cong-grad-derived
state. See [`docs/general/PROGRESS.md`](docs/general/PROGRESS.md) for the latest
post-pruning measurements.

> ⚠ **Requires numba** for full speed — it JITs the routing-apply (~half the
> runtime). numba is in `requirements.txt` but **not** `pyproject.toml`, so
> `uv sync` alone won't install it; **install `requirements.txt`**. Without numba
> the placer still runs (numpy fallback) but materially slower and gives up
> deadline-bound refinement. See `docs/general/ISSUES.md` S13.
Driven by a **family of dual-field soft + hard moves** (cong-field +
density-field for every move type, plus HXS hard ⇄ soft cross-swaps), a
bit-exact incremental scoring core, a parallelized pipeline, an **adaptive
round/pass scheduler** that re-iterates whenever a pass keeps finding
moves and bails when it saturates, plus a **persistent shared scorer +
numba-JIT'd routing apply** that frees ~15-25s/benchmark of compute —
which the R2 loop spends on additional productive rounds AND fixes the
ibm18 starvation that previously cost +0.28 on that single benchmark. The current
default also enables the shipped XGBoost hard-relocation ranker when the model
artifact and `xgboost` are available and no `ML_*` env var is preset; it widens the
hard-relocation pool to 32 and exact-scores the model's top 16, preserving the
same strict true-proxy accept gate.
The dominant algorithmic levers:
(a) **single-soft relocation** R3 (cong) + R5 (density) — 1.4216 → 1.2799,
(b) **A1 soft-soft 2-opt** + A1b cong-field + A1c cold-teleport — 1.2737
→ 1.2195,
(c) **A4 WL-aware candidate ordering + A5 adaptive multi-pass 2-opt +
adaptive R2 round termination + adaptive skip-empty replacing hardcoded
round caps** — 1.2195 → 1.2092,
(d) **HXS hard ⇄ soft cross-swap + R6 combined cong+density relocation +
WL-delta prefilter for soft-2opt + persistent shared scorer per R2 round
+ numba-JIT routing apply (with numpy fallback)** — 1.2092 → 1.1993
(14/17 wins, ibm18 starvation fixed: +0.28 → −0.036),
(e) **HS3 hard-soft 3-cycle (H → S₁ → S₂ → H) + 3-pin routing dispatcher
numba-JIT** — 1.1993 → **1.1782** (11/17 wins, biggest mover ibm16 −0.029).
Layered on top: (i) **incremental congestion cost** (cache smoothed H/V;
re-smooth only the touched-net bbox per move), (ii) **#1 subset-cumsum
strip-batch**, (iii) **#2 topology-struct cache** for the routing apply,
(iv) a **floor-reservation budget allocator** (every benchmark ≥110 s — no
last-benchmark starvation), (v) **round-3 cong cap + density `top_hot=192`
boost**, (vi) **S1 prep/trial/commit/revert + S3 bincount strip-batch**
(hoist the loop-invariant subtract-old — 25–43% faster per-trial),
(vii) **A3 net-centroid candidate ordering** for soft passes,
(viii) **H5 hard density relocation** (the R5-for-hards symmetry),
(ix) **Phase 9 + DREAMPlace ×3 parallelization**, (x) **DREAMPlace ABI fix**
so the three DP basins actually run under the Python 3.10 DP build env, and
(xi) **default ML hard-relocation filtering** for the wide-32 candidate pool.
The entire chain is
**bit-exact verified** (every scoring path — including the new HXS
score_swap_hard_soft and the numba-JIT strip-batch — has its own
verifier; Δ ≤ 4.4e-16).
Stacked progression: 1.4854 (v12) → 1.2799 (R5) → 1.2767 (inc cong) →
1.2755 (+ #1+#2+floor-res+A+C) → 1.2737 (+ S1+S3) → 1.2433 (+ A1+A3) →
1.2195 (+ H5+A1b+A1c+A1×2+Phase9-parallel) → 1.2092 (+ A4+A5+adaptive
R2/skip-empty) → 1.1993 (+ HXS+R6+WL-prefilter+shared-scorer+numba) →
1.1782 (+ HS3+3pin-JIT, 11/17 wins) → 1.1379 (S14 hand-JIT, DP-off) →
1.1272 (S16 DP restored) → **1.1252** (ML hard-relocation filter default).

> Source of truth for numbers and experiment history is [`docs/general/PROGRESS.md`];
> open issues / closed dead-ends are in [`docs/general/ISSUES.md`]. This README is the
> architectural overview.

## What's being optimized

```
proxy_cost = 1.0·wirelength + 0.5·density + 0.5·congestion
```
After normalization, **congestion ≈ 65% of proxy**, density ≈ 30%, wirelength
≈ 5%. The whole strategy follows from this: local moves are ranked and accepted
against the exact congestion-aware proxy, and WL-only optimization reliably makes
proxy *worse* (clustering spikes congestion).

## Pipeline

```
0    Baseline           legalize from initial.plc (vectorized _will_legalize)
─    Multi-DP (async)   3 DREAMPlace candidates launched in parallel:
                          lo-fix (td=0.65, soft fixed), hi-mov (td=0.85, soft
                          movable), hi-fix (td=0.85, soft fixed); scored as
                          normal candidates only
─    random restarts     Gaussian perturbations of initial hard positions,
                          legalize + exact-score
9    random-order        legalize with randomized tie-break order
─    R2 interleave       alternate relocation/swap/cycle passes until neither
                          improves (≤20 budget-gated rounds) — see "Relocation" below
─    post-R2 soft reloc  leftover soft congestion/density cleanup
─    LSMC final explore  random hard-macro kick -> legalize -> hard/soft descent
                          -> strict exact accept, over generic local seed pool
```
All candidates legalized then scored via exact `PlacementCost` proxy; lowest
wins. The floor-reservation allocator uses a 150s first-benchmark soft budget,
a 110s per-benchmark floor, and an 83s overrun reserve under the
3300s internal `--all` place-time cap; thresholds admit all 17.
Every return path goes through a final movable-macro in-bounds clamp; hard macros
are already legalized, so this safety net primarily catches stray soft macro
coordinates from input or DREAMPlace output.

## The three things that make v2 ≫ v1 (1.4854 → 1.2755)

### 1. Fully-incremental proxy scorer (`IncrementalScorer`)

A 2-opt/relocation move changes only 1–2 macros, so re-scoring the whole proxy
each trial is wasteful. The scorer maintains all three terms as state and updates
only what a move touches:

| Term | Incremental strategy | Tag |
|---|---|---|
| Wirelength | recompute HPWL for the moved macro's nets only | B3p2 |
| Congestion | subtract/add the touched-net routing demand + the macro's routing blockage on the maintained H/V flats | B3p4 |
| Density | maintain the occupancy grid; update only the moved macro's footprint cells | P3 |

Net: **~1.4 ms/move-eval** (vs full recompute scattering all ~1100–2800 macros).
`score_swap`/`score_move` are **verified bit-exact** vs the full `_exact_proxy`
(`test/verification/_verify_incremental_scorer.py`, `_verify_score_move.py`;
Δ ≤ 1e-8, no drift over sequential commits). This speed is what makes the
move-based local search affordable.

### 2. Congestion- & density-directed relocation (R1 / R2 / R2b / R3 / R5 — the dominant lever)

2-opt only *exchanges* two macros' positions — it can **never relocate a routing-
heavy macro into empty low-congestion space** (a swap would dump some other macro
into the vacated hot spot). Relocation adds exactly that missing move:

- **R1** — a post-2-opt pass that moves the hottest *hard* macros (by live
  `max(H,V)` congestion) into the nearest low-congestion legal cells, accept-on-
  true-proxy. Legality = in-bounds + no overlap with other hard macros (softs may
  overlap). `--all 1.4422 → 1.4326`, all 17 improved.
- **R2** — *interleave* relocation ⇄ 2-opt: each relocation opens new swaps and
  vice versa. The early implementation used ≤6 rounds (`1.4326 → 1.4243`);
  the current loop is budget-gated up to 20 rounds and stops on saturation.
- **R2b** — widen the per-round candidate set (`top_hot` 24→48, `n_targets`
  12→16) so large benchmarks relieve >3% of their hot macros/round.
  `1.4243 → 1.4216`, and faster.
- **R3 — soft-macro relocation.** Soft macros are the bulk of the routing demand
  and were frozen at `initial.plc` by every prior placer. Relocating the hottest
  soft clusters into low-congestion space (`score_move_soft`, verified bit-exact;
  no legality check since softs may overlap), as a third move type in the loop,
  compounds: **`1.4216 → 1.3764`**, all 17 improved. Corrects O3 (which only
  tested *bulk* soft moves).
- **R3b / R5 — soft DENSITY relocation (the dominant win of the relocation family).**
  Softs are the bulk of the *density* term too (and may overlap, so the cong pass
  can pile them). A second soft pass targeting the **density** field
  (`use_density`) finds moves the cong pass can't: a cong-converged best_pl
  still yielded 22–68 density moves. Interleaved (hard ⇄ soft-cong ⇄
  soft-density ⇄ 2-opt) + widened candidates (top_hot 128): **`1.3764 → 1.2799`**,
  all 17 improved (ibm13/02/08 −0.122, ibm18 −0.21).

All moves are accept-on-true-proxy, so the whole local search is **strictly
non-regressing by construction**.

**Leverage** (`test/diagnostic/_reloc_leverage.py`): per-benchmark gain is driven
by **hard-macro utilization × congestion headroom** — relocation helps where hard
macros occupy enough canvas to drive congestion (ibm04/10/02/12, util 0.42–0.60)
AND there's congestion above the floor. Low-hard-util benchmarks (ibm17/18) are
soft/net-dominated and barely move → soft-macro relocation is the flagged next
lever.

### 3. Bit-exact scoring-speedup stack (1.2799 → 1.2755)

Five mutually compounding changes, each *bit-exact* (every accept-on-true-proxy
guarantee preserved; every change passes the same scorer verifiers as the base):

- **Incremental congestion cost.** `_compute_cong_cost` used to full-re-smooth the
  whole grid and full-partition every move (~17% of a trial). The smoother is a
  separable box filter — H per column, V per row, each independent — so the scorer
  now **caches the smoothed normalized H/V** as 2D state and per move re-smooths
  only the touched-net pin-bbox columns/rows *from raw flats* (recomputing from
  raw, not accumulating deltas, keeps it bit-identical to a full re-smooth with
  no drift). All six move paths thread the bbox through `_resmooth_bbox`. Swap
  Δ stays at machine eps (≤4.4e-16); hard/soft move ≤1.8e-9.
  Isolated `--all`: **1.2799 → 1.2767**.
- **Idea #1 subset-cumsum strip-batch.** `_apply_h/v_strips_batch` was the
  inner-inner-loop of the 67% routing-apply path — it allocated a full
  `(grid_row, grid_col+1)` diff array, scattered with `np.add.at`, then
  cumsummed *every row*. The diff-array cumsum is per-row independent, so
  unique-ing the touched rows and cumsumming only those is bit-identical, and
  cuts both the alloc and the cumsum to the touched subset.
- **Idea #2 topology-struct cache.** The routing apply mixes **placement-
  independent bookkeeping** (which pins, lengths, 2/3/≥4-pin classification,
  ≥4-pin sink index layout) with the **position-dependent fill** (gcell
  extraction + dispatch). Split into `_build_net_routing_struct` (cacheable per
  macro) + `_apply_net_routing_struct`; the scorer keeps a per-module struct
  cache so single-macro paths build the structure *once per macro* and reuse it
  across every candidate target and across the −1 / +1 applies. Swap builds
  once per call. Init path keeps the original `_apply_net_routing_subset`
  (additive — the full-build path is unchanged).
- **Floor-reservation budget allocator.** Closes the ibm18-starvation bug: in
  the old fair-share allocator a few large benchmarks' overruns ate the tail's
  budget, and the guard returned baseline whenever `cumulative > 95%·3300`. The
  new allocator reserves `(PER_BENCH_FLOOR_S=110 + BUDGET_OVERRUN_S=60)·(remaining−1) + 60`
  for the others' overrun + own overrun, clamps to a 3540 s hard-cap headroom,
  and floors at 110 s. Worst-case simulation (every benchmark overruns by 60 s)
  has all 17 benchmarks at ≥110 s and cumulative ending exactly at 3300. The
  guard reduces to `eff < 45 s → baseline` (only fires on genuine exhaustion).
- **A: round-3 cong cap + C: density `top_hot` boost.** The cong soft-pass
  saturates by round 3 (ibm09: round 4+ accepts ≤2 moves, ~zero gain) while
  density keeps finding moves through round 6. Skip cong on `_r2 ≥ 3` (A) and
  bump density's candidate set 128 → 192 on those rounds (C) so the freed
  ~4–5 s/round is spent on more density attempts. Combined with the speedup
  stack: `--all` **1.2767 → 1.2755**.

The whole stack is **strictly bit-exact** (verified by the three move-path
verifiers: `_verify_incremental_scorer.py`, `_verify_score_move.py`,
`_verify_score_move_soft.py`) and **strictly non-regressing** (accept-on-true-
proxy is preserved end-to-end). Diagnostics that produced and constrained the
plan: the fixed-overhead measurement (retired the shared-scorer refactor — per-pass fixed
overhead is 0.1–0.28 s/round, not the 60–75 s estimated), `_profile_move.py`
and `_profile_move_internals.py` (cong cost 17%, density 0.7%, routing-apply
67% → the latter two are where the speedups were targeted), and
`_profile_move_realistic.py` (isolates the topology-struct cache benefit by A/B-ing
the same-macro / nearby pattern vs the cache-defeating random-k pattern).

## Closed dead-ends (don't re-run without a specific reason — see ISSUES.md)

| Direction | Outcome |
|---|---|
| **DP1** congestion-aware DREAMPlace (`routability_opt`) | CLOSED — DREAMPlace's RUDY congestion ≠ TILOS proxy; no-op or worse across a 64× capacity sweep. Details are in `docs/general/ISSUES.md` DP1. |
| **Phase 7b** post-hoc DP-basin repair | REVERTED — recoverable in a probe but budget-hungry, high-variance, not reproducible at fixed seed. |
| **S1** basin-hopping 2-opt (cong-grad kick) | DISPROVEN — slicing the budget starves the deadline-bound search; 6/7 worse. |
| **O3** soft-macro repositioning (bulk/gradient) | CLOSED for bulk methods — R5 discrete soft relocation is what works. |
| **R4** WL-aware hard-relocation (net-centroid target bias) | DISPROVEN — slightly worse than nearest-to-current; probe scaffolding removed. |
| **Shared-scorer interleave refactor** (the original P5 plan) | RETIRED — fixed-overhead profiling measured 0.1–0.28 s/round (not the projected 60–75 s), so the refactor would save <1.7 s/benchmark and risk the bit-exact core. Replaced by the incremental-cong-cost + #1 + #2 stack above. |

## Source layout

The submission now lives under `src/`. `src/main.py` is the evaluator-facing
entrypoint; it imports `MacroPlacer` from `placer.pipeline.macro_placer` and
keeps compatibility delegation for diagnostics that still reach private helpers
through the submission module.

```
.
├── README.md
├── requirements.txt
├── scripts/                 # collect_ml_data.sh — ML training-data collection
├── ml_data/                 # collected candidate traces + logs (gitignore-worthy)
├── docs/
├── test/
└── src/
    ├── main.py
    ├── place_design.py      # eda_io CLI: standard EDA files in/out
    ├── eda_io/              # LEF/DEF/Verilog/SDC/Liberty readers, DEF/Tcl/report writers
    ├── dreamplace_bridge/
    │   ├── bookshelf_to_pb.py
    │   ├── pb_to_bookshelf.py
    │   └── run_bridge.py
    └── placer/
        ├── config.py
        ├── geometry.py
        ├── pipeline/
        │   └── macro_placer.py
        ├── scoring/
        │   ├── exact.py
        │   ├── incremental.py
        │   ├── wirelength.py
        │   ├── density.py
        │   └── congestion.py
        ├── routing/
        │   └── apply.py
        ├── plc/
        │   ├── loader.py
        │   └── placement.py
        ├── legalize/
        │   ├── spiral.py
        │   └── swap.py
        ├── local_search/
        │   ├── fields.py
        │   ├── two_opt.py
        │   ├── relocation.py
        │   ├── soft_moves.py
        │   ├── hard_soft.py
        │   └── workers.py
        ├── perturb/
        │   └── congestion_gradient.py
        └── ml/
            ├── data_collection.py
            ├── dataset.py
            ├── modeling.py
            ├── shadow.py
            └── train.py
```

### Module responsibilities

| Path | Purpose |
|---|---|
| `src/main.py` | **Submission entrypoint** for `uv run evaluate`; wraps `MacroPlacer` so the evaluator sees class module `main`. |
| `src/placer/pipeline/macro_placer.py` | Top-level orchestration: budgeted candidate generation, DREAMPlace integration, R2 loop scheduling, final placement selection. |
| `src/placer/config.py` | Runtime config, GPU backend detection, numba feature flag, and `_log`. |
| `src/placer/geometry.py` | Shared geometry helpers — `separation_matrices` (pairwise minimum non-overlap separations used by the legalizer + 2-opt / relocation conflict checks). |
| `src/placer/scoring/exact.py` | Exact proxy wrapper over patched `PlacementCost`: WL + density + congestion. |
| `src/placer/scoring/incremental.py` | `IncrementalScorer`, the stateful bit-exact scorer used by swaps and relocation moves. |
| `src/placer/scoring/{wirelength,density,congestion}.py` | Vectorized PLC scoring patches and cost helpers. |
| `src/placer/routing/apply.py` | Vectorized routing demand, strip batching, 2-pin/3-pin dispatch, smoothing, and routing subset/struct apply helpers. |
| `src/placer/plc/loader.py` | `PlacementCost` loader. |
| `src/placer/plc/placement.py` | Position cache and fast placement setter used by scoring. |
| `src/placer/legalize/` | Minimum-displacement legalization and hard-macro swap legality helpers. |
| `src/placer/local_search/` | 2-opt, relocation, soft moves, hard-soft moves, hot/cold cell fields (`fields.py`), and multiprocessing workers. |
| `src/placer/perturb/congestion_gradient.py` | Congestion-gradient perturbation used by global move phases. |
| `src/placer/ml/` | Training-data collection and inference for the learned candidate ranker — `data_collection.py` (`CandidateTrace`, `TraceFields`, `net_degree_features`; active only when `ML_TRACE_PATH` is set), `dataset.py` (trace-JSONL loaders + `add_group_relevance`), `modeling.py` / `train.py` (offline model tooling), and `shadow.py` (shadow diagnostics + production filtering fallback guards). See "ML candidate-ranker data collection" below. |
| `scripts/collect_ml_data.sh` | Runs the placer with `ML_TRACE_PATH` set across a seed sweep (`--all` or `--ng45`) to produce the training traces in `ml_data/`. |
| `src/dreamplace_bridge/` | pb.txt ↔ Bookshelf converters + async DREAMPlace subprocess launcher. |
| `src/eda_io/` | Plug-and-play EDA I/O layer: parses LEF/DEF/Verilog/SDC/Liberty into a neutral `Design`, converts to ICCAD04 pb+plc (placer + exact scorer unchanged), writes updated DEF / ICC2-Innovus Tcl / QoR reports. See `src/eda_io/README.md`. |
| `src/place_design.py` | CLI for the eda_io layer — any input combo in, any output combo out. |
| `docs/general/ARCHITECTURE.md` | Design overview + pipeline visualization + algorithm explanations. Start here for the "how it works" tour. |
| `docs/general/PROGRESS.md` | Per-benchmark results + full experiment history. Source of truth for "what works". |
| `docs/general/ISSUES.md` | Open issues + closed dead-ends with evidence. |
| `test/verification/` | Bit-exactness checks vs the scalar reference. |
| `test/diagnostic/` | Profiling + analysis. |

### Recent system changes

- **ML hard-relocation filter connected as default (2026-06-11).**
  `src/main.py` now enables the validated S10 config B (wide-32 pool, ranker
  keeps 16) whenever no `ML_*` env var is set and the shipped model +
  `xgboost` are available. The pipeline logs
  `R2 hard relocation ML filter on (pool=32, top_k=16)` when active.
  Verified: `test/verification/_verify_ml_filter_wiring.py` + ibm01
  end-to-end (proxy 0.9146, VALID, 71s, filter line present) + same-day
  `--all` re-baseline: **avg 1.1252, 17/17 VALID, 0 overlaps, 2337s** (new
  best; was 1.1272). **Gain confirmed by same-day paired multi-seed `--all`**:
  Δ(ON−OFF) = −0.0051 / −0.0044 / −0.0029 across 3 seeds (mean **−0.0041**,
  filter wins 3/3; ON mean 1.1245 vs OFF 1.1286; all 6 runs 17/17 VALID).
  Logs: `ml_data/compare/all_20260611_{on,off}_s{def,43,44}.log`.
- **ML candidate-ranker data collection (2026-06-04).** Added
  `scripts/collect_ml_data.sh` + a default-preserving `V2_SEED` knob in
  `src/main.py` to capture the training traces (see the section below). No
  change to the placer's algorithm or inference path; `V2_SEED` is unset in
  real evaluation.
- **Readability refactor (2026-06-04, no algorithm change).** Consolidated the
  ML-trace per-candidate congestion/density feature lookups into a `TraceFields`
  helper (`ml/data_collection.py`); deduped the 7× pairwise separation matrices
  into `geometry.separation_matrices`; extracted `place()`'s budget and
  DREAMPlace-launch setup into `_effective_budget` / `_launch_dreamplace_seeds`
  methods (the cong-grad phase descent kept inline). Pure code-motion, ML data
  collection byte-identical (`test/verification/test_trace_fields_equivalence.py`),
  validated non-degrading at `--all` (avg 1.1500, 17/17 VALID, 0 overlaps).
- Replaced the old monolithic `placer.py` submission with `src/main.py` plus
  a package under `src/placer/`.
- Moved `dreamplace_bridge/` under `src/` and updated bridge root discovery so
  it still finds the repository from the nested package location.
- Split scoring, routing, PLC state management, legalization, local-search
  moves, perturbation, and pipeline orchestration into separate modules.
- Moved the multiprocessing 2-opt seed worker into
  `src/placer/local_search/workers.py`; it is now importable as a normal module
  instead of relying on the old synthetic pickle wrapper.
- Updated package `__init__.py` exports for `scoring`, `routing`, `plc`,
  `local_search`, `legalize`, `perturb`, and `pipeline`.
- Verified the reorganized entrypoint with bytecode compilation, import smoke,
  and `uv run evaluate src/main.py -b ibm01`
  (`VALID`, proxy `0.9078`, CUDA backend detected locally).

## ML candidate-ranker data collection

Status: **the hard-relocation ranker is wired into the placer by default
(2026-06-11).** `src/main.py` enables the S10 equal-budget config B when no
`ML_*` env var is set: the R2 hard-relocation pool widens to 32 candidates and
the shipped XGBoost ranker (`ml_data/models/clean-wide32-holdout-ibm13-001`)
picks the 16 to exact-score. The accept-on-true-proxy gate is unchanged, so the
search stays strictly non-regressing. Setting any `ML_*` env var (e.g.
`ML_FILTER_OPERATORS=""`) skips the defaults entirely, keeping trace
collection, shadow diagnostics, and sweeps at their exact prior semantics; a
missing model file or missing `xgboost` also falls back to the pure-heuristic
narrow-16 path. Wiring check:
`test/verification/_verify_ml_filter_wiring.py`. Full design + validation:
`docs/general/ISSUES.md` S10; conceptual notes: `docs/ml_nn/`.

How the data is produced:

```bash
# IBM (17 benchmarks) and NG45 (Tier 2) seed sweeps; runs detached for hours.
scripts/collect_ml_data.sh 42 43 44          # IBM
scripts/collect_ml_data.sh --ng45 42 43 44   # NG45
```

- The script sets `ML_TRACE_PATH`, so `placer.ml.data_collection.CandidateTrace`
  writes one JSONL row per local-search candidate trial: pre-score features +
  `score_gain`/`improves` labels + a `group_id` per decision. `V2_SEED` varies
  the seed for distinct trajectories.
- Output lands in `ml_data/traces/*.jsonl.gz` (IBM `s*`, NG45 `ng45_s*`) with
  per-run logs in `ml_data/logs/`. Load it via `placer.ml.dataset.load_candidates`
  + `add_group_relevance` (LambdaMART labels).
- **Tracing changes timing, which changes scores** — these runs are for data
  only; never read a placement score off a traced run.
- Current dataset (seeds 42/43/44, IBM + NG45): ~12.6M candidate rows
  (~1.2 GB). `hard_relocation` is the leanest operator (~190k rows), so the NG45
  cross-design data matters most for it.
- Training deps (`xgboost`, `scikit-learn`) are in `requirements.txt`,
  offline-only — not imported on the submission's inference path. `ml_data/`
  is large and gitignore-worthy.

## Using the placer outside the challenge (eda_io)

The v2 placer is usable in any physical-design flow via `src/eda_io/` +
`src/place_design.py`: it accepts standard EDA inputs (**LEF, DEF, structural
Verilog, SDC, Liberty** — mix freely, minimum is one geometry source + one
instance source) and emits standard outputs (**updated DEF** with exact
locations/orientations/PLACED-FIXED flags, **Tcl** sourceable in ICC2 or
Innovus, **QoR .rpt** with HPWL/legality/proxy breakdown, and the standard
visualization PNG). Every input combo is merged into one neutral `Design`
and converted to the ICCAD04 `netlist.pb.txt` + `initial.plc` pair, so the
unchanged placer and the exact TILOS scorer run on external designs exactly
as on the challenge benchmarks. SDC raises weights on timing-critical nets
so critical macros are pulled together; fixed components and DEF blockages
are honored. Full documentation: [`src/eda_io/README.md`](src/eda_io/README.md).

```bash
uv run python src/place_design.py \
    --lef tech.lef --lef macros.lef --def floorplan.def --sdc top.sdc \
    --out-def placed.def --out-tcl place.tcl --report qor.rpt
```

Tests: `uv run --with pytest python -m pytest test/eda_io/ -v`

## Reproducing the DREAMPlace build (`dreamplace_build/`, gitignored ~500MB)

```
sudo apt install -y flex bison libboost-all-dev
# clone DREAMPlace into dreamplace_src/, then:
cmake .. -DCMAKE_CXX_ABI=1 -DPython_EXECUTABLE=$(which python)
make -j2 install      # NOT -j$(nproc) — OOM
sed -i 's/np\.string_/np.bytes_/g' install/dreamplace/PlaceDB.py   # NumPy 2.0
```
The retired routability experiment and vendor-patch notes are summarized in
`docs/general/ISSUES.md` DP1; the production bridge keeps only the active seed flow.

## Commands

```bash
uv run evaluate src/main.py -b ibm04      # single benchmark
uv run evaluate src/main.py --all         # headline (~39 min)
uv run python scripts/compare_placers.py system/v1/placer.py src/main.py
uv run python test/verification/_verify_score_move.py
```
