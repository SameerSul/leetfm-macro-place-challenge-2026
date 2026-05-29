# v2 — Varrahan's Submission

Active placer for the Partcl/HRT Macro Placement Challenge. A multi-restart
legalization placer with **congestion-gradient global moves**, a **fully-
incremental proxy scorer**, and **move-based local search** (2-opt swaps +
congestion-directed relocation) on top.

**Headline (`--all`, 2026-05-29 — combined stack): avg `1.2755`** — beats the
RePlAce target (`1.4578`) by **12.5%**, all 17 IBM benchmarks VALID / 0
overlaps. **Beats the #1 leaderboard** (UT Austin DREAMPlace, `1.4076`) by
**0.132 (−9.4%)**. Driven by the **relocation family** (R1/R2/R3/R5) plus a
**bit-exact scoring-speedup stack**: (i) **incremental congestion cost** (cache
the smoothed normalized H/V; re-smooth only the touched-net bbox per move,
bit-identical), (ii) **#1 subset-cumsum strip-batch** (cumsum only the touched
rows/cols), (iii) **#2 topology-struct cache** for the routing apply (the
position-independent gather is built once per macro and reused across moves and
the −1/+1 applies), (iv) a **floor-reservation budget allocator** (every
benchmark in `--all` is guaranteed ≥110 s — no last-benchmark starvation), and
(v) a **round-3 cong cap + density `top_hot=192` boost** (cong soft-pass
saturates by round 3 — reclaim those cycles for more density attempts).
Prior milestone (R5 alone): 1.2799; incremental cong cost alone: 1.2767.

> Source of truth for numbers and experiment history is [`docs/PROGRESS.md`];
> open issues / closed dead-ends are in [`docs/ISSUES.md`]; DREAMPlace patches
> are in [`docs/DREAMPLACE_FIXES.md`]. This README is the architectural overview.

## What's being optimized

```
proxy_cost = 1.0·wirelength + 0.5·density + 0.5·congestion
```
After normalization, **congestion ≈ 65% of proxy**, density ≈ 30%, wirelength
≈ 5%. The whole strategy follows from this: our edge is **direct hard-macro
congestion optimization**, and WL-only optimization reliably makes proxy *worse*
(clustering spikes congestion).

## Pipeline

```
0    Baseline           legalize from initial.plc (vectorized _will_legalize)
─    Multi-DP (async)   3 DREAMPlace candidates launched in parallel:
                          lo-fix (td=0.65, soft fixed), hi-mov (td=0.85, soft
                          movable), hi-fix (td=0.85, soft fixed)
1-3  cong-grad          iterative max(H,V) gradient descent from baseline
                          (frac 0.04, wide 0.08/0.12, adaptive halving)
5b/5c cong-grad         from best_pl / wide-from-best
7    DP-rescue          cong-grad chains seeded from each DP candidate
8    TOP-K cong-grad     move only the K hottest macros from best_pl
9    random-order        legalize with randomized tie-break order
─    multi-seed 2-opt    proxy-driven 2-opt (k=20) from best_pl + each DP basin,
                          select by true _exact_proxy (prune window 0.02)
─    R2 interleave       alternate {relocation pass, 2-opt cleanup} until neither
                          improves (≤6 rounds) — see "Relocation" below
```
All candidates legalized then scored via exact `PlacementCost` proxy; lowest
wins. Adaptive 200s + 60s-overrun per-benchmark budget; thresholds admit all 17.

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
  vice versa, compounding over ≤6 rounds. `1.4326 → 1.4243`.
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
  (`use_density`) finds moves the cong pass can't (`DENS_SOFT_PROBE`: cong-converged
  best_pl still yields 22–68 density moves). Interleaved (hard ⇄ soft-cong ⇄
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
plan: `_profile_init.py` (retired the shared-scorer refactor — per-pass fixed
overhead is 0.1–0.28 s/round, not the 60–75 s estimated), `_profile_move.py`
and `_profile_move_internals.py` (cong cost 17%, density 0.7%, routing-apply
67% → the latter two are where the speedups were targeted), and
`_profile_move_realistic.py` (isolates the topology-struct cache benefit by A/B-ing
the same-macro / nearby pattern vs the cache-defeating random-k pattern).

## Closed dead-ends (don't re-run without a specific reason — see ISSUES.md)

| Direction | Outcome |
|---|---|
| **DP1** congestion-aware DREAMPlace (`routability_opt`) | CLOSED — DREAMPlace's RUDY congestion ≠ TILOS proxy; no-op or worse across a 64× capacity sweep. (Required a real bug-fix to even run: NCTUgr-map guard, see DREAMPLACE_FIXES.md.) |
| **Phase 7b** post-hoc DP-basin repair | REVERTED — recoverable in a probe but budget-hungry, high-variance, not reproducible at fixed seed. |
| **S1** basin-hopping 2-opt (cong-grad kick) | DISPROVEN — slicing the budget starves the deadline-bound search; 6/7 worse. |
| **O3** soft-macro repositioning (bulk/gradient) | CLOSED for bulk methods — R5 discrete soft relocation is what works. |
| **R4** WL-aware hard-relocation (net-centroid target bias) | DISPROVEN — slightly worse than nearest-to-current; scaffolding kept inert (`wl_blend`, `hard_net_centroids()`, `WLAWARE_PROBE`). |
| **Shared-scorer interleave refactor** (the original P5 plan) | RETIRED — `_profile_init.py` measured the per-pass fixed overhead at 0.1–0.28 s/round (not the projected 60–75 s), so the refactor would save <1.7 s/benchmark and risk the bit-exact core. Replaced by the incremental-cong-cost + #1 + #2 stack above. |

## File / docs index

| Path | Purpose |
|---|---|
| `placer.py` | **The submission** (~5000 lines). Pipeline above + `IncrementalScorer` + `_two_opt_proxy_swap` + `_relocation_moves`. |
| `docs/ARCHITECTURE.md` | **Design overview + pipeline visualization + algorithm explanations** (incl. DREAMPlace integration). Start here for the "how it works" tour. |
| `docs/PROGRESS.md` | Per-benchmark results + full experiment history. Source of truth for "what works". |
| `docs/ISSUES.md` | Open issues + closed dead-ends with evidence (R1/R2/DP1/S1/S9/O3/P3…). |
| `docs/DREAMPLACE_FIXES.md` | DREAMPlace bridge/source patches (gitignored vendor trees → recorded here for reapply). |
| `dreamplace_bridge/` | pb.txt ↔ Bookshelf converters + async subprocess launcher (`launch_dreamplace_async`). |
| `test/verification/` | Bit-exactness checks vs the scalar reference (`_verify_incremental_scorer.py`, `_verify_score_move.py`, …). |
| `test/diagnostic/` | Profiling + analysis (`_profile_density.py`, `_term_breakdown.py`, `_reloc_leverage.py`, …). |
| `test/dreamplace/` | DREAMPlace bridge tests + DP1 probes (`_routopt_poc.py`, `_routopt_calib.py`, …). |

### Env-gated diagnostics in `placer.py` (no effect unless set)

`DP_DIAG=1` (decompose DP candidates vs best), `DP_PROBE=1` (DP-basin
recoverability ceiling test), `RELOC_PROBE=1` (relocation-on-best probe).

## Reproducing the DREAMPlace build (`dreamplace_build/`, gitignored ~500MB)

```
sudo apt install -y flex bison libboost-all-dev
# clone DREAMPlace into dreamplace_src/, then:
cmake .. -DCMAKE_CXX_ABI=1 -DPython_EXECUTABLE=$(which python)
make -j2 install      # NOT -j$(nproc) — OOM
sed -i 's/np\.string_/np.bytes_/g' install/dreamplace/PlaceDB.py   # NumPy 2.0
```
Plus the NCTUgr-map guard patch in `docs/DREAMPLACE_FIXES.md` if enabling
`routability_opt` (otherwise it crashes on Bookshelf inputs).

## Commands

```bash
uv run evaluate submissions/varrahan/v2/placer.py -b ibm04      # single benchmark
uv run evaluate submissions/varrahan/v2/placer.py --all         # headline (~25 min)
uv run python scripts/compare_placers.py submissions/varrahan/v1/placer.py submissions/varrahan/v2/placer.py
uv run python submissions/varrahan/v2/test/verification/_verify_score_move.py
```
