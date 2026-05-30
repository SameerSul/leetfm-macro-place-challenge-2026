# v2 — Architecture

This document is the **architectural overview** of the v2 placer. For
per-benchmark numbers + experiment history, see [`PROGRESS.md`]; for open issues
and closed dead-ends, see [`ISSUES.md`]; for DREAMPlace bridge / source patches,
see [`DREAMPLACE_FIXES.md`].

Headline (`--all`, 2026-05-29 — combined stack): **avg `1.2755`** — beats
RePlAce (1.4578) by **12.5%** and the UT Austin DREAMPlace leaderboard (1.4076)
by **9.4%**. All 17 IBM benchmarks VALID / 0 overlaps, bit-exact verified.

---

## 1. Design philosophy

v2 is a **multi-restart legalizer + move-based local-search placer**. It starts
from the hand-tuned `initial.plc` spread and refines it through a chain of
progressively-finer optimizations, each gated by a strict accept-on-true-proxy
rule so the search is **non-regressing by construction**.

Three observations drive every design choice:

### 1.1 The proxy is congestion-dominated

After the TILOS evaluator's normalization,

```
proxy = 1.0 · wirelength + 0.5 · density + 0.5 · congestion
```

has weights of roughly **WL 5% / density 30% / congestion 65%** of the
proxy value. WL-only optimization (clustering connected macros) reliably makes
proxy *worse* because the clustering spikes density and congestion. **Our edge
is direct congestion + density optimization.** Every algorithmic choice in
v2 follows from this.

### 1.2 `initial.plc` is already a good seed

`initial.plc` comes from a prior EDA flow with hand-tuned spread. The job of
legalization is to resolve overlaps **without destroying that spread** — small
perturbations + local search beats construct-from-scratch. This is empirically
proven: WireMask-BBO (rebuild from scratch with WL+congestion penalty),
DREAMPlace standalone, gradient-descent on soft macros — all lost to
restart-from-`initial.plc` + cong-grad + local search.

### 1.3 Soft macros are the bulk of both terms

There are ~900–2000 soft macros per benchmark (vs ~250–800 hard). They may
overlap each other, they carry the bulk of the routing demand AND occupy the
bulk of grid cells, and **every prior placer froze them at `initial.plc`**.
Relocating them is the dominant lever — **R5 soft density relocation alone
took us 1.3764 → 1.2799 (−0.0965)**, larger than every other change combined.

---

## 2. What makes v2 special

### 2.1 Soft-macro relocation as a first-class move type

Every other placer in the benchmark set leaves soft macros at `initial.plc`.
We relocate them. Specifically, in each round of the local-search loop we run
**two distinct soft-pass operations**:

- **R3 (congestion field):** find the hottest softs by the routing
  `max(H, V)` field; move each to the nearest colder cell that lowers the
  proxy.
- **R5 (density field):** find the softs in the densest grid cells (the scorer's
  maintained `grid_occupied`); move each to the lowest-occupancy cell that
  lowers the proxy.

The two passes target *different fields* and find *different moves*. Softs may
overlap, so the cong pass can pile them in a low-congestion cell without
relieving density — the density pass cleans that up. Net: R3 alone 1.4216 →
1.3764; R5 added on top 1.3764 → 1.2799.

### 2.2 Bit-exact fully-incremental proxy scorer

A move-based local search is only useful if the scorer is fast enough to
evaluate thousands of candidates per deadline. Off-the-shelf `_exact_proxy`
scatters every macro into the congestion + density grids and recomputes WL
over every net — way too slow for inner-loop scoring.

The `IncrementalScorer` maintains the full proxy state and updates only what a
single move touches:

| Term       | Maintained state          | Per-move update                                            |
|------------|---------------------------|-------------------------------------------------------------|
| Wirelength | `per_net_hpwl`, `total_wl_raw` | Subtract old HPWL of touched nets, add new (B3p2)        |
| Congestion | H/V routing flats + cached smoothed H/V (2D) | Apply ∓1 routing on touched nets; re-smooth touched bbox cols/rows from raw (B3p4 + incremental cong cost) |
| Density    | `grid_occupied`           | Subtract/add the moved macro's footprint cells (P3)         |

Net per-move cost: **~1.4 ms**, vs ~10–30 ms for a full `_exact_proxy`. And
crucially, the incremental scorer is **bit-exact** vs the full scorer:

- `score_swap`: Δ ≤ **4.4e-16** (machine epsilon)
- `score_move` (hard): Δ ≤ 1.8e-9
- `score_move_soft`: Δ ≤ 5e-10
- No drift over sequential commits (verified in `_verify_incremental_scorer.py`,
  `_verify_score_move.py`, `_verify_score_move_soft.py`).

The accept gate uses the incremental score directly; the interleave loop
validates each pass's net result with a true `_exact_proxy` re-score. The
strict accept-on-true-proxy rule is what makes the entire chain
**non-regressing by construction**.

### 2.3 Two-field, multi-pass interleave

Each round of the local-search outer loop runs four passes in sequence:

1. **Hard relocation** — move hot hard macros into cold legal gaps.
2. **Soft cong relocation (R3)** — hottest softs into cold-cong cells.
3. **Soft density relocation (R5)** — softs in dense cells into low-occupancy cells.
4. **2-opt cleanup** — small swaps around the relocations.

Each pass opens new moves for the next pass, and the round repeats until no
pass finds a true-proxy improvement (up to 6 rounds). Bit-exact + non-regressing
across the entire chain.

### 2.4 Bit-exact scoring-speedup stack (1.2799 → 1.2755)

Five mutually compounding changes, **each verified bit-exact**, that buy more
search per deadline (and contention-robustness for the official evaluation):

1. **Incremental congestion cost** (cache smoothed H/V; per move re-smooth
   only the touched-net bbox from raw flats — bit-identical to a full re-smooth,
   no drift).
2. **Idea #1 subset-cumsum strip-batch** (cumsum only the touched rows/cols
   in the difference-array routing fill).
3. **Idea #2 topology-struct cache** for the routing apply (the placement-
   independent bookkeeping is built once per macro and reused across moves and
   the −1/+1 applies within each).
4. **Floor-reservation budget allocator** (every benchmark in `--all` is
   guaranteed ≥110 s — closes the ibm18-starvation bug we hit during
   development).
5. **A: round-3 cong cap + C: density `top_hot` boost** (cong soft-pass
   saturates by round 3; skip on rounds 4–6 and spend the freed ~4–5 s/round
   on density attempts with a wider candidate set, 128 → 192).

### 2.5 DREAMPlace as a seed, not a destination

DREAMPlace is the academic SOTA (leaderboard #1 at 1.4076 standalone). Its
strength is WL + density via Nesterov-accelerated analytical placement; its
blind spot is the TILOS proxy's **top-5% congestion peaks**, which its
objective doesn't see. Standalone, DP wins only 2/17 final-seed races against
our cong-grad pipeline. We use DP as a **side-channel seed** — launched
async, its result becomes one of several candidates the multi-seed 2-opt
mines. See § 5 for the bridge architecture and our DREAMPlace patches.

---

## 3. What gives us the high score (decomposition)

The lever stack, ranked by magnitude:

| # | Lever | `--all` Δ | Cumulative |
|---|---|---|---|
| 1 | **R5 soft density relocation** | −0.0965 | 1.3764 → 1.2799 |
| 2 | **R3 soft cong relocation** | −0.0452 | 1.4216 → 1.3764 |
| 3 | **A2: DREAMPlace as candidate** | −0.0161 | 1.4647 → 1.4486 |
| 4 | **R1 hard relocation pass** | −0.0096 | 1.4422 → 1.4326 |
| 5 | **R2 interleave + 2-opt widening** | −0.0083 | 1.4326 → 1.4243 |
| 6 | **A1 proxy 2-opt + B3 incremental scorer** | −0.0059 | 1.4782 → 1.4723 |
| 7 | **Incremental cong cost** (bit-exact speedup) | −0.0032 | 1.2799 → 1.2767 |
| 8 | **R2b widened relocation candidates** | −0.0027 | 1.4243 → 1.4216 |
| 9 | **#1 + #2 + floor-res + A+C** (bit-exact stack) | −0.0012 | 1.2767 → 1.2755 |
| — | (small wins) S2 k=20, S9 cong-aware 2-opt, etc. | ~ −0.003 each | |

**The dominant lever — by an order of magnitude — is soft-macro relocation
(R5 + R3 = −0.142 combined).** Everything else is necessary infrastructure:
the incremental scorer (so the moves are affordable), the interleave
(so the passes compound), DREAMPlace and cong-grad (so the local search has
a good basin to start from), and the speedup stack (so more search fits in
the budget under contention).

**The lever fails where soft macros aren't the bottleneck** — but it doesn't
exist where soft macros aren't the bottleneck either. The leverage analysis
(`test/diagnostic/_reloc_leverage.py`) shows per-benchmark gain correlates with
**hard-macro utilization × congestion headroom**, NOT with macro dominance or
open space.

---

## 4. The pipeline

The placer is invoked once per benchmark via `MacroPlacer.place(benchmark) →
torch.Tensor[num_macros, 2]`. Internally, `place()` runs the following pipeline:

```
                       ┌───────────────────────────────────────┐
                       │  initial.plc   (hand-tuned spread)    │
                       └───────────────────────────────────────┘
                                          │
                ┌─────────────────────────┼─────────────────────────┐
                │                         │                         │
                ▼                         ▼                         ▼
   ┌────────────────────────┐  ┌────────────────────────┐  ┌──────────────────────┐
   │ Phase 0 — Baseline     │  │ Phase 5: DREAMPlace ×3 │  │ Noise restarts       │
   │ _will_legalize         │  │ async subprocess       │  │ (k=4-50 fracs)       │
   │ (vectorized rings)     │  │ lo-fix / hi-mov /      │  │ small Gaussian       │
   │                        │  │ hi-fix configs         │  │ perturbations        │
   └────────────────────────┘  └────────────────────────┘  └──────────────────────┘
                │                         │                         │
                ▼                         ▼                         │
   ┌────────────────────────┐  ┌────────────────────────┐           │
   │ Phase 1-3 cong-grad    │  │ legalize DP outputs    │           │
   │ iterative descent +    │  │ + score                │           │
   │ wide-step fallback     │  └────────────────────────┘           │
   └────────────────────────┘             │                         │
                │                         │                         │
                ▼                         │                         │
   ┌────────────────────────┐             │                         │
   │ Phase 5b/5c cong-grad  │             │                         │
   │ from best_pl + wide-   │             │                         │
   │ from-best              │             │                         │
   └────────────────────────┘             │                         │
                │                         │                         │
                ▼                         ▼                         │
   ┌────────────────────────────────────────────────────┐           │
   │ Phase 7 DP-rescue                                  │           │
   │ cong-grad chain from each DP candidate basin       │           │
   └────────────────────────────────────────────────────┘           │
                │                                                   │
                ▼                                                   │
   ┌────────────────────────┐                                       │
   │ Phase 8 TOP-K          │                                       │
   │ cong-grad on K hottest │                                       │
   │ macros only            │                                       │
   └────────────────────────┘                                       │
                │                                                   │
                ▼                                                   │
   ┌────────────────────────┐                                       │
   │ Phase 9 random-order   │◄──────────────────────────────────────┘
   │ legalize variants      │
   └────────────────────────┘
                │
                ▼
   ┌────────────────────────────────────────────────────────────────┐
   │  Multi-seed 2-opt (k_neighbors=20, max_iters=6)                │
   │  from best_pl + each DP candidate basin                        │
   │  with S9 cong-aware candidate selection:                       │
   │    hot-first outer ordering + cold-region teleport             │
   │  Final selection across seeds by true _exact_proxy             │
   └────────────────────────────────────────────────────────────────┘
                │
                ▼
   ╔════════════════════════════════════════════════════════════════════╗
   ║  R2 interleave loop (≤6 rounds, accept-on-true-proxy)              ║
   ║  ┌──────────────────────────────────────────────────────────────┐  ║
   ║  │  Round r:                                                    │  ║
   ║  │  ┌─────────────────────────────────────────────────────┐    │  ║
   ║  │  │  Hard relocation (R1/R2/R2b)                        │    │  ║
   ║  │  │    top_hot=48 by max(H,V), n_targets=16             │    │  ║
   ║  │  │    accept on true-proxy drop                        │    │  ║
   ║  │  └─────────────────────────────────────────────────────┘    │  ║
   ║  │                       ▼                                      │  ║
   ║  │  ┌─────────────────────────────────────────────────────┐    │  ║
   ║  │  │  Soft cong relocation (R3) — IF r ≤ 3 (A: hard cap) │    │  ║
   ║  │  │    top_hot=128, n_targets=24                        │    │  ║
   ║  │  │    field = plc routing max(H,V)                     │    │  ║
   ║  │  └─────────────────────────────────────────────────────┘    │  ║
   ║  │                       ▼                                      │  ║
   ║  │  ┌─────────────────────────────────────────────────────┐    │  ║
   ║  │  │  Soft density relocation (R5)                       │    │  ║
   ║  │  │    top_hot = 128 (r ≤ 3) or 192 (r > 3, C: boost)   │    │  ║
   ║  │  │    field = grid_occupied / dens_grid_area           │    │  ║
   ║  │  └─────────────────────────────────────────────────────┘    │  ║
   ║  │                       ▼                                      │  ║
   ║  │  ┌─────────────────────────────────────────────────────┐    │  ║
   ║  │  │  2-opt cleanup (8s budget slice)                    │    │  ║
   ║  │  │    k=20 spatial kNN + S9 cold-teleport              │    │  ║
   ║  │  └─────────────────────────────────────────────────────┘    │  ║
   ║  │                       ▼                                      │  ║
   ║  │       round_improved? — yes → next round                     │  ║
   ║  │                       — no  → terminate                      │  ║
   ║  └──────────────────────────────────────────────────────────────┘  ║
   ╚════════════════════════════════════════════════════════════════════╝
                │
                ▼
                ┌──────────────────────────┐
                │  best placement returned │
                │  (centers, [num_macros, 2]) │
                └──────────────────────────┘
```

Everything above runs inside a single per-benchmark budget
(`effective_budget_s`) allocated by the **floor-reservation allocator** (§4.2).

### 4.1 Pipeline phase reference

| Phase | What it does | Why |
|---|---|---|
| **0 Baseline** | Legalize from `initial.plc` via `_will_legalize` (vectorized greedy spiral) | Establish a valid baseline; preserve the hand-tuned spread |
| **5 DREAMPlace ×3 (async)** | Launch DP at `place()` entry in 3 configs (lo-fix, hi-mov, hi-fix) | Side-channel seeds with different basins; legalize each and add to candidates |
| **1-3 cong-grad** | Iterative max(H,V) gradient-descent perturbation from baseline (`frac=0.04`, ≤4 steps; wide-step fallback at frac=0.08/0.12) | Escape WL-trap local minima; relieve congestion peaks |
| **5b/5c cong-grad-from-best** | One more cong-grad from `best_pl` + wide-from-best (frac=0.08) | Refine after multi-DP candidates settle |
| **7 DP-rescue** | Cong-grad chain from each DP candidate's basin | Mine DP's WL+density edge for proxy-friendly local minima |
| **8 TOP-K cong-grad** | Restrict perturbation to the K hottest macros only (K ∈ {5, 10, 20}, 3-iter chains) | Focus motion on routing peaks instead of spreading across all congested cells |
| **9 random-order legalize** | N=3 trials with randomized secondary-sort key in `_will_legalize` | Different legalization arrangements from the same starting positions |
| **Multi-seed 2-opt** | Proxy-driven 2-opt (k=20) from `best_pl` + each DP basin; true-proxy selection | A DP seed's basin can 2-opt to a deeper minimum than `best_pl`'s; pruning at `+0.02` skips unreachable seeds |
| **R2 interleave (≤6 rounds)** | Hard reloc ⇄ soft-cong reloc ⇄ soft-density reloc ⇄ 2-opt cleanup | The dominant lever — see § 2.3 |

### 4.2 Budget allocation (floor-reservation)

The placer is invoked 17 times by `evaluate --all` on the same instance, and
must keep total wall-clock under the harness's 3600s hard cap. Each call
computes `effective_budget_s` from the cumulative monotonic elapsed:

```python
remaining_total = HARNESS_TOTAL_BUDGET_S - cumulative_elapsed       # 3300s internal cap
remaining_benchmarks = max(1, HARNESS_TOTAL_BENCHMARKS - benchmarks_done)
# Reserve (floor + overrun) for every OTHER remaining benchmark + own overrun
reserve_others = (PER_BENCH_FLOOR_S + BUDGET_OVERRUN_S) * (remaining_benchmarks - 1)
this_cap = remaining_total - reserve_others - BUDGET_OVERRUN_S
effective_budget_s = min(time_budget_s, max(PER_BENCH_FLOOR_S, this_cap))
# Hard-cap safety: never push past the 3540s headroom under the 3600s harness cap
effective_budget_s = min(effective_budget_s, HARD_CAP_SAFE_S - cumulative_elapsed - BUDGET_OVERRUN_S)
```

Constants: `PER_BENCH_FLOOR_S=110`, `BUDGET_OVERRUN_S=60`,
`HARNESS_TOTAL_BUDGET_S=3300`, `HARD_CAP_SAFE_S=3540`.

Worst-case simulation (every benchmark overruns its soft budget by 60s):

```
b01    cum=     0    eff=200    actual=260
b02    cum=  260    eff=200    actual=260
b03    cum=  520    eff=200    actual=260
b04    cum=  780    eff=200    actual=260
b05    cum= 1040    eff=160    actual=220  ← transition
b06–b17  cum stepping by 170    eff=110    actual=170
final cum = 3300                            ← exact internal-cap landing
```

Even in the worst case, **every benchmark gets ≥110 s** and the total lands
exactly at the 3300 s internal cap — well under the 3600 s harness cap. The
pre-floor-reservation allocator (`adaptive_cap = remaining/remaining_benchmarks·0.9`
plus a blunt `cumulative > 95% × cap → baseline` guard) starved ibm18 in one
real `--all` run; the floor-reservation fix makes that structurally
impossible.

---

## 5. Algorithm explanations

### 5.1 Vectorized greedy spiral legalize (`_will_legalize`)

The base legalizer. Sort macros by largest area first; for each macro, search
expanding rings of cells around its starting position for the nearest
unoccupied legal slot. The naive scalar implementation is a Python loop over
ring radii and over candidates within each ring. The vectorized version
generates all `8r` candidates per ring at once via `_ring_offsets`, runs a
single `[K, P]` conflict matrix against placed macros, and picks the argmin
distance. ~12× faster than the scalar reference.

**Critical correctness detail** (one of the harder bugs we fixed): the
original scalar code computes `d² = (cx − pos[idx, 0])²` where `cx` is a
Python float and `pos[idx, 0]` is `float32`. NumPy demotes the Python float to
float32 for the subtraction (the Python-scalar-meets-numpy-scalar rule), so
ties between symmetric ring candidates are broken at float32 precision. The
vectorized version must cast its candidate arrays to `float32` **before** the
subtraction — otherwise the float64 ties pick a different ring direction, and
ibm04 (where the tie-break matters for the cong-grad trajectory) lands at
1.3364 instead of 1.3316.

### 5.2 Congestion-gradient perturbation (`_routing_congestion_perturb`)

The global move that escapes WL-trap local minima. For each macro in a cell
whose congestion exceeds a threshold, compute the finite-difference gradient
of the cell-congestion field at its position, and move it **against** the
gradient (toward lower-congestion neighbors). A small random component breaks
symmetry. Uses an isolated `RandomState(seed+1)` so the main numpy RNG state
isn't perturbed — noise restarts get identical draws regardless of cong-grad
participation.

The congestion field is `plc.get_horizontal/vertical_routing_congestion()` —
real H/V routing congestion after a `get_congestion_cost()` call. This
captures both the net routing demand and the hard-macro routing blockage.

Used in Phases 1/2/3/5b/5c/7/8 with different perturbation strengths
(`frac ∈ {0.04, 0.08, 0.12}`) and different top-K restrictions.

### 5.3 Vectorized routing fill (`_apply_3pin_routing_vec`, `_apply_h/v_strips_batch`)

The hot path of the incremental scorer's congestion update (cProfile shows
67% of a per-move trial). For a touched net, the routing demand is "filled"
into the H/V flats along L-shaped or steiner paths between the net's pins.

The vectorized fill:

1. **Classify** all touched nets by length (2-pin, 3-pin, ≥4-pin steiner).
2. **For each class**, compute all the L/steiner segments in one batched numpy
   expression. The 3-pin path is the trickiest (4 sub-cases depending on
   collinearity); we mirror the scalar reference exactly via two sorts
   (one by `(col, row)` for cases 1–3, one by `(row, col)` for the T-route
   case 4).
3. **Apply** the segments to the H/V flats via a **difference-array + cumsum
   trick**: for an H-strip on row `r` from column `lo` to `hi`, store
   `+weight` at `(r, lo)` and `−weight` at `(r, hi)` in a `(grid_row, grid_col+1)`
   events array, then `cumsum(events, axis=1)[:, :grid_col]` recovers the
   filled values. Multiple overlapping strips on the same row accumulate via
   `np.add.at` on the endpoints (unbuffered scatter for correctness on
   duplicates).

**Idea #1 optimization (subset-cumsum strip-batch):** the
`(grid_row, grid_col+1)` diff array is wasteful when only a few rows are
touched. `np.unique` the touched row indices and cumsum only those rows.
The cumsum is per-row independent, so the result is **bit-identical** to the
full version — duplicate accumulation order is preserved because `np.add.at`
processes entries in the original index order, and the unique remap doesn't
change which entries land in which compact row.

**Idea #2 optimization (topology-struct cache):** the gather indices, length
classification, and the ≥4-pin sink-index layout are placement-independent.
Split `_apply_net_routing_subset` into `_build_net_routing_struct` (cacheable
per macro) + `_apply_net_routing_struct` (does only the gcell extraction +
dispatch + fill given a precomputed struct). The scorer caches the struct
per module index; single-macro paths reuse it across every candidate target
(within a relocation pass) and across the −1 / +1 within each move.

### 5.4 Bit-exact incremental congestion cost (`_compute_cong_cost`)

Cost computation from the maintained H/V flats:

1. Normalize H/V routing flats by routes-per-cell.
2. Add the macro-routing blockage (Hm/Vm flats).
3. Smooth H/V via a 1D box filter (`_smooth_routing_cong_vec`, separable per
   axis).
4. Concatenate H_total + V_total, take the **top-5% of all cells**, return
   `mean(top)`.

Steps 3 + 4 cost ~17% of a per-move trial. The smoother is a fixed-kernel
convolution (`smooth_range=2`), and crucially, it's **separable**:

- H is smoothed *along rows-within-a-column* (each column independent).
- V is smoothed *along cols-within-a-row* (each row independent).

So changing the raw flats in a small region only affects a localized set of
smoothed values in the same row/column.

**The incremental cong cost:** cache `H_smoothed` and `V_smoothed` as
`(grid_row, grid_col)` 2D state. On a move:

- The routing apply (`_apply_net_routing_struct`) returns the bounding box
  of touched-net pin gcells.
- Re-smooth only the affected **columns** for H (since H smooths per column)
  and affected **rows** for V (per row), using the bbox.
- Crucially, **recompute from the raw flats**, not from accumulated deltas.
  Each value in the cache always equals exactly what a full re-smooth would
  produce → **no drift, no float-accumulation error**.
- `_compute_cong_cost` becomes: V_total = cached V_smoothed + Vm,
  H_total = cached H_smoothed + Hm, concat, top-5% partition. No re-smoothing.

The trial path (`score_swap`/`score_move`/`score_move_soft`) snapshots the
affected columns/rows of the smoothed cache before applying, and restores on
revert; the commit path persists the re-smoothed values. All six move paths
wire through a shared `_resmooth_bbox` helper.

This is the **bit-exact incremental cong cost** lever (1.2799 → 1.2767 in
isolation, fully verified). The "from raw, not deltas" detail is what makes
it bit-exact and drift-free — a delta-accumulating version would lose
~1e-16 / commit and slowly diverge.

### 5.5 Multi-seed proxy-driven 2-opt (`_two_opt_proxy_swap`)

Local search via macro pair-swaps, scored by the incremental proxy. Naive
2-opt is O(N²) candidates per iteration; we restrict to a **spatial kNN**
(`k=20` nearest macros) per macro, sorted by distance. Each candidate swap
is scored via `score_swap` (~1.4 ms incremental); accept on strict
true-proxy drop.

**S9 augmentations** (cong-aware candidate selection):

- **Hot-first ordering.** Sort the outer loop by descending local cong
  (`macro_cong[i]`), so the deadline-bound search spends its time on the
  routing peaks that dominate the proxy.
- **Cold-region teleport.** Spatial kNN can only swap nearby macros — a
  routing-heavy macro can never relocate across the chip via local swaps
  alone. For the 20 hottest macros, append the 8 *coldest* as extra
  candidates — a long-range edge that expands the reachable placement set.
  Size-incompatible teleports fail the free conflict check before scoring.

The proxy gate validates every swap, so candidate selection only changes
*which* swaps are tried, never *accepts* a worse placement.

**Multi-seed selection.** The final 2-opt runs from `best_pl` *plus each DP
candidate basin*, with true-proxy selection across seeds. A DP seed's basin
can yield a deeper 2-opt result even when its standalone score lost the
best_pl race (ibm04 hi-mov basin: 1.3210 standalone → 1.2797 after 2-opt).
A seed whose raw proxy is more than `DP_SEED_2OPT_WINDOW=0.02` above the
current best is pruned (max observed 2-opt gain ~0.04, so a +0.02 seed can't
catch up). This is provably score-neutral and cuts `--all` wall-clock from
~1198 s (no prune) to ~722 s.

Selection uses a fresh `_exact_proxy` on the final candidates, not the
incremental score — see O2 in `ISSUES.md` for the gotcha (the incremental WL
can drift across seeds when plc state mutates between scorers).

### 5.6 Relocation moves (`_relocation_moves`, `_soft_relocation_moves`)

The move type 2-opt **can't** make. 2-opt only *exchanges* two macros'
positions — it can never relocate a routing-heavy macro into an empty
low-congestion gap (a swap would dump some other macro into the vacated hot
spot). Relocation adds exactly that missing move:

For each of the hottest movable macros:

1. Read its current cell's local cong (or density) → `local_cong[i]`.
2. From the global pool of low-cong (or low-density) candidate cells, keep
   only those *strictly colder* than `local_cong[i]`.
3. Sort the candidates by Euclidean distance to the macro's current position
   (nearest first) — local moves cost less wirelength, more likely to pass
   the proxy gate.
4. For each candidate (up to `n_targets`), clip to keep in-bounds, then
   `score_move(i, target)`. Hard moves include an overlap check vs other
   hard macros; soft moves don't (softs may overlap).
5. Track the best target that strictly lowers the true incremental proxy.
   If found, `commit_move(i, best)` persists the move and `best_score`
   ratchets down.

**The hot/cold fields:**

| Move type | Field source | Reads from |
|---|---|---|
| Hard reloc | `max(H, V)` routing cong | `plc.get_*_routing_congestion()` |
| Soft cong (R3) | `max(H, V)` routing cong | `plc.get_*_routing_congestion()` |
| Soft density (R5) | occupancy `grid_occupied / dens_grid_area` | `incremental_scorer.grid_occupied` |

The soft passes use `top_hot=128` and `n_targets=24` (top ~6–14% of softs
per round on IBM). R5 boosts to `top_hot=192` on rounds 4–6 (where cong is
skipped by the A+C optimization) to spend the freed budget on more density
attempts. Hard relocation uses `top_hot=R2_HOT=48`, `n_targets=R2_TGT=16`.

**Mean-field coupling.** Softs don't relocate "against each other" pairwise
or "against hard macros" specifically. The fields are *aggregates* of all
contributions (hard + soft) on the shared grid, and each move is decided by
its effect on the global proxy. Hard macros are *fixed* during a soft pass;
they contribute to the field but don't move. Softs co-adapt to each other
*through the shared grid* — a later soft in the same pass sees the field
updated by earlier commits.

### 5.7 IncrementalScorer internals (B3 phases + P3)

The scorer is built in `IncrementalScorer.__init__` from the current
placement (~25–48 ms one-time per benchmark per pass). It maintains:

- `per_net_hpwl`, `total_wl_raw` — for B3p2 (per-net incremental WL).
- `H_flat`, `V_flat`, `H_macro_flat`, `V_macro_flat` — for B3p4
  (per-net incremental routing).
- `H_smoothed`, `V_smoothed` — 2D smoothed cache for incremental cong cost
  (§5.4).
- `grid_occupied` — for P3 (incremental density).
- `committed_hard_pos`, `committed_soft_pos` — running state.
- `_route_struct_cache: dict[module → struct]` — for idea #2 topology cache.
- `macro_to_nets` — gathered from `wl_cache`'s `ref_idx`/`pin_to_net` via
  vectorized stable-sort + boundary-partition (built once).

Move methods:

| Method | Inputs | What it does |
|---|---|---|
| `score_swap(i, xy_i, j, xy_j)` | Two hard macros + new positions | Trial-score the swap; snapshot/apply/compute/revert; bit-exact vs `_exact_proxy` |
| `commit_swap(i, xy_i, j, xy_j)` | Same | Persist the swap (no revert) |
| `score_move(i, xy)` | One hard macro + new position | Single-macro relocation trial |
| `commit_move(i, xy)` | Same | Persist |
| `score_move_soft(k, xy)` | One soft macro + new position | Soft trial (no macro-routing blockage) |
| `commit_move_soft(k, xy)` | Same | Persist |
| `hard_net_centroids()` | — | `[n_hard, 2]` WL-anchor per hard macro (mean of connected nets' centroids); kept inert after R4 was disproven |

### 5.8 Verification regime (the foundation)

The accept-on-true-proxy guarantee is only as good as the incremental
scorer's correctness. Every move-path is verified bit-exact against
`_exact_proxy`:

| Verifier | Path | Tolerance | Drift over commits |
|---|---|---|---|
| `_verify_incremental_scorer.py` | swap | Δ ≤ 4.4e-16 (machine eps) | 0 |
| `_verify_score_move.py` | hard move | Δ ≤ 1.8e-9 | stable |
| `_verify_score_move_soft.py` | soft move | Δ ≤ 5e-10 | stable |
| `_verify_subset_routing.py` | `_apply_net_routing_subset` vs full routing | bit-exact | — |
| `_verify_congestion.py` | `_compute_cong_cost` vs `plc.get_congestion_cost` | bit-exact | — |
| `_verify_density.py` | `_compute_density_cost` vs `plc.get_density_cost` | bit-exact | — |
| `_stress_verify.py` | Many sequential commits, observe drift | none over 1000s of moves | |

Every speedup added to the scoring path must pass these verifiers before
shipping. This is the discipline that lets us add five mutually compounding
speedups without ever introducing a regression.

---

## 6. DREAMPlace integration

DREAMPlace is the academic SOTA placer (leaderboard #1 on this benchmark set
at 1.4076 standalone). Its strength is WL + density via Nesterov-accelerated
analytical placement; its blind spot is that its objective doesn't see the
**TILOS proxy's top-5% congestion peaks**. The empirical decomposition
(DP_DIAG, env-gated logging in `place()`): on congestion-heavy benchmarks DP
loses entirely on congestion (Δ ~+0.064 on ibm10, ~+0.075 on ibm12) while
being *better* on WL and density.

We use DP as a **side-channel seed** — launched async at `place()` entry, its
result becomes one of several candidates the multi-seed 2-opt mines. Standalone,
DP wins 2/17 final-seed races; counting basins selected by the multi-seed
2-opt is somewhat higher.

### 6.1 Bridge architecture (`dreamplace_bridge/`)

```
.pb.txt netlist (TILOS format)
        │
        ▼
  pb_to_bookshelf.py     ─▶   .aux / .nodes / .nets / .scl / .pl / .wts
                                       (Bookshelf format)
                                              │
                                              ▼
                                    DREAMPlace (Nesterov NLP)
                                              │
                                              ▼
                                    legalized .pl output
                                              │
                                              ▼
                              bookshelf_to_pb.py
                                              │
                                              ▼
                                  macro centers [N, 2]
```

Three configurations are launched in parallel by
`launch_dreamplace_async`:

| Tag | `target_density` | `soft_macros_movable` | Purpose |
|-----|-----------------|----------------------|---------|
| **lo-fix** | 0.65 | False | Loose density target, softs anchored to `initial.plc` |
| **hi-mov** | 0.85 | True  | Tight density target, softs co-optimized with hards |
| **hi-fix** | 0.85 | False | Tight density target, softs anchored |

These three basins are diverse enough that the multi-seed 2-opt's
true-proxy selection regularly picks different winners across benchmarks
(O2 / S4).

The async launcher:

- `subprocess.Popen` with `start_new_session=True` so we can clean up the
  whole process group on timeout.
- Watchdog thread enforces `timeout_s` even when the placer is blocked in
  scoring (without this, a hung DP saturates CPU and slows our scoring
  100× — observed on ibm06 in early v15 debugging).
- `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`, `OPENBLAS_NUM_THREADS=2`,
  `NUMEXPR_NUM_THREADS=2` cap so DP's internal BLAS pools don't
  oversubscribe with the parent's scoring thread.
- `os.killpg()` for clean teardown if the placer needs to give up on a
  DP candidate.

### 6.2 DREAMPlace modifications

All recorded in `docs/DREAMPLACE_FIXES.md` (kept synchronized with the
gitignored `dreamplace_build/` and `dreamplace_src/` trees so the patches
can be reapplied on a fresh build). Summary:

**Input format fixes (`dreamplace_bridge/pb_to_bookshelf.py`):**

- **`.scl` row structure.** Originally emitted a single canvas-height row
  (`Height: 34081` for ibm04). DREAMPlace's density bins + macro legalizer
  need stdcell-row-height rows to function. The reference `simple.scl`
  benchmark uses 8 rows of 12 over a 96-tall canvas. We now emit
  `num_rows_target=8` rows of height `canvas_h/8` each. Without this fix,
  DP's Nesterov optimizer plateaus at iter=1 with `wHPWL` frozen at 5.31e7
  and iter times of 0.3 ms (vs the typical 50–500 ms when the optimizer is
  doing real work). After the fix, real iter-by-iter motion appears.

**Config flags (`run_bridge._default_dreamplace_config`):**

- **`macro_place_flag=1` + `use_bb=1`.** Engages DP's 2-stage BB-step → NLP
  pipeline for actual macro placement. Without these, DP treats macros as
  huge stdcells and the gradient step is effectively zero (~0.5 ms/iter wall
  time, way below the ~50 ms/iter needed for real per-cell gradient
  computation). With them, real `wHPWL` trajectories appear:
  5.56e7 → 5.10e7 → 5.22e7 over 150 iters with Overflow 0.20 → 0.40 (real
  convergence). Standalone DP proxy on ibm04 drops from 1.7714 → 1.5207.
- **`iter=300`.** Standard `iter=150` is under-converged (Overflow stuck at
  0.4 vs target 0.10). Bumped to 300 → ibm04 standalone DP = **1.3196**
  (vs Phase 3's 1.3316 — beats it by 0.012). Larger values (500–1000)
  trigger DensityWeight runaway (Obj jumps to 1e12) with no proxy
  improvement. 300 is the sweet spot.
- **`soft_macros_movable=False`** (for lo-fix / hi-fix). Movable softs
  inflate congestion +0.011 on ibm04 (measured 2026-05-20). The hi-mov
  variant tests the alternative; multi-seed 2-opt picks per benchmark.
- **`routability_opt_flag=0`** (DP1: closed). DP's built-in `routability_opt`
  uses RUDY/RISA congestion to inflate cell areas in routing hotspots —
  but RUDY ≠ TILOS proxy congestion, so across a 64× capacity sweep + grid-
  matched route bins, routopt was either a no-op or a regression on every
  config tested (`test/dreamplace/_routopt_poc.py`, `_routopt_calib.py`).
  The bridge knob is wired but defaulted off.

**DREAMPlace source patches** (`dreamplace_src/dreamplace/PlaceObj.py`,
mirrored into `dreamplace_build/install/dreamplace/PlaceObj.py`):

- **NCTUgr-map guard.** `PlaceObj.build_nctugr_congestion_map` requires
  per-layer `unit_horizontal_capacities`, which is None for Bookshelf
  inputs (it's an LEF/DEF concept). Patched to only build the NCTUgr map
  when `adjust_nctugr_area_flag` is set — RUDY (the default) is the path
  that runs on our inputs. Without this guard, enabling `routability_opt`
  crashes with a NoneType error. This is a genuine bug fix that we kept
  in case `routability_opt` becomes useful with a different upstream
  capacity model.

**NumPy 2.0 compat:**

- `np.string_` → `np.bytes_` in `install/dreamplace/PlaceDB.py`
  (`sed -i 's/np\.string_/np.bytes_/g' install/dreamplace/PlaceDB.py`).

### 6.3 Why DP alone isn't enough

DP_DIAG ran on the congestion-heavy benchmarks shows the standalone DP
basins lose to cong-grad-best **entirely on congestion**:

| | wl | den | cong | proxy |
|---|---|---|---|---|
| ibm10 raw `dp[hi-fix]` | 0.0574 | 0.3774 | **0.9543** | 1.3891 |
| ibm10 final best | 0.0636 | 0.3804 | **0.8904** | 1.3344 |
| Δ (dp − best) | −0.006 | −0.003 | **+0.064** | +0.055 |
| ibm12 raw `dp[hi-fix]` | 0.0626 | 0.3968 | **1.2497** | 1.7090 |
| ibm12 final best | 0.0608 | 0.4017 | **1.1749** | 1.6375 |
| Δ (dp − best) | +0.002 | −0.005 | **+0.075** | +0.071 |

DP is *better* on WL and density and only loses on the term it can't see.
Post-hoc cong-grad from the DP basin recovers some of the gap (DP_PROBE
confirmed ibm10 can reach 1.3279 from DP's basin), but in-pipeline it's
budget-hungry, high-variance, and not reproducible at fixed seed (Phase 7b
was REVERTED). The shipped path is to let DP keep its own basin and have
the multi-seed 2-opt pick when DP's trajectory yields a better local min.

---

## 7. References

- [`PROGRESS.md`](./PROGRESS.md) — per-benchmark numbers, full experiment
  history, the v1 → v2 progression.
- [`ISSUES.md`](./ISSUES.md) — open issues, closed dead-ends, resolved bugs.
- [`DREAMPLACE_FIXES.md`](./DREAMPLACE_FIXES.md) — full inventory of
  DREAMPlace patches (kept in sync so the gitignored `dreamplace_build/`
  and `dreamplace_src/` trees can be rebuilt).
- [`../README.md`](../README.md) — top-level overview and reproduction
  commands.
- `../test/verification/` — bit-exactness verifiers (the foundation of
  the non-regression guarantee).
- `../test/diagnostic/` — profiles + leverage analyses that produced and
  constrained the design (`_profile_init.py`, `_profile_move.py`,
  `_profile_move_internals.py`, `_profile_move_realistic.py`,
  `_reloc_leverage.py`, `_term_breakdown.py`, …).
