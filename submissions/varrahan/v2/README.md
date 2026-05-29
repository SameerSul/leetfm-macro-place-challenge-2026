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

## The two things that make v2 ≫ v1 (1.4854 → 1.2799)

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
- **R3b / R5 — soft DENSITY relocation (the dominant win).** Softs are the bulk of
  the *density* term too (and may overlap, so the cong pass can pile them). A
  second soft pass targeting the **density** field (`use_density`) finds moves the
  cong pass can't (`DENS_SOFT_PROBE`: cong-converged best_pl still yields 22–68
  density moves). Interleaved (hard ⇄ soft-cong ⇄ soft-density ⇄ 2-opt) + widened
  candidates (top_hot 128): **`1.3764 → 1.2799`**, all 17 improved (ibm13/02/08
  −0.122, ibm18 −0.21). Beats the leaderboard by 9.1%.

All moves are accept-on-true-proxy, so the whole local search is **strictly
non-regressing by construction**. (Open: budget margin under contention — R5 fits
at 2639s clean but ibm09 hit 307s; a speedup pass is queued. See ISSUES.md.)

Disproven: **R4** (WL-aware hard-relocation targeting toward net centroids) —
slightly worse, reverted; scaffolding kept inert.

**Leverage** (`test/diagnostic/_reloc_leverage.py`): per-benchmark gain is driven
by **hard-macro utilization × congestion headroom** — relocation helps where hard
macros occupy enough canvas to drive congestion (ibm04/10/02/12, util 0.42–0.60)
AND there's congestion above the floor. Low-hard-util benchmarks (ibm17/18) are
soft/net-dominated and barely move → soft-macro relocation is the flagged next
lever.

## Closed dead-ends (don't re-run without a specific reason — see ISSUES.md)

| Direction | Outcome |
|---|---|
| **DP1** congestion-aware DREAMPlace (`routability_opt`) | CLOSED — DREAMPlace's RUDY congestion ≠ TILOS proxy; no-op or worse across a 64× capacity sweep. (Required a real bug-fix to even run: NCTUgr-map guard, see DREAMPLACE_FIXES.md.) |
| **Phase 7b** post-hoc DP-basin repair | REVERTED — recoverable in a probe but budget-hungry, high-variance, not reproducible at fixed seed. |
| **S1** basin-hopping 2-opt (cong-grad kick) | DISPROVEN — slicing the budget starves the deadline-bound search; 6/7 worse. |
| **O3** soft-macro repositioning (bulk/gradient) | CLOSED for bulk methods (R1-style discrete *soft* relocation is the open follow-up). |

## File / docs index

| Path | Purpose |
|---|---|
| `placer.py` | **The submission** (~4500 lines). Pipeline above + `IncrementalScorer` + `_two_opt_proxy_swap` + `_relocation_moves`. |
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
