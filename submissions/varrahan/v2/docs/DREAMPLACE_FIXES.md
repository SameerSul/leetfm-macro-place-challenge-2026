# DREAMPlace Integration — Required Fixes

## STATUS (2026-05-21): SUPERSEDED — see PROGRESS.md v15 section for outcome

**Bottom line:** the diagnosis in this doc was *wrong about the cause*. The true root cause of DREAMPlace producing junk output was a broken Bookshelf conversion (single-row `.scl` + `macro_place_flag=0` + under-converged 150 iters), not a soft-macro / density-tuning issue. With the corrected bridge, standalone DP proxy on ibm04 dropped from **1.7714 → 1.3196** (winning as Phase 5 additive). Fix 1 (soft movable) was actually a NEUTRAL/NEGATIVE change once the bridge worked (softs movable inflates congestion); Fix 2 (density sweep) was a no-op on the broken NLP and remained marginal even after the bridge worked.

**What was actually needed (see PROGRESS.md v15 → "Bridge architecture fix"):**
1. **`.scl`** → 8 rows of `canvas_h/8` instead of 1 canvas-height row.
2. **`run_bridge._default_dreamplace_config`** → `macro_place_flag=1`, `use_bb=1`.
3. **Iterations** → 300 (was 150 — under-converged).
4. **Subprocess env** → `OMP_NUM_THREADS=2` etc. to prevent CPU oversubscription with the parent scoring thread (caused 100× slowdown on ibm06 under --all).
5. **Watchdog thread** in `AsyncDreamplaceHandle` to enforce `timeout_s` when the placer is blocked in scoring.

**What was tried from this doc's Fix 3 and rejected:**
- "DP as PRIMARY baseline_pos" (replace `_will_legalize(initial.plc)` with legalized DP output) regressed ibm06 1.6684 → 1.6789 (+0.011). Phase 3 cong-grad from DP's placement converges to a different (worse) basin than from initial.plc.
- Additive Phase 6 cong-grad-from-DP regressed ibm08 by +0.017 (budget displacement).

**Net result:** --all avg 1.4854 → 1.4804 (−0.0050). Wins from this work: ibm01 (−0.044), ibm04 (−0.012), ibm10 (−0.037, via separate Improvement #1 on n>400 benchmarks), ibm14 (−0.003).

---

# (Below is the original 2026-05-20 plan, kept as historical record)

This document specifies the changes needed to make the DREAMPlace bridge actually improve our placer score. As of 2026-05-20, the bridge is **built, integrated, and async-launched, but yields zero improvement** because of three compounding issues. Each fix below is independently necessary; **Fix 1 is critical** (without it, the others can't help).

## Repository state (what you're working with)

- **`submissions/varrahan/v1/placer.py`** — main placer (894 lines). Async DREAMPlace launch is at `place()` start; Phase 5 check after Phase 3. Look for the comment blocks: `Async DREAMPlace launch` and `Async DREAMPlace check`.
- **`submissions/varrahan/v1/dreamplace_bridge/`** — 4 files:
  - `run_bridge.py` — `launch_dreamplace_async`, `AsyncDreamplaceHandle`, `_default_dreamplace_config`.
  - `pb_to_bookshelf.py` — TILOS → Bookshelf forward converter. `convert(soft_macros_movable=True)` works correctly (verified 2026-05-20: produces 287 terminals = ports only, 1380 movable).
  - `bookshelf_to_pb.py` — Bookshelf → TILOS back-converter. `read_dreamplace_positions_full()` (added 2026-05-20) returns BOTH hard and soft positions.
  - `__init__.py` — trivial.
- **`submissions/varrahan/dreamplace_build/install/`** — built DREAMPlace (~75-min build, NumPy 2.0 patch applied). DREAMPlace is fully functional.

**Important:** if any code change "doesn't take effect", delete `submissions/varrahan/v1/dreamplace_bridge/__pycache__/` first. Stale `.pyc` cache caused the initial soft_macros_movable bug to silently persist after the code was already correct.

---

## Current evidence of the problem

Standalone DREAMPlace on ibm04 (measured 2026-05-20):

| Variant | DP standalone proxy | Final placer proxy | Notes |
|---|---|---|---|
| Baseline (no DP) | n/a | **1.3316** | Phase 3 wins |
| DP with `soft_macros_movable=False` (broken — stale `.pyc` made it ineffective) | 1.6455 | 1.3316 | Phase 3 unchanged |
| DP with `soft_macros_movable=True` and Fix 2 NOT applied | **1.7790** ⚠️ | 1.3316 | **WORSE** — confirms Fix 2 is necessary |
| DP with Fix 1 + Fix 2 (tuned density) | **predicted ~1.3-1.4** | **predicted ≤ 1.32** | TBD — needs Fix 2 |

**Key empirical finding (2026-05-20):** Enabling soft macros movable WITHOUT tuning density made the standalone DREAMPlace proxy *worse* (1.6455 → 1.7790). This is because DREAMPlace with default config (`target_density=1.0, density_weight=8e-5`) does pure HPWL minimization. With softs movable, it clusters them maximally → density and congestion both blow up.

**The conclusion: Fix 1 ALONE is a regression. Fix 1 + Fix 2 are an inseparable pair.** Do not apply Fix 1 without Fix 2.

Compare: UT Austin's DREAMPlace-based leaderboard entry achieves **1.4076 average** on ICCAD04. Our gap: 0.078 to that ceiling.

---

## Fix 1 — Soft macros movable + read back (CRITICAL, partially done)

### Status: PARTIALLY APPLIED. Verify and complete.

### What's already done (verify before re-doing):
- `bookshelf_to_pb.py` has `read_dreamplace_positions_full()` returning `(hard_pos, soft_pos)`.
- `run_bridge.AsyncDreamplaceHandle.wait_for_result_full()` returns the full tuple; legacy `wait_for_result()` returns hard only.
- `placer.py` Phase 5 check calls `wait_for_result_full`, applies both hard and soft positions to a fresh `dp_pl` tensor, then scores directly (no `_try_restart`).
- `placer.py` passes `soft_macros_movable=True` to `launch_dreamplace_async`.

### What needs to be verified:
1. **Run on ibm04 standalone** (`uv run evaluate submissions/varrahan/v1/placer.py -b ibm04`). Look for log line `DREAMPlace launched async (soft-movable, ...)` and `Candidate N (dreamplace hard+soft): proxy=...`. The proxy should be ≤ 1.6 ideally; if it's still ≥ 1.5, Fix 2 (density tuning) is what's actually needed — not Fix 1's fault.
2. **Run on ibm11**. v13 (PROGRESS.md) reported wins here. Look for whether the DREAMPlace candidate beats baseline 1.2354.
3. **Inspect the `.nodes` file** at `/tmp/dreamplace_v1/<benchmark>/<benchmark>.nodes`. The line `NumTerminals : X` should equal the **port count** (~287 for ibm04). If it's >1000, the soft_macros_movable flag isn't propagating — clear `__pycache__/` and retry.

### What might still be broken:
- The soft positions DREAMPlace returns may have **overlaps** between soft cells (DREAMPlace's NLP doesn't enforce strict non-overlap for movable cells when `enable_legalize=0` and `legalize_flag=0` in the config). Our scoring should still work — the proxy doesn't enforce soft overlap — but verify scoring doesn't crash.
- The mapping from Bookshelf node names → TILOS soft indices uses `_sanitize(node.get_name())`. If any soft macro has a sanitized-name collision with another (suffixed `_1`, `_2`), the back-converter might mismap. Add an assertion or log warning if `len(plc.soft_macro_indices)` and the number of soft positions read don't match.

### Smoke test for Fix 1 success:

```bash
# Should show ~287 terminals (just ports), ~1380 movable (hard + soft)
rm -rf /tmp/dreamplace_v1 submissions/varrahan/v1/dreamplace_bridge/__pycache__
uv run python -c "
import sys
sys.path.insert(0, 'submissions/varrahan/v1')
from dreamplace_bridge.pb_to_bookshelf import convert
convert('external/MacroPlacement/Testcases/ICCAD04/ibm04', '/tmp/dp_test', soft_macros_movable=True)
"
grep "NumTerminals" /tmp/dp_test/ibm04.nodes
# Expected: NumTerminals : 287
```

---

## Fix 2 — Tune DREAMPlace density parameters (CRITICAL for proxy)

### Status: NOT DONE. Highest expected gain.

### Current config (in `run_bridge._default_dreamplace_config`, line ~85):

```python
"target_density": 1.0,      # 100% canvas utilization allowed = ZERO spreading penalty
"density_weight": 8e-5,     # density weight 80,000× weaker than HPWL term
"wirelength": "weighted_average",  # pure HPWL objective
"gamma": 4.0,
```

### The problem

`target_density=1.0` tells DREAMPlace "the canvas can be 100% full" — no spreading required. `density_weight=8e-5` is essentially zero compared to the HPWL term (which has effective weights ~1-10). So DREAMPlace minimizes **HPWL almost in isolation**.

For our congestion-dominated proxy (proxy = 1×WL + 0.5×density + 0.5×congestion, with congestion ~1.3-2.5 and WL ~0.06), this is the *opposite* of what helps. CLAUDE.md explicitly warns: "Optimizing for wirelength alone reliably makes proxy cost worse because clustering connected macros spikes density and congestion." DREAMPlace with current params does exactly that.

### Recommended change

```python
"target_density": 0.75,     # 75% utilization → forces ~25% empty space → less congestion
"density_weight": 5e-3,     # ~60× stronger; comparable to HPWL term
"gamma": 4.0,               # leave as-is
```

### Tuning protocol

`target_density` and `density_weight` interact — sweep both:

```
target_density ∈ {0.6, 0.7, 0.8, 0.85, 0.9, 1.0}
density_weight ∈ {8e-5, 5e-4, 1e-3, 5e-3, 1e-2}
```

That's 30 combinations × 1 benchmark = 30 DREAMPlace runs (~5 minutes total at ~10s each, sequential). Score each on ibm04 + ibm11 (the two v13-reported wins). Pick the combo with lowest **standalone proxy** on both.

### Expected outcome

If tuned correctly, standalone DREAMPlace proxy should drop from ~1.6 to ~1.3-1.4 on ibm04. That's the gap that lets the additive candidate actually beat Phase 3.

### Where to change

Just edit the dict in `run_bridge._default_dreamplace_config` (around line 74-108). All callers use this default.

---

## Fix 3 — Use DREAMPlace as PRIMARY, not additive (optional, largest upside)

### Status: NOT DONE. Most invasive change.

### Background

UT Austin's leaderboard entry at 1.4076 almost certainly uses DREAMPlace as the **main pipeline**:
1. DREAMPlace global placement (hard + soft, tuned density).
2. DREAMPlace's built-in legalization (set `legalize_flag=1` in the config).
3. Return.

That's it. No multi-restart, no cong-grad. The whole thing is one DREAMPlace call.

We currently use DREAMPlace as **one additive candidate** competing against a 20+ candidate restart pipeline. The DREAMPlace structure advantage gets washed out:
- We legalize DP's output ourselves with greedy spiral (which fragments DP's careful spread).
- We compare against Phase 3's 7-iter cong-grad which is already tuned to find good local minima.

### Recommended approach

After Fix 1 + Fix 2 produce a competitive DREAMPlace standalone proxy (say <1.4 on ibm04), restructure `place()` to:

1. **Launch DREAMPlace synchronously at place() entry**. Wait for it (it's the BASELINE now, not an additive).
2. **If DREAMPlace succeeds**: use its output as `baseline_pos`. Skip our `_will_legalize` call entirely (or use it only to fix residual overlaps DP didn't legalize).
3. **Phase 1/2/3 cong-grad runs FROM DREAMPlace's placement** instead of from `initial.plc` legalization. This is the "salvage path (c)" PROGRESS.md noted.
4. **If DREAMPlace fails or times out** (>60s on a single benchmark): fall back to current `_will_legalize` baseline path.

### Config changes for Fix 3

In `_default_dreamplace_config`:
```python
"legalize_flag": 1,         # use DP's own legalize (currently 0)
"detailed_place_flag": 0,   # keep off — we don't need detailed placement
"abacus_legalize_flag": 1,  # already set; uses Abacus row-based legalize
```

### Why this is risky

- **We lose ibm04 Phase 3's 1.3316 win** if DP-based baseline doesn't reach that level. PROGRESS.md notes Phase 3 is highly tuned around the initial.plc spread.
- **Bookshelf legalization differs from greedy spiral**. The placement may not satisfy TILOS's exact non-overlap rule for hard macros (TILOS uses a different overlap threshold than Bookshelf).
- **Async benefit is lost**: this is the sync v13 pattern with DREAMPlace overhead on the critical path. The benefit must come from DREAMPlace being a better placement, not from parallelism.

### When to do Fix 3

After Fix 1 + Fix 2 are verified:
- If standalone DREAMPlace proxy < Phase 3's win on ≥3 benchmarks → Fix 3 is worth it.
- If standalone DREAMPlace proxy is consistently worse than Phase 3 → skip Fix 3, accept Fix 1+2's modest gain.

---

## Verification protocol

After each fix:

1. **Smoke test** the bridge code in isolation (avoids subprocess overhead):
   ```bash
   rm -rf /tmp/dreamplace_v1 submissions/varrahan/v1/dreamplace_bridge/__pycache__
   uv run python -c "
   import sys, time
   sys.path.insert(0, 'submissions/varrahan/v1')
   from dreamplace_bridge.run_bridge import run_dreamplace
   t0 = time.perf_counter()
   pos = run_dreamplace('external/MacroPlacement/Testcases/ICCAD04/ibm04',
                        iterations=150, num_threads=2,
                        soft_macros_movable=True)
   print(f'time={time.perf_counter()-t0:.1f}s, pos.shape={pos.shape}')
   "
   ```

2. **Standalone benchmark test** on ibm04 (the v13 win candidate):
   ```bash
   timeout 280 uv run evaluate submissions/varrahan/v1/placer.py -b ibm04
   ```
   Look for the `Candidate N (dreamplace hard+soft): proxy=X.XXXX` line. Target: X.XXXX < 1.4101 (baseline). Ideal: X.XXXX < 1.3316 (Phase 3 win).

3. **Standalone tests** on ibm01, ibm06, ibm07, ibm08, ibm09, ibm11. Same target: DREAMPlace candidate should beat baseline on most.

4. **--all run** (only if standalone tests show consistent wins):
   ```bash
   timeout 3600 uv run evaluate submissions/varrahan/v1/placer.py --all
   ```
   v12 baseline: avg 1.4854. Target: ≤ 1.470 (any improvement is meaningful given the failures we've already tried).

### Recovery if regression observed

The placer has a `dp_handle.kill()` fallback if DREAMPlace fails. The integration is wrapped in `try/except` so a DREAMPlace failure logs and skips, never crashes the placer. Worst case: revert by setting `dp_handle = None` unconditionally near `place()` entry, which puts us back to v14 (current pre-DREAMPlace state).

---

## Quick reference — DREAMPlace config knobs

Documented in `run_bridge._default_dreamplace_config`. The most impactful knobs:

| Knob | Current | Tuned for proxy | Meaning |
|---|---|---|---|
| `target_density` | 1.0 | **0.75** | Max allowed density per bin (0-1). Lower = more spread, less congestion. |
| `density_weight` | 8e-5 | **5e-3** | Density penalty weight in the loss function. Higher = stronger spreading force. |
| `wirelength` | weighted_average | leave | HPWL approximation. Other options: log_sum_exp, sigmoid. |
| `gamma` | 4.0 | leave | Softness of WL approximation. Higher = sharper but noisier gradient. |
| `iteration` (in global_place_stages) | 150 | leave-or-200 | Nesterov optimization iterations. More = better convergence, slower. |
| `stop_overflow` | 0.10 | leave | Early-stop threshold for density overflow. |
| `random_center_init_flag` | 0 | leave | 0 = warm-start from initial.plc. 1 = cold-start from center. |
| `enable_fillers` | 0 | maybe try 1 | Fillers create artificial cells to enforce density. 1 = enable. |
| `legalize_flag` | 0 | **1 for Fix 3** | Run DREAMPlace's built-in legalize after global. |
| `macro_place_flag` | 0 | leave | Macro-specific placement stage (different algorithm). |

---

## Summary of fix priority

| Fix | Expected gain on --all | Effort | Risk |
|---|---|---|---|
| **Fix 1 + Fix 2 (together)** (soft movable + read back + tuned density) | **−0.005 to −0.020** to avg | 2-4h total | Low-Medium |
| **Fix 3** (DP as primary, after Fix 1+2 verified) | **−0.010 to −0.030** to avg additional | 2-4h | Medium (risk losing ibm04 1.3316) |

**Fix 1 and Fix 2 are INSEPARABLE.** Empirical evidence (2026-05-20): Fix 1 alone made standalone DP proxy go from 1.6455 → 1.7790 (regression). Without density tuning, enabling movable softs makes DREAMPlace cluster everything.

Recommended order:
1. **Apply Fix 1 + Fix 2 together.** Don't bother testing Fix 1 in isolation.
2. **Sweep density params** (Fix 2 protocol). 30-combo grid on ibm04 + ibm11; ~5 minutes total.
3. **Measure standalone DP proxy** on ibm04, ibm11, ibm06, ibm08, ibm09. Compare against each benchmark's baseline.
4. **If standalone DP proxy ≥ Phase 3 win on most benchmarks**: stop. Async-additive integration is the best we can do. Gain will be small (maybe −0.005).
5. **If standalone DP proxy < Phase 3 win on ≥3 benchmarks**: apply Fix 3 (DP as primary). This is where the −0.020+ gain materializes.

If only one fix is feasible: **NEITHER alone works**. Skip the integration entirely and revert to v14 (current pre-DP state). The cleanup is one line: set `dp_handle = None` unconditionally in `place()`.
