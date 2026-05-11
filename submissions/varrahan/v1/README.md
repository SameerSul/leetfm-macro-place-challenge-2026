# v1 — Varrahan's Submission

Active placer for the Partcl/HRT Macro Placement Challenge. Built on top of `sameer_v1` with one targeted change.

## What's different from `sameer_v1`

Structurally `placer.py` is **almost identical** to `sameer_v1/placer.py` — same legalization, restart schedule, congestion-gradient iterative loop, wide-step trick. Three concrete code changes:

| Change | sameer_v1 | v1 | Effect |
|---|---|---|---|
| `EXACT_MACRO_THRESHOLD` | 340 | **400** | Re-includes ibm11 (n=373), ibm15 (n=393) in exact pipeline |
| `EXACT_GRID_CELL_LIMIT` | 2000 | **2200** | Re-includes ibm15 (grid=2166), ibm18 (grid=2145) |
| `BUDGET_OVERRUN_S` | (n/a) | **60.0s** | Allows directed-restart phases (cong-grad Phase 1/2/3) up to 60s overrun beyond `time_budget_s`. Noise loop stays strict. |

Plus minor pattern change: cong-grad call sites switched from `if not _try_restart(...): return best_pl` (terminate placer) to `if not _try_restart(...): break` (let later phases try). Without this, a budget-failure in Phase 1 was killing Phase 2/3.

### Why the threshold change

Re-measured per-benchmark scoring time on clean CPU (2026-05-08):

| Benchmark | PROGRESS.md estimate | Actually measured |
|---|---|---|
| ibm11 | 75–263s | **17.7s** |
| ibm15 | 160s | **42.8s** |
| ibm18 | 220s | **61.7s** |

PROGRESS.md numbers were 4–13× too high (likely measured under heavy CPU load). All three new inclusions are well under the existing `SLOW_SCORE_THRESHOLD_S=100s` safety guard.

### Why the budget fix

In --all run 1 (2026-05-10), ibm04's cong-grad iter=1 spiked from 7s typical to 200s, putting total time at 209s — over the 200s budget — and the post-scoring guard fired. The placer then `return best_pl`'d immediately, halting Phase 2/3 and returning iter=1's 1.3882 instead of Phase 3's 1.3316. ibm04 collapsed by +0.0566 → +0.0033 to the avg.

Adding `BUDGET_OVERRUN_S=60s` for directed phases gives the productive cong-grad pipeline a 30% slack to absorb transient spikes. Confirmed reproducible: ibm04 holds at 1.3316 across multiple --all runs.

## Result vs `sameer_v1` / v11

| Benchmark | v11 (PROGRESS.md) | v12 (--all stable) | Δ | Notes |
|---|---|---|---|---|
| ibm01 | 1.1854 | 1.1860 | +0.0006 | v11 was lucky outlier |
| ibm02 | 1.5823 | 1.5923 | +0.0100 | v11 was lucky outlier (stale-plc lottery) |
| ibm03 | 1.3547 | 1.3603 | +0.0056 | v11 was lucky outlier |
| **ibm04** | 1.3390 | **1.3316** | **−0.0074** | clean rediscovery + budget fix makes it stable |
| **ibm06** | 1.6797 | **1.6684** | **−0.0113** | clean CPU rediscovery |
| **ibm07** | 1.4950 | **1.4924** | **−0.0026** | 1% noise restart wins |
| ibm08 | 1.5251 | 1.5251 | 0.0000 | |
| ibm09 | 1.1304 | 1.1304 | 0.0000 | |
| ibm10–17 large/baseline-only | unchanged | unchanged | 0.0000 | |
| **ibm18** | 1.7941 | **1.7896** | **−0.0045** | threshold change re-includes; cong-grad iter=1 wins |

**17-benchmark --all average:**

| Run | Avg | Notes |
|---|---|---|
| PROGRESS.md v11 estimate | 1.4860 | composite, never actually --all'd |
| PROGRESS.md v10b actual --all | 1.4877 | 2026-04-30 historical |
| **v12 --all stable (with budget fix)** | **1.4854** | reproducible across runs |

**Net delta vs v11 estimate: −0.0006 to avg.** Wins (−0.0258 across ibm04/06/07/18) partially offset by clean-CPU regressions vs v11 outliers (+0.0162 across ibm01/02/03).

The wins that are genuinely v1-attributable: ibm18 (threshold change). The others (ibm04, ibm06, ibm07) are sameer_v1-era results that emerge naturally on clean CPU; v12's budget fix makes ibm04 in particular reliably reachable under --all conditions.

**Gap to RePlAce (1.4578):** −2.2% / 0.028. Not closeable by further tuning of this restart pipeline — needs a structural change (DREAMPlace bridge or similar).

## File index

| File | Purpose | Carry to v2? |
|---|---|---|
| `placer.py` | **The submission.** Active placer entry point. Same code as `sameer_v1/placer.py` except for the two threshold constants above. | **Yes** — v2 builds on this directly. |
| `surrogate.py` | WL-only surrogate ranker for large benchmarks. Designed to let SA-style search work on benchmarks where exact scoring is too slow. Spearman +0.83/+0.94 vs real proxy on ibm11/ibm15, but ties near the optimum — picks the wrong tied candidate when several score identically in the surrogate. **Net effect: zero / slightly negative.** Never wired into `placer.py`. | **No** — failed experiment. Archive value only. v2 should not import it. |
| `_calibration_test.py` | Calibration harness that drove `surrogate.py`'s WL-only weighting decision and proved (on ibm11+ibm15) that the surrogate's tie-breaking is too coarse to reliably outpick the baseline. Documents *why* the surrogate was rejected. | **No** — one-off experiment. |
| `_path3_incremental_test.py` | Tested whether `plc.set_use_incremental_cost(True)` would enable fast SA against the real evaluator. **Found that incremental mode freezes the density and congestion components — only the WL term refreshes, and WL is anti-correlated with proxy.** Path 3 confirmed dead. | **No** — one-off experiment, documents the dead-end. |
| `_ibm15_timing_test.py` | Reusable scoring-time harness — measures legalization + 3 scoring calls for any benchmark. Used 2026-05-08 to discover the stale PROGRESS.md estimates that motivated the threshold change. | **Yes** — useful any time we want to re-verify scoring assumptions or stress-test new threshold ideas. |
| `PROGRESS.md` | Local copy of the team's `/PROGRESS.md`, extended with v12+v13 entries (varrahan/v1's findings: threshold change, ibm04 1.3316 rediscovery, ibm07 1% noise win, surrogate/Path 3/Phase 4/WireMask/DREAMPlace dead-ends). The team copy at the repo root is read-only for this submission slot. | **Yes** — keep updating as iterations progress; merge back to team copy when permitted. |
| `dreamplace_bridge/` | DREAMPlace bridge module: `pb_to_bookshelf.py` (forward), `bookshelf_to_pb.py` (back), `run_bridge.py` (end-to-end runner via subprocess). Working code that successfully runs DREAMPlace on TILOS benchmarks. **Not wired into `placer.py`** — integration was tested 2026-05-11 and reverted (v13 --all = 1.4897 vs v12 = 1.4854, +0.0043 worse). Two real wins (ibm04 −0.0075, ibm11 −0.0019) overshadowed by 10-15s subprocess overhead displacing productive restarts on 7 other benchmarks. See PROGRESS.md "Key Findings" for salvage paths. | **Maybe** — if v2 explores async DREAMPlace, conditional invocation, or a "Phase 5: cong-grad-from-dreamplace", these files are the foundation. Otherwise can be deleted. |
| `dreamplace_build/` | Local DREAMPlace install (~500MB build artifacts, gitignored). Built with `_GLIBCXX_USE_CXX11_ABI=1` to match torch wheel. Required if you want to use the `dreamplace_bridge/` module. Rebuild with: `sudo apt install -y flex bison libboost-all-dev && uv pip install scipy shapely cairocffi torch_optimizer ncg_optimizer pyunpack patool pkgconfig`, clone DREAMPlace, `cmake .. -DCMAKE_CXX_ABI=1 -DPython_EXECUTABLE=$(which python)`, `make -j2` (avoid -j$(nproc), causes OOM), `make install`, then `sed -i 's/np\\.string_/np.bytes_/g' install/dreamplace/PlaceDB.py` for NumPy 2.0 compat. | **No** — gitignored. Reproduce via the steps above (~75min). |

## v2 starting point

When starting v2:

1. Copy `placer.py`, `_ibm15_timing_test.py`, and `PROGRESS.md`.
2. Skip `surrogate.py`, `_calibration_test.py`, `_path3_incremental_test.py` — they document negative results, not reusable building blocks.
3. Treat v1's thresholds (`EXACT_MACRO_THRESHOLD=400`, `EXACT_GRID_CELL_LIMIT=2200`) and `BUDGET_OVERRUN_S=60.0` as the new floor. Don't drop them without re-measuring scoring time / re-validating ibm04 stability.
4. The remaining gap to RePlAce (~0.028) cannot be closed by tuning the existing restart pipeline — see the next-steps notes in `PROGRESS.md` "When in doubt" / "Next Experiments". Highest-leverage open items: (a) DREAMPlace bridge (multi-day, only known structural answer), (b) larger cong-grad fracs (0.10, 0.15) on high-cong benchmarks (cheap, speculative).
