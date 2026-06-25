# Iteration Progress Log

All scores are proxy cost (lower is better).
Target: beat RePlAce avg of 1.4578.

> Only the first status entry is current production state; all later entries are
> historical experiment records.

> **Status (2026-06-25 — graph-tension ordering plus decompression survivor):**
> Added a hierarchy graph-tension signal that compares current cluster-centroid
> relations with the selected hierarchy seed, weights stretched inter-cluster
> edges by edge weight and congestion along the edge corridor, and normalizes the
> result into per-cluster ordering scores. The signal is default-on only for
> large hard-macro designs (`HIER_GRAPH_TENSION_HARD_MIN=600`) and currently
> orders decompression/coldspot opportunities; direct swap ordering remains
> default-off (`HIER_GRAPH_TENSION_SWAP_WEIGHT=0.0`) because focused tests showed
> it burned exact-score budget on weaker swaps. The default-on decompression
> graph survivor exact-scores a tiny hard/soft local-polish pool for legal,
> hierarchy-safe, graph-favorable decompression near misses. Exact proxy, hard
> legality, and hierarchy audit remain the only commit gates.
>
> Verification: `uv run python -m py_compile $(find src -type f -name '*.py')`
> passed; `git diff --check` passed; focused checks were `ibm08=1.1338`,
> `ibm10=1.1675`, `ibm12=1.6704`, and `ibm15=1.3618`, all VALID and
> audit-passing. Current full `uv run evaluate src/main.py --all` after the
> graph-utilization follow-ups = **AVG 1.1657**, 17/17 VALID, 0 overlaps, all
> final hierarchy audits passed, 1128.80s. This
> supersedes the prior audit-preserving **AVG 1.1664** result. Per-benchmark
> proxies:
> `ibm01=0.9258`, `ibm02=1.1180`, `ibm03=1.0050`, `ibm04=1.0115`,
> `ibm06=1.2097`, `ibm07=1.0525`, `ibm08=1.1331`, `ibm09=0.8541`,
> `ibm10=1.1715`, `ibm11=0.9939`, `ibm12=1.6723`, `ibm13=1.0200`,
> `ibm14=1.2794`, `ibm15=1.3595`, `ibm16=1.2147`, `ibm17=1.4046`,
> `ibm18=1.3906`.
>
> Follow-up graph-utilization infrastructure, not promoted as a new full-suite
> result: added `scripts/gnn/analyze_graph_tension.py` to summarize
> `hier_graph_tension`, decompression, coldspot candidate, rejection, and
> ego-net trace rows; added a default-off `HIER_COLDSPOT_EGONET` scaffold that
> synthesizes temporary graph-neighbor candidate groups for coldspot generation
> while leaving the real hierarchy model and all exact/audit gates unchanged.
> Trace mining on `ibm12` showed the first ego-net candidates were too large
> (`~118` hard, `~155` soft, `~46` micron mean hard displacement). The tuned
> opt-in path now prefers small neighbors
> (`HIER_COLDSPOT_EGONET_MAX_NEIGHBOR_HARD=32`), moves hard macros only by
> default (`HIER_COLDSPOT_EGONET_SOFT_MODE=none`), starts with low-displacement
> variants, and requires `HIER_COLDSPOT_EGONET_MIN_GAIN=0.001` before commit.
> Verification: bytecode compile passed; `git diff --check` passed; traced
> `ibm10` smoke with `HIER_GNN_TRACE=1` was VALID/audit-pass at `1.1682`; default
> `ibm06` smoke was VALID/audit-pass at `1.2130`; tuned opt-in `ibm12` was
> VALID/audit-pass at `1.6810`, with ego-net rows rejected unless they cleared
> the stricter gain gate. Ego-net remains default-off pending broader A/B because
> current accepted default `ibm12` remains better at `1.6704`.
>
> Added a default-off graph-aware coldspot anchor ranker
> (`HIER_COLDSPOT_GRAPH_ANCHOR_WEIGHT`) that keeps congestion as the primary
> anchor signal and uses distance to the selected cluster's weighted
> graph-neighbor centroid as a tie/near-tie bias. Focused opt-in traces were
> legal and audit-passing, but not promoted: `ibm12` with weight `0.10` finished
> at `1.6744` and weight `0.25` at `1.6736`, both behind accepted `1.6704`;
> `ibm10` with weight `0.10` finished at `1.1684`, behind accepted `1.1679`.
> Analyzer output now reports `graph_anchor_candidate_rows` and
> `graph_anchor_summary`.
>
> Added a default-off graph/local-relief prefilter (`HIER_GRAPH_PREFILTER`) for
> decompression and coldspot candidate generation. It rejects low-tension
> candidates before exact scoring/refinement when their cheap local congestion
> estimate does not improve, and logs `prefiltered`/`local_relief` trace fields.
> Focused opt-in traces were legal and audit-passing but mixed: `ibm08=1.1341`
> with 12 prefiltered rows and `ibm15=1.3619` with 9 prefiltered rows improved
> their accepted sweep numbers, while `ibm10=1.1676` with 6 prefiltered rows was
> worse than the prefilter-disabled control `ibm10=1.1667`. `ibm12=1.6744` was
> unchanged by disabling the filter. The hook remains default-off pending a more
> selective predictor.
>
> Added graph-corridor-aware component region expansion
> (`HIER_REGION_GRAPH_COMPONENT_WEIGHT`, default `0.0`). The hook biases
> component-aware early region expansion toward cold components near hierarchy
> graph edge corridors. It is opt-in only: `ibm10` with weight `0.10` was
> VALID/audit-pass but regressed to `1.1744` despite `graph_component=19`.
>
> Added default-on decompression feasibility screening
> (`HIER_DECOMPRESS_FEASIBILITY_FILTER`) that estimates free area and neighbor
> blockage for the proposed decompression bbox before legalization/exact scoring.
> Focused checks: `ibm10=1.1669`, VALID/audit-pass with 3
> `feasibility_blocked` rows; `ibm12=1.6744`, VALID/audit-pass with 0
> feasibility rejects; `ibm15=1.3627`, VALID/audit-pass with 1 feasibility
> reject. Analyzer output now reports `feasibility_rejected_rows` and
> `feasibility_summary`.
>
> Added graph-edge candidate delta tracing for coldspot and decompression:
> `edge_stretch_delta`, `corridor_congestion_delta`, `weighted_edge_delta`,
> `graph_candidate_delta`, and `graph_delta_edges`. The fields are diagnostic
> only. Focused traces: `ibm12=1.6758`, VALID/audit-pass, with the accepted
> coldspot candidate showing a large positive graph delta (`committed_delta`
> `13.9369`), meaning it improved proxy while worsening graph-edge geometry;
> decompression candidates often had favorable negative graph deltas but died on
> legality. `ibm10=1.1693`, VALID/audit-pass, showed the same pattern: average
> decompression graph delta was negative, while coldspot candidates were
> graph-expensive and rejected. Analyzer output now reports
> `graph_delta_candidate_rows` and `graph_delta_summary`.
>
> Added a default-off graph-delta coldspot ranker
> (`HIER_COLDSPOT_GRAPH_DELTA_RANK`, `HIER_COLDSPOT_GRAPH_DELTA_WEIGHT`) that
> adds a small proxy-equivalent penalty for graph-worsening exact coldspot
> candidates before the usual graph-score tie-break. Analyzer output now reports
> `graph_delta_ranked_rows` and `graph_delta_rank_summary`. Focused opt-in
> checks were valid/audit-passing but not promotable: `ibm12` with weight
> `0.0002` and `0.001` both finished at `1.6744` and still committed the same
> graph-worsening coldspot move (`committed_delta` `13.9381`); `ibm10` with
> weight `0.0002` finished worse at `1.1727` with no coldspot accepts.
>
> Implemented default-off graph-guided decompression legalization rescue
> (`HIER_DECOMPRESS_GRAPH_RESCUE`). When a decompression candidate has a
> favorable `graph_candidate_delta` but fails feasibility or hard overlap, the
> hook tries a bounded set of smaller and cold-component-shifted variants before
> returning to the normal hard legality, hierarchy-quality, exact-proxy, and
> audit gates. Focused traced `ibm10` showed the intended behavior:
> `graph_rescue_attempted_rows=51`, `used=7`, and 2 exact-accepted rescued
> decompression candidates, finishing VALID/audit-pass at `1.1664`. Required
> full-suite validation was legal but not promotable:
> `uv run evaluate src/main.py --all` = **AVG 1.1663**, 17/17 VALID, 0
> overlaps, all final hierarchy audits passed, 1134.11s. It improved `ibm12`
> (`1.6691` vs prior `1.6744`) and `ibm15`/`ibm16` slightly, but regressed
> `ibm10` (`1.1755`) and `ibm18` (`1.3957`) enough that the default remains
> off. Analyzer output now reports `graph_rescue_attempted_rows` and
> `graph_rescue_summary`.
>
> Implemented and promoted the narrower decompression graph survivor
> (`HIER_DECOMPRESS_GRAPH_SURVIVOR`). It only runs after a legal,
> hierarchy-safe decompression candidate misses exact proxy by a small margin
> while having favorable graph-edge delta; it exact-scores a capped local
> hard/soft one-cell polish pool and commits only if the polished candidate
> clears the normal exact-proxy gain and audit gates. Focused `ibm10` traces
> showed survivor attempts (`graph_survivor_attempted_rows=6`) but no direct
> survivor accept; the full suite nevertheless improved slightly through
> downstream ordering effects. Required full-suite validation with the hook
> enabled and then promoted to default:
> `uv run evaluate src/main.py --all` = **AVG 1.1657**, 17/17 VALID, 0
> overlaps, all final hierarchy audits passed, 1128.80s. Analyzer output now
> reports `graph_survivor_attempted_rows` and `graph_survivor_summary`.

> **[HISTORICAL] Status (2026-06-25 — audit-preserving local relief recovery):**
> Fixed the stale region-escape verifier so it uses the active `HierarchyModel`
> and restored the shared `any_outside_region()` helper. Added final
> hierarchy-quality audit telemetry against the selected hierarchy seed. The
> hierarchy pipeline now tracks a separate audit-safe checkpoint, independent of
> proxy-best state, and the finalizer rolls back to the best audit-passing
> checkpoint when the current state exceeds
> `HIER_FINAL_HIER_AUDIT_MAX_DEGRADATION=0.05`. Pass boundaries now enforce that
> checkpoint immediately, region-bounded hard-hard and hard-soft swap candidates
> are rejected before commit when they exceed the hierarchy-quality budget, and
> small-design polish restores the best audit-passing exact-scored subpass state
> instead of waiting for the final rollback.
> Region expansion now prefers nearby contiguous cold congestion
> components before falling back to side-band expansion, and cluster
> decompression can bias its local expansion/shift toward a nearby cold
> component while keeping exact-proxy and hierarchy-quality gates. Verification:
> targeted bytecode compile passed; `uv run python
> test/verification/_verify_region_escape_gate.py` passed on `ibm01`, `ibm04`,
> and `ibm10`; focused recovery checks passed on `ibm03`, `ibm08`, `ibm12`,
> and `ibm15`; `git diff --check` passed; full
> `uv run evaluate src/main.py --all` = **AVG 1.1664**, 17/17 VALID, 0 overlaps,
> all final hierarchy audits passed, 1146.64s. The previous strict-final-rollback
> result was **AVG 1.1999**; the current path recovers most of that proxy loss
> while keeping hierarchy audit as an enforced production invariant. The earlier
> **AVG 1.1627** hierarchy sweep remains a proxy reference only because its final
> hierarchy audit was report-only. Per-benchmark proxies:
> `ibm01=0.9254`, `ibm02=1.1180`, `ibm03=1.0057`, `ibm04=1.0079`,
> `ibm06=1.2130`, `ibm07=1.0544`, `ibm08=1.1339`, `ibm09=0.8535`,
> `ibm10=1.1692`, `ibm11=1.0001`, `ibm12=1.6756`, `ibm13=1.0200`,
> `ibm14=1.2770`, `ibm15=1.3612`, `ibm16=1.2147`, `ibm17=1.4050`,
> `ibm18=1.3933`.

> **Status (2026-06-24 — SA-ratio secondary subset cleanup):**
> The hierarchy flow now splits the SA-ratio secondary subset into two
> structural lanes. Small-secondary designs continue through the post-survivor
> small-design polish; no-release low-net cases shift candidate breadth from
> hard relocation toward soft relocation and soft-involving swaps. Medium/large
> congestion cases add a default-on soft-only continuation hook after strong
> soft repair, gated by structural shape and prior strong-soft exact gain. In
> the verified production sweep, the medium hook matched `ibm12` but did not run
> because spare-time gating was false; `ibm12` still improved through the normal
> strong-soft/coldspot path. Post-swap hard propose-all was not expanded.
> Verification: `uv run python -m py_compile $(find src -type f -name "*.py")`
> passed; `git diff --check` passed; `uv run evaluate src/main.py --all` =
> **AVG 1.1627**, 17/17 VALID, 0 overlaps, 1116.90s. The follow-up candidate
> gate keeps weak/hot early region reshape default-off and limits the opt-in
> path to small-design-sized low-confidence release candidates.
>
> Full per-benchmark proxies:
> `ibm01=0.9262`, `ibm02=1.1194`, `ibm03=1.0009`, `ibm04=1.0067`,
> `ibm06=1.2090`, `ibm07=1.0527`, `ibm08=1.1400`, `ibm09=0.8537`,
> `ibm10=1.1475`, `ibm11=0.9935`, `ibm12=1.6668`, `ibm13=1.0172`,
> `ibm14=1.2721`, `ibm15=1.3588`, `ibm16=1.2087`, `ibm17=1.4049`,
> `ibm18=1.3877`.

> **[REJECTED] Status (2026-06-24 — broad weak/hot early region reshape):**
> Tested early extra region room for hot clusters with weak inferred hierarchy
> confidence (`HIER_REGION_WEAK_HOT_RESHAPE=1`, confidence cutoff `0.92`,
> cap `2` clusters, extra side fraction `0.03`, side floor `0.45`). Focused
> `ibm04` improved to `1.0055`, but the full sweep regressed:
> `uv run evaluate src/main.py --all` = **AVG 1.1637**, 17/17 VALID,
> 0 overlaps, 1123.29s. Wins included `ibm02`, `ibm04`, `ibm07`, `ibm08`,
> `ibm09`, `ibm15`, `ibm17`, and `ibm18`; regressions on `ibm01`, `ibm06`,
> `ibm10`, `ibm11`, `ibm12`, `ibm13`, `ibm14`, and `ibm16` outweighed them.
> The hook remains in code but is default-off. Follow-up tightened the opt-in
> candidate gate to small-design-sized placements and to clusters in the same
> low-confidence release-candidate pool used by late small-design polish.

> **[HISTORICAL] Status (2026-06-24 — adaptive pass continuation by latest exact-gain):**
> The hierarchy flow now uses gain-gated pass continuation for major plateaus and
> stage transitions: interleaved soft repair, region swaps, post-swap hard/soft
> cleanup, swap and post-coldspot micro-shift replay, and coldspot refinement now
> skip to the next stage when the most recent exact-proxy gain is not above
> `HIER_PLATEAU_PROXY_GAIN` (`0.00005`).
> Full `uv run evaluate src/main.py --all` = **AVG 1.1714**, 17/17 VALID,
> 0 overlaps, 961.79s.

> **[HISTORICAL] Status (2026-06-24 — numba swap legality and no-trace swap fast path):**
> hard-hard and hard-soft region-swap legality now use numba short-circuit
> loops when available, avoiding repeated candidate-by-hard overlap matrix
> allocations. Default production also skips per-candidate swap trace
> dictionaries when both `HIER_GNN_TRACE` and region-swap GNN ranking are off,
> and scores the ranked legal list directly. While adding that fast path, the
> soft-soft trace path was corrected to use each row's own outside-region flag
> instead of a stale loop variable, so traced/ranked and default modes now share
> the same region gate. Verification: `_verify_score_region_swaps.py ibm01
> ibm04 ibm10` passed; direct numba-vs-vectorized legality parity on `ibm10`
> was 4096/4096 hard-hard and 4096/4096 hard-soft; `uv run evaluate
> src/main.py -b ibm10` = `proxy=1.1493` (wl=0.078, den=0.586, cong=1.556),
> VALID, 89.37s.

> **[HISTORICAL] Status (2026-06-23 — local CUDA scoring guards and cache cleanup):**
> bounded hard relocation, bounded soft relocation, and micro-shift now have a
> guarded path to reuse the exact `cuda_delta` relocation scorer for larger
> per-source target batches (`HIER_LOCAL_RELOC_CUDA_DELTA=auto`,
> `HIER_LOCAL_RELOC_CUDA_DELTA_MIN_TARGETS=64`). The scorer is intentionally
> dormant for the normal 8-24 target cleanup batches: a trial threshold of `8`
> was VALID on `ibm10` but consumed cleanup budget and regressed to
> `proxy=1.2896`, 89.62s. Restoring the high threshold recovered
> `uv run evaluate src/main.py -b ibm10` = `proxy=1.1573` (wl=0.080, den=0.585,
> cong=1.569), VALID, 89.36s. This turn also cached immutable legalizer spiral
> rings and reused coldspot window-integral/anchor results per field/window.
> Verification: bytecode compile passed; hard and soft CUDA delta parity
> verifiers passed; `_verify_coldspot_kick.py ibm10` passed.

> **[HISTORICAL] Status (2026-06-23 — whole-cluster coldspot diversity and predictor):**
> coldspot kick generation now adds default-on shape-preserving whole-cluster
> variants across the top two opportunity-ranked clusters: multiple cold
> anchors, compact original orientation, rotated layouts, source-facing border
> compaction, and a lower-displacement centroid-blended candidate. The hard
> cluster and owned/bridge softs still move as one state through legalization,
> local refinement, exact-proxy gating, and hierarchy-quality gating. Coldspot
> scheduling also has a cheap opportunity predictor that checks field relief,
> open cold cells, and source-to-window displacement before generating
> candidates, plus a dry-round limit for repeated no-commit pools.
> Non-GNN production now commits from exact-proxy-ranked refined candidates
> rather than graph-selected prefixes. Weak-opportunity and dry-limit exits skip
> graph-local and soft-only coldspot fallbacks too. This is implemented but not
> yet recorded as an accepted full-suite result.

> **[HISTORICAL] Status (2026-06-23 — promoted default-on gates to unconditional behavior):**
> removed production constants that were always `True` and used only as feature
> gates. The accepted hierarchy flow now runs those operators unconditionally:
> tag-prefix/oversize clustering, connectivity-ordered legalization, seed
> portfolio variants, bridge soft roles, region relief and expansion,
> hierarchy-aware congestion proposal ranking, micro-shift replays, decompression,
> region swaps, additive candidate pools, plateau/component scheduling,
> strong/interleaved soft repair, coldspot local refinement and graph fallback,
> survivor search, scorer congestion-field reuse, and numba routing/structural
> paths when available. Numeric tuning constants and default-off experiments
> remain configurable. Verification for this cleanup: Python bytecode compile
> across `src/` and `test/` passes; benchmark smoke is tracked in the current
> implementation turn rather than as a new accepted full-suite result.

> **[HISTORICAL] Status (2026-06-23 — default-off soft-only coldspot fallback):**
> added `HIER_COLDSPOT_SOFT_ONLY=0`, which runs only when the hard coldspot kick
> path and graph-local fallback commit no candidate. The pass keeps all hard
> macros fixed, builds a soft relocation target pool from remembered open cold
> cells, preserves hierarchy region boxes and the cold-cell mask, and accepts
> only exact-proxy improvements above `HIER_COLDSPOT_SOFT_ONLY_MIN_GAIN`.
> This is intended as the next low-risk coldspot experiment because it cannot
> alter hard legality or hard hierarchy quality by construction. It is not
> promoted and does not change default production output.
> Follow-up replaced the after-hard refill idea with always-on simultaneous
> hard+soft coldspot candidates. Coldspot kick generation now includes movable
> bridge soft macros tied to the same hierarchy cluster in the candidate's soft
> set, so hard and soft positions are proposed together and accepted or rejected
> as one exact-gated full placement.

> **[HISTORICAL] Status (2026-06-23 — default-off partial frontier coldspot prototype):**
> added `HIER_COLDSPOT_PARTIAL_FRONTIER=0`, which can append one
> capacity-aware partial frontier candidate to a coldspot candidate pool. The
> prototype estimates connected cold area near the selected anchor, clamps the
> moved area to a source-cluster fraction, selects frontier hard macros with a
> low-fanout connectivity bias, optionally co-moves directly connected soft
> macros, and relies on the existing legalization, local refinement,
> exact-proxy, and hierarchy-quality gates. This is not promoted and does not
> change default production output. Verification:
> `uv run python test/verification/_verify_coldspot_kick.py ibm04 ibm10`
> passed, including one partial frontier pool on each benchmark.
> Follow-up enabled traces showed the first implementation was mostly
> under-ranked by graph selection; on `ibm10`, widening graph top-k plus oracle
> scoring exact-labeled the partial candidate as a proxy regression
> (`proxy_delta=+0.0005`). A five-benchmark enabled smoke (`ibm04`, `ibm10`,
> `ibm12`, `ibm15`, `ibm17`) was 5/5 VALID, 0 overlaps, AVG **1.3356**,
> 431.44s, with per-benchmark proxies `1.0277`, `1.1577`, `1.6776`,
> `1.4036`, `1.4112`. The only exact-improving partial candidate was on
> `ibm15` (`proxy_delta=-0.0097`) but it failed hierarchy quality badly
> (`0.8022 -> 2.9406`) because it moved two macros out of a three-macro
> cluster. The prototype now skips tiny source clusters by default with
> `HIER_COLDSPOT_PARTIAL_MIN_CLUSTER_HARD=6`; explicit source-hotspot
> reorganization was not added because the hierarchy metric is dominated by the
> far split itself, not by unfilled area left behind.
> Follow-up implementation added a cheap pre-exact split-quality predictor for
> surviving partial candidates: after hard legalization, the generator rejects
> candidates whose full source-cluster radius, bbox radius, or moved-vs-remaining
> centroid separation exceeds configured ratios
> (`HIER_COLDSPOT_PARTIAL_MAX_RADIUS_RATIO=1.15`,
> `HIER_COLDSPOT_PARTIAL_MAX_BBOX_RATIO=1.20`,
> `HIER_COLDSPOT_PARTIAL_MAX_SEPARATION_RATIO=1.50`). This keeps obviously
> hierarchy-damaging partial splits out of the expensive exact/refinement pool.
> The generator now also rejects majority splits, splits that leave too few
> hard macros behind, and disconnected selected subsets when local low-fanout
> hard-hard edges exist. `HIER_GNN_TRACE=1` writes
> `hier_coldspot_partial_reject` rows for generated-but-filtered partials, so
> future threshold tuning can count rejected candidates by reason before
> spending exact score time.
> Capping the selected subset during construction, instead of only rejecting
> majority splits after selection, was tested with another enabled trace smoke.
> `ibm10` produced a smaller 10-of-21 split but the pre-exact predictor still
> rejected it as a far stretch (`radius_ratio=2.10`, `bbox_ratio=1.64`,
> `separation_ratio=1.79`). The five-benchmark enabled capped smoke was 5/5
> VALID, 0 overlaps, AVG **1.3385**, 433.99s, with per-benchmark proxies
> `ibm04=1.0277`, `ibm10=1.1609`, `ibm12=1.6858`, `ibm15=1.4051`,
> `ibm17=1.4130`. It generated 11 rejected partial attempts and 0 surviving
> partial candidates: most rejects were tiny clusters, with the only larger
> attempt rejected by radius ratio. No source-hotspot reorganization was added;
> the traced blocker remains far hierarchy splitting, not lack of source-side
> area relaxation.

> **[HISTORICAL] Status (2026-06-23 — accepted swap-round micro-shift, stronger
> opportunity gates, and component-aware scheduling):** promoted
> `HIER_SWAP_ROUND_MICRO_SHIFT=1`, `HIER_STRONG_OPPORTUNITY_GATES=1`, and
> `HIER_COMPONENT_AWARE_SCHEDULING=1`. Region swaps now replay exact-gated
> micro-shift after each swap round as well as after the full swap pass.
> Optional decompression/coldspot work now has stronger hot-vs-cold field gates;
> this changes scheduling only, not exact acceptance. Late strong soft repair
> records exact wirelength/density/congestion components and can treat
> congestion-dominated states as useful cleanup opportunities. Rejected and
> removed during the same audit: early strong-soft repair, early swap-lite,
> early survivor search, and ArchGen-style seed top-k repair. Full verification:
> `uv run evaluate src/main.py --all` = **AVG 1.1793**, 17/17 VALID,
> 0 overlaps, 1421.12s. `ibm10` in the full sweep was `1.1576`, VALID.
>
> Full per-benchmark proxies:
> `ibm01=0.9392`, `ibm02=1.1345`, `ibm03=1.0085`, `ibm04=1.0277`,
> `ibm06=1.2092`, `ibm07=1.0698`, `ibm08=1.1593`, `ibm09=0.8692`,
> `ibm10=1.1576`, `ibm11=1.0242`, `ibm12=1.6850`, `ibm13=1.0230`,
> `ibm14=1.2889`, `ibm15=1.4043`, `ibm16=1.2093`, `ibm17=1.4141`,
> `ibm18=1.4234`.

> **[HISTORICAL] Status (2026-06-22 — Archgen-inspired buffered telemetry, GPU proposal
> ranking, and plateau escape proposal class):** added buffered default-on
> plateau telemetry (`HIER_PLATEAU_TRACE_BUFFERED=1`) with per-benchmark flush
> plus `atexit` fallback. Added CUDA sorting for large swap and relocation
> target-rank arrays (`HIER_GPU_RANK_SWAP_CANDIDATES=auto`,
> `HIER_GPU_RANK_RELOCATION_TARGETS=auto`,
> `HIER_GPU_RANK_MIN_CANDIDATES=512`). Added a plateau escape proposal class:
> if region swaps plateau and spare budget remains, the scheduler spends a
> short exact-gated soft-relocation slice inside soft hierarchy regions before
> post-swap polish. Verification: `uv run python -m py_compile $(find src -type
> f -name '*.py')` passed; `_verify_score_region_swaps.py ibm01 ibm04 ibm10`
> passed; `uv run evaluate src/main.py -b ibm10` was VALID with
> `proxy=1.1539`, 0 overlaps, 92.17s. On that smoke, region swaps were not
> plateaued, so the new escape pass did not trigger; post-swap hard/soft
> relocation plateaus still scheduled strong soft repair as before. Follow-up
> full sweep reported by the user: `uv run evaluate src/main.py --all` =
> **AVG 1.1791**, 17/17 VALID, 0 overlaps, 1376.96s.
> Follow-up plateau expansion scopes the escape proposal class to both
> region-swap plateaus and post-swap hard/soft cleanup plateaus
> (`HIER_PLATEAU_ESCAPE_AFTER_POST_POLISH=1`). The new soft-relocation GPU
> batch ranker is used only by this plateau escape class by default, so normal
> interleaved/post/strong soft repair keeps the previously accepted CPU
> hierarchy ordering. The CUDA prefix applies the same hierarchy-aware target
> filter before ranking, then exact incremental scoring remains the accept
> gate. Verification after scoping: py-compile passed;
> `_verify_score_region_swaps.py ibm01 ibm04 ibm10` passed; `uv run evaluate
> src/main.py -b ibm10` was VALID with `proxy=1.1543` (wl=0.081, den=0.586,
> cong=1.561), 0 overlaps, 89.97s. On that smoke, post-swap soft relocation
> plateaued, `plateau_escape_post_soft_relocation` accepted 38 soft moves
> (`1.1623 -> 1.1583`), and strong/coldspot cleanup finished at `1.1543`.
> Staged GPU swap prescore follow-up: added a guarded CUDA swap-prescore helper
> that can add a small distance-aware batch ranking term before exact swap
> scoring. Soft-soft was tested first and left default-off after ibm10
> regressed to `proxy=1.1553`, VALID, 91.22s. Hard-soft was tested second and
> left default-off after ibm10 regressed to `proxy=1.1554`, VALID, 91.42s.
> Hard-hard was tested last and kept default-on
> (`HIER_GPU_SWAP_PRESCORE_HH=auto`) after ibm10 improved to `proxy=1.1525`
> (wl=0.081, den=0.588, cong=1.555), VALID, 91.56s. Exact swap verification
> still passed on ibm01/ibm04/ibm10.
> Follow-up all-prescore validation promoted hard-hard, hard-soft, and soft-soft
> to default `auto` rather than default-off gates. The condition is now runtime
> capability and scale: CUDA must be available and the candidate list must meet
> `HIER_GPU_SWAP_PRESCORE_MIN_CANDIDATES`. A temporary monkeypatched all-on
> wrapper reached `proxy=1.1513` (wl=0.080, den=0.586, cong=1.557), VALID,
> 90.21s, but after promoting the constants the normal command repeated at
> `proxy=1.1599` (wl=0.077, den=0.587, cong=1.578), VALID, 91.47s and 91.58s.
> The all-auto behavior remains per the no-default-off-gates direction, but the
> stable normal-command smoke is weaker than the previous HH-only setting.
> Exact reversible batched swap scoring was then tested for soft-soft,
> hard-soft, and hard-hard `score_swap_*_many()` paths. Direct many-vs-scalar
> checks on ibm01/ibm04/ibm10 matched to numerical precision, and the standard
> `_verify_score_region_swaps.py ibm01 ibm04 ibm10` verifier passed. However,
> the reversible scorer was slower than the current full-grid snapshot path on
> ibm10: all swap types active finished `proxy=1.1633`, VALID, 92.00s; isolated
> soft-soft reversible scoring finished `proxy=1.1613`, VALID, 90.30s. The
> reversible exact batch scorer was not promoted and was removed from the active
> code path. Current production remains the cached-struct exact scorer.

> **[HISTORICAL] Status (2026-06-22 — swap scoring throughput speedup, smoke-validated):**
> added exact-equivalent cached routing structs for repeated multi-macro swap
> scoring and vectorized region-bbox masks for hard-hard, hard-soft, and
> soft-soft swap candidate outside-region flags. This does not change candidate
> ranking, legality, hierarchy escape rules, or exact-proxy accept gates.
> Verification: `uv run python -m py_compile $(find src -type f -name '*.py')`
> passed; `uv run python test/verification/_verify_score_region_swaps.py ibm01
> ibm04 ibm10` matched full exact proxy on all trial and sequential commit
> checks; `uv run evaluate src/main.py -b ibm10` was VALID with
> `proxy=1.1545`, 0 overlaps, 92.37s. Compared with the pre-speedup smoke
> (`proxy=1.1598`, 91.55s), wall time is similar because region swaps are
> deadline-bound, but the swap stage scored more soft-soft candidates
> (`25860 -> 29194`) and improved proxy on the smoke. Additional smoke:
> `uv run evaluate src/main.py -b ibm01` was VALID with `proxy=0.9448`,
> 0 overlaps, 76.71s.

> **CURRENT SYSTEM (2026-06-22): hierarchy path with exact-prescored seed
> portfolio.** The active code no longer
> ships the proxy-optimized production path. `MacroPlacer.place()` always routes
> through `_hierarchy_floorplan()` and raises if grouped DREAMPlace is
> unavailable. Deleted proxy-only pieces include candidate restarts, R2/2-opt,
> hard-soft/soft swap and cycle passes, generic LSMC, generic cluster kicks, ML
> ranker defaults, and their verifiers. Current verified full run after adding
> seed portfolio prescoring and hierarchy-aware congestion-weighted proposal
> ranking: **AVG 1.1879**, 17/17 VALID, 0 overlaps, 1225.52s.
> BeyondPPA-style structural metrics,
> optional hierarchy candidate ordering, and opt-in GNN trace logging remain
> integrated into the hierarchy flow with production defaults disabled where
> noted. Current verified full run after adding strong soft repair, plateau
> telemetry, and budget-aware pass scheduling: **AVG 1.1827**, 17/17 VALID,
> 0 overlaps, 1328.33s. The older proxy-score history below is retained as
> experiment context.

> **[HISTORICAL] Status (2026-06-22 — six-stage hierarchy-aware revamp):** applied the
> staged revamp with full `uv run evaluate src/main.py --all` gates after each
> active stage. Stage 1 verified NG45 path-tag hierarchy support on IBM with
> **AVG 1.1822**, 17/17 VALID, 0 overlaps. Stage 2 added a hierarchy-safe
> route-channel seed but gates it to explicit slash-separated hierarchy tags;
> flat IBM uses the original seed portfolio. Stage 2 IBM verification remained
> valid at **AVG 1.1824**, so the feature is useful for tagged designs but not
> a promoted IBM proxy contributor. Stage 3 added interleaved exact-gated soft
> repair after decompression and improved to **AVG 1.1815**, 17/17 VALID,
> 0 overlaps. Stage 4 added plateau-driven strong-soft budget/round bonuses
> and improved to **AVG 1.1808**, 17/17 VALID, 0 overlaps. Stage 5 implemented
> GPU-ranked additive hard-relocation tails, but the direct sweep regressed to
> **AVG 1.1810** and showed 0 post-swap hard propose-all accepts, so
> `HIER_GPU_RANK_ADDITIVE_TAILS=0` remains default-off. With that default-off
> setting, the recovery sweep produced **AVG 1.1796**, 17/17 VALID, 0 overlaps,
> driven by normal coldspot variance rather than the disabled tail ranker.
> Stage 6 added audit-only scorer-compatible legality margin telemetry
> (`HIER_LEGALITY_MARGIN_AUDIT=1`, `HIER_LEGALITY_MARGIN_EPS=0.05`). Final
> current-code verification was **AVG 1.1817**, 17/17 VALID, 0 overlaps,
> 1383.28s. The audit found many evaluator-valid placements with strict
> internal margins around `-0.045` to `-0.050`, while some large cases
> (`ibm10`, `ibm12`, `ibm17`) were margin-clean. This gives future work a
> concrete signal for optional clearance repair without changing current
> placement behavior.

> **[HISTORICAL] Status (2026-06-22 — NG45 hierarchy-tag verification added and passed):**
> ran `uv run evaluate src/main.py --ng45` after the strong soft repair changes:
> all four NG45 designs were VALID with 0 overlaps, `AVG 0.7300`
> (`ariane133=0.6845`, `ariane136=0.7283`, `mempool_tile=0.7730`,
> `nvdla=0.7340`). A new tag-prefix verifier using NG45 hard macro instance
> paths initially failed `nvdla`: fine-grained SRAM pair tags had
> `radius_growth=1.421` and nearest-tag purity dropped by `0.102`, even though
> legality and proxy were valid. This showed the hierarchy model was not using
> NG45's explicit hierarchy tags strongly enough.
>
> Added `HIER_TAG_PREFIX_CLUSTERING=1` so `HierarchyModel.build()` first derives
> hard clusters from slash-separated instance-path prefixes when those tags have
> useful coverage; flat IBM names still fall back to connectivity clustering.
> Added `test/verification/_verify_ng45_hierarchy_tags.py`, which checks hard
> overlap count, same-tag normalized radius, and nearest-prefix purity. Final
> verifier result passed all four designs:
> `ariane133 radius_growth=1.018`, `ariane136=0.998`, `mempool_tile=0.900`,
> `nvdla final_radius=0.0432` of die diagonal. Final canonical
> `uv run evaluate src/main.py --ng45` after tag clustering = **AVG 0.7320**,
> all four VALID, 0 overlaps:
> `ariane133=0.6888`, `ariane136=0.7361`, `mempool_tile=0.7730`,
> `nvdla=0.7303`. This trades a small NG45 average proxy increase for verified
> explicit hierarchy-tag preservation; `nvdla` improves versus the pre-tag NG45
> run while the Ariane designs become more hierarchy-constrained.

> **[HISTORICAL] Status (2026-06-22 — strong soft repair, plateau telemetry, and
> budget-aware scheduling added):** added pass-level `PlateauTelemetry` records
> for region relief, swaps, post-swap polish, decompression, split evacuation,
> and the new strong soft repair. Plateau rows are stored by default under
> `ml_data/beyondppa_gnn/plateau/plateau_telemetry.jsonl` and include pass
> runtime, candidate/legal/scored counts, accepts, proxy gain, plateau flag, and
> scheduler decisions. `HIER_PLATEAU_TRACE=0` disables the lightweight storage;
> `HIER_PLATEAU_TRACE_PATH` redirects it. The existing opt-in GNN trace remains
> separate.
>
> The new default-on strong soft repair spends spare local pass budget on larger
> exact-gated soft relocation target pools after normal post-swap hard/soft
> cleanup. `HIER_BUDGET_AWARE_SCHEDULING=1` starts it only when at least
> `HIER_STRONG_SOFT_REPAIR_MIN_SPARE_S=2` seconds remain and recent telemetry
> shows plateaued cleanup or useful soft movement. Verification:
> `uv run python -m py_compile $(find src -type f -name '*.py')` passed.
> Focused smokes improved two sensitive cases: `ibm10` improved from the prior
> hierarchy-aware weighted ranking smoke/full entry `1.1699/1.1717` to
> `proxy=1.1633` VALID (`wl=0.078`, `den=0.591`, `cong=1.579`, 91.86s), with
> the strong repair accepting 69 soft moves and improving post-soft
> `1.1716 -> 1.1633`; `ibm17` improved from `1.4118` to `proxy=1.4053` VALID
> (`wl=0.066`, `den=0.677`, `cong=2.002`, 88.63s), with the strong repair
> accepting 51 soft moves and coldspot tightening one additional move.
>
> Full `uv run evaluate src/main.py --all` = **AVG 1.1827**, 17/17 VALID,
> 0 overlaps, 1328.33s. This improves the previous accepted full result
> `1.1879 -> 1.1827` (-0.0052). `ibm17` was the only full-suite regression
> because the scheduler skipped strong soft repair when less than 2s local
> spare remained (`1.4118 -> 1.4128`); the aggregate gain was still positive.
>
> Full per-benchmark proxies:
> `ibm01=0.9444`, `ibm02=1.1369`, `ibm03=1.0127`, `ibm04=1.0334`,
> `ibm06=1.2152`, `ibm07=1.0684`, `ibm08=1.1617`, `ibm09=0.8690`,
> `ibm10=1.1630`, `ibm11=1.0279`, `ibm12=1.6973`, `ibm13=1.0268`,
> `ibm14=1.2853`, `ibm15=1.4161`, `ibm16=1.2138`, `ibm17=1.4128`,
> `ibm18=1.4215`.

> **[HISTORICAL] Status (2026-06-22 — congestion-weighted ranking made hierarchy-aware):**
> kept the first-place-inspired weighted proposal field, but added
> `HIER_PROPOSAL_HIERARCHY_AWARE=1` so out-of-region candidates do not dominate
> ranking merely because they are cold. In hard/soft relocation and region
> swaps, when an in-region option exists, an out-of-region option must beat the
> best in-region field relief by `HIER_PROPOSAL_OUTSIDE_RELIEF_MARGIN=0.08`
> times the active proposal-field span before it can enter the truncated ranked
> set. Exact proxy, legality, fixed-macro, bounds, region-escape, and
> hierarchy-quality gates remain unchanged.
>
> Verification: `uv run python -m py_compile $(find src -type f -name '*.py')`
> passed. Focused smokes improved versus the prior weighted-only portfolio
> run: `ibm10 1.1729 -> 1.1699` VALID and `ibm17 1.4200 -> 1.4118` VALID.
> Full `uv run evaluate src/main.py --all` = **AVG 1.1879**, 17/17 VALID,
> 0 overlaps, 1225.52s. This slightly improves the prior `1.1880` portfolio
> run while better preserving hierarchy in weighted proposal ranking.
>
> Full per-benchmark proxies:
> `ibm01=0.9472`, `ibm02=1.1409`, `ibm03=1.0153`, `ibm04=1.0344`,
> `ibm06=1.2190`, `ibm07=1.0718`, `ibm08=1.1651`, `ibm09=0.8693`,
> `ibm10=1.1717`, `ibm11=1.0310`, `ibm12=1.7159`, `ibm13=1.0297`,
> `ibm14=1.2904`, `ibm15=1.4232`, `ibm16=1.2231`, `ibm17=1.4118`,
> `ibm18=1.4340`.

> **[HISTORICAL] Status (2026-06-22 — first-place lessons implemented: seed portfolio +
> congestion-weighted proposal ranking):** added two ArchGen-inspired ideas
> without restoring the deleted proxy-only path. `HIER_SEED_PORTFOLIO=1`
> exact-prescores grouped DREAMPlace, legalized `initial.plc`, two
> DREAMPlace/initial blends, radial expansion, and synthetic clearance before
> region relief. `HIER_CONGESTION_WEIGHTED_PROPOSALS=1` makes hard/soft
> relocation and region swaps rank their congestion pass by a normalized
> `2.5*congestion + 1.0*density` proposal field while keeping exact proxy as
> the only accept gate.
>
> Verification: `uv run python -m py_compile $(find src -type f -name '*.py')`
> passed. Focused smokes: `ibm10=1.1729` VALID (selected `initial`,
> `dreamplace=1.8080` prescore) and `ibm17=1.4200` VALID (selected `initial`,
> `dreamplace=2.5025` prescore). Full `uv run evaluate src/main.py --all` =
> **AVG 1.1880**, 17/17 VALID, 0 overlaps. The evaluator printed an anomalous
> `ibm13` elapsed time (`35354.51s`), inflating reported total runtime to
> `36564.51s`; wall-clock progress was normal and the suite completed.
>
> Full per-benchmark proxies:
> `ibm01=0.9520`, `ibm02=1.1428`, `ibm03=1.0156`, `ibm04=1.0340`,
> `ibm06=1.2190`, `ibm07=1.0710`, `ibm08=1.1650`, `ibm09=0.8697`,
> `ibm10=1.1729`, `ibm11=1.0312`, `ibm12=1.6948`, `ibm13=1.0316`,
> `ibm14=1.2994`, `ibm15=1.4006`, `ibm16=1.2299`, `ibm17=1.4284`,
> `ibm18=1.4385`.
>
> Seed selections in the full run:
> DREAMPlace selected on `ibm01`, `ibm02`, `ibm03`, `ibm04`, `ibm06`, `ibm08`,
> `ibm09`, `ibm11`, and `ibm13`; synthetic clearance selected on `ibm07`;
> `initial.plc` selected on `ibm10`, `ibm12`, `ibm14`, `ibm15`, `ibm16`,
> `ibm17`, and `ibm18`. This confirms the portfolio is not merely reverting to
> `initial.plc`; it selects different basins by exact prescore.

> **[HISTORICAL] Status (2026-06-21 — gated oversized-cluster split promoted over flat
> default):** investigated whether `ibm17`'s loss after disabling recursive
> clustering was random. It was deterministic: repeated focused toggles gave
> `ibm17 flat/no-rooms=2.1529`, `recursive/no-rooms=2.1220`, and
> `recursive+rooms=2.1073`. The correlated losers `ibm03`, `ibm11`, and
> `ibm17` all had one oversized flat cluster that hid useful substructure
> (`ibm03 234/290`, `ibm11 171/373`, `ibm17 351/760`). A naive 30%/15%
> splitter improved those cases but was not safe globally: it regressed
> `ibm07` by `+0.0129`, `ibm14` by `+0.0446`, and `ibm16` by `+0.0301`.
> Accepted production rule is therefore gated: `HIER_OVERSIZE_CLUSTER_SPLIT=1`,
> `HIER_OVERSIZE_CLUSTER_START_FRAC=0.40`,
> `HIER_OVERSIZE_CLUSTER_TARGET_FRAC=0.15`,
> `HIER_OVERSIZE_CLUSTER_TARGET_TOL=1.10`, and
> `HIER_OVERSIZE_CLUSTER_MIN_BRIDGE_SOFTS=5`. Full recursive clustering and
> cluster room/corridor branches were later deleted from production code.
>
> Verification: `uv run python -m py_compile $(find src -type f -name '*.py')`
> passed. `uv run evaluate src/main.py --all` = **AVG 1.3781**, 17/17 VALID,
> 0 overlaps, 1162.84s. This improves the flat/no-room default
> `1.3798 -> 1.3781` (-0.0017) and recovers `ibm17` by `2.1529 -> 2.1241`
> (-0.0288), while avoiding the large `ibm14`/`ibm16` regressions from the
> naive 30% splitter. It is still worse than the lower-proxy accepted hierarchy
> reference **AVG 1.3631**.
>
> Final gated per-benchmark proxies:
> `ibm01=0.9558`, `ibm02=1.1520`, `ibm03=1.0135`, `ibm04=1.0343`,
> `ibm06=1.2252`, `ibm07=1.0703`, `ibm08=1.1566`, `ibm09=0.8676`,
> `ibm10=1.6185`, `ibm11=1.0286`, `ibm12=2.2064`, `ibm13=1.0358`,
> `ibm14=1.6538`, `ibm15=1.8603`, `ibm16=1.6849`, `ibm17=2.1241`,
> `ibm18=1.7395`.

> **[HISTORICAL] Status (2026-06-21 — recursive clustering and room/corridor regions removed
> from production, then deleted):** removed the recursive weighted clustering
> path and explicit room/corridor region boxes after the staged revamp showed
> they were detrimental overall. The active system now keeps the first-class
> `HierarchyModel`, additive candidate pools, composite hierarchy quality, and
> pass state/context/result objects, but uses tag/oversized inferred clusters
> plus the existing hard/soft congestion-expanded region boxes. Verification:
> `uv run evaluate src/main.py --all` = **AVG 1.3798**, 17/17 VALID,
> 0 overlaps, 1152.01s. This improves the Stage 6 revamp sweep
> `1.3846 -> 1.3798` (-0.0048) while remaining worse than the lower-proxy
> accepted hierarchy reference **AVG 1.3631**.
>
> Per-benchmark change versus Stage 6 revamp:
> `ibm01 +0.0031`, `ibm02 +0.0000`, `ibm03 +0.0158`, `ibm04 -0.0011`,
> `ibm06 +0.0000`, `ibm07 -0.0160`, `ibm08 +0.0016`, `ibm09 -0.0006`,
> `ibm10 -0.0127`, `ibm11 +0.0114`, `ibm12 -0.0084`, `ibm13 +0.0012`,
> `ibm14 -0.0226`, `ibm15 -0.0433`, `ibm16 -0.0604`, `ibm17 +0.0456`,
> `ibm18 +0.0049`. The removal mostly helps dense packed cases
> `ibm10`, `ibm12`, and `ibm14`-`ibm16`; it hurts `ibm03`, `ibm11`, and
> especially `ibm17`.

> **[HISTORICAL] Status (2026-06-21 — six-stage hierarchy architecture revamp valid, not
> lower-proxy promoted):** implemented the requested six revamp directions in
> staged order and ran `uv run evaluate src/main.py --all` after each stage.
> The new code introduces a first-class `HierarchyModel`, recursive hard-cluster
> partitioning, cluster-room/bridge-corridor regions, spare-budget additive
> candidate tails, a composite hierarchy-quality metric, and shared
> `PlacementState` / `PassContext` / `PassResult` orchestration objects. Final
> Stage 6 verification was **AVG 1.3846**, 17/17 VALID, 0 overlaps, 1170.17s.
> This is valid, but it regresses proxy versus the lower-proxy accepted
> hierarchy reference **AVG 1.3631**, so treat it as an architecture revamp and
> diagnostic base rather than a proxy-score promotion.
>
> Stage contribution table, lower proxy is better:
>
> | Stage | Change | `--all` AVG | Delta vs prior |
> |---|---:|---:|---:|
> | 1 | First-class hierarchy model and trace metadata | 1.3803 | baseline for this revamp run |
> | 2 | Recursive weighted hierarchy partitioning | 1.3839 | +0.0036 worse |
> | 3 | Cluster rooms and bridge corridors before macro moves | 1.3864 | +0.0025 worse |
> | 4 | Spare-budget additive candidate pools | 1.3849 | -0.0015 better |
> | 5 | Composite hierarchy-quality objective | 1.3843 | -0.0006 better |
> | 6 | Split pass orchestration into state/context/result objects | 1.3846 | +0.0003 worse |
>
> Final Stage 6 per-benchmark proxies:
> `ibm01=0.9504`, `ibm02=1.1520`, `ibm03=1.0053`, `ibm04=1.0354`,
> `ibm06=1.2252`, `ibm07=1.0863`, `ibm08=1.1549`, `ibm09=0.8719`,
> `ibm10=1.6307`, `ibm11=1.0277`, `ibm12=2.2030`, `ibm13=1.0340`,
> `ibm14=1.6764`, `ibm15=1.9036`, `ibm16=1.7436`, `ibm17=2.1073`,
> `ibm18=1.7311`.

> **[HISTORICAL] Status (2026-06-20/21 — coldspot selector demoted; regional ops confirmed
> as GNN target):** added default-off additive coldspot selector infrastructure:
> `HIER_GNN_COLDSPOT_SELECT`, `HIER_GNN_COLDSPOT_MODEL`,
> `HIER_GNN_COLDSPOT_KICKS`, `HIER_GNN_COLDSPOT_TOP_K`,
> `HIER_GNN_COLDSPOT_SKIP_MICRO`, and trace-only
> `HIER_GNN_COLDSPOT_ORACLE`. The selector ranks generated coldspot kick
> outcomes; it does not generate coordinates, and all existing hard legality,
> hierarchy-quality, budget, and exact-proxy gates remain mandatory. Default
> production is unchanged. Full default `--all` after the implementation was
> **AVG 1.3667**, 17/17 VALID, 0 overlaps, 634.76s, so this is not a new
> accepted baseline versus **AVG 1.3631**. Oracle-pool traces were collected for
> `ibm10`, `ibm12`, and `ibm17` with `HIER_GNN_COLDSPOT_ORACLE=1`,
> `HIER_GNN_COLDSPOT_SELECT=0`, and `HIER_GNN_COLDSPOT_KICKS=8`; artifacts live
> under `ml_data/beyondppa_gnn/traces/20260620_coldspot_oracle_k8/` and
> `ml_data/beyondppa_gnn/datasets/20260620_coldspot_oracle_k8_only_*`. The
> isolated dataset has 216 coldspot candidates across 24 pools and only 8
> accepted/proxy-gain candidates, all in one `ibm12` pool. Diagnostic result:
> `trace_order` and `field_delta` miss the best-gain candidate at top-1/top-4;
> oracle top-1 succeeds by definition. This is too sparse and benchmark-specific
> to train a meaningful coldspot GNN. Do not promote or train a coldspot model
> from this dataset. Correction: the intended learned target is regional
> relocation and regional hard-hard, hard-soft, and soft-soft swaps, with an
> optional soft-macro barrier for soft-involving moves.

> **[HISTORICAL] Status (2026-06-21 — regional swap GNN hook added, closed-loop rejected for
> sequential reordering):** added default-off `region_swaps` inference support
> to `src/placer/local_search/gnn_ranker.py`, plus guarded-prefix controls
> `HIER_GNN_SWAP_PRESERVE_TOP_N` and `HIER_GNN_SWAP_TOP_K`. Added
> `HIER_SOFT_BARRIER_GAIN` (production default `0.0`; use `0.01` for soft
> barrier diagnostics) and applied it only to soft relocation, hard-soft swaps,
> and soft-soft swaps. Offline regional diagnostics on
> `20260620_coldspot_oracle_k8_only_ibm10_12_17.pt` show real ranking signal:
> `region_swaps` G4 exact-gain recall@4 `0.5984` versus trace order `0.1933`,
> accepted recall@4 `0.7675` versus trace order `0.3259`. Closed-loop `ibm12`
> smokes were all VALID but worse: full GNN region-swap reorder plus
> `HIER_SOFT_BARRIER_GAIN=0.01` gave `2.3090`; full GNN reorder without the
> barrier gave `2.2400`; guarded prefix
> `HIER_GNN_SWAP_PRESERVE_TOP_N=8,HIER_GNN_SWAP_TOP_K=8` gave `2.2199`.
> Deterministic/default `ibm12` in the same current stack is around
> `2.1847-2.1855`. Conclusion: the GNN has useful regional-swap candidate
> ranking signal offline, but sequentially reordering the swap stream changes
> state too aggressively and is rejected. Next regional GNN work should be
> additive or budgeted: preserve deterministic swap order, exact-score a small
> GNN-ranked supplemental set only when budget remains, and keep
> `HIER_SOFT_BARRIER_GAIN=0.01` as a diagnostic soft-macro gate rather than a
> production default.

> **[HISTORICAL] Status (2026-06-19 — GNN Stage G5 default-off relocation hook smoke
> accepted):** added `src/placer/local_search/gnn_ranker.py`, a default-off
> inference hook that can reorder the existing hard propose-all relocation
> candidate list with the accepted G4 macro-net ranker. Runtime controls:
> `HIER_GNN_RANK=1`,
> `HIER_GNN_MODEL=ml_data/beyondppa_gnn/models/20260619_g4_macro_net_ranker_v1/model.pt`,
> `HIER_GNN_OPERATORS=relocation`, and `HIER_GNN_TOP_K=32`. The hook runs after
> existing proposal scoring and before proposal truncation/exact checking; it
> does not alter hard legality, fixed-macro immobility, bounds,
> hierarchy-region constraints, hierarchy-quality gates, or exact-proxy
> acceptance. Smoke pair on `ibm10`: default-off baseline VALID
> `proxy=1.6192`, 45.80s; GNN-ranked relocation-only VALID `proxy=1.6180`,
> 50.16s. This is smoke acceptance only, not default-on promotion. Next stage is
> G6 multi-benchmark closed-loop validation before expanding operators.

> **[HISTORICAL] Status (2026-06-19 — GNN Stage G6 closed-loop validation valid but not
> promoted):** ran the default-off relocation hook through the documented G6
> sequence and full suite with
> `HIER_GNN_RANK=1`,
> `HIER_GNN_MODEL=ml_data/beyondppa_gnn/models/20260619_g4_macro_net_ranker_v1/model.pt`,
> `HIER_GNN_OPERATORS=relocation`, and `HIER_GNN_TOP_K=32`. Four-benchmark
> sequence passed: `ibm10=1.6177` VALID in 50.00s, `ibm01=0.9434` VALID in
> 35.03s, `ibm12=2.1718` VALID in 46.84s, and `ibm17=2.1007` VALID in 49.24s.
> Full `--all` also passed legality: **AVG 1.3676**, 17/17 VALID, 0 overlaps,
> 786.25s. Individual full-suite results were `ibm01=0.9434`,
> `ibm02=1.1415`, `ibm03=1.0170`, `ibm04=1.0267`, `ibm06=1.2185`,
> `ibm07=1.0640`, `ibm08=1.1483`, `ibm09=0.8685`, `ibm10=1.6181`,
> `ibm11=1.0245`, `ibm12=2.1818`, `ibm13=1.0256`, `ibm14=1.6354`,
> `ibm15=1.8447`, `ibm16=1.6671`, `ibm17=2.1054`, `ibm18=1.7187`.
> Because AVG `1.3676` regresses versus the accepted hierarchy baseline
> `1.3631` and runtime is higher, the GNN relocation hook is **not promoted**.
> Keep `HIER_GNN_RANK=0` for production. Next GNN work should diagnose candidate
> ordering/regression by benchmark and operator before any broader integration.

> **[HISTORICAL] Status (2026-06-19 — GNN post-G6 ranking diagnostics started):** added
> `scripts/gnn/diagnose_gnn_ranking.py` to compare deterministic trace order, G3
> MLP, and G4 macro-net rankings against both accepted labels and exact
> proxy-gain labels inside candidate pools. Also added relocation trace fields
> `gnn_score` and `gnn_rank_error` for default-off ranked runs. First diagnostic
> report:
> `ml_data/beyondppa_gnn/diagnostics/20260619_gnn_ranking_value_top4.json`.
> Overall exact-gain recall@4: trace order `0.2513`, G3 MLP `0.5239`, G4
> macro-net `0.6318`; G4 mean best-gain rank `7.37` versus trace order `13.17`;
> G4 top-4 overlap with trace order only `0.1889`. Region swaps show the same
> pattern: trace `0.1964`, G3 `0.4889`, G4 `0.6048`, overlap `0.1856`.
> Interpretation: the G4 model is not merely learning accepted labels while
> missing exact proxy-gain candidates; it improves local exact-gain ranking
> offline but strongly changes the candidate order. The next improvement work
> should use paired heuristic/GNN traces to isolate model-induced distribution
> shift, downstream pass interactions, and runtime budget displacement before
> expanding operators or promoting the hook.

> **[HISTORICAL] Status (2026-06-19 — GNN post-G6 paired trace comparison):** added
> `scripts/gnn/compare_gnn_trace_pairs.py` and collected paired heuristic versus
> `HIER_GNN_RANK=1` traces for `ibm01`, `ibm10`, `ibm12`, and `ibm17` with
> `HIER_GNN_TRACE_MAX_CANDIDATES=64`. Pair reports live under
> `ml_data/beyondppa_gnn/diagnostics/20260619_postg6_*_trace_pair.json`.
> Final proxy deltas were effectively flat: `ibm01=-0.0001`,
> `ibm10=-0.0001`, `ibm12=+0.0001`, `ibm17=+0.0002`. The important signal is
> pass-level behavior: hard propose-all accepts changed `ibm01 0->0`,
> `ibm10 1->0`, `ibm12 4->1`, `ibm17 0->0`; all GNN paired traces had scored
> relocation samples and zero `gnn_rank_error` samples. `ibm12` is now the
> targeted diagnostic benchmark because `HIER_GNN_TOP_K=32` suppresses three of
> four deterministic hard propose-all accepts, and later coldspot/micro-shift
> work only recovers the final proxy to a near tie. Next test: paired `ibm12`
> traces with `HIER_GNN_TOP_K=8` and `16` before retraining or expanding
> operators.

> **[HISTORICAL] Status (2026-06-19 — GNN ibm12 top-k and guarded-prefix variants
> rejected):** ran `ibm12` follow-ups after the paired-trace finding.
> `HIER_GNN_TOP_K=8` was VALID but worse (`2.1878`, delta `+0.0106`) with hard
> propose-all accepts `4->3`; `HIER_GNN_TOP_K=16` matched the same worse result.
> Added default-off `HIER_GNN_PRESERVE_TOP_N` to keep a deterministic prefix and
> fill the tail with GNN-ranked candidates; `HIER_GNN_PRESERVE_TOP_N=12` plus
> `HIER_GNN_TOP_K=4` was also VALID but worse (`2.1898`, delta `+0.0126`) with
> hard propose-all accepts `4->3`, so it is not promoted. A controlled
> `PYTHONHASHSEED=0` rerun did not fully stabilize upstream region-swap counts,
> which means current paired trace reports are diagnostic but not clean causal
> A/Bs. Next improvement work should create a repeatable diagnostic mode or use
> repeated-run variance before retraining the GNN or expanding operators.

> **[HISTORICAL] Status (2026-06-20 — GNN repeatable diagnostic mode added):** added
> `HIER_DIAGNOSTIC_NO_DEADLINES=1`, a diagnostic-only mode that disables local
> hierarchy relief deadlines while preserving production defaults. This fixed
> `ibm12` repeatability: two heuristic no-deadline traces both produced VALID
> final `2.1719`, region swaps `391`, hard propose-all accepts `3`; two
> GNN-ranked no-deadline traces both produced VALID final `2.1707`, region
> swaps `391`, hard propose-all accepts `5`. Pair report:
> `ml_data/beyondppa_gnn/diagnostics/20260620_diag_nodeadline_ibm12_topk32_trace_pair.json`.
> Controlled delta is `-0.0012` proxy with upstream region swaps unchanged. This
> means the GNN ranker has a real positive hard propose-all signal on `ibm12`
> when timing noise is removed; the production G6 regression is more likely a
> budget/timing displacement problem than a pure ranking-quality failure. This
> mode is not production-safe or promoted. Next: run no-deadline paired
> comparisons on `ibm01`, `ibm10`, and `ibm17`, then target the production
> budget interaction if the controlled signal holds.

> **[HISTORICAL] Status (2026-06-20 — GNN additive controlled mode accepted for timed
> smoke):** added default-off `HIER_GNN_EXTRA_TOP_K` so diagnostics can preserve
> deterministic post-swap hard propose-all candidates and append extra
> GNN-ranked candidates. Controlled no-deadline pure GNN top-k 32 results on the
> four-benchmark set were mixed: `ibm01 +0.0000`, `ibm10 +0.0019`, `ibm12
> -0.0012`, `ibm17 +0.0000` for total `+0.0007`. The additive setting
> `HIER_GNN_PRESERVE_TOP_N=16`, `HIER_GNN_TOP_K=8`,
> `HIER_GNN_EXTRA_TOP_K=8` produced `ibm01 +0.0000`, `ibm10 +0.0000`, `ibm12
> -0.0024`, `ibm17 +0.0000`, total `-0.0024`. It preserved the `ibm10`
> deterministic hard propose-all accept (`1->1`) and improved `ibm12` accepts
> (`3->6`). This is not promoted because no-deadline diagnostics disable local
> budgets and the additive mode increases exact checks from 16 to 24 in
> no-gain cases. Next gate: timed production-mode smoke on `ibm10` and `ibm12`
> with additive mode.

> **[HISTORICAL] Status (2026-06-20 — GNN additive timed smoke mixed, not promoted):** ran
> timed production-mode smoke for additive GNN with
> `HIER_GNN_PRESERVE_TOP_N=16`, `HIER_GNN_TOP_K=8`, and
> `HIER_GNN_EXTRA_TOP_K=8` under normal local deadlines. Current timed
> heuristic reruns: `ibm10=1.6184` VALID in 46.27s, `ibm12=2.1857` VALID in
> 44.37s. Additive timed runs: `ibm10=1.6192` VALID in 50.93s, delta `+0.0008`
> with hard propose-all accepts `1->1` and exact checks `15->23`; `ibm12=2.1703`
> VALID in 48.93s, delta `-0.0154` with accepts `4->4` and exact checks `4->4`.
> Decision: keep additive mode default-off and do not run promotion G6 yet. Next
> useful step is a stricter production guard so additive GNN only consumes spare
> post-swap hard-propose budget, or repeated timed smoke to quantify the
> `ibm10` risk.

> **[HISTORICAL] Status (2026-06-19 — GNN Stage G4 offline macro-net ranker accepted,
> default-off):** extended `scripts/gnn/build_gnn_dataset.py` to dataset schema v2
> with MacroDiff+-inspired macro-net graph tensors: net nodes, macro-net
> incidence edges, normalized pin-offset edge features, net degree/fanout, net
> weight, normalized HPWL x/y, and weighted HPWL pressure. Added
> `scripts/gnn/train_gnn_ranker.py`, a small CPU macro-net message-passing ranker,
> and `test/verification/_verify_gnn_ranker.py`. Rebuilt the 4-benchmark G3
> trace dataset as schema v2 with the same 183,452 examples and 1,082 accepted
> labels. Trained artifact
> `ml_data/beyondppa_gnn/models/20260619_g4_macro_net_ranker_v1/` with
> train=`ibm01,ibm17`, validation=`ibm10,ibm12`, hidden=32, 2 graph layers,
> 80 epochs. Overall validation top-4 recall improved from accepted G3 MLP
> artifact `0.5368` and same-split G3 MLP `0.5216` to G4 `0.7922`; overall MRR
> improved from accepted G3 `0.4437` and same-split G3 `0.4199` to G4 `0.6412`.
> Region-swap top-4 recall improved from accepted G3 `0.5328` and same-split G3
> `0.5175` to G4 `0.7904`. CPU scoring smoke on the `ibm10` candidate pool
> scored 37,414 candidates in `0.0336s` to `0.0500s` after warmup. Promotion
> decision remains `default_off`: no inference integration or placement behavior
> changed. The model's proxy-delta correlation is weaker than G3, so G5 must
> use it only for candidate ordering and keep exact-proxy gates authoritative.

> **[HISTORICAL] Status (2026-06-19 — GNN Stage G3 offline baseline accepted,
> default-off):** collected fresh schema-v1 traces for the minimum G3 split:
> `ibm01`, `ibm10`, `ibm12`, and `ibm17`. All four trace runs were VALID:
> `ibm01=0.9435` in 36.38s, `ibm10=1.6172` in 43.90s,
> `ibm12=2.1797` in 55.10s, and `ibm17=2.1038` in 52.14s. The combined
> Stage-G2 dataset contains 4 graphs, 183,452 candidate examples, and 1,082
> accepted labels. Trained the Stage-G3 candidate-feature baselines with
> train=`ibm01,ibm17` and validation=`ibm10,ibm12`; artifact:
> `ml_data/beyondppa_gnn/models/20260619_g3_candidate_baseline_min4/`.
> Overall validation top-4 recall improved from trace order `0.3268` to
> logistic `0.5758` and MLP `0.5368`; overall MRR improved from `0.2611` to
> logistic `0.4772` and MLP `0.4437`. Region swaps met the G3 acceptance gate:
> top-4 recall improved from trace order `0.3210` to logistic `0.5721` and MLP
> `0.5328`, with MLP MRR `0.4389` versus trace-order `0.2557`. Promotion
> decision remains `default_off`: this proves offline label learnability and
> does not enable inference or change placement. Next stage is G4 graph
> extension and first GNN ranker training.

> **[HISTORICAL] Status (2026-06-19 — GNN Stage G2 graph dataset builder):** added
> `scripts/gnn/build_gnn_dataset.py`, a deterministic schema-v1 JSONL-to-PyTorch
> dataset builder. The output contains one graph per benchmark with hard macro,
> soft macro, and inferred-cluster nodes; netlist clique, macro-cluster
> membership, and spatial-neighbor edges; and a stacked candidate table with
> source/target node references, 27 candidate features, accepted labels,
> proxy-delta labels with known-mask, rejection ids, and trace provenance.
> Added the dataset contract in
> `docs/ml_nn/beyondppa_results/gnn_dataset_schema.md` and the verifier
> `test/verification/_verify_gnn_dataset_builder.py`. Smoke build from the
> Stage-G1 `ibm01` trace produced 1 graph, 4526 examples, and 78 accepted
> labels; repeated builds were tensor-identical. Verification: broad
> `py_compile` over `src`, `scripts`, and `test/verification` passed
> (with an unrelated existing regex escape warning in
> `scripts/generate_macro_placement_tcl.py`); the G2 verifier passed.

> **[HISTORICAL] Status (2026-06-19 — GNN Stage G1 trace completeness and schema v1):**
> completed candidate-level trace coverage for the active hierarchy GNN data
> path without changing placement behavior. `log_gnn_event()` now emits
> `schema_version=1`; `hier_decompression_candidate` records cluster expansion,
> axis scale, hierarchy-quality delta, exact proxy delta when scored, accepted
> flag, and rejection reason; `hier_swap_candidates` records sampled hard/hard,
> hard/soft, and soft/soft region-swap candidate pools with legality, region,
> score, proxy delta, accepted flag, and rejection reason; and
> `hier_coldspot_candidate` records skipped, missing, rejected, and accepted
> coldspot proposals with selected-cluster metadata, field gap, quality delta,
> and proxy delta. Added the schema contract in
> `docs/ml_nn/beyondppa_results/gnn_trace_schema.md`. Verification:
> `py_compile` passed; focused structural tests passed (`4 passed`);
> `_verify_score_region_swaps.py` passed on ibm01/04/10;
> `_verify_coldspot_kick.py ibm10` passed; GNN trace smoke on `ibm01` was
> VALID with proxy `0.9435` and wrote 1539 schema-v1 events with the new
> decompression, swap, and coldspot candidate event families.

> **[HISTORICAL] Status (2026-06-18 — BeyondPPA structural objective and GNN trace logging
> integrated, defaults unchanged):** added deterministic edge-keepout,
> grid-alignment, notch, and combined structural metrics in
> `src/placer/local_search/structural_fields.py`; and integrated the structural term
> into existing hierarchy relocation candidate ordering behind
> the `HIER_OBJECTIVE_STRUCTURAL_WEIGHT=0.0` constant. This is not a
> second BeyondPPA path: legality, fixed-macro immobility, bounds,
> hierarchy-region constraints, hierarchy-quality gates, and exact-proxy gates
> still decide accepted moves. Added opt-in GNN JSONL tracing through
> `src/placer/local_search/gnn_trace.py` with events for relocation candidates,
> accepted relocation labels, hierarchy pass summaries, and final placement
> summaries. Verification: `py_compile` passed; focused structural tests passed
> (`4 passed`); GNN trace smoke on `ibm01` was VALID with proxy `0.9435` and
> wrote 24 trace events. A later default-off full sweep produced **AVG 1.3626**,
> 17/17 VALID, 0 overlaps, 595.52s, in-family with the accepted 1.3631 result.
> Because the worktree also contains unrelated local changes, do not attribute
> the small delta solely to the structural/GNN logging work without a clean A/B.

> **[HISTORICAL] Status (2026-06-18 — Stage 3 micro-shift replay stack accepted,
> `--all` avg = 1.3631):** promoted two exact-gated replay passes that rerun
> `_micro_shift_polish()` after region swaps and again after coldspot tightening.
> Full `uv run evaluate src/main.py --all`: **AVG 1.3631**, 17/17 VALID,
> 0 overlaps, 602.76s. Individual gates were: post-swap replay alone
> **AVG 1.3650**, post-coldspot replay alone **AVG 1.3645**, and both together
> **AVG 1.3631**. The stack wins on average and improves the largest late cases
> (`ibm15=1.8355`, `ibm17=2.0977`, `ibm18=1.7162`), with a small `ibm14`
> tradeoff (`1.6272` vs post-swap alone `1.6256`). Rejected companion gates:
> combined congestion-density hard relocation was only **AVG 1.3704** and
> emitted an invalid-sqrt warning on `ibm01`; deterministic hot coldspot
> selection regressed to **AVG 1.3714**. Both rejected gates were removed from
> the production pipeline.

> **[HISTORICAL] Status (2026-06-18 — Stage 3 DREAMPlace selector rejected at smoke):**
> implemented a bounded grouped-DREAMPlace selector that tried candidate
> `(group_weight, seed)` variants before relief, legalizing and scoring each
> with exact proxy plus hierarchy-quality blend. `ibm10` group-weight variants
> `(8,1000),(6,1000),(10,1000)` selected the existing `(8,1000)` baseline:
> candidate proxies `1.7942`, `1.9100`, `1.9177`; final smoke **1.6179** VALID
> versus default **1.6165** VALID. Seed variants `(8,1000),(8,1001),(8,1002)`
> also selected the baseline: candidate proxies `1.7942`, `1.9114`, `1.9396`;
> final smoke **1.6168** VALID. Because the selector added runtime and found no
> better `ibm10` candidate, the selector code path was removed and no `--all`
> was run.

> **[HISTORICAL] Status (2026-06-18 — Stage 2 micro-shift + anisotropic decompression
> accepted, `--all` avg = 1.3707):** promoted exact-gated one/two-grid-cell
> micro-shift polish for hot hard and soft macros inside their hierarchy regions
> (`HIER_MICRO_SHIFT=1`, radius 2, top 96, min gain `1e-5`) and promoted
> anisotropic cluster decompression (`HIER_DECOMPRESS_ANISO=1`). Full
> `uv run evaluate src/main.py --all`: **AVG 1.3707**, 17/17 VALID, 0 overlaps,
> 563.00s. Micro-shift alone was **AVG 1.3718**, 17/17 VALID, 0 overlaps,
> 591.59s, so anisotropic decompression adds a small but real full-suite gain.
> Micro-shift + fused swap prescreen was **AVG 1.3780**, 17/17 VALID,
> 0 overlaps, 568.20s, so the fused prescreen code path was removed. Prior
> individual checks: fused swap prescreen alone **AVG 1.3997** (regression),
> anisotropic decompression alone **AVG 1.3932** (small heat-baseline win but
> worse than micro-shift), and all three together **AVG 1.3777**. Pre-swap hard
> propose-all relocation on top of the accepted combo regressed the `ibm10`
> smoke to **1.6405** VALID, so that pre-swap propose-all plumbing was removed;
> accepted post-swap propose-all relocation remains active.

> **[HISTORICAL] Status (2026-06-16 — post-swap soft polish accepted, `--all`
> avg = 1.3947):** promoted ordinary post-swap soft relocation with
> `HIER_POST_SOFT_RELOC=1`, `HIER_POST_SOFT_RELOC_MIN_GAIN=0.0005`, and
> lowered the post-swap hard propose-all margin to
> `HIER_RELOC_PROPOSE_MIN_GAIN=0.0005`. This is not the rejected soft
> propose-all path; it is sequential exact-gated `_soft_relocation_moves()` after
> swaps. Full `uv run evaluate src/main.py --all`: **AVG 1.3947**, 17/17 VALID,
> 0 overlaps, 534.30s. Margin comparison on ibm10/12/15/17/14/18 favored
> `0.0005` over `0.00075` on every targeted case. Main full-run gains vs the
> prior accepted 1.3974: ibm15 1.8894→1.8743, ibm17 2.2096→2.2045, ibm18
> 1.7832→1.7772, ibm14 1.6784→1.6733, ibm12 2.2514→2.2496, plus smaller wins on
> ibm01/06/07/09/11/16.

> **[HISTORICAL] Status (2026-06-16 — post-swap hard propose-all polish accepted, `--all`
> avg = 1.3974):** kept `HIER_RELOC_PROPOSE_ALL=0` for the pre-swap hard
> relocation loop, but promoted CUDA-only
> `HIER_POST_RELOC_PROPOSE_ALL=auto` after region swaps with footprint-averaged
> hard-macro field ranking, `HIER_POST_RELOC_PROPOSE_TOP_M=16`, and
> `HIER_RELOC_PROPOSE_MIN_GAIN=0.001`. Full
> `uv run evaluate src/main.py --all`: **AVG 1.3974**, 17/17 VALID, 0 overlaps,
> 526.21s. Accepted post-swap hard moves were sparse and helped the congestion
> cases without reintroducing the earlier pre-swap basin regression: ibm10
> 1.6506→1.6485 and ibm12 2.2535→2.2514; other cases were neutral or within
> run noise.

> **[HISTORICAL] Status (2026-06-16 — connectivity legalize order accepted, `--all`
> avg = 1.3978):** promoted `HIER_LEGALIZE_CONNECTIVITY_ORDER=1`, which keeps
> cluster-consecutive legalization but orders members by connectivity-pressure x
> area instead of area alone. Full
> `uv run evaluate src/main.py --all`: **AVG 1.3978**, 17/17 VALID, 0 overlaps,
> 518.68s; beats RePlAce avg 1.4578 by +4.1%. Key gains vs the prior hierarchy
> result 1.4452 are broad and congestion-led: ibm12 2.3297→2.2535, ibm17
> 2.2374→2.2109, ibm15 1.9494→1.8894, ibm14 1.6991→1.6790, ibm18
> 1.7869→1.7832, and ibm10 1.6759→1.6506. The rejected Stage-1 bundle
> (`HIER_RELOC_PROPOSE_ALL=auto`, `HIER_SOFT_RELOC_PROPOSE_ALL=auto`,
> connectivity order, and `HIER_SS_SWAP_MAX_SCORES=12000`) regressed badly:
> **AVG 1.6000**, 17/17 VALID, 0 overlaps, 1029.78s. Follow-up full ablations:
> hard propose-all only **AVG 1.4019** / 546.26s; hard top-M 16 **AVG 1.4066** /
> 545.26s; hard congestion-pass top-32 hot macros **AVG 1.4030** / 526.26s; soft
> propose-all only **AVG 1.5650** / 996.04s. Soft propose-all and the score-cap
> experiment were removed. Pre-swap hard propose-all remains diagnostic-only and
> default off.

> **[HISTORICAL] Status (2026-06-16 — soft-swap breadth tuning accepted, `--all`
> avg = 1.4452):** kept owned/bridge soft classification, congestion-expanded
> regions, exact-gated cluster decompression, proxy-aware coldspot tightening,
> per-operator region-swap controls, strict hard-swap legality, and best-state
> rollback; then raised the default soft swap candidate count
> `HIER_SOFT_SWAP_K` from 24 to 48. Full
> `uv run evaluate src/main.py --all`: **AVG 1.4452**, 17/17 VALID, 0 overlaps,
> 520.08s; beats RePlAce avg 1.4578 (+0.9% vs RePlAce). Net gain vs the prior
> tuned region-swap `--all` 1.4471 is -0.0019, led by the intended congestion
> targets: ibm12 2.3454→2.3297, ibm17 2.2481→2.2374, ibm15 1.9555→1.9494, plus
> ibm10 1.6836→1.6759 and ibm16 1.7376→1.7322. Main regressions to watch next:
> ibm18 1.7761→1.7869, ibm14 1.6931→1.6991, and ibm11 1.1305→1.1326. Current
> bottlenecks remain congestion-heavy: ibm12 2.3297 (cong 2.983), ibm17 2.2374
> (cong 2.897), ibm15 1.9494 (cong 2.390).

> **HISTORICAL HEADLINE (2026-06-14 — cluster-coherent LSMC kicks were shipped
> in the then-current proxy path, then deleted on 2026-06-16):
> paired multi-seed `--all` ON vs OFF, ON wins 3/3.** OFF (random kicks) means
> 1.1206 → ON (pure cluster kicks, p=1.0, both modes) means **1.1183**
> (Δ = **−0.0023**), all six runs 17/17 VALID / 0 overlaps. Per-seed:
> seed 0 1.1228→1.1174 (−0.0054, 16/17 benchmarks improved), seed 42
> 1.1192→1.1183 (−0.0009), seed 44 1.1199→1.1192 (−0.0007). Seed 0 was a
> favorable draw; steady-state gain ~−0.0008/seed, so the mean is partly
> noise-adjacent but consistently negative with **0 regressions** (the kick is
> behind the exact post-descent accept gate). **Mechanism:** in these flat
> netlists hard macros connect THROUGH standard cells, not to each other (ibm01
> has 0 hard-to-hard nets), so "subsystems" are derived via union-find over the
> bipartite hard↔soft graph on low-fanout nets (`local_search/clusters.py`).
> LSMC then kicked a whole cluster as a unit — `gather` (collapse members to one
> anchor, legalizer packed them) or `translate` (rigid relocate) — instead of
> scattering random macros. It was enabled in the then-current `src/main.py`
> through `_enable_cluster_kick_defaults` and `GPU_EXPLORE_CLUSTER_*`; those
> integration points were later deleted with the proxy path.
> isolation-harness `LSMC_ISOLATE=1` confirmed cluster kicks beat random
> 6/6 from an identical incumbent (−0.0053 avg, congestion-driven). The old
> `test/verification/_verify_cluster_kick.py` verifier was deleted with that
> code.
>
> **HEADLINE (2026-06-14, current `varrahan` 786e749): `--all` avg = 1.1203,
> 17/17 VALID, 0 overlaps, 2806s (~47min), default seed.** This is the
> simplified pipeline: cong-grad spine deleted, multi-seed LSMC explore (other
> agent), all Stage 4 prunes (multi-seed 2-opt / 5c / 8 / cong-grad), dead code
> removed. +0.0034 vs the prior high-water 1.1169 (multi-seed-2-opt prune, my
> earlier branch) — a deliberate near-noise score cost for a much leaner
> codebase; not a clean A/B since the LSMC engine differs. Still −23% vs RePlAce
> (1.4578) and −20% vs the DREAMPlace leaderboard (1.4076), beating both on every
> benchmark. Per-bench: ibm01 0.9233, ibm02 1.1496, ibm03 0.9864, ibm04 1.0002,
> ibm06 1.1855, ibm07 1.1721, ibm08 1.1527, ibm09 0.8384, ibm10 1.0903, ibm11
> 0.9358, ibm12 1.3145, ibm13 0.9809, ibm14 1.2091, + back four. Single-seed, so
> ±~0.002 noise.

> **[HISTORICAL] Status (2026-06-14 — ALL cong-grad code DELETED (on the reverted multi-seed
> LSMC base, e2c8d04).** Removed the congestion-gradient spine (phases 1-3, 5b
> DP-rescue, P7 DP-chains; 136 lines) from `macro_placer.py` + the orphaned
> `placer/perturb/` package and its test. `rng_cong` kept (Phase 9 tie-breaks);
> the `_remember_lsmc_seed` machinery stays (it seeds from random-noise restarts
> + r2-best, not cong-grad). Verified: ibm04 1.0002 VALID, no cong-grad lines,
> multi-seed LSMC seeds from survivors `{post/pre-r2-best, random noise}`.
> **Accepted regression, by directive:** cong-grad was net-positive on the
> reverted code (+0.0134 over ibm01/04/09/12/17/18 single-seed, worst ibm17
> +0.0153) and the regression is somewhat larger on this multi-seed base
> (ibm04 1.0002 vs cong-on ~0.9947) because cong-grad indirectly improves the
> seed pool. Deleted for pipeline simplification; git-revertable. A full 2-seed
> `--all` should size the true cost before relying on the number.

> **(historical, my earlier branch) Stage 4 MAJOR CLEANUP: pruned phases DELETED from
> source.** Removed 272 lines from `macro_placer.py` + the whole
> `local_search/workers.py` (and the `mp` import + `__init__` export): the
> multi-seed 2-opt phase, Phase 5c (wide-from-best), and Phase 8 (TOP-K
> cong-grad chains), plus their `PRUNE_*` flags. These were all pruned-by-
> default already, so deletion is behaviorally equivalent — verified on ibm01/
> 04/12/18 (seed1) within ±0.0003 of the flag-pruned references, all VALID.
> Shipped `--all` stays ~1.1170–1.1176 (the all-pruned value); the lone real
> score win in Stage 4 was multi-seed 2-opt removal (1.1169), 5c/8 were
> near-noise simplification. NB: the "restore via PRUNE_*=0" knobs no longer
> exist; ARCHITECTURE.md / README phase lists for these three are now superseded.

> **[HISTORICAL] Status (2026-06-14 — Stage 4: Phase 5c (wide-from-best) PRUNED by default
> for pipeline simplification, despite a near-noise score cost.** Pruned `--all`
> = **1.1170** (seed1, full 17/17). The paired gate actually favored KEEPING 5c
> (keep seed1 1.1156 / seed2 1.1191 vs prune seed1 1.1170; 5c does real work on
> ibm09/12/17), so its "pure insurance" label in older notes was stale. Pruned
> anyway by directive to keep the pipeline lean; `PRUNE_P5C=0` restores it.
> So 1.1156 (5c kept) is the lower achievable; the shipped lean default is 1.1170.
> Gate stopped after 3/4 runs (decision was made); logs `ml_data/compare/stage4p5c_*`.

> **[HISTORICAL] Status (2026-06-13 — Stage 4: multi-seed 2-opt PRUNED by default; NEW BEST
> `--all` 1.1169):** the pre-R2 multi-seed 2-opt phase is now skipped by default
> (`PRUNE_MULTISEED_2OPT=0` restores it). Paired gate keep-vs-prune,
> full-stack (dp=17 both arms): seed1 1.1175→1.1169 (−0.0006), seed2
> 1.1229→1.1178 (−0.0051) — 2/2 prune-wins, mean −0.0029, and faster
> (seed2 2816s vs 2926s). The phase was net-harmful: it steers R2 into worse
> basins on more benchmarks than it helps. Big winners from removal: ibm04
> (−0.020/−0.026), ibm12 (−0.015/−0.023), ibm17 (−0.024 s2), ibm11/13.
> Consistent small regressions: ibm01 +0.007, ibm02 +0.010, ibm05 +0.006;
> ibm14 high-variance (+0.026 s1 / −0.004 s2). Net win on the scored 17-average
> on both seeds → shipped. (The per-benchmark split flags a future targeted
> question: why does 2-opt help ibm01/02/05 but hurt ibm04/12/17? — a conditional,
> not a blanket phase.) Logs `ml_data/compare/stage4_*`. Earlier ibm04 smoke
> (0.9850) was directionally right but the early 3-benchmark gate peek
> (ibm01/02/03 all regress) was a small-sample trap — the full 17 reversed it.
> Next Stage 4 target: the cong-grad restart phases.

> **[HISTORICAL] Status (2026-06-13 — Stage 2b kick pre-screen SHIPPED AS DEFAULT
> (PRESCREEN=8); on-arm best 1.1176 is the NEW BEST `--all`):** each LSMC
> iteration now scores a batch of kicks (`GPU_EXPLORE_PRESCREEN`, default 8)
> and descends only the best one — the cuGenOpt evaluate→reduce→descend-one
> pattern at the kick level, since descent dominates iteration cost.
> **Full-stack paired gate (DP+ML active, dp=17 both arms; B8 vs B1=2a
> behavior): seed1 1.1198→1.1176 (−0.0022), seed2 1.1237→1.1219 (−0.0018) —
> 2/2 wins, mean −0.0020.** Accepts roughly doubled (seed1 8→17, seed2 11→16)
> and land as broad small gains (ibm12 −0.0091, ibm13 −0.0065, ibm15 −0.0066,
> ibm09 −0.0061), not one lucky basin. B8 is also marginally faster
> (3050s vs 3107s) — descending one well-chosen kick beats descending several
> mediocre ones. Lone seed1 regression ibm16 +0.0046 is upstream R2 timing
> jitter (final-phase gate can't itself regress). `PRESCREEN=1` is
> bit-equivalent to the prior 2a default. Logs `ml_data/compare/stage2b_*`.
> Default-path ibm01 smoke: 0.9134 VALID. Next: 2c options re-analysis.

> **Historical status (2026-06-12 evening — Stage 2 LSMC exploration shipped as
> default in the old proxy path;
> seed-1 on-arm avg 1.1194 is the NEW BEST `--all`):** the post-R2 LSMC
> kick/descent/accept phase (`local_search/lsmc_explore.py`, hook as the FINAL
> quality phase in `macro_placer.py`) was default-on under CUDA
> (`GPU_EXPLORE` unset/auto; opt out with 0), kick=0.02, 30s slice, in that
> deleted proxy path —
> exactly the measured gate config. **Full-stack paired gate (DP + ML filter
> verified active in every log; user-authorized 2-seed bar): seed1 off 1.1245 /
> on 1.1194 (−0.0051), seed2 1.1275 / 1.1242 (−0.0033) — 2/2 wins, mean
> −0.0042.** On-arm totals 3107s/3024s (~51min, watch the 1h cap under
> contention; the per-bench budget clamp shrinks the slice as t_one_score
> rises). Logs `ml_data/compare/` (stage2 full-stack set). Two structural
> lessons baked into the design: (1) acceptance must gate the FINAL score —
> pre-R2 and pre-post-soft hooks both accepted states that lost after later
> refinement; (2) an earlier gate run was invalidated because the code-pinning
> worktree silently lacked gitignored assets (dreamplace_build, ml_data) — DP
> and the ML filter were OFF (+0.018 on the off-arm); symlink assets into any
> worktree that runs the placer and grep logs for "DREAMPlace launched" + "ML
> filter on". Degraded-stack pair (DP/ML absent): explore was worth −0.0067
> there — relevant if eval hardware ever lacks the full stack.

> **[HISTORICAL] Status (2026-06-12 — GPU staged rollout: Stage 0 re-baseline + Stage 1
> propose-all A/B done; verdict WASH, stays opt-in; see docs/gpu/GPU-ops.md and
> ISSUES.md S17):**
>
> **Stage 0 re-baseline (2026-06-11): avg 1.1243, 17/17 VALID, 2679s** — same
> placer as the 1.1252 record within single-seed noise, locked as the pre-GPU
> reference. CUDA relocation diagnostic PASS (torch 2.10.0+cu128, cuda_delta on
> cuda:0, exact parity 1.541e-07); DREAMPlace dpenv healthy (torch 2.4.1+cu121,
> **built sm_89-only** — rebuild required on any other GPU arch); numba 0.65.1
> present. Hardware unchanged: 1× RTX 4050 Laptop 6GB. The new multi-GPU
> machines are not reachable from this box yet — second half of Stage 0
> (inventory, DP rebuild, re-baseline there) pending access.
>
> **Stage 1 (2026-06-12): `RELOC_PROPOSE_ALL=auto` paired multi-seed A/B —
> WASH, stays opt-in.** Both verifiers PASS (in-loop scorer-vs-exact delta
> 1.4e-11). Paired same-box sequential `--all` runs, seeds 1/2/3
> (logs `ml_data/compare/all_20260612_propall_*`): seed1 off 1.1231 / auto
> 1.1237 (+0.0090 cum), seed2 1.1277 / 1.1280 (+0.0047), seed3 1.1250 / 1.1246
> (−0.0076). Mean +0.0020 cumulative (+0.0001/bench), 2/3 seeds worse — far
> from the S10 ship bar (3/3, −0.0041 mean). No `--all` wall-time win either
> (2415 vs 2408s, 2454 vs 2492s): the budget allocator reabsorbs per-benchmark
> speedups (ibm01 alone runs 77.6s vs 109.4s at identical proxy 0.9146).
> 10–13 of 17 benchmarks per seed are bit-identical between arms; divergences
> are deterministic but seed-dependent in sign (ibm11 −0.0103/−0.0047/+0.0030;
> ibm18 +0.0188 on seed1 only, replay-confirmed bit-exact 1.3967 under clean
> conditions — a genuinely worse basin, not a budget artifact). Caveats noted:
> the seed-1 pair straddles dead-code commit 04ae002 (behavior-neutral — 10/17
> bit-identical across the boundary); seed-1 auto crossed a laptop sleep (wall
> 35233s) with per-benchmark monotonic budgets unaffected. **Takeaway for
> Stage 2:** the GPU pool-scoring machinery is validated and fast; the ±0.02
> per-benchmark policy divergence is exactly what an exact-gated multi-candidate
> selector can harvest. Next: Stage 2 exploration engine (GPU_EXPLORE) per
> docs/gpu/GPU-ops.md §2–3.

> **[HISTORICAL] Status (2026-06-11 — ML hard-relocation ranker connected as production
> default, ISSUES.md S10):** The trained XGBoost filter was validated 2026-06-05
> (equal-budget net ≈ −0.008/10, no robust regressions) but had been left
> opt-in — no env var was ever set in the run path, so every evaluate run since
> was pure-heuristic and the ranker never loaded. `src/main.py` now enables
> config B by default (wide-32 hard-relocation pool, ranker exact-scores 16,
> model `clean-wide32-holdout-ibm13-001`) when no `ML_*` env var is set; any
> preset `ML_*` var, a missing model file, or missing `xgboost` falls back to
> the prior pure-heuristic path. The pipeline logs
> `R2 hard relocation ML filter on (pool=32, top_k=16)` when active. Verified:
> `test/verification/_verify_ml_filter_wiring.py` (defaults set, bank loads,
> 32→16 filtering, opt-out respected) + ibm01 end-to-end ×2 (proxy **0.9146**
> VALID, 71s; DP-fixed --all reference was 0.9215).
>
> **`--all` re-baseline (same day): avg 1.1252 — all 17 VALID / 0 overlaps,
> 2337s — NEW BEST** (vs 1.1272 DP-fixed reference, Δ −0.0020/17 cumulative
> −0.0345). Per-benchmark: ibm01 0.9146, ibm02 1.1621, ibm03 0.9896, ibm04
> 1.0137, ibm06 1.2059, ibm07 1.1819, ibm08 1.1543, ibm09 0.8409, ibm10 1.0945,
> ibm11 0.9354, ibm12 1.3100, ibm13 0.9988, ibm14 1.2133, ibm15 1.2130, ibm16
> 1.1608, ibm17 1.3502, ibm18 1.3885. 13 of 17 improved; movers beyond the
> ±0.005–0.01 single-run noise: ibm11 −0.0094, ibm13 −0.0083 (the model's
> held-out benchmark), ibm16 −0.0088, ibm17 −0.0078 vs ibm06 +0.0166, ibm10
> +0.0133, ibm18 +0.0085. NB ibm10 regressing here despite being the robust S10
> compare win underlines that single reps are noisy — which the paired
> multi-seed below resolved.
>
> **Multi-seed paired confirmation (same day): the filter gain is REAL — mean
> Δ −0.0041, filter wins 3/3 pairs.** Six sequential same-day `--all` runs,
> paired ON-vs-OFF per seed (OFF = `ML_FILTER_OPERATORS=""`, the exact prior
> production path; logs
> `ml_data/compare/all_20260611_{on,off}_s{def,43,44}.log`): default seed
> 1.1252 vs 1.1303 (−0.0051); seed 43 1.1248 vs 1.1292 (−0.0044); seed 44
> **1.1235** vs 1.1264 (−0.0029). ON mean 1.1245, OFF mean 1.1286; all 6 runs
> 17/17 VALID / 0 overlaps / 2219–2504s. ON runs cluster tighter (spread
> 0.0017 vs 0.0039 OFF). Note the same-day OFF baseline (1.1303) was worse
> than the prior-day 1.1272 reference — day-to-day machine drift exceeds the
> effect size, so only paired same-day runs can resolve filter-sized deltas;
> never compare against a different-day reference. **Headline stays the
> default-seed 1.1252** (the harness runs the default seed; seed 44's 1.1235
> is seed variance, not a selectable config). Next lever: budget-aware pruning
> (`ML_FILTER_TOP_K` sweep under time pressure only).

> **[HISTORICAL] Status (2026-06-10 — fixed a silent DREAMPlace ABI break, ISSUES.md S16):**
> **Avg 1.1272 — all 17 VALID / 0 overlaps, 2645s (~44 min) — NEW BEST.** Root
> cause: the DP bridge launched DREAMPlace with the repo `.venv`, which was upgraded
> to **Python 3.14** for numba (S13), but DP's compiled extensions are **cpython-310**
> — so `import dreamplace.ops.place_io.place_io_cpp` failed with `ModuleNotFoundError`
> ~4s after launch, and the bridge logged it as a benign "not ready (elapsed=4.4s);
> killing subprocess." Net effect: **DREAMPlace produced ZERO seed basins on every
> benchmark from S13 onward** — the multi-seed 2-opt ran single-basin, and the
> **1.1379 @2117s (S14) headline was a DP-OFF run.** Fix (one spot): `VENV_PYTHON`
> now prefers the DP build env's interpreter (`dreamplace_build/dpenv/bin/python`,
> 3.10), falling back to `.venv` only if dpenv is absent. With DP restored: `--all`
> **1.1379 → 1.1272 (−0.0107)**, **51/51 DP launches ready / 0 failures**, DP basins
> used (not pruned) in the 2-opt on all 17. The +528s wall (2117→2645) is the DP
> candidate-scoring + DP-basin 2-opt work — comfortably under the 3300s soft cap.
> Per-benchmark (DP-fixed --all): ibm01 0.9215, ibm02 1.1618, ibm03 0.9939, ibm04
> 1.0159, ibm06 1.1893, ibm07 1.1832, ibm08 1.1619, ibm09 0.8421, ibm10 1.0812,
> ibm11 0.9448, ibm12 1.3113, ibm13 1.0071, ibm14 1.2206, ibm15 1.2198, ibm16
> 1.1696, ibm17 1.3580, ibm18 1.3800. Beats RePlAce (1.4578) by 22.7% and the
> leaderboard (1.4076) by ~19.9%, on every benchmark. **Lesson:** single-benchmark
> spot-checks (ibm12/17/18) read this as noise-level (one even regressed) — the
> −0.0107 only resolves above the noise floor at the 17-benchmark aggregate.
>
> **Also this session — LAHC disproven (reverted).** Late-Acceptance Hill Climbing
> on the 2-opt-on-winner (env `LAHC`=history length): strictly worse 2-opt on
> ibm12/17/18 (ibm17 1.7299→1.7401 at L=1000, →1.7328 at L=50 — i.e. tighter L only
> recovers greedy, never beats it), ~85% accept rate = random-walk on the plateau.
> The deadline-bound 2-opt converges fast to a strong basin min, so non-monotonic
> acceptance just wastes budget (consistent with the S1 basin-hopping disproof).
> Reverted in full; the accept gate stays strict-improvement greedy.

> **[HISTORICAL] Status (2026-06-07 — hand-JIT the scoring hot paths, ISSUES.md S14):**
> **Avg 1.1379 — all 17 VALID / 0 overlaps, 2117s (~35 min).** cProfile (post-numba)
> found three vectorized-numpy scoring functions with no JIT path dominating:
> `_apply_macro_routing` (per-cell macro-routing scatter), `_macro_occ` (density
> footprint), `_compute_per_net_hpwl_subset` (per-net HPWL). Added explicit-loop
> numba versions of each, **bit-exact** (stress Hcong/Vcong ~1e-15, density Δ=0,
> swap Δ=0). `--all` **2563s → 2117s (~17% faster; ~39% vs no-numba 3486s)** — pure
> speed (avg unchanged, JITs bit-exact), a big margin under the 1 h cap. ibm13:
> 200s (no-numba) → 162s (numba) → 119s (+3 JITs). Remaining: `_resmooth_h/v`
> (cumsum-based, deprioritized).

> **[HISTORICAL] Status (2026-06-07 — numba JIT was silently disabled; re-enabled, ISSUES.md S13):**
> **Avg 1.1380 — all 17 VALID / 0 overlaps, 2563s (~43 min).** cProfile found the
> routing-apply (half the runtime) running the slow **numpy fallbacks** because
> **numba was not installed** (`HAS_NUMBA=False`): numba is in `v2/requirements.txt`
> but NOT `pyproject.toml`, so `uv sync` alone never installed it — and the placer
> falls back silently. **Every prior measurement this session ran in slow mode.**
> Installing numba 0.65.1 (supports py3.14): `--all` **3486s→2563s (~26% faster)**
> *and* **1.1403→1.1380** (freed speed → more refinement on the deadline-bound
> benchmarks). Without numba the placer still runs but ~25% slower (~58 min, near
> the 1 h cap) — a src/utils/config.py warning now fires when numba is missing. **Action: the
> eval env must install `v2/requirements.txt` (or numba must reach `pyproject`)** to
> realize 1.1380; the graceful fallback is 1.1403. Both still beat RePlAce (1.4578)
> and the leaderboard (1.4076) on every benchmark.

> **[HISTORICAL] Status (2026-06-07 — spend freed budget on soft_relocation, ISSUES.md S12):**
> **Avg 1.1403 — all 17 VALID / 0 overlaps, 3486.53s (new best).** With each
> soft-reloc group ~37% cheaper (S11 WL prefilter), the freed budget is spent on the
> score MVP's per-macro depth: **soft_relocation n_targets 24 → 32**. Validated:
> ibm13 −0.012, ibm17 −0.0054, ibm15 neutral; widening `top_hot` too over-widens
> (rejected). Trajectory: 1.1500 → 1.1496 (2-opt cuts) → 1.1423 (soft-reloc
> prefilter) → **1.1403**. Also tried (NEGATIVE, shelved env-gated off):
> **adaptive per-pass budget control** — yield-weighted deadline caps were
> consistently worse on deadline-bound benchmarks (shrink → early termination;
> boost → saturates), so the static caps + skip-if-empty stay. Env knobs:
> `SOFT_TGT` / `SOFT_HOT`, `ADAPTIVE_BUDGET` (off).

> **[HISTORICAL] Status (2026-06-06 — scoring-cost reduction in 2-opt + soft-relocation, ISSUES.md S11):**
> **Avg 1.1423 — all 17 VALID / 0 overlaps, 3433.76s (new best, beats prior 1.1500
> by −0.0077).** Three accept-gate-safe WL-delta prefilter / budget cuts (only
> change which candidates get exact-scored; every accept still validated):
> 1. **soft_relocation WL-delta prefilter = 1e-4** — skips ~37% of `_trial_at_soft`
>    (~10% of total scoring time). soft_relocation commits best-per-group, so
>    skipping non-best improving candidates is free. ibm15 replicates: 1.2136 vs
>    1.2219 off (−0.008, and faster); ibm13 no regression. The biggest single win.
> 2. **soft_2opt WL-delta prefilter 0.01 → 3e-4** — rejects ~23% of
>    `score_swap_soft`, drops ~0.2% of improving swaps (calibrated).
> 3. **R2-cleanup hard_2opt k_neighbors 20 → 16** — frees time for the productive
>    soft passes; multi-seed 2-opt-on-winner stays k=20 (S2).
>
> Motivated by per-operator profiling: hard_2opt was ~48% of scoring time for the
> smallest per-move gains; soft_relocation (28%) is the score MVP. The freed budget
> converts to deeper refinement on the deadline-bound benchmarks (1.1500 → 1.1496
> 2-opt-only → 1.1423 + soft-reloc). A hard_2opt WL prefilter was built but shipped
> OFF (WL doesn't separate hard spatial-kNN swaps — calibrated). New bit-exact
> scorer methods `wl_delta_swap` / `wl_delta_move_soft` (verified). Env knobs:
> `SOFT_RELOC_WL_PREFILTER` / `SOFT_2OPT_WL_PREFILTER` / `HARD_2OPT_K` /
> `HARD_2OPT_WL_PREFILTER`. Tests: `_verify_wl_delta_swap.py`,
> `_verify_wl_delta_move_soft.py`.

> **[HISTORICAL] Status (2026-06-04 — readability refactor, no algorithm change):**
> **Avg 1.1500 — all 17 VALID / 0 overlaps, 3300.10s.** Pure code-simplification
> pass, validated non-degrading via `--all`. No move-generation, scoring, or RNG
> logic was touched, so the 1.1500 is a favorable full-budget run within normal
> timing variance — not an algorithmic gain. Changes:
> 1. **ML-trace consolidation.** The per-candidate congestion/density feature
>    lookups scattered through `two_opt`/`relocation`/`hard_soft`/`soft_moves`
>    (5 setup blocks + ~30 verbose `float(field[ri,ci]/max) if … else 0.0`
>    ternaries) collapsed into a `TraceFields` helper in `ml/data_collection.py`.
>    Data collection stays fully functional and byte-identical (pinned by
>    `test/verification/test_trace_fields_equivalence.py`). Search files −111 lines.
> 2. **`separation_matrices()` helper** (`placer/geometry.py`) replaces the
>    7×-duplicated `(sizes[:,0:1]+sizes[:,0:1].T)/2` across the legalizer + 2-opt
>    / relocation passes.
> 3. **`place()` setup extracted** into `_effective_budget()` and
>    `_launch_dreamplace_seeds()` methods (the cong-grad phase descent left intact).
> 4. pyflakes now clean across `src`; fixed `congestion_gradient.py` annotations.
>
> Verification note: ibm01's single-bench score is **timing-sensitive**
> (0.9084–0.9094 depending on wall-time via deadline-gated R2 passes), so it is
> not a bit-identity oracle — non-degradation was confirmed at the `--all`
> aggregate instead.
>
> **[HISTORICAL] Status (2026-05-31 — full-stack `--all` incl. HS3 hard-soft 3-cycle + 3-pin routing JIT):**
> **Avg 1.1963 — beats RePlAce (1.4578) by 0.262 (−17.9%), and beats the UT
> Austin DREAMPlace leaderboard (1.4076) by 0.211 (−15.0%).** All 17 VALID /
> 0 overlaps. **11/17 wins** vs 1.1993 baseline. Cumulative Δ −0.0504,
> avg −0.0030/bench. Biggest movers: **ibm16 −0.0287** (recovers the
> +0.0108 fluke-loss from the prior run AND adds a net win), ibm07 −0.0151,
> ibm01 −0.0069, ibm13/ibm12 −0.005, ibm14 −0.0048, ibm09 −0.0034,
> ibm17/ibm18 −0.0033. Losses (all small): ibm10 +0.0172 (the
> mirror-image of ibm16: prior big winner became this run's main loser —
> RNG sensitivity, swap nets cumulative wins), ibm11 +0.0077, ibm06/ibm08
> +0.0017, ibm03 +0.0010, ibm04 +0.0003. Total runtime 4429s wall (74min,
> harness monotonic well under 3300s — no host-suspend drift this run).
>
> **HS3 (hard-soft 3-cycle rotation):** new move type. Captures
> configurations where H wants S1's slot but swapping H↔S1 hurts because
> S1's connections need to go elsewhere — 2-opt can't accept that chain
> individually, but the single combined 3-cycle (H → S1's old pos, S1 →
> S2's old pos, S2 → H's old pos) can. New `score_cycle_hard_soft_soft`
> + `commit_cycle_hard_soft_soft` on `IncrementalScorer` (extension of
> HXS to 3 modules via _touched_nets3). Bit-exact verified
> (`_verify_score_cycle_hard_soft_soft.py`: Δ ≤ 2.22e-16 across all
> trials and sequential commits on ibm01/04/10). New pass
> `_three_opt_hard_soft_soft` in the R2 round, dual-field, top_hot=15
> hards × k_inner=5 S1 × k_inner+1=6 S2 = ~375 trials/pass, 3s tight
> deadline cap, adaptive skip-if-empty. Cubic-in-knn but knn-truncated.
> **3-pin routing dispatcher numba JIT (#35):** speedup. The 3-pin
> dispatcher was 38% of move time (per profile) — the numpy gather /
> scatter / per-case mask dance carries meaningful overhead beyond the
> arithmetic. Collapsed into a single per-net numba loop with manual
> 3-element sort + case branching + direct H/V strip writes. Bit-exact
> within ≤4.4e-16. Saves another ~13-15s/bench → freed ~250s over the
> full `--all` (the ibm04 smoke went from 138.8s to 124.8s).
> ibm04 progression: 1.2092 baseline 1.0304 → ... → 1.0062 (prior shared-
> scorer + numba strips) → **1.0067** (+ HS3 + 3pin JIT). HS3 fired 4
> cycles on ibm04 (R1: 7 cycles, R2: 2). Note ibm04 score barely changed
> but runtime dropped 14s — the freed budget compounds across other
> benchmarks.
>
> Prior milestones (stacked):
> **[HISTORICAL] Status (2026-05-31 — full-stack `--all` incl. HXS+R6+WL-prefilter+shared-scorer+numba-JIT):**
> **Avg 1.1993 — beats RePlAce (1.4578) by 0.259 (−17.7%), and beats the UT
> Austin DREAMPlace leaderboard (1.4076) by 0.208 (−14.8%).** All 17 VALID /
> 0 overlaps. **14/17 wins** vs 1.2092 baseline (only ibm07 +0.004, ibm15
> +0.0004, ibm16 +0.0108 — the latter likely a fluke-loss back toward the
> ibm16 long-run mean; the prior 1.2092 run got an unusually-good 1.2641 on
> ibm16). Cumulative Δ −0.1683, avg −0.0099/bench. Biggest movers:
> **ibm18 −0.0359** (starvation FIXED — went from +0.283 with the previous
> HXS+R6 budget overrun to −0.036 with the shared scorer + numba freed-up
> budget), ibm17 −0.0252, ibm04 −0.0226, ibm10 −0.0209, ibm11 −0.0186,
> ibm06 −0.0158. Total runtime 11486s wall (host-suspend inflated; harness
> monotonic ≤3300s).
>
> **HXS (hard ⇄ soft cross-swap):** new move type. Exchanges a hard macro
> with a soft macro. Neither hard-2opt nor soft-2opt can find such pairs
> (each swaps only within its own kind). New `score_swap_hard_soft` /
> `commit_swap_hard_soft` on `IncrementalScorer` — hybrid of score_swap
> (hard's routing blockage via macro_subset) + score_swap_soft (no
> macro_subset for the soft). Bit-exact verified
> (`_verify_score_swap_hard_soft.py`: Δ ≤ 4.4e-16 across all trials and
> sequential commits on ibm01/04/10). New pass `_two_opt_hard_soft_swap`
> in the R2 round, dual-field (cong + density), 2.5s tight deadline cap,
> adaptive skip-if-empty.
> **R6 (combined cong+density relocation):** third hard-reloc pass per
> round, hotness = geometric mean of normalized cong & density. Catches
> macros moderately hot on both fields that neither pure pass prioritized
> (each ranking favors pure-field extremes). 4s deadline cap. Same proxy
> gate, same overlap check. Sparse firings (1-3/round) before adaptive
> skip-if-empty triggers.
> **WL-delta prefilter for soft-2opt:** new cheap `wl_delta_swap_soft`
> method on `IncrementalScorer` computes per-net HPWL change in ~50µs
> (vs ~5-10ms for the full score_swap_soft). Used in
> `_two_opt_soft_swap` as a prefilter — skip the full score call when
> predicted WL delta exceeds 0.01 (loose enough to keep every
> historically-accepted swap; typical accepted ΔWL is <0.002).
> **Persistent shared scorer per R2 round (#33):** the R2 round body has
> ~10 distinct passes (hard reloc cong / density / combined, soft reloc
> cong / density, soft-2opt cong / density × A5 passes, HXS cong /
> density, 2-opt cleanup); the status quo rebuilt an `IncrementalScorer`
> per pass (~0.1-0.3s each → ~10-20 s/benchmark). Now the scorer is
> built ONCE per round, lazily rebuilt on the rare case a pass's
> committed accepts don't pass the cumulative `cand_true < best_score`
> gate. Saves ~15-25s/benchmark, which the R2 loop spends on additional
> productive rounds.
> **Numba-JIT routing apply (#34):** soft-import numba; if available,
> JIT-compile `_apply_h_strips_batch` / `_apply_v_strips_batch` (the
> inner-inner loops of the 2-pin / 3-pin / big-net routing apply,
> ~10% of move time per profile). Pure numpy fallback when numba is
> absent. Bit-exact within ≤4.4e-16 (verified by the existing scorer
> verifier on ibm01/04/10). Saves another ~10-15s/benchmark.
> ibm04 progression (validating the stack incrementally):
> 1.2092-baseline 1.0304 → + HXS+R6 (tight caps) 1.0162 → + WL prefilter
> 1.0139 (187s) → + shared scorer 1.0074 (163s, **−24s**) → + numba JIT
> **1.0062 (138s, −49s vs pre-shared)** — total −0.0242 score,
> −49s/bench freed.
>
> Prior milestones (stacked):
> **[HISTORICAL] Status (2026-05-30 — full-stack `--all` incl. A4+A5+adaptive R2/skip-empty):**
> **Avg 1.2092 — beats RePlAce (1.4578) by 0.249 (−17.1%), and beats the UT
> Austin DREAMPlace leaderboard (1.4076) by 0.198 (−14.1%).** All 17 VALID /
> 0 overlaps. **15/17 wins** vs 1.2195 baseline (only ibm04 +0.0017 and ibm18
> +0.0063 — both near noise). Cumulative Δ −0.1755, avg −0.0103/bench.
> Biggest movers: ibm15 −0.0311, ibm06 −0.0259, ibm12 −0.0194, ibm13 −0.0174,
> ibm08 −0.0148, ibm14 −0.0135, ibm11 −0.0121. Total runtime 2716s.
>
> **A4 (WL-aware soft-2opt candidate ordering):** `_two_opt_soft_swap` now
> takes `net_centroid` + `wl_blend=0.3`, blending Euclidean distance with
> distance-to-net-centroid in the candidate ordering — the soft-2opt analog
> of A3. Pure ordering change; strictly non-regressing.
> **A5 (adaptive multi-pass soft-2opt):** each soft-2opt call in R2 now runs
> up to `A5_NUM_PASSES=2` passes with early-stop if the first pass made no
> improvement. Pass 2 fired 186/189 opportunities across the run — nearly
> every round had a productive 2nd pass.
> **Adaptive R2 round termination:** added `TINY_R2_ROUNDS_TO_STOP=2`
> consecutive rounds of Δ < `R2_DELTA_THRESHOLD=1e-3` to short-circuit
> diminishing-returns rounds. In practice the tiny-streak guard never fired
> on the winning run (every round productively > 1e-3) — confirms the rounds
> are doing real work.
> **Adaptive skip-empty replacing hardcoded `R3_CONG_MAX_ROUNDS`:** both the
> single-soft cong-relocation pass and the A1b cong-field soft-2opt now skip
> a round only after `SKIP_EMPTY_AFTER=1` empty round in a row. The earlier
> hardcoded round-3 cap on A1b was found to regress scores by killing
> productive late-round work (A1b finds 7–35 swaps even at round 6 on some
> benchmarks). Density `top_hot` boost still triggers, but adaptively (when
> the cong empty-streak counter saturates).
> **#3v2 time-shifted multi-seed 2-opt subprocess pool (drafted, env-gated
> off):** `MULTISEED_MP=1` runs the main "best" 2-opt inline first (full
> solo CPU during the 15s deadline), then submits DP seed 2-opts to a
> ProcessPoolExecutor afterward. Default off — direct subprocess parallelism
> on the deadline-bound search caused regression due to CPU contention.
> Total runtime 2716s (clean, no host suspend, well under 3600s hard cap).
>
> Prior milestones (stacked):
> **[HISTORICAL] Status (2026-05-30 — full-stack `--all` incl. H5+A1b+A1c+A1×2+Phase9-parallel):**
> **Avg 1.2195 — beats RePlAce (1.4578) by 0.238 (−16.3%), and beats the UT
> Austin DREAMPlace leaderboard (1.4076) by 0.188 (−13.4%).** All 17 VALID /
> 0 overlaps. We **beat RePlAce on every benchmark** (ibm01 flipped from
> +2.6% to −1.0%). All 17 benchmarks improved vs the 1.2433 baseline
> (17/17 wins, cumulative Δ −0.4044, avg −0.024/bench).
> **H5 (hard density relocation):** new pass — the R5-analog for hard macros.
> `_relocation_moves` now switches its hot/cold field via `use_density=True`;
> a new pass in the R2 round runs the hard-density variant after the existing
> cong-based hard reloc. Modest (1-3 moves/round) but consistent contribution.
> **A1b (cong-field soft-2opt):** soft-2opt now runs TWICE per round — once
> on the cong hotness field, once on density — same dual-field symmetry that
> gave R3 + R5 their compound gain. Finds 7-35 swaps/round on the cong pass.
> **A1c (cold-teleport):** each A1 pass appends 4 globally-coldest movable
> softs to the kNN candidate set per hot — analog of S9 cold-teleport for the
> hard 2-opt.
> **Phase 9 parallelization:** ThreadPoolExecutor on the 3 random-order
> legalize trials (numpy releases the GIL on the heavy work). Score step
> stays sequential (plc state). Saves ~0.3s/bench.
> **DREAMPlace ×3 already parallel** (confirmed) — 3 async subprocess
> handles, no change needed.
> Combined `--all`: 1.2433 → **1.2195** (−0.0238). Biggest movers: ibm12
> −0.069, ibm11 −0.041, ibm10 −0.029, ibm08 −0.030, ibm15 −0.028, ibm17
> −0.028. Total runtime 2598s (clean, under cap).
> Prior milestones (stacked):
> **A1 + A3 (added 2026-05-29) — the dominant new lever.**
> **A1 (soft-soft 2-opt):** new pair-swap move type that exchanges two soft
> macros' positions. Single-soft relocation can't find moves where two softs
> need to swap places (e.g., both at suboptimal cells where their connections
> would be happier in each other's slot). New `score_swap_soft` /
> `commit_swap_soft` on `IncrementalScorer` (analog of `score_swap` minus
> macro_subset since softs don't block routing), new `_two_opt_soft_swap`
> pass in the R2 interleave round (between soft-density and the hard 2-opt
> cleanup): top_hot=64 density-hot softs × k_neighbors=12 nearest movable
> softs, accept-on-true-proxy, ~6s budget slice. Bit-exact verified
> (`_verify_score_swap_soft.py`: Δ ≤ 2.2e-16 machine eps across trials and
> sequential commits). **A3 (smart soft candidate ordering):** new
> `soft_net_centroids()` method;
> `_soft_relocation_moves` now blends Euclidean distance with distance-to-
> net-centroid via `wl_blend=0.3` so candidates aligned with the soft's
> WL anchor are tried first. Pure ordering change — strictly non-regressing.
> Combined `--all`: 1.2737 → **1.2433** (−0.0304, **ALL 17 wins**, biggest
> movers ibm17 −0.059, ibm07 −0.050, ibm13 −0.043, ibm16/15 −0.039,
> ibm14 −0.035, ibm18 −0.032). Per-round soft-2opt accepted 9–41 swaps
> consistently across all 6 rounds, confirming A1 finds many real moves the
> single-soft passes couldn't reach. Total runtime 2291s (clean, no WSL
> inflation this run). **A1 is the largest single algorithmic improvement
> since R5** — they're now co-dominant levers, both around −0.03 to −0.1
> magnitude.
> Prior milestones (stacked):
> **S1 + S3 (added 2026-05-29):** S1 hoists the loop-invariant "subtract k's
> old routing + density" out of the relocation candidate inner loop via a new
> `_prepare_move(_soft)` / `_trial_at(_soft)` / `_commit_after_prep(_soft)` /
> `_revert_prep(_soft)` quartet on `IncrementalScorer`. Per-trial cost in the
> realistic same-macro / nearby-target pattern drops 25–43% (ibm10
> 1.50→0.90 ms, ibm15 1.50→0.86 ms, ibm17 1.82→1.36 ms). Bit-exact verified
> (`_verify_prep_trial.py`: Δ=0.00e+00 on every trial vs `score_move(_soft)`).
> S3 replaces `np.add.at` with `np.bincount` in the strip-batch routing fill —
> same-order accumulation, swap verifier still passes at Δ≤4.4e-16.
> Combined `--all`: 1.2755 → **1.2737** (−0.0018; 10/17 wins; ibm18 −0.021
> and ibm06 −0.019 the biggest movers).
> Prior milestones (stacked):
> **Latest changes stacked this session** (each one bit-exact-verified before
> the next): (1) **Incremental congestion cost** — `IncrementalScorer` caches
> the smoothed normalized H/V and per move re-smooths only the touched-net
> bbox from raw flats (bit-identical to a full re-smooth; swap Δ≤4.4e-16);
> isolated `--all` 1.2799 → **1.2767**. (2) **Idea #1 subset-cumsum strip-batch**
> (only the unique touched rows/cols are cumsummed). (3) **Idea #2 topology-
> struct cache** for the routing apply (position-independent gather + 2/3/≥4-pin
> classification built once per macro, reused across the −1/+1 applies and
> across moves; the position-dependent fill is still recomputed → bit-exact).
> (4) **Floor-reservation budget allocator** — closes the ibm18-starvation bug:
> reserve `(110+60)·remaining` for every other remaining benchmark plus 60s of
> own-overrun slack so the last benchmark always gets ≥110s. Worst-case
> simulation: all 17 ≥110s, cumulative ends at 3300. (5) **A: cong soft-pass
> hard-cap at round 3 + C: density `top_hot` boost 128→192 on rounds 4–6** —
> cong saturates by round 3 (ibm09 round 4+ accepts ≤2 moves, ~zero gain);
> skip it and spend the freed ~4–5s/round on density. Combined `--all`:
> 1.2767 → **1.2755** (−0.0012; 12/17 wins vs the cong-only baseline). **ibm18
> = 1.5787** (vs the floor-res-only run's starved 1.7941 — confirms the
> allocator works). Biggest movers: ibm17 −0.034, ibm16 −0.019, ibm07 −0.015.
> (Wall-time reported 3860s under WSL host-suspend inflation; the placer's
> `monotonic` budget held — no benchmark returned baseline.)
>
> **Disproven this session:** the "shared scorer" lever — measure-first
> profiling showed the per-pass fixed overhead is only
> ~0.1–0.28s/round (not the projected 60–75s), so a shared-scorer refactor
> would save <1.7s/benchmark and risk correctness. NOT implemented. **Disproven:** R4 WL-aware hard
> relocation (net-centroid target bias) — slightly worse, reverted (scaffolding
> kept). Prior: **R3** soft cong relocation 1.4216→1.3764; **R2/R2b** 1.4326→
> 1.4216; **R1** 1.4422→1.4326; **S9/P3** before that.
> Earlier detail — **R1 congestion-directed relocation moves** — a post-2-opt
> pass that RELOCATES the hottest macros into empty low-congestion legal gaps
> (a move the swap-only 2-opt can't make). Uses the incremental scorer's new
> `score_move` (verified bit-exact, ≤6e-9). --all 1.4422 → 1.4326 (all 17
> improved). Before that: **S9 congestion-aware 2-opt** (hot-first ordering +
> cold teleport augmentation), 1.4424 → 1.4422; and **P3 incremental
> density** — `IncrementalScorer` now keeps
> the occupancy grid as state and updates only the 2 swapped macros' cells
> per score (verified bit-exact vs full recompute, ≤4.4e-16). score_swap is
> −22% to −29% faster → +40–56% more 2-opt scores fit the 15s deadline →
> avg 1.4435 → **1.4424** (the gain lands on the deadline-bound large
> benchmarks: ibm10 1.3381→1.3346, ibm16 1.5057→1.5041). `--all` 979s
> (WSL-inflated). S1 (basin-hopping 2-opt) is implemented but DORMANT pending
> its own --all (P3 now makes small benchmarks converge early, freeing kick
> budget). Prior changes: multi-seed 2-opt-on-winner (O2), k=20 (S2),
> IncrementalScorer clean-init (O5). See section below for headlines.
>
> History notes (2026-05-20): this file started as v1's local copy of
> the team's PROGRESS.md, updated through v14. The "Iteration Log"
> section below tracks the v1-era progression (v1 → v14). The v2
> session (2026-05-23 → 2026-05-25) is summarized in the new section
> immediately after the Baselines table.

---

## Baselines (reference)

| Placer | Avg (17 benchmarks) | Notes |
|---|---|---|
| SA baseline | 2.1251 | challenge organizer SA |
| will_seed | 1.5338 | challenge organizer legalization |
| sameer_v1 leg-only | 1.5062 | our legalize-only, confirmed |
| RePlAce | 1.4578 | Grand Prize target |
| UT Austin (DREAMPlace) | 1.4076 | leaderboard #1 |
| **v2 (this submission)** | **1.2799** | **BEATS RePlAce by 0.178 (−12.2%); below leaderboard 1.4076 by 0.128** (R5 soft density relocation + R3 + R2 + R1 + S9 + P3) |

---

## v2 — Submission state (2026-05-25)

### Headline

| Metric | Value |
|---|---|
| 17 IBM benchmarks avg | **1.2799** |
| RePlAce target | 1.4578 |
| **Gap to RePlAce** | **−12.2% (beat by 0.178)** |
| v12 starting point | 1.4854 |
| **Total v2 improvement** | **−0.1090** |
| DREAMPlace leaderboard | 1.4076 — **v2 BEATS it by 0.128 (−9.1%)** |
| `--all` wall-clock | 2639s (< 3600s cap) |
| NG45 avg (Tier 2) | 0.7830 |

### Per-benchmark results (v12 → R5 1.2799)

| Bench | v12 | R5 (1.2799) | Δ vs v12 |
|---|---|---|---|
| ibm01 | 1.1860 | 1.0544 | −0.132 |
| ibm02 | 1.5923 | 1.3302 | −0.262 |
| ibm03 | 1.3603 | 1.0787 | −0.282 |
| ibm04 | 1.3316 | 1.0648 | −0.267 |
| ibm06 | 1.6684 | 1.3104 | −0.358 |
| ibm07 | 1.4924 | 1.2955 | −0.197 |
| ibm08 | 1.5251 | 1.3048 | −0.220 |
| ibm09 | 1.1304 | 0.9720 | −0.158 |
| ibm10 | 1.4037 | 1.2071 | −0.197 |
| ibm11 | 1.2354 | 1.0862 | −0.149 |
| ibm12 | 1.6507 | 1.5127 | −0.138 |
| ibm13 | 1.4011 | 1.1769 | −0.224 |
| ibm14 | 1.6033 | 1.3945 | −0.209 |
| ibm15 | 1.6061 | 1.4290 | −0.177 |
| ibm16 | 1.5323 | 1.3494 | −0.183 |
| ibm17 | 1.7437 | 1.6172 | −0.127 |
| ibm18 | 1.7896 | 1.5754 | −0.214 |
| **AVG** | **1.4854** | **1.2800** | **−0.205** |

(R1 column = `--all` 2026-05-27: P3 incremental density + S9 cong-aware 2-opt +
R1 congestion-directed relocation. ALL 17 improved vs the prior 1.4435; R1's
relocation pass alone was −0.0096 avg over the 1.4422 state. Total runtime 751s,
all 17
VALID / 0 overlaps.)

**All 17 benchmarks improved.** No regressions vs v12.

### Architecture changes vs v1

1. **`MacroPlacer.__init__` cross-benchmark state** (B1) — tracks
   cumulative wall-clock with `time.monotonic()` for adaptive
   per-benchmark budget under `--all`'s 3600s harness cap.
2. **Proxy-driven 2-opt-on-winner** (A1) — `_two_opt_proxy_swap` uses
   `_exact_proxy` rescoring per swap (was: displacement-from-init,
   anti-correlated with proxy).
3. **B3 incremental scoring** (4 phases) —
   - Phase 1: global position cache eliminates per-call get_pos loops.
   - Phase 2: per-net HPWL incremental via macro→nets index.
   - Phase 3: numpy abu (np.partition) replaces Python sorted +
     .tolist() conversions.
   - Phase 4: per-net incremental ROUTING via subset dispatch
     helpers (`_apply_net_routing_subset`, `_apply_macro_routing_subset`).
     Per-score on ibm10 dropped 22.5ms → ~3ms (7.5× faster).
4. **B4 dispatch cache** — pre-compute topology-fixed index arrays in
   `_build_cong_cache` (idx2/idx3/idx_big/net_local_ids/global_pin_idx).
5. **A6 axis #1: Phase 8 TOP-K cong-grad** with multi-iter chains —
   restrict cong-grad to K hottest macros; chain up to 3 iters per K
   in {5, 10, 20}.
6. **A6 axis #4: Phase 9 random-tiebreak legalize order** — N=3
   variant orderings of `_will_legalize` with random secondary sort
   key (primary key −area preserved).
7. **2-opt widening** — k_neighbors 5 → 10, max_iters 3 → 6.
8. **A2 DREAMPlace soft_movable diversification** — 2-DP launch:
   lo-fix (td=0.65, soft_movable=False) + hi-mov (td=0.85,
   soft_movable=True). Best-of-both candidate per benchmark.
9. **WSL2 clock-drift hardening** — all 56 `time.time()` calls
   replaced with `time.monotonic()` to prevent host-suspend-induced
   wall-clock jumps from corrupting deadlines / budgets.
10. **NG45 disambiguation** — `_load_plc` matches NG45 designs by
    canvas dimensions when `benchmark.name == "output_CT_Grouping"`
    (all 4 NG45 designs share that name due to load_benchmark's
    basename logic).

### Reproducibility

Multiple `--all` runs confirmed avg 1.4475 ± noise (typically
≤ 0.001 per-benchmark variance). Largest run-to-run swing observed:
ibm10 ±0.0024 due to non-deterministic CPU scheduling affecting
2-opt deadline-bound decisions.

### Headline progression through the v2 session (2026-05-23 → 2026-05-25)

| Milestone | Avg | Δ from prior | Gap vs RePlAce 1.4578 |
|---|---|---|---|
| v12 (session start) | 1.4854 | — | +1.9% |
| + B1 cumulative-budget guard | 1.4782 | −0.0072 | +1.4% |
| + A1 proxy 2-opt | 1.4723 | −0.0059 | +1.0% |
| + B3 phase 1 (pos cache) | 1.4719 | −0.0004 | +1.0% |
| + B3 phase 2 (per-net HPWL incr) | 1.4714 | −0.0005 | +0.9% |
| + B3 phase 3 (numpy abu) | 1.4711 | −0.0003 | +0.9% |
| + A6 Phase 8 (TOP-K cong-grad) | 1.4701 | −0.0010 | +0.8% |
| + Phase 9 (random-order legalize) | 1.4698 | −0.0003 | +0.8% |
| + B4 dispatch cache | 1.4698 | 0 | +0.8% |
| + B3 phase 4 (per-net cong incr) | 1.4690 | −0.0008 | +0.8% |
| + 2-opt widening (k=10, iters=6) + Phase 8 chains | 1.4647 | −0.0043 | +0.5% |
| + A2 (DP soft_movable best-of-both) | 1.4486 | −0.0161 | **−0.6%** |
| + A2 refined (lo-fix + hi-mov) | **1.4475** | **−0.0011** | **−0.7%** |
| (+ WSL2 monotonic clock fix — no score Δ, ↓ wall-clock 720s → 526s) | | | |

---

## Iteration Log

### v1: Legalization only
- Strategy: legalize directly from initial.plc, no restarts
- All benchmarks: return baseline legalized position
- ibm01: 1.2253, avg: 1.5062

### v2/v3: Multi-restart with exact proxy scoring (broken by density fallback regression)
- Strategy: 5 random Gaussian restarts, score all with exact proxy, pick best
- For n>350 benchmarks: density fallback to rank restarts (ANTI-CORRELATED, see below)
- ibm01: 1.1854, ibm03: 1.3944, ibm08: 1.5251 (exact benchmarks improved)
- ibm11: 1.3770 (density fallback selected 8% noise → actual proxy 11.5% WORSE than baseline!)
- Full eval avg = 1.5656 (REGRESSED from v1! Density fallback hurt large benchmarks by +0.14 each)

### v4: Density fallback disabled, exact scoring for ibm11 (CURRENT)
- Fix 1: Non-exact benchmarks (n>400 or grid>2000 cells) return baseline immediately
- Fix 2: Raised EXACT_MACRO_THRESHOLD from 350 to 400 → ibm11 (n=373) now uses exact scoring
- ibm11 with exact scoring: baseline=1.2354 (81s), restart 1 (2%)=1.2591 → baseline wins
- Expected avg: ~1.501 (v1 for non-exact benchmarks + improved exact benchmarks)
- Full eval running (2026-04-29)

### v5: Budget-filling restarts
- Extended noise_fracs from 4 entries to 35 entries
- n_restarts=50 (budget check is the actual limit, not n_restarts)
- Core 4 fracs [0.02, 0.04, 0.06, 0.08] unchanged → preserves ibm01/03/08 wins
- Fast benchmarks now fill their budget:
  - ibm01 (~5s/score): ~20 restarts vs 4 before
  - ibm03 (~10s/score): ~9 restarts vs 4
  - ibm04 (~14s/score): ~10 restarts vs 4
  - ibm06 (~16s/score): ~8 restarts vs 4
  - ibm09 (~20s/score): ~6 restarts vs 4
  - ibm08 (~36s/score): ~4 restarts (unchanged, already at budget limit)
  - ibm11 (~81s/score): ~1 restart (unchanged)

### v8: Iterative congestion-gradient descent + wide step (HISTORICAL — RETIRED)
- Phase 1: Iterative gradient descent at frac=0.04, up to 4 steps. After each improving step,
  extract legalized position from best_pl and use it with plc's updated congestion map for the
  next gradient step. Stop when a step fails to improve or budget < 3×t_score.
- Phase 2: After any improvement from phase 1, try frac=0.08 then frac=0.12 from baseline_pos
  using current (possibly stale) plc congestion state. Stop when a wide step fails to improve.
  Key insight: stale plc from failed iter=2 provides 2nd-order info that guides a larger jump.
- Benchmarks where cong-grad doesn't improve (iter=1 fails): wide steps skipped, exact same
  behavior as v6 for ibm07, ibm08, ibm11.
- ibm15 confirmed at 164s scoring (SLOW_SCORE_THRESHOLD catches it), EXACT_GRID_CELL_LIMIT stays 2000.
- Confirmed improvements vs v6 (2026-04-30):
  - ibm02: 1.6203 → **1.5823** (-0.038; stale iter=2 plc + wide=8% from baseline is key)
  - ibm03: 1.3854 → **1.3583** (-0.027; 2 iterative steps)
  - ibm04: 1.3882 → **1.3479** (-0.040; 4 iterative steps, budget fills)
  - ibm06: 1.6838 → **1.6810** (-0.003; 2 iterative steps)
- No regressions: ibm08=1.5251, ibm09=1.1304 both confirmed clean
- Est. avg: ~1.4867 (gap to RePlAce: 0.029, down from 0.035 in v6)

---

### v6: Routing-congestion-gradient perturbation
- After baseline scoring, plc has the routing congestion map from get_congestion_cost().
- New restart (k=1 for IBM benchmarks): perturb baseline_pos using the REAL H/V routing
  congestion map from PlacementCost.get_horizontal/vertical_routing_congestion().
- For each macro in a cell with congestion > 0.5: move against the finite-difference
  gradient of the congestion map (toward lower-congestion neighbors). Small random noise
  (0.1× scale) added to break symmetry.
- Uses separate RandomState(seed+1) so main np.random state unchanged; noise restarts
  get identical draws to v5 (ibm01 6% win preserved at k shifted by 1).
- Confirmed improvements (2026-04-29):
  - ibm02 (cong=2.375): 1.6800 → 1.6203 (-0.0597)
  - ibm06 (cong=2.503): 1.7198 → 1.6838 (-0.0360)
  - ibm01 (cong=1.274): no improvement (congestion too low for gradient signal)
- ibm07, ibm08 tests contaminated by system load (scoring inflated 3-4x); clean results pending
- Full clean eval running (2026-04-29)

---

## Per-Benchmark Detail (confirmed from full evals)

v12 = current best (system/v1, --all confirmed 2026-05-10 with budget-relaxation fix).
v12 stable --all avg = **1.4854**. Reproduced in 2 of 3 runs (3rd run had Run-1 ibm04 spike,
fixed by adding `BUDGET_OVERRUN_S=60s` allowance for directed-restart phases).

| Benchmark | hard_n | grid_cells | v1 (leg) | v8 | v11 | **v12 (current)** | RePlAce | vs RePlAce | Notes |
|---|---|---|---|---|---|---|---|---|---|
| ibm01 | 246 | 45x41=1845 | 1.2253 | 1.1854 | 1.1854 | **1.1860** | 0.9976 | -18.9% | t_score=2-3s clean; 6% noise wins; v11's 1.1854 was a lucky outlier |
| ibm02 | 271 | 30x27=810 | 1.6800 | 1.5823 | 1.5823 | **1.5923** | 1.8370 | +13.3% | t_score=7-8s clean; wide=8% wins; v11's 1.5823 was a lucky outlier (stale-plc lottery) |
| ibm03 | 290 | 32x29=928 | 1.4100 | 1.3583 | 1.3547 | **1.3603** | 1.3222 | -2.9% | t_score=5-6s clean; iter=2 cong-grad wins; v11's 1.3547 was a lucky outlier |
| ibm04 | 295 | 31x30=930 | 1.4101 | 1.3479 | 1.3390 | **1.3316** | 1.3024 | -2.2% | t_score=6-7s clean; 7 iter steps + Phase 2 + Phase 3 wins. **STABLE under --all with budget fix** (was fragile in run 1 without fix) |
| ibm06 | 178 | 31x28=868 | 1.7198 | 1.6810 | 1.6797 | **1.6684** | 1.6187 | -3.1% | clean CPU rediscovery: −0.0113 vs v11 stale (frac=0.02 at iter=4 + Phase 3) |
| ibm07 | 291 | 35x32=1120 | 1.4950 | 1.4950 | 1.4950 | **1.4924** | 1.4633 | -2.0% | clean CPU, 1% noise restart wins (−0.0026 vs v11); cong-grad doesn't help |
| ibm08 | 301 | 38x34=1292 | 1.5582 | 1.5251 | 1.5251 | **1.5251** | 1.4285 | -6.8% | cong-grad worse; 6% noise wins; stable across runs |
| ibm09 | 253 | 36x38=1368 | 1.1363 | 1.1304 | 1.1304 | **1.1304** | 1.1194 | -1.0% | 1 cong-grad iter wins |
| ibm10 | 786 | 55x41=2255 | 1.4037 | 1.4037 | 1.4037 | **1.4037** | 1.5009 | +6.5% | n>400; returns baseline |
| ibm11 | 373 | 39x45=1755 | 1.2354 | 1.2354 | 1.2354 | **1.2354** | 1.1774 | -4.9% | v12: re-included in exact pipeline (t_score=17s clean); 10 restarts attempted, baseline wins |
| ibm12 | 651 | 47x47=2209 | 1.6507 | 1.6507 | 1.6507 | **1.6507** | 1.7261 | +4.4% | n>400; returns baseline |
| ibm13 | 424 | 43x43=1849 | 1.4011 | 1.4011 | 1.4011 | **1.4011** | 1.3355 | -4.9% | n>400; returns baseline |
| ibm14 | 614 | 49x44=2156 | 1.6033 | 1.6033 | 1.6033 | **1.6033** | 1.5436 | -3.9% | n>400; returns baseline |
| ibm15 | 393 | 57x38=2166 | 1.6061 | 1.6061 | 1.6061 | **1.6061** | 1.5159 | -5.9% | v12: re-included (t_score=43s clean); restarts attempted, baseline wins |
| ibm16 | 458 | 45x48=2160 | 1.5323 | 1.5323 | 1.5323 | **1.5323** | 1.4780 | -3.7% | n>400; returns baseline |
| ibm17 | 760 | 51x44=2244 | 1.7437 | 1.7437 | 1.7437 | **1.7437** | 1.6446 | -6.0% | n>400; returns baseline |
| ibm18 | 285 | 55x39=2145 | 1.7941 | 1.7941 | 1.7941 | **1.7896** | 1.7722 | -1.0% | v12: re-included (t_score=62s clean); cong-grad iter=1 wins (−0.0045) |

**v10b full eval avg (2026-04-30):** 1.4877 (ibm04=1.3390 new best; ibm08=1.5539 under load)
**v11 clean estimate:** 1.4860 (composite — never actually --all'd; numbers were lucky outliers for ibm01/02/03)
**v12 stable --all avg (2026-05-10):** **1.4854** with `BUDGET_OVERRUN_S=60.0s` fix; reproducible across runs

---

### v12 = system/v1 (2026-05-08 → 2026-05-10): threshold change + budget-relaxation fix

Three concrete code changes vs sameer_v1 and v11:

1. **EXACT_MACRO_THRESHOLD: 340 → 400** (re-includes ibm11, ibm15)
2. **EXACT_GRID_CELL_LIMIT: 2000 → 2200** (re-includes ibm15, ibm18)
3. **BUDGET_OVERRUN_S = 60.0s** for directed-restart phases (Phase 1/2/3 cong-grad). Allows
   the placer to spend up to `time_budget_s + 60s` on directed restarts, while keeping the
   noise loop strict (`time_budget_s` only).

#### Why the threshold change

Re-measurement of scoring time on clean CPU (2026-05-08) revealed PROGRESS.md v11 estimates were
4–13× too high:

| Benchmark | v11 estimate | v12 measured (clean) |
|---|---|---|
| ibm11 | 75–263s | **17.7s** |
| ibm15 | 160s | **42.8s** |
| ibm18 | 220s | **61.7s** |

All three well under `SLOW_SCORE_THRESHOLD_S=100s`. Threshold change re-includes them. Restarts
attempted on each:
- ibm11: 10 restarts, baseline (1.2354) wins — no change vs v11
- ibm15: restarts attempted, baseline (1.6061) wins — no change vs v11
- ibm18: 2 restarts, **cong-grad iter=1 wins → 1.7896** (−0.0045 vs baseline-only 1.7941)

#### Why the budget fix

**Problem found in --all run 1 (2026-05-10):** ibm04 normally scores 7s/call on clean CPU. But
during run 1, iter=1 of cong-grad spiked to 200s (likely transient CPU contention). This pushed
total time to 209s — over the 200s soft budget — and the post-scoring guard fired, returning
False from `_try_restart`. The calling code `if not _try_restart(...): return best_pl` then
terminated the entire placer, returning iter=1's result (1.3882) instead of Phase 3's 1.3316.

ibm04 collapsed by +0.0566. That single benchmark cost +0.0033 on the avg (1.4854 → 1.4888).

**Fix:**
- Add `allow_overrun: bool = False` parameter to `_try_restart`. When True, use
  `time_budget_s + 60s` as the cap for both pre- and post-scoring checks.
- Pass `allow_overrun=True` for all directed-restart calls (density-grad, all Phase 1 cong-grad
  iters, Phase 2 wide steps, Phase 3 cong-grad).
- Change cong-grad call sites from `if not _try_restart(...): return best_pl` to
  `if not _try_restart(...): break` so a budget exhaustion in one phase doesn't kill subsequent
  phases.
- Noise loop calls keep default `allow_overrun=False` — they're exploratory and shouldn't
  push us over budget on dead-end benchmarks.

**Result:** ibm04's 1.3316 win is now reproducible under --all conditions. Confirmed in --all
run 3 (2026-05-10): ibm04 = 1.3316. Bonus: ibm18 ticks 1.7898 → 1.7896 (one extra cong-grad iter
fits within the relaxed cap).

#### Bonus rediscoveries (clean CPU, no code change required)

- **ibm06 = 1.6684** (was 1.6797 in v11). Clean CPU runs hit a different cong-grad iteration
  pattern that lands at 1.6684 consistently. −0.0113 improvement.
- **ibm07 = 1.4924** (was 1.4950 in v11). Restart 6 (1% noise) wins. PROGRESS.md v11 said "noise
  doesn't help" but never tested 1% noise specifically on ibm07. −0.0026 improvement.
- **ibm04 = 1.3316** (was 1.3390 in v11). On clean CPU (t_score=6.4s instead of 15s) the placer
  fits more iterations and Phase 3 lands at 1.3316. −0.0074 improvement.

#### v11 numbers were a mix of outliers

The v11 PROGRESS.md figures for ibm01, ibm02, ibm03 turned out to be **lucky outliers**, not
stable targets. Today's clean runs (and the --all results) show:
- ibm01: 1.1860 (was 1.1854 in v11) +0.0006
- ibm02: 1.5923 (was 1.5823 in v11) +0.0100 — stale-plc trick is timing-sensitive
- ibm03: 1.3603 (was 1.3547 in v11) +0.0056

These regressions partially offset the v12 wins. Net delta vs v11 estimate: **−0.0006 to the avg**
(small, but in the right direction, and now reproducible).

#### Final numbers

- **v11 estimate (composite, never measured --all):** 1.4860
- **v10b actual --all (2026-04-30):** 1.4877
- **v12 stable --all (2026-05-10):** **1.4854**

Wins (vs v11 estimate): ibm04 −0.0074, ibm06 −0.0113, ibm07 −0.0026, ibm18 −0.0045 = −0.0258
Regressions (vs v11 estimate): ibm01 +0.0006, ibm02 +0.0100, ibm03 +0.0056 = +0.0162
Net: −0.0096 / 17 ≈ −0.00057 to avg.

---

### v14 = today's session (2026-05-19 → 2026-05-20): speed-only kept, structural attempts deferred to DREAMPlace

#### Kept changes (verified, no regression)

1. **Tier 3: Vectorize `_will_legalize`** (2026-05-19). Greedy spiral search rewritten in numpy: per ring, generate all 8r candidates at once via `_ring_offsets`, run a single `[K, P]` conflict matrix instead of nested Python loops. ibm04 legalize: 3.2s → 0.27s (12×). All cong-grad iters now run sub-second.

   **Critical correctness fix**: the original scalar computed `d² = (cx - pos[idx, 0])²` where `cx` is a Python float and `pos[idx, 0]` is numpy float32. NumPy demotes the Python float to float32 for the subtraction (Python-scalar-meets-numpy-scalar rule), so d² is computed at float32 precision. This causes symmetric ring candidates like (-1, 0) and (0, -1) to break ties at float32 noise instead of being truly equal. My initial vectorized version computed d² in float64 (cand_x is a strong-typed float64 array, no demotion), which made ties exact and changed which candidate `np.argmin` picked. Result: ibm04's cong-grad iter-2 diverged and the placer landed at 1.3364 instead of 1.3316. **Fix**: cast `cand_x`/`cand_y` to `pos.dtype` before subtraction, mirroring scalar's float32 demotion. Bit-equivalent legalize confirmed via diff harness on ibm04 iter-2 input. See `placer.py:178-191`.

2. **Running-max `t_one_score`** (2026-05-19). Re-added v11's running-max budget guard (removed in v12 for noise sensitivity). Under --all CPU contention, scoring can be 3-5× slower than the baseline measurement (not "jitter"). Without adaptation, the budget check approves restarts that then exceed cap, causing Phase 3 to skip on benchmarks like ibm04 (observed 1.3316 → 1.3449 regression in multi-order --all). The running-max tightens budget when contention is observed; brief blips that double t_one_score still leave 60s overrun for directed phases. Defensive — no improvement on its own, enables other experiments.

3. **2-opt swap post-pass on baseline-only branch** (2026-05-19). After greedy spiral-search legalize, scan macro pairs within K=5 nearest neighbors; accept swaps that legal AND strictly reduce per-pair displacement. Applied only to the `n>400` baseline-only branch (no cong-grad trajectory to disrupt). Tested gains: ibm10 −0.0006, ibm12 −0.0001, ibm13 −0.0005, ibm14 −0.0005, ibm16 +0.0001, ibm17 +0.0001. Net sum: −0.0015 across 6 baseline-only benchmarks ≈ −0.0001 to 17-bench avg.

#### Cleanup (2026-05-20, prepping for DREAMPlace integration)

- **Removed from placer.py**: `_wiremask_place`, `_build_wm_net_cache`, `_density_gradient_perturb`, `_congestion_heatmap`, `_box_blur` (all dead code on IBM — density-grad never fires for n>100).
- **Deleted files**: `surrogate.py`, `_calibration_test.py`, `_path3_incremental_test.py` (rejected experiments, never wired in).
- **placer.py**: 1159 → 894 lines.

#### Rejected today (sporadic / catastrophic / dead)

1. **Multi-order baseline (Phase 1-disrupting)** — adds smallest-area / tallest / widest orderings as extra Phase-0 candidates before Phase 1. Under --all CPU contention, the 4 extra scorings (3 multi-order + 1 baseline re-score) consumed enough budget that Phase 3 didn't fit on ibm04 (regression 1.3316 → 1.3449). Net negative.

2. **Displacement-ranked multi-order on baseline-only** — pick the legalization with smallest total displacement from initial.plc among 3 orderings. Catastrophically wrong: ibm10 with `tallest` order had lowest displacement (414 vs 1051 default) but proxy was 1.5658 vs 1.4037 (+0.162 regression — congestion blew up). ibm12 with `smallest-area` produced INVALID placement (27 overlaps) because large macros couldn't find slots within the 60s spiral deadline. Conclusion: displacement-sum is NOT a useful proxy ranker across orderings.

3. **2-opt-everywhere (in `_try_restart`)** — applied to every legalize result (baseline + cong-grad iters + noise). On ibm04, baseline 2-opt nudge cascades through cong-grad: −0.0115 win (1.3201). But on ibm06, the same baseline 2-opt nudge improves iter-1 enough that iter-2 can't beat it → Phase 1 break-on-no-improvement fires at cong_iter=1 (<2 halving threshold) → Phase 3 skipped → +0.0087 regression. Sporadic. Root cause: 2-opt's "snap back toward target" interferes with cong-grad's "push away from congested cells" trajectory.

4. **Multi-frac Phase 3** — try `frac ∈ {0.02, 0.04, 0.06}` instead of just 0.04. Tested ibm04/06/02/09: f=0.04 always wins; extra fracs add 2 scorings of overhead per Phase-3 benchmark for no improvement. Safe but ineffective.

5. **WireMask-BBO + congestion penalty** (alpha=30, G=25) — the v13 salvage path from PROGRESS.md. Tested ibm01/04/06: sporadic. Helps sparse (ibm01: 1.1964 vs baseline 1.2253 = −0.029) but hurts dense (ibm04: 1.5070 vs 1.4101 = +0.097; ibm06: 1.8890 vs 1.7197 = +0.169). Root cause: WireMask is constructive — rebuilds from scratch and loses initial.plc's hand-tuned spread that the pipeline operates around. A single alpha cannot satisfy all benchmarks (would need per-benchmark tuning, which violates the "no benchmark-specific tweaks" rule). Implementation removed from placer.py.

6. **`plc.optimize_stdcells` post-pass** — academic force-directed soft-macro re-placement (`external/MacroPlacement/CodeElements/Plc_client/plc_client_os.py` line 2886). Timed on ibm01 (smallest, n_soft=894) at num_steps=10: **126.6s** (~13s per step), and the result was **+0.1296 WORSE** than baseline (1.2253 → 1.3549 with default attract/repel factors). Pure Python iteration over ~1000 soft macros and ~10000 nets per step; no C++ binding. Effectively infeasible inside our 200s budget. Would need a multi-day rewrite in vectorized numpy/torch with tuned parameters to ever be useful. Dead path.

7. **Vectorized soft-macro re-placement (the rewrite of #6)** — implemented and tested 2026-05-20. Three algorithms, all in a new `soft_relax.py` module: (a) HPWL² gradient descent (textbook analytical placement); (b) connectivity-weighted displacement-follow (translate softs by the avg displacement of their connected hards); (c) HPWL + grid-bin density repulsion (the "do it right" combined version). Edge extraction from `plc.nets` cached on plc object (~1.5s one-time per benchmark); per-call runtime 0.5–10ms — performance was never the issue. **All three regress proxy on every benchmark tested.** Best result was hpwl+density with 2 steps × 0.005 max_frac × dw=0.5: +0.003 (ibm06) to +0.031 (ibm01). Tested across hard-perturbation magnitudes 0% → 80%; no regime where any variant netted negative delta. Module deleted after the test; see this entry for the negative result.

   **Decomposition that explains the loss** (ibm04, 15% hard perturb):
   ```
   stale softs:           WL=0.082, D=0.951, C=1.815, proxy=1.465
   HPWL-only relax:       WL=0.077, D=1.074, C=1.715, proxy=1.472  (D ↑)
   HPWL + density repel:  WL=0.072, D=0.950, C=1.747, proxy=1.420 → 1.420 vs base 1.410 = +0.010
   ```
   HPWL relax DOES improve WL (−0.005) and congestion (−0.10), and density repulsion successfully cancels the density rise — but the residual is still net positive. Initial.plc softs sit in a steep local minimum on the *joint* (WL, density, congestion) surface; any motion away from there pays one component faster than it gains on others, no matter how the forces are balanced.

   **Corrects #6's "soft mismatch" theory.** PROGRESS.md previously attributed DREAMPlace standalone's 0.2–0.3 regression to stale softs around moved hards. The decomposition above shows stale-soft cost is at most **~0.05** even at 30%–80% random hard perturbation (ibm04 stale-softs at perturb=0.30: 1.579; at perturb=0.80: 1.500 — actually goes *down* as hards spread to fill canvas). DREAMPlace standalone's real failure is that **its WL-optimized hard placement lands in a WL basin, which is uncorrelated with the congestion-dominated proxy basin** — same root cause as the WireMask-BBO failure. No amount of soft re-placement can fix that. The async DREAMPlace integration retains value as a side-channel for plc-state mutation (which seeds new cong-grad basins), not for its placement quality per se.

   **Implication: hard-placement search is the only useful axis.** Stop trying to optimize softs.

#### In progress: async DREAMPlace bridge (2026-05-20)

The v13 sync bridge was rejected in May because its 10-15s subprocess overhead displaced productive restarts on 7/17 benchmarks (net +0.0043 worse). PROGRESS.md notes the salvage path: async invocation so DREAMPlace runs in parallel with our scoring.

**Status**: restored `dreamplace_bridge/` from commit 111f315; added `AsyncDreamplaceHandle` + `launch_dreamplace_async` for non-blocking subprocess management. Integrated into `placer.py` as Phase 5: launch DREAMPlace at `place()` entry, check after Phase 3 as additive candidate ("dreamplace global"), and follow with one cong-grad iter from DREAMPlace's legalized position ("cong-grad from-dreamplace") to exploit the plc-state-mutation effect that PROGRESS.md noted as the source of v13's real wins. DREAMPlace build in progress (cmake done, `make -j2` ~70% on 2026-05-20 10:15).

**Expected gain (if async parallelism works)**: −0.005 to −0.025 to avg. Lower bound (−0.005) is the v13 wins (ibm04, ibm11) without the displacement cost. Upper bound (−0.025) assumes DREAMPlace also mutates plc-state usefully on 3-5 other benchmarks. Won't reach DREAMPlace standalone's 1.4076 because our pipeline still owns the basin search; DREAMPlace is just one additional seed.

**Risks**:
- *Async parallelism may not materialize* — depends on whether plc's C++ scoring releases the GIL. If it doesn't, DREAMPlace burns CPU contending with the scoring thread.
- *Soft-macro mismatch* — v13's standalone DREAMPlace was ~0.2-0.3 worse than baseline because soft macros stayed at initial positions while hard macros moved. The cong-grad-from-DREAMPlace step partially compensates (cong-grad nudges hard macros and softs are re-scored via plc), but doesn't fix the underlying issue. `optimize_stdcells` would, but it's too slow.

---

### v15 = current session (2026-05-20 → 2026-05-21): DREAMPlace bridge functional, Improvement #1 enabled

**Headline: --all avg 1.4854 → 1.4804 (−0.0050 absolute)** — confirmed via partial v5 run (16/17 benchmarks; ibm17 timed out at 3600s cumulative). Wins: ibm01 (−0.044), ibm04 (−0.012), ibm10 (−0.037), ibm14 (−0.003). Regression: ibm07 (+0.003).

#### Bridge architecture fix (was: DP NLP plateaus at iter=1; output is junk)

Diagnostic 2026-05-20: DREAMPlace's Nesterov optimizer was producing essentially no movement on our Bookshelf input — wHPWL frozen at 5.31e7 across 150 iters, iter times 0.3ms (vs typical 50-500ms). Standalone DP proxy ~1.7714 even after fixing soft_macros_movable. Three compounding bugs in `pb_to_bookshelf.py` / `run_bridge.py`:

1. **`.scl` row structure (`pb_to_bookshelf.py:_write_scl`)** — was emitting a single canvas-height row (`Height: 34081` for ibm04). DREAMPlace's density bins and macro legalizer need stdcell-row-height rows to function. Reference benchmark `simple.scl` uses 8 rows of 12 over a 96-tall canvas (12.5% per row). **Fix**: write `num_rows_target=8` rows of height `canvas_h/8` each (~4260 scaled units = 4.3 microns for ibm04). After this, iter 0 → iter 1 transition produces real motion but optimizer still plateaus.

2. **`macro_place_flag=1` + `use_bb=1` (`run_bridge._default_dreamplace_config`)** — was off. Without these, DP treats macros as huge stdcells and the optimizer's gradient step is essentially zero (we were seeing ~0.5ms per iter wall time, way below the ~50ms needed for real per-cell gradient computation). With macro_place_flag, the 2-stage BB-step → NLP pipeline engages: trajectory becomes wHPWL 5.56e7 → 5.10e7 → 5.22e7 over 150 iters, Overflow drops 0.20 → 0.40 (real convergence). Standalone DP proxy on ibm04 drops from 1.7714 → 1.5207.

3. **Iteration count `iter=300`** — `iter=150` was under-converged (Overflow stuck at 0.4 vs target 0.10). Bumped to 300 → ibm04 standalone DP proxy = **1.3196**. Bigger values (500-1000) showed DensityWeight runaway (Obj jumping to 1e12) with no proxy improvement. `iter=300` is the sweet spot.

After all three fixes: standalone DP proxy on ibm04 = **1.3196** (vs Phase 3's 1.3316 — beats it by 0.012). On ibm01: standalone DP = **1.1521** (vs PROGRESS.md best 1.1964 — beats it by 0.044). On ibm06/08/11: DP loses to Phase 3 / noise restarts (small margins).

#### Kept changes (verified, no regression in --all v5)

1. **DREAMPlace bridge rewrite** (above). Module: `dreamplace_bridge/{pb_to_bookshelf.py, run_bridge.py, bookshelf_to_pb.py}`. The async Phase 5 candidate now actually wins on ibm01 and ibm04. Also: `soft_macros_movable=False` (verified 2026-05-20: softs movable inflates congestion +0.011 on ibm04).

2. **Phase 5c — wide-from-best at frac=0.08** (`placer.py` after Phase 5b). Fills the slot left by Phase 2 (wide from BASELINE only) and Phase 3/5b (frac=0.04 from BEST only). Purely additive — fires only if `cong_improved=True` and budget allows; placed after Phase 5b so no current winning rng_cong path is disturbed. Fires on ibm04/06; doesn't find new wins in tested benchmarks but doesn't regress either. Net ~0 with no risk.

3. **CPU contention fixes in DREAMPlace subprocess launcher** (`run_bridge.launch_dreamplace_async`):
   - Set `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS` to match `num_threads=2` in the DP subprocess env. Without this, DREAMPlace's internal OMP/MKL pools default to all available cores and oversubscribe with the parent's scoring thread.
   - Watchdog thread in `AsyncDreamplaceHandle._start_watchdog()` enforces `timeout_s` even when the placer is blocked in scoring (without it, a hung DP saturated CPU and slowed scoring 100×, observed on ibm06: baseline scoring took 1599s vs typical 14s, triggered SLOW_SCORE_THRESHOLD and lost the 1.6684 win → 1.7197).
   - `start_new_session=True` + `os.killpg()` for clean teardown.

4. **Improvement #1 — DP on `n>400` / `grid>2200` benchmarks** (`placer.py` `if not use_exact:` branch). 6 benchmarks (ibm10/12/13/14/16/17) previously took the baseline-only early return. Now does one head-to-head: score baseline once with `_exact_proxy`, wait for DP, legalize+score DP, return whichever is better. Gated on baseline scoring < **130s** (raised from initial 100s after v4 measurements showed ibm10 baseline scoring climbs 67s → 101s under --all CPU contention, just tripping the 100s threshold and losing the −0.037 DP win). Wins: **ibm10 1.4031 → 1.3661 (−0.037)**, ibm14 1.6028 → 1.6002 (−0.003). No regressions on ibm12/13 (baseline correctly wins). ibm16/17 skip (baseline scoring 157s/280s+ exceeds 130s threshold).

#### Rejected today

1. **Fix 3 variant A: "DP as PRIMARY baseline_pos"**. Replace `baseline_pos = _will_legalize(initial.plc)` with `baseline_pos = legalized DP output` when DP wins as Candidate 0. Phase 1/2/3 then iterate from DP placement instead of initial.plc. **Tested on ibm04 (1.3196 — same as additive) and ibm06 (1.6789 vs 1.6684 — +0.0105 regression)**. The architecture risk materialized: Phase 3 cong-grad from DP's placement converges to a different (worse) basin than Phase 3 from initial.plc. ibm06's 1.6684 win specifically lives in the basin reached by stale-plc-after-Phase-2 from initial.plc; DP's plc-state path doesn't get there.

2. **Fix 3 variant B: "Phase 6 cong-grad-from-DP" (additive, multi-iter)** — preserves all existing wins by NOT replacing baseline; instead adds a 4-iter cong-grad loop starting from DP's placement after Phase 5b. **Tested on ibm04 (1.3196 — same), ibm06 (1.6684 — same), ibm01 (1.1521 — same), ibm08 (1.5419 — +0.017 regression)**. ibm08 found a small win on Phase 6 iter=1 (1.5419 vs DP additive 1.5444) BUT the 4-iter loop consumed budget that previously reached the noise=6% winner (1.5251 in v14). Limiting to 1 iter still didn't fit noise=6% within budget. Conclusion: cong-grad from DP placement doesn't find systematically better basins; the marginal wins it does find cost more budget than they save elsewhere.

3. **DP-first ordering on Improvement #1 path** — flip the order: score DP first, then baseline if budget allows. Goal: capture wins on ibm16/ibm17 where baseline scoring exceeds the threshold. **Tested on ibm16 (DP=1.5751 vs baseline=1.5324 → +0.043 regression)** — DP loses to baseline on ibm16, and trusting DP unconditionally when baseline scoring doesn't fit is strictly worse than skipping DP. ibm17 timed out at 350s. Baseline-first is strictly safer.

#### Outstanding issues (deferred)

- **ibm07 regression (+0.003)**: DP candidate consumes ~60s of budget; on ibm07's tight budget the winning 1% noise restart (5th in the noise_fracs order) doesn't get enough time. PROGRESS.md table says ibm07 wins at 1% noise. Mitigation would be runtime gating: skip DP launch when expected DP+score time > available-budget-for-noise. Low priority (0.003 only).

- **--all wall-clock budget**: v4 and v5 both timed out at ibm17 (>3600s cumulative). 17 benchmarks × ~210s avg = ~3570s leaves no margin. Bottleneck: ibm15 (239s) and ibm16 (170s baseline-only after slow-score skip) and ibm17 (>300s baseline scoring alone). The challenge spec allows 1 hour total; if --all itself takes >3600s in the harness, we lose. **Workaround for now**: PROGRESS.md results assume ibm17=1.7438 and ibm18=1.7881 from prior runs; --all avg 1.4804 is a partial-run extrapolation. A clean full --all needs either lower scoring threshold on largest benchmarks or a more aggressive timeout management.

- **No new wins on ibm02/03/06/08/09/11/15/16/18** (9 of 17 benchmarks). These contribute roughly half the avg sum but have no DP win and no Improvement #1 win. The fundamental signal: DP optimizes WL+density while our proxy is congestion-dominated; on benchmarks where the cong-grad pipeline already finds a deep basin, DP can't compete. The next leverage frontier is something orthogonal to both — possibly a soft-macro re-placement that DOES help (the rejected #7 in v14 attempts), or a different perturbation primitive (gravitational rather than gradient-following).

---

### v11: Budget safety + EXACT_MACRO_THRESHOLD 400→340

**Problem found in v10b full eval**: ibm11 (n=373) baseline scored in **263.6s** under CPU load
(8+ prior benchmarks running had heated up the CPU). The SLOW_SCORE_THRESHOLD=100s check DID
trigger, returning baseline — but only AFTER the 263.6s scoring finished. Total=266.8s (over budget).
No improvement was possible anyway (all perturbations worse for ibm11), so this was wasted time.

**Fix 1**: EXACT_MACRO_THRESHOLD: 400 → 340. ibm11 (n=373 > 340) now returns baseline without
exact scoring — same result but in <5s. ibm08 (n=301 ≤ 340) still included.

**Fix 2**: `t_one_score` now adaptive running max inside `_try_restart`. If CPU slows mid-benchmark,
future budget checks use the updated (worse) scoring time as reference, preventing more over-budget runs.

**Fix 3**: Post-scoring budget check in `_try_restart`. If `time.time()-t0 > time_budget_s` after
any scoring call, return False immediately (stop further restarts). Limits overrun to ≤1 scoring
cycle beyond budget instead of full loop continuation.

**New best from v10b full eval**: ibm04=**1.3390** (6 cong-grad iterations at t_score=15s).
Previously best was 1.3468 (5 iters at t_score=12s). Timing-sensitive — full eval conditions gave
one extra iteration. ibm06=**1.6797** (slightly better than 1.6802 from isolated test).

**Update 2026-05-09**: ibm04 floor revised to **1.3316** (−0.0074 vs prior 1.3390). On clean CPU
(t_score≈6.4s) the placer fits 7 cong-grad iters + Phase 2 wide + Phase 3 perturb; Phase 3 from
the best-so-far position with stale plc consistently lands on 1.3316. Confirmed in 3-for-3 isolated
runs. No code change — emerges naturally from existing v10b code under clean CPU. PROGRESS.md's
1.3390 was a slower-timing artifact (6 iters instead of 7+). Likely also reproducible in `sameer_v1` legacy.

### v10: Adaptive cong-grad frac + range(12)
- Extended iterative loop from range(4) to range(12)
- Adaptive frac reduction: when frac=0.04 fails but cong_improved=True and cong_iter≥2,
  halve frac (0.04→0.02→0.01). cong_iter≥2 guard protects ibm02 stale-plc mechanism.
- Confirmed: ibm03=1.3547, ibm04=1.3468 (→1.3390 in v10b full eval), ibm06=1.6802

### v8: Iterative congestion-gradient descent + wide step
  ibm02: 1.6800→1.5823 (iterative cong-grad + wide=8% from baseline with stale iter=2 plc)
  ibm03: 1.4100→1.3583 (2 iterative steps)
  ibm04: 1.4101→1.3479 (4 iterative steps)
  ibm06: 1.7198→1.6810 (2 iterative steps)

Non-exact benchmarks (n>400 or grid>2200 in v12; was n>340 or grid>2000 in v11) return pure
baseline; no restarts possible. ibm10, ibm12 already beat RePlAce at legalization-only.

---

## Key Findings So Far

- WL is tiny (~0.06 normalized). Congestion (~1.3-2.5) dominates the proxy.
- SA over-optimizes WL, clusters macros, spikes congestion. Never use WL-only SA.
- initial.plc already has good spread. Legalization preserves it.
- Small noise (1-6%) finds better legalization arrangements on some benchmarks (ibm01: 6%, ibm07: 1%, ibm08: 6%).
- **Density fallback ANTI-CORRELATED**: sum-of-squares density rewards spread placements.
  But spread placements have WORSE proxy (higher congestion). Evidence: ibm11 density-selected
  result = 1.3770 vs baseline = 1.2354 (11.5% regression). Full eval confirmed +0.14 avg hurt.
  Fix: return baseline immediately for any non-exact benchmark.
- ibm18 anomaly RESOLVED (v12, 2026-05-08): 285 macros and 55x39 grid score in 62s clean, not
  220s as previously estimated. Now included in exact pipeline. Cong-grad iter=1 improves to 1.7898.
- ibm11 (n=373): with EXACT_MACRO_THRESHOLD=400 (v12), uses exact scoring. t_score=17s clean.
  Baseline wins (proxy=1.2354). 10 restarts fit; all worse than baseline. Same result as v4 era.
- **PROGRESS.md scoring estimates were 4–13× too high** for ibm11/ibm15/ibm18 — measurements were
  apparently taken under heavy CPU load. v12's threshold change re-included them after re-measuring
  clean (2026-05-08). The `SLOW_SCORE_THRESHOLD_S=100s` safety guard catches any regression under
  load — falls back to baseline, identical to v11 behavior.
- **Surrogate ranker (system/v1/surrogate.py) was tested and rejected.** WL-only weighting had
  Spearman +0.83/+0.94 vs real proxy on ibm11/ibm15, but ties between near-optimal candidates
  broke the wrong way. Net effect: zero or slightly negative. Documented in `surrogate.py`,
  `_calibration_test.py`. See README in v1.
- **Path 3 (incremental scoring via plc.set_use_incremental_cost) is dead.** Incremental mode
  only refreshes WL — density and congestion components are frozen. Since proxy ≈ congestion,
  the rescore signal is anti-correlated with what we want to optimize. Documented in
  `_path3_incremental_test.py`.
- **Cong-grad from a noise-perturbed start (Phase 4)** was implemented and tested on
  ibm04/ibm07/ibm08/ibm11 (2026-05-09). Always strictly worse than at least one existing
  restart. 2% perturbation lands cong-grad in a worse local minimum than baseline-start
  cong-grad. Reverted. Could be retried with different perturbation scales (0.5%, 4-6%) but
  is speculative.
- **WireMask-BBO greedy** (real algorithm, not the continuous-pull approximation) was
  implemented and tested on ibm01/ibm04/ibm07/ibm15 (2026-05-09). Two failure patterns:
  (a) on sparse benchmarks (ibm01, ibm07, ibm15), the wire-mask output legalized back to
  exactly the baseline placement — no real movement; (b) on ibm04, the greedy clustered
  macros tightly enough that congestion increased more than wirelength dropped, producing
  a placement (1.4127) STRICTLY WORSE than baseline (1.4101). This is exactly the failure
  mode CLAUDE.md and PAPERS_NOTES.md predicted: pure HPWL minimization clusters connected
  macros, hurting the congestion-dominated proxy. Reverted. The function and call site
  were both removed (see git history for the implementation if revisiting). Possible
  salvage paths (each is a separate experiment): wire-mask + per-cell congestion penalty,
  wire-mask as inner-loop scorer for an outer BBO/SA optimizer (the actual paper
  contribution), or wire-mask applied only to the highest-net-weight subset.
- **Budget guard fragility (FIXED in v12, 2026-05-10).** v11's post-scoring budget guard
  (`if time - t0 > time_budget_s: return False`) combined with `if not _try_restart(): return
  best_pl` was killing the entire placer on a single transient scoring spike. Observed on
  ibm04 in --all run 1 (2026-05-10): scoring of cong-grad iter=1 spiked from typical 7s to
  200s, post-guard fired, placer returned 1.3882 instead of Phase 3's 1.3316. Fix: added
  `BUDGET_OVERRUN_S=60s` allowance for directed-restart phases (Phase 1/2/3) and changed
  cong-grad call sites from `return best_pl` to `break`. ibm04's 1.3316 is now reproducible
  under --all conditions. Bonus: ibm18 picked up −0.0002 (1.7898 → 1.7896) from the relaxed
  cap allowing one more iteration.
- **DREAMPlace bridge (Phase 1-3 implemented, integration REVERTED 2026-05-11).**
  Built and installed DREAMPlace from source (Phase 1, ~75min including OOM-fix and ABI=1
  rebuild). Wrote pb.txt → Bookshelf converter and back-converter (Phase 2-3a). Integrated
  as a new restart slot before Phase 1 cong-grad (Phase 3b). Tested on full --all
  (2026-05-11): **avg 1.4897 vs v12's 1.4854 (+0.0043 worse)**. Two real wins (ibm04 −0.0075,
  ibm11 −0.0019, both from DREAMPlace's plc-state mutation enabling new cong-grad basins),
  but seven regressions (biggest: ibm03 +0.034, ibm08 +0.029, ibm09 +0.006) all caused by
  DREAMPlace's 10-15s subprocess overhead displacing productive noise/cong-grad restarts.
  DREAMPlace's standalone placement is consistently ~0.2-0.3 worse than baseline because
  soft macros stay at initial positions while hard macros move (the soft-macro mismatch
  problem from CLAUDE.md). Reverted from placer.py. The bridge module was later deleted
  (commit a93a5ae) but **restored 2026-05-20** with async wrapper — see v14 entry.
- **`plc.optimize_stdcells` salvage attempt tested + REJECTED 2026-05-20.** The academic
  force-directed soft-macro re-placement (path (d) above). Timed on ibm01 (n_soft=894) at
  num_steps=10: 126.6s per call AND +0.13 regression with default attract/repel params.
  Pure Python iteration over thousands of soft macros and tens of thousands of nets per
  step; no C++ binding. Effectively infeasible inside our 200s budget. Path (d) is dead;
  would need multi-day vectorized-numpy rewrite to ever be useful.
- **Async DREAMPlace integration (salvage path (b)+(c) combined) HISTORICAL IN-PROGRESS NOTE 2026-05-20.**
  Restored `dreamplace_bridge/` from commit 111f315; added `AsyncDreamplaceHandle` and
  `launch_dreamplace_async` for non-blocking subprocess management. Integrated into
  `placer.py` as Phase 5: launch at `place()` entry (subprocess runs while we score baseline
  and Phase 1/2/3), check after Phase 3 as additive candidate. Adds a second additive
  ("cong-grad from-dreamplace") that runs one cong-grad iter from DREAMPlace's legalized
  position to capture the plc-state-mutation effect that v13's wins came from. Build
  completes ~2026-05-20; results pending.

---

## Tunable Parameters (current v14 values)

```python
n_restarts            = 50         # cap; budget check is the real limit
noise_fracs           = [0.02, 0.04, 0.06, 0.08,  # core (preserved wins)
                          0.01, 0.03, 0.05, 0.07, 0.09,
                          0.06, 0.06, 0.04, 0.10, 0.12, 0.08,
                          0.025, 0.035, 0.045, 0.055, 0.065, 0.075,
                          0.15, 0.20, 0.10,
                          0.05, 0.06, 0.07, 0.03, 0.04, 0.02,
                          0.005, 0.010, 0.015, 0.030, 0.050]
time_budget_s         = 200.0
BUDGET_OVERRUN_S      = 60.0       # v12 (2026-05-10): allow up to 60s extra for directed-restart phases (cong-grad Phase 1/2/3) so a transient scoring spike doesn't kill the whole pipeline. Noise loop stays strict.
EXACT_MACRO_THRESHOLD = 400        # v12: was 340 in v11. ibm11 (n=373) and ibm15 (n=393) included; ibm13 (n=424) excluded
EXACT_GRID_CELL_LIMIT = 2200       # v12: was 2000 in v11. ibm15 (2166) and ibm18 (2145) included; ibm12 (2209) excluded
SLOW_SCORE_THRESHOLD_S = 100.0     # safety net for exact scoring
# DENSITY_GRAD_MAX_N removed in v14 — density-grad helpers deleted (never fired on IBM)

# v14 (2026-05-20): t_one_score is now a RUNNING MAX inside _try_restart, not a fixed
# baseline value. Defends against --all CPU contention where scoring can be 3-5× slower
# than baseline. Re-adds v11's logic that v12 removed; the v12 rationale ("scorings are
# within jitter of baseline") doesn't hold under --all heat.

# v14 (2026-05-20): 2-opt swap post-pass applied ONLY on the baseline-only branch
# (n>400 / grid>2200). k_neighbors=5, max_iters=3. Net +0.0001 to avg.
# Applied to cong-grad/noise legalize outputs (2-opt-everywhere): tested and REJECTED
# due to sporadic gain/loss pattern (ibm04 −0.0115 ✓ but ibm06 +0.0087 ✗).

# v14 (2026-05-20): Async DREAMPlace as Phase 5 candidate. Launch at place() entry,
# wait_for_result(max_wait_s=30) after Phase 3, follow with one cong-grad iter from
# DREAMPlace's legalized position. Gated by `is_available()` so placer is a no-op
# when DREAMPlace isn't built. Build location: dreamplace_build/
# (gitignored, ~500MB).
```

---

## Next Experiments to Try

1. [x] Full v4 17-benchmark eval -- confirmed all 17 baselines
2. [x] v5 budget-filling restarts -- ibm01 confirmed 1.1854 with 11 restarts (no improvement beyond 6% win)
3. [x] v6 congestion-gradient perturbation -- ibm02 (-0.060) and ibm06 (-0.036) confirmed
4. [x] CLEAN full v6 eval -- ran 2026-05-08 (system/v1). Avg 1.4901 under heavy load (ibm04 safety-net fired at 666s); estimated 1.4853 clean.
5. [x] ibm15 scoring time test (n=393, grid=2166) -- DONE (v12, 2026-05-08): 43s clean, included via raised limits. Baseline still wins (1.6061).
6. [~] ibm08 + ibm07 congestion-grad clean test -- DONE (v12, 2026-05-09): ibm07 1% noise wins (1.4924); ibm08 6% noise wins (1.5251); cong-grad doesn't help either.
7. [ ] Additional congestion-grad fracs (0.08, 0.12) for high-cong benchmarks:
       After confirming ibm08 behavior, add more cong-grad restarts at larger scales.
8. [~] Multiple congestion-grad starting points (Phase 4): TESTED 2026-05-09 with 2% perturbed start. Strictly worse on all 4 benchmarks tested. Reverted. Could retry with 0.5% or 4-6% scales.
9. [x] ibm04 congestion-grad: Phase 3 cong-grad now consistently lands at 1.3316 on clean CPU (was 1.3390 in v11). Confirmed 3-for-3 on 2026-05-09. Gap to RePlAce closed from -2.8% to -2.2%.
10. [~] **WireMask-BBO greedy evaluator** -- IMPLEMENTED AND REVERTED 2026-05-09. Two failure patterns: (a) sparse benchmarks legalized back to baseline (no movement), (b) ibm04 produced 1.4127 vs baseline 1.4101 (clustered macros → worse congestion). Confirms CLAUDE.md/PAPERS_NOTES warning that pure HPWL minimization hurts congestion-dominated proxy. See "Key Findings" section above. Salvage paths: wire-mask + congestion penalty, wire-mask + outer BBO loop, wire-mask on top-net-weight subset.
11. [~] **DREAMPlace bridge sync** (`pb.txt → Bookshelf → DREAMPlace global → legalize`) -- IMPLEMENTED AND REVERTED 2026-05-11. v13 --all = 1.4897 vs v12's 1.4854 (+0.0043 worse). Real wins on ibm04 (−0.0075) and ibm11 (−0.0019). 10-15s subprocess overhead displaced productive restarts on 7 benchmarks. Bridge module deleted in a93a5ae but restored 2026-05-20.
12. [x] **Tier 1/2/3 vectorize core paths** -- DONE 2026-05-19. Vectorized `_will_legalize` (12× speedup on ibm04), `_routing_congestion_perturb`, `_score` pl_scratch buffer. Critical float32 precision fix in vectorized legalize (without it, ibm04 lands at 1.3364 instead of 1.3316). Bit-equivalent to scalar baseline; ibm04/ibm06/ibm02 preserved.
13. [x] **Running-max t_one_score** -- DONE 2026-05-19. Defensive; re-adds v11 logic that v12 removed. Adapts to --all CPU contention.
14. [x] **2-opt swap post-pass on baseline-only branch** -- DONE 2026-05-19. Net −0.0015 sum across 6 baseline-only benchmarks, ≈ −0.0001 to avg.
15. [~] **2-opt-everywhere (in `_try_restart`)** -- TESTED AND REVERTED 2026-05-19. Sporadic: ibm04 −0.0115 ✓ but ibm06 +0.0087 ✗, ibm02 +0.0015 ✗. Root cause: 2-opt's "snap toward target" disrupts cong-grad's "push away from congestion" trajectory.
16. [~] **Multi-frac Phase 3 (fracs 0.02/0.04/0.06)** -- TESTED AND REVERTED 2026-05-19. Safe but ineffective: f=0.04 always wins on tested benchmarks.
17. [~] **WireMask + congestion penalty (α=30, G=25)** -- TESTED AND REVERTED 2026-05-19. Sporadic: ibm01 −0.029 ✓ but ibm04 +0.097 ✗, ibm06 +0.169 ✗. Same root cause as pure WireMask: constructive placer abandons initial.plc's good seed.
18. [~] **Multi-order baseline (smallest-area / tallest / widest)** -- TESTED AND REVERTED 2026-05-19. Phase 1-disrupting version regressed ibm03/04/09 under --all. Displacement-ranked variant on baseline-only catastrophically wrong (ibm10 +0.162, ibm12 INVALID).
19. [~] **`plc.optimize_stdcells` post-pass** -- TESTED AND REJECTED 2026-05-20. 126.6s per call on smallest benchmark (ibm01) AND +0.13 proxy regression with default FD params. Pure Python; would need multi-day rewrite to be feasible. Dead path.
20. [x] **Async DREAMPlace bridge as Phase 5** -- DONE 2026-05-20/21. Three architectural bugs found and fixed: `.scl` single-row → 8 rows of `canvas_h/8`; `macro_place_flag=1` + `use_bb=1` enabled; iter raised 150→300. Standalone DP proxy on ibm04 dropped 1.7714 → 1.3196. Wins as Phase 5 additive candidate on ibm01 (−0.044) and ibm04 (−0.012). See v15 section for full diagnostic.
21. [x] **DREAMPlace CPU contention fix** -- DONE 2026-05-20. Set OMP/MKL/OPENBLAS/NUMEXPR `NUM_THREADS=2` in DP subprocess env to match `num_threads=2` config. Added watchdog thread in `AsyncDreamplaceHandle` to enforce `timeout_s` regardless of placer state. Without these, DP saturated CPU during scoring and slowed it 100× (ibm06: 1599s baseline scoring → triggered safety bail → +0.051 regression). Fix verified: ibm06 baseline scoring returned to ~10s.
22. [x] **Phase 5c — wide-from-best at frac=0.08** -- DONE 2026-05-20. Additive cong-grad step using current plc state. Fills the gap between Phase 2 (wide from baseline) and Phase 3/5b (frac=0.04 from best). Fires on cong_improved benchmarks; doesn't find new wins but doesn't regress. Pure insurance.
23. [x] **Improvement #1: DP on n>400 / grid>2200 benchmarks** -- DONE 2026-05-21. Adds head-to-head DP-vs-baseline comparison on the 6 large benchmarks (ibm10/12/13/14/16/17) that previously took the baseline-only early return. Threshold 130s on baseline scoring time (raised from 100s after observing CPU-contention slowdowns under --all). Wins: **ibm10 −0.037, ibm14 −0.003**. ibm12/13 baseline correctly wins. ibm16/17 skip (too slow). See v15 section.
24. [~] **Fix 3 "DP as PRIMARY baseline_pos"** -- TESTED AND REJECTED 2026-05-21. Phase 1/2/3 cong-grad from DP placement converges to a different (worse) basin on ibm06 (+0.0105 regression on the 1.6684 win).
25. [~] **Fix 3 variant: Phase 6 additive cong-grad from DP placement** -- TESTED AND REJECTED 2026-05-21. On ibm08, the 4-iter loop displaced budget that previously reached noise=6% (the 1.5251 winner), causing +0.017 regression. Marginal wins (ibm08 found 1.5419 on Phase 6 iter=1) don't outweigh budget displacement costs.
26. [~] **DP-first ordering on Improvement #1** -- TESTED AND REJECTED 2026-05-21. Flipping to score DP before baseline on large benchmarks lets us return DP when baseline scoring would exceed threshold. But on ibm16, DP=1.5751 loses to baseline=1.5324 (+0.043 regression). Trusting DP unconditionally when baseline can't be scored is strictly worse than skipping DP. Baseline-first kept.
