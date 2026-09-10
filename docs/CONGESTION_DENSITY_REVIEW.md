# Congestion and density review — 2026-09-08

## Individual implementation trials

The recommendations below were tested individually after the review. Acceptance
uses per-design proxy non-regression and preserved hierarchy, rather than only
a better suite average or a passing slack limit. Rejected implementations were
removed; their patches, controls, and full-precision results are archived under
`ml_data/proxy_individual/20260908/` (`ledger.json` records each actual control).
Early search changes use fixed-work controls and normal-deadline follow-ups.
The final pass uses paired evaluator comparisons against its own unchanged
search prefix, including the API's bounds clamp.

| Trial | Result | Decision |
|---|---|---|
| Shared checkpoint-cache repair | Focused IBM09/12 gains, but paired normal runs regress IBM01 by 0.000943 and IBM10 by 0.024574. | Reverted; the underlying cache defect remains open. |
| Non-mutating transfer density reads | Reuses `_density_field`; repeated heat reads preserve occupancy. Five focused scores unchanged. | Retained. |
| Independent density lane after congestion stalls | NVDLA regresses by 0.008437. | Reverted. |
| Soft footprint contribution to density-tail source order | IBM04/12/18 improve, but IBM09 regresses by 0.022466. | Reverted. |
| Incident-route contribution to congestion-tail source order | IBM09 regresses by 0.013685 and takes longer. | Reverted. |
| Bounded rescue of a wirelength-filter rejection | IBM09 regresses by 0.015177. | Reverted. |
| Skip leaf-topology nets with no clustered hard endpoint | Five focused placements, scores, and work counts are identical. The skipped pair loops contribute nothing. | Retained. |
| Full neighboring-leaf checks on hard micro winners | IBM04 improves, but IBM09 regresses by 0.011714. | Reverted. |
| Full contract on improving soft winners | Five fixed-work placements identical; extra audit time skips IBM12 strong repair under normal deadlines, costing 0.046272. | Reverted. |
| Remove consolidation's strict structural-gain gate | IBM04 proxy improves by 0.006112, but fragmented leaves rise 4→5 and foreign intrusions 26→35; seed-stage leaf limits also change. | Reverted for hierarchy deterioration. |
| Hard-blockage contribution to congestion-tail source order | IBM12 regresses by 0.004296. | Reverted. |
| Bounded final density relief for free soft macros | Paired evaluator checks improve 19/31 designs, leave 12 unchanged, and preserve every measured hierarchy value and hard coordinate. | Retained. |

The first two retained changes passed **112 verification tests** and all **31** full
hierarchy/legality checks, including independent NG45 tags and synthetic truth.
Headline averages remain **IBM 1.1881, NG45 0.7177, synthetic 1.4361**. A small
IBM17 difference begins in deadline-limited micro-shifts before either modified
function executes; these runs do not establish an end-to-end runtime gain.

The final density pass reuses the existing soft
relocation operator after checkpoint selection, freezes all hard macros, every
active/child/parent soft role, every soft bundle, fixed soft macros, and direct
hard-connected soft macros. It tests at most 64 candidates from 16 hot sources
and four targets, with a two-second search guard. Fresh baseline and candidate
scoring use the same `clamp_in_bounds()` helper as the API return. Only a
full-contract float32 gain above 0.000001 survives. The global checkpoint-cache
repair remains reverted; this pass initializes a separate scoring transaction.

The completed **2026-09-09 paired validation** scores each clamped input and
returned output with the external evaluator. Both states must be legal, with
identical hard/fixed coordinates, active/child/parent vectors, and island
metrics. No design regresses. All **113 verification tests**, **31 final
hierarchy audits**, **4 independent NG45 tag checks**, and **10 synthetic
truth checks** pass.

| Suite | Same-run input average | Output average | Improved / unchanged | Time in final pass |
|---|---:|---:|---:|---:|
| IBM | 1.188733634 | 1.188504272 | 16 / 1 | 3.079 s |
| NG45 | 0.717651725 | 0.717062488 | 3 / 1 | 0.216 s |
| Synthetic | 1.436111467 | 1.436111467 | 0 / 10 | 0.015 s |

IBM retains 68 moves from 1,053 candidate scores; NG45 retains 11 from 95.
Synthetic softs are fully assigned and therefore skip the new search.

| Suite | Mean Δ wirelength | Mean Δ density | Mean Δ congestion | Mean Δ proxy |
|---|---:|---:|---:|---:|
| IBM | +0.000010448 | −0.000246760 | −0.000232901 | −0.000229362 |
| NG45 | +0.000120595 | −0.001477316 | +0.000057745 | −0.000589237 |

Components may trade against each other while total proxy improves. These are
small attributable gains. Separate timed runs vary more: the earlier tail
sweep reached IBM 1.187981609, while its IBM13 prefix retained 20 micro moves
instead of the control's 24 and produced a raw cross-run regression. The
direct IBM13 comparison verifies **1.137223363 → 1.136410356**, with identical
hierarchy. IBM14/17 also have timing-sensitive prefixes. Do not compare the
paired input average with the earlier 1.1881 headline as if they were the same
placement, or claim an end-to-end speedup.

Final evidence: `paired_summary.json`, `12_*_paired/results.json`, saved
before/after placements, `12_verification.log`, and `validation12.log` under
`ml_data/proxy_individual/20260908/`. Its `results.md` gives reproduction commands.

## Original review

The best next placement work is better soft-macro proposals inside the existing
hierarchy regions, followed by targeted large-macro blockage relief. Before
tuning either, fix the two reproduced scoring-state defects below.
Keep hierarchy ownership, fixed macros, legal bounds, the six component limits,
per-leaf safeguards, and final audits unchanged.

This section records the review before implementation; the trial results above
supersede its proposed next steps. Evidence comes from the accepted 31-design
rebalance traces, a fresh four-design hotspot run, and two small reproductions.

## Current cost and useful work

The accepted IBM average is **1.1881**. From the evaluator's rounded component
reports, congestion contributes approximately **66.0%**, density **27.4%**, and
wirelength **6.6%**. Highest final proxies are IBM12 1.7030, IBM17 1.4599,
IBM18 1.3967, IBM15 1.3500, and IBM14 1.2826.

| Current IBM pass | Reported retained proxy gain, summed over designs | Time inside pass | Exact candidates |
|---|---:|---:|---:|
| Regional soft relocation | 1.698049 | 40.96 s | 318,940 |
| Interleaved soft repair | 0.226012 | 9.85 s | 57,459 |
| Region swaps | 0.258977 | 115.58 s | 982,478 |
| Strong soft repair | 0.051616 | 31.37 s | 324,800 |
| Post-relief small-leaf consolidation | 0.000104 | 26.39 s | 272 |
| Final internal leaf floorplan | 0.000597 | 48.80 s | 409 |
| Final void relocation | 0.000916 | 147.06 s | 1,253 |

These are stage-local telemetry values, not independent end-to-end ablations.
Changing an early pass changes later search. The cache issue also limits trust
in score attribution around rollback; micro-shift and transfer gains are
deliberately excluded from this comparison. A zero aggregate `scored` value
for adjacent transfer is an instrumentation gap: its `inter_scored` and
`intra_scored` counters contain real work.

## Prioritized findings

### 1. Correct shared scoring state before expanding search — two confirmed defects

**Locations:** `src/placer/plc/placement.py:45`,
`src/placer/scoring/incremental.py:2058`, and
`src/placer/pipeline/hierarchy_floorplan.py:1641`.

`_fast_set_placement()` skips writes when the requested coordinates equal
`_last_pos_cache`. Incremental `_apply_pos()` changes the actual PLC position
and global position cache without updating that last-applied cache. Scoring a
saved checkpoint can therefore skip a macro that actually moved and evaluate
the wrong coordinates. The rollback code computes its scalar score before
constructing the fresh scorer that forces a complete position reset.

The bounded IBM09 reproduction commits one movable soft macro, then requests
the original checkpoint. The macro remains **2.596667 µm** away from its
requested position. The reported proxy is **1.112744726**, whereas a forced
full restore gives the original **1.112624305**. The accepted full-run trace
also shows IBM09 reporting a micro-shift rollback score of 1.176747, followed
by a transfer stage ending at 1.278152 while reporting a positive local gain.
The reproduction confirms the underlying cache bug; it does not quantify the
entire downstream effect of fixing it.

**Smallest next change:** synchronize placement-cache validity at the shared
position-write boundary. Retain a regression check for incremental commit →
checkpoint restore → full score, including the actual coordinates. Do not add
separate workarounds at every rollback caller. This changes no hierarchy rule.

**Second defect: transfer heat mutates density.** In
`src/placer/local_search/adjacent_cluster_transfer.py:146`, `np.asarray(...).reshape(...)`
aliases the scorer's float64 `grid_occupied` array. The next line divides that
array in place by cell area. The pass calls this helper first for hard macros
and then for soft macros (`:469`), so occupancy is divided twice merely while
reading heat. A four-cell reproduction with area 4 changes occupancy
`[4,8,12,16] → [1,2,3,4] → [0.25,0.5,0.75,1]`. Proposal field weights and
source priorities then depend on those accidental rescalings. Candidate exact
components are computed separately, so this establishes corrupted proposal
state, not that every final evaluator score is incorrect.

**Smallest next change:** make normalization non-mutating, preferably by using
the existing `_density_field()` helper. Check that repeated hard/soft heat
reads leave occupancy unchanged. This also changes no hierarchy rule.

### 2. Improve soft source and target selection inside existing regions

**Locations:** `src/placer/local_search/relocation.py:1444`, `:1505`, `:1532`,
and `src/placer/local_search/fields.py:40`.

Soft relocation ranks a source by the field at its center, chooses globally
cold cells, then takes nearby targets. Its field combines maximum H/V heat
with density, each normalized by its grid maximum. That is useful but does
not measure the source's contribution to the evaluator's worst cells:

- A soft rectangle can cover dense cells while its center is relatively cool.
- A macro can drive routes through a distant hotspot while its center is cool.
- The nearest six cold target centers need not offer good capacity over the
  moved rectangle or relieve the congested routing direction.
- Source heat and target fields are computed once per invocation, so later
  proposals use heat from before earlier commits.

**Next experiment:** within the existing score allowance, prioritize soft
rectangle area in the density tail and incident-net demand in the directional
congestion tail. Preserve some current candidates as a control. Try nearby
cold targets inside the original soft regions; use the existing exact scorer
and full contract on improving candidates. Refresh target capacity after
meaningful retained moves. Begin with hard positions and all ownership frozen.
This improves the information used to propose moves; widening every search
quota is not supported by the present yield.

IBM18 is the clearest spatial target below. For owned softs, prefer targets
that preserve or reduce owner/bridge distance before considering candidates
that merely fit the existing contract allowance. Do not promote routing
cohorts into hierarchy ownership.

### 3. Give density an independent chance after congestion stalls

**Location:** `src/placer/pipeline/hierarchy_floorplan.py:3703` and the analogous
medium-soft loop below it.

The strong-soft loop tries congestion first and stops all remaining lanes
when that lane has weak retained gain. In the accepted IBM sweep, this prevents
even the first density lane on **IBM06, IBM08, and IBM12**. The same pattern
occurs on ariane133, mempool_tile, syn03_sram, and syn09_seedless.

Across the IBM lanes that did run, density retained **0.033931** gain from
139,611 exact candidates; congestion retained **0.017685** from 185,189.
Failure of one proposal field does not establish failure of the other.

**Small next experiment:** let the other field finish one bounded lane before
declaring both stalled. Keep the existing total allowance, full source tail,
audits, and deadline; do not restart extra rounds after both fail. Start on
IBM12 and use IBM06/08 as additional controls. This is an untested opportunity:
the older lane-stop policy had preserved scores under the July pipeline, so
the new policy needs current A/B evidence.

### 4. Audit the fixed wirelength cutoff for missed total-cost winners

**Location:** `src/placer/local_search/relocation.py:1557`.

Ordinary soft relocation rejects `delta_wirelength > 0.0001` before scoring
density/congestion; strong repair uses 0.0005. Regional soft relocation alone
rejects **68,115** targets this way in the accepted IBM sweep. This is a
heuristic, not a lower bound on total proxy: a wirelength increase of 0.0002
can be offset by a 0.001 congestion reduction.

**Next experiment:** exact-score a bounded sample of rejected, in-region
targets to measure false rejections. Only if those contain useful contract-safe
winners should the filter change, for example by reserving a small candidate
allowance for high-relief targets. Keep the batched wirelength calculation;
removing the filter wholesale could spend substantial time for no gain.
The rejection count alone is not proof of lost improvement.

### 5. Preserve valid move prefixes instead of discarding entire lanes

**Locations:** `src/placer/pipeline/hierarchy_floorplan.py:1519`, `:1559`,
`:1582`, `:1953`, and `:2014`.

Ordinary bulk hard relocation checks hard containment before scoring but
defers most vector/per-leaf checks to the pass checkpoint. Bulk soft relocation
normally has no per-candidate contract callback. The accepted traces discard
**0.080656** proposed soft gain on syn03_sram for bridge distance, **0.003378**
hard gain on NVDLA, and **0.001186** hard gain across IBM08/14. These totals are
discarded proposals, not fully recoverable improvements.

Micro-shifts already check improving candidates, but the hard callback limits
per-leaf checking to the moved macro's own leaf. A hard macro can also alter
another leaf's nearest-neighbor impurity. Full audits still roll back micro
lanes on IBM04/09. `hierarchy_quality.py:89` shows that impurity compares each
leaf against all clustered hard macros.

**Next change after the cache fix:** check the full affected contract on
exact-improving winners before commit, including affected neighboring leaves;
preserve the valid sequential prefix. For soft moves with hard positions
frozen, reuse the unchanged hard metrics and check the affected soft terms.
Do not calculate the complete expensive audit for every losing target and do
not weaken the final rollback gate.

### 6. Spend less on broad leaf rearrangement; target real blockage

**Locations:** `src/placer/local_search/cluster_consolidation.py:243`,
`src/placer/local_search/cluster_internal_floorplan.py:372`,
`src/placer/local_search/cluster_void_relocation.py:681`, and
`src/placer/local_search/relocation.py:1185`.

The two final leaf passes spend **195.86 s** for only **0.001513** summed IBM
gain, approximately **0.000089** in suite average. They can still be useful on
other inputs: void relocation retained 0.002143 on NG45. Use actual hotspot
opportunity and capacity to gate their expensive topology/legalization work;
do not remove ownership or hierarchy modeling with the search passes.

Small-leaf consolidation retains another redundant hierarchy objective: it
requires a strict composite hierarchy improvement before checking the full
contract and exact proxy. Across seed assembly and its replay, **435** candidates
stop at that structural test. The post-relief replay retains only 0.000104 IBM
gain in 26.39 s. A bounded control can test contract-preserving translation
without requiring extra compactness; no missed winner has yet been established.

For congestion relief, prioritize a few legal moves of hard macros whose
footprints actually block bad routing cells. IBM12's `a12317` and `a26309` are
large **unclustered** hard macros implicated in its central hotspot. The
existing hard-relocation operator already handles movable unclustered macros;
improve its source ranking instead of adding another whole-leaf floorplanner.
All fixed-macro, overlap, bounds, and complete hierarchy checks still apply.

## Fresh physical targets

Coordinates below are canvas fractions, `(x0,y0)–(x1,y1)`. These are bounding
boxes of the highest-pressure connected tail component, not an assertion that
every cell inside the box is hot.

| Design | Congestion component | Density component | Useful next direction |
|---|---|---|---|
| IBM12 | (0.32,0.21)–(0.68,0.68) | (0.32,0.32)–(0.74,0.81) | Separate soft overlap relief from large-hard blockage. Worst density component is 52% soft area; dominant-direction congestion has 17% macro blockage. |
| IBM18 | (0.44,0.05)–(0.65,0.46) | (0.40,0.08)–(0.47,0.21) | Soft relocation within leaf 4 and existing residual regions. Worst density component is 100% soft area; main congestion component has no hard blockage. |
| IBM17 | (0.63,0.45)–(0.88,0.84) | (0.16,0.20)–(0.27,0.30) | Preserve time for separate route and density lanes; these hotspots are spatially distinct. |
| IBM10 | (0.60,0.22)–(0.71,0.49) | (0.55,0.12)–(0.75,0.51) | Local route/blockage relief; worst density component is 64% soft area and congestion has 15% macro blockage. |

The fresh run returned 4/4 valid placements and passing final hierarchy audits.
IBM12 reproduced its accepted 1.7030. IBM17/18/10 were respectively
1.6464/1.4036/1.2055 rather than accepted 1.4599/1.3967/1.1744 because the
deadline-dependent schedules differed. IBM17 returned a hierarchy checkpoint.
Use those three maps as diagnostic placements, not accepted-control maps or an
algorithm comparison. The accepted 31-design results remain the baseline.

The hotspot script's `internal` route class means a net has exactly one
assigned owner label; unlabeled endpoints may still be present. Its route
attribution is raw demand in a smoothing-expanded neighborhood. It is not a
causal fraction of exact proxy cost or proof that every such net is internal
to a confirmed IP. In particular, the August claim that unassigned soft/IO
dominates is not a valid current classification after soft-role propagation.

## Implementation order and acceptance

1. Fix shared placement-cache coherence and non-mutating density reads; verify
   checkpoint scoring and scorer-state preservation.
2. Test one independent density lane on the existing schedule.
3. Test footprint/tail-informed soft proposals with hard positions and labels
   fixed; separately sample wirelength-filter rejections.
4. Improve candidate contract checking to retain valid prefixes.
5. Gate low-yield leaf work and investigate a few hotspot-driven hard moves.

Compare final exact wirelength, density, and congestion on matched inputs and
cache/deadline settings. Report all hierarchy vectors and per-leaf metrics
against the control: passing an existing slack limit alone does not establish
that hierarchy improved or stayed numerically identical. Promote only changes
that preserve the intended hierarchy, pass legality and independent NG45/tag
and synthetic/truth checks, and improve final suite proxy. No benchmark-name
branch, enlarged hierarchy slack, new hierarchy tier, or learned model is
needed for these experiments.

## Reproducible evidence

- Accepted source traces: `ml_data/proxy_rebalance/20260908/{ibm,ng45,synthetic}_final.jsonl`
- Rounded evaluator components and pass summaries:
  `ml_data/congestion_density_review/20260908/{accepted_components,ibm_passes,ibm_quotas,contracts}.json`
- Fresh maps, placements, and trace:
  `ml_data/congestion_density_review/20260908/20260908T165933*`, `hotspots.log`, `hotspots.jsonl`
- Cache reproduction: `ml_data/congestion_density_review/20260908/reproduce_cached_restore.py`
  and adjacent JSON/log.
- Density-read reproduction: `ml_data/congestion_density_review/20260908/reproduce_transfer_density.py`
  and adjacent JSON.

Run the cache reproduction with:

```bash
rtk uv run python ml_data/congestion_density_review/20260908/reproduce_cached_restore.py
rtk uv run python ml_data/congestion_density_review/20260908/reproduce_transfer_density.py
```

The fresh hotspot run reused `test/diagnostic/analyze_final_hotspots.py` with
its benchmark list limited in memory to IBM12, IBM17, IBM18, and IBM10. No
production settings or benchmark inputs were edited.
