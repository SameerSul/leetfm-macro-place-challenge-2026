# VivaPlace v2 — Architecture

## Overview

`MacroPlacer.place()` always routes through `_hierarchy_floorplan()` in
`src/placer/pipeline/macro_placer.py`. There is no proxy-only fallback path:
the placer raises if grouped DREAMPlace is unavailable. Exact proxy is the
optimization objective within the existing hierarchy and legality constraints.
Seed ranking pays no premium for additional hierarchy headroom. Terminal
cluster isolation and its hard/soft repolish sequence are removed.

The constructor exposes keyword-only `seed`, `event_sink`, and
`dreamplace_sample_every`. The old restart/noise options and overall
`time_budget_s` were unused and are removed. Existing per-pass time limits and
exact-score quotas control runtime. `src/place_design.py` no longer accepts
the ignored `--budget`; the synthetic runner supports an explicit budget
override only for alternative placers that expose `time_budget_s`.

The maintenance cleanup also removes unused local-search arguments, a dead
environment parser, and an always-true coldspot callback. The effective
coldspot candidate counts, search order, scoring, and hierarchy gates remain
the same. Current validation is recorded in [PROGRESS.md](PROGRESS.md).

Placement and diagnostic settings are source constants in
`src/utils/constants.py`; process environment variables cannot override the
seed, backend, quotas, weights, diagnostic selection, or telemetry destination.
The evaluator entrypoint keeps a local adapter class for class discovery and
inherits the explicit `seed` constructor argument without reading `SEED`. Diagnostic CLIs assign constants in their own
process; spawned audit workers receive the trace path explicitly. Defaults and
all 16 per-pass quotas are unchanged. DREAMPlace progress sampling travels in
its JSON input. Native subprocess paths/thread pools and the fixed cuBLAS
workspace setting remain library setup, not placement configuration.

The September 12 cleanup removes rejected CUDA overlap filtering and its
selector, the regressing graph-corridor region-expansion bias, and three unused
single-macro scorer methods (`score_move`, `score_move_soft`, `commit_move_soft`).
Production keeps its existing NumPy/Numba legality filters, batched and group
scoring, ordinary cold-component region ranking, deterministic quotas, and
complete hierarchy gates. The test-only soft-role alias and completed
ablation switches are removed; their callers use the canonical APIs.

The optional live visualizer adds a dependency-free schema-v1 event sink at
the placer/scorer boundary. With no sink (the evaluator default), no events,
Qt imports, trace writes, DREAMPlace progress output, or cache changes occur.
With a sink, accepted scorer commits emit only changed output-tensor indices,
old/new centers, and exact committed wirelength/density/congestion/proxy values;
the pipeline adds the production hierarchy composite and stage/checkpoint
events. Audit restoration emits a full rollback state. The GUI runs outside the
placer in the parent process and therefore cannot participate in candidate
ordering or acceptance.

```text
benchmark input
  -> build HierarchyModel
       - infer hard-macro clusters from connectivity (or RTL instance-path
         prefixes when names provide useful coverage, e.g. NG45)
       - refine a nearly all-covering single flat component from shared
         low-fanout hard-to-soft affinity, with a strict graph-cut fallback
       - retain exactly one parent/child level without recursive discovery:
         use the nearest useful explicit path ancestor, retain an original
         connectivity component above existing split leaves, or bisect an
         eligible active cluster once from structural edges reinforced by
         initial placement proximity, local density, and placed wire pressure
       - classify soft macros as owned (one dominant cluster) or bridge and
         record independent repeated-support/ambiguity confidence
       - preserve explicit soft instance-path bundles when names expose them
       - record inter-cluster edge weights and confidence
  -> grouped DREAMPlace global placement (synthetic clique nets per cluster)
  -> cluster-consecutive hard legalization
       - reserve every fixed obstacle before placing any movable macro, including recurrent seed anchors
  -> exact-proxy seed portfolio selection
       (grouped DREAMPlace, legalized initial.plc, two Re²MaP-inspired
        recursive prototypes, a conditional initial-anchored recurrent repair,
        and an explicit-tag route-channel candidate;
        every candidate gets a complete
        hierarchy vector, the vector is constrained component-by-component
        relative to legalized initial.plc, except that a legal raw reference is
        retained for single-component affinity refinement and an illegal raw
        reference falls back to grouped DREAMPlace; a mandatory lower-proxy
        candidate that misses one component may be repaired toward the passing
        reference when at least 95% of its displacement survives; the lowest
        exact-proxy contract-passing candidate enters relief)
  -> congestion-expanded hard/soft hierarchy regions
  -> seed-stage small-leaf assembly
       - target ordinary 2-8-hard leaves, independent of parent inference
       - compact hard members and pull directly owned movable softs inward
       - test whole-leaf slot exchanges to repair interleaved cluster ordering
       - legalize affected hard sets and require hierarchy-composite plus exact-proxy gain
  -> freeze confidence-calibrated hierarchy islands
       - explicit leaves and inferred leaves with confidence >=0.65 are strict islands
       - medium-confidence leaves receive wider per-colour limits; low-confidence evidence is advisory
       - intersect strict hard/owned-soft movement boxes with the compact post-assembly footprint
       - constrain spread, bounding span, and nearest-neighbour colour impurity per leaf
       - retain fragmentation, owned-soft p90 distance, and foreign intrusion as per-leaf audit telemetry
  -> exact-gated local micro-shift polish
       - check the complete contract only for exact-improving winners
       - retain the best valid sequential prefix instead of rolling back a whole lane
  -> region-locked hard relocation + soft cleanup
       - reject hard candidates that exceed the selected seed's inexpensive
         hard-containment limit before exact batch scoring
       - keep the complete six-component checkpoint authoritative after the pass
       - keep strict hard and directly owned soft macros inside their frozen island boxes
       - on refined graphs with at most 64 hard macros, reject individual hard
         and soft candidates that fail the complete contract before exact scoring
  -> small-leaf assembly replay after ordinary region relief
  -> parent-bounded child-cluster search
       - keep the production DREAMPlace/leaf partition unchanged
       - require confidence >=0.65 for inferred child search; explicit parents bypass the gate
       - rigidly translate hot child groups toward cold space inside the parent
       - co-move every leaf-owned movable soft macro; bridge softs stay independent
       - test sibling slot swaps inside the same parent
       - when a rigid state is blocked, compact and legalize only the affected children
       - enforce child and parent hierarchy contracts before exact mixed-group scoring
       - activate the multilevel contract for later passes only after a move is retained
  -> exact-gated cluster decompression with composite hierarchy quality
       - large designs can order opportunities by hierarchy graph tension
  -> adjacent-cluster ownership transfer
       - consider only persistent-graph adjacent leaves in the same retained parent
       - derive revision-scoped utilization, side capacity, heat, internal/external demand, frontier, and cut-delta features
       - propose hard-hard, soft-soft, hard-soft swaps and one-way hard/soft relocation into cold destination-leaf cells
       - grow bounded connected frontier bundles and translate them rigidly into capacity-safe cold targets
       - stage position and ownership changes across active/child membership, soft ownership, graph nodes, and the shared active-edge list
       - preserve at least one hard macro in every source leaf
       - require the complete dynamic hierarchy/island contract after reassignment
       - require exact proxy gain >= 0.00035 and density-plus-congestion contribution gain >= 0.0005
       - commit ownership only after all gates pass; otherwise restore the previous hierarchy maps and graph roles
       - after a retained transfer, run one bounded same-cluster hard-hard/soft-soft swap and hard/soft relocation round
       - rebuild capacity-directed hard/soft regions from the committed ownership revision
  -> budget-aware interleaved soft repair
  -> region-bounded hard-hard / hard-soft / soft-soft swaps
       - hard-moving swap candidates must stay inside the hierarchy audit budget
       - exact-score two stable prefixes before the untouched suffix
       - optional micro-shift replay after each swap round
  -> post-swap micro-shift replay
  -> post-swap hard propose-all relocation with spare-budget additive candidates
  -> telemetry scheduler skips the duplicate ordinary post-swap soft pass
       - two clean attributable full suites produced 0 accepts in 34 runs
       - its time remains as deadline and final-audit headroom
  -> plateau-triggered compound related-soft relocation
       - form groups only from explicit high-confidence path bundles
       - keep every member inside its individual hierarchy region
       - preserve group-relative geometry while testing pair/quartet/full-group shifts
       - enforce the six-component hierarchy contract before scoring
       - exact-score and accept only the completed multi-soft state
  -> plateau- and component-aware strong soft repair when telemetry shows useful spare work
       - medium/large soft continuation runs only when structural shape and prior soft gain justify it
  -> coldspot tightening:
       - refresh current congestion field and cold-cell graph memory
       - generate coldspot kick candidates
       - co-move owned/bridge soft macros
       - legalize candidate hard macros
       - expand local graph border through adjacent open cold cells
       - apply hard-core padding
       - run graph-bordered local swaps and hard/soft relocations
       - exact-rank refined outcomes with deterministic graph tie-breaks
       - exact proxy + hierarchy-quality gate before commit
       - large designs can rank hot clusters by hierarchy graph tension
  -> graph-local fallback when no coldspot kick commits:
       - select hottest eligible current clusters
       - reuse the same graph-expanded border
       - run the same swaps and relocations without a kick
       - exact proxy + hierarchy-quality gate before commit
  -> post-coldspot micro-shift replay
  -> structurally eligible small-design polish:
       - seed release candidates with weakest-k inferred hierarchy clusters
       - keep only clusters below the confidence threshold
       - release the hottest eligible weak clusters, capped by max clusters and weakest-k
       - no-release low-net small designs shift candidate breadth toward soft relocation and soft-involving swaps
       - build cold connected-component target pools
       - run bounded hard/soft relocation, hard swaps only after useful released hard relocation, soft-involving swaps, and micro-shift polish
       - restore the best audit-passing exact-scored state seen inside the small-design pass
       - exact proxy, hard legality, and hierarchy audit budget remain the commit gates
  -> topology-aware internal leaf survivor floorplanning
       - derive a weighted induced graph for each eligible leaf directly from low-fanout nets
       - use spectral coordinates to keep strongly connected hard macros adjacent
       - classify macros by external-demand ratio and place them on the leaf side facing their weighted external centroid
       - reserve a demand-scaled inset at those boundary ports and test 65%, 55%, and 45% utilization layouts
       - move directly owned soft macros toward hard-affinity barycentres inside frozen island boxes
       - affected-set legalize, enforce the complete hierarchy/island contract, and exact-score the complete hard/soft state
       - schedule after the main congestion search so a small local win cannot replace a stronger downstream trajectory
  -> inter-macro void relocation survivor
       - preserve the verified opposing-edge corridor prefix and also extract hard-clear fragments between large macros and canvas edges
       - subtract intervening hard blockages before soft-only occupancy and track interior/edge provenance
       - identify existing clusters that own the large hard macros bounding each clear fragment
       - move the physically aligned boundary band 25%, 50%, or 75% into the corridor, backing off until float64 and returned-float32 hard legality pass
       - also build a graph-tapered decompression candidate: the boundary band takes the full shift while low-fanout hard-graph followers move 55% per hop for at most three hops; disconnected cluster macros remain fixed
       - reject projected void occupancy above 78% before exact scoring so the follower wave fills the vacated interior without saturating the opening
       - co-move only the owned soft band aligned with the same opening, leaving the cluster interior fixed so its footprint expands outward
       - rank hot 2-12-hard leaves, stable residual soft bundles, transient low-fanout routing cohorts, and hot residual singletons against cold voids
       - place the rigid hard leaf at its weighted hierarchy-graph-neighbour centroid projected into the void
       - co-move and compact at most 32 directly owned soft macros inside the same void
       - place residual soft units toward their weighted external-pin centroid using rigid or deterministic non-overlapping shelf layouts
       - keep routing cohorts pass-local: they never create hard ownership, a hierarchy label, or a parent/child relation
       - reject projected utilization above 78%, soft/hard or internal cohort overlaps, and complete hierarchy/island contract failures before exact scoring
       - exact-score at most 96 states; hard moves require 0.0001 gain and soft-only moves require 0.00005
       - among contract-passing expansions above the exact-gain floor, prioritize the largest visible outward displacement; then recompute voids and allow disjoint rigid hard or soft-only fills
  -> gain-controlled passes: stop repeats when latest exact gain <= HIER_PLATEAU_PROXY_GAIN
  -> final scorer-compatible hard legality margin audit
  -> final hierarchy-quality audit against the selected hierarchy seed:
       - enforce both the legacy hard-cluster budget and independent limits for
         all six rich-vector components
       - roll back to the best saved audit-passing checkpoint when needed
  -> bounded final density relief for free soft macros
       - freeze all hard macros, fixed softs, active/child/parent soft roles, soft bundles, and direct hard-connected softs
       - reuse region-bounded soft relocation: 16 hot sources, 4 targets, at most 64 candidate scores, 2-second search guard
       - independently rescore baseline and proposed float32 output; retain only a full-contract gain above 0.000001
  -> bounded final cluster-tile rearrangement
       - attribute directional routing-tail, hard-blockage, and density pressure through complete incident nets
       - reuse the canonical location graph to form 2–4 connected units within one leaf/child partition
       - keep fixed and bridge macros frozen; move complete explicit bundles rigidly
       - rank joint assignments to current slots and nearby cold tiles, then exact-score at most 16 per patch and 64 total
       - spend at most 3 seconds of unused internal-floorplan/void search allowance
       - retain only a complete-contract float32 gain above 0.000001 after fresh baseline/winner scoring
  -> final legality and bounds checks
  -> return center coordinates for hard and soft macros
```

The two recursive prototype candidates are an independent, bounded adaptation
of Re²MaP's recurrent prototyping idea ([reference 28](REFERENCES.md)).
After ordinary grouped DREAMPlace is legalized, round one freezes one complete
movable 2–16-hard leaf chosen deterministically by hierarchy confidence, footprint
utilization, member count, and stable leaf ID. Round two freezes one additional
eligible leaf from the first recursive result. Fixed-containing and
zero-confidence leaves are ineligible. The remaining hard and soft macros are
re-prototyped from real nets and the existing grouping cliques; no guide net,
ellipse constraint, packing-tree search, or imported clustering is used.
Temporary names and coordinates, exact grouping membership, and group weight
are cache-separated. Both results enter the ordinary immutable prefilter,
legalization, soft cleanup, exact scoring, and component-contract selection.
A third recurrent candidate bridges a different failure mode. When grouped
DREAMPlace fails the complete component contract but improves exact proxy over
the legalized initial seed by at least 15%, VivaPlace freezes two to four
positive-confidence leaves at their distributed initial positions and reruns
grouped DREAMPlace for the remainder. The anchors are selected strongest-first,
then normalized farthest-first across the canvas. The candidate uses the same
immutable prefilter, legalization, soft cleanup, exact scoring, selection, and
final rollback as every other seed. The gate avoids paying for another global
solve when grouped DREAMPlace is already eligible or offers too little upside.
A deterministic hierarchy-leaf B*-tree compaction seed was also evaluated and
removed after selecting on 0/17 IBM designs. Its implementation and dedicated
tests were pruned; production does not execute packing-tree relocation or the
upstream evolutionary search.

## Live diagnostics architecture

`src/visualizer/main.py` starts the placer with the multiprocessing `spawn`
context. A bounded queue transports plain dictionaries to the main Qt process.
The consumer writes every event to schema-v1 JSONL before its 30 FPS renderer
coalesces rapid updates. Traces start with metadata and a full placement, store
accepted moves as deltas, add full keyframes at algorithm-stage and explicit
pipeline boundaries and every 250 moves, and tolerate a truncated final line
during replay. Pausing or scrubbing affects only rendering; the queue continues
to drain and every event continues to be recorded. Trace playback supports
0.02× through 8× speed, including explicit 10× and 50× slower modes, and can
resume from any selected timeline event.

Completed or partial traces can be exported to H.264 MP4 either from the replay
window or with `--export-mp4`. Frames capture the full dashboard and its current
layer configuration at the selected replay speed. Encoding is deliberately
replay-only: it cannot stall the live consumer and back-pressure the bounded
worker queue. ImageIO streams frames to its packaged FFmpeg executable, and an
atomic temporary-file replacement prevents a failed or cancelled export from
overwriting an existing demo.
For speeds below 1×, the exporter lowers the encoded stream frame rate rather
than duplicating identical images. Each recorded placement state is rendered
once, the requested slow duration is preserved, and CLI exports report frame
progress plus an estimated completion time.
The launcher accepts either an explicit trace or a trace directory, selecting
the directory's newest JSONL file deterministically by modification time and
filename. Replay paths are validated before Qt starts, avoiding GUI tracebacks
for placeholders, missing files, or empty directories.

In visualizer mode only, `run_dreamplace()` bypasses final-position cache reads
and writes and consumes `VIVAPLACE_PROGRESS` records from `Popen`. The tracked
bootstrap patch samples float32 lower-left movable-node coordinates every ten
optimizer updates by default. The bridge converts Bookshelf order/scale/size
back to VivaPlace output indices and centers. These frames retain the previous
metrics with a visible stale badge; the next exact scorer checkpoint refreshes
them. A malformed frame disables subsequent progress decoding without turning
an otherwise successful DREAMPlace process into a placement failure.

The tracked patcher owns protocol version 1 and normalizes both
`dreamplace_src/dreamplace/NonLinearPlace.py` and the installed copy. Bootstrap
applies it before compilation and after installation and recognizes the pinned
upstream commit plus the legacy and current local patch commits. Bootstrap
`preflight` runs the patcher's `--check` mode before native import checks, so
stale, partial, duplicated, or unsupported insertion state is reported without
writing files. Explicit patch targets follow the bootstrap source/build paths;
missing files fail verification.

Real-net metadata is extracted from the scorer's global pin/net arrays after
the PLC is loaded. The dashboard stable-ranks low-fanout/high-weight nets, adds
all nets incident to the selected macro, and uses macro pin offsets and fixed
I/O endpoints when available. Repeated synthetic grouping nets collapse into a
single dashed centroid/spoke hyperedge per leaf cluster. Hierarchy-model edges
remain a separate, default-off centroid layer.

Passes advance on gain, not fixed repeat counts: each stage keeps running
while its most recent exact-proxy improvement exceeds
`HIER_PLATEAU_PROXY_GAIN`, then moves on.

There are no Boolean switches around promoted production behavior. BB and
DREAMPlace cache reads, component-aware expansion/decompression, decompression
feasibility and survivor handling, graph-mask fallback, adaptive gain control,
cold-component targets, structurally eligible small/medium soft polish, final
audit rollback, and plateau telemetry always run when their data, structural,
budget, and safety preconditions apply. Default-off research hooks remain
separate experiments. The former `HIER_DREAMPLACE_BB`,
`HIER_DREAMPLACE_CACHE`, `HIER_ADAPTIVE_PASSES`, `HIER_PLATEAU_TRACE`, and
`HIER_PLATEAU_TRACE_BUFFERED` feature switches are not read by production code;
legacy values cannot disable the selected behavior.

```text
proxy_cost = wirelength + 0.5 * density + 0.5 * congestion
```

Exact proxy guides optimization and evaluator reporting, subject to the
existing hierarchy and legality contracts. Extra hierarchy compactness does
not justify a proxy increase. The required structure and constraints are
documented in [OBJECTIVES.md](OBJECTIVES.md).

The structural objectives that drive the hierarchy flow are documented in
[OBJECTIVES.md](OBJECTIVES.md). The retired learned-GNN stack is preserved only
as historical experiment results in [PROGRESS.md](PROGRESS.md); its inference,
candidate tracing, offline training, artifacts, and dedicated documentation
are no longer part of the system.

The final ordinary-guard run matches the independently audited cluster-tile
replay on all 31 placements. The replay improves four designs and leaves 27
unchanged while preserving legality and the complete hierarchy contract.
Current validation and maintenance checks are in [PROGRESS.md](PROGRESS.md);
paired scores, source snapshots, and timing controls are in the
[algorithm review](../ml_data/algorithm_review/20260909/results.md).

The graph-tension signal is advisory and applies to structurally eligible large designs. It
orders decompression/coldspot opportunities but does not change commit gates.
Direct graph-tension swap ordering remains available through
`HIER_GRAPH_TENSION_SWAP_WEIGHT`, but defaults to `0.0` after focused tests
regressed `ibm08` and `ibm10`.
Swap candidate ranking uses temporary graph-derived masks and soft mask
penalties whenever a graph mask is available:

```text
HIER_SWAP_GRAPH_MASK_MAX_EDGES=0
HIER_SWAP_GRAPH_MASK_PAD_CELLS=1
HIER_SWAP_GRAPH_MASK_PENALTY_WEIGHT=0.30
HIER_SWAP_GRAPH_DELTA_WEIGHT=0.0
HIER_GRAPH_TENSION_CORRIDOR_SAMPLES=9
HIER_SWAP_GRAPH_FALLBACK_BUDGET_S=2.5
```

These controls are diagnostic/ranking only in default mode; no candidate is
rejected for violating the mask, and final commit still requires hard legality,
hierarchy quality, exact proxy gain, and the active audit checks.
Decompression uses graph-edge stretch, corridor congestion, and weighted
edge-length deltas to identify its bounded graph-survivor polish candidates.
Coldspot candidates retain exact-proxy ordering and the ordinary graph-score
tie-break.
Decompression always estimates the proposed bbox's free area and neighbor
blockage before legalization and exact scoring, and logs `feasibility_blocked`
rejects.
The graph-survivor path is narrower: for legal,
hierarchy-safe decompression candidates that miss exact proxy by a small amount
while improving graph-edge geometry, it exact-scores a tiny hard/soft local
polish pool around the moved cluster. It commits only if the final candidate
clears the normal exact-proxy gain and audit gates. Historical score and timing comparisons are retained in
[PROGRESS.md](PROGRESS.md). They span different hierarchy contracts and search
schedules and must not be used as interchangeable current baselines.

## Main Components

| Path | Role |
|---|---|
| `src/main.py` | Evaluator entrypoint. Exposes `MacroPlacer`; applies `SEED` only. |
| `src/utils/constants.py` | All tunable hierarchy constants (see appendix below). |
| `src/placer/pipeline/macro_placer.py` | Production flow entrypoint; raises if `_hierarchy_floorplan()` cannot run. |
| `src/placer/pipeline/hierarchy_floorplan.py` | The hierarchy pipeline itself: seed portfolio, region relief, swaps, and coldspot cleanup. |
| `src/placer/local_search/cluster_consolidation.py` | Bounded small-leaf hard/owned-soft compaction and whole-leaf slot exchange with structural, contract, and exact-proxy gates. |
| `src/placer/pipeline/hierarchy_context.py` | Shared `PlacementState`, `PassContext`, `PassResult`, `PlateauTelemetry` used across pipeline stages. |
| `src/placer/local_search/hierarchy_model.py` | Inferred hierarchy: active clusters, one parent/child level, soft roles, graphs, region builders. |
| `src/placer/local_search/location_graph.py` | Persistent macro/cluster topology with live coordinates, sizes, ownership, adjacency, cluster centroids/bounds, graph-hop, and spatial-box queries. |
| `src/placer/local_search/soft_hierarchy.py` | Confidence-calibrated soft bundles: explicit instance paths can be active; flat-netlist connectivity and affinity remain diagnostic evidence. |
| `src/placer/local_search/hierarchy_quality.py` | Complete hierarchy vector, including cached-JIT stable nearest-four impurity selection. |
| `src/placer/local_search/clusters.py` | Hard-cluster derivation, oversized-cluster splitting, region-box primitives. |
| `src/placer/local_search/relocation.py` | Hard and soft relocation used by region-locked relief and post-swap polish. |
| `src/placer/local_search/subcluster_relocation.py` | Parent-bounded child relocation and sibling slot swaps. |
| `src/placer/local_search/hierarchy_swaps.py` | Region-bounded hard-hard, hard-soft, soft-soft swap relief. |
| `src/placer/local_search/cluster_tile_rearrange.py` | Final directional tile attribution and bounded joint assignments within immutable leaf/child partitions. |
| `src/placer/local_search/cluster_decompress.py` | Exact-gated decompression of hot hierarchy blobs. |
| `src/placer/local_search/adjacent_cluster_transfer.py` | Transactional graph-adjacent swaps, relocations, ownership transfer, exact component gating, and post-transfer intra-leaf repair. |
| `src/placer/local_search/region_expand.py` | Expands hot cluster regions toward colder congestion bands. |
| `src/placer/local_search/lsmc_explore.py` | Coldspot kick candidate generation. |
| `src/placer/local_search/fields.py` | Congestion/coldspot fields used by relocation and coldspot tightening. |
| `src/placer/local_search/plateau_telemetry.py` | Buffered schema-v2 pass-yield telemetry with run/revision provenance. |
| `src/placer/scoring/exact.py` | Exact TILOS proxy wrapper. |
| `src/placer/scoring/incremental.py` | Incremental scorer for relocation, swaps, and complete mixed hard/soft group moves, including cached-JIT bbox smoothing and batched density tails. |
| `src/placer/legalize/spiral.py` | Hard-macro legalization, with cluster-consecutive order support. |
| `src/dreamplace_bridge/` | ICCAD04 pb/plc → Bookshelf, cluster grouping injection, DREAMPlace launcher, read-back. |
| `scripts/dreamplace/` | Pinned source/toolchain bootstrap, CUDA-12 CUB patch, and native-extension preflight. |
| `scripts/analyze_plateau_telemetry.py` | Provenance-filtered pass-yield aggregation and conservative skip-candidate report. |
| `src/eda_io/` | Standard EDA file I/O; converts inputs to the same benchmark object. |

## Hierarchy Pipeline

### 1. Cluster Derivation

`HierarchyModel.build()` derives hard-macro clusters from RTL instance-path
prefixes when macro names carry useful coverage (e.g. NG45), otherwise from
low-fanout net connectivity. Oversized bridge-connected flat clusters are
selectively split toward a target leaf size. If flat connectivity instead
collapses nearly all hard macros into one component, bridge-soft evidence is
mathematically unavailable because every soft has only one current owner. That
narrow topology is refined by cosine-connected hard-to-soft affinity vectors
from shared low-fanout nets. Tiny fragments merge into their strongest
positive-affinity group; if the affinity partition is inconclusive, a stricter
partial hard-graph cut is the fallback. The result remains inferred evidence,
not an explicit IP tag, and multi-component designs retain the bridge-evidence
rule. Soft macros are classified as
**owned** (one cluster dominates their connectivity, so they move with that
cluster) or **bridge** (comparable affinity to multiple clusters, so they get
a region spanning the clusters they connect). Direct soft-role evidence accepts
fanout up to 16 even though hard-cluster construction remains at fanout 8. The
wider role net must still contain an anchored hard cluster; geometry and
unanchored soft-only connectivity cannot assign an owner. When soft names themselves carry
useful slash-separated RTL paths, the deepest useful shared prefix creates a
high-confidence soft bundle. Explicit bundles take precedence over owner or
bridge-signature evidence during compound soft relocation; flat `Grp_*`
benchmarks do not form compounds from owner/bridge roles. A second diagnostic
layer builds a soft-only graph from repeated low-fanout shared nets and retains
only mutually strong components of at most 16 soft macros. These connectivity communities are
scored against common owned/bridge hard-cluster affinity. Combined evidence is
assigned deterministic `high` (≥0.90), `medium` (≥0.75), or `low` confidence.
Only explicit instance-path membership is high confidence and can form a
compound relocation group. Flat-netlist connectivity plus hard affinity is
limited to medium/low confidence: it continues to determine individual owned/
bridge regions and proposal evidence, but is not proof that the soft macros are
one IP.
Oversized split eligibility counts unique bridge softs per flat hard component;
evidence from another component cannot authorize a split.

Each placement call also creates one persistent `LocationAwareGraph` and stores
it on `HierarchyModel.location_graph`; the same object is exposed as
`plc._location_graph` and `benchmark._location_graph` for diagnostics. A macro
node has a stable placement index, PLC module index, name, hard/soft role, size,
active leaf, retained child/parent identifiers, bridge memberships, and weighted
low-fanout macro adjacency. A cluster node contains hard and owned-soft members
plus live centroid/bounding-box geometry. The graph and `HierarchyModel` share
one canonical active-leaf edge list; neighbor maps are projected on demand. Dense
NumPy placement arrays remain authoritative for scoring. The orchestration layer
synchronizes the graph at seed selection, before and after final survivor
search, after every retained void move, and after final audit rollback. Node
objects retain identity across synchronization and `revision` increases after
each committed snapshot. Consumers can query `cluster_of()`, `neighbors()`,
`macros_in_box()`, or `graph_hop_profile()` without reconstructing topology.
Each analyzed revision derives leaf area/utilization/free capacity, four-sided
whitespace, congestion/density heat, internal/external demand, geometric
frontiers, destination affinity, and cut delta. It also attributes routing to
internal, cross-leaf, bridge-soft, unowned-soft, and boundary-pair demand. The
same state grows connected frontier bundles, directs decompression waves,
constructs live capacity-directed regions, and orders final legalization from
boundary seeds through graph followers to interiors. Bounded checkpoints keep
positions, hierarchy roles, proxy components, and accepted-transfer history;
the dashboard consumes a JSON-safe graph payload for replay.

Hard ownership changes rebuild the shared active edge list from cached eligible
net endpoints with the same full-net weighting used during hierarchy inference.
Soft ownership changes do not affect the hard graph. Retained-child and parent
edge lists remain separate projections because they encode different hierarchy
levels. The decompressor has no private topology fallback; it requires the
persistent macro graph.

The model separately retains one additional hierarchy level without changing
the active clusters sent to DREAMPlace. Path-tag designs keep the nearest
useful ancestor prefix above the selected leaf depth. If oversized-connectivity
refinement already split a flat component, that original component becomes the
parent of its active leaves. Otherwise, an active cluster with at least 12 hard
macros may be bisected exactly once when both children contain at least four
hard macros. The bisection graph starts with weighted low-fanout hard-hard
edges and hard connections through the same soft macro. Initial
hard/soft proximity reinforces those edges; a Gaussian macro-area neighborhood
estimates local density, while placed low-fanout net span and endpoint demand
estimate routing pressure. These physical signals cannot create an edge where
there is no structural connection. The resulting split is retained only when
its raw structural cut is at most 0.20, within-child mean distance improves by
at least 10%, and combined confidence is at least 0.54. No child is recursively
split. Parent and child layers each receive their own labels, hard/soft roles,
cluster graph, confidence, regions, reference vector, and six-component limits.
Inferred child relocation requires mean confidence at
least 0.65; lower-confidence discovered structure remains audited and visible
but does not spend exact-search work. Explicit and retained connectivity parents
bypass this scheduling threshold.

After ordinary active-cluster relocation, the child pass ranks hot eligible
children and tests rigid translations toward cold connected components and
available parent boundaries. It co-moves the child's owned movable soft macros.
If rigid geometry overlaps, only the affected child set is compacted and spiral
legalized against fixed outside macros. Siblings may also exchange slots; the
same affected-only legalization is available when a rigid exchange is blocked.
Every complete hard/soft state must remain in the parent regions and pass the
active, child, and parent contracts before exact incremental scoring. The pass
shares a 24-state quota, a 4s deadline guard, and requires a local exact gain of
0.0001. Multilevel limits become authoritative for downstream passes and final
rollback only after a child state is retained; pure discovery therefore cannot
perturb an otherwise unchanged placement trajectory.

The separate deepest-child internal search has been removed. Its isolated
31-design ablation preserves every returned coordinate, score, and hierarchy
measurement. Parent/child inference, whole-child relocation, sibling swaps, and
all corresponding hierarchy contracts remain. The final cluster-tile pass
provides bounded member rearrangement after the other search and checkpoint
selection, without activating new downstream constraints.

### 2. Grouped DREAMPlace

`utils.constants.DREAMPLACE_GPU = True` runs fresh DREAMPlace solves on the first
visible CUDA GPU; `False` (the default) keeps CPU execution. The setting is independent of the main
process's CUDA detection and applies to ordinary and recurrent seeds. CUDA
cache keys add a backend suffix, while existing CPU cache keys remain valid.
Cache hits report their backend and skip the subprocess on either setting.
Grouping, BB updates, legalization, and all seed/final hierarchy gates are the
same. The focused backend comparison is recorded in
`ml_data/dreamplace_cuda/20260909/results.md`: CUDA improves IBM04 and preserves
IBM10/12 exactly, but the nine fresh seed calls take 88.76s versus 72.01s on
the RTX 4050. CUDA remains opt-in; no full-suite promotion is claimed.
The requested normal-guard IBM `--all` follow-up reaches AVG 1.1806, 17/17 VALID,
zero hard overlaps, and all final audits passing in 980.93s placer time, with
55 fresh CUDA seed calls. This has no paired CPU control; see
`ml_data/dreamplace_cuda/20260909/cuda_all/results.md`.

`run_dreamplace()` accepts `cluster_groups` and `group_weight`; the bridge
adds synthetic clique nets among each cluster's hard and soft members so
DREAMPlace's global placement pulls each subsystem together. DREAMPlace is
required — there is no proxy-only fallback. Runtime availability is a real
subprocess import probe using the Python ABI that compiled DREAMPlace, including
representative native density, HPWL, and boundary ops plus the DREAMPlace 4.1
BB-Nesterov optimizer used by this stage. The bridge sets `macro_place_flag=1`
and `use_bb=1`. At each global-placement update, DREAMPlace uses the short
Barzilai-Borwein step
`alpha = (s^T y) / (y^T y)`, where `s` is the change in reference position and
`y` is the corresponding gradient change. This is a scalar inverse-Hessian
approximation that scales the Nesterov step from observed curvature without
forming or storing a Hessian. A non-positive BB step falls back to the predicted
Lipschitz step. A clean checkout can reproduce the local CUDA 12.1 build with
`scripts/dreamplace/bootstrap.sh all`. The bootstrap pins Python 3.10.20 and
all installed Python dependencies, plus 113 toolchain archives with checksums
in `environment-linux-64.lock`. It initializes its own source repository even
inside the parent checkout and supports fresh or recorded local source revisions.
The CUDA-12 CUB patch, runtime-fixes patch, and progress patch are applied before
building. The runtime patch captures existing zero-filler handling, the optional
NCTUgr construction guard, and RUDY logging from the configured install; it does
not enable area adjustment or restore the retired seed experiment. Native
rebuild fixes explicitly instantiate the greedy-legalizer template. CUDA headers and link-time library
search paths point to the pinned environments, avoiding local symlinks and host
CUDA contamination. The HeteroSTA archive is checksum-verified before building.
`scripts/dreamplace/bootstrap.sh preflight` checks patches, source/install
agreement, exact Python dependencies, the ABI and CUDA build, native imports,
and BB-Nesterov support. Custom source/build directories propagate through every
patch and check. See [SETUP.md](../SETUP.md) for reproduction commands and the
build-environment exceptions to constants-only placement settings.
BB and cache reads are fixed production behavior rather than runtime-gated
options. A bounded Zhang-Hager non-monotone Armijo trial was evaluated on
ibm04 and ibm10, regressed DREAMPlace seed quality on both, and was removed;
the focused numbers remain in `PROGRESS.md`.

### 3. Seed Portfolio Selection

Grouped DREAMPlace is one candidate seed among several: legalized
`initial.plc`, two recurrent hierarchy prototypes, and an explicit-tag-only
route-channel candidate. The ordinary initial
candidate is explicitly hierarchy-aware: clusters and members legalize
consecutively in low-fanout connectivity-pressure × area order. The reference
`initial.plc` is legalized before
any immutable limit is built; the same legalized coordinates are then
exact-scored as the ordinary initial candidate. Each scored candidate records a
richer hierarchy vector covering mean and worst hard-cluster spread,
nearest-neighbor cluster impurity, weighted inter-cluster edge stretch,
owned-soft distance, and bridge-soft corridor distance. Each vector component
must remain within its independent absolute-or-relative slack from the
legalized reference. Non-mandatory alternatives whose immutable hard
components already fail are rejected before exact scoring. Production first
removes every candidate that fails the immutable component contract, then
selects the lowest exact proxy, breaking exact ties by stable candidate name.
Hierarchy composite and headroom remain diagnostic values; neither can promote
a higher-cost seed. The former hierarchy-band/headroom modes and their runtime
switch are removed. The selected seed becomes the reference for the same
six-component contract at pass checkpoints and final rollback.
If no candidate satisfies the component contract, selection fails closed unless
the reference candidate itself passes; an invalid fallback is never promoted
to become the hierarchy baseline.
A mandatory candidate that has lower exact proxy than the selected seed and
misses exactly one component is a repair candidate. Production deterministically
interpolates it toward the authoritative passing reference, legalizes each
trial, and bisects the passing boundary. Only a repair that retains at least
`HIER_SEED_CONTRACT_REPAIR_MIN_FRACTION=0.95` of the source displacement is
exact-scored and returned to the ordinary selector. This guard rejected broad
IBM03/13 projections that changed the search basin; IBM09's accepted fraction
was `0.99609375`.
Single-component soft-affinity refinement adds one reference rule to avoid
compounding contract slack: when raw `initial.plc` is already in bounds and
overlap-free and its legalized form satisfies the raw limits, the raw vector
stays authoritative through final rollback. If the raw hard placement is
illegal, it cannot define trustworthy geometry; grouped DREAMPlace becomes the
reference for that topology. Seed telemetry records which reference was used.
Final reports classify evidence coverage as `high` (hard >= 0.75 and soft >=
0.25), `partial` (hard >= 0.25 and soft >= 0.10), or `low`; this is diagnostic
provenance and does not change proxy or hierarchy acceptance gates. Path-tag
clusters are reported as `explicit`, while flat-net connectivity is `inferred`.
The production 15% relative allowance and component absolute allowances were
retained after replay over 31 IBM, NG45, and synthetic final rows. A 10%
relative profile invalidated `ibm18`'s final state and `ibm07`'s selected seed;
a uniform 20% absolute reduction invalidated `ibm08`/`ibm11` finals and the
selected NG45 `nvdla` seed. The active profile therefore has measured real-
design support rather than being inferred from the aggregate composite score.

Soft hierarchy is confidence-tiered after hard labels are final. Tier 1 is the
direct fanout-16 hard affinity. Tier 2 performs exactly two synchronous
soft-to-soft propagation rounds with repeated support, 0.60/0.67 dominance,
and non-propagating bridge results. Tier 3 examines only still-unassigned soft
macros and retains a 2-16 member community only when repeated mutually strong
edges produce identical membership at 0.60 and 0.75 thresholds and a repeated-
edge cut ratio at most 0.35. Tier 2 roles enter the existing owned/bridge
regions and contracts. Tier 3 groups enter grouped DREAMPlace and compound
relocation but do not enter `cluster_softs`, so no soft-only evidence can
manufacture hard ownership or alter the retained hard hierarchy levels.

Neighbor impurity needs only the nearest four clustered hard macros. A cached
Numba kernel therefore keeps a four-entry insertion-ordered selection per
macro instead of materializing and stably sorting an N-by-N distance matrix.
It compares squared distances (the same ordering as Euclidean distance) and
uses the original clustered-row order for equal-distance ties, preserving the
previous stable-sort result.

Synthetic-clearance pair pushes are accumulated by a cached Numba kernel; the
seed update, clipping, legalization, scoring, and selection semantics are
unchanged.

### 4. Cluster-Consecutive Legalization

Hard macros legalize in an order that keeps cluster members adjacent
(largest clusters first, then connectivity-pressure × area within each
cluster, then unclustered macros), followed by a default-order safety pass
to guarantee legality. Each macro's expanding-ring search runs in a cached
Numba kernel with the original lexicographic candidate order, strict overlap
tests, and minimum-displacement tie behavior. Python retains the between-macro
deadline check, and the former vectorized conflict-matrix path remains the
diagnostic reference.

The former constraint-graph candidate used coordinate order and overlap
geometry without hierarchy labels, edges, ownership, or confidence. It was
deleted with its standalone legalizer on 2026-08-11 rather than treating
post-hoc contract eligibility as hierarchy optimization.

### 5. Soft Cleanup

`_soft_relocation_moves()` relocates soft macros by congestion and density.
Soft macros may overlap, so this pass has no legality constraint.

### 6. Region-Locked Relief

`HierarchyModel.hard_regions()` / `soft_regions()` build region boxes per
cluster, expanded toward nearby cold congestion components where available.
Hard and soft relocation inside these regions rank candidates by a
congestion-weighted proposal field plus density; moves that leave the
assigned region are accepted only when the exact-proxy gain clears
`HIER_REGION_ESCAPE_MIN`.

Before a region hard-relocation candidate reaches exact batch scoring, the
pipeline temporarily applies that one move and evaluates the inexpensive
legacy hard-cluster containment metric. Candidates above the selected seed's
hard audit limit are discarded and counted as `hierarchy_rejects`. The rich
six-component contract is still checked after the completed pass, so this
prefilter reduces known-ineligible work without weakening the authoritative
checkpoint. The accepted full sweep rejected 654 candidates this way,
exact-scored 18,637 remaining relocation candidates, and required no hard-
relocation rollback.

For refined single-component graphs with at most 64 hard macros, micro-shift
and hard/soft relocation candidates also pass the complete vector contract
before exact scoring. This retains useful work on the small SRAM-shaped case.
Larger refined graphs use the normal pass checkpoint and final rollback rather
than recomputing the rich vector for every proposal.

Relocation has a default-off structural ordering term
(`HIER_OBJECTIVE_STRUCTURAL_WEIGHT=0.0`) that combines edge clearance, grid
alignment, and local gap penalties. It only reorders proposals; legality,
region, hierarchy-quality, and exact-proxy gates are unaffected.

### 7. Cluster Decompression

`_cluster_decompression_relief()` expands hot clusters inside their regions
to open local routing channels, biasing toward nearby cold congestion
components when present. Candidates re-legalize hard macros, move owned softs
with their cluster, and are accepted only if exact proxy improves while a
composite hierarchy-quality metric (mean radius + bbox spread + crowding
penalty) stays within `HIER_QUALITY_BUDGET`.

### 8. Region-Bounded Swaps

`_region_bounded_swap_relief()` runs hard-hard, hard-soft, and soft-soft
swaps against the same congestion/density proposal field. In-region swaps use
the exact-proxy gate directly; out-of-region swaps must also clear the escape
threshold. Swap scoring uses the `IncrementalScorer` global topology arrays;
candidate ranking can use CUDA batch sorting when available, but exact
scoring remains the acceptance authority either way.

Each swap source preserves its ranked candidate list. Hard-hard batches first
score four candidates, hard-soft batches first score eight, and soft-soft
score twelve. If the first prefix has no acceptable candidate, the scheduler
scores a second prefix of the same size before the untouched remainder. If
either prefix contains the first candidate that clears exact proxy, region, and
hierarchy gates, the untouched suffix is skipped. Otherwise the remainder is
scored as one batch and the original first-winner order is unchanged. Logical score
counts continue to drive the accepted deterministic quota, while telemetry
separately records physical exact work, the skipped suffix count, and accepted
candidate-rank buckets. Hard-hard and hard-soft sources rank and truncate their
candidate lists before running the independent legality kernel, so candidates
that cannot reach exact scoring do not allocate overlap matrices. When graph
masks and deltas are disabled, the same loop also skips their zero-valued array
construction.

When a source schedule reaches beyond its first prefix, the scorer prepares the
source-invariant pair topology and coordinate snapshots once for the complete
stable candidate list. Each prefix receives a view of its candidate rows and a
rebased view of the ragged touched-net offsets. The untouched suffix is still
unprepared for exact field scoring until needed, and any mock/scalar scorer
without the prepared API follows the original batch calls.

Hard-macro widths and heights do not change during the schedule. The pipeline
therefore constructs their two pairwise center-separation matrices once before
the congestion/density rounds and passes the immutable tuple through ordinary
and graph-fallback swap calls. Standalone callers retain a local-computation
fallback.

The incremental scorer keeps raw and smoothed routing grids synchronized after
each trial move. A prepared routing structure caches pin-to-module references,
offsets, topology groups, weights, and scratch buffers. One cached Numba call
gathers current pin cells, applies two-, three-, and high-fanout routes, and
returns the touched bbox without Python buckets or sorting. Horizontal-column
and vertical-row bbox smoothing then uses cached Numba kernels with reusable
prefix buffers. The NumPy formulas remain the diagnostic oracle; both JIT
stages preserve their reference accumulation order and incremental deltas.

Soft-relocation grid IDs first pass through a compiled stable filter that
converts and clips target centers, constructs exact symbolic coordinate keys,
applies the optional region mask, and stamps first-winner duplicates. The
hierarchy callback remains scalar and ordered because it evaluates the rich
contract. Prepared target sets then use a compiled exact wirelength
delta batch. The existing threshold is applied in stable proposal order before
field scoring; this changes neither candidate order nor acceptance semantics.
Surviving target sets of size two or more use a true batched CPU path. Cached
JIT loops build per-target routing grids, touched bboxes,
wirelength, and density occupancy without mutating committed scorer state;
the congestion and density tail reductions then operate over the batch. The
congestion grids are disposable after reduction, so their float64 rows use
in-place `ndarray.partition` and avoid a second full batch copy while retaining
the evaluator's exact top-five-percent sum. The density reducer is itself a
cached Numba kernel that retains the scalar nonzero filtering, tiny-grid mean,
and top-tail partition semantics. Scalar trials remain the one-target path and
the parity oracle.

Soft-soft swap sets sharing one endpoint no longer build or flatten one routing
structure per endpoint pair. The JIT loop consumes sorted touched-net ids and
the scorer's global net starts, lengths, weights, pin references, and offsets.
It packs selected pin cells once per old/new state, applies two-pin, three-pin,
then high-fanout routes in evaluator order, and performs the exact
remove/swap/add sequence. The committed routing, smoothing, density, and
position caches remain read-only until the existing scalar commit method
accepts a winner.

Swap tail reduction is baseline-plus-delta rather than full-result-grid. For
congestion, a candidate recomputes only H columns and V rows in the route bbox
plus cells whose hard blockage changed; the reducer combines those values with
the highest unchanged baseline values before the exact top-five-percent sum.
For density, four rectangle deltas define every affected cell, and the same
merge recovers the scalar nonzero/top-ten-percent semantics. Raw routing trial
grids remain batched, but the former full congestion-value and density-result
matrices are gone. The sorted baseline values and density summary are cached
until a commit mutates routing, blockage, or occupancy state; every hard,
soft, swap, and multi-move commit invalidates that cache.

Two further memory/parallel experiments remain deliberately absent. Fusing
hard-blockage construction into one reducer scratch regressed IBM04 and IBM12,
and unbounded candidate-row `prange` launched 22 workers for thousands of
small batches, exhausting IBM18's 20s phase guard after 32,498 candidates
instead of completing 69,152 in 8.51s. Productive parallelism would require a
coarser schedule boundary, not parallel reduction inside each small batch.

Hard-hard and hard-soft sets sharing one hard endpoint also use exact batched
scoring. Candidate-specific hard-blockage grids reproduce the reference
remove/swap/add order, including top-row and right-column correction terms,
while prepared route, wirelength, and density kernels evaluate the full set.
Scalar scoring remains the one-candidate/fallback oracle. Cross-design parity
was exact to floating-point roundoff (maximum `2.22e-16`) and every committed
scorer cache remained unchanged during batch trials.

### 9. Post-Swap Polish

Hard propose-all relocation and soft relocation each run once more over the
swap-relieved state with a stronger exact-gain margin, followed by a
micro-shift replay. A budget-aware strong-soft-repair pass can spend any
remaining time on soft macros (without reopening hard legality) when recent
pass telemetry shows plateaued cleanup or a useful soft signal.

### 10. Coldspot Tightening

`_coldspot_cluster_kick_candidates()` selects hot clusters, gathers it and its owned/
bridge soft macros into a low-congestion window, legalizes, and refines
locally (swaps + relocation) before accepting only if exact proxy improves
and the hierarchy-quality metric stays within budget. When no kick commits,
a graph-local fallback runs the same bordered refinement on the hottest
eligible cluster without a kick.

### 11. Final Audit and Bounded Relief

Post-coldspot replay, eligible small-design polish, topology-aware leaf search,
and hard-clear void relocation feed the final exact score directly. The finalizer
keeps the lowest exact-scored legal checkpoint that passes the hard hierarchy
budget, all six component limits, per-leaf island limits, and any active
parent/child contract. Terminal disjoint-envelope packing and both repair
replays are removed: envelope intersections alone do not violate macro legality.
There are no terminal exemptions for edge stretch, impurity, or soft distance.

After checkpoint selection, one bounded density lane can move only free soft
macros. It excludes every active, child, and parent owner/bridge role, every
retained or evidenced soft bundle, fixed soft macros, and any soft macro on a
multi-pin net with a hard endpoint. The existing regional relocation operator
tests at most 64 candidates from 16 hot sources and four targets per source.
The search has a two-second guard; scorer setup and final verification are
additional bounded work. A fresh scoring transaction evaluates the actual
float32 baseline and candidate through the same `clamp_in_bounds()` helper
used by the API return, and the complete contract must pass. A gain
above 0.000001 is required; otherwise the original coordinates survive.
Protected coordinates retain their original internal precision. The subsequent
cluster-tile pass requires a fresh exact improvement over this result.

The final cluster-tile pass builds a temporary attribution of the evaluator's
directional congestion tail, hard blockage, and density tail. Incident nets
include external endpoints; routing pressure can therefore belong to a macro
whose footprint is far from the hot tile. The existing canonical location graph
connects small patches inside one leaf/child partition. It never changes
ownership, hierarchy edges, or confidence. Fixed and bridge macros remain
frozen, and eligible complete explicit soft bundles move rigidly.

Each of at most four patches contains two to four units, capped at 32 macros.
Bounded enumeration ranks joint assignments to the current slots and up to two
cold tile centers. The rank is a proposal heuristic; hard legality, center
regions, the complete hierarchy contract, and exact mixed-group scoring decide
acceptance. There are at most 16 exact trials per patch and 64 overall. Search
uses at most three seconds of unused internal-floorplan/void allowance; fresh
baseline/winner scoring and final reporting are additional bounded work.
The winner passes the API bounds clamp in float32 and fresh full scoring before
commit, with a required gain above 0.000001. A failed check restores the input.

### 12. Plateau Telemetry

The hierarchy flow always buffers schema-v2 pass-level telemetry: proposed and
retained proxy/accepts, audit rollback and violations, discarded gain, scorer
rebuild time, elapsed time, accept rate, plateau flag, run id, code revision,
process id, dirty-worktree flag, and a deterministic scoped worktree
fingerprint. Seed creation/prescore, cache lookup, coldspot work, full exact
scores, and final audits are separate stage events. The outer orchestration also
records mutually exclusive setup, seed-portfolio, hierarchy-search, coldspot,
and post-coldspot phases plus `hierarchy_floorplan_total` and
`placer_api_total` boundaries. Exact-scored seed
candidates and the final placement also emit structured
`hierarchy_contract_audit` events containing vectors, limits, margins,
violations, coverage, and provenance. The default output is
`ml_data/plateau_telemetry/plateau_telemetry.jsonl`;
The `utils.constants.HIER_PLATEAU_TRACE_PATH` constant can redirect it. Candidate-level trace logging and
learned ranking were removed because they added overhead and repeatedly failed
to improve placement. `scripts/analyze_plateau_telemetry.py` filters the
remaining scheduling telemetry by run, revision, worktree fingerprint, or
benchmark, reports aggregate retained yield and conservative skip candidates,
prints stage timing with `--stages`, and summarizes exact-score limits,
per-benchmark distributions, and exhaustion with `--quotas`. Every pass row
records its exact quota, usage before and after the invocation, and whether the
limit was exhausted. Passes with no exact-scored candidates
report gain-per-score as `n/a`, never as an artificial infinite yield.
`--coverage` reconciles the five exclusive phases to the floorplan and API
boundaries. On the accepted IBM sweep they cover at least 99.86% of every API
call: 297.33s inside `MacroPlacer.place()` versus 318.55s for the complete
evaluator command, exposing 21.22s of external load/final-score overhead.
`scripts/analyze_hierarchy_contract.py` separately aggregates per-component
headroom, relevant-row counts, allowance utilization, coverage/provenance
cohorts, and failures. It can replay alternative relative and per-component
absolute slacks without changing production. NG45 audit rows use their design
names (`ariane133`, `ariane136`, `mempool_tile`, and `nvdla`) even though their
source directories share the leaf name `output_CT_Grouping`.

The synthetic runner also emits `hierarchy_truth_audit` rows from cluster labels
preserved by `generate_benchmarks.py`. These are independent accuracy checks,
not production acceptance gates. The current ten-design sweep passes all ten.
The previously failing `syn03_sram` case now recovers its four truth groups
exactly (purity, pair precision, and recall all `1.0`) and improves proxy
`4.3964 -> 4.3257`; this is structural recovery rather than looser slack.

### 13. Deterministic Exact-Score Quotas

Wall-clock deadlines remain safety guards, but the high-volume region,
interleaved, plateau, compound, and strong/medium repair operators stop at
deterministic work ceilings first. Repeated region rounds consume one shared
pass allowance. Regional hard-hard, hard-soft, and soft-soft swaps likewise
share one allowance, so no swap type receives unbounded work. The next stable
candidate batch is sliced to the remaining allowance without changing
candidate or commit order.

| Pass | Exact-score ceiling |
|---|---:|
| region hard relocation | 2,600 |
| region soft relocation | 24,000 |
| parent-bounded child relocation / sibling swaps | 24 |
| interleaved soft repair | 4,096 |
| regional swaps | 72,000 |
| regional-swap graph fallback | 100 |
| plateau escape, first / post | 5,000 / 7,000 |
| compound soft relocation | 60 |
| strong / medium soft repair | 40,000 / 2,048 |

These ceilings were derived from attributable accepted-run maxima with modest
headroom. The IBM and NG45 validation sweeps preserved every preceding score.
Only IBM interleaved soft repair reached a ceiling, on ibm11 and ibm17, where
the reference already scored exactly 4,096 candidates. An intentionally
aggressive binding profile regressed ibm11 and was not promoted.

## Scoring and Legality

Candidate island checks request only spread, bounding span, and nearest-neighbor
impurity. Reporting-only fragmentation, owned-soft p90, and foreign intrusion
remain in seed/final diagnostics. The limits and acceptance rules are unchanged.
The void survivor shares initial interior/edge geometry between its two lanes
and obtains each soft routing unit's nets from the scorer's cached incident-net
union, preserving net/pin accumulation order and lane-specific filtering.

Mixed hard/soft group scoring reuses the sparse swap congestion reducer and
retained grid snapshots. A compiled density scratch transaction removes all old
rectangles, then adds all new rectangles in the original order; changed values
merge with the cached unchanged density tail. It leaves committed occupancy and
smoothed routing untouched, and retains scalar scoring for tiny grids or the
diagnostic NumPy fallback. Commits and complete final audits remain unchanged.

- Fixed macros stay fixed.
- Hard macros must not overlap.
- All macro centers must be in bounds (`_clamp_in_bounds()` runs on every
  returned placement).
- Soft macros may overlap.

Exact proxy scoring drives evaluator reports, the initial hierarchy score
measurement, and every accept gate listed above (relocation, decompression,
swaps, coldspot tightening). `PlacementState` carries hard positions, soft
positions, and the current exact proxy through the pipeline; pass results
are returned as structured `PassResult` trace payloads.

## Verification

```bash
uv run python -m compileall -q src
uv run python test/verification/_verify_region_escape_gate.py
uv run python test/verification/_verify_score_region_swaps.py
uv run python test/verification/_verify_coldspot_kick.py ibm10
uv run --with pytest python -m pytest test/verification/ test/eda_io/ test/visualizer/ -q
uv run evaluate src/main.py -b ibm10
uv run evaluate src/main.py --all
```

## Constants Reference (`src/utils/constants.py`)

Grouped by the pipeline stage that consumes them.

**Hierarchy / clustering**
```text
HIER_TAG_PREFIX_MAX_DEPTH=5            HIER_TAG_PREFIX_MIN_GROUP=2
HIER_TAG_PREFIX_MIN_COVERAGE=0.25      CLUSTER_MIN_EDGE=2
CLUSTER_MAX_FANOUT=8                   HIER_OVERSIZE_CLUSTER_START_FRAC=0.40
HIER_OVERSIZE_CLUSTER_TARGET_FRAC=0.15 HIER_OVERSIZE_CLUSTER_TARGET_TOL=1.10
HIER_OVERSIZE_CLUSTER_MIN_BRIDGE_SOFTS=5
HIER_OVERSIZE_CLUSTER_MIN_SIZE=6       HIER_OVERSIZE_CLUSTER_MAX_CUT_RATIO=0.45
HIER_SUBCLUSTER_MIN_PARENT_HARD=12     HIER_SUBCLUSTER_MIN_CHILD_HARD=4
HIER_SUBCLUSTER_MAX_CUT_RATIO=0.20     HIER_SUBCLUSTER_SHARED_SOFT_WEIGHT=0.75
HIER_SUBCLUSTER_SPATIAL_PROXIMITY_WEIGHT=1.0
HIER_SUBCLUSTER_SPATIAL_PRESSURE_WEIGHT=0.50
HIER_SUBCLUSTER_SPATIAL_NEIGHBORS=8    HIER_SUBCLUSTER_SPATIAL_MAX_SOFT_DEGREE=24
HIER_SUBCLUSTER_SPATIAL_MIN_COMPACTNESS_GAIN=0.10
HIER_SUBCLUSTER_SPATIAL_MIN_CONFIDENCE=0.54
HIER_SUBCLUSTER_RELOCATION_MAX_HARD=64
HIER_GROUP_WEIGHT=8
```

**Seed portfolio**
```text
HIER_RE2MAP_RECURSIVE_SEEDS=2      HIER_RE2MAP_MAX_LEAF_HARD=16
HIER_SEED_ROUTE_CHANNEL_MIN_CLUSTER=4
HIER_VECTOR_CONTRACT_REL_SLACK=0.15
HIER_VECTOR_CONTRACT_ABS_SLACK={compactness:0.005,worst_spread:0.015,
  neighbor_impurity:0.05,edge_stretch:0.015,owned_soft:0.015,bridge_soft:0.015}
```

**Regions and relocation**
```text
HIER_REGION_DENSITY=0.65        REGION_BIAS=1.0
HIER_REGION_ROUNDS=2            HIER_REGION_BUDGET_S=40
HIER_REGION_ESCAPE_MIN=0.002
HIER_REGION_COMPONENT_COLD_PCT=45     HIER_REGION_COMPONENT_MIN_CELLS=4
HIER_SUBCLUSTER_RELOCATION_BUDGET_S=4 HIER_SUBCLUSTER_RELOCATION_MIN_SPARE_S=12
HIER_SUBCLUSTER_RELOCATION_TOP_CHILDREN=4 HIER_SUBCLUSTER_RELOCATION_TOP_SWAPS=4
HIER_SUBCLUSTER_RELOCATION_MIN_GAIN=0.0001
HIER_PROPOSAL_CONGESTION_WEIGHT=2.5   HIER_PROPOSAL_DENSITY_WEIGHT=1.0
HIER_PROPOSAL_OUTSIDE_RELIEF_MARGIN=0.08
HIER_RELOC_PROPOSE_MIN_GAIN=0.0005
HIER_COMPOUND_SOFT_BUDGET_S=4         HIER_COMPOUND_SOFT_MIN_SPARE_S=5
HIER_COMPOUND_SOFT_TOP_GROUPS=4       HIER_COMPOUND_SOFT_GROUP_SIZE=6
HIER_COMPOUND_SOFT_COLD_PCT=35        HIER_COMPOUND_SOFT_ANCHORS=2
HIER_COMPOUND_SOFT_SHIFT_FRACTIONS=0.25,0.5,1.0
HIER_COMPOUND_SOFT_MIN_FIELD_DROP=0.02 HIER_COMPOUND_SOFT_MIN_GAIN=0.00005
HIER_PLATEAU_ESCAPE_BUDGET_S=4        HIER_PLATEAU_ESCAPE_SOFT_TOP_K=384
HIER_PLATEAU_ESCAPE_SOFT_TARGETS=10
```

**Deterministic structural candidate ordering (opt-in)**
```text
HIER_OBJECTIVE_STRUCTURAL_WEIGHT=0.0   HIER_KEEP_OUT_WEIGHT=0.2
HIER_GRID_ALIGN_WEIGHT=0.2             HIER_NOTCH_WEIGHT=0.6
```

**Decompression**
```text
HIER_DECOMPRESS_ROUNDS=2          HIER_DECOMPRESS_BUDGET_S=18
HIER_QUALITY_BUDGET=0.03          HIER_QUALITY_RADIUS_WEIGHT=0.75
HIER_QUALITY_BBOX_WEIGHT=0.20     HIER_QUALITY_CROWD_WEIGHT=0.05
HIER_DECOMPRESS_LOCAL_SHIFT_FRAC=0.20
```

Rounds with no cheap hot-cluster to cold-window opportunity are skipped before
candidate generation and exact candidate scoring. The predictor blends
hot-to-cold field gap, open cold-cell capacity around the candidate window, and
source-to-window displacement. Default production tries the top two opportunity
clusters with five whole-cluster variants per cluster, then commits from
exact-proxy-ranked refined candidates rather than from a graph-ranked prefix.
Coldspot also stops after repeated generated pools fail to commit.
Weak-opportunity and dry-limit exits also skip the graph-local fallback.

This is not the old generic LSMC path. It is a narrow hierarchy-tightening
helper. Candidate-local refinement runs hard-hard and hard-soft swaps with the
kicked hard cluster locked in the local box, plus soft-soft swaps and soft
relocation that may leave the local box only after a `0.0025` exact-proxy gain.
The local box includes owned/bridge soft macros, but its base pad is derived from
the kicked hard-core max dimension rather than the soft-inclusive bbox.
The phase tracks a current cold-cell grid from the active congestion field,
refreshes it after every finalized coldspot kick, masks out cells occupied by the
candidate, and expands the pre-margin local border through adjacent open cold
cells before applying the hard-core pad. This lets finalized cluster locations
use nearby coldspots for local relief while preserving swap and soft-locked
relocation room.
The graph supplies coldspot-local relocation target pools and gates relocation
targets by graph mask; default candidate commitment uses exact-proxy-ranked
refined outcomes.
Coldspot kick candidate generation augments each cluster's owned soft set with
movable bridge soft macros tied to the same hierarchy cluster. The default pool
now tries multiple opportunity-ranked clusters, with shape-preserving variants
for each instead of only repeating one random gather: multiple cold anchors,
compact original orientation, rotated orientation, source-facing border
compaction, and a lower-displacement centroid-blended candidate. The hard
cluster and those soft macros are placed into the cold window together, then the
existing legalization, local refinement, exact-proxy gate, and hierarchy-quality
gate accept or reject the resulting full candidate as one state.
When no coldspot kick commits, the
graph-local fallback runs the same bordered swaps and relocations on the
current placement for the hottest eligible clusters.
Production then reruns `_micro_shift_polish()` once more after coldspot
tightening; deterministic hot-cluster coldspot selection was tested and removed
after regressing the full sweep.

**Swap ranking**

The current CPU/Numba swap breadth is:

```text
HIER_HARD_SWAP_K=16          HIER_SOFT_SWAP_K=48
HIER_SWAP_MIN_GAIN=0.00001
```

Repeated batched swap evaluations reuse a bounded cache of static pair-topology
packing inside the incremental scorer; exact score values and acceptance gates
are unchanged.

Fallback congestion expansion now skips a hot cluster when no adjacent fallback
side is colder than the cluster; component-guided expansion remains unchanged.

**Post-swap / plateau scheduling**
```text
HIER_POST_SWAP_MICRO_SHIFT_BUDGET_S=8   HIER_STRONG_SOFT_REPAIR_BUDGET_S=12
HIER_STRONG_SOFT_REPAIR_MIN_SPARE_S=2   HIER_STRONG_SOFT_REPAIR_ROUNDS=2
HIER_SOFT_QUOTA_REFERENCE_MACROS=1200   HIER_SOFT_QUOTA_MIN_SCALE=0.50
HIER_PLATEAU_ACCEPT_RATE=0.002          HIER_PLATEAU_PROXY_GAIN=0.00005
HIER_PLATEAU_ESCAPE_BUDGET_S=4
```

**Coldspot tightening**
```text
HIER_COLDSPOT_ROUNDS=8              HIER_COLDSPOT_BUDGET_S=30
HIER_COLDSPOT_MIN_GAIN=0.0001       HIER_COLDSPOT_QUALITY_BUDGET=0.01
HIER_COLDSPOT_MIN_FIELD_GAP=0.02    HIER_COLDSPOT_MAX_DRY_ROUNDS=2
HIER_COLDSPOT_OPPORTUNITY_TOP_CLUSTERS=1
HIER_COLDSPOT_WHOLE_VARIANTS=5      HIER_COLDSPOT_ANCHOR_VARIANTS=3
```

**Runtime and diagnostic constants (`src/utils/constants.py`)**
```python
DREAMPLACE_GPU = False
HIER_DIAGNOSTIC_NO_DEADLINES = False
HIER_PLATEAU_TRACE_PATH = "ml_data/plateau_telemetry/plateau_telemetry.jsonl"
HIER_PLATEAU_TRACE_RUN = ""
```

`HIER_PASS_BUDGETS` contains each pass's exact-score and candidate ceilings;
zero means unlimited for that counter. Revision and worktree fingerprints are
computed from source; archived-source diagnostics supply their fingerprint
directly. No environment override is read.

The rejected CUDA overlap/bounds prefilter and graph isolation selector are
removed. The [GPU graph report](../ml_data/gpu_graphs/20260911/results.md)
preserves the nine failed workload/design gates; GPU construction/affinity
kernels, repeated-query probes, and duplicate comparison controls are retired.
`test/diagnostic/profile_hierarchy_splits.py` retains the promising compiled
CPU neighbor-sum experiment on captured production split inputs. It requires
identical partitions and cut ratios and remains outside the production path.

The physical-net RUDY DREAMPlace wrapper is also retired after adding seed
runtime without improving the three tested final placements. Its measurements
remain in the [GPU placement investigation](GPU_PLACEMENT_EXPERIMENTS.md).
The shared rectangular occupancy helper now lives in the retained
`replay_gpu_soft_refinement.py`. Fresh baseline capture and backend comparisons
share `run_dreamplace_cuda_comparison.py --capture-final`; the duplicate
experimental runner is removed. Soft refinement remains offline.
Its prerequisite correction removes a stale extra `plc` positional argument
from seed soft cleanup. DREAMPlace candidates now reach the intended scoring
and hierarchy selection instead of being discarded by `TypeError`.

The [September 12 refinement comparison](GPU_SOFT_REFINEMENT_20260912.md)
adds eight-step replay with checkpoints 1/2/4/8 and an isolated guidance variant
using the existing exact tile-search tail weights. Weights include directional
capacity and boundary smoothing conversions and stay resident for the short
block. All variants preserve the same eligible macros and exact/contract gates;
none enters the production pipeline.
Independent scalar/tag/truth checks run in spawned CPU processes after all
timed proposals, and verified results are published only after those checks.
The diagnostic acceptance path repairs return-precision canvas overhang using
the existing float32 clamp on eligible soft coordinates only, then rechecks
validity and the complete contract. Already-valid proposals are unchanged.

Experiments that were tried and not promoted (full recursive bisection,
cluster-room/bridge-corridor modeling, broad weak-hot region reshape, early
strong-soft repair, early swap-lite, deterministic hot-cluster coldspot
selection, learned candidate reordering at full-suite scale) are recorded in
`PROGRESS.md`, not here — this document describes only the
active system.
