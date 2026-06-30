"""Accepted production constants for hierarchy placement."""

# Per-benchmark placement time budget used by the MacroPlacer constructor default.
TIME_BUDGET_S = 150.0

# Maximum net fanout considered as hierarchy signal when forming hard clusters.
CLUSTER_MAX_FANOUT = 8
# Minimum shared low-fanout net count required to merge two hard macros.
CLUSTER_MIN_EDGE = 2
# Prefer explicit slash-separated instance-path hierarchy when benchmark macro
# names provide it. This auto no-ops on flat-name IBM benchmarks.
HIER_TAG_PREFIX_MAX_DEPTH = 5
HIER_TAG_PREFIX_MIN_GROUP = 2
HIER_TAG_PREFIX_MIN_COVERAGE = 0.25
# Flat clusters above this hard-macro fraction are eligible for selective splitting.
HIER_OVERSIZE_CLUSTER_START_FRAC = 0.40
# Eligible oversized clusters are recursively split until leaves are below this fraction.
HIER_OVERSIZE_CLUSTER_TARGET_FRAC = 0.15
# Small tolerance for accepting integer-sized leaves near the target fraction.
HIER_OVERSIZE_CLUSTER_TARGET_TOL = 1.10
# Minimum flat bridge-soft count required before selective oversized splitting is allowed.
HIER_OVERSIZE_CLUSTER_MIN_BRIDGE_SOFTS = 5
# Minimum child size accepted from oversized hierarchy bisection.
HIER_OVERSIZE_CLUSTER_MIN_SIZE = 6
# Maximum cut/total edge-weight ratio accepted for oversized hierarchy bisection.
HIER_OVERSIZE_CLUSTER_MAX_CUT_RATIO = 0.45

# DREAMPlace group attraction weight for cluster grouping constraints.
HIER_GROUP_WEIGHT = 8
# A small hierarchy-compatible seed portfolio around the grouped
# DREAMPlace candidate. Seeds are legalized and prescored before region relief.
# Blend ratios from DREAMPlace toward initial.plc for prescored seed basins.
HIER_SEED_BLEND_ALPHAS = (0.35, 0.65)
# Radial expansion applied to the grouped DREAMPlace basin.
HIER_SEED_EXPANSION_FRAC = 0.06
# Extra temporary half-extent fraction used by synthetic-clearance push-apart.
HIER_SEED_CLEARANCE_FRAC = 0.08
# Number of Jacobi-style synthetic-clearance push-apart iterations.
HIER_SEED_CLEARANCE_ITERS = 3
# Percentile of hard-macro area eligible for synthetic clearance.
HIER_SEED_CLEARANCE_AREA_PCT = 97.0
# Minimum hard macros required before a hierarchy cluster receives route lanes.
HIER_SEED_ROUTE_CHANNEL_MIN_CLUSTER = 4
# Center-lane half-width as a fraction of the cluster local span.
HIER_SEED_ROUTE_CHANNEL_LANE_FRAC = 0.10
# Fraction of remaining center-lane overlap used as outward push distance.
HIER_SEED_ROUTE_CHANNEL_PUSH_FRAC = 0.35
# Maximum route-channel push as a fraction of the cluster local span.
HIER_SEED_ROUTE_CHANNEL_MAX_SHIFT_FRAC = 0.04
# Congestion-heavy proposal ranking. Exact proxy remains the accept gate.
HIER_PROPOSAL_CONGESTION_WEIGHT = 2.5
HIER_PROPOSAL_DENSITY_WEIGHT = 1.0
# Keeps congestion-weighted proposal ranking hierarchy-aware. Out-of-region
# targets can still compete, but only when their field relief beats the best
# in-region relief by this fraction of the active proposal-field span.
HIER_PROPOSAL_OUTSIDE_RELIEF_MARGIN = 0.08

# Enables post-swap hard propose-all relocation automatically on CUDA backends.
HIER_POST_RELOC_PROPOSE_ALL = "auto"
# Caps hard propose-all candidates per hot macro after region swaps.
HIER_POST_RELOC_PROPOSE_TOP_M = 16
# Number of hottest hard macros considered for post-swap propose-all relocation.
HIER_RELOC_PROPOSE_HOT_K = 32
# Number of hottest soft macros considered by post-swap soft relocation.
HIER_POST_SOFT_RELOC_TOP_K = 256
# Minimum exact-proxy gain required to accept a post-swap soft relocation.
HIER_POST_SOFT_RELOC_MIN_GAIN = 0.0005
# Minimum exact-proxy gain required to accept hard propose-all relocation.
HIER_RELOC_PROPOSE_MIN_GAIN = 0.0005

# Pass-level candidate/accept/gain/time telemetry is emitted for plateau analysis.
# Rows are buffered and flushed once per benchmark / process exit.
# Budget-aware and component-aware scheduling use telemetry and exact proxy snapshots.
# If normalized congestion dominates density by this margin, preserve budget
# for soft/coldspot cleanup and allow strong-soft repair to run even when
# plateau telemetry is ambiguous.
HIER_COMPONENT_CONG_DOMINANCE = 0.10
# Minimum seconds preserved for strong soft repair and coldspot cleanup when
# congestion still dominates.
HIER_COMPONENT_RESERVED_CLEANUP_S = 12.0
# Accept-rate threshold below which a pass is considered plateaued.
HIER_PLATEAU_ACCEPT_RATE = 0.002
# Proxy-gain threshold below which a pass is considered low-yield.
HIER_PLATEAU_PROXY_GAIN = 0.00005
# When local hard/swap operators plateau, spend a small bonus on soft-only repair.
HIER_PLATEAU_SOFT_REPAIR_BONUS_BUDGET_S = 4.0
HIER_PLATEAU_SOFT_REPAIR_BONUS_ROUNDS = 1
HIER_PLATEAU_SOFT_REPAIR_BONUS_MIN_SPARE_S = 6.0
# Adds a short alternate soft-relocation proposal class after plateaued swaps.
HIER_PLATEAU_ESCAPE_BUDGET_S = 4.0
HIER_PLATEAU_ESCAPE_MIN_SPARE_S = 5.0
HIER_PLATEAU_ESCAPE_SOFT_TOP_K = 384
HIER_PLATEAU_ESCAPE_SOFT_TARGETS = 10
HIER_PLATEAU_ESCAPE_MIN_GAIN = 0.00005

# Stronger exact-gated late soft repair. This spends spare budget on soft
# macros, which can relieve congestion without reopening hard legality.
HIER_STRONG_SOFT_REPAIR_BUDGET_S = 12.0
HIER_STRONG_SOFT_REPAIR_MIN_SPARE_S = 2.0
HIER_STRONG_SOFT_REPAIR_ROUNDS = 2
HIER_STRONG_SOFT_REPAIR_TOP_K = 512
HIER_STRONG_SOFT_REPAIR_TARGETS = 12
HIER_STRONG_SOFT_REPAIR_MIN_GAIN = 0.00005
HIER_STRONG_SOFT_REPAIR_WL_PREFILTER = 0.0005
# Extra soft-only continuation for medium/large congestion cases where the
# normal strong-soft pass is still producing exact proxy gain. This is
# structural, not benchmark-name gated.
HIER_MEDIUM_SOFT_CONTINUATION = True
HIER_MEDIUM_SOFT_HARD_MIN = 520
HIER_MEDIUM_SOFT_HARD_MAX = 760
HIER_MEDIUM_SOFT_MACRO_MIN = 2200
HIER_MEDIUM_SOFT_MACRO_MAX = 3200
HIER_MEDIUM_SOFT_NETS_PER_MACRO_MIN = 12.0
HIER_MEDIUM_SOFT_NETS_PER_MACRO_MAX = 24.0
HIER_MEDIUM_SOFT_TRIGGER_GAIN = 0.004
HIER_MEDIUM_SOFT_BUDGET_S = 6.0
HIER_MEDIUM_SOFT_MIN_SPARE_S = 2.0
HIER_MEDIUM_SOFT_ROUNDS = 2
HIER_MEDIUM_SOFT_TOP_K = 768
HIER_MEDIUM_SOFT_TARGETS = 14
HIER_MEDIUM_SOFT_MIN_GAIN = 0.00005
# Soft-only repair interleaved before hard/soft swap search.
HIER_INTERLEAVED_SOFT_REPAIR_BUDGET_S = 3.0
HIER_INTERLEAVED_SOFT_REPAIR_MIN_SPARE_S = 12.0
HIER_INTERLEAVED_SOFT_REPAIR_TOP_K = 256
HIER_INTERLEAVED_SOFT_REPAIR_TARGETS = 8
HIER_INTERLEAVED_SOFT_REPAIR_MIN_GAIN = 0.00005

# Region-locked hard and soft congestion relief.
# Target packing density used to size cluster region boxes.
HIER_REGION_DENSITY = 0.65
# Optional canvas-fraction margin fallback for region sizing.
HIER_REGION_MARGIN = 0.0
# Local-window half-width fraction for unclustered singleton macros.
HIER_REGION_SINGLETON = 0.05
# Extra region area fraction granted to hot clusters before relief.
HIER_REGION_HEAT_FRAC = 0.04
# Percentile threshold used to decide which clusters are heat-expanded.
HIER_REGION_HEAT_HOT_PCT = 70.0
# Minimum normalized heat scale applied when expanding hot regions.
HIER_REGION_HEAT_ESCAPE_MIN = 0.25
# Directional region expansion toward colder congestion bands.
# Congestion percentile used to select clusters for directional expansion.
HIER_REGION_EXPAND_HOT_PCT = 60.0
# Maximum fractional side expansion applied by congestion-band expansion.
HIER_REGION_EXPAND_FRAC = 0.08
# Number of grid cells sampled on each side when choosing expansion direction.
HIER_REGION_EXPAND_BAND = 3
# Component-aware expansion grows hot hierarchy regions toward contiguous cold
# congestion components rather than only comparing thin side bands.
HIER_REGION_COMPONENT_EXPAND = True
HIER_REGION_COMPONENT_COLD_PCT = 45.0
HIER_REGION_COMPONENT_MIN_CELLS = 4
HIER_REGION_COMPONENT_MAX_DISTANCE_CELLS = 4
# Bias component-aware region expansion toward cold components near hierarchy
# graph edge corridors for graph-tension-enabled large designs.
HIER_REGION_GRAPH_COMPONENT_WEIGHT = 0.0
# Extra early region room for hot clusters with weak inferred hierarchy
# confidence. This reshapes the hierarchy boxes before local relief instead of
# waiting for post-survivor small-design release.
HIER_REGION_WEAK_HOT_RESHAPE = False
HIER_REGION_WEAK_CONFIDENCE_MAX = 0.92
HIER_REGION_WEAK_HOT_MAX_CLUSTERS = 2
HIER_REGION_WEAK_HOT_EXTRA_FRAC = 0.03
HIER_REGION_WEAK_HOT_SIDE_FLOOR = 0.45
# Candidate gate for weak/hot reshape. Keep the experiment scoped to the
# small-design region where the full sweep showed useful wins and away from
# larger region-relief cases that regressed.
HIER_REGION_WEAK_HOT_HARD_MIN = 240
HIER_REGION_WEAK_HOT_HARD_MAX = 420
HIER_REGION_WEAK_HOT_MACRO_MAX = 1600
HIER_REGION_WEAK_HOT_REQUIRE_RELEASE_CANDIDATE = True
# Weight favoring candidates that stay inside hierarchy regions.
REGION_BIAS = 1.0
# Minimum proxy improvement required for moves that escape their hierarchy region.
HIER_REGION_ESCAPE_MIN = 0.002
# Number of region-relief rounds before final legalization.
HIER_REGION_ROUNDS = 2
# Wall-clock budget for region relief and its nested passes.
HIER_REGION_BUDGET_S = 40.0
# Bridge-soft classification instead of assigning every soft to one owner.
# Relative affinity threshold for classifying soft macros as bridge softs.
HIER_BRIDGE_SOFT_RATIO = 0.6

# Tiny exact-gated hard/soft shifts inside hierarchy regions.
# Maximum grid-cell radius considered by micro-shift polish.
HIER_MICRO_SHIFT_RADIUS = 2
# Number of hottest macros considered by each micro-shift pass.
HIER_MICRO_SHIFT_TOP = 96
# Minimum exact-proxy gain required for a micro-shift move.
HIER_MICRO_SHIFT_MIN_GAIN = 0.00001
# Replays micro-shift polish after region swaps.
# When region swaps run multiple rounds, replay micro-shift after
# each swap round instead of waiting until the full swap pass completes.
# Wall-clock budget for the post-swap micro-shift replay.
HIER_POST_SWAP_MICRO_SHIFT_BUDGET_S = 8.0
# Replays micro-shift polish after coldspot tightening.
# Wall-clock budget for the post-coldspot micro-shift replay.
HIER_POST_COLDSPOT_MICRO_SHIFT_BUDGET_S = 8.0

# Exact-gated decompression of congested hierarchy clusters.
# Stronger opportunity gates for expensive decompression/coldspot
# passes. These gates skip optional passes when the current congestion field has
# too little hot-vs-cold separation to justify spending runtime.
HIER_DECOMPRESS_MIN_FIELD_GAP = 0.08
HIER_COLDSPOT_STRONG_MIN_FIELD_GAP = 0.04
# Wall-clock budget for cluster decompression inside a region-relief round.
HIER_DECOMPRESS_BUDGET_S = 18.0
# Number of cluster-decompression rounds to attempt.
HIER_DECOMPRESS_ROUNDS = 2
# Congestion percentile used to select hot clusters for decompression.
HIER_DECOMPRESS_HOT_PCT = 65.0
# Candidate expansion factors tested when decompressing a hot cluster.
HIER_DECOMPRESS_FACTORS = (1.08, 1.16, 1.25)
# Minimum exact-proxy gain required to keep a decompression move.
HIER_DECOMPRESS_MIN_GAIN = 0.0001
# Maximum allowed degradation in hierarchy-quality score for decompression.
HIER_QUALITY_BUDGET = 0.03
# Composite hierarchy-quality metric weights. Lower quality score is better.
HIER_QUALITY_RADIUS_WEIGHT = 0.75
HIER_QUALITY_BBOX_WEIGHT = 0.20
HIER_QUALITY_CROWD_WEIGHT = 0.05
# Anisotropic decompression toward the colder axis.
# Grid-cell band sampled to choose anisotropic decompression direction.
HIER_DECOMPRESS_ANISO_BAND = 3
# Secondary-axis expansion ratio during anisotropic decompression.
HIER_DECOMPRESS_ANISO_SECONDARY = 0.25
# Bias decompression toward a nearby contiguous cold component while preserving
# the cluster as a local shape instead of splitting the hierarchy group.
HIER_DECOMPRESS_LOCAL_COMPONENT = True
HIER_DECOMPRESS_LOCAL_COLD_PCT = 45.0
HIER_DECOMPRESS_LOCAL_MIN_CELLS = 4
HIER_DECOMPRESS_LOCAL_MAX_DISTANCE_CELLS = 4
HIER_DECOMPRESS_LOCAL_SHIFT_FRAC = 0.20
# Cheap decompression feasibility screen. Before legalizing/scoring a candidate,
# estimate whether the proposed cluster bbox has enough unblocked local area.
HIER_DECOMPRESS_FEASIBILITY_FILTER = True
HIER_DECOMPRESS_FEASIBILITY_MIN_FREE_RATIO = 0.70
HIER_DECOMPRESS_FEASIBILITY_MAX_BLOCKAGE = 0.75
# Graph-guided decompression rescue. When a candidate improves graph-edge
# geometry but fails cheap feasibility or hard legalization, try smaller and
# slightly shifted variants before discarding it. Final acceptance still requires
# hard legality, hierarchy quality, exact proxy gain, and final audit pass.
HIER_DECOMPRESS_GRAPH_RESCUE = False
HIER_DECOMPRESS_GRAPH_RESCUE_MAX_DELTA = 0.0
HIER_DECOMPRESS_GRAPH_RESCUE_SHRINKS = (0.75, 0.55, 0.35)
HIER_DECOMPRESS_GRAPH_RESCUE_SHIFT_MULTS = (1.35, 0.70)
HIER_DECOMPRESS_GRAPH_RESCUE_MAX_VARIANTS = 5
# Default-on survivor for legal, hierarchy-safe decompression near misses with
# strongly favorable graph-edge delta. It exact-scores a tiny local hard/soft
# micro-shift pool around moved cluster members; final commit still requires a
# normal exact-proxy gain and audit pass.
HIER_DECOMPRESS_GRAPH_SURVIVOR = True
HIER_DECOMPRESS_GRAPH_SURVIVOR_MAX_DELTA = -0.01
HIER_DECOMPRESS_GRAPH_SURVIVOR_PROXY_MISS = 0.0015
HIER_DECOMPRESS_GRAPH_SURVIVOR_TOP_HARD = 8
HIER_DECOMPRESS_GRAPH_SURVIVOR_TOP_SOFT = 8
HIER_DECOMPRESS_GRAPH_SURVIVOR_RADIUS_CELLS = 1
HIER_DECOMPRESS_GRAPH_SURVIVOR_MAX_TRIALS = 48
# Advisory hierarchy-graph pressure ordering. These weights only reorder
# existing candidates; exact proxy, hard legality, and hierarchy audit gates
# still decide whether a candidate commits.
HIER_GRAPH_TENSION_ORDER = True
HIER_GRAPH_TENSION_WEIGHT = 0.10
HIER_GRAPH_TENSION_DECOMP_WEIGHT = 0.10
HIER_GRAPH_TENSION_COLDSPOT_WEIGHT = 0.10
HIER_GRAPH_TENSION_SWAP_WEIGHT = 0.0
HIER_GRAPH_TENSION_HARD_MIN = 600
HIER_GRAPH_TENSION_HARD_MAX = 1000000
HIER_GRAPH_TENSION_CORRIDOR_SAMPLES = 9
HIER_GRAPH_TENSION_TRACE_TOP = 6
# Conservative graph-aware rejection prefilters. These only skip low graph-tension
# candidates whose cheap local congestion estimate does not improve before exact
# scoring/refinement; high-tension candidates still reach the normal exact gates.
HIER_GRAPH_PREFILTER = False
HIER_GRAPH_PREFILTER_LOW_TENSION = 0.05
HIER_GRAPH_PREFILTER_MIN_RELIEF = 0.0
# Region-bounded hard-hard, hard-soft, and soft-soft swap relief.
# Number of region-bounded swap rounds to attempt.
HIER_REGION_SWAP_ROUNDS = 2
# Wall-clock budget for region-bounded swap relief.
HIER_REGION_SWAP_BUDGET_S = 20.0
# Number of hard candidates considered per hard swap source.
HIER_HARD_SWAP_K = 16
# Number of soft candidates considered per soft swap source.
HIER_SOFT_SWAP_K = 48
# Minimum exact-proxy gain required for a swap move.
HIER_SWAP_MIN_GAIN = 0.00001
# Optional soft-macro barrier for soft relocation and soft-involving swaps.
# Keep 0.0 for production parity; set to 0.01 in regional GNN diagnostics.
HIER_SOFT_BARRIER_GAIN = 0.0
# Minimum congestion-field relief required for a swap move.
HIER_SWAP_MIN_FIELD_RELIEF = 0.0
# Enable mask-aware swap region checks when a region mask is available.
HIER_SWAP_GRAPH_MASK_AWARE = True
# Draw graph-aware swap masks from the strongest graph relations only.
HIER_SWAP_GRAPH_MASK_MAX_EDGES = 0
HIER_SWAP_GRAPH_MASK_PAD_CELLS = 1
# Soft penalty for candidates outside graph-derived swap masks.
# Outside-mask candidates remain legal; they are only ranked with a small
# additional advisory penalty.
HIER_SWAP_GRAPH_MASK_PENALTY_WEIGHT = 0.30
# Graph-edge ranking pressure for swap candidates (diagnostic/rank only).
# Non-negative deltas indicate longer graph edges / worse corridor pressure.
HIER_SWAP_GRAPH_DELTA_WEIGHT = 0.0
# Optional single-shot fallback when a masked swap pass makes no accepts.
HIER_SWAP_GRAPH_FALLBACK = True
# Wall-clock budget reserved for the masked-pass fallback.
HIER_SWAP_GRAPH_FALLBACK_BUDGET_S = 2.5

# Optional bias for source-anchored coldspot expansion around graph/components.
# Source points are used to prefer low-congestion windows near cluster anchors.
HIER_COLDSPOT_SOURCE_POINT_WEIGHT = 0.20
# Region-bounded swap relief uses hard-hard, hard-soft, soft-soft,
# congestion-field, density-field, and batched exact scoring paths.

# Uses CUDA top-k/order kernels for large candidate-ranking arrays when available.
HIER_GPU_RANK_SWAP_CANDIDATES = "auto"
HIER_GPU_RANK_RELOCATION_TARGETS = "auto"
HIER_GPU_RANK_SOFT_RELOCATION_TARGETS = "auto"
HIER_GPU_RANK_MIN_CANDIDATES = 512
HIER_GPU_RANK_SOFT_MIN_CANDIDATES = 1024
HIER_GPU_SWAP_PRESCORE_SS = "auto"
HIER_GPU_SWAP_PRESCORE_HS = "auto"
HIER_GPU_SWAP_PRESCORE_HH = "auto"
HIER_GPU_SWAP_PRESCORE_MIN_CANDIDATES = 512
HIER_GPU_SWAP_PRESCORE_DISTANCE_WEIGHT = 0.02

# Adds supplemental candidates after the deterministic prefix when local budget remains.
# Rank only additive relocation tails with the lightweight torch/GPU heuristic.
# Default-off on IBM: the Stage 5 sweep saw no hard propose-all accepts and a
# small aggregate regression, so keep this as infrastructure for opt-in runs.
HIER_GPU_RANK_ADDITIVE_TAILS = False
# Extra hard propose-all relocation candidates exact-checked after the deterministic prefix.
HIER_ADDITIVE_RELOC_EXTRA_TOP_K = 8
# Extra swap candidates per source exact-checked after the deterministic neighbor prefix.
HIER_ADDITIVE_SWAP_EXTRA_K = 4
# Minimum seconds of local pass budget required before spending additive candidates.
HIER_ADDITIVE_MIN_SPARE_S = 2.0

# Final audit of hard-macro clearance using the same tolerance as local legality tests.
HIER_LEGALITY_MARGIN_EPS = 0.05
# Final audit of hierarchy-quality drift versus the selected hierarchy seed.
# If rollback is enabled, the finalizer returns the tracked best valid state
# when it satisfies this budget and the current state does not.
HIER_FINAL_HIER_AUDIT_MAX_DEGRADATION = 0.05
HIER_FINAL_HIER_AUDIT_ROLLBACK = True

# Wall-clock budget for post-swap hard propose-all relocation.
HIER_POST_RELOC_PROPOSE_BUDGET_S = 8.0
# Wall-clock budget for post-swap soft relocation.
HIER_POST_SOFT_RELOC_BUDGET_S = 8.0

# Proxy-aware coldspot tightening for hot clusters.
# Per-move proxy slack allowed during coldspot tightening.
HIER_COLDSPOT_BUDGET = 0.0
# Total proxy slack allowed versus coldspot baseline.
HIER_COLDSPOT_TOTAL = 0.0
# Minimum exact-proxy gain required to accept a coldspot tightening move.
HIER_COLDSPOT_MIN_GAIN = 0.0001
# Maximum hierarchy-quality degradation allowed during coldspot tightening.
HIER_COLDSPOT_QUALITY_BUDGET = 0.01
# Minimum hot-vs-cold field gap required before coldspot tightening runs.
HIER_COLDSPOT_MIN_FIELD_GAP = 0.02
# Cheap opportunity score required before a coldspot round spends candidate and
# exact-scoring time. The score blends hot-cold relief, open cold capacity, and
# source-to-cold displacement.
HIER_COLDSPOT_OPPORTUNITY_MIN_SCORE = 0.0
# Minimum open cold cells around the selected window before trying a whole-cluster kick.
HIER_COLDSPOT_OPPORTUNITY_MIN_COLD_CELLS = 1
# Stop coldspot tightening after this many generated-but-uncommitted rounds.
HIER_COLDSPOT_MAX_DRY_ROUNDS = 2
# Number of opportunity-ranked clusters tried in one coldspot round.
HIER_COLDSPOT_OPPORTUNITY_TOP_CLUSTERS = 2
# Number of coldspot tightening rounds to attempt.
HIER_COLDSPOT_ROUNDS = 8
# Wall-clock budget for coldspot tightening.
HIER_COLDSPOT_BUDGET_S = 30.0
# Refines each coldspot-kick candidate inside a local region before exact gating.
# Fraction of the kicked hard-core max dimension used as local-region pad.
HIER_COLDSPOT_LOCAL_HARD_PAD_FRAC = 0.50
# Minimum local-region pad in grid cells.
HIER_COLDSPOT_LOCAL_MIN_PAD_CELLS = 1
# Maximum local-region pad as a canvas fraction.
HIER_COLDSPOT_LOCAL_MAX_PAD_FRAC = 0.12
# Local hard-hard / hard-soft / soft-soft swap rounds per kicked candidate.
HIER_COLDSPOT_LOCAL_SWAP_ROUNDS = 1
# Hard swap candidates per local source macro.
HIER_COLDSPOT_LOCAL_HARD_SWAP_K = 12
# Soft swap candidates per local source macro.
HIER_COLDSPOT_LOCAL_SOFT_SWAP_K = 24
# Minimum exact-proxy gain that lets soft-only moves leave the local region.
HIER_COLDSPOT_LOCAL_SOFT_ESCAPE_MIN = 0.0025
# Hard relocation hot-source cap inside the local coldspot region.
HIER_COLDSPOT_LOCAL_HARD_RELOC_TOP_K = 24
# Soft relocation hot-source cap inside the local coldspot region.
HIER_COLDSPOT_LOCAL_SOFT_RELOC_TOP_K = 64
# Candidate cold cells considered per local relocation source.
HIER_COLDSPOT_LOCAL_RELOC_TARGETS = 8
# Number of shape-preserving whole-cluster variants generated for each selected
# opportunity-ranked coldspot cluster when the GNN selector is not driving the
# candidate count.
HIER_COLDSPOT_WHOLE_VARIANTS = 5
# Number of source clusters considered when GNN/Oracle policy is active.
# Default 1 keeps historical single-source behavior.
HIER_COLDSPOT_GNN_MAX_CLUSTERS = 1
# Number of distinct low-congestion anchors considered by whole-cluster variants.
HIER_COLDSPOT_ANCHOR_VARIANTS = 3
# Component-derived anchor candidates are ranked by coldness, area, and source
# proximity when source context is available.
HIER_COLDSPOT_COMPONENT_ANCHORS = 2
# Optional graph-aware coldspot anchor ranking. When positive, cold window
# anchors are still cold-first, but ties and near-ties prefer locations closer
# to the selected cluster's weighted graph-neighbor centroid.
HIER_COLDSPOT_GRAPH_ANCHOR_WEIGHT = 0.0
HIER_COLDSPOT_GRAPH_ANCHOR_CANDIDATE_MULT = 32
# Default-off exact-candidate reordering by graph-edge delta. Positive deltas
# stretch hierarchy graph edges or worsen graph corridors, so the opt-in ranker
# adds a small proxy-equivalent penalty before the usual graph-score tie-break.
HIER_COLDSPOT_GRAPH_DELTA_RANK = False
HIER_COLDSPOT_GRAPH_DELTA_WEIGHT = 0.0
# Default-off ego-net coldspot candidates. When enabled, the selected high
# tension/hot cluster may co-move a small strongest-neighbor cluster set through
# the same whole-candidate, local-refine, exact-proxy, and hierarchy gates.
HIER_COLDSPOT_EGONET = False
HIER_COLDSPOT_EGONET_MAX_NEIGHBORS = 1
HIER_COLDSPOT_EGONET_MAX_HARD = 96
HIER_COLDSPOT_EGONET_MAX_NEIGHBOR_HARD = 32
HIER_COLDSPOT_EGONET_MIN_EDGE_WEIGHT = 0.0
HIER_COLDSPOT_EGONET_CANDIDATES = 2
HIER_COLDSPOT_EGONET_SOFT_MODE = "none"
HIER_COLDSPOT_EGONET_LOW_DISP_BLEND = 0.75
HIER_COLDSPOT_EGONET_MIN_GAIN = 0.001
# Compacting scale used by shape-preserving whole-cluster layouts.
HIER_COLDSPOT_COMPACT_SPREAD = 0.72
# Blend toward the current cluster centroid for lower-displacement candidates.
HIER_COLDSPOT_LOW_DISP_BLEND = 0.45
# Stage G2/G6: use graph-derived cell pools and masks for local hard/soft
# relocation targets. Graph-local fallback runs when coldspot kicks produce no
# accepted move.
# Number of hot clusters considered by graph-local fallback.
HIER_COLDSPOT_GRAPH_FALLBACK_TOP_K = 3
# Default-off soft-only fallback for coldspot cleanup. When no hard coldspot
# candidate commits, this tries exact-gated movable soft relocation into open
# remembered cold cells while preserving hierarchy region boxes.
HIER_COLDSPOT_SOFT_ONLY = False
# Hot soft macros considered by the soft-only coldspot fallback.
HIER_COLDSPOT_SOFT_ONLY_TOP_K = 96
# Candidate cold cells considered per soft source in the soft-only fallback.
HIER_COLDSPOT_SOFT_ONLY_TARGETS = 10
# Minimum exact-proxy gain required by the soft-only coldspot fallback.
HIER_COLDSPOT_SOFT_ONLY_MIN_GAIN = 0.00005
# Remembered cold-cell graph expansion for local coldspot refinement.
# Field percentile used to remember cold cells for adaptive local regions.
HIER_COLDSPOT_MEMORY_COLD_PCT = 35.0
# Maximum grid-cell distance flooded from a cluster box into adjacent cold cells.
HIER_COLDSPOT_ADAPTIVE_MAX_CELLS = 5
# Generates one default-off capacity-aware partial frontier candidate alongside
# the normal whole-cluster coldspot kick. Exact proxy and hierarchy gates still
# decide whether the candidate can commit.
HIER_COLDSPOT_PARTIAL_FRONTIER = False
# Maximum number of partial frontier candidates added to one coldspot pool.
HIER_COLDSPOT_PARTIAL_CANDIDATES = 1
# Fill fraction applied to the connected cold-area capacity estimate.
HIER_COLDSPOT_PARTIAL_FILL_FRAC = 0.75
# Maximum fraction of the source hard-cluster area a partial frontier candidate
# may move. This keeps the mode distinct from the whole-cluster kick.
HIER_COLDSPOT_PARTIAL_MAX_AREA_FRAC = 0.55
# Minimum source hard-cluster size for partial frontier. Tiny clusters tend to
# become far 2-of-3 splits that improve proxy but fail hierarchy quality.
HIER_COLDSPOT_PARTIAL_MIN_CLUSTER_HARD = 6
# Minimum hard macros moved by a partial frontier candidate.
HIER_COLDSPOT_PARTIAL_MIN_HARD = 2
# Minimum hard macros left behind in the source cluster.
HIER_COLDSPOT_PARTIAL_MIN_REMAINING_HARD = 3
# Maximum selected hard-macro fraction before rejecting majority splits.
HIER_COLDSPOT_PARTIAL_MAX_MEMBER_FRAC = 0.50
# Maximum selected-vs-remaining connectivity cut ratio before rejecting a split.
HIER_COLDSPOT_PARTIAL_MAX_CUT_RATIO = 0.85
# Selected hard macros must form one local low-fanout connectivity component
# when such edges are available.
# Cheap pre-exact split-shape guard. Reject partial candidates predicted to
# stretch the source hierarchy cluster beyond these local shape ratios.
HIER_COLDSPOT_PARTIAL_MAX_RADIUS_RATIO = 1.15
HIER_COLDSPOT_PARTIAL_MAX_BBOX_RATIO = 1.20
HIER_COLDSPOT_PARTIAL_MAX_SEPARATION_RATIO = 1.50

# Bounded go-with-the-winners survivor search after coldspot cleanup. The pass
# keeps a small pool of valid hierarchy-preserving states instead of continuing
# from one greedy state.
# Wall-clock budget for survivor search.
HIER_SURVIVOR_BUDGET_S = 12.0
# Number of survivor generations.
HIER_SURVIVOR_ROUNDS = 2
# Number of placement states kept between generations.
HIER_SURVIVOR_WIDTH = 4
# Number of hottest hierarchy clusters used to generate each candidate pool.
HIER_SURVIVOR_HOT_CLUSTERS = 6
# Candidate cluster translation distances in congestion-grid cells.
HIER_SURVIVOR_STEP_CELLS = (2.0, 4.0, 7.0)
# Exact-score only the best cheap-ranked candidates per generation.
HIER_SURVIVOR_EXACT_TOP_K = 10
# Minimum exact-proxy gain required to commit the final survivor result.
HIER_SURVIVOR_MIN_GAIN = 0.0001
# Maximum hierarchy-quality degradation allowed for survivor candidates.
HIER_SURVIVOR_QUALITY_BUDGET = 0.015
# Uses CUDA for cheap candidate-pool ranking when available.
HIER_SURVIVOR_GPU_RANK = "auto"

# Extra exact-gated late polish for the SA-ratio primary struggle subset shape:
# small hard-macro population, no fixed hard macros, and moderate total macro
# count. This is structural, not benchmark-name gated.
HIER_SMALL_DESIGN_POLISH = True
HIER_SMALL_DESIGN_HARD_MIN = 240
HIER_SMALL_DESIGN_HARD_MAX = 420
HIER_SMALL_DESIGN_MACRO_MAX = 1600
HIER_SMALL_DESIGN_BUDGET_S = 14.0
HIER_SMALL_DESIGN_ROUNDS = 2
HIER_SMALL_DESIGN_MIN_GAIN = 0.00005
# Release weak inferred hierarchy regions inside the small-design polish.
# Selection starts from the weakest-k clusters by confidence, keeps only clusters
# below the confidence cutoff, then releases the hottest eligible weak clusters.
HIER_SMALL_DESIGN_RELEASE_CONFIDENCE_MAX = 0.92
HIER_SMALL_DESIGN_RELEASE_WEAKEST_K = 4
HIER_SMALL_DESIGN_RELEASE_MAX_CLUSTERS = 8
HIER_SMALL_DESIGN_RELEASE_ESCAPE_MIN = 0.00005
# Split the small-design pass into a low-connectivity lane and a high-net lane
# for ibm15-like cases. Exact proxy remains the accept gate in both lanes.
HIER_SMALL_DESIGN_HIGH_NETS_PER_MACRO = 24.0
HIER_SMALL_DESIGN_LOW_HARD_TOP_K = 96
HIER_SMALL_DESIGN_LOW_HARD_TARGETS = 20
HIER_SMALL_DESIGN_LOW_HARD_PROPOSE_TOP_M = 24
HIER_SMALL_DESIGN_LOW_SOFT_TOP_K = 256
HIER_SMALL_DESIGN_LOW_SOFT_TARGETS = 10
# Small low-net designs with no releasable weak cluster get more soft/SS
# candidate breadth and less hard-relocation breadth. This targets the shape,
# not a benchmark name, and exact proxy still gates every accepted move.
HIER_SMALL_DESIGN_NO_RELEASE_LOW_NET_HARD_TOP_K = 64
HIER_SMALL_DESIGN_NO_RELEASE_LOW_NET_HARD_TARGETS = 12
HIER_SMALL_DESIGN_NO_RELEASE_LOW_NET_HARD_PROPOSE_TOP_M = 16
HIER_SMALL_DESIGN_NO_RELEASE_LOW_NET_SOFT_TOP_K = 384
HIER_SMALL_DESIGN_NO_RELEASE_LOW_NET_SOFT_TARGETS = 12
HIER_SMALL_DESIGN_NO_RELEASE_LOW_NET_SWAP_SOFT_K = 24
HIER_SMALL_DESIGN_HIGH_HARD_TOP_K = 128
HIER_SMALL_DESIGN_HIGH_HARD_TARGETS = 24
HIER_SMALL_DESIGN_HIGH_HARD_PROPOSE_TOP_M = 32
HIER_SMALL_DESIGN_HIGH_SOFT_TOP_K = 384
HIER_SMALL_DESIGN_HIGH_SOFT_TARGETS = 12
HIER_SMALL_DESIGN_HARD_SWAP_K = 8
HIER_SMALL_DESIGN_SWAP_HARD_K = 8
HIER_SMALL_DESIGN_SWAP_SOFT_K = 16
HIER_SMALL_DESIGN_SWAP_MIN_GAIN = 0.00005
# Cold connected-component pools for small-design relocation target selection.
# The pool builder finds connected low-field grid components, prefers larger and
# colder components, and passes a component penalty into relocation ranking.
HIER_COLD_COMPONENT_TARGETS = True
HIER_COLD_COMPONENT_PCT = 45.0
HIER_COLD_COMPONENT_MAX_COMPONENTS = 8
HIER_COLD_COMPONENT_MIN_CELLS = 4
HIER_COLD_COMPONENT_SIZE_WEIGHT = 0.35
HIER_COLD_COMPONENT_RANK_WEIGHT = 0.04

# Weight for structural candidate ordering inside hierarchy relocation.
HIER_OBJECTIVE_STRUCTURAL_WEIGHT = 0.0
# Relative weight for edge keep-out structural penalty.
HIER_KEEP_OUT_WEIGHT = 0.2
# Relative weight for grid-alignment structural penalty.
HIER_GRID_ALIGN_WEIGHT = 0.2
# Relative weight for notch-avoidance structural penalty.
HIER_NOTCH_WEIGHT = 0.6
# Minimum pair count before structural notch scoring uses numba.
HIER_STRUCTURAL_NOTCH_NUMBA_MIN_PAIRS = 24
# Enables experimental GPU path for structural notch scoring.
HIER_STRUCTURAL_NOTCH_GPU = False
# Minimum macro count before structural notch scoring may use GPU.
HIER_STRUCTURAL_NOTCH_GPU_MIN_N = 128

# Scorer implementation used for propose-all relocation candidates.
RELOC_PROPOSE_SCORER = "cuda_delta"
# Default CUDA chunk size for relocation proposal scoring.
RELOC_PROPOSE_DEFAULT_CUDA_CHUNK_SIZE = 128
# Safety multiplier for estimated relocation proposal memory usage.
RELOC_PROPOSE_MEM_SAFETY = 1.0
# Default CUDA memory fraction used for automatic proposal scoring chunk sizing.
RELOC_PROPOSE_AUTO_MEM_FRAC = 0.75
# Reuse the exact CUDA delta scorer for bounded/local sequential relocation
# target batches when CUDA is available. Falls back to the serial scorer.
HIER_LOCAL_RELOC_CUDA_DELTA = "auto"
HIER_LOCAL_RELOC_CUDA_DELTA_MIN_TARGETS = 64

# Enables aggregate profiling of exact proxy scoring calls.
PROFILE_EXACT = False
# Routing congestion uses numba strip application when available. Incremental
# scoring reuses cached congestion fields when available.
