"""Accepted production constants for hierarchy placement."""

# Per-benchmark placement time budget used by the MacroPlacer constructor default.
TIME_BUDGET_S = 150.0

# Maximum net fanout considered as hierarchy signal when forming hard clusters.
CLUSTER_MAX_FANOUT = 8
# Minimum shared low-fanout net count required to merge two hard macros.
CLUSTER_MIN_EDGE = 2

# DREAMPlace group attraction weight for cluster grouping constraints.
HIER_GROUP_WEIGHT = 8
# Orders legalization by connectivity pressure inside clusters instead of area only.
HIER_LEGALIZE_CONNECTIVITY_ORDER = True

# Enables post-swap hard propose-all relocation automatically on CUDA backends.
HIER_POST_RELOC_PROPOSE_ALL = "auto"
# Caps hard propose-all candidates per hot macro after region swaps.
HIER_POST_RELOC_PROPOSE_TOP_M = 16
# Number of hottest hard macros considered for post-swap propose-all relocation.
HIER_RELOC_PROPOSE_HOT_K = 32
# Enables post-swap soft macro relocation inside soft hierarchy regions.
HIER_POST_SOFT_RELOC = True
# Number of hottest soft macros considered by post-swap soft relocation.
HIER_POST_SOFT_RELOC_TOP_K = 256
# Minimum exact-proxy gain required to accept a post-swap soft relocation.
HIER_POST_SOFT_RELOC_MIN_GAIN = 0.0005
# Minimum exact-proxy gain required to accept hard propose-all relocation.
HIER_RELOC_PROPOSE_MIN_GAIN = 0.0005

# Enables region-locked hard and soft congestion relief.
HIER_REGION_RELIEF = True
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
# Enables directional region expansion toward colder congestion bands.
HIER_CONG_EXPAND_REGIONS = True
# Congestion percentile used to select clusters for directional expansion.
HIER_REGION_EXPAND_HOT_PCT = 60.0
# Maximum fractional side expansion applied by congestion-band expansion.
HIER_REGION_EXPAND_FRAC = 0.08
# Number of grid cells sampled on each side when choosing expansion direction.
HIER_REGION_EXPAND_BAND = 3
# Weight favoring candidates that stay inside hierarchy regions.
REGION_BIAS = 1.0
# Minimum proxy improvement required for moves that escape their hierarchy region.
HIER_REGION_ESCAPE_MIN = 0.002
# Number of region-relief rounds before final legalization.
HIER_REGION_ROUNDS = 2
# Wall-clock budget for region relief and its nested passes.
HIER_REGION_BUDGET_S = 40.0
# Enables bridge-soft classification instead of assigning every soft to one owner.
HIER_BRIDGE_SOFTS = True
# Relative affinity threshold for classifying soft macros as bridge softs.
HIER_BRIDGE_SOFT_RATIO = 0.6

# Enables tiny exact-gated hard/soft shifts inside hierarchy regions.
HIER_MICRO_SHIFT = True
# Maximum grid-cell radius considered by micro-shift polish.
HIER_MICRO_SHIFT_RADIUS = 2
# Number of hottest macros considered by each micro-shift pass.
HIER_MICRO_SHIFT_TOP = 96
# Minimum exact-proxy gain required for a micro-shift move.
HIER_MICRO_SHIFT_MIN_GAIN = 0.00001
# Replays micro-shift polish after region swaps.
HIER_POST_SWAP_MICRO_SHIFT = True
# Wall-clock budget for the post-swap micro-shift replay.
HIER_POST_SWAP_MICRO_SHIFT_BUDGET_S = 8.0
# Replays micro-shift polish after coldspot tightening.
HIER_POST_COLDSPOT_MICRO_SHIFT = True
# Wall-clock budget for the post-coldspot micro-shift replay.
HIER_POST_COLDSPOT_MICRO_SHIFT_BUDGET_S = 8.0

# Enables exact-gated decompression of congested hierarchy clusters.
HIER_DECOMPRESS = True
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
# Enables anisotropic decompression toward the colder axis.
HIER_DECOMPRESS_ANISO = True
# Grid-cell band sampled to choose anisotropic decompression direction.
HIER_DECOMPRESS_ANISO_BAND = 3
# Secondary-axis expansion ratio during anisotropic decompression.
HIER_DECOMPRESS_ANISO_SECONDARY = 0.25
# Rolls back decompression when accepted moves fail to clear minimum gain.
HIER_ROLLBACK_WEAK_DECOMP = True

# Enables region-bounded hard-hard, hard-soft, and soft-soft swap relief.
HIER_REGION_SWAPS = True
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
# Minimum congestion-field relief required for a swap move.
HIER_SWAP_MIN_FIELD_RELIEF = 0.0
# Enables hard-hard swaps in region-bounded swap relief.
HIER_SWAP_HH = True
# Enables hard-soft swaps in region-bounded swap relief.
HIER_SWAP_HS = True
# Enables soft-soft swaps in region-bounded swap relief.
HIER_SWAP_SS = True
# Runs swap relief against both congestion and density fields.
HIER_SWAP_DENSITY_FIELD = True
# Enables experimental batched swap scoring path.
HIER_BATCH_SWAP_SCORES = False

# Wall-clock budget for post-swap hard propose-all relocation.
HIER_POST_RELOC_PROPOSE_BUDGET_S = 8.0
# Wall-clock budget for post-swap soft relocation.
HIER_POST_SOFT_RELOC_BUDGET_S = 8.0

# Enables proxy-aware coldspot tightening for hot clusters.
HIER_COLDSPOT_KICK = True
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
# Number of coldspot tightening rounds to attempt.
HIER_COLDSPOT_ROUNDS = 8
# Wall-clock budget for coldspot tightening.
HIER_COLDSPOT_BUDGET_S = 30.0

# Weight for structural candidate ordering inside hierarchy relocation.
HIER_OBJECTIVE_STRUCTURAL_WEIGHT = 0.0
# Relative weight for edge keep-out structural penalty.
HIER_KEEP_OUT_WEIGHT = 0.2
# Relative weight for grid-alignment structural penalty.
HIER_GRID_ALIGN_WEIGHT = 0.2
# Relative weight for notch-avoidance structural penalty.
HIER_NOTCH_WEIGHT = 0.6
# Enables numba acceleration for structural notch scoring.
HIER_STRUCTURAL_NOTCH_NUMBA = True
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

# Enables numba strip-application path for routing congestion construction.
ROUTE_STRUCT_JIT = False
# Enables aggregate profiling of exact proxy scoring calls.
PROFILE_EXACT = False
# Prefers the incremental scorer's cached congestion field when available.
USE_SCORER_CONGESTION_FIELD = True
