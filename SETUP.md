# Setup & API Reference

## Installation

```bash
# Clone the repository
git clone https://github.com/partcleda/macro-place-challenge-2026.git
cd macro-place-challenge-2026

# Initialize TILOS MacroPlacement submodule (required for evaluation)
git submodule update --init external/MacroPlacement

# Create virtual environment and install the package (editable)
uv sync

# Build the required pinned DREAMPlace runtime. This creates the ignored
# dreamplace_src/ and dreamplace_build/ trees without changing tracked code.
scripts/dreamplace/bootstrap.sh all
```

The bootstrap pins DREAMPlace commit
`37214b40fe3837cc7d392c7d6092ccd6ff04a02c`, CUDA 12.1, GCC 11.4,
PyTorch 2.4.1+cu121, and the repository's CUDA-12 CUB compatibility patch.
Set `DREAMPLACE_CUDA_ARCH` when building for a GPU other than compute
capability 8.9. To diagnose an existing install without rebuilding it:

```bash
scripts/dreamplace/bootstrap.sh preflight
```

Production performs the same native-extension import probe before placement.
An ABI-incompatible or incomplete build produces an actionable error instead
of silently falling back to another placer.

## Project Structure

```
├── macro_place/            # Installable Python package
│   ├── __init__.py
│   ├── benchmark.py        # Benchmark dataclass (PyTorch tensors)
│   ├── loader.py           # Load benchmarks from ICCAD04 format
│   ├── objective.py        # Proxy cost computation
│   ├── utils.py            # Validation and visualization
│   └── def_writer.py       # DEF file export
├── src/                    # Active hierarchy placer and EDA I/O
│   ├── main.py             # Evaluator-facing entrypoint
│   ├── placer/             # Hierarchy pipeline, scoring, routing, legalization
│   ├── dreamplace_bridge/  # ICCAD04 pb/plc <-> Bookshelf bridge
│   └── eda_io/             # LEF/DEF/Verilog/SDC/Liberty import/export
├── docs/                   # Active docs plus archived experiment notes
├── test/                   # Smoke tests, diagnostics, verification scripts
├── scripts/                # Comparison and conversion helpers
│   └── dreamplace/         # Pinned DREAMPlace bootstrap, patch, and preflight
├── ml_data/                # Historical traces/models/logs and generated data
├── system/                 # Optional historical checkpoints if present
│   └── v1/                 # Frozen checkpoint placer; read-only if present
├── external/
│   └── MacroPlacement/     # TILOS evaluator and ICCAD04 testcases
├── benchmarks/
│   └── processed/          # Pre-processed .pt benchmark files
├── pyproject.toml          # Package & dependency config (used by uv)
└── SETUP.md                # This file
```

## API Reference

### Loading a Benchmark

```python
from macro_place.loader import load_benchmark_from_dir

benchmark, plc = load_benchmark_from_dir('external/MacroPlacement/Testcases/ICCAD04/ibm01')
```

Returns:
- `benchmark`: A `Benchmark` dataclass with PyTorch tensors
- `plc`: A `PlacementCost` object (needed for cost computation)

### Benchmark Object

The `Benchmark` dataclass contains:

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Benchmark name (e.g., "ibm01") |
| `canvas_width` | `float` | Canvas width in microns |
| `canvas_height` | `float` | Canvas height in microns |
| `num_macros` | `int` | Total macros (hard + soft) |
| `num_hard_macros` | `int` | Number of hard macros (indices `[0, num_hard)`) |
| `num_soft_macros` | `int` | Number of soft macros (indices `[num_hard, num_macros)`) |
| `macro_positions` | `Tensor [N, 2]` | (x, y) center positions (hard first, then soft) |
| `macro_sizes` | `Tensor [N, 2]` | (width, height) of each macro |
| `macro_fixed` | `Tensor [N]` | Boolean mask of fixed macros |
| `macro_names` | `List[str]` | Macro names for debugging |
| `num_nets` | `int` | Number of nets |
| `grid_rows`, `grid_cols` | `int` | Grid dimensions for density/congestion |
| `hard_macro_indices` | `List[int]` | Map tensor index → PlacementCost module index (hard) |
| `soft_macro_indices` | `List[int]` | Map tensor index → PlacementCost module index (soft) |

Helper methods:
- `benchmark.get_movable_mask()` — returns `~macro_fixed`
- `benchmark.get_hard_macro_mask()` — True for hard macros (first `num_hard_macros` entries)
- `benchmark.get_soft_macro_mask()` — True for soft macros
- `benchmark.save(path)` / `Benchmark.load(path)` — serialize to/from `.pt` files

### Computing Proxy Cost

```python
from macro_place.objective import compute_proxy_cost

costs = compute_proxy_cost(placement, benchmark, plc)
```

**Input**: `placement` — a `[num_macros, 2]` tensor of (x, y) center positions.

**Output**: A dictionary with:

| Key | Description |
|-----|-------------|
| `proxy_cost` | Weighted sum: 1.0 × WL + 0.5 × density + 0.5 × congestion |
| `wirelength_cost` | Normalized HPWL across all nets |
| `density_cost` | Top 10% grid cell density |
| `congestion_cost` | Top 5% routing congestion with smoothing |
| `overlap_count` | Number of overlapping macro pairs |
| `total_overlap_area` | Total overlap area in μm² |
| `overlap_ratio` | Fraction of macros involved in overlaps |

### Validating a Placement

```python
from macro_place.utils import validate_placement

is_valid, violations = validate_placement(placement, benchmark)
```

Checks:
- Correct tensor shape
- No NaN/Inf values
- All macros within canvas bounds
- Fixed macros at original positions
- Zero macro-to-macro overlaps

### Visualizing a Placement

```python
from macro_place.utils import visualize_placement

visualize_placement(placement, benchmark, save_path='output.png')
```

## Writing a Placer

Your placer takes a `Benchmark` and returns a `[num_macros, 2]` tensor of positions. The tensor contains both hard macros (indices `[0, num_hard_macros)`) and soft macros (indices `[num_hard_macros, num_macros)`).

**Both hard and soft macros are movable.** Hard macros are the primary optimization targets (SRAMs, IPs, etc.). Soft macros are standard cell clusters — co-optimizing their positions alongside hard macros will improve wirelength, density, and congestion. The SA baseline does this by running force-directed placement on soft macros after each batch of hard macro moves.

```python
import torch
from macro_place.benchmark import Benchmark

class MyPlacer:
    def place(self, benchmark: Benchmark) -> torch.Tensor:
        placement = benchmark.macro_positions.clone()

        # Hard macros: indices [0, num_hard_macros)
        # Soft macros: indices [num_hard_macros, num_macros)
        # Both are movable — optimize both for best results

        hard_movable = benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()
        movable_indices = torch.where(hard_movable)[0]

        # Your algorithm here
        # - Move hard macros to optimize placement
        # - Optionally reposition soft macros to follow hard macro changes
        #   (the SA baseline uses PlacementCost.optimize_stdcells() for this)

        return placement
```

Key constraints:
- Positions are **center coordinates** (not corners)
- Fixed macros must stay at their original positions
- All macros must be fully within canvas bounds
- **Zero hard macro overlaps** required (soft macros may overlap — they are standard cell cluster abstractions)
- Moving hard macros without repositioning soft macros will degrade wirelength and density

See `src/main.py` for the current submission entrypoint. Historical examples
under `system/` may be absent after the root-layout migration; if `system/v1/`
exists, treat it as a frozen checkpoint rather than active code.

## Net Connectivity

Net connectivity is stored inside the `PlacementCost` object (`plc`), not in the `Benchmark` tensors. The proxy cost computation uses it automatically.

If you need direct access to net data for your algorithm (e.g., for a GNN), you can access it through the PlacementCost API:

```python
# Number of nets
print(plc.net_cnt)

# Access individual modules and their connections
for i, module in enumerate(plc.modules_w_pins):
    print(module.get_name(), module.get_pos())
```

See the [TILOS MacroPlacement source](https://github.com/TILOS-AI-Institute/MacroPlacement/blob/main/CodeElements/Plc_client/plc_client_os.py) for the full PlacementCost API.

## Soft Macro Optimization

Soft macros (standard cell clusters) are connected to hard macros via nets. When you move hard macros, the optimal soft macro positions change. The PlacementCost API provides a built-in force-directed placer for soft macros:

```python
# After setting hard macro positions, reoptimize soft macros:
canvas_size = max(benchmark.canvas_width, benchmark.canvas_height)
plc.optimize_stdcells(
    use_current_loc=False, move_stdcells=True, move_macros=False,
    log_scale_conns=False, use_sizes=False, io_factor=1.0,
    num_steps=[100, 100, 100],
    max_move_distance=[canvas_size/100]*3,
    attract_factor=[100, 1.0e-3, 1.0e-5],
    repel_factor=[0, 1.0e6, 1.0e7],
)
```

This is what the SA baseline does between iterations. Note: this is slow in Python (~minutes per call). You can reduce `num_steps` for faster but less optimal results, or implement your own soft macro optimization (e.g., jointly in a GPU-based optimizer).

## Running Benchmarks

### IBM Benchmarks (Tier 1 — Proxy Cost)

The 17 IBM ICCAD04 benchmarks are in `external/MacroPlacement/Testcases/ICCAD04/`.
Run a single benchmark or the full suite using the active hierarchy placer:

```bash
# Single benchmark
uv run evaluate src/main.py -b ibm10

# All 17 benchmarks with comparison table
uv run evaluate src/main.py --all
```

To evaluate your own placer on all benchmarks, follow the same pattern — loop over the benchmark directories:

```python
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost

BENCHMARKS = [
    "ibm01", "ibm02", "ibm03", "ibm04", "ibm06", "ibm07", "ibm08", "ibm09",
    "ibm10", "ibm11", "ibm12", "ibm13", "ibm14", "ibm15", "ibm16", "ibm17", "ibm18",
]

for name in BENCHMARKS:
    benchmark, plc = load_benchmark_from_dir(f'external/MacroPlacement/Testcases/ICCAD04/{name}')
    placement = my_placer.place(benchmark)
    costs = compute_proxy_cost(placement, benchmark, plc)
    print(f"{name}: proxy={costs['proxy_cost']:.4f}  overlaps={costs['overlap_count']}")
```

### NG45 Designs (Tier 2 — OpenROAD Flow)

The top 7 submissions by proxy score will be evaluated through the full OpenROAD PnR flow on NanGate45 designs. These designs are located in the TILOS repository:

```
external/MacroPlacement/Flows/NanGate45/
├── ariane133/    # RISC-V core, 133 macros
├── ariane136/    # RISC-V core, 136 macros
├── mempool_tile/ # Memory pool, 20 macros
└── nvdla/        # NVIDIA DLA, 128 macros
```

Pre-processed `.pt` versions are available in `benchmarks/processed/public/` for quick loading:

```python
from macro_place.benchmark import Benchmark

benchmark = Benchmark.load('benchmarks/processed/public/ariane133_ng45_random.pt')
```

The OpenROAD flow evaluation measures WNS (worst negative slack), TNS (total negative slack), and Area. Participants do not need to run OpenROAD themselves — the judges will run it on top submissions.

#### Running ORFS Locally (Optional)

If you want to test your placement through the full PnR flow locally, we provide `scripts/evaluate_with_orfs.py` which automates the entire process.

**Prerequisites**: Install [OpenROAD-flow-scripts](https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts) adjacent to this repository:

```bash
cd ..
git clone --depth=1 https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts
cd macro-place-challenge-2026
```

**Run the ORFS evaluation**:

```bash
# Evaluate a single NG45 design (uses default placement)
python scripts/evaluate_with_orfs.py --benchmark ariane133_ng45 --no-docker

# Evaluate with your own placement (saved as a [num_macros, 2] tensor)
python scripts/evaluate_with_orfs.py --benchmark ariane133_ng45 --no-docker \
    --placement my_placement.pt

# Evaluate all NG45 designs
python scripts/evaluate_with_orfs.py --all --no-docker

# Point to a custom ORFS installation
python scripts/evaluate_with_orfs.py --benchmark ariane133_ng45 \
    --orfs-root /path/to/OpenROAD-flow-scripts --no-docker
```

The script will:
1. Load the benchmark and compute proxy cost
2. Generate a macro placement TCL script (handling the name mapping between protobuf and ODB formats)
3. Copy the design config into ORFS with necessary patches
4. Run the full ORFS flow (synthesis → floorplan → placement → CTS → routing)
5. Parse and report WNS, TNS, Area, and other metrics

A full ORFS run takes approximately 3-8 hours per design depending on the benchmark and machine.
