# Partcl/HRT Macro Placement Challenge: Team Guide

> A complete reference for understanding, developing, and contributing to this project.

---

## Table of Contents

1. [What Is This Problem? (Simple Terms)](#1-what-is-this-problem)
2. [The Full Picture: Chip Design Flow](#2-the-full-chip-design-flow)
3. [The Proxy Cost Metric (What We're Optimizing)](#3-the-proxy-cost-metric)
4. [The Competition Setup](#4-the-competition-setup)
5. [Software Installation](#5-software-installation)
6. [Repository Structure](#6-repository-structure)
7. [How the Evaluation Works](#7-how-the-evaluation-works)
8. [Our Algorithm (sameer_v1)](#8-our-algorithm)
9. [How to Develop and Contribute as a Team](#9-team-development-workflow)
10. [How to Submit](#10-how-to-submit)
11. [Team Roles and Responsibilities](#11-team-roles)
12. [Related Research: Papers and Code](#12-related-research)
13. [Ideas for Improvement](#13-ideas-for-improvement)
14. [Glossary](#14-glossary)

---

## 1. What Is This Problem?

### Simple Analogy

Imagine you're designing a university campus. You have:
- **Buildings** (macros): fixed-size structures like the library, gym, lecture halls
- **Pathways** (wires): connections between buildings that students walk
- **Zones** (placement grid): rows/columns where buildings can sit

Your goal: **arrange the buildings so that:**
1. Frequently-connected buildings are close together (less walking = less wirelength)
2. Buildings are spread evenly (no zone is overcrowded = low density)
3. Pathways don't bottleneck (no corridor has too much traffic = low congestion)

In chip design, the "buildings" are **hard macros** (SRAM blocks, IP cores, analog circuits), and the "pathways" are metal wires connecting them.

### What Makes This Hard

- There are hundreds of macros with varying sizes
- Macros cannot overlap
- Moving one macro to improve wirelength often worsens congestion elsewhere
- The search space is astronomically large (continuous positions × orientations)
- Traditional tools (SA, force-directed) work but leave room for improvement

### The ML Opportunity

Google's 2021 Nature paper ("A graph placement methodology for fast chip design") showed that a reinforcement learning agent could place macros competitively with human engineers in seconds, after training on thousands of prior placements. This sparked a wave of research into ML-based placement, which is what this competition is about.

---

## 2. The Full Chip Design Flow

Here's the journey from idea to silicon, with our competition highlighted:

```
Specification
    ↓
RTL (Verilog/VHDL)          ← Human engineers write behavioral descriptions
    ↓ Logic Synthesis (Yosys, Design Compiler)
Gate-level Netlist           ← Logic gates (AND, OR, FF, etc.)
    ↓
Physical Design:
  ┌─────────────────────────────────────────────────────┐
  │  1. Floorplanning      ← Define canvas size, I/O   │
  │  2. MACRO PLACEMENT    ← ** THIS COMPETITION **     │
  │  3. Power Planning     ← VDD/GND grid               │
  │  4. Standard Cell PnR  ← Place small logic cells   │
  │  5. Clock Tree Synthesis ← Balance clock paths      │
  │  6. Routing            ← Connect all wires          │
  │  7. Sign-off           ← Timing, DRC, LVS checks   │
  └─────────────────────────────────────────────────────┘
    ↓ GDSII file (the "blueprint")
Semiconductor Fabrication (TSMC, Samsung, etc.)
    ↓
Physical Chip
```

### Step 2 in Detail: Macro Placement

At this step, we have:
- A **netlist** (`.pb.txt`): which macros connect to which, with what weight
- An **initial placement** (`initial.plc`): a starting guess for macro positions
- A **canvas**: the rectangular region where macros must fit

We output:
- A **final placement**: (x, y) coordinates for each macro's center

The evaluation tool (TILOS MacroPlacement, from Google/UCSD) then computes the **proxy cost**.

### Key EDA Tools

| Tool | Role | Used In Competition |
|------|------|---------------------|
| Yosys | Logic synthesis (RTL → netlist) | No (benchmarks pre-synthesized) |
| OpenROAD | Full physical design flow | Yes (Tier 2 evaluation) |
| ABC | Logic optimization | No |
| TILOS MacroPlacement | Proxy cost evaluator | Yes (Tier 1 evaluation) |

### What Is OpenROAD?

OpenROAD is an open-source EDA tool that runs the complete physical design flow from netlist to GDSII. It includes:
- Floorplanning
- Macro placement (what we're improving!)
- Standard cell placement
- Routing

In the competition, the **top 7 proxy-score submissions** are run through OpenROAD on NG45 designs to measure real PnR outcomes (WNS = worst negative slack, TNS = total negative slack, Area).

---

## 3. The Proxy Cost Metric

The competition ranks all submissions by:

```
Proxy Cost = 1.0 × Wirelength + 0.5 × Density + 0.5 × Congestion
```

All three components are normalized by the evaluation framework.

### Wirelength (WL)

Measures the total half-perimeter wirelength (HPWL) of all nets:

```
For each net with macros at positions p1, p2, ..., pk:
  HPWL(net) = (max_x - min_x) + (max_y - min_y)

Total WL = sum of HPWL across all nets
```

Lower = better (macros connected together are physically close).

### Density

Measures how evenly macros are distributed across the placement grid cells. High density = macros are clustered in certain areas, leaving others empty.

The grid is defined in the `.plc` file (e.g., 45×41 for ibm01). The evaluator computes the utilization of each cell and penalizes high-utilization cells.

### Congestion

Measures routing demand vs. routing capacity across horizontal/vertical channels. When macros cluster, wires must detour around them, creating hot spots.

The evaluator uses **routes per micron** parameters (from the `.plc` header) to compute congestion.

### Why It Matters

- Reducing WL usually improves timing and power
- Reducing density/congestion improves routability and prevents design-rule violations
- **The tension**: moving macros closer together reduces WL but increases density and congestion

This tension is the core algorithmic challenge. A naive WL optimizer can easily worsen proxy cost.

---

## 4. The Competition Setup

### Benchmarks

**Tier 1 (IBM ICCAD04)**: 17 benchmarks (`ibm01`–`ibm18`, excluding `ibm05`)
- Sizes: 99–942 hard macros
- Canvas: 12–160 μm²
- These are classic placement benchmarks from 2004

**Tier 2 (NG45 commercial designs)**: 4 designs
- `ariane133`, `ariane136` (RISC-V processors)
- `nvdla` (NVIDIA Deep Learning Accelerator)
- `mempool_tile` (parallel computing tile)

### Baselines

| Method | Avg Proxy Cost | Description |
|--------|---------------|-------------|
| SA (simulated annealing) | 1.9072 (ibm02 example) | Standard SA baseline |
| RePlAce | 1.8370 (ibm02 example) | Academic analytical placer |
| will_seed | ~1.5338 avg | Challenge organizer's seed solution |

### Prizes

- **$20K Grand Prize**: Best OpenROAD results among top 7 proxy scorers
- **$20K First Place**: #1 by proxy score (if no Grand Prize winner)
- **$5K Second Place**: Runner-up
- **$4K Innovation Award**: Most creative approach

### Deadline

**May 21, 2026, 11:59 PM Pacific**

### Current Competition Standing (as of April 2026)

| Benchmark | sameer_v1 | will_seed | RePlAce |
|-----------|-----------|-----------|---------|
| ibm01 | **1.2253** | ~1.29 | 0.998 |
| ibm02 | 1.6800 | ~1.68 | 1.837 |
| ibm03 | **1.4100** | ~1.74 | 1.322 |
| ibm07 | **1.4950** | ~2.02 | 1.463 |
| ibm09 | **1.1363** | ~1.39 | 1.119 |
| ibm17 | **1.7437** | ~3.67 | 1.645 |
| **AVG** | **1.5062** | **1.5338** | **1.4578** |

Our submission already beats will_seed by ~2.9%. The gap to RePlAce is 3.3%, which is worth chasing.

---

## 5. Software Installation

### Prerequisites

```bash
# Python 3.10 or higher
python --version  # should be 3.10+

# Git with submodule support
git --version
```

### Step 1: Clone and Initialize

```bash
git clone https://github.com/partcleda/macro-place-challenge-2026.git
cd macro-place-challenge-2026
git submodule update --init external/MacroPlacement
```

### Step 2: Install Python Dependencies

The project uses `uv` for fast dependency management:

```bash
# Install uv (package manager)
pip install uv
# OR on Linux/Mac:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies
uv sync

# Verify installation
uv run evaluate system/v0/greedy_row_placer.py -b ibm01
```

If `uv` is not available, use standard Python:
```bash
pip install -e .
python -m macro_place.evaluate system/v0/greedy_row_placer.py -b ibm01
```

### Step 3: (Optional) OpenROAD for Full Flow

OpenROAD is only needed for Tier 2 validation. It's complex to install. See the [OpenROAD installation guide](https://openroad.readthedocs.io/).

For development, the Tier 1 proxy evaluation is sufficient.

### Dependencies Summary

| Package | Version | Purpose |
|---------|---------|---------|
| torch | ≥2.0 | Tensor operations, GPU support |
| numpy | any | Array math |
| protobuf | any | Netlist file parsing |
| scipy | any | Scientific computing utils |

---

## 6. Repository Structure

```
macro-place-challenge-2026/
│
├── system/                       # Varrahan placers and active system work
│   ├── v0/                       # Simple reference placers
│   │   ├── greedy_row_placer.py  # Simple greedy baseline
│   │   └── simple_random_placer.py
│   ├── v1/placer.py              # Frozen checkpoint
│   └── v2/src/main.py            # Active placer entrypoint
│
├── macro_place/                  # Core framework
│   ├── evaluate.py               # Main evaluation harness
│   ├── benchmark.py              # Benchmark data structures
│   ├── loader.py                 # Netlist/plc file loading
│   └── visualize.py              # Placement visualization
│
├── external/MacroPlacement/      # TILOS evaluation submodule
│   ├── CodeElements/Plc_client/  # PlacementCost evaluator
│   └── Testcases/ICCAD04/        # ibm01-ibm18 benchmarks
│
├── scripts/
│   └── compare_placers.py        # Side-by-side comparison
│
├── benchmarks/                   # Benchmark metadata
│   └── metadata/                 # Per-benchmark statistics
│
├── SETUP.md                      # Detailed setup instructions
├── README.md                     # Competition overview
└── TEAM_GUIDE.md                 # This document
```

### Key Files to Understand

**`macro_place/benchmark.py`**: The `Benchmark` class:
```python
class Benchmark:
    name: str
    num_hard_macros: int          # Number of macros to place
    canvas_width: float           # Canvas size in microns
    canvas_height: float
    macro_sizes: torch.Tensor     # [N, 2] (width, height) per macro
    macro_positions: torch.Tensor # [N, 2] (x, y) initial positions

    def get_movable_mask(self) -> torch.Tensor  # Which macros can move
    def get_hard_macro_mask(self) -> torch.Tensor
    def evaluate(self, placement) -> dict       # Compute proxy cost
```

**`macro_place/evaluate.py`**: Running evaluations:
```bash
# Single benchmark
python -m macro_place.evaluate system/v1/placer.py -b ibm01

# All 17 IBM benchmarks
python -m macro_place.evaluate system/v1/placer.py --all

# NG45 designs (requires setup)
python -m macro_place.evaluate system/v1/placer.py --ng45
```

---

## 7. How the Evaluation Works

### Interface Contract

Your placer must be a Python file with a class `MacroPlacer` that has exactly this interface:

```python
class MacroPlacer:
    def __init__(self):
        pass  # constructor with no required arguments

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        """
        Place hard macros on the canvas.

        Args:
            benchmark: Contains netlist, initial positions, canvas size

        Returns:
            torch.Tensor of shape [N, 2] with (x, y) center positions
            for each of the N hard macros (in order).

            IMPORTANT: Return all positions (hard + soft macros).
            Typically: start from benchmark.macro_positions and modify
            only the hard macro positions (indices 0..num_hard_macros-1).
        """
```

### What Gets Evaluated

1. **Proxy cost** = WL + 0.5*density + 0.5*congestion (lower is better)
2. **Validity**: no macro-macro overlaps (overlap > 0.4% triggers INVALID)
3. **Runtime**: must complete in under 1 hour for ALL 17 benchmarks combined

### How PlacementCost Works Internally

The `external/MacroPlacement/CodeElements/Plc_client/plc_client_os.py` evaluator:
1. Places macros at the returned positions on the placement grid
2. Snaps each macro center to the nearest grid intersection
3. Computes HPWL across all nets (hard macro pins + soft macro proxies)
4. Computes per-cell density and congestion
5. Returns the normalized proxy cost

**Important**: The evaluator snaps positions to the placement grid. Your placer can output any position, but the actual evaluated position may differ slightly.

---

## 8. Our Algorithm (sameer_v1)

### High-Level Strategy

```
initial.plc
    ↓
Min-displacement Legalization    (resolve overlaps, keep good initial spread)
    ↓
Return placement
```

### Why Just Legalization?

After extensive testing, legalization alone (avg **1.5062**) beats will_seed (1.5338). Here's what we learned:

1. **initial.plc positions are already excellent**: The initial placements provided by the benchmark have good macro spread and low congestion. Our job is to resolve overlaps without destroying this quality.

2. **SA consistently hurts proxy cost**: Adding SA (Simulated Annealing) reduces wirelength (WL) but increases density and congestion faster. Net result: proxy cost goes up, not down.
   - Example: ibm01 legalize-only = 1.2253, with SA = 1.3002 (SA is 6% worse)
   - Example: ibm07 legalize-only = 1.4950, with SA = 1.7313 (SA is 16% worse!)

3. **Why SA hurts**: The proxy cost weights are: `1×WL + 0.5×density + 0.5×congestion`. When SA moves connected macros closer (reducing WL), it clusters them, increasing density and congestion. The density+congestion penalty outweighs the WL gain because the initial spread is already good.

4. **Root cause**: The initial.plc comes from a prior algorithm that already optimized placement. SA from these good positions tends to over-optimize WL at the cost of routability.

### Key Insight for Future Work

To beat RePlAce (1.4578), we need to reduce density and congestion, NOT wirelength. The WL is already small (0.05-0.08 normalized). The dominant cost is congestion (1.3-2.7 normalized). **The winning algorithm will improve congestion, not WL.**

### Runtime

- Legalization: ~5-15s per benchmark
- **Total for all 17 benchmarks: ~3 minutes** (well within 1-hour limit)

---

## 9. Team Development Workflow

### Fork and Setup (One-time)

```bash
# Fork the repo on GitHub under sameersul account
# Then clone your fork:
git clone https://github.com/sameersul/macro-place-challenge-2026.git
cd macro-place-challenge-2026
git remote add upstream https://github.com/partcleda/macro-place-challenge-2026.git
git submodule update --init external/MacroPlacement
uv sync
```

### Branch Strategy

```bash
# Main development branch
git checkout -b dev/sameer_v2

# Each team member works on a branch
git checkout -b feature/gradient-descent-phase3  # member A
git checkout -b feature/gnn-initial-placement    # member B
git checkout -b feature/proxy-aware-sa           # member C
```

### Testing Changes

```bash
# Quick test: single benchmark
python -m macro_place.evaluate system/v1/placer.py -b ibm01

# Full test: all 17 benchmarks (takes ~8 min)
python -m macro_place.evaluate system/v1/placer.py --all

# Compare two placers head-to-head
python scripts/compare_placers.py \
    system/v1/placer.py \
    system/v0/greedy_row_placer.py
```

### Key Metrics to Track

For each benchmark, track:
- `proxy` = final score (MINIMIZE)
- `wl` = wirelength component
- `den` = density component
- `cong` = congestion component
- `runtime` = seconds per benchmark

Target: avg proxy < 1.5338 (will_seed) → competitive; < 1.4578 (RePlAce) → strong entry

### Understanding the Numbers

For ibm01:
```
proxy=1.2253  (wl=0.075 den=0.953 cong=1.347)  VALID  [7.9s]
              ↑                                  ↑
              Breaking down: 0.075 + 0.5×0.953 + 0.5×1.347
                           = 0.075 + 0.477 + 0.674 = 1.226 ✓
```

Congestion dominates proxy for most benchmarks. Reducing congestion has 2x the impact of reducing wirelength (after normalization).

---

## 10. How to Submit

### Step 1: Prepare Your Submission

Your submission must be a single Python file at `system/v1/placer.py` (or your own folder) with a `MacroPlacer` class. It must:
- Work with the exact `place(benchmark) -> torch.Tensor` interface
- Complete all 17 IBM benchmarks in under 1 hour total
- Produce no overlaps (the evaluator checks this)
- Be open-source under Apache 2.0 or GPL (for winning submissions)

### Step 2: Test Thoroughly

```bash
# Run all 17 IBM benchmarks
python -m macro_place.evaluate system/v1/placer.py --all

# Check average proxy score in the final output line
# Format: AVG our_score sa_score replace_score
```

### Step 3: Push to GitHub

```bash
git add system/v1/placer.py
git commit -m "Add sameer_v1 competitive macro placer"
git push origin main
```

The repository must be accessible to judges (public or shared with judges).

### Step 4: Submit via Google Form

Submit at: **https://forms.gle/YDRtYV5Vq68SZgKW9**

Include:
- Your name: Sameer Suleman
- GitHub: sameersul
- Repository URL
- Team size and members
- Brief description of your algorithm

---

## 11. Team Roles

Three roles split by expertise. Each person has a clear onboarding path, owns a piece of the codebase, and has concrete next steps tied to the competition target (beat RePlAce avg 1.4578).

---

### Role A: Algorithm Lead (Sameer)

**Owns:** `system/v1/placer.py` · competition timeline · final submission

**Why this role:** You built the project from scratch, understand every file, and have already beaten will_seed. You drive the main submission loop.

#### Week 1 Onboarding Checklist
- [ ] Re-read `system/v1/placer.py` top-to-bottom, especially `_will_legalize()` and the restart loop
- [ ] Run `python -m macro_place.evaluate system/v1/placer.py --all` and record the full 17-benchmark avg
- [ ] Read `macro_place/objective.py` (`compute_proxy_cost`) to understand exactly how WL, density, and congestion are computed from positions
- [ ] Read `macro_place/evaluate.py` (`evaluate_benchmark`) for the full harness pipeline
- [ ] Read `external/MacroPlacement/CodeElements/Plc_client/plc_client_os.py`, which is the PlacementCost evaluator used by `compute_proxy_cost`

#### Immediate Next Steps (see §13 for full list)
1. **Run the full 17-benchmark eval** with the new restart placer → get baseline avg
2. **Tune noise levels**: try finer noise grid (0.01, 0.02, 0.03, 0.04, 0.05). ibm01 improved at 4%, and 2-3% might be even better for other benchmarks.
3. **Increase restarts for fast benchmarks**: benchmarks that legalize in under 5s can afford 10-15 restarts within the same runtime budget. Add per-benchmark adaptive restart count.
4. **Implement congestion-aware perturbation**: instead of uniform Gaussian noise, push macros away from high-congestion zones found in the baseline legalization

#### Files You Own
```
system/v1/placer.py     ← main submission
test/diagnostic/test_smoke.py ← smoke-test harness
TEAM_GUIDE.md                       ← this document
```

---

### Role B: ML Research Lead

**Owns:** `system/ml_placer/` (create this) · literature review · learning-based approach

**Why this role:** The gap from our current avg (≈1.49) to RePlAce (1.4578) is small with perturbation heuristics alone. Closing it and going below likely needs a learned model. This role brings machine learning to the placement problem.

#### Week 1 Onboarding Checklist
- [ ] Read §12 (Related Research) in this document to understand all 10 papers at a high level
- [ ] Run the existing placer on ibm01: `python -m macro_place.evaluate system/v1/placer.py -b ibm01`
- [ ] Read `macro_place/benchmark.py` to understand the `Benchmark` dataclass fields
- [ ] Understand the netlist format: open `external/MacroPlacement/Testcases/ICCAD04/ibm01/netlist.pb.txt` in a text editor and look at the node, net, and pin structure
- [ ] Clone and browse **WireMask-BBO** (`github.com/lamda-bbo/WireMask-BBO`), the most practically competitive ML method

#### Immediate Next Steps
1. **Reproduce WireMask-BBO on ibm01**: WireMask uses a greedy wire-density heuristic as the evaluator inside a black-box optimizer. Try applying their evaluator to our benchmarks, as it may directly give better placements.
2. **Build a GNN feature extractor**: represent the ibm01 netlist as a graph (macros = nodes, nets = edges weighted by `1/(k-1)`). Use PyTorch Geometric (`torch_geometric`) to build a simple GCN. Output per-node embeddings. These embeddings are the foundation of any learned placement policy
3. **Read ChiPFormer** (`arxiv.org/abs/2306.14744`) carefully. Their offline dataset approach doesn't require per-circuit RL training. We already have a dataset: 17 benchmarks × 5 restarts × proxy scores, which is a small supervised dataset for imitation learning.
4. **Explore Circuit Training's gridding** (`github.com/google-research/circuit_training`). This is the soft-macro grouping step that converts millions of standard cells into around 500 clusters. Understand whether this preprocessing improves our netlist representation.

#### Files You Will Create
```
system/ml_placer/
    __init__.py
    placer.py          ← MacroPlacer class using a learned model
    gnn_model.py       ← GNN architecture
    train.py           ← training script
    data/              ← training data (placements + proxy scores)
```

---

### Role C: Infrastructure and Experiments Lead

**Owns:** `scripts/` · experiment tracking · DREAMPlace integration · evaluation pipeline

**Why this role:** As the team runs dozens of experiments (tuning noise, trying new algorithms, comparing methods), someone needs to make sure results are logged, reproducible, and comparable. This role also investigates the GPU path (DREAMPlace) which could unlock 30× speedup.

#### Week 1 Onboarding Checklist
- [ ] Run `python scripts/compare_placers.py system/v1/placer.py system/v0/greedy_row_placer.py` and understand the comparison output format
- [ ] Read `scripts/compare_placers.py` end-to-end
- [ ] Read `macro_place/evaluate.py` `main()` function to understand all CLI flags
- [ ] Set up a results log: create `results/runs.csv` with columns: `date, placer, benchmark, proxy, wl, density, congestion, runtime, notes`
- [ ] Browse **DREAMPlace** (`github.com/limbo018/DREAMPlace`) to understand its input format and how it compares to our benchmark format

#### Immediate Next Steps
1. **Automated experiment logging**: modify `scripts/compare_placers.py` to append results to `results/runs.csv` automatically. Every run of any placer should log to this file with a timestamp and git commit hash
2. **DREAMPlace compatibility check**: DREAMPlace takes LEF/DEF or Bookshelf format. Our benchmarks are in `.pb.txt` format. Write a converter: `scripts/pb_to_bookshelf.py` that converts ibm01 netlist to Bookshelf format so DREAMPlace can process it. This unlocks GPU-accelerated global placement as an initial position generator
3. **Perturbation noise sweep**: write `scripts/noise_sweep.py` that runs `sameer_v1` with noise_fracs=`[0.01, 0.02, ..., 0.10]` on ibm01 and plots proxy cost vs noise level. Identify the optimal noise range per benchmark
4. **Per-benchmark adaptive restarts**: ibm10 takes 60s to legalize (too slow for 15 restarts). ibm07 takes 2s (fast, can afford 20+). Write a time-budget wrapper that decides how many restarts to run per benchmark given a per-benchmark time cap of `3600/17 ≈ 210s`

#### Files You Own
```
scripts/compare_placers.py     ← comparison harness (already exists)
scripts/noise_sweep.py         ← create this
scripts/pb_to_bookshelf.py     ← create this (DREAMPlace bridge)
results/runs.csv               ← create this (experiment log)
```

---

### Shared Responsibilities (All Three)

| Task | Who Leads | Deadline |
|------|-----------|----------|
| Beat will_seed avg (1.5338) | Sameer (A) | Done ✅ |
| Full 17-benchmark avg with restarts | Sameer (A) | This week |
| Literature deep-dive on WireMask-BBO | ML Lead (B) | Week 1 |
| Noise sweep script | Infra (C) | Week 1 |
| DREAMPlace format bridge | Infra (C) | Week 2 |
| GNN feature extractor prototype | ML Lead (B) | Week 2 |
| Beat RePlAce avg (1.4578) | All | Week 3 |
| Final submission via Google Form | Sameer (A) | May 21, 2026 |

---

## 12. Related Research: Papers and Code

This field exploded after Google's 2021 Nature paper. Every significant paper below includes its core idea, key result, and GitHub link, organized by approach type. **Bolded rows** are highest priority to read and implement first.

---

### The Essential Starting Point: TILOS MacroPlacement

**GitHub:** https://github.com/TILOS-AI-Institute/MacroPlacement

Before reading any other paper, understand this. The TILOS group (UCSD) published a landmark reproducibility study showing Google's Circuit Training results **could not be reproduced** from its open-source release, and that standard SA baselines match or beat CT on most benchmarks. They also:
- Released the IBM ICCAD04 and Ariane/MemPool/NVDLA benchmarks with proper enablements (the same benchmark set our competition uses)
- Implemented all of CT's missing components (soft-macro clustering, gridding)
- Provide a full open-source evaluation flow

---

### Approach 1: Reinforcement Learning

#### Google Circuit Training (Nature 2021)
- **Paper:** "A graph placement methodology for fast chip design" by Mirhoseini et al. (Google Brain)
- **GitHub:** https://github.com/google-research/circuit_training
- **Core idea:** GNN policy places macros one-at-a-time on a grid canvas, trained with PPO. A preprocessing step clusters standard cells with macros using force-directed "gridding." Used in production for Google TPU-v5.
- **Key result:** Claims to match human expert placements; TILOS showed SA is competitive with far less compute.
- **Reusable for us:** Netlist→graph representation, soft-macro clustering code, proxy cost formulation (same as ours).

#### MaskPlace (NeurIPS 2022)
- **Paper:** "MaskPlace: Fast Chip Placement via Reinforced Visual Representation Learning" by Lai et al.
- **arXiv:** https://arxiv.org/abs/2211.13382 · **GitHub:** https://github.com/laiyao1/maskplace
- **Core idea:** Canvas as a pixel mask rather than a grid. Dense rewards every step (vs CT's sparse end-of-episode reward), giving faster and more stable training.
- **Key result:** 60-90% HPWL reduction over CT baselines on Ariane.
- **Reusable:** Clean codebase; pixel-canvas state representation and dense reward formulation.

#### **ChiPFormer (ICML 2023) - Most Relevant for ML Role**
- **Paper:** "ChiPFormer: Transferable Chip Placement via Offline Decision Transformer" by Lai et al.
- **arXiv:** https://arxiv.org/abs/2306.14744 · **GitHub:** https://github.com/laiyao1/chipformer
- **Core idea:** Trains once on a dataset of expert placements (500 per circuit), then transfers to new circuits via few-shot fine-tuning. Uses a Decision Transformer instead of per-circuit RL, making placement take minutes instead of hours.
- **Key result:** Outperforms CT and MaskPlace on HPWL across 12 circuits. Includes a released dataset of 12 circuits × 500 expert placements.
- **Reusable for us:** We can generate our own dataset: run N restarts per benchmark, record (netlist features, placement, proxy score) tuples, train a supervised model. Skips RL entirely.

#### MaskRegulate (NeurIPS 2024)
- **Paper:** "RL Policy as Macro Regulator Rather than Macro Placer" by the LAMDA Group (Nanjing U.)
- **GitHub:** https://github.com/lamda-bbo/macro-regulator
- **Core idea:** RL iteratively adjusts an existing placement (doesn't place from scratch). "Regularity" metric used as both input feature and reward signal.
- **Key result:** -17% routing wirelength and -73% congestion overflow vs MaskPlace.
- **Reusable:** Start from our legalized placement and use RL to improve it, which is much simpler than full RL placement from scratch.

---

### Approach 2: Black-Box Optimization

#### **WireMask-BBO (NeurIPS 2023) - Most Actionable Right Now**
- **Paper:** "Macro Placement by Wire-Mask-Guided Black-Box Optimization" by the LAMDA Group
- **arXiv:** https://arxiv.org/abs/2306.16844 · **GitHub:** https://github.com/lamda-bbo/WireMask-BBO
- **Core idea:** A wire-density heatmap guides a greedy evaluator that rapidly scores any candidate placement. Any black-box optimizer (Bayesian, evolutionary, random) wraps this evaluator. No ML training required.
- **Key result:** Up to 50% HPWL improvement over CT using far less compute.
- **Reusable:** Instead of random Gaussian perturbations, use the wire-density heatmap to guide where to perturb macros, pushing them toward low-density zones. This is a direct upgrade to our current restart strategy.

---

### Approach 3: Analytical Placement (GPU)

#### RePlAce (IEEE TCAD 2019)
- **Paper:** "RePlAce: Advancing Solution Quality and Routability Validation in Global Placement"
- **GitHub:** https://github.com/The-OpenROAD-Project/RePlAce (also `gpl` inside OpenROAD)
- **Core idea:** Nonlinear analytical placement using Nesterov's method. Minimizes WL + electrostatic density penalty. The standard global placement engine in OpenROAD.
- **Key result:** ~10-15% routability improvement over ePlace. **This is the competition's 1.4578 baseline.**
- **Reusable:** Understanding its density model is essential because this is what we need to beat.

#### **DREAMPlace (DAC 2019 / TCAD 2020 / v4.0) - GPU Infrastructure**
- **Paper:** "DREAMPlace: Deep Learning Toolkit-Enabled GPU Acceleration for Modern VLSI Placement" by Lin et al. (UT Austin / NVIDIA)
- **GitHub:** https://github.com/limbo018/DREAMPlace
- **Core idea:** Treats analytical placement as neural network training, with WL and density as the loss function and PyTorch autograd computing gradients. Custom CUDA kernels handle HPWL and electrostatic density.
- **Key result:** 30× speedup over CPU RePlAce on a V100 GPU.
- **Reusable:** If we bridge our `.pb.txt` format to DREAMPlace's Bookshelf input, we get GPU-accelerated global placement positions that are far better than `initial.plc` as starting points.

#### GiFt (ICCAD 2024)
- **Paper:** "The Power of Graph Signal Processing for Chip Placement Acceleration"
- **arXiv:** https://arxiv.org/abs/2502.17632
- **Core idea:** Graph spectral analysis of the netlist gives initial placement coordinates in <1s on GPU. Used as DREAMPlace warm-start: -33% iterations, -46% total runtime.
- **Reusable:** If we integrate DREAMPlace, use GiFt as the initialization instead of random.

---

### Approach 4: Generative Models

#### Chip Placement with Diffusion Models (ICML 2025)
- **Paper:** "Chip Placement with Diffusion Models" by Lee, Nguyen, Elzeiny et al. (UC Berkeley)
- **arXiv:** https://arxiv.org/abs/2407.12282 · **GitHub:** https://github.com/vint-1/chipdiffusion
- **Core idea:** A diffusion model conditioned on the circuit netlist places ALL macros simultaneously rather than one at a time like RL. Guided sampling at inference optimizes quality. Pre-trains on large synthetic datasets for zero-shot generalization to unseen circuits.
- **Key result:** Competitive with RL baselines on HPWL and congestion, with zero per-circuit training.
- **Reusable:** Synthetic dataset generation algorithm; the most architecturally novel approach.

---

### Full Summary Table

| Paper | Year | Venue | Type | GitHub | Key Advantage |
|-------|------|-------|------|--------|---------------|
| TILOS Benchmarks | 2022+ | Open | Eval framework | [link](https://github.com/TILOS-AI-Institute/MacroPlacement) | Same benchmarks; reproducible baselines |
| Circuit Training | 2021 | Nature | RL (GNN) | [link](https://github.com/google-research/circuit_training) | Production-proven; controversial reproducibility |
| MaskPlace | 2022 | NeurIPS | RL (pixel) | [link](https://github.com/laiyao1/maskplace) | Dense rewards; clean code |
| **WireMask-BBO** | **2023** | **NeurIPS** | **BBO** | **[link](https://github.com/lamda-bbo/WireMask-BBO)** | **No training; wire-density guided perturbation** |
| **ChiPFormer** | **2023** | **ICML** | **Offline RL** | **[link](https://github.com/laiyao1/chipformer)** | **Minutes not hours; transferable across chips** |
| MaskRegulate | 2024 | NeurIPS | RL (regulator) | [link](https://github.com/lamda-bbo/macro-regulator) | Adjusts existing placement; -73% congestion |
| Chip+Diffusion | 2025 | ICML | Diffusion | [link](https://github.com/vint-1/chipdiffusion) | Zero-shot; parallel placement |
| **DREAMPlace** | **2019+** | **DAC/TCAD** | **Analytical (GPU)** | **[link](https://github.com/limbo018/DREAMPlace)** | **30× speedup; extensible PyTorch framework** |
| RePlAce | 2019 | TCAD | Analytical | [link](https://github.com/The-OpenROAD-Project/RePlAce) | The 1.4578 baseline to beat |

---

## 13. Ideas for Improvement

### Understanding the Numbers First

Before trying to improve, understand what's actually being optimized:

```
proxy = 1.0×WL + 0.5×density + 0.5×congestion

Current avg (sameer_v1): 1.5062
  WL component: ~0.06  (already very small, not the problem)
  Density:      ~0.90  (moderate)
  Congestion:   ~2.0   (THIS is what's killing us, it's 20-30x larger than WL)
```

**The winning algorithm reduces congestion, not wirelength.** This completely changes the approach.

### Low Effort, High Impact

1. **Perturbation + Re-legalization**: Randomly perturb the initial.plc positions (add noise to macro positions), re-legalize, evaluate, keep best. Run 5-10 random restarts per benchmark. If the noise happens to space macros more evenly, congestion drops.
   ```python
   best = legalize(initial_pos)
   for _ in range(10):
       noise = initial_pos + random.gauss(0, 0.5)  # small perturbation
       candidate = legalize(noise)
       if proxy(candidate) < proxy(best): best = candidate
   ```

2. **Spread macros from congested zones**: After legalization, identify the most congested grid cells. Move macros OUT of those cells (push them to the boundary or low-density areas). This directly reduces the congestion term.

3. **Better initial positions**: Instead of using `initial.plc`, try placing macros on a regular grid (evenly spaced), then legalizing. Maximizes spread, minimizes congestion.

### Medium Effort

4. **Congestion-Map SA**: Run SA where the objective is `congestion_estimate` rather than WL. Move macros away from congested regions. The congestion can be estimated with a simple routing demand model.

5. **Force-Directed with Repulsion**: Add repulsive forces between nearby macros (not just spring attractions to connected ones). This spreads macros evenly, reducing congestion.

6. **Gradient Descent on Congestion (GPU)**: Model congestion as differentiable using a soft kernel to estimate routing demand per cell, then minimize via gradient descent. Much more effective than WL gradient descent.

### High Effort, Potentially Huge Impact

7. **Graph Neural Network**: Train a GNN on the netlist to predict CONGESTION-OPTIMAL macro positions. Key: train it to minimize proxy cost (including congestion), not just WL. Features: macro connectivity, size, canvas aspect ratio, benchmark statistics.

8. **Reinforcement Learning**: Like Google's Circuit Training, train an RL agent to sequentially place macros using proxy cost as the reward signal. Requires significant training time.

9. **Learned Legalization**: Instead of min-displacement legalization, train a model to predict WHERE to push macros during legalization to minimize congestion.

### What NOT to Try

- WL-only optimization (we tested this exhaustively and it hurts proxy)
- Hardcoding positions for specific benchmarks (against rules)
- Running more than 1 hour total runtime (hardware limit)
- Using external proprietary tools (against rules)

---

## 14. Glossary

| Term | Definition |
|------|------------|
| **Hard macro** | Large fixed-size block (SRAM, IP, analog) that cannot be split or resized |
| **Soft macro** | Standard cell cluster treated as a movable rectangle for placement purposes |
| **Netlist** | Graph of connections between circuit elements |
| **Hyperedge / net** | A net connecting 2+ macros (the netlist file uses hyperedges) |
| **Clique expansion** | Converting a k-way hyperedge to k(k-1)/2 pairwise edges; enables vectorized WL computation |
| **HPWL** | Half-Perimeter Wirelength: bounding box perimeter of a net. Standard WL approximation |
| **Placement grid** | Regular grid of rows × columns on the canvas; macros snap to grid intersections |
| **Legalization** | Resolving macro-macro overlaps while minimizing displacement from current positions |
| **Proxy cost** | Weighted sum of WL + density + congestion; a cheap estimate of final routed quality |
| **SA** | Simulated Annealing: probabilistic optimization that accepts worse solutions sometimes |
| **WNS** | Worst Negative Slack: the most negative timing violation (Tier 2 evaluation) |
| **TNS** | Total Negative Slack: sum of all negative slack values (Tier 2 evaluation) |
| **NG45** | NanGate 45nm process design kit, used for Tier 2 evaluation |
| **PB text format** | Protocol Buffer text format, which is how netlists are stored (`.pb.txt` files) |
| **PLC file** | Placement cost file that stores initial macro positions and grid configuration |
| **initial.plc** | Starting placement file; generated by a prior EDA flow |
| **Anchor constraint** | SA constraint: macro must stay within k × half-size of its starting position |
| **OpenROAD** | Open-source EDA suite for complete physical design flow |
| **EDA** | Electronic Design Automation: software tools for chip design |
