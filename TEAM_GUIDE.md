# Partcl/HRT Macro Placement Challenge: Team Guide

> A complete reference for understanding, developing, and contributing to this project.

> **Current system note (2026-06-16):** this guide started as an early
> onboarding document for the April `sameer_v1` proxy experiments. The active
> submission now lives at the repository root (`src/main.py`, `src/placer/`,
> `src/dreamplace_bridge/`, `src/eda_io/`) and is hierarchy-only. It requires
> grouped DREAMPlace, preserves connected subsystems, and currently reports
> `uv run evaluate src/main.py --all` = **AVG 1.4452**, 17/17 VALID, 0 overlaps,
> 520.08s. Older `sameer_v1`, `system/v0`, restart, ML-ranker, R2, and generic
> LSMC references below are historical unless explicitly marked as current.

---

## Table of Contents

1. [What Is This Problem? (Simple Terms)](#1-what-is-this-problem)
2. [The Full Picture: Chip Design Flow](#2-the-full-chip-design-flow)
3. [The Proxy Cost Metric (What We're Optimizing)](#3-the-proxy-cost-metric)
4. [The Competition Setup](#4-the-competition-setup)
5. [Software Installation](#5-software-installation)
6. [Repository Structure](#6-repository-structure)
7. [How the Evaluation Works](#7-how-the-evaluation-works)
8. [Our Current Algorithm](#8-our-current-algorithm)
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

### Current Competition Standing

As of **June 16, 2026**, the selected system is the hierarchy-preserving
placer in `src/main.py`, not the April `sameer_v1` legalizer. The current IBM
Tier 1 reference is:

```text
uv run evaluate src/main.py --all
AVG 1.4452  17/17 VALID  0 overlaps  520.08s
```

This is slightly better than RePlAce's 1.4578 proxy average, but the system's
chosen objective is hierarchy preservation. It intentionally avoids returning
to the deleted spread-oriented proxy optimizer, whose historical best proxy
average was lower but no longer reflects the active submission.

Historical April snapshot:

| Benchmark | sameer_v1 | will_seed | RePlAce |
|-----------|-----------|-----------|---------|
| ibm01 | **1.2253** | ~1.29 | 0.998 |
| ibm02 | 1.6800 | ~1.68 | 1.837 |
| ibm03 | **1.4100** | ~1.74 | 1.322 |
| ibm07 | **1.4950** | ~2.02 | 1.463 |
| ibm09 | **1.1363** | ~1.39 | 1.119 |
| ibm17 | **1.7437** | ~3.67 | 1.645 |
| **AVG** | **1.5062** | **1.5338** | **1.4578** |

The April `sameer_v1` submission beat `will_seed` by ~2.9% but trailed RePlAce.
That state is no longer the active system.

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
uv run evaluate src/main.py -b ibm10
```

If `uv` is not available, use standard Python:
```bash
pip install -e .
python -m macro_place.evaluate src/main.py -b ibm10
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
├── src/                          # Active hierarchy placer and EDA I/O
│   ├── main.py                   # Evaluator-facing entrypoint
│   ├── placer/                   # Hierarchy pipeline, scoring, routing, legalization
│   ├── dreamplace_bridge/        # ICCAD04 pb/plc <-> Bookshelf bridge
│   └── eda_io/                   # LEF/DEF/Verilog/SDC/Liberty import/export
│
├── system/                       # Optional historical checkpoints if present
│   └── v1/                       # Frozen checkpoint; read-only if present
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
python -m macro_place.evaluate src/main.py -b ibm10

# All 17 IBM benchmarks
python -m macro_place.evaluate src/main.py --all

# NG45 designs (requires setup)
python -m macro_place.evaluate src/main.py --ng45
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

## 8. Our Current Algorithm

### High-Level Strategy

```
initial.plc
    ↓
Derive hard clusters from low-fanout hard/soft connectivity
    ↓
Classify soft macros as owned or bridge
    ↓
Grouped DREAMPlace with synthetic cluster clique nets
    ↓
Cluster-consecutive hard legalization
    ↓
Soft relocation cleanup
    ↓
Congestion-expanded hard/soft regions
    ↓
Region-locked hard and soft relief
    ↓
Exact-gated cluster decompression
    ↓
Region-bounded hard-hard / hard-soft / soft-soft swaps
    ↓
Proxy-aware coldspot tightening
    ↓
Return placement
```

### Why Hierarchy Preservation?

The current selected system is not the old lowest-proxy optimizer. It keeps
connected subsystems together because that is the desired placement behavior,
even though the proxy metric often rewards spreading connected macros apart.
The exact proxy still gates local relief moves and evaluator reports, but it is
not the primary design objective.

What remains true from the early work:

1. **`initial.plc` is a strong seed.** Random or grid restarts were consistently worse.
2. **Proxy and hierarchy are opposed.** Compact connected groups usually raise congestion.
3. **Soft macros must move with hierarchy.** Owned and bridge soft roles are a live part of the current system.
4. **DREAMPlace is required.** `MacroPlacer.place()` raises if grouped DREAMPlace cannot run.

### Current Reference

```text
uv run evaluate src/main.py -b ibm10
proxy=1.6759  VALID

uv run evaluate src/main.py --all
AVG 1.4452  17/17 VALID  0 overlaps  520.08s
```

See `docs/general/DESIGN_FLOW.md`, `docs/general/ARCHITECTURE.md`, and
`docs/general/OBJECTIVES.md` for the current implementation details and
placement objectives.

### Historical April Insight

The old `sameer_v1` legalize-only system averaged 1.5062 and showed that
congestion dominates the proxy. That insight still explains why hierarchy
needs congestion-aware relief, but the legalize-only algorithm is historical.

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
uv run evaluate src/main.py -b ibm10

# Full test: all 17 benchmarks
uv run evaluate src/main.py --all

# Compare two placers head-to-head
uv run python scripts/compare_placers.py path/to/placer_a.py src/main.py
```

### Key Metrics to Track

For each benchmark, track:
- `proxy` = final score (MINIMIZE)
- `wl` = wirelength component
- `den` = density component
- `cong` = congestion component
- `runtime` = seconds per benchmark

Current hierarchy reference: avg proxy **1.4452** with 17/17 valid placements.
Treat older sub-1.13 proxy-path averages as historical; that code path was
deleted and is not the active capability of this repository.

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

The active submission entrypoint is `src/main.py`, which exposes a
`MacroPlacer` class. It must:
- Work with the exact `place(benchmark) -> torch.Tensor` interface
- Complete all 17 IBM benchmarks in under 1 hour total
- Produce no overlaps (the evaluator checks this)
- Be open-source under Apache 2.0 or GPL (for winning submissions)

### Step 2: Test Thoroughly

```bash
# Run all 17 IBM benchmarks
uv run evaluate src/main.py --all

# Check average proxy score in the final output line
# Format: AVG our_score sa_score replace_score
```

### Step 3: Push to GitHub

```bash
git add src docs test scripts
git commit -m "Update hierarchy placer submission"
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

### Role A: Algorithm Lead

**Owns:** `src/placer/pipeline/macro_placer.py`, `src/placer/local_search/`,
`docs/general/PROGRESS.md`, competition timeline, final submission.

**Why this role:** This role owns the hierarchy objective, legality, benchmark
quality, and the decision to accept or reject system-level changes.

#### Week 1 Onboarding Checklist
- [ ] Re-read `src/placer/pipeline/macro_placer.py` top-to-bottom, especially `_hierarchy_floorplan()`
- [ ] Run `uv run evaluate src/main.py --all` only when a change is ready for full verification
- [ ] Read `macro_place/objective.py` (`compute_proxy_cost`) to understand exactly how WL, density, and congestion are computed from positions
- [ ] Read `macro_place/evaluate.py` (`evaluate_benchmark`) for the full harness pipeline
- [ ] Read `external/MacroPlacement/CodeElements/Plc_client/plc_client_os.py`, which is the PlacementCost evaluator used by `compute_proxy_cost`

#### Immediate Next Steps (see §13 for full list)
1. **Use ibm10 as the hierarchy smoke** before broader sweeps.
2. **Investigate current bottlenecks**: ibm12, ibm15, ibm17 congestion, plus ibm14/ibm18 regressions from the latest soft-swap tuning.
3. **Keep hierarchy metrics visible** alongside proxy when accepting changes.
4. **Update `docs/general/PROGRESS.md`** with concrete numbers for every accepted system result.

#### Files You Own
```
src/main.py                         <- evaluator entrypoint
src/placer/pipeline/macro_placer.py <- main hierarchy pipeline
src/placer/local_search/            <- relief, swaps, decompression, coldspot helper
docs/general/PROGRESS.md            <- accepted results and experiment history
```

---

### Role B: ML Research Lead

**Owns:** archived ML notes, any future hierarchy-specific learning prototype,
and experiment design for learned scoring or placement.

**Why this role:** The old proxy-path ML ranker was deleted. Future ML work must
define a hierarchy-aware target before adding model code.

#### Week 1 Onboarding Checklist
- [ ] Read §12 (Related Research) in this document to understand all 10 papers at a high level
- [ ] Run the existing placer on ibm10: `uv run evaluate src/main.py -b ibm10`
- [ ] Read `macro_place/benchmark.py` to understand the `Benchmark` dataclass fields
- [ ] Understand the netlist format: open `external/MacroPlacement/Testcases/ICCAD04/ibm01/netlist.pb.txt` in a text editor and look at the node, net, and pin structure
- [ ] Clone and browse **WireMask-BBO** (`github.com/lamda-bbo/WireMask-BBO`), the most practically competitive ML method

#### Immediate Next Steps
1. **Define a hierarchy-aware label** before training anything. Proxy-only labels recreate the deleted spread-oriented optimizer.
2. **Prototype on diagnostics first**: use cluster closeness, region congestion, and accepted/rejected exact-gated moves as features.
3. **Keep ML optional** until it has a verified current integration point in the hierarchy path.
4. **Update `docs/ml_nn/`** if a new hierarchy-specific ML direction replaces the archived proxy-ranker notes.

#### Files You Will Create
```
src/placer/ml/         <- create only if a hierarchy-specific integration is accepted
ml_data/               <- generated traces, labels, and model artifacts
docs/ml_nn/            <- design notes; current files are historical
```

---

### Role C: Infrastructure and Experiments Lead

**Owns:** `scripts/`, DREAMPlace install health, experiment tracking, evaluation pipeline.

**Why this role:** The current production path requires DREAMPlace. Reliable
evaluation depends on keeping the bridge, install tree, and run logs healthy.

#### Week 1 Onboarding Checklist
- [ ] Run `uv run python scripts/compare_placers.py path/to/placer_a.py src/main.py` and understand the comparison output format
- [ ] Read `scripts/compare_placers.py` end-to-end
- [ ] Read `macro_place/evaluate.py` `main()` function to understand all CLI flags
- [ ] Set up a results log: create `results/runs.csv` with columns: `date, placer, benchmark, proxy, wl, density, congestion, runtime, notes`
- [ ] Browse **DREAMPlace** (`github.com/limbo018/DREAMPlace`) to understand its input format and how it compares to our benchmark format

#### Immediate Next Steps
1. **Automated experiment logging**: modify `scripts/compare_placers.py` to append results to `results/runs.csv` automatically. Every run of any placer should log to this file with a timestamp and git commit hash
2. **DREAMPlace health checks**: after rebuilds, run `scripts/patch_dreamplace_install.py` and verify grouped DREAMPlace produces ready results.
3. **Targeted sweeps**: use existing diagnostics such as `test/diagnostic/_sweep_region_swaps.py` for current hierarchy operators.
4. **Runtime tracking**: keep the all-17 run under the 1-hour harness limit; record contention-sensitive timings in `docs/general/ISSUES.md` or `PROGRESS.md`.

#### Files You Own
```
scripts/compare_placers.py     ← comparison harness (already exists)
src/dreamplace_bridge/         ← active DREAMPlace bridge
results/runs.csv               ← create this (experiment log)
```

---

### Shared Responsibilities (All Three)

| Task | Who Leads | Deadline |
|------|-----------|----------|
| Maintain current hierarchy `--all` reference | Algorithm Lead | Active |
| Keep DREAMPlace bridge healthy | Infrastructure Lead | Active |
| Investigate ibm12/15/17 congestion bottlenecks | Algorithm Lead | Active |
| Define hierarchy-aware ML objective before model work | ML Lead | Open |
| Keep documentation current after accepted changes | All | Active |
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

This section keeps early proxy-optimization ideas for context, but do not treat
them as the current implementation plan. The active code is hierarchy-only and
the proxy-only restart, R2, ML-ranker, and generic LSMC paths were deleted.

### Understanding the Numbers First

Before trying to improve, understand what's actually being optimized:

```
proxy = 1.0×WL + 0.5×density + 0.5×congestion

Current hierarchy avg: 1.4452
  WL component: ~0.06  (already very small, not the problem)
  Density:      ~0.90  (moderate)
  Congestion:   ~2.0   (THIS is what's killing us, it's 20-30x larger than WL)
```

For the selected system, "better" means preserving hierarchy while relieving
enough congestion to stay valid and competitive. Pure proxy spread is not the
current target.

### Current Directions

1. **Congestion relief inside hierarchy regions**: improve region expansion, decompression, and soft-heavy swap choices for ibm12/15/17.
2. **Soft macro placement around hard hierarchy**: keep owned/bridge behavior accurate and avoid soft moves that undo subsystem structure.
3. **Hierarchy-quality metrics**: make accept gates sensitive to both proxy and structural closeness.
4. **DREAMPlace bridge robustness**: grouped DREAMPlace is required, so install health and cache correctness are production concerns.
5. **Hierarchy-specific learning**: only add ML after defining labels that reward the selected structural objective, not just lower proxy.

### Historical Ideas

### What NOT to Try

- WL-only optimization (tested exhaustively and hurts proxy/hierarchy tradeoffs)
- Reintroducing deleted proxy-only restarts, R2, generic LSMC, or ML ranker code by default
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
