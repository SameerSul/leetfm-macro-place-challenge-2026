# v2 Design Flow

This document describes the production flow implemented by
`src/placer/pipeline/macro_placer.py`.

## TLDR

`MacroPlacer.place()` is a budget-gated, accept-only search over macro
placements. It legalizes the benchmark's initial hard macro positions, launches
three standard DREAMPlace subprocesses, plus an optional grouped-DP subprocess
when `V2_DP_GROUP` is enabled, as asynchronous candidate generators when the
bridge is available. It then explores generic random/local restart candidates,
runs a deep R2 local-search finisher, and uses LSMC as the final global
exploration layer.

Every candidate that can affect the incumbent is judged by the exact proxy:

```text
proxy = wirelength + 0.5 * density + 0.5 * congestion
```

`best_pl` and `best_score` are the main incumbent state. They update only when
an exact proxy score is lower.

## Flow

```mermaid
flowchart TD
    A[Benchmark input] --> B[Read benchmark data]
    B --> B1[Hard macro indices and movable mask]
    B --> B2[Soft macro indices and movable mask]
    B --> B3[Nets, pins, macro sizes, canvas, grid]
    B --> C[Load PLC for exact proxy scoring]
    B --> D[Read hard positions from initial benchmark placement]

    C --> DP0[Launch DREAMPlace x3 async if available]
    DP0 --> DP1[lo-fix: target density 0.65, soft fixed]
    DP0 --> DP2[hi-mov: target density 0.85, soft movable]
    DP0 --> DP3[hi-fix: target density 0.85, soft fixed]
    DP0 -. V2_DP_GROUP .-> DP4[optional grouped DP seed with synthetic cluster nets]

    D --> P0[Baseline hard legalization]
    P0 --> GUARD{Exact scoring path usable?}
    GUARD -->|No| LB2[Optional baseline-only hard 2-opt]
    LB2 --> LB3[Optionally compare first DP candidate on large-DP path]
    LB3 --> LBR[Return baseline or first DP if it wins]

    GUARD -->|Yes| S0[Score baseline with exact proxy]
    S0 --> BEST[best_pl and best_score]
    S0 --> LSEED0[Remember LSMC baseline seed]

    DP1 --> DPH[Harvest completed DP candidates]
    DP2 --> DPH
    DP3 --> DPH
    DP4 --> DPH
    DPH --> DPL[Clip and legalize DP hard macros]
    DPL --> DPS[Copy DP soft positions, then exact proxy score]
    DPS --> DPK{Improved best?}
    DPK -->|Yes| BEST
    DPK -->|No| DPDROP[Do not update best]

    BEST --> N[Random Gaussian restarts from initial positions]
    N --> NL[Legalize each noisy hard placement]
    NL --> NS[Exact proxy score]
    NS --> NK{Improved best?}
    NK -->|Yes| BEST
    NK -->|No| NDROP[Discard candidate]
    NS --> LSEED1[Remember near-best random LSMC seeds]

    BEST --> P9[Random-order legalization]
    P9 --> P9L[Three legalize trials from initial positions with randomized tie-breaks]
    P9L --> P9S[Sequential exact proxy scoring]
    P9S --> BEST
    P9S --> LSEED2[Remember near-best P9 LSMC seeds]

    BEST --> LSEED3[Remember pre-R2 best LSMC seed]
    LSEED3 --> R2[R2 refinement loop, up to 20 budget-gated rounds]
    R2 --> R2S[Build or reuse round IncrementalScorer]
    R2S --> HR1[Hard relocation by congestion field]
    HR1 --> HML[Optional ML filter: wide-32 hard-relocation pool, rank top 16]
    HML --> HCUDA[Optional CUDA propose-all ranking when V2_RELOC_PROPOSE_ALL enables it]
    HCUDA --> HR2[Hard relocation by density field if not skipped]
    HR2 --> HR3[Hard relocation by combined cong-density field if not skipped]
    HR3 --> SR1[Soft relocation by congestion field, WL-prefiltered, if not skipped]
    SR1 --> SR2[Soft relocation by density field, WL-prefiltered]
    SR2 --> SS1[Soft-soft 2-opt by congestion field, WL-prefiltered, if not skipped]
    SS1 --> SS2[Soft-soft 2-opt by density field, WL-prefiltered, if not skipped]
    SS2 --> HXS1[Hard-soft cross-swap by congestion field if not skipped]
    HXS1 --> HXS2[Hard-soft cross-swap by density field if not skipped]
    HXS2 --> HS31[Hard-soft-soft 3-cycle by congestion field if not skipped]
    HS31 --> HS32[Hard-soft-soft 3-cycle by density field if not skipped]
    HS32 --> R2O[Hard 2-opt cleanup, k=16 plus cold-region teleport]
    R2O --> R2V[Exact proxy verify after passes with accepts]
    R2V --> R2Q{Round improved enough and budget remains?}
    R2Q -->|Yes| R2
    R2Q -->|No| POST[Post-R2 leftover soft relocation]

    POST --> POST1[Soft relocation by congestion then density, top_hot 1024, n_targets 4]
    POST1 --> POST2[Exact proxy verify]
    POST2 --> LSMC0[Remember post-R2 best LSMC seed]
    LSMC0 --> LSMC1[Select generic LSMC seeds by exact score]
    LSMC1 --> LSMC2[For each seed: cluster/default or random hard kick, legalize, descend]
    LSMC2 --> LSMC3[Exact proxy accept only if global best improves]
    LSMC3 --> CLAMP[Final movable-macro in-bounds clamp]
    CLAMP --> RET[Return best_pl]
```

## Active Candidate Sources

- **Baseline:** legalize hard macros from `initial.plc`.
- **DREAMPlace:** when available, three async DP variants are scored as ordinary
  candidates. They may update `best_pl`, but they are not retained as special
  LSMC seed basins.
- **Grouped DREAMPlace:** optional and env-gated by `V2_DP_GROUP`. It adds
  synthetic cluster clique nets to the Bookshelf conversion and scores the
  resulting DP placement as another ordinary candidate.
- **Random restarts:** Gaussian perturbations of initial hard positions,
  legalized and exact-scored.
- **Random-order legalization:** three alternate legalizer tie-break orders from
  the initial placement.
- **R2:** the main local-search finisher.
- **Post-R2 soft relocation:** short leftover-budget soft cleanup.
- **LSMC:** final generic multi-incumbent exploration over baseline/random/P9/
  pre-R2/post-R2 seeds.

## Important Details

- Hard and soft macro labels come from the benchmark API. The placer does not
  infer them.
- Congestion-gradient phases are no longer part of the active flow. Historical
  notes may mention phases 1/2/3/5b/5c/7/8; those labels refer to retired
  experiments.
- Non-DREAMPlace candidates leave soft macro positions as they are in the source
  placement until R2/post-R2/LSMC soft moves operate on them. DREAMPlace
  candidates can carry DREAMPlace-produced soft positions.
- Random-noise and random-order phases restart from the initial hard positions,
  not from `best_pl`.
- LSMC seed collection is intentionally generic. It does not use
  DREAMPlace/bridge-specific seed pools and does not use cong-grad-derived
  state.
- R2 is richer than "relocation plus 2-opt": each round can run hard relocation,
  soft relocation, soft-soft swaps, hard-soft swaps, hard-soft-soft 3-cycles, and
  hard 2-opt cleanup.
- By default, `src/main.py` enables the shipped hard-relocation ML filter when no
  `ML_*` env var is preset and the model artifact plus `xgboost` are available.
  It widens the hard-relocation pool to 32 and exact-scores the ranked top 16;
  any preset `ML_*` var, missing model, or missing `xgboost` falls back to the
  pure-heuristic path.
- `V2_RELOC_PROPOSE_ALL=auto` can enable the CUDA propose-all hard-relocation
  ranking path only when the runtime backend is CUDA. It is an opt-in search
  variant; exact incremental scoring still gates every committed move.
- `V2_GPU_EXPLORE=auto` enables the final LSMC layer when CUDA is visible.
  `V2_GPU_EXPLORE_MULTI_INCUMBENT`, `V2_GPU_EXPLORE_MAX_SEEDS`, and
  `V2_GPU_EXPLORE_SEED_MARGIN` control the generic seed pool.
- `src/main.py` enables cluster-coherent LSMC kicks by default unless the caller
  already set `V2_GPU_EXPLORE_CLUSTER_P`. The default is
  `V2_GPU_EXPLORE_CLUSTER_P=1.0` and `V2_GPU_EXPLORE_CLUSTER_MODE=both`, with
  random-kick fallback when no usable cluster exists. Direct imports of
  `placer.pipeline.macro_placer` do not apply this wrapper default.
- Every return path passes through a final in-bounds clamp for movable macros.
  Hard macros are already legalized by the normal path; the clamp mainly protects
  against soft macro coordinates inherited from input data or DREAMPlace output.

## Entry points

The flow above starts from a `Benchmark`. There are two ways to produce one:

- **Challenge path** — the harness calls `load_benchmark` on an ICCAD04
  `netlist.pb.txt` + `initial.plc` pair (the 17 IBM benchmarks, NG45, and the
  synthetic suite under `test/benchmarks/`).
- **eda_io path** — `src/place_design.py` accepts standard EDA inputs
  (LEF / DEF / structural Verilog / SDC / Liberty), merges them into a neutral
  `Design`, converts to the same ICCAD04 pair, and loads it through the same
  `load_benchmark`. The flow in this document then runs completely unchanged,
  including exact PLC proxy scoring. Results are written back out as an
  updated DEF, ICC2/Innovus Tcl, and/or a QoR report. See
  `src/eda_io/README.md`.
