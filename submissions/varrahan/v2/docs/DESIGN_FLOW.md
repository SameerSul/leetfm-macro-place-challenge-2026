# v2 Design Flow

This document describes the production flow implemented by
`src/placer/pipeline/macro_placer.py`.

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

    D --> P0[Phase 0 baseline legalization]
    P0 --> GUARD{Exact scoring path usable?}
    GUARD -->|No| LB2[Optional baseline-only hard 2-opt]
    LB2 --> LB3[Optionally exact-score baseline and first DP candidate on large-DP path]
    LB3 --> LBR[Return baseline or first DP if it wins]

    GUARD -->|Yes| S0[Score baseline with exact proxy]
    S0 --> BEST[best_pl and best_score]

    BEST --> CG1[Phase 1 iterative congestion-gradient from baseline]
    CG1 --> CGM[Read H and V routing congestion from PLC]
    CGM --> CGH[Use max H,V per grid cell]
    CGH --> CGP[Push movable hard macros away from hot cells]
    CGP --> CGL[Legalize hard macros]
    CGL --> CGS[Exact proxy score]
    CGS --> CGK{Improved best?}
    CGK -->|Yes| BEST
    CGK -->|No| CGSTOP[Stop or halve step depending on phase rules]

    CG1 --> CG2[Phase 2 wide cong-grad from baseline if Phase 1 improved]
    CG2 --> CGS
    CG2 --> CG3[Phase 3 cong-grad from current best if cong-grad improved]
    CG3 --> CGS

    DP1 --> DPH[Harvest completed DP candidates after Phase 3]
    DP2 --> DPH
    DP3 --> DPH
    DPH --> DPL[Clip and legalize DP hard macros]
    DPL --> DPS[Copy DP soft positions, then exact proxy score]
    DPS --> DPK{Improved best?}
    DPK -->|Yes| BEST
    DPK -->|No| DPDROP[Do not update best]
    DPS --> DPB[Save every scored DP placement as a DP basin]

    DPB --> P5B[Phase 5b cong-grad from best using latest PLC map]
    P5B --> CGS
    BEST --> P5C[Phase 5c wide-from-best if cong-grad improved]
    P5C --> CGS

    BEST --> N[Random Gaussian restarts from initial positions]
    N --> NL[Legalize each noisy hard placement]
    NL --> NS[Exact proxy score]
    NS --> NK{Improved best?}
    NK -->|Yes| BEST
    NK -->|No| NDROP[Discard candidate]

    DPB --> P7[Phase 7 DP-rescue cong-grad chains]
    P7 --> P7L[Legalize each rescue step]
    P7L --> P7S[Exact proxy score]
    P7S --> BEST

    BEST --> P8[Phase 8 top-K cong-grad from best]
    P8 --> P8K[Try K in 5, 10, 20 hottest movable hard macros]
    P8K --> P8L[Legalize and score]
    P8L --> BEST

    BEST --> P9[Phase 9 random-order legalization]
    P9 --> P9L[Three legalize trials from initial positions with randomized tie-breaks]
    P9L --> P9S[Sequential exact proxy scoring]
    P9S --> BEST

    BEST --> T[Multi-seed hard 2-opt]
    DPB --> T
    T --> TSEED[Seeds are best_pl plus retained DP basins]
    TSEED --> TPRUNE[Prune DP seeds more than 0.02 above best]
    TPRUNE --> TTRY[Try hard-macro swaps]
    TTRY --> TKNN[k=20 spatial nearest neighbors]
    TKNN --> TTELE[S9 hot-first ordering plus cold-region teleport]
    TTELE --> TSCORE[Score swaps with IncrementalScorer when init succeeds, else full exact fallback]
    TSCORE --> TVERIFY[Re-score each seed finalist exactly]
    TVERIFY --> BEST

    BEST --> R2[R2 refinement loop, up to 20 budget-gated rounds]
    R2 --> R2S[Build or reuse round IncrementalScorer]
    R2S --> HR1[Hard relocation by congestion field]
    HR1 --> HR2[Hard relocation by density field if not skipped]
    HR2 --> HR3[Hard relocation by combined cong-density field if not skipped]
    HR3 --> SR1[Soft relocation by congestion field, WL-prefiltered, if not skipped]
    SR1 --> SR2[Soft relocation by density field, WL-prefiltered]
    SR2 --> SS1[Soft-soft 2-opt by congestion field, WL-prefiltered, if not skipped]
    SS1 --> SS2[Soft-soft 2-opt by density field, WL-prefiltered, if not skipped]
    SS2 --> HXS1[Hard-soft cross-swap by congestion field if not skipped]
    HXS1 --> HXS2[Hard-soft cross-swap by density field if not skipped]
    HXS2 --> HS31[Hard-soft-soft 3-cycle by congestion field if not skipped]
    HS31 --> HS32[Hard-soft-soft 3-cycle by density field if not skipped]
    HS32 --> R2O[Hard 2-opt cleanup, k=16 plus S9 cold-teleport]
    R2O --> R2V[Exact proxy verify after passes with accepts]
    R2V --> R2Q{Round improved enough and budget remains?}
    R2Q -->|Yes| R2
    R2Q -->|No| POST[Post-R2 leftover soft relocation]

    POST --> POST1[Soft relocation by congestion then density, top_hot 1024, n_targets 4]
    POST1 --> POST2[Exact proxy verify]
    POST2 --> RET[Return best_pl]

    C --> SCORE[Exact proxy formula]
    SCORE --> EQ[proxy = wirelength + 0.5 density + 0.5 congestion]
```

## TLDR

`MacroPlacer.place()` is a budget-gated, accept-only search over macro
placements. It launches up to three DREAMPlace subprocesses as asynchronous seed
generators, legalizes the benchmark's initial hard macro positions, and then
repeatedly creates candidate placements through congestion-gradient
perturbations, noisy restarts, alternate legalization orders, and local-search
moves.

Placement-level candidates are judged by the same exact proxy wrapper:

```text
proxy = wirelength + 0.5 * density + 0.5 * congestion
```

`best_pl` and `best_score` are the main incumbent quality state: a placement
updates them only when its exact proxy score is lower.

## Important Details

- Hard and soft macro labels come from the benchmark API. The placer does not
  infer them.
- Congestion-gradient only proposes hard-macro position perturbations. The
  proposed hard positions are then legalized and exact-scored before they can
  update `best_pl`.
- Non-DREAMPlace candidates leave soft macro positions as they are in the source
  placement. DREAMPlace candidates can carry DREAMPlace-produced soft positions.
- DREAMPlace candidates are all tested when they finish in time. The best one
  may update `best_pl`, but every scored DP candidate is also retained as a
  later 2-opt / rescue seed.
- Random-noise and random-order phases restart from the initial hard positions,
  not from `best_pl`.
- R2 is richer than "relocation plus 2-opt": each round can run hard relocation,
  soft relocation, soft-soft swaps, hard-soft swaps, hard-soft-soft 3-cycles, and
  hard 2-opt cleanup.
- R2 local passes use `IncrementalScorer` for candidate scoring when it can be
  initialized, then verify accepted pass results with `_exact_proxy`.
- The optional `V2_MULTISEED_MP` path parallelizes DP-seed 2-opt after the inline
  best-seed 2-opt. With the env var unset, seeds run sequentially in-process.
