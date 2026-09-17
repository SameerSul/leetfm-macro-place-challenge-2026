# Short GPU refinement and exact congestion guidance

The corrected 31-design baseline and all three refinement variants are
validated. The 24-step variant retains the lowest mean exact proxy in each
benchmark family. Eight-step exact-hotspot guidance improves more designs and
retains 96.3% of the long variant's total proxy gain with 43.0% less measured
proposal time. All variants remain offline; production behavior is unchanged.

| Variant | Improved / unchanged | IBM mean proxy | NG45 mean proxy | Synthetic mean proxy | Sum of warm proposal medians, s |
| --- | --- | ---: | ---: | ---: | ---: |
| Baseline | — | 1.181101299 | 0.717060425 | 1.436273508 | — |
| `long` | 20 / 11 | **1.179760243** | **0.716155049** | **1.432496211** | 14.680 |
| `short` | 22 / 9 | 1.179836373 | 0.716313093 | 1.433350080 | 8.608 |
| `guided` | 23 / 8 | 1.179806573 | 0.716363450 | 1.432574308 | 8.362 |

Scores use independent scalar evaluation on the same float32 coordinates
promoted to float64. Each variant has zero regressions against the baseline;
all 93 final outputs pass legality and the complete saved hierarchy contract.
All four NG45 tag designs and all ten synthetic truth designs pass their
independent audits. The final verification suite has **138 passing tests**.

## Interpretation

The short variant cuts measured proposal time by **41.4%**. Compared directly
with long, it has lower proxy on six designs, higher proxy on ten, and equal
proxy on fifteen. Guided beats short on thirteen, loses on eleven, and ties
on seven (comparison tolerance `1e-7`). Guided's IBM and synthetic means are
better than short's, while its NG45 mean is worse. More improved designs does
not imply the largest total gain: long still has the best mean in every family.

Exact congestion decreases on only two designs for long and short, and three
for guided; those same designs also improve wirelength and density. Most gains
come from density and wirelength reductions that outweigh congestion rises.
A congestion increase alone is not a rejection condition: acceptance minimizes
exact proxy subject to legality and the existing hierarchy contract.

| Variant | IBM mean congestion change | NG45 mean congestion change | Synthetic mean congestion change |
| --- | ---: | ---: | ---: |
| `long` | +0.003209128 | +0.000060146 | +0.007517317 |
| `short` | +0.002099231 | -0.000057047 | +0.005572468 |
| `guided` | +0.002517611 | -0.000026692 | +0.004835504 |

The exact baseline field helps some proposals but does not establish consistent
congestion relief. It still weights an approximate rectangular routing model
and remains fixed throughout the short block. Under the production objective,
keep long as the quality reference. The eight-step variants offer a measured
runtime trade-off; neither unconditionally replaces long. Production
integration still needs ordinary-guard pipeline validation and measurement
of the complete added runtime. No end-to-end or production-Numba speedup is
claimed by this experiment.

## Corrected production baseline

The baseline uses the default CPU DREAMPlace backend, normal time guards,
fresh private seed caches, and the corrected seed-cleanup call from the
[September 11 investigation](GPU_PLACEMENT_EXPERIMENTS.md). It finishes with
31 legal outputs, zero overlaps, 31 passing final hierarchy audits, and 31
saved captures. Production source hashes match the initial manifest.

| Family | Designs | Ordinary evaluator mean proxy | Placer time, seconds |
| --- | ---: | ---: | ---: |
| IBM | 17 | 1.181128351 | 956.699 |
| NG45 | 4 | 0.717060223 | 147.055 |
| Synthetic | 10 | 1.436273508 | 428.165 |

These means match the recorded September 9 reference means. This does not
establish coordinate equality with the archived run. Brief implementation
tests and smoke checks overlap the baseline run; its times are validation
measurements, not isolated performance comparisons. Differences from the
paired scalar baseline in the first table reflect scoring precision, not
placement changes.

## Fixed treatments and acceptance

The treatment definitions were fixed before observing the new replay results:

| Variant | Steps | Exact checkpoints | Congestion guidance |
| --- | ---: | --- | --- |
| `long` | 24 | 4, 8, 16, 24 | Existing differentiable RUDY top tail |
| `short` | 8 | 1, 2, 4, 8 | Same RUDY top tail |
| `guided` | 8 | 1, 2, 4, 8 | Exact baseline directional hotspot weights |

All variants preserve float64 math, the Adam learning rate, maximum half-cell
displacement, eligible owned singleton selection, wirelength and density
terms, and fresh float32 exact acceptance. Hard/fixed/bridge macros and all
bundles remain frozen. Ownership, confidence, hierarchy edges, and contract
limits never change. All 31 designs have eligible soft singletons; the driver
also handles an empty eligible set by retaining the baseline.

Guided reuses `_tail_view()` from the production tile search. Its directional
weights include exact top-tail selection, macro blockages, the transpose of
source-normalized smoothing, and the 0.5 proxy weight. `exact_route_weights()`
converts raw-route derivatives to normalized demand units and transposes the
row-major CPU maps to the GPU's x-major maps. The weights are uploaded once
and held fixed for eight steps. They weight differentiable rectangular demand;
they do not make its routing gradients exact. This independently adapts
existing repository code; underlying PyTorch/RUDY references remain
[30 and 31](REFERENCES.md).

Each GPU batch downloads four checkpoints together. Bounds, fixed-state
preservation, and the complete saved hierarchy contract precede fresh exact
scoring. The baseline remains available and a retained gain must exceed
`1e-6`. Three repetitions produce identical candidates within each backend;
shared long/short checkpoints match exactly. Independent transactions do not
repair the known shared `_last_pos_cache` rollback defect.

## Float32 projection correction

The initial full replay rejected twenty proposal states on IBM07, IBM08, and
IBM15 because of float32 canvas-boundary rounding. All 372 proposed states
passed the hierarchy contract; only 352 passed validity and received exact
scores. The initial retained results were legal and independently verified.
They improved 19 / 22 / 21 designs for long / short / guided and remain in
`replay/results.json` for attribution.

For example, an IBM07 soft center at 37.95985794067383 with half-width
0.5501428842544556 exceeds the return-precision canvas edge 38.5099983215332
when added in float32, despite being bounded before conversion. The existing
`clamp_in_bounds()` fixes all twenty rejections in a 36-state diagnostic.
Every corrected state passes the unchanged hierarchy contract.

`prepare_candidate()` now applies that clamp only when validity fails, copies
back only eligible soft coordinates, and checks validity again. Already-valid
states return the same array untouched. The regression test checks this
numerical boundary, preservation of frozen coordinates, and identity of the
valid path. The proposal and scalar-oracle functions are unchanged, verified
by comparing their ASTs with the archived `unclamped_driver.py`.

All three variants were rerun on the three affected designs with three
repetitions and the same four-checkpoint quota. All 36 states are now legal
and contract-passing; twenty receive the projection. Nine retained outputs
pass fresh independent scalar checks. The other 28 designs had no invalid
states and take the unchanged valid path. `final_results.json` combines those
28 verified outcomes with the three corrected replays and records each row's
source. The final table uses this corrected comparison.

## Timing and independent validation

The machine is an RTX 4050 Laptop GPU with 6 GB VRAM, PyTorch 2.10.0+cu128,
and NumPy 2.4.2. No other placement job runs during formal proposal timing.
Warm medians use repetitions two and three. The first table sums one proposal
batch per design, including CPU packing, device setup, optimization, and
readback. It excludes file parsing, acceptance validation, and scalar audits.
Optimization-loop sums are 11.297 / 5.232 / 4.994 s for long / short / guided.
Small sample counts and laptop clock variation limit timing precision.

Proposal timings come from the initial full sweep. The projection correction
runs in acceptance, outside the unchanged measured proposal function; the
extra acceptance work is not included in those timing ratios.

The focused IBM10/IBM17/NVDLA CPU/CUDA comparison independently verifies
18 outputs. All CPU/CUDA retained coordinates match bit-for-bit and match
the corresponding full-replay outputs. Its checkpoints are already valid and
unaffected by the projection correction. Ratios below compare the same
PyTorch implementation, not the production Numba search.

| Design | Long CPU / CUDA s | Short CPU / CUDA s | Guided CPU / CUDA s |
| --- | ---: | ---: | ---: |
| IBM10 | 2.231 / 0.781 | 0.973 / 0.405 | 1.067 / 0.511 |
| IBM17 | 3.850 / 1.252 | 1.358 / 0.816 | 1.457 / 0.833 |
| NVDLA | 1.900 / 0.666 | 0.581 / 0.345 | 0.490 / 0.333 |

All timed proposals finish before independent scalar auditing starts. Unique
baseline/winner coordinates are checked in spawned CPU processes, configurable
with `--oracle-workers` (four in the full runs). No live CUDA context is forked.
`proposals.json` marks states unverified; `results.json` is written only after
all scalar comparisons and truth/tag audits pass. The initial 88 unique scalar
checks take 796.971 s, outside proposal timing. Their maximum difference from
the fast scorer is 1.34e-15. Per-row `validation_s` covers fast acceptance and
fresh scoring state; scalar audit times are separate.

## Reproduction and records

Choose fresh output directories when repeating these commands:

```bash
rtk proxy uv run python test/diagnostic/run_dreamplace_cuda_comparison.py \
  --capture-final --gpu 0 --normal \
  --out ml_data/gpu_placement/new-baseline ibm ng45 synthetic
rtk proxy uv run python test/diagnostic/replay_gpu_soft_refinement.py \
  --inputs ml_data/gpu_placement/new-baseline \
  --out ml_data/gpu_placement/new-replay \
  --variants long short guided --devices cuda --repeats 3 --oracle-workers 4
rtk proxy uv run --with pytest python -m pytest test/verification/ -q
```

The current drivers configure backends and telemetry through source constants,
with trace paths passed explicitly to spawned scalar workers. The fixed native
cuBLAS workspace setting preserves deterministic matrix operations. This
configuration cleanup leaves the experiment treatments and measured results
unchanged; historical source manifests retain the original implementation.

All September 12 records are under `ml_data/gpu_placement/20260912/`:
`baseline/` holds coordinates, hierarchy captures, source hashes, seed calls,
and complete telemetry; `replay/` holds the initial full comparison;
`backend-check/` holds the focused CPU/CUDA check; `bounds-diagnostic.json`
records boundary diagnosis; `rounding-fix/` holds the three corrected replays.
`final_results.json` records final quality with per-row provenance, and
`final_summary.json` records the corrected family and variant comparisons.
