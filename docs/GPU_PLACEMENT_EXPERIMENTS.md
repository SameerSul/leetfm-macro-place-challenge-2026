# GPU placement experiments — September 11, 2026

The ordered investigation tests a congestion-aware grouped DREAMPlace seed,
then coordinated soft refinement on captured final placements. Production GPU
defaults, scoring, hierarchy inference, ownership, and acceptance gates remain
unchanged. Results below distinguish experiments from accepted behavior.

## Prerequisite correction

The initial control exposed a stale `plc` positional argument in
`run_seed_portfolio._score_seed`'s call to `_soft_relocation_moves`. The callee
no longer accepts that argument. It bound the score to `deadline` and then
received `deadline` again by keyword, raising `TypeError` and discarding every
DREAMPlace candidate requiring soft cleanup. Remove the obsolete argument;
keep the current shared relocation API. The focused regression check binds
the actual seed call against the actual callee signature.

The interrupted `ml_data/gpu_placement/20260911/control` run is invalid for
attribution. Both subsequent experiment arms use the corrected source.

Normal-guard validation on the default CPU backend returns IBM10 at
1.174197912 in 105.482 s and NVDLA at 0.730595589 in 45.655 s. Both are legal,
have zero overlaps, and pass all final contracts. NVDLA selects the recurrent
`re2map_recursive_2` seed, exercising the restored scoring path. These are
validation runs, not a paired comparison with the CUDA experiment arms.

## Congestion-aware seed

`test/diagnostic/run_gpu_placement_experiments.py` reuses the maintained
evaluator runner and captures final hierarchy contracts for later replay.
Both arms run CUDA with separate fresh caches. The RUDY arm changes only the
DREAMPlace subprocess entrypoint to `test/diagnostic/dreamplace_rudy.py`;
the installed DREAMPlace tree is unchanged.

The subprocess enables at most one native area-adjustment round per native
stage, uses benchmark
routing grid dimensions and tracks-per-micron capacities, and disables pin
area adjustment. Coordinates are scaled by 1000 for Bookshelf, so capacities
are divided by the same factor. Grouping-net weights remain in the placement
objective but are zeroed in the separate routing-demand tensor. Every RUDY
update adds hard-macro footprint occupancy times the benchmark's directional
routing allocation/capacity. Footprints follow current macro centers and use
original sizes, including during native area inflation. Ports do not become
hard blockages. This is an additive surrogate inspired by
[reference 30](REFERENCES.md), not an exact routing implementation.

An initial IBM10 trial left native area adjustment inert: its minimum density
overflow was 0.213792, above the native 0.15 activation threshold. Its ordinary
seed made 600 RUDY calls but no area adjustments. That interrupted trial is
stored under `rudy/`; `activation-probe/` records the threshold diagnosis.
The active experiment uses a uniform 0.30 activation threshold on all designs
and logs every attempted/retained native adjustment. It does not change any
VivaPlace hierarchy-contract limit.

Each run saves source hashes, fresh Bookshelf inputs, configuration and routing
metadata, subprocess logs and routing-call counts, seed timings, coordinates,
exact evaluator results, and complete hierarchy telemetry. Native kernel and
subprocess overhead are included in the recorded seed-call times. Search
deadlines are disabled by default through the existing diagnostic option;
seed preparation still retains its existing safety deadlines. Different
seeds can change subsequent work, so these runs are quality comparisons, not
kernel speedup measurements.

```bash
rtk proxy uv run python test/diagnostic/run_gpu_placement_experiments.py \
  --mode control --out ml_data/gpu_placement/new-control ibm10 ibm17 nvdla
rtk proxy uv run python test/diagnostic/run_gpu_placement_experiments.py \
  --mode rudy --out ml_data/gpu_placement/new-rudy ibm10 ibm17 nvdla
# Add --normal to retain ordinary hierarchy-search deadlines.
```

## Coordinated soft refinement

`test/diagnostic/replay_gpu_soft_refinement.py` uses the saved final float32
coordinates and their captured hierarchy limits. It optimizes eligible owned
soft singletons together, freezes hard/fixed/bridge macros and every bundle,
and leaves graph ownership and confidence untouched. Existing leaf/child
membership and region boxes bound proposals. A macro outside its region at
the baseline is frozen rather than projected to a different initial state.

One PyTorch implementation runs on CPU and CUDA with float64 placement math.
All topology, coordinates, gradients, and fields stay on the selected device
for 24 Adam steps. Segmented physical-pin reductions form a smooth HPWL term;
rectangular occupancy forms density and directional RUDY terms. Congestion
includes hard blockages and source-normalized smoothing. The surrogate uses
the existing objective coefficients. It is not the evaluator's routing model.
See [references 30 and 31](REFERENCES.md).

Displacement is bounded to half a routing cell per axis, and four completed
checkpoints are downloaded together. Each must pass bounds, fixed-state
preservation, and the complete saved hierarchy contract before fresh exact
scoring. Hard coordinates remain bit-identical to the validated baseline.
Only gains above `1e-6` survive. The baseline and retained winner also receive
fresh scalar evaluator checks on the same float32 coordinates promoted to
float64. This avoids measuring scalar accumulation precision as movement.
These local independent transactions do not repair the shared checkpoint
`_last_pos_cache` defect.

Three repetitions check identical outputs within each backend. Timing records
separate CPU packing, device setup, the optimization loop, readback, and
validation. First calls and warm calls are distinguishable; device timers are
synchronized. This compares the prototype's CPU and CUDA implementations,
not CUDA against the production Numba search. Cross-backend outputs may
differ; each retained result must independently satisfy the exact gates.

```bash
rtk proxy uv run python test/diagnostic/replay_gpu_soft_refinement.py \
  --inputs ml_data/gpu_placement/new-control \
  --out ml_data/gpu_placement/new-soft ibm10 ibm17 nvdla
rtk proxy uv run --with pytest python -m pytest test/verification/ -q
```

## Measurements and decision

Hardware is an NVIDIA GeForce RTX 4050 Laptop GPU with 6 GB VRAM. The main
environment uses PyTorch 2.10.0+cu128 and NumPy 2.4.2. DREAMPlace runs in its
separate existing build environment.

The completed `control-fixed/` versus `rudy-active/` comparison returns
identical final coordinates, exact evaluator scores, and complete hierarchy
metrics on all three designs. All six outputs are legal with zero overlaps.

| Design | Control final proxy | RUDY final proxy | Control API s | RUDY API s |
| --- | ---: | ---: | ---: | ---: |
| IBM10 | 1.173217535 | 1.173217535 | 98.895 | 99.844 |
| IBM17 | 1.393702984 | 1.393702984 | 89.233 | 91.357 |
| NVDLA | 0.815043032 | 0.815043032 | 44.884 | 56.271 |

All three ordinary grouped Bookshelf inputs are byte-identical between arms.
Nine fresh seed calls take 77.570 s in the control and 90.712 s with RUDY;
total API time is 233.012 s versus 247.472 s. These single sequential runs
include unrelated test/smoke activity and are not an isolated speed benchmark.
The ordinary and both recurrent IBM seeds each retain one native adjustment;
NVDLA retains one ordinary adjustment and none in its recurrent solves.

On IBM10, the best DREAMPlace seed improves from approximately 1.8242 to
1.6359, but legalized initial placement still wins at 1.4666. IBM17's ordinary
RUDY seed reaches 1.7001, below initial's 1.7506, but fails the unchanged
hierarchy contract. NVDLA also retains initial placement. The RUDY hypothesis
is not promoted because it produces no final gain and adds seed work.

The completed `soft-repeated/` replay improves all three captured placements.
All 24 CPU/CUDA checkpoint candidates pass legality and the complete saved
contracts. Each backend repeats its candidates exactly across three runs;
retained CPU/CUDA coordinates are also identical. Independent scalar scores
agree within `1e-7`. The table uses those fresh scalar scores on float32
coordinates promoted to float64, accounting for the small baseline differences
from the ordinary evaluator table above.

| Design | Eligible softs | Before proxy | Retained proxy | Reduction | CPU proposal s | CUDA proposal s | Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| IBM10 | 704 | 1.173218182 | 1.173139663 | 0.0067% | 2.020 | 0.684 | 2.95x |
| IBM17 | 571 | 1.393703334 | 1.392391389 | 0.0941% | 3.302 | 1.251 | 2.64x |
| NVDLA | 53 | 0.815042601 | 0.814817224 | 0.0277% | 1.282 | 0.528 | 2.43x |

Proposal times are medians of repetitions two and three, including fresh CPU
packing, device setup, 24 optimization steps, and readback. Both backends use
the same PyTorch implementation and two CPU threads. CUDA first-proposal times
are 1.340 / 1.258 / 0.622 s; only IBM10 includes the process's first CUDA use.
Warm optimization loops alone take 0.512 / 1.023 / 0.427 s on CUDA versus
1.870 / 3.074 / 1.186 s on CPU. The small sample count and laptop clock
variation limit precision. These measurements exclude input-file parsing and
validation and do not establish an end-to-end or production-Numba speedup.

| Design | Wirelength change | Density change | Congestion change |
| --- | ---: | ---: | ---: |
| IBM10 | -0.000195529 | -0.005212419 | +0.005446441 |
| IBM17 | +0.000005687 | -0.010329783 | +0.007694519 |
| NVDLA | -0.000109449 | -0.000164051 | -0.000067804 |

The retained IBM gains come primarily from density; exact congestion rises.
Only NVDLA improves all three terms. Later checkpoints often regress exact
proxy and are rejected. The separate one-step IBM10 smoke achieved a larger
gain, 0.000473269, than the fixed four-checkpoint experiment. It is a diagnostic
observation, not a retrospectively tuned replacement result. This prototype
demonstrates useful resident GPU proposal work but does not yet establish a
reliable congestion-relief operator or improve hierarchy compactness. Existing
hierarchy constraints remain hard acceptance requirements.

The independent NVDLA RTL hierarchy-tag audit passes on the retained output;
its hard coordinates are unchanged. All 135 verification tests and 34
EDA/visualizer tests pass. The three new focused tests cover the stale seed
call signature, physical-net exclusion and moving blockage geometry, and
finite-difference surrogate gradients with CUDA value/gradient parity.
Formatting, undefined-name checks, and bytecode compilation also pass.

## Exact-scoring decision

The final probe alternates baseline and retained coordinates on an independent
resident CPU scoring state, checking every score against the fresh scalar
results. After one warmup pair, six samples give warm median exact times of
**12.596 ms IBM10, 17.757 ms IBM17, and 6.354 ms NVDLA**. No placement job runs
alongside this probe. Four calls at those medians cost approximately
**50 / 71 / 25 ms**, before contract and transaction setup. This is a direct
measurement of the existing fast exact scorer, excluding file loading and
the slow independent scalar oracle. The replay's multi-second validation
totals include those diagnostic costs and are not production scorer timings.
Sequential scoring here performs no checkpoint rollback and does not resolve
the separate shared rollback-cache defect.

```bash
rtk proxy uv run python ml_data/gpu_placement/20260911/exact_score_probe.py
```

Do not add a GPU exact scorer for four completed checkpoints. The demonstrated
GPU opportunity is the resident coordinated proposal loop. Keep that prototype
offline: promotion needs a broader paired replay and ordinary-guard pipeline
validation, plus evidence that added runtime earns sufficient retained gains.
Improving agreement between its routing surrogate and exact congestion is
more relevant than accelerating these few exact calls. The native RUDY seed
experiment is rejected for production; the seed-call correction is the only
production code change in this investigation. No new end-to-end speedup or
31-design quality result is claimed.

Raw records are in `ml_data/gpu_placement/20260911/`; `comparison.json`
records RUDY output/contract equality, and `soft-repeated/results.json`
contains every timing, checkpoint score, component score, and input hash.
