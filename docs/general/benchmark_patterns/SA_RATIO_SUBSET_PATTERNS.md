# SA-ratio benchmark subsets (2026-06-24 `--all` run)

This page records benchmark grouping patterns using:

`ratio = current_proxy / SA_base`.

- `ratio` is computed from the latest `--all` results in
  `docs/general/PROGRESS.md` (current system) and `SA_BASELINES` in
  `macro_place/evaluate.py`.
- Higher `ratio` means weaker relative gain vs SA for this benchmark.
- All values are exact to four decimals.

## 1) Full SA-ratio table (all 17 IBM benchmarks)

| Benchmark | Current | SA baseline | Ratio | Gain (SA - current) |
|---|---:|---:|---:|---:|
| ibm01 | 0.9262 | 1.3166 | **0.7035** | 0.3904 |
| ibm02 | 1.1194 | 1.9072 | **0.5869** | 0.7878 |
| ibm03 | 1.0009 | 1.7401 | **0.5752** | 0.7392 |
| ibm04 | 1.0067 | 1.5037 | **0.6695** | 0.4970 |
| ibm06 | 1.2090 | 2.5057 | **0.4825** | 1.2967 |
| ibm07 | 1.0527 | 2.0229 | **0.5204** | 0.9702 |
| ibm08 | 1.1400 | 1.9239 | **0.5925** | 0.7839 |
| ibm09 | 0.8537 | 1.3875 | **0.6153** | 0.5338 |
| ibm10 | 1.1475 | 2.1108 | **0.5436** | 0.9633 |
| ibm11 | 0.9935 | 1.7111 | **0.5806** | 0.7176 |
| ibm12 | 1.6668 | 2.8261 | **0.5898** | 1.1593 |
| ibm13 | 1.0172 | 1.9141 | **0.5314** | 0.8969 |
| ibm14 | 1.2721 | 2.2750 | **0.5592** | 1.0029 |
| ibm15 | 1.3588 | 2.3000 | **0.5908** | 0.9412 |
| ibm16 | 1.2087 | 2.2337 | **0.5411** | 1.0250 |
| ibm17 | 1.4049 | 3.6726 | **0.3825** | 2.2677 |
| ibm18 | 1.3877 | 2.7755 | **0.5000** | 1.3878 |

Sorted by ratio descending (`struggle` under this convention):
`ibm01 -> ibm04 -> ibm09 -> ibm08 -> ibm15 -> ibm12 -> ibm02 -> ibm11 -> ibm03 -> ibm14 -> ibm10 -> ibm16 -> ibm13 -> ibm07 -> ibm18 -> ibm06 -> ibm17`.

## 2) Struggling subsets (from the current ratio convention)

### Primary subset (top-4)
- **ibm01, ibm04, ibm09, ibm08**

### Secondary subset (next-5)
- **ibm15, ibm12, ibm02, ibm11, ibm03**

### Combined struggling band (top-9)
- **ibm01, ibm04, ibm09, ibm08, ibm15, ibm12, ibm02, ibm11, ibm03**

### If you want the opposite convention
Some readers prefer “struggle” as “smallest ratio” (least SA-gain retained). That would be:
- `ibm17, ibm06, ibm18, ibm07, ibm13, ibm16, ...`

Keep one convention and use it consistently; this page uses **higher ratio = weaker relative gain**.

## 3) Intra-subset common patterns (data-backed)

Using `benchmarks/processed/public/*.pt` metadata:

### Common to primary subset (ibm01, ibm04, ibm09, ibm08)
- Hard macro count is small and tightly clustered: **246–301** (avg **273.8**).
- Total macros: **1140–1380** (avg **1288**), total much smaller than extreme high-net/high-complexity designs.
- Soft macro count: **894–1085** (avg **1014.25**).
- Net count is low-to-moderate: **7269–20694**.
- Average macro area and canvas occupancy are stable in this data family (`macro area ≈ 1.19`, occupancy ratio around **0.8001**).
- `ibm15` now ranks fifth by a narrow margin and stays in the same high-ratio
  small-design lane, but it is the high-connectivity exception
  (`46467` nets, `30.35` nets/macro).

### Implemented response for primary subset
- `src/placer/pipeline/segments/floorplan_post_coldspot.py` now runs a structural small-design polish after survivor search.
- Gate: `240 <= num_hard <= 420`, `num_hard + num_soft <= 1600`, and all hard macros movable.
- The pass release pool starts from the weakest-k inferred hierarchy clusters by confidence, keeps only clusters below the confidence cutoff (`cluster_confidence <= 0.92`), and releases the hottest eligible weak clusters.
- It splits into low-net and high-net lanes at `nets_per_macro >= 24.0`, so ibm15-like cases get a larger exact-gated candidate budget.
- It runs up to two adaptive rounds and includes an explicit released-region hard-hard swap pass before the soft-involving released-region swap subpass only when weak-cluster release happened and hard relocation produced enough exact gain.
- It keeps a local best exact-scored state across adaptive rounds and restores that state before returning to the main hierarchy flow.
- Relocation targets now come from cold connected components. Larger/colder components are favored inside candidate ordering, but exact proxy still decides acceptance.
- It never branches on benchmark name; all accepted moves still require exact-proxy improvement and hard legality.

### Shared patterns in secondary subset (ibm15, ibm12, ibm02, ibm11, ibm03)
- Larger spread in nets and topology than primary subset; several low-to-mid designs mixed with large-connectivity cases (`ibm15`, `ibm12`).
- This band likely represents an “intermediate” mode: not the easiest, not the hardest, and therefore sensitive to small operator-threshold changes.
- Ratios are typically in the **0.575–0.591** band where gains are flatter than
  top-tier primary cases and worse than low-ratio designs.

### Implemented response for secondary subset
- The secondary subset is split into two structural shapes, not one operator lane.
- Small high-ratio designs (`ibm02`, `ibm03`, `ibm08`, `ibm11`, `ibm15`) satisfy the
  small-design polish gate. Latest telemetry shows that pass is useful on all
  of the low/mid-net cases, with exact-proxy gains from about **0.0063** to
  **0.0187**.
- When a small low-net design has no releasable weak hierarchy cluster, the
  small-design pass now shifts candidate breadth away from hard relocation and
  toward soft relocation plus soft-involving swaps. This is intended for the
  no-release shape observed in the secondary band while keeping exact-proxy and
  hard-legality gates unchanged.
- `ibm12` is a separate medium/large congestion shape: **651** hard macros,
  **2636** total macros, and **40996** nets. It misses the small-design gate by
  design and instead benefits from late soft-only repair.
- A medium/large soft-continuation lane is available after normal strong-soft
  repair only when the design shape matches (`520 <= hard <= 760`, `2200 <=
  total <= 3200`, and `12 <= nets/macro <= 24`), the preceding strong-soft pass
  already produced enough exact-proxy gain, and spare time remains. It does not
  branch on benchmark name. In the accepted sweep, `ibm12` matched the shape and
  gain gates but skipped continuation because the spare-time gate was false.
- Post-swap hard propose-all is not expanded for this subset; telemetry shows
  zero useful gain there. Runtime is reserved for exact-gated soft continuation
  instead.

### Distinguishing observation across all subsets
- Across all IBM `.pt` inputs examined here, fixed-macro count is **0** for every benchmark.
- The current struggle group is **not explained by fixed macros, canvas occupancy, or explicit hierarchy tags in metadata**.
- Hardness signal is more aligned with **structure/connectivity shape mismatch**: subsets with low raw size but higher relative SA ratio often have weaker relative gains.

## 4) Reproducible extraction

To reproduce this split quickly:

```bash
uv run python - <<'PY'
from pathlib import Path
import torch
import macro_place.evaluate as E

SA = E.SA_BASELINES
current = {
    'ibm01':0.9262,'ibm02':1.1194,'ibm03':1.0009,'ibm04':1.0067,'ibm06':1.2090,
    'ibm07':1.0527,'ibm08':1.1400,'ibm09':0.8537,'ibm10':1.1475,'ibm11':0.9935,
    'ibm12':1.6668,'ibm13':1.0172,'ibm14':1.2721,'ibm15':1.3588,'ibm16':1.2087,
    'ibm17':1.4049,'ibm18':1.3877,
}
rows = sorted(
    [(name, score / SA[name], score, SA[name]) for name, score in current.items()],
    reverse=True,
)
print(rows)
PY
```
