# Hierarchy GNN Trace Schema

Candidate trace rows are JSONL events emitted only when `HIER_GNN_TRACE=1`.
Plateau telemetry is a separate lightweight JSONL stream for ML/DL scheduling
data and is controlled by `HIER_PLATEAU_TRACE` instead.

Every row includes:

- `schema_version`: integer schema version. Current version: `1`.
- `time_s`: wall-clock timestamp from `time.time()`.
- `event`: event type.
- `benchmark`: benchmark name when available.

## Common Candidate Fields

Candidate-level events use these fields when applicable:

- `operator`: hierarchy operator that generated the candidate.
- `candidate_id`: monotonically increasing candidate id inside that operator call when available.
- `old_proxy`: exact proxy before testing the candidate.
- `candidate_proxy`: exact proxy after the candidate, or `null` when not scored.
- `proxy_delta`: `candidate_proxy - old_proxy`, or `null` when not scored.
- `accepted`: boolean candidate outcome.
- `rejection_reason`: `null` for accepted candidates; otherwise one of:
  - `illegal_overlap`
  - `out_of_bounds`
  - `out_of_hierarchy_region`
  - `hierarchy_quality_failed`
  - `exact_proxy_failed`
  - `proxy_budget_failed`
  - `field_gap_below_threshold`
  - `no_eligible_cluster`
  - `not_scored`

## Event Types

## Plateau Telemetry Stream

Plateau rows are written to
`ml_data/beyondppa_gnn/plateau/plateau_telemetry.jsonl` by default. Override
with `HIER_PLATEAU_TRACE_PATH`, or disable with `HIER_PLATEAU_TRACE=0`.

Common fields:

- `schema_version`: plateau schema version. Current version: `1`.
- `event`: `hier_plateau_telemetry` or `hier_budget_schedule`.
- `benchmark`
- `diagnostic_no_deadlines`

`hier_plateau_telemetry` rows include:

- `plateau_pass`
- `proxy_before`, `proxy_after`, `proxy_gain`
- `elapsed_s`
- `candidates`, `legal`, `scored`, `accepts`, `accept_rate`
- `plateaued`

`hier_budget_schedule` rows include:

- `pass_name`
- `run`
- `has_spare`
- `plateau_trigger`
- `useful_soft_trigger`
- `budget_s`
- `min_spare_s`

### `hier_relocation_candidates`

Hard propose-all relocation candidate pool after candidate scoring and ordering.
The `candidates` list contains sampled rows with:

- `macro`
- `hot_rank`
- `candidate_rank`
- `target_index`
- `score`
- `local_field`
- `target_field`
- `structural_delta`
- `x`, `y`
- `gnn_score`: model score when `HIER_GNN_RANK=1`, otherwise `null`
- `gnn_rank_error`: diagnostic rank-error field when available, otherwise
  `null`

### `hier_relocation_result`

Accepted hard or soft relocation moves. The `accepted` list contains sampled
accepted moves with macro id, target coordinate, proxy delta, and structural
delta when available.

### `hier_decompression_candidate`

Cluster decompression candidate. Additional fields:

- `cluster`
- `movable_count`
- `member_count`
- `soft_count`
- `expansion_factor`
- `axis_scale`
- `hierarchy_quality_before`
- `hierarchy_quality_after`
- `hierarchy_quality_delta`

### `hier_swap_candidates`

Region-bounded swap candidate pool for one source macro. Additional fields:

- `kind`: `hard_hard`, `hard_soft`, or `soft_soft`
- `field`: `congestion` or `density`
- `source`
- `candidate_count`
- `candidates`: sampled candidate rows with:
  - `candidate_rank`
  - `target`
  - `source_field`
  - `target_field`
  - `outside_region`
  - `legal`
  - proxy fields and outcome fields

### `hier_coldspot_candidate`

Coldspot tightening candidate. Additional fields:

- `field_gap`
- `min_field_gap`
- `cluster`
- `candidate_pool_size`
- `selector_enabled`
- `oracle_enabled`
- `selector_rank`
- `selected_by_gnn`
- `selected_by_policy`
- `gnn_score`
- `gnn_rank_error`
- `is_noop`
- `movable_count`
- `member_count`
- `member_area`
- `cluster_heat`
- `source_field`, `target_field`, `score`
- `anchor_x`, `anchor_y`
- `window_microns`
- `window_cells`
- `target_density`
- `pick`
- `soft_count`
- `soft_moved`
- displacement summary fields such as `hard_disp_mean`, `hard_disp_max`,
  `soft_disp_mean`, and `soft_disp_max`
- before/after cluster bbox and centroid fields
- hierarchy-quality fields
- `committed`, indicating the candidate actually changed the placement

Skipped coldspot rounds may omit cluster fields when no candidate was generated.
When `HIER_GNN_COLDSPOT_SELECT=1`, each round may emit the no-op candidate plus
all generated kick outcomes. Only selected candidates have exact proxy fields
unless a candidate was evaluated before an earlier selected candidate accepted.
When `HIER_GNN_COLDSPOT_ORACLE=1`, generated candidates are exact-scored for
offline labels. Nonselected candidates may have `accepted=true` but
`committed=false`; these are useful oracle-positive selector labels and do not
change placement.

### `hier_pass_result`

Pass-level summary for hierarchy operators. This is not a candidate label.

### `hier_final`

Final placement summary. This is not a candidate label.

## Sampling

`HIER_GNN_TRACE_MAX_CANDIDATES` limits sampled candidate lists and candidate
events per operator call. It does not change placement behavior.

## Production Boundary

Schema v1 traces are training data only. They do not enable a model and do not
change candidate ordering or acceptance.
