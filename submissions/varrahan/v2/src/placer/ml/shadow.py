"""Shadow-mode model scoring for local-search candidate groups.

Shadow mode is opt-in via ``ML_MODEL_MANIFEST``. It scores exactly the candidates
the heuristic already generated and records diagnostics, but it never filters,
reorders, accepts, or rejects a move.
"""

from __future__ import annotations

import os
import time
import warnings
from pathlib import Path
from typing import Mapping, Sequence

from placer.ml.modeling import ModelBank


_BANK = None
_BANK_INITIALIZED = False
_BANK_ERROR = None


def _parse_top_ks(value: str | None = None) -> tuple[int, ...]:
    text = value if value is not None else os.environ.get("ML_SHADOW_TOP_K", "")
    if not text:
        return (1, 3, 5, 10, 16)
    out = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        k = int(item)
        if k > 0 and k not in out:
            out.append(k)
    return tuple(out) or (1, 3, 5, 10, 16)


def get_shadow_model_bank():
    """Return the process-wide shadow model bank, or None when disabled."""
    global _BANK, _BANK_INITIALIZED, _BANK_ERROR
    if not _BANK_INITIALIZED:
        _BANK_INITIALIZED = True
        manifest = os.environ.get("ML_MODEL_MANIFEST")
        if manifest:
            try:
                _BANK = ModelBank.from_manifest(Path(manifest))
            except Exception as exc:  # pragma: no cover - defensive runtime guard
                _BANK_ERROR = exc
                warnings.warn(f"failed to load ML_MODEL_MANIFEST={manifest!r}: {exc}")
                _BANK = None
    return _BANK


def shadow_rank_group(
    *,
    operator: str,
    candidates: Sequence[Mapping],
    trace=None,
    field: str | None = None,
    group_id: str | None = None,
    top_ks: Sequence[int] | None = None,
) -> dict | None:
    """Score one candidate group and optionally emit an ``ml_shadow_group`` event."""
    if not candidates:
        return None
    bank = get_shadow_model_bank()
    if bank is None:
        return None
    ranker = bank.get(operator)
    if ranker is None:
        return None

    try:
        start_ns = time.perf_counter_ns()
        scores = ranker.scores(candidates)
        inference_ns = time.perf_counter_ns() - start_ns
    except Exception as exc:  # pragma: no cover - model errors must not affect placement
        warnings.warn(f"ML shadow scoring failed for {operator}: {exc}")
        return None

    top_ks = tuple(int(k) for k in (top_ks or _parse_top_ks()) if int(k) > 0)
    gains = [float(candidate.get("score_gain", 0.0)) for candidate in candidates]
    best_gain = max(gains)
    improving = best_gain > 0.0
    best_indices = {idx for idx, gain in enumerate(gains) if gain == best_gain}
    pred_order = sorted(range(len(scores)), key=lambda idx: (-float(scores[idx]), idx))
    pred_rank = {idx: rank + 1 for rank, idx in enumerate(pred_order)}
    best_exact_model_rank = min(pred_rank[idx] for idx in best_indices)

    data = {
        "operator": operator,
        "field": field,
        "group_id": group_id,
        "rows": len(candidates),
        "model_backend": ranker.spec.backend,
        "model_inference_ns": int(inference_ns),
        "best_exact_gain": best_gain,
        "best_exact_model_rank": best_exact_model_rank,
        "improving": improving,
        "predicted_top_indices": pred_order[: min(8, len(pred_order))],
        "predicted_top_scores": [
            float(scores[idx]) for idx in pred_order[: min(8, len(pred_order))]
        ],
    }

    for k in top_ks:
        kk = min(k, len(pred_order))
        chosen = pred_order[:kk]
        chosen_best_gain = max((gains[idx] for idx in chosen), default=float("-inf"))
        data[f"best_recall@{k}"] = any(idx in best_indices for idx in chosen)
        data[f"improving_recall@{k}"] = (
            any(gains[idx] > 0.0 for idx in chosen) if improving else None
        )
        data[f"mean_regret@{k}"] = max(0.0, best_gain - chosen_best_gain)

    if trace is not None:
        trace.event("ml_shadow_group", **data)
    return data


def _reset_shadow_model_bank_for_tests() -> None:
    global _BANK, _BANK_INITIALIZED, _BANK_ERROR
    _BANK = None
    _BANK_INITIALIZED = False
    _BANK_ERROR = None
