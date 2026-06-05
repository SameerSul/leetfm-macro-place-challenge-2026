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


def _parse_filter_operators(value: str | None = None) -> set[str]:
    text = value if value is not None else os.environ.get("ML_FILTER_OPERATORS", "")
    return {part.strip() for part in text.split(",") if part.strip()}


def is_filter_enabled(operator: str) -> bool:
    return operator in _parse_filter_operators()


def _parse_int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


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


def filter_candidate_indices(
    *,
    operator: str,
    candidates: Sequence[Mapping],
    trace=None,
    field: str | None = None,
    group_id: str | None = None,
    top_k: int | None = None,
    keep_heuristic_first: int | None = None,
) -> list[int]:
    """Return original-order candidate indices to exact-score.

    Filtering is disabled unless ``operator`` appears in ``ML_FILTER_OPERATORS``.
    Any missing model or model error falls back to scoring every candidate.
    """
    all_indices = list(range(len(candidates)))
    if not candidates or operator not in _parse_filter_operators():
        return all_indices

    bank = get_shadow_model_bank()
    if bank is None:
        if trace is not None:
            trace.event(
                "ml_filter_group",
                operator=operator,
                field=field,
                group_id=group_id,
                enabled=True,
                applied=False,
                reason="missing_model_bank",
                generated=len(candidates),
                selected=len(candidates),
                skipped=0,
            )
        return all_indices
    ranker = bank.get(operator)
    if ranker is None:
        if trace is not None:
            trace.event(
                "ml_filter_group",
                operator=operator,
                field=field,
                group_id=group_id,
                enabled=True,
                applied=False,
                reason="missing_operator_model",
                generated=len(candidates),
                selected=len(candidates),
                skipped=0,
            )
        return all_indices

    try:
        k = _parse_int_env("ML_FILTER_TOP_K", 0) if top_k is None else int(top_k)
        if k <= 0:
            k = len(candidates)
        keep = (
            _parse_int_env("ML_FILTER_KEEP_HEURISTIC_FIRST", ranker.spec.keep_heuristic_first)
            if keep_heuristic_first is None
            else int(keep_heuristic_first)
        )
        start_ns = time.perf_counter_ns()
        model_selected = ranker.select_top_k(
            candidates,
            top_k=k,
            keep_heuristic_first=keep,
        )
        inference_ns = time.perf_counter_ns() - start_ns
    except Exception as exc:  # pragma: no cover - model errors must not affect placement
        warnings.warn(f"ML filtering failed for {operator}: {exc}")
        if trace is not None:
            trace.event(
                "ml_filter_group",
                operator=operator,
                field=field,
                group_id=group_id,
                enabled=True,
                applied=False,
                reason="model_error",
                error=str(exc),
                generated=len(candidates),
                selected=len(candidates),
                skipped=0,
            )
        return all_indices

    selected = sorted(set(int(idx) for idx in model_selected if 0 <= int(idx) < len(candidates)))
    if not selected:
        selected = all_indices
        applied = False
        reason = "empty_selection"
    else:
        applied = len(selected) < len(candidates)
        reason = "filtered" if applied else "selected_all"

    if trace is not None:
        trace.event(
            "ml_filter_group",
            operator=operator,
            field=field,
            group_id=group_id,
            enabled=True,
            applied=applied,
            reason=reason,
            generated=len(candidates),
            selected=len(selected),
            skipped=len(candidates) - len(selected),
            top_k=k,
            keep_heuristic_first=keep,
            model_backend=ranker.spec.backend,
            model_inference_ns=int(inference_ns),
            selected_indices=selected[:32],
        )
    return selected


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
