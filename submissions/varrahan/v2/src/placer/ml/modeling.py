"""Inactive model framework for learned candidate ranking.

This module deliberately has no hooks into the production placer. It defines the
operator schemas, artifact format, and ranking helpers that later integration can
call before exact candidate scoring.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence


OPERATORS = (
    "hard_relocation",
    "soft_relocation",
    "hard_2opt",
    "soft_2opt",
    "hard_soft_swap",
    "hard_soft_soft_cycle",
)

COMMON_FEATURES = (
    "accepted_in_pass",
    "source_hot_rank_norm",
)

OPERATOR_FEATURES = {
    "hard_relocation": (
        "net_degree",
        "net_degree_log1p",
        "net_degree_norm",
        "macro_w_norm",
        "macro_h_norm",
        "x_norm",
        "y_norm",
        "target_x_norm",
        "target_y_norm",
        "dx_norm",
        "dy_norm",
        "source_field_norm",
        "target_field_norm",
        "source_congestion_norm",
        "target_congestion_norm",
        "source_density_norm",
        "target_density_norm",
        "target_cold_rank_norm",
    ),
    "soft_relocation": (
        "net_degree",
        "net_degree_log1p",
        "net_degree_norm",
        "macro_w_norm",
        "macro_h_norm",
        "x_norm",
        "y_norm",
        "target_x_norm",
        "target_y_norm",
        "dx_norm",
        "dy_norm",
        "source_field_norm",
        "target_field_norm",
        "source_congestion_norm",
        "target_congestion_norm",
        "source_density_norm",
        "target_density_norm",
        "target_cold_rank_norm",
    ),
    "hard_2opt": (
        "i_net_degree",
        "i_net_degree_log1p",
        "i_net_degree_norm",
        "j_net_degree",
        "j_net_degree_log1p",
        "j_net_degree_norm",
        "i_w_norm",
        "i_h_norm",
        "j_w_norm",
        "j_h_norm",
        "i_x_norm",
        "i_y_norm",
        "j_x_norm",
        "j_y_norm",
        "distance_norm",
        "i_congestion_norm",
        "j_congestion_norm",
        "i_density_norm",
        "j_density_norm",
    ),
    "soft_2opt": (
        "k1_net_degree",
        "k1_net_degree_log1p",
        "k1_net_degree_norm",
        "k2_net_degree",
        "k2_net_degree_log1p",
        "k2_net_degree_norm",
        "distance_norm",
        "k1_field_norm",
        "k2_field_norm",
        "k1_congestion_norm",
        "k2_congestion_norm",
        "k1_density_norm",
        "k2_density_norm",
        "wl_delta",
    ),
    "hard_soft_swap": (
        "hard_net_degree",
        "hard_net_degree_log1p",
        "hard_net_degree_norm",
        "soft_net_degree",
        "soft_net_degree_log1p",
        "soft_net_degree_norm",
        "hard_w_norm",
        "hard_h_norm",
        "distance_norm",
        "hard_field_norm",
        "hard_congestion_norm",
        "soft_congestion_norm",
        "hard_density_norm",
        "soft_density_norm",
    ),
    "hard_soft_soft_cycle": (
        "hard_net_degree",
        "hard_net_degree_log1p",
        "hard_net_degree_norm",
        "s1_net_degree",
        "s1_net_degree_log1p",
        "s1_net_degree_norm",
        "s2_net_degree",
        "s2_net_degree_log1p",
        "s2_net_degree_norm",
        "hard_w_norm",
        "hard_h_norm",
        "hard_s1_distance_norm",
        "s1_s2_distance_norm",
        "hard_field_norm",
        "hard_congestion_norm",
        "s1_congestion_norm",
        "s2_congestion_norm",
        "hard_density_norm",
        "s1_density_norm",
        "s2_density_norm",
        "s1_rank_norm",
    ),
}


def feature_names_for(operator: str) -> tuple[str, ...]:
    """Return the stable feature order for an operator model."""
    if operator not in OPERATOR_FEATURES:
        raise ValueError(f"unknown ML operator: {operator}")
    names = [*COMMON_FEATURES, *OPERATOR_FEATURES[operator]]
    return tuple(dict.fromkeys(names))


def _feature_value(candidate: Mapping, name: str, default: float) -> float:
    features = candidate.get("features")
    if isinstance(features, Mapping) and name in features:
        value = features[name]
    else:
        value = candidate.get(f"feature.{name}", default)
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def vectorize_candidate(
    candidate: Mapping,
    feature_names: Sequence[str],
    *,
    missing_value: float = 0.0,
) -> list[float]:
    """Convert a trace candidate row or flattened row into a model vector."""
    return [_feature_value(candidate, name, missing_value) for name in feature_names]


@dataclass(frozen=True)
class ModelSpec:
    """Metadata for one model artifact.

    Supported backends:
    - ``linear_json``: dependency-free placeholder/integration-test scorer.
    - ``xgboost_json``: lazy-loads ``xgboost.Booster`` from ``model_path``.
    """

    operator: str
    backend: str
    feature_names: tuple[str, ...]
    model_path: str | None = None
    missing_value: float = 0.0
    top_k_default: int | None = None
    keep_heuristic_first: int = 0
    random_exploration_fraction: float = 0.0

    @classmethod
    def from_dict(cls, data: Mapping, base_dir: Path | None = None) -> "ModelSpec":
        operator = str(data["operator"])
        feature_names = tuple(data.get("feature_names") or feature_names_for(operator))
        model_path = data.get("model_path")
        if model_path is not None and base_dir is not None:
            model_path = str((base_dir / str(model_path)).resolve())
        return cls(
            operator=operator,
            backend=str(data["backend"]),
            feature_names=feature_names,
            model_path=None if model_path is None else str(model_path),
            missing_value=float(data.get("missing_value", 0.0)),
            top_k_default=(
                None if data.get("top_k_default") is None
                else int(data["top_k_default"])
            ),
            keep_heuristic_first=int(data.get("keep_heuristic_first", 0)),
            random_exploration_fraction=float(data.get("random_exploration_fraction", 0.0)),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "ModelSpec":
        path = Path(path)
        with path.open("rt", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle), base_dir=path.parent)

    def validate(self) -> None:
        if self.operator not in OPERATORS:
            raise ValueError(f"unknown ML operator: {self.operator}")
        if not self.feature_names:
            raise ValueError("model spec must contain at least one feature")
        if self.backend not in {"linear_json", "xgboost_json"}:
            raise ValueError(f"unsupported model backend: {self.backend}")
        if self.backend == "xgboost_json" and not self.model_path:
            raise ValueError("xgboost_json model spec requires model_path")


@dataclass(frozen=True)
class TrainingMatrix:
    """Training-ready rows for regression or grouped ranking."""

    operator: str
    feature_names: tuple[str, ...]
    X: list[list[float]]
    y: list[float]
    group_sizes: list[int]
    rows: list[Mapping]


def build_training_matrix(
    rows: Iterable[Mapping],
    operator: str,
    *,
    label: str = "score_gain",
    feature_names: Sequence[str] | None = None,
    missing_value: float = 0.0,
) -> TrainingMatrix:
    """Build X/y/group arrays from flattened candidate rows.

    Rows are filtered to one operator and sorted by ``(run_id, group_id,
    candidate_rank)`` so ranking libraries can consume ``group_sizes`` directly.
    Use ``label='score_gain'`` for regression or ``label='relevance'`` after
    calling ``add_group_relevance`` for LambdaMART-style ranking.
    """
    if operator not in OPERATORS:
        raise ValueError(f"unknown ML operator: {operator}")
    names = tuple(feature_names or feature_names_for(operator))
    selected = [row for row in rows if row.get("operator") == operator]
    selected.sort(
        key=lambda row: (
            str(row.get("run_id", "")),
            str(row.get("group_id", "")),
            -1 if row.get("candidate_rank") is None else int(row.get("candidate_rank")),
        )
    )

    X = [
        vectorize_candidate(row, names, missing_value=missing_value)
        for row in selected
    ]
    y = []
    for row in selected:
        if row.get(label) is None:
            raise ValueError(f"row missing training label {label!r}")
        y.append(float(row[label]))

    group_sizes = []
    last_key = None
    for row in selected:
        key = (row.get("run_id"), row.get("group_id"))
        if key != last_key:
            group_sizes.append(1)
            last_key = key
        else:
            group_sizes[-1] += 1

    return TrainingMatrix(
        operator=operator,
        feature_names=names,
        X=X,
        y=y,
        group_sizes=group_sizes,
        rows=selected,
    )


class CandidateRanker:
    """Rank candidates for one operator without mutating placer state."""

    def __init__(
        self,
        spec: ModelSpec,
        predictor: Callable[[Sequence[Sequence[float]]], Sequence[float]],
    ):
        spec.validate()
        self.spec = spec
        self._predictor = predictor

    @classmethod
    def from_spec(cls, spec: ModelSpec) -> "CandidateRanker":
        if spec.backend == "linear_json":
            return cls(spec, _load_linear_json(spec))
        if spec.backend == "xgboost_json":
            return cls(spec, _load_xgboost_json(spec))
        raise ValueError(f"unsupported model backend: {spec.backend}")

    @classmethod
    def from_file(cls, path: str | Path) -> "CandidateRanker":
        return cls.from_spec(ModelSpec.from_file(path))

    def scores(self, candidates: Sequence[Mapping]) -> list[float]:
        matrix = [
            vectorize_candidate(cand, self.spec.feature_names, missing_value=self.spec.missing_value)
            for cand in candidates
        ]
        return [float(score) for score in self._predictor(matrix)]

    def rank_indices(self, candidates: Sequence[Mapping]) -> list[int]:
        scores = self.scores(candidates)
        return sorted(range(len(scores)), key=lambda idx: (-scores[idx], idx))

    def select_top_k(
        self,
        candidates: Sequence[Mapping],
        top_k: int | None = None,
        *,
        keep_heuristic_first: int | None = None,
        rng: random.Random | None = None,
    ) -> list[int]:
        """Return candidate indices to exact-score later.

        The returned order is model-preferred order after preserving the first
        few heuristic candidates requested by the spec or caller. If
        ``random_exploration_fraction`` is set, reserve that fraction of the
        top-K budget for random candidates not already selected.
        """
        if not candidates:
            return []
        k = self.spec.top_k_default if top_k is None else top_k
        if k is None:
            k = len(candidates)
        k = max(0, min(int(k), len(candidates)))
        keep = (
            self.spec.keep_heuristic_first
            if keep_heuristic_first is None
            else keep_heuristic_first
        )
        keep = max(0, min(int(keep), len(candidates), k))
        selected = list(range(keep))
        selected_set = set(selected)
        explore_n = int(k * self.spec.random_exploration_fraction)
        explore_n = max(0, min(explore_n, k - len(selected)))
        if explore_n:
            chooser = rng or random
            pool = [idx for idx in range(len(candidates)) if idx not in selected_set]
            for idx in chooser.sample(pool, k=min(explore_n, len(pool))):
                selected.append(idx)
                selected_set.add(idx)
        for idx in self.rank_indices(candidates):
            if len(selected) >= k:
                break
            if idx not in selected_set:
                selected.append(idx)
                selected_set.add(idx)
        return selected


class ModelBank:
    """Collection of independent operator rankers."""

    def __init__(self, rankers: Iterable[CandidateRanker] = ()):
        self._rankers = {ranker.spec.operator: ranker for ranker in rankers}

    @classmethod
    def from_manifest(cls, path: str | Path) -> "ModelBank":
        path = Path(path)
        with path.open("rt", encoding="utf-8") as handle:
            data = json.load(handle)
        specs = data.get("models", data)
        return cls(
            CandidateRanker.from_spec(ModelSpec.from_dict(item, base_dir=path.parent))
            for item in specs
        )

    def get(self, operator: str) -> CandidateRanker | None:
        return self._rankers.get(operator)

    def require(self, operator: str) -> CandidateRanker:
        ranker = self.get(operator)
        if ranker is None:
            raise KeyError(f"no ML ranker configured for operator {operator!r}")
        return ranker

    def operators(self) -> tuple[str, ...]:
        return tuple(sorted(self._rankers))


def _load_linear_json(spec: ModelSpec):
    if not spec.model_path:
        raise ValueError("linear_json model spec requires model_path")
    with Path(spec.model_path).open("rt", encoding="utf-8") as handle:
        data = json.load(handle)
    intercept = float(data.get("intercept", 0.0))
    weights = [float(x) for x in data.get("weights", [])]
    if len(weights) != len(spec.feature_names):
        raise ValueError(
            f"linear_json weight count {len(weights)} does not match "
            f"{len(spec.feature_names)} features"
        )

    def predict(matrix: Sequence[Sequence[float]]) -> list[float]:
        return [intercept + sum(w * x for w, x in zip(weights, row)) for row in matrix]

    return predict


def _load_xgboost_json(spec: ModelSpec):
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise RuntimeError(
            "xgboost is required to load xgboost_json model artifacts"
        ) from exc
    booster = xgb.Booster()
    booster.load_model(str(spec.model_path))

    def predict(matrix: Sequence[Sequence[float]]):
        dmat = xgb.DMatrix(matrix, feature_names=list(spec.feature_names))
        return booster.predict(dmat)

    return predict
