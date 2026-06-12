"""ML-assisted local-search utilities."""

from .data_collection import get_candidate_trace
from .dataset import add_group_relevance, load_candidates, trace_summary
from .modeling import (
    OPERATORS,
    CandidateRanker,
    ModelBank,
    ModelSpec,
    TrainingMatrix,
    build_training_matrix,
    feature_names_for,
    vectorize_candidate,
)

__all__ = [
    "OPERATORS",
    "CandidateRanker",
    "ModelBank",
    "ModelSpec",
    "TrainingMatrix",
    "add_group_relevance",
    "build_training_matrix",
    "feature_names_for",
    "get_candidate_trace",
    "load_candidates",
    "trace_summary",
    "vectorize_candidate",
]
