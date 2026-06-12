import json
import random

import pytest

from placer.ml.modeling import (
    OPERATORS,
    CandidateRanker,
    ModelBank,
    ModelSpec,
    build_training_matrix,
    feature_names_for,
    vectorize_candidate,
)


def test_feature_schema_exists_for_each_operator():
    assert set(OPERATORS) == {
        "hard_relocation",
        "soft_relocation",
        "hard_2opt",
        "soft_2opt",
        "hard_soft_swap",
        "hard_soft_soft_cycle",
    }
    for operator in OPERATORS:
        names = feature_names_for(operator)
        assert "accepted_in_pass" in names
        assert len(names) == len(set(names))


def test_vectorize_candidate_accepts_raw_and_flat_rows():
    names = ("dx_norm", "target_density_norm", "missing_feature")
    raw = {"features": {"dx_norm": 0.25, "target_density_norm": 0.75}}
    flat = {"feature.dx_norm": 0.5, "feature.target_density_norm": None}

    assert vectorize_candidate(raw, names) == [0.25, 0.75, 0.0]
    assert vectorize_candidate(flat, names, missing_value=-1.0) == [0.5, -1.0, -1.0]


def test_linear_json_ranker_scores_and_selects_top_k(tmp_path):
    model_path = tmp_path / "hard-reloc-linear.json"
    model_path.write_text(
        json.dumps({"intercept": 0.1, "weights": [2.0, -1.0]}),
        encoding="utf-8",
    )
    spec = ModelSpec(
        operator="hard_relocation",
        backend="linear_json",
        feature_names=("dx_norm", "target_density_norm"),
        model_path=str(model_path),
        top_k_default=2,
        keep_heuristic_first=1,
    )
    ranker = CandidateRanker.from_spec(spec)
    candidates = [
        {"features": {"dx_norm": 0.0, "target_density_norm": 0.0}},
        {"features": {"dx_norm": 0.2, "target_density_norm": 0.0}},
        {"features": {"dx_norm": 0.1, "target_density_norm": 1.0}},
    ]

    assert ranker.scores(candidates) == pytest.approx([0.1, 0.5, -0.7])
    assert ranker.rank_indices(candidates) == [1, 0, 2]
    assert ranker.select_top_k(candidates) == [0, 1]


def test_model_bank_loads_manifest_with_relative_paths(tmp_path):
    model_path = tmp_path / "soft-linear.json"
    model_path.write_text(json.dumps({"weights": [1.0]}), encoding="utf-8")
    manifest = tmp_path / "models.json"
    manifest.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "operator": "soft_2opt",
                        "backend": "linear_json",
                        "feature_names": ["distance_norm"],
                        "model_path": model_path.name,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    bank = ModelBank.from_manifest(manifest)
    assert bank.operators() == ("soft_2opt",)
    assert bank.require("soft_2opt").scores(
        [{"features": {"distance_norm": 0.3}}]
    ) == pytest.approx([0.3])
    assert bank.get("hard_2opt") is None


def test_top_k_can_reserve_random_exploration(tmp_path):
    model_path = tmp_path / "linear.json"
    model_path.write_text(json.dumps({"weights": [1.0]}), encoding="utf-8")
    spec = ModelSpec(
        operator="soft_2opt",
        backend="linear_json",
        feature_names=("distance_norm",),
        model_path=str(model_path),
        random_exploration_fraction=0.5,
    )
    ranker = CandidateRanker.from_spec(spec)
    candidates = [
        {"features": {"distance_norm": 0.1}},
        {"features": {"distance_norm": 0.2}},
        {"features": {"distance_norm": 0.3}},
        {"features": {"distance_norm": 0.4}},
    ]

    selected = ranker.select_top_k(candidates, top_k=2, rng=random.Random(1))

    assert len(selected) == 2
    assert selected[0] in range(len(candidates))
    assert selected[1] == 3


def test_model_spec_validation_rejects_unknown_operator():
    spec = ModelSpec(operator="unknown", backend="linear_json", feature_names=("x",))
    with pytest.raises(ValueError, match="unknown ML operator"):
        spec.validate()


def test_build_training_matrix_groups_rows_for_rankers():
    rows = [
        {
            "operator": "hard_relocation",
            "run_id": "r",
            "group_id": "g2",
            "candidate_rank": 1,
            "score_gain": 0.2,
            "feature.dx_norm": 0.2,
        },
        {
            "operator": "soft_2opt",
            "run_id": "r",
            "group_id": "ignored",
            "candidate_rank": 0,
            "score_gain": 9.0,
            "feature.distance_norm": 9.0,
        },
        {
            "operator": "hard_relocation",
            "run_id": "r",
            "group_id": "g1",
            "candidate_rank": 0,
            "score_gain": 0.1,
            "feature.dx_norm": 0.1,
        },
        {
            "operator": "hard_relocation",
            "run_id": "r",
            "group_id": "g2",
            "candidate_rank": 0,
            "score_gain": 0.3,
            "feature.dx_norm": 0.3,
        },
    ]

    matrix = build_training_matrix(
        rows,
        "hard_relocation",
        feature_names=("dx_norm",),
    )

    assert matrix.X == [[0.1], [0.3], [0.2]]
    assert matrix.y == [0.1, 0.3, 0.2]
    assert matrix.group_sizes == [1, 2]
