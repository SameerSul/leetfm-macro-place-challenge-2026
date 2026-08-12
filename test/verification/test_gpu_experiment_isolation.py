"""Contract checks for isolated diagnostic CUDA experiments."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from utils.config import gpu_experiment_selected


def test_unset_gpu_experiment_selects_no_diagnostic_route(monkeypatch):
    monkeypatch.delenv("HIER_GPU_EXPERIMENT", raising=False)

    assert not gpu_experiment_selected("overlap_prefilter")
    assert not gpu_experiment_selected("graph_tension_batches")


def test_selected_experiment_matches_only_its_named_route(monkeypatch):
    monkeypatch.setenv("HIER_GPU_EXPERIMENT", "graph_tension_batches")

    assert gpu_experiment_selected("graph_tension_batches")
    assert not gpu_experiment_selected("overlap_prefilter")
    assert not gpu_experiment_selected("compound_coldspot")
