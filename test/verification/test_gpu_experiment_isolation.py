"""Contract checks for isolated diagnostic CUDA experiments."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from utils.config import gpu_experiment_allows, gpu_experiment_selected


def test_unset_gpu_experiment_preserves_all_existing_cuda_routes(monkeypatch):
    monkeypatch.delenv("HIER_GPU_EXPERIMENT", raising=False)

    assert gpu_experiment_allows("overlap_prefilter")
    assert gpu_experiment_allows("graph_tension_batches")
    assert not gpu_experiment_selected("overlap_prefilter")


def test_selected_experiment_disables_other_optional_cuda_routes(monkeypatch):
    monkeypatch.setenv("HIER_GPU_EXPERIMENT", "graph_tension_batches")

    assert gpu_experiment_selected("graph_tension_batches")
    assert not gpu_experiment_allows("overlap_prefilter")
    assert not gpu_experiment_allows("compound_coldspot")
