import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "dreamplace"))
sys.path.insert(0, str(ROOT / "src"))

import dreamplace_bridge.run_bridge as bridge
from dreamplace_bridge.run_bridge import (
    _cache_key,
    _default_dreamplace_config,
    _write_cache,
    dreamplace_design_name,
    run_dreamplace,
)
from preflight import DEFAULT_BUILD_ROOT, probe


def test_legacy_env_cannot_disable_iccad2023_bb_nesterov(monkeypatch):
    monkeypatch.setenv("HIER_DREAMPLACE_BB", "0")
    cfg = _default_dreamplace_config("design.aux", "results")

    assert cfg["macro_place_flag"] == 1
    assert cfg["use_bb"] == 1
    assert cfg["global_place_stages"][0]["optimizer"] == "nesterov"


def test_legacy_env_cannot_disable_dreamplace_cache_reads(tmp_path, monkeypatch):
    benchmark_dir = tmp_path / "benchmark"
    benchmark_dir.mkdir()
    scratch_root = tmp_path / "scratch"
    design = dreamplace_design_name(benchmark_dir)
    work_dir = scratch_root / design
    work_dir.mkdir(parents=True)
    key = _cache_key(
        benchmark_dir.resolve(),
        iterations=200,
        random_seed=1000,
        num_threads=4,
        soft_macros_movable=False,
        random_center_init=False,
    )
    expected_hard = np.array([[1.0, 2.0]], dtype=np.float64)
    expected_soft = np.array([[3.0, 4.0]], dtype=np.float64)
    _write_cache(work_dir, key, expected_hard, expected_soft)
    monkeypatch.setenv("HIER_DREAMPLACE_CACHE", "0")
    monkeypatch.setattr(bridge, "is_available", lambda: True)

    hard, soft = run_dreamplace(
        str(benchmark_dir),
        scratch_root=str(scratch_root),
        return_full=True,
    )

    np.testing.assert_array_equal(hard, expected_hard)
    np.testing.assert_array_equal(soft, expected_soft)


@pytest.mark.integration
@pytest.mark.cuda
@pytest.mark.skipif(
    not (DEFAULT_BUILD_ROOT / "install" / "dreamplace" / "Placer.py").exists(),
    reason="run scripts/dreamplace/bootstrap.sh all to install DREAMPlace",
)
def test_dreamplace_native_extensions_match_pinned_python_abi():
    ok, detail = probe(DEFAULT_BUILD_ROOT)

    assert ok, detail
