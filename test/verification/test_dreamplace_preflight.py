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
    _temporary_fixed_sig,
    _write_cache,
    dreamplace_design_name,
    run_dreamplace,
)
from dreamplace_bridge.pb_to_bookshelf import extract_bookshelf_data
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


def test_recursive_prototype_cache_signature_covers_fixed_state_and_grouping():
    fixed = {"hard/a": (10.0, 20.0)}
    base = _temporary_fixed_sig(fixed, [["hard/a", "soft/b"]], 8)

    assert base != _temporary_fixed_sig({"hard/a": (11.0, 20.0)}, [["hard/a", "soft/b"]], 8)
    assert base != _temporary_fixed_sig(fixed, [["hard/a", "soft/c"]], 8)
    assert base != _temporary_fixed_sig(fixed, [["hard/a", "soft/b"]], 9)
    assert base == _temporary_fixed_sig(fixed, [["soft/b", "hard/a"]], 8)


def test_dreamplace_cache_key_covers_target_density(tmp_path):
    common = dict(
        benchmark_dir=tmp_path,
        iterations=300,
        random_seed=1000,
        num_threads=2,
        soft_macros_movable=True,
        random_center_init=False,
    )

    assert _cache_key(**common, target_density=0.70) != _cache_key(
        **common, target_density=0.75
    )


def test_bookshelf_temporary_fixed_position_makes_movable_hard_a_terminal():
    class Node:
        macro_name = None

        def __init__(self, name, pos, size=(4.0, 2.0), fixed=False):
            self._name = name
            self._pos = pos
            self._size = size
            self._fixed = fixed

        def get_name(self):
            return self._name

        def get_pos(self):
            return self._pos

        def get_width(self):
            return self._size[0]

        def get_height(self):
            return self._size[1]

        def get_fix_flag(self):
            return self._fixed

    class Plc:
        hard_macro_indices = [0, 1]
        soft_macro_indices = []
        port_indices = []
        modules_w_pins = [Node("a", (5.0, 5.0)), Node("b", (8.0, 8.0))]
        nets = {}
        mod_name_to_indices = {}

        @staticmethod
        def get_canvas_width_height():
            return 100.0, 80.0

    nodes, _nets, _cw, _ch, scale = extract_bookshelf_data(
        Plc(),
        temporary_fixed_positions={"a": (20.0, 30.0)},
    )
    by_name = {node.name: node for node in nodes}

    assert by_name["a"].is_terminal
    assert by_name["a"].fixed
    assert by_name["a"].x_ll == (20.0 - 2.0) * scale
    assert by_name["a"].y_ll == (30.0 - 1.0) * scale
    assert not by_name["b"].is_terminal


@pytest.mark.integration
@pytest.mark.cuda
@pytest.mark.skipif(
    not (DEFAULT_BUILD_ROOT / "install" / "dreamplace" / "Placer.py").exists(),
    reason="run scripts/dreamplace/bootstrap.sh all to install DREAMPlace",
)
def test_dreamplace_native_extensions_match_pinned_python_abi():
    ok, detail = probe(DEFAULT_BUILD_ROOT)

    assert ok, detail
