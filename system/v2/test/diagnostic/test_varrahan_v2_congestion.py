"""Regression tests for varrahan v2 routing-congestion perturbation."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np


class _FakePlc:
    def __init__(self, h_cong, v_cong):
        self._h_cong = h_cong
        self._v_cong = v_cong

    def get_horizontal_routing_congestion(self):
        return self._h_cong.ravel().tolist()

    def get_vertical_routing_congestion(self):
        return self._v_cong.ravel().tolist()


class _ZeroNoise:
    def normal(self, _loc, _scale, size):
        return np.zeros(size, dtype=np.float64)


def _load_v2_placer():
    path = Path("system/v2/src/main.py")
    spec = importlib.util.spec_from_file_location("varrahan_v2_placer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_routing_congestion_gradient_uses_max_hv_objective():
    placer = _load_v2_placer()

    h_cong = np.zeros((3, 3), dtype=np.float64)
    v_cong = np.zeros((3, 3), dtype=np.float64)

    # Local center is hot enough to perturb. On the left, one channel is hotter;
    # on the right, both channels are moderately hot. max(H,V) (the production
    # field) makes the LEFT side hottest, so the macro steps right (away from it)
    # - the opposite of what an H+V objective would do.
    h_cong[1, 1] = 0.6
    v_cong[1, 1] = 0.6
    h_cong[1, 0] = 2.0
    h_cong[1, 2] = 1.1
    v_cong[1, 2] = 1.1

    pos = np.array([[1.5, 1.5]], dtype=np.float64)
    perturbed = placer._routing_congestion_perturb(
        pos=pos,
        plc=_FakePlc(h_cong, v_cong),
        benchmark=SimpleNamespace(grid_rows=3, grid_cols=3),
        n=1,
        cw=3.0,
        ch=3.0,
        hw=np.array([0.1], dtype=np.float64),
        hh=np.array([0.1], dtype=np.float64),
        movable=np.array([True]),
        frac=0.1,
        rng=_ZeroNoise(),
    )

    # max(H,V): left cell (2.0) is hotter than right (1.1), so step right (+x).
    assert perturbed[0, 0] > pos[0, 0]
    assert perturbed[0, 1] == pos[0, 1]
