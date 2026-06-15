"""Verify TraceFields reproduces the exact inline feature arithmetic it replaced.

The local-search operators used to inline expressions of the form

    float(field[ri, ci] / field_max) if field is not None else 0.0
    float(field.ravel()[flat] / field_max) if field is not None else 0.0

where ``field_max = max(float(field.max()), 1e-12)``. TraceFields bundles that
arithmetic into ``cong_at`` / ``dens_at`` / ``cong_flat`` / ``dens_flat``. This
test pins the helper to the original expressions bit-for-bit, including the
``None`` (tracing-inactive / grid-unavailable) path.
"""

import numpy as np

from placer.ml.data_collection import TraceFields


def _inline_at(field, field_max, ri, ci):
    return float(field[ri, ci] / field_max) if field is not None else 0.0


def _inline_flat(field, field_max, idx):
    return float(field.ravel()[idx] / field_max) if field is not None else 0.0


def test_trace_fields_match_inline_expressions():
    rng = np.random.RandomState(0)
    for _ in range(50):
        rows, cols = int(rng.randint(2, 12)), int(rng.randint(2, 12))
        cong = rng.rand(rows, cols) * rng.choice([1.0, 1e-9, 1e3])
        dens = rng.rand(rows, cols) * rng.choice([1.0, 1e-9, 1e3])
        tf = TraceFields(cong=cong, dens=dens)

        cong_max = max(float(cong.max()), 1e-12)
        dens_max = max(float(dens.max()), 1e-12)

        for _ in range(20):
            ri, ci = int(rng.randint(rows)), int(rng.randint(cols))
            flat = int(rng.randint(rows * cols))
            assert tf.cong_at(ri, ci) == _inline_at(cong, cong_max, ri, ci)
            assert tf.dens_at(ri, ci) == _inline_at(dens, dens_max, ri, ci)
            assert tf.cong_flat(flat) == _inline_flat(cong, cong_max, flat)
            assert tf.dens_flat(flat) == _inline_flat(dens, dens_max, flat)


def test_trace_fields_none_grids_return_zero():
    tf = TraceFields(cong=None, dens=None)
    assert tf.cong_max == 1.0 and tf.dens_max == 1.0
    assert tf.cong_at(0, 0) == 0.0
    assert tf.dens_at(3, 7) == 0.0
    assert tf.cong_flat(5) == 0.0
    assert tf.dens_flat(2) == 0.0


def test_trace_fields_mixed_one_grid_present():
    cong = np.arange(12, dtype=np.float64).reshape(3, 4)
    tf = TraceFields(cong=cong, dens=None)
    cong_max = max(float(cong.max()), 1e-12)
    assert tf.cong_at(2, 3) == float(cong[2, 3] / cong_max)
    assert tf.cong_flat(7) == float(cong.ravel()[7] / cong_max)
    assert tf.dens_at(2, 3) == 0.0
    assert tf.dens_flat(7) == 0.0
