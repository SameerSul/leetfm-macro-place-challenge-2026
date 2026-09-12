"""Check routing surrogates, exact-tail conversion, and refinement eligibility."""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "diagnostic"))
from replay_gpu_soft_refinement import blockage_map


def test_blockage_coverage_and_empty_input():
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for device in devices:
        edges = torch.tensor([0.0, 1.0, 2.0], device=device, dtype=torch.float64)
        lower = torch.tensor([[0.5, 0.0], [1.0, 1.0]], device=device, dtype=torch.float64)
        upper = torch.tensor([[1.5, 1.0], [2.0, 2.0]], device=device, dtype=torch.float64)
        result = blockage_map(lower, upper, edges, edges)
        torch.testing.assert_close(
            result.cpu(), torch.tensor([[0.5, 0.0], [0.5, 1.0]], dtype=torch.float64)
        )
        assert blockage_map(lower[:0], upper[:0], edges, edges).count_nonzero() == 0


def test_soft_surrogate_gradients_and_cuda_parity():
    from replay_gpu_soft_refinement import net_terms

    def evaluate(device):
        pins = torch.tensor(
            [[0.2, 0.3], [1.4, 1.6], [0.4, 1.3], [1.6, 0.7]],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )
        lengths = torch.tensor([2, 2], device=device)
        pin_net = torch.tensor([0, 0, 1, 1], device=device)
        weights = torch.tensor([1.0, 2.0], dtype=torch.float64, device=device)
        edges = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64, device=device)
        capacities = torch.tensor([2.0, 3.0], dtype=torch.float64, device=device)

        def objective(p):
            wl, h, v = net_terms(p, lengths, pin_net, weights, 0.1, edges, edges, capacities)
            return wl + h.square().sum() + v.square().sum()

        if device == "cpu":
            assert torch.autograd.gradcheck(objective, (pins,))
        value = objective(pins)
        value.backward()
        assert torch.isfinite(pins.grad).all()
        return value.detach().cpu(), pins.grad.cpu()

    reference = evaluate("cpu")
    if torch.cuda.is_available():
        for actual, expected in zip(evaluate("cuda"), reference):
            torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)


def test_exact_tail_weights_preserve_axes_units_and_boundary_smoothing():
    from replay_gpu_soft_refinement import exact_route_weights, smoothing_matrix

    congestion = np.zeros(40)
    congestion[0], congestion[-1] = 10, 9
    scorer = SimpleNamespace(
        grid_row=5,
        grid_col=4,
        grid_h_routes=7,
        grid_v_routes=11,
        smooth_range=1,
        dens_grid_area=2,
        _swap_tail_baseline=lambda: dict(
            congestion_order=np.argsort(-congestion, kind="stable"),
            congestion=congestion,
            density_nonzero=0,
            density=np.zeros(20),
        ),
    )
    h, v = exact_route_weights(scorer)
    expected_h, expected_v = np.zeros((4, 5)), np.zeros((4, 5))
    expected_h[3, 3], expected_h[3, 4] = 1 / 12, 1 / 8
    expected_v[0, 0], expected_v[1, 0] = 1 / 8, 1 / 12
    np.testing.assert_allclose(h, expected_h, atol=1e-15)
    np.testing.assert_allclose(v, expected_v, atol=1e-15)
    for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        horizontal = torch.ones((4, 5), device=device, dtype=torch.float64, requires_grad=True)
        vertical = torch.ones_like(horizontal, requires_grad=True)
        hs = horizontal @ smoothing_matrix(5, 1, device).T
        vs = smoothing_matrix(4, 1, device) @ vertical
        loss = 0.25 * (hs[3, 4] + vs[0, 0])
        gradients = torch.autograd.grad(loss, (horizontal, vertical))
        for actual, expected in zip(gradients, (h, v)):
            torch.testing.assert_close(actual.cpu(), torch.from_numpy(expected))


def test_refinement_without_eligible_macros_needs_no_regions():
    from replay_gpu_soft_refinement import movable_bounds

    data = dict(hierarchy=SimpleNamespace(location_graph=None), region=None)
    benchmark = SimpleNamespace(num_hard_macros=2, macro_fixed=torch.zeros(3, dtype=torch.bool))
    indices, lower, upper = movable_bounds(data, benchmark, np.zeros((3, 2)), 0.5)
    assert len(indices) == 0 and lower.shape == upper.shape == (0, 2)


def test_float32_boundary_projection_preserves_frozen_macros_and_valid_states():
    from replay_gpu_soft_refinement import prepare_candidate
    from macro_place.utils import validate_placement

    positions = np.array([[37.95985794067383, 5], [5, 5], [10, 10]], dtype=np.float32)
    benchmark = SimpleNamespace(
        num_macros=3,
        num_hard_macros=0,
        canvas_width=38.51,
        canvas_height=20,
        macro_sizes=torch.tensor([[1.1002857685089111, 1], [1, 1], [1, 1]]),
        macro_fixed=torch.tensor([False, False, True]),
        macro_positions=torch.from_numpy(positions.copy()),
    )
    benchmark.get_movable_mask = lambda: ~benchmark.macro_fixed
    assert not validate_placement(torch.from_numpy(positions), benchmark)[0]
    original = positions.copy()
    projected, valid, count = prepare_candidate(positions, benchmark, np.array([0]))
    assert valid and count == 1
    assert validate_placement(torch.from_numpy(projected), benchmark)[0]
    np.testing.assert_array_equal(positions, original)
    np.testing.assert_array_equal(projected[1:], positions[1:])
    repeated, valid, count = prepare_candidate(projected, benchmark, np.array([0]))
    assert repeated is projected and valid and count == 0
