"""Check physical-net isolation and the moving blockage surrogate."""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "diagnostic"))
from dreamplace_rudy import blockage_map, physical_net_mask


def test_blockage_and_synthetic_net_exclusion():
    assert physical_net_mask([b"n1", b"grp0_0", "n2"]) == [True, False, True]
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
