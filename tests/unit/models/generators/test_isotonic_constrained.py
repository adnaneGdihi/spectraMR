"""The isotonic generator is non-decreasing in its input (#2404)."""

from __future__ import annotations

import torch

from spectramr.models.generators.isotonic_constrained import IsotonicConstrainedGenerator


def _tiny() -> IsotonicConstrainedGenerator:
    torch.manual_seed(0)
    return IsotonicConstrainedGenerator(
        in_channels=1, out_channels=1, feature_dim=4, num_layers=2, num_constraints=2
    ).double()


def test_raising_the_input_never_lowers_the_output():
    g = _tiny().eval()
    for seed in range(10):
        torch.manual_seed(seed)
        x = torch.rand(1, 1, 12, 12, dtype=torch.float64)
        d = 0.1 * torch.rand_like(x)
        with torch.no_grad():
            assert ((g(x + d) - g(x)) >= -1e-12).all()


def test_monotone_after_a_training_step():
    # The constraint is structural, so arbitrary weight updates must keep it.
    g = _tiny()
    opt = torch.optim.SGD(g.parameters(), lr=1.0)
    x = torch.rand(2, 1, 12, 12, dtype=torch.float64)
    (-(g(x) - x.flip(-1)).square().mean()).backward()
    opt.step()
    d = 0.1 * torch.rand_like(x)
    with torch.no_grad():
        assert ((g(x + d) - g(x)) >= -1e-12).all()


def test_output_shape_matches_input():
    g = _tiny()
    assert g(torch.rand(2, 1, 12, 12, dtype=torch.float64)).shape == (2, 1, 12, 12)
