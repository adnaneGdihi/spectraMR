"""The ULF low-pass operator and its closed-form DC prox.

Paired with ``src/spectramr/infrastructure/physics/ulf_lowpass.py``.
"""

from __future__ import annotations

import torch

from spectramr.infrastructure.physics.fft_ops import fft2c
from spectramr.infrastructure.physics.ulf_lowpass import _lowpass_otf, lowpass_dc_prox, ulf_degrade


def _h(n: int = 16, cutoff: float = 0.3) -> torch.Tensor:
    return _lowpass_otf(n, n, cutoff, torch.device("cpu"), torch.float32)


def test_the_prox_is_the_minimiser_of_its_objective() -> None:
    """Gradient of w/2||y - Ax||^2 + rho/2||x - z||^2 vanishes at the returned x."""
    torch.manual_seed(0)
    y, z = torch.rand(1, 1, 16, 16), torch.rand(1, 1, 16, 16)
    h, w, rho = _h(), 3.0, 0.7
    x = lowpass_dc_prox(fft2c(y), z, h, w, rho).requires_grad_(True)
    objective = 0.5 * w * (y - ulf_degrade(x, 0.3)).pow(2).sum() + 0.5 * rho * (x - z).pow(2).sum()
    (grad,) = torch.autograd.grad(objective, x)
    assert grad.abs().max() < 1e-4


def test_no_data_weight_returns_the_prior_point() -> None:
    torch.manual_seed(1)
    y, z = torch.rand(1, 1, 16, 16), torch.rand(1, 1, 16, 16)
    torch.testing.assert_close(lowpass_dc_prox(fft2c(y), z, _h(), 0.0, 1.0), z, atol=1e-5, rtol=0)


def test_the_ulf_map_strategy_reexports_the_one_operator() -> None:
    from spectramr.infrastructure.training.strategies import ulf_map_strategy as ulf

    assert ulf.ulf_degrade is ulf_degrade and ulf.lowpass_dc_prox is lowpass_dc_prox
