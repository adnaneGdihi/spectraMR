"""Noise2Self normalises by masked ELEMENTS; SURE stays unbiased for a linear denoiser."""

from __future__ import annotations

import torch
from torch import nn

from spectramr.models.losses.sure_n2self_losses import Noise2SelfLoss, SURELoss


def test_noise2self_does_not_scale_with_the_batch_size() -> None:
    """A shared [H, W] J-mask broadcasts over B; counting its pixels once reads 4x at B=4."""
    torch.manual_seed(0)
    pred, target = torch.randn(1, 2, 16, 16), torch.randn(1, 2, 16, 16)
    j_mask = (torch.rand(16, 16) > 0.7).float()
    loss = Noise2SelfLoss()
    one = loss(pred, target, context={"j_mask": j_mask})
    four = loss(pred.repeat(4, 1, 1, 1), target.repeat(4, 1, 1, 1), context={"j_mask": j_mask})
    torch.testing.assert_close(one, four)


def test_noise2self_is_the_mean_over_masked_elements() -> None:
    torch.manual_seed(1)
    pred, target = torch.randn(2, 3, 8, 8), torch.randn(2, 3, 8, 8)
    j_mask = (torch.rand(8, 8) > 0.5).float()
    for norm, expected in (
        ("l2", (pred - target).pow(2)[..., j_mask.bool()].mean()),
        ("l1", (pred - target).abs()[..., j_mask.bool()].mean()),
    ):
        torch.testing.assert_close(
            Noise2SelfLoss(norm=norm)(pred, target, context={"j_mask": j_mask}), expected
        )


def test_noise2self_accepts_a_complex_prediction() -> None:
    """The mask takes the REAL dtype: clamp_min rejects a complex denominator."""
    pred = torch.randn(1, 1, 8, 8, dtype=torch.complex64)
    target = torch.randn(1, 1, 8, 8, dtype=torch.complex64)
    j_mask = torch.ones(8, 8)
    out = Noise2SelfLoss()(pred, target, context={"j_mask": j_mask})
    torch.testing.assert_close(out, (pred - target).abs().pow(2).mean())


class _Scale(nn.Module):
    def __init__(self, a: float) -> None:
        super().__init__()
        self.a = a

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.a * y


def test_sure_matches_the_closed_form_for_a_linear_denoiser() -> None:
    """f(y) = a y has divergence a per element: SURE = mean(res^2) - s^2 + 2 s^2 a."""
    torch.manual_seed(2)
    sigma, a = 0.3, 0.6
    y = torch.randn(8, 1, 16, 16)
    model = _Scale(a)
    pred = model(y)
    out = SURELoss(sigma=sigma, n_samples=2)(pred, y, context={"model": model, "noisy_input": y})
    closed = (pred - y).pow(2).mean() - sigma**2 + 2 * sigma**2 * a
    torch.testing.assert_close(out, closed, rtol=0, atol=1e-4)
