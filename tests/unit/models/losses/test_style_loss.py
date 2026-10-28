"""``GramStyleLoss`` (``models/losses/style_loss.py``)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from spectramr.models.losses.style_loss import _gram


def test_gram_of_half_precision_features_is_finite_and_matches_float32() -> None:
    """Each entry sums H*W products: on a 128x128 map of O(1) activations that is
    far past float16's 65504, and the old float16 ``bmm`` returned inf."""
    torch.manual_seed(0)
    features = torch.rand(1, 64, 128, 128) * 4.0
    got = _gram(features.half())
    assert got.dtype == torch.float32
    assert torch.isfinite(got).all()
    torch.testing.assert_close(got, _gram(features), rtol=2e-3, atol=1e-4)


def test_gram_is_normalised_by_channels_and_pixels() -> None:
    features = torch.ones(1, 2, 3, 4)
    torch.testing.assert_close(_gram(features), torch.full((1, 2, 2), 12.0 / 24.0))


@pytest.fixture
def style_loss(monkeypatch):
    import torchvision

    torch.manual_seed(0)
    layers = [nn.Conv2d(3, 3, 3, padding=1) for _ in range(37)]
    monkeypatch.setattr(
        torchvision.models, "vgg19", lambda **_: SimpleNamespace(features=nn.Sequential(*layers))
    )
    from spectramr.models.losses.style_loss import GramStyleLoss

    return GramStyleLoss(layers=["relu1_1", "relu2_1"])


def test_identical_images_have_zero_style_loss(style_loss) -> None:
    x = torch.rand(1, 1, 16, 16)
    torch.testing.assert_close(style_loss(x, x.clone()), torch.tensor(0.0))


def test_style_loss_backpropagates_to_the_prediction_only(style_loss) -> None:
    x = torch.rand(1, 1, 16, 16, requires_grad=True)
    y = torch.rand(1, 1, 16, 16, requires_grad=True)
    style_loss(x, y).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert y.grad is None
