"""Tests for ``spectramr.models.losses.latent_losses``."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch import nn

from spectramr.models.losses.latent_losses import (
    LatentRegularizationLoss,
    PerceptualLatentLoss,
)


def _encoder() -> nn.Module:
    enc = nn.Sequential()
    enc.add_module("conv1", nn.Conv2d(1, 2, 3, padding=1))
    enc.add_module("conv2", nn.Conv2d(2, 2, 3, padding=1))
    return enc


def test_perceptual_latent_loss_trains_the_encoder() -> None:
    """The term must be a graph-attached tensor, not a ``.item()``-rebuilt float."""
    torch.manual_seed(0)
    enc = _encoder()
    loss_fn = PerceptualLatentLoss(enc, layers=("conv1", "conv2"))
    x1 = torch.randn(2, 1, 8, 8, requires_grad=True)
    x2 = torch.randn(2, 1, 8, 8)
    loss = loss_fn(x1, x2)

    assert isinstance(loss, torch.Tensor)
    loss.backward()
    assert x1.grad is not None and float(x1.grad.abs().sum()) > 0
    assert all(p.grad is not None for p in enc.parameters())


def test_perceptual_latent_loss_value_is_the_weighted_feature_mse() -> None:
    torch.manual_seed(0)
    enc = _encoder()
    loss_fn = PerceptualLatentLoss(enc, layers=("conv1",), weights=(0.3,))
    x1, x2 = torch.randn(2, 1, 8, 8), torch.randn(2, 1, 8, 8)
    expected = 0.3 * nn.functional.mse_loss(enc.conv1(x1), enc.conv1(x2))
    torch.testing.assert_close(loss_fn(x1, x2), expected)


def test_perceptual_latent_loss_rejects_a_layer_the_encoder_lacks() -> None:
    with pytest.raises(ValueError, match="conv9"):
        PerceptualLatentLoss(_encoder(), layers=("conv1", "conv9"))


def test_latent_regularization_rejects_an_unknown_type() -> None:
    """An unrecognised type used to return a constant zero with no gradient."""
    with pytest.raises(ValueError, match="regularization_type"):
        LatentRegularizationLoss(regularization_type="bogus")


def _masked_diversity_reference(latent: torch.Tensor, weight: float) -> torch.Tensor:
    flat = latent.reshape(latent.size(0), -1)
    distances = torch.cdist(flat, flat, p=2)
    mask = torch.eye(distances.size(0), dtype=torch.bool)
    off = distances[~mask].view(distances.size(0), -1)
    return weight * torch.mean(1.0 / (off + 1e-8))


def test_diversity_matches_the_masked_reference() -> None:
    torch.manual_seed(0)
    latent = torch.randn(5, 3, 4, 4, requires_grad=True)
    loss = LatentRegularizationLoss("diversity", weight=0.5)(latent)
    ref_latent = latent.detach().clone().requires_grad_(True)
    ref = _masked_diversity_reference(ref_latent, 0.5)
    torch.testing.assert_close(loss, ref)
    loss.backward()
    ref.backward()
    torch.testing.assert_close(latent.grad, ref_latent.grad)


def test_diversity_needs_two_latents() -> None:
    """A single latent has no pairs; the mean over none used to be NaN."""
    with pytest.raises(ValueError, match="at least 2"):
        LatentRegularizationLoss("diversity")(torch.randn(1, 8))


@pytest.mark.parametrize(
    ("kind", "expected"),
    [("l2", lambda z: z.pow(2).mean()), ("sparsity", lambda z: z.abs().mean())],
)
def test_elementwise_regularizers(
    kind: str, expected: Callable[[torch.Tensor], torch.Tensor]
) -> None:
    torch.manual_seed(0)
    z = torch.randn(4, 6)
    out = LatentRegularizationLoss(kind, weight=2.0)(z)
    torch.testing.assert_close(out, 2.0 * expected(z))
