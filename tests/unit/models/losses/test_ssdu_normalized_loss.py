"""Unit tests for :mod:`spectramr.models.losses.ssdu_normalized_loss`."""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.fft_ops import fft2c
from spectramr.models.losses.ssdu_normalized_loss import SSDUNormalizedLoss

B, H, W = 2, 16, 16


def _case():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(B, 1, H, W, dtype=torch.complex64, generator=g)
    theta = (torch.rand(B, 1, H, W, generator=g) < 0.4).float()
    return x, fft2c(x), theta


def test_zero_at_the_measurement_and_two_at_a_zero_prediction():
    x, y, theta = _case()
    loss = SSDUNormalizedLoss()
    ctx = {"target_kspace": y, "theta_mask": theta}
    assert loss(x, None, context=ctx) == pytest.approx(0.0, abs=1e-5)
    # Predicting zero leaves a residual of -y: each normalised norm is exactly 1.
    assert loss(torch.zeros_like(x), None, context=ctx) == pytest.approx(2.0, abs=1e-5)


def test_it_is_scale_free_per_slice():
    x, y, theta = _case()
    loss = SSDUNormalizedLoss()
    pred = x * 0.9
    a = loss(pred, None, context={"target_kspace": y, "theta_mask": theta})
    b = loss(pred * 1e3, None, context={"target_kspace": y * 1e3, "theta_mask": theta})
    assert a == pytest.approx(float(b), rel=1e-4)


def test_unscored_locations_do_not_count():
    x, y, theta = _case()
    loss = SSDUNormalizedLoss()
    pred = fft2c(x)
    noisy = pred + 5.0 * (1 - theta)  # error only where Theta is zero
    from spectramr.infrastructure.physics.fft_ops import ifft2c

    assert loss(
        ifft2c(noisy), None, context={"target_kspace": y, "theta_mask": theta}
    ) == pytest.approx(0.0, abs=1e-4)


@pytest.mark.parametrize(
    ("context", "match"),
    [
        (None, "requires"),
        (
            {"target_kspace": torch.zeros(1), "theta_mask": torch.zeros(1), "loss_weights": 1},
            "Eq. 35",
        ),
    ],
)
def test_it_refuses_what_it_cannot_score(context, match):
    x, _, _ = _case()
    with pytest.raises(ValueError, match=match):
        SSDUNormalizedLoss()(x, None, context=context)


def test_it_is_registered_under_an_ssdu_name_the_strategy_sums():
    from spectramr.models.losses.registry import LossRegistry

    assert LossRegistry.get_loss_class("ssdu_normalized") is SSDUNormalizedLoss
    assert "ssdu" in "ssdu_normalized"  # SSDUReconstructionStrategy sums names containing it
