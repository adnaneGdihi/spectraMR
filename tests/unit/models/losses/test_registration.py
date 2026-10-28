"""Tests for registration losses.

Targets ``spectramr.models.losses.registration.LocalCrossCorrelationLoss``.

Regression: the local variances ``Var = E[X^2] - E[X]^2`` can dip a few ULP
below zero from float round-off. The denominator ``I_var * J_var + eps`` must
not be allowed to flip sign (which would spike the loss / produce NaN on flat
regions), so each variance is clamped to be non-negative.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from spectramr.models.losses.registration import LocalCrossCorrelationLoss  # noqa: E402


def test_lncc_finite_on_constant_input() -> None:
    """A constant image has zero local variance — the clamped denominator must
    keep the loss finite (no NaN/Inf from a sign-flipped variance product)."""
    loss_fn = LocalCrossCorrelationLoss(kernel_size=9)
    img = torch.full((1, 1, 32, 32), 0.5)

    loss = loss_fn(img, img)
    assert torch.isfinite(loss).all()


def test_lncc_bounded_on_random_input() -> None:
    """With non-negative variances, ``cc = cov^2 / (var_I var_J)`` is in [0, 1]
    by Cauchy-Schwarz, so the negated loss stays within roughly [-1, 0]."""
    torch.manual_seed(0)
    pred = torch.rand(2, 1, 32, 32)
    target = torch.rand(2, 1, 32, 32)

    loss = LocalCrossCorrelationLoss(kernel_size=9)(pred, target)
    assert torch.isfinite(loss).all()
    # Small boundary-padding slack on the [-1, 0] range.
    assert -1.05 <= loss.item() <= 0.05


def _reference_single_channel_lncc(pred: torch.Tensor, target: torch.Tensor, k: int) -> torch.Tensor:
    """The float32, single-channel formulation the depthwise version must reproduce."""
    win = torch.ones(1, 1, k, k, dtype=pred.dtype) / (k * k)
    pad = k // 2

    def mean(x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.conv2d(x, win, padding=pad)

    mp, mt = mean(pred), mean(target)
    cross = mean(pred * target) - mp * mt
    vp = (mean(pred * pred) - mp * mp).clamp_min(0.0)
    vt = (mean(target * target) - mt * mt).clamp_min(0.0)
    return -(cross * cross / (vp * vt + 1e-5)).mean()


def test_lncc_single_channel_matches_reference() -> None:
    torch.manual_seed(1)
    pred, target = torch.rand(2, 1, 20, 20), torch.rand(2, 1, 20, 20)
    torch.testing.assert_close(
        LocalCrossCorrelationLoss(kernel_size=7)(pred, target),
        _reference_single_channel_lncc(pred, target, 7),
    )


def test_lncc_accepts_float64_input() -> None:
    """The window used to be float32 regardless of the input, so conv2d raised."""
    pred = torch.rand(1, 1, 16, 16, dtype=torch.float64)
    loss = LocalCrossCorrelationLoss(kernel_size=5)(pred, pred.clone())
    assert loss.dtype == torch.float64
    assert torch.isfinite(loss)


def test_lncc_multichannel_is_the_mean_of_per_channel_lncc() -> None:
    """A (1, 1, k, k) window raised on C=2; each channel is now windowed on its own."""
    torch.manual_seed(2)
    pred, target = torch.rand(2, 2, 16, 16), torch.rand(2, 2, 16, 16)
    loss_fn = LocalCrossCorrelationLoss(kernel_size=5)
    per_channel = torch.stack([loss_fn(pred[:, c : c + 1], target[:, c : c + 1]) for c in range(2)])
    torch.testing.assert_close(loss_fn(pred, target), per_channel.mean())


def test_lncc_tolerates_registry_kwargs() -> None:
    pred = torch.rand(1, 1, 12, 12)
    assert torch.isfinite(LocalCrossCorrelationLoss(kernel_size=3)(pred, pred, mask=None))


def test_smoothness_accepts_the_registry_prediction_target_call() -> None:
    """``loss(prediction, target)`` raised TypeError: forward took the field only."""
    from spectramr.models.losses.registration import SmoothnessLoss

    flow = torch.rand(1, 2, 8, 8)
    loss_fn = SmoothnessLoss()
    torch.testing.assert_close(loss_fn(flow, torch.rand(1, 2, 8, 8)), loss_fn(flow))
