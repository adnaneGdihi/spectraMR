"""Equivalence of ``PMAGlobalLoss``'s batched form with its per-contrast definition."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from spectramr.models.losses.m4_losses import PMAGlobalLoss
from spectramr.models.losses.ssim_loss import SSIMLoss


def _per_contrast_reference(
    loss: PMAGlobalLoss,
    prediction: torch.Tensor,
    target: torch.Tensor,
    marker_mask: torch.Tensor | None,
    marker_prior: torch.Tensor | None,
) -> torch.Tensor:
    """The documented ``sum_c (L1_c + l_s (1 - SSIM_c) + l_m ||M (x_c - m_c)||^2)``."""
    pred_mag = prediction.abs() if prediction.is_complex() else prediction
    targ_mag = target.abs() if target.is_complex() else target
    data_range = float(targ_mag.max())
    ssim = SSIMLoss()
    total = torch.zeros((), dtype=pred_mag.dtype)
    for c in range(prediction.shape[1]):
        p_c, t_c = prediction[:, c : c + 1], target[:, c : c + 1]
        l1 = (p_c - t_c).abs().mean()
        l_ssim = ssim(pred_mag[:, c : c + 1], targ_mag[:, c : c + 1], data_range=data_range)
        l_marker = torch.zeros((), dtype=pred_mag.dtype)
        if marker_mask is not None and marker_prior is not None:
            prior_c = marker_prior[:, c : c + 1] if marker_prior.shape[1] > 1 else marker_prior
            l_marker = (p_c * marker_mask - prior_c * marker_mask).abs().pow(2).mean()
        total = total + loss.lambda_l1 * l1 + loss.lambda_ssim * l_ssim
        total = total + loss.lambda_marker * l_marker
    return total


@pytest.mark.parametrize("with_marker", [False, True])
@pytest.mark.parametrize("shared_prior", [False, True])
def test_batched_loss_matches_the_per_contrast_sum(with_marker: bool, shared_prior: bool) -> None:
    torch.manual_seed(0)
    prediction = torch.rand(2, 3, 24, 24, dtype=torch.float64, requires_grad=True)
    target = torch.rand(2, 3, 24, 24, dtype=torch.float64)
    mask = (torch.rand(1, 1, 24, 24) > 0.7).double() if with_marker else None
    prior_channels = 1 if shared_prior else 3
    prior = torch.rand(2, prior_channels, 24, 24, dtype=torch.float64) if with_marker else None
    loss = PMAGlobalLoss(lambda_l1=1.0, lambda_ssim=0.5, lambda_marker=5.0)

    out = loss(prediction, target, marker_mask=mask, marker_prior=prior)
    ref = _per_contrast_reference(loss, prediction, target, mask, prior)
    torch.testing.assert_close(out, ref)
    (grad,) = torch.autograd.grad(out, prediction)
    (ref_grad,) = torch.autograd.grad(ref, prediction)
    torch.testing.assert_close(grad, ref_grad)


def test_complex_prediction_matches_the_per_contrast_sum() -> None:
    torch.manual_seed(1)
    prediction = torch.randn(2, 2, 20, 20, dtype=torch.complex128)
    target = torch.randn(2, 2, 20, 20, dtype=torch.complex128)
    loss = PMAGlobalLoss()
    torch.testing.assert_close(
        loss(prediction, target), _per_contrast_reference(loss, prediction, target, None, None)
    )


def test_l1_term_equals_mean_absolute_error_for_real_input() -> None:
    torch.manual_seed(2)
    p, t = torch.rand(1, 2, 16, 16), torch.rand(1, 2, 16, 16)
    loss = PMAGlobalLoss(lambda_l1=1.0, lambda_ssim=0.0, lambda_marker=0.0)
    torch.testing.assert_close(loss(p, t), 2 * F.l1_loss(p, t))
