"""SSDU and multi-mask SSDU: Θ-restricted k-space residual, mask broadcasting, pairing."""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.fft_ops import fft2c
from spectramr.models.losses.ssdu_loss import MultiMaskSSDULoss, SSDULoss


def _case(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    img = torch.randn(3, 1, 8, 8, dtype=torch.complex64)
    y = fft2c(torch.randn(3, 1, 8, 8, dtype=torch.complex64))
    theta = (torch.rand(3, 1, 8, 8) > 0.6).float()
    return img, y, theta


def test_ssdu_is_zero_when_the_prediction_reproduces_the_held_out_bins() -> None:
    img, _, theta = _case()
    y = fft2c(img)
    assert SSDULoss()(img, img, context={"target_kspace": y, "theta_mask": theta}).item() == 0.0


def test_ssdu_matches_the_masked_residual() -> None:
    img, y, theta = _case(1)
    expected = torch.view_as_real((fft2c(img) - y) * theta).pow(2).mean()
    out = SSDULoss()(img, img, context={"target_kspace": y, "theta_mask": theta})
    torch.testing.assert_close(out, expected)


def test_a_per_sample_mask_without_a_channel_axis_stays_on_the_batch_axis() -> None:
    """Padding [B, H, W] at the front would broadcast into a [B, B, H, W] cross-sample loss."""
    img, y, theta = _case(2)
    loss = SSDULoss()
    with_axis = loss(img, img, context={"target_kspace": y, "theta_mask": theta})
    without = loss(img, img, context={"target_kspace": y, "theta_mask": theta[:, 0]})
    torch.testing.assert_close(without, with_axis)


def test_multi_mask_prerolled_is_the_mean_of_single_splits() -> None:
    img, y, _ = _case(3)
    masks = [(torch.rand(3, 1, 8, 8) > 0.6).float() for _ in range(2)]
    preds = torch.stack([img, img * 0.5])
    out = MultiMaskSSDULoss()(preds, img, context={"target_kspace": y, "theta_masks": masks})
    single = SSDULoss()
    expected = (
        sum(
            single(p, img, context={"target_kspace": y, "theta_mask": m})
            for p, m in zip(preds, masks, strict=True)
        )
        / 2
    )
    torch.testing.assert_close(out, expected)


def test_multi_mask_rejects_a_prerolled_stack_it_cannot_pair() -> None:
    """A 5-D stack whose length differs from the mask list must raise, not broadcast."""
    img, y, theta = _case(4)
    preds = torch.stack([img, img, img])
    with pytest.raises(ValueError, match="pair one-to-one"):
        MultiMaskSSDULoss()(preds, img, context={"target_kspace": y, "theta_masks": [theta, theta]})


def test_multi_mask_refuses_robust_weights_it_would_drop() -> None:
    img = torch.randn(1, 2, 8, 8)
    with pytest.raises(ValueError, match="does not apply"):
        MultiMaskSSDULoss()(
            img,
            img,
            context={
                "target_kspace": img,
                "theta_masks": [torch.ones(1, 1, 8, 8)],
                "loss_weights": torch.ones(1, 1, 8, 8),
            },
        )


def test_missing_context_raises() -> None:
    img, _, _ = _case()
    with pytest.raises(ValueError):
        SSDULoss()(img, img)
    with pytest.raises(ValueError):
        MultiMaskSSDULoss()(img, img, context={"target_kspace": img})


# --- #2327: the loss applies F S, and Robust SSDU's weights -----------------


def _maps(batch: int = 2, coils: int = 4, size: int = 8) -> torch.Tensor:
    torch.manual_seed(7)
    maps = torch.randn(batch, coils, size, size, dtype=torch.complex64)
    return maps / maps.abs().pow(2).sum(1, keepdim=True).sqrt()


def test_with_maps_the_loss_scores_f_s_x_against_every_coil() -> None:
    from spectramr.infrastructure.physics.fft_ops import sense_forward

    maps = _maps()
    img = torch.randn(2, 1, 8, 8, dtype=torch.complex64)
    y = sense_forward(img, maps, None)
    theta = (torch.rand(2, 1, 8, 8) > 0.5).float()
    ctx = {"target_kspace": y, "theta_mask": theta, "coil_sensitivities": maps}
    assert SSDULoss()(img, img, context=ctx).item() == 0.0


def test_without_maps_a_single_image_cannot_match_multicoil_y() -> None:
    """The planted blocker: fft2c(pred) against 4-coil y raised, and must keep raising."""
    y = torch.randn(2, 4, 8, 8, dtype=torch.complex64)
    img = torch.randn(2, 1, 8, 8, dtype=torch.complex64)
    with pytest.raises(ValueError, match="k-space shape"):
        SSDULoss()(img, img, context={"target_kspace": y, "theta_mask": torch.ones(2, 1, 8, 8)})


def test_weights_multiply_the_residual_inside_the_norm() -> None:
    img, y, theta = _case(5)
    weights = torch.full_like(theta, 3.0)
    plain = SSDULoss()(img, img, context={"target_kspace": y, "theta_mask": theta})
    weighted = SSDULoss()(
        img, img, context={"target_kspace": y, "theta_mask": theta, "loss_weights": weights}
    )
    torch.testing.assert_close(weighted, 9.0 * plain)


def test_the_coil_operator_refuses_a_real_prediction() -> None:
    maps = _maps()
    with pytest.raises(ValueError, match="complex image prediction"):
        SSDULoss()(
            torch.randn(2, 2, 8, 8),
            None,
            context={
                "target_kspace": torch.randn(2, 4, 8, 8, dtype=torch.complex64),
                "theta_mask": torch.ones(2, 1, 8, 8),
                "coil_sensitivities": maps,
            },
        )
