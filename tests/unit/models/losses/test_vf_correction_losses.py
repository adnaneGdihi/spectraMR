"""Unit tests for dual-marker correction losses (vf_correction_losses.py)."""

import pytest
import torch
import torch.nn.functional as F

from spectramr.models.losses import create_loss, list_available

CORRECTION_LOSSES = [
    "gnl_consistency",
    "ghost_suppression",
    "super_resolution_consistency",
]


class TestCorrectionLossRegistration:
    """All correction losses must be in registry."""

    @pytest.mark.parametrize("name", CORRECTION_LOSSES)
    def test_registered(self, name: str) -> None:
        assert name in list_available()

    @pytest.mark.parametrize("name", CORRECTION_LOSSES)
    def test_instantiable(self, name: str) -> None:
        assert create_loss(name) is not None


class TestGNLConsistencyLoss:
    """Tests for GNLConsistencyLoss."""

    def test_zero_when_perfect_match(self) -> None:
        loss_fn = create_loss("gnl_consistency", lambda_gnl=5.0)
        mask = torch.zeros(1, 1, 8, 8)
        mask[:, :, 0:4, 0:4] = 1.0
        prior = torch.ones(1, 1, 8, 8) * 0.5
        pred = torch.randn(1, 1, 8, 8)
        pred[:, :, 0:4, 0:4] = 0.5
        loss = loss_fn(pred, pred, marker_mask=mask, marker_prior=prior)
        assert loss.item() < 1e-6

    def test_nonzero_on_mismatch(self) -> None:
        loss_fn = create_loss("gnl_consistency", lambda_gnl=5.0)
        mask = torch.ones(1, 1, 8, 8)
        prior = torch.zeros(1, 1, 8, 8)
        pred = torch.ones(1, 1, 8, 8)
        loss = loss_fn(pred, pred, marker_mask=mask, marker_prior=prior)
        assert loss.item() > 0.0

    def test_gradient_flow(self) -> None:
        loss_fn = create_loss("gnl_consistency")
        pred = torch.randn(1, 1, 8, 8, requires_grad=True)
        target = torch.randn(1, 1, 8, 8)
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None


class TestGhostSuppressionLoss:
    """Tests for GhostSuppressionLoss."""

    def test_no_ghost_gives_low_loss(self) -> None:
        loss_fn = create_loss("ghost_suppression", lambda_ghost=10.0)
        pred = torch.zeros(1, 1, 16, 16)  # No signal anywhere
        mask = torch.zeros(1, 1, 16, 16)
        mask[:, :, 0:4, 0:4] = 1.0
        loss = loss_fn(pred, pred, marker_mask=mask)
        assert loss.item() < 1e-6

    def test_ghost_gives_high_loss(self) -> None:
        loss_fn = create_loss("ghost_suppression", lambda_ghost=10.0)
        pred = torch.zeros(1, 1, 16, 16)
        mask = torch.zeros(1, 1, 16, 16)
        mask[:, :, 0:4, 0:4] = 1.0
        # Place signal at ghost position (FOV/2 shift)
        pred[:, :, 8:12, 0:4] = 1.0
        loss = loss_fn(pred, pred, marker_mask=mask)
        assert loss.item() > 0.5


class TestSuperResolutionConsistencyLoss:
    """Tests for SuperResolutionConsistencyLoss."""

    def test_perfect_reconstruction(self) -> None:
        """HR that exactly reproduces LR should have zero loss."""
        loss_fn = create_loss("super_resolution_consistency", upscale_factor=2)
        # Create HR and generate LR via downsampling
        hr = torch.randn(1, 1, 16, 16)
        lr = F.avg_pool2d(hr, kernel_size=2, stride=2)
        shifts = torch.zeros(1, 2)
        loss = loss_fn(hr, hr, lr_frames=lr, shifts=shifts)
        assert loss.item() < 0.1

    def test_gradient_flow(self) -> None:
        loss_fn = create_loss("super_resolution_consistency", upscale_factor=2)
        hr = torch.randn(1, 1, 16, 16, requires_grad=True)
        lr = torch.randn(2, 1, 8, 8)
        shifts = torch.zeros(2, 2)
        loss = loss_fn(hr, hr, lr_frames=lr, shifts=shifts)
        loss.backward()
        assert hr.grad is not None


def test_an_integer_hr_shift_samples_whole_pixels() -> None:
    """align_corners=True puts one HR pixel at 2 / (W - 1); 2 / W sampled 0.875 px."""
    from spectramr.models.losses.vf_correction_losses import SuperResolutionConsistencyLoss

    img = torch.arange(64, dtype=torch.float64).view(1, 1, 8, 8)
    shifted = img.roll(-1, dims=-1)
    shifted[..., -1] = img[..., -1]  # border padding repeats the last column
    loss = SuperResolutionConsistencyLoss(upscale_factor=1)(
        img, img, lr_frames=shifted, shifts=torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    )
    assert loss.item() == pytest.approx(0.0, abs=1e-9)
