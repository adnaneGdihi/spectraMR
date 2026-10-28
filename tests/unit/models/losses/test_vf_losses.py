"""Unit tests for VF losses (vf_losses.py) — registration and gradient flow."""

import pytest
import torch

from spectramr.models.losses import create_loss, list_available

VF_LOSSES = [
    "zero_shot_denoising",
    "marker_masked_discriminator",
    "marker_conditioned_vae",
    "marker_prior",
    "diffusion_variance_calibration",
]


class TestVFLossRegistration:
    """All VF losses must be discoverable via the loss registry."""

    @pytest.mark.parametrize("name", VF_LOSSES)
    def test_registered(self, name: str) -> None:
        avail = list_available()
        assert name in avail, f"{name} not found in loss registry"

    @pytest.mark.parametrize("name", VF_LOSSES)
    def test_instantiable(self, name: str) -> None:
        loss_fn = create_loss(name)
        assert loss_fn is not None


class TestZeroShotDenoisingLoss:
    """Tests for ZeroShotDenoisingLoss."""

    def test_without_marker(self) -> None:
        loss_fn = create_loss("zero_shot_denoising")
        pred = torch.randn(2, 1, 8, 8, requires_grad=True)
        target = torch.randn(2, 1, 8, 8)
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None

    def test_with_marker(self) -> None:
        loss_fn = create_loss("zero_shot_denoising", lambda_marker=5.0)
        pred = torch.randn(2, 1, 8, 8, requires_grad=True)
        target = torch.randn(2, 1, 8, 8)
        mask = torch.zeros(1, 1, 8, 8)
        mask[:, :, 0:2, 0:2] = 1.0
        prior = torch.ones(1, 1, 8, 8) * 0.5
        loss = loss_fn(pred, target, marker_mask=mask, marker_prior=prior)
        loss.backward()
        assert pred.grad is not None


class TestMarkerPriorLoss:
    """Tests for MarkerPriorLoss."""

    def test_zero_when_perfect(self) -> None:
        loss_fn = create_loss("marker_prior", lambda_prior=10.0)
        mask = torch.zeros(1, 1, 8, 8)
        mask[:, :, 0:4, 0:4] = 1.0
        prior = torch.ones(1, 1, 8, 8) * 0.5
        # Prediction matches prior in marker region
        pred = torch.randn(1, 1, 8, 8)
        pred[:, :, 0:4, 0:4] = 0.5
        loss = loss_fn(pred, pred, marker_mask=mask, marker_prior=prior)
        assert loss.item() < 1e-6

    def test_nonzero_when_mismatch(self) -> None:
        loss_fn = create_loss("marker_prior", lambda_prior=10.0)
        mask = torch.ones(1, 1, 8, 8)
        prior = torch.zeros(1, 1, 8, 8)
        pred = torch.ones(1, 1, 8, 8)
        loss = loss_fn(pred, pred, marker_mask=mask, marker_prior=prior)
        assert loss.item() > 0.0


class TestMarkerConditionedVAELoss:
    """Tests for MarkerConditionedVAELoss."""

    def test_gradient_flow(self) -> None:
        loss_fn = create_loss("marker_conditioned_vae", beta=1.0, lambda_marker=5.0)
        pred = torch.randn(1, 1, 8, 8, requires_grad=True)
        target = torch.randn(1, 1, 8, 8)
        mu = torch.randn(1, 16, requires_grad=True)
        logvar = torch.randn(1, 16, requires_grad=True)
        mask = torch.ones(1, 1, 8, 8)
        prior = torch.zeros(1, 1, 8, 8)
        loss = loss_fn(
            pred, target, mu=mu, logvar=logvar, marker_mask=mask, marker_prior=prior
        )
        loss.backward()
        assert pred.grad is not None
        assert mu.grad is not None


class TestDiffusionVarianceCalibrationLoss:
    """The measured marker variance is a calibration target, not a trainable term."""

    def _inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        target = torch.ones(1, 1, 4, 4)
        pred = (target + 0.1).requires_grad_(True)
        est = torch.tensor([0.5], requires_grad=True)
        return pred, target, est, torch.ones(1, 1, 4, 4)

    def test_calibration_term_sends_no_gradient_to_the_prediction(self) -> None:
        """With est > measured it pushed the marker residual up, away from the target."""
        from spectramr.models.losses.vf_losses import DiffusionVarianceCalibrationLoss

        pred, target, est, mask = self._inputs()
        loss = DiffusionVarianceCalibrationLoss(lambda_cal=1.0)(
            pred, target, estimated_noise_var=est, marker_mask=mask
        )
        loss.backward()
        # Only the L1 term reaches the prediction: sign(0.1) / 16 per pixel.
        assert pred.grad is not None
        torch.testing.assert_close(pred.grad, torch.full_like(pred, 1.0 / 16.0))

    def test_calibration_term_trains_the_estimate(self) -> None:
        from spectramr.models.losses.vf_losses import DiffusionVarianceCalibrationLoss

        pred, target, est, mask = self._inputs()
        loss = DiffusionVarianceCalibrationLoss(lambda_cal=1.0)(
            pred, target, estimated_noise_var=est, marker_mask=mask
        )
        loss.backward()
        # d/d est of (est - 0.01)^2 at est = 0.5.
        assert est.grad is not None
        torch.testing.assert_close(est.grad, torch.tensor([2.0 * (0.5 - 0.01)]))
