"""Unit tests for FocalFrequencyLoss.

Tests the focal frequency loss that up-weights k-space periphery
errors for Gibbs ringing correction and super-resolution.
"""

import pytest
import torch

from spectramr.models.losses.focal_frequency_loss import (
    FocalFrequencyLoss,
    _build_radial_weight,
)
from spectramr.models.losses.registry import create_loss


class TestBuildRadialWeight:
    """Tests for the radial weight matrix builder."""

    def test_output_shape(self):
        """Weight matrix should be (1, 1, H, W)."""
        w = _build_radial_weight(32, 32, alpha=2.0)
        assert w.shape == (1, 1, 32, 32)

    def test_center_near_zero(self):
        """DC component (center) should have near-zero weight."""
        w = _build_radial_weight(64, 64, alpha=2.0)
        center_val = w[0, 0, 32, 32].item()
        assert center_val < 0.01

    def test_periphery_near_one(self):
        """Corner frequencies should have near-maximum weight."""
        w = _build_radial_weight(64, 64, alpha=1.0)
        # Corner at (0, 0) → r ≈ √2 clamped to 1.0
        corner_val = w[0, 0, 0, 0].item()
        assert corner_val > 0.9

    def test_alpha_zero_uniform(self):
        """alpha=0 should produce uniform weights (all near 1)."""
        w = _build_radial_weight(16, 16, alpha=0.0)
        # r^0 = 1 for r > 0, but center is 0^0 = 1 as well
        assert w.min() >= 0.0
        assert w.max() <= 1.0


class TestFocalFrequencyLoss:
    """Tests for the Focal Frequency Loss module."""

    def test_identical_inputs_zero_loss(self):
        """Loss should be zero for identical pred and target."""
        ffl = FocalFrequencyLoss(alpha=2.0)
        x = torch.randn(1, 1, 32, 32)
        loss = ffl(x, x)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_gradient_flows(self):
        """Gradients should flow through the loss."""
        ffl = FocalFrequencyLoss(alpha=2.0)
        pred = torch.randn(1, 1, 32, 32, requires_grad=True)
        target = torch.randn(1, 1, 32, 32)

        loss = ffl(pred, target)
        loss.backward()

        assert pred.grad is not None
        assert pred.grad.abs().sum() > 0

    def test_peripheral_errors_produce_higher_loss(self):
        """Errors at k-space periphery should produce higher loss than central errors."""
        ffl = FocalFrequencyLoss(alpha=4.0)  # Strong peripheral weighting

        target = torch.zeros(1, 1, 64, 64)

        # Central error (low frequency)
        pred_center = torch.zeros(1, 1, 64, 64)
        pred_center[:, :, 30:34, 30:34] = 1.0

        # Peripheral error (high frequency — fine checkerboard)
        pred_edge = torch.zeros(1, 1, 64, 64)
        pred_edge[:, :, 0::2, 0::2] = 0.01

        loss_center = ffl(pred_center, target)
        loss_edge = ffl(pred_edge, target)

        # Both should produce non-zero loss
        assert loss_center.item() > 0
        assert loss_edge.item() > 0

    def test_complex_input_support(self):
        """Complex-valued inputs should work."""
        ffl = FocalFrequencyLoss(alpha=2.0)
        pred = torch.complex(torch.randn(1, 1, 16, 16), torch.randn(1, 1, 16, 16))
        target = torch.complex(torch.randn(1, 1, 16, 16), torch.randn(1, 1, 16, 16))
        loss = ffl(pred, target)
        assert loss.item() > 0

    def test_2channel_input_support(self):
        """2-channel real/imag inputs should work."""
        ffl = FocalFrequencyLoss(alpha=2.0)
        pred = torch.randn(1, 2, 16, 16)
        target = torch.randn(1, 2, 16, 16)
        loss = ffl(pred, target)
        assert loss.item() > 0

    def test_registry_resolution(self):
        """FFL should be resolvable via create_loss."""
        loss_fn = create_loss("focal_frequency_radial")
        assert isinstance(loss_fn, FocalFrequencyLoss)

    def test_registry_alias_resolution(self):
        """FFL aliases should resolve correctly."""
        for alias in ["ffl_radial", "RadialFocalFrequencyLoss"]:
            loss_fn = create_loss(alias)
            assert isinstance(loss_fn, FocalFrequencyLoss)

    def test_weight_caching(self):
        """Weight matrix should be cached for same spatial size."""
        ffl = FocalFrequencyLoss(alpha=2.0)
        pred = torch.randn(1, 1, 32, 32)
        target = torch.randn(1, 1, 32, 32)

        # First call builds cache
        ffl(pred, target)
        assert ffl._cached_size == (32, 32)

        # Second call should reuse cache
        ffl(pred, target)
        assert ffl._cached_size == (32, 32)


def test_unknown_weight_mode_raises() -> None:
    """A misspelt mode used to fall through to ``radial`` without a word."""
    with pytest.raises(ValueError, match="parabolic"):
        FocalFrequencyLoss(mode="parabollic")
