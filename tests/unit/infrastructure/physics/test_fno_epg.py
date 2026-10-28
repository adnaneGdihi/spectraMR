"""Unit tests for FNO EPG Surrogate."""

import pytest
import torch

from spectramr.infrastructure.physics.fno_epg import FNOEPGSurrogate, SpectralConv1d


class TestSpectralConv1d:
    """Tests for 1D spectral convolution."""

    def test_output_shape(self):
        """Output shape should match expected dimensions."""
        conv = SpectralConv1d(in_channels=4, out_channels=8, modes=4)
        x = torch.randn(2, 4, 16)  # [B, C, N]
        out = conv(x)
        assert out.shape == (2, 8, 16)

    def test_no_nans(self):
        """Output should not contain NaN values."""
        conv = SpectralConv1d(in_channels=4, out_channels=4, modes=8)
        x = torch.randn(4, 4, 32)
        out = conv(x)
        assert not torch.isnan(out).any()


class TestFNOEPGSurrogate:
    """Tests for FNO EPG surrogate model."""

    @pytest.fixture
    def model(self):
        """Create FNO EPG Surrogate (seeded: an untrained net's B1+ sensitivity is ~1e-4)."""
        torch.manual_seed(0)
        return FNOEPGSurrogate(
            tissue_channels=4,
            hidden_width=16,
            n_layers=2,
            max_etl=16,
        )

    def test_output_shape(self, model):
        """Output should be [B, 1, H, W]."""
        B, H, W = 2, 32, 32
        rho = torch.rand(B, 1, H, W)
        t1 = torch.rand(B, 1, H, W) * 4000 + 500
        t2 = torch.rand(B, 1, H, W) * 200 + 50
        b1 = torch.ones(B, 1, H, W)

        signal = model(rho, t1, t2, b1, echo_train_length=16)
        assert signal.shape == (B, 1, H, W)

    def test_no_nans(self, model):
        """Output should not contain NaN values."""
        rho = torch.rand(1, 1, 16, 16)
        t1 = torch.rand(1, 1, 16, 16) * 2000 + 500
        t2 = torch.rand(1, 1, 16, 16) * 100 + 30

        signal = model(rho, t1, t2, echo_train_length=16)
        assert not torch.isnan(signal).any()
        assert not torch.isinf(signal).any()

    def test_non_negative_output(self, model):
        """Echo amplitudes should be non-negative (Softplus output)."""
        rho = torch.rand(2, 1, 16, 16)
        t1 = torch.rand(2, 1, 16, 16) * 2000 + 500
        t2 = torch.rand(2, 1, 16, 16) * 100 + 30

        signal = model(rho, t1, t2, echo_train_length=16)
        assert (signal >= 0).all()

    def test_gradient_flow(self, model):
        """Gradients should flow back through the model."""
        rho = torch.rand(1, 1, 8, 8, requires_grad=True)
        t1 = torch.rand(1, 1, 8, 8) * 2000 + 500
        t2 = torch.rand(1, 1, 8, 8) * 100 + 30

        signal = model(rho, t1, t2, echo_train_length=16)
        loss = signal.sum()
        loss.backward()

        assert rho.grad is not None
        assert not torch.isnan(rho.grad).any()

    def test_backward_through_a_predicted_b1_plus_map(self, model):
        """A B1+ map that requires grad must backpropagate (it is the network's own output).

        Writing the B1+-modulated flip angles into the sequence vector in place edited a
        tensor autograd had saved for the product, so backward raised on the first step.
        """
        rho = (torch.rand(2, 1, 4, 4) + 0.2).requires_grad_(True)
        t1 = (torch.full((2, 1, 4, 4), 1000.0)).requires_grad_(True)
        t2 = (torch.full((2, 1, 4, 4), 80.0)).requires_grad_(True)
        b1 = (torch.rand(2, 1, 4, 4) * 0.4 + 0.8).requires_grad_(True)

        model(rho, t1, t2, b1, echo_train_length=16).sum().backward()

        assert b1.grad is not None and torch.isfinite(b1.grad).all()
        assert b1.grad.abs().sum() > 0

    def test_b1_plus_modulation(self, model):
        """B1+ should affect the output (different B1+ → different signal)."""
        rho = torch.ones(2, 1, 8, 8) * 0.8
        t1 = torch.ones(2, 1, 8, 8) * 1000
        t2 = torch.ones(2, 1, 8, 8) * 100

        b1_low = torch.ones(2, 1, 8, 8) * 0.5
        b1_high = torch.ones(2, 1, 8, 8) * 1.5

        signal_low = model(rho, t1, t2, b1_low, echo_train_length=16)
        signal_high = model(rho, t1, t2, b1_high, echo_train_length=16)

        # Different B1+ should produce different signals
        assert not torch.allclose(signal_low, signal_high, atol=1e-4)
