import torch

from spectramr.infrastructure.physics.data_consistency import SoftDataConsistency


class TestSoftDataConsistency:
    def test_initialization(self):
        sdc = SoftDataConsistency(lambda_init=5.0)
        assert abs(sdc.lam.item() - 5.0) < 1e-5
        assert sdc.lambda_param.requires_grad is True

    def test_forward_shapes(self):
        B, C, H, W = 2, 1, 32, 32
        sdc = SoftDataConsistency(lambda_init=1.0)

        # Real image input (B, C, H, W)
        pred = torch.randn(B, C, H, W)
        # Complex kspace (B, C, H, W)
        k_meas = torch.randn(B, C, H, W, dtype=torch.complex64)
        # Mask (B, 1, H, W)
        mask = torch.ones(B, 1, H, W)

        out = sdc(pred, k_meas, mask)
        assert out.shape == pred.shape
        assert not torch.is_complex(out)

    def test_soft_behavior(self):
        """Test that output INTERPOLATES between pred and measurement."""
        B, C, H, W = 1, 1, 4, 4
        sdc = SoftDataConsistency(lambda_init=1.0)

        # Pred: all zeros
        pred = torch.zeros(B, C, H, W)

        # Measured: all ones (in k-space means spike at DC, but let's just make kspace ones)
        # If we just put ones in kspace
        k_meas = torch.ones(B, C, H, W, dtype=torch.complex64)

        # Mask: all ones
        mask = torch.ones(B, 1, H, W)

        # Formula: k_out = (k_pred + lambda*k_meas) / (1 + lambda)
        # k_pred = 0
        # k_out = (0 + 1*1) / (1 + 1) = 0.5

        # So output image should be IFFT(0.5)
        # IFFT of 0.5 everywhere -> Impulse at 0,0?
        # Actually if kspace is 0.5 everywhere, image is delta?

        # Let's check kspace output directly if possible or check image energy
        # k_out = 0.5

        # But we can't easily check internal K.
        # Let's use logic:
        # If lambda -> infinity, out -> measurement
        # If lambda -> 0, out -> prediction

        # Case 1: lambda = 1e9 (Hard)
        sdc_hard = SoftDataConsistency(lambda_init=1e9)
        out_hard = sdc_hard(pred, k_meas, mask)

        # Should be ifft2c(k_meas) because SoftDC does proper centered FFT/IFFT
        # We need to use the SAME inverse transform as the class uses
        from spectramr.infrastructure.physics.fft_ops import ifft2c

        expected_hard = ifft2c(k_meas).abs()

        assert torch.allclose(out_hard, expected_hard, atol=1e-5)

        # Case 2: lambda = 1e-9 (Identity)
        sdc_identity = SoftDataConsistency(lambda_init=1e-9)
        out_identity = sdc_identity(pred, k_meas, mask)
        assert torch.allclose(out_identity, pred.abs(), atol=1e-5)

    def test_gradients(self):
        """Test that gradients flow through lambda."""
        sdc = SoftDataConsistency(lambda_init=1.0, learnable=True)
        pred = torch.randn(1, 1, 16, 16, requires_grad=True)
        k_meas = torch.randn(1, 1, 16, 16, dtype=torch.complex64)
        mask = torch.ones(1, 1, 16, 16)

        out = sdc(pred, k_meas, mask)
        loss = out.mean()
        loss.backward()

        assert sdc.lambda_param.grad is not None
        assert pred.grad is not None


    def test_lambda_gradient_survives_a_tiny_weight(self):
        """Planted: the old clamp(min=1e-6) zeroed the gradient once lambda went negative."""
        sdc = SoftDataConsistency(lambda_init=1e-7)
        k_pred = torch.randn(1, 2, 8, 8)
        k_meas = torch.randn(1, 2, 8, 8)
        mask = torch.ones(1, 1, 8, 8)
        sdc(k_pred, k_meas, mask, is_kspace_domain=True).sum().backward()
        assert sdc.lambda_param.grad is not None
        assert sdc.lambda_param.grad.abs() > 0
        assert sdc.lam.item() > 0

    def test_log_scaled_blend_is_computed_on_linear_kspace(self):
        """The proximal blend of decompressed values, recompressed -- not a blend of logs."""
        from spectramr.data.transforms.normalization import (
            compress_kspace_log,
            decompress_kspace_log,
        )

        torch.manual_seed(0)
        lin_pred = torch.randn(1, 2, 8, 8) * 30
        lin_meas = torch.randn(1, 2, 8, 8) * 30
        mask = torch.ones(1, 1, 8, 8)
        sdc = SoftDataConsistency(lambda_init=1.0, learnable=False, log_scaled=True)
        out = sdc(
            compress_kspace_log(lin_pred, channel_dim=1),
            compress_kspace_log(lin_meas, channel_dim=1),
            mask,
            is_kspace_domain=True,
        )
        expected = (lin_pred + lin_meas) / 2.0
        torch.testing.assert_close(
            decompress_kspace_log(out, channel_dim=1), expected, rtol=1e-3, atol=1e-3
        )

    def test_log_scaled_must_be_a_bool(self):
        import pytest

        with pytest.raises(TypeError, match="log_scaled"):
            SoftDataConsistency(log_scaled=None)
