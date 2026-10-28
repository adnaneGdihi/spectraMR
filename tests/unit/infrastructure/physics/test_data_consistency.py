"""Unit tests for data consistency layers."""

import math

import pytest
import torch

from spectramr.infrastructure.physics.data_consistency import (
    AdaptiveDataConsistency,
    NoiseAdaptiveDataConsistency,
    NoiseAdjustedDataConsistency,
    NoiseSimulatingDataConsistency,
    SimpleDataConsistency,
    data_consistency,
)
from spectramr.infrastructure.physics.fft_ops import fft2c


class TestSimpleDataConsistencyMethod:
    """The ``method`` knob ('hard' vs 'soft') must actually change the output.

    ``self.method`` was stored in ``__init__`` but never read in ``forward`` —
    both settings produced the identical soft proximal blend
    ``(x + λ·measured)/(1+λ)``, so ``method='hard'`` was a silent no-op
    (CLAUDE.md pitfall #15). Hard DC must replace the prediction with the
    measurement *exactly* at sampled k-space locations.
    """

    @staticmethod
    def _inputs():
        torch.manual_seed(0)
        pred = torch.randn(1, 2, 8, 8)  # k-space domain, real-interleaved (even C)
        measured = torch.randn(1, 2, 8, 8)
        mask = torch.zeros(1, 1, 8, 8)
        mask[..., :4] = 1.0  # sample the left half of k-space
        return pred, measured, mask

    def test_hard_replaces_measurement_at_sampled_locations(self):
        pred, measured, mask = self._inputs()
        out = SimpleDataConsistency(weight=1.0, method="hard")(
            pred, measured_kspace=measured, mask=mask, is_kspace_domain=True
        )
        m = mask.float()
        # At sampled locations: output == measurement exactly (hard replacement).
        assert torch.allclose(out * m, measured * m, atol=1e-6)
        # At unsampled locations: prediction is preserved.
        assert torch.allclose(out * (1 - m), pred * (1 - m), atol=1e-6)

    def test_hard_and_soft_outputs_differ(self):
        pred, measured, mask = self._inputs()
        hard = SimpleDataConsistency(weight=1.0, method="hard")(
            pred, measured_kspace=measured, mask=mask, is_kspace_domain=True
        )
        soft = SimpleDataConsistency(weight=1.0, method="soft")(
            pred, measured_kspace=measured, mask=mask, is_kspace_domain=True
        )
        assert not torch.allclose(hard, soft), "method knob is inert (hard == soft)"


class TestDataConsistencyLayerNoiseTypeValidation:
    """Regression: noise_type knob must be validated, not silently no-op'd.

    Previously NoiseSimulatingDataConsistency accepted any noise_type string but
    ``_add_realistic_noise`` only branched on 'gaussian'; any other value
    (e.g. 'rician', or the typo 'guassian') silently added ZERO noise
    (CLAUDE.md pitfalls #9/#15). The fix validates the knob in __init__.
    """

    def test_rejects_unsupported_noise_type(self):
        """An advertised-but-unimplemented noise model must RAISE at construction."""
        with pytest.raises(ValueError, match="unsupported noise_type"):
            NoiseSimulatingDataConsistency(noise_type="rician")

    def test_rejects_typo_noise_type(self):
        """A typo in noise_type must RAISE, not silently disable noise."""
        with pytest.raises(ValueError, match="unsupported noise_type"):
            NoiseSimulatingDataConsistency(noise_type="guassian")  # typo, codespell:ignore

    def test_accepts_gaussian(self):
        """The one supported value is accepted and normalised to lowercase."""
        layer = NoiseSimulatingDataConsistency(noise_type="GAUSSIAN")
        assert layer.noise_type == "gaussian"

    def test_gaussian_noise_actually_added_in_training(self):
        """The gaussian path adds non-zero noise (no silent no-op fallback)."""
        torch.manual_seed(0)
        layer = NoiseSimulatingDataConsistency(train_noise_level=0.5)
        layer.train()
        kspace = torch.zeros(1, 1, 8, 8, dtype=torch.complex64)
        noised = layer._add_realistic_noise(kspace)
        assert noised.abs().sum().item() > 0.0


class TestDataConsistencyLayerBasic:
    """Test basic NoiseSimulatingDataConsistency functionality."""

    def test_initialization_default(self):
        """Test initialization with default parameters."""
        dc_layer = NoiseSimulatingDataConsistency()

        assert dc_layer is not None
        assert dc_layer.noise_lvl is None

    def test_initialization_with_noise_level(self):
        """Test initialization with noise level."""
        noise_lvl = 0.01
        dc_layer = NoiseSimulatingDataConsistency(noise_lvl=noise_lvl)

        assert dc_layer.noise_lvl == noise_lvl

    def test_forward_basic(self):
        """Test basic forward pass."""
        dc_layer = NoiseSimulatingDataConsistency()

        # Create tensors
        # Use (B, C, H, W) complex format which is robust
        predicted_img = torch.randn(1, 1, 16, 16, dtype=torch.complex64)
        measured_kspace = torch.randn(1, 1, 16, 16, dtype=torch.complex64)
        mask = torch.ones(1, 1, 16, 16)

        result = dc_layer(predicted_img, measured_kspace, mask)

        assert result is not None
        assert result.shape == predicted_img.shape


class TestDataConsistencyLayerComplexFormat:
    """Test data consistency with complex tensor format."""

    def test_forward_complex_input(self):
        """Test forward pass with complex tensors."""
        dc_layer = NoiseSimulatingDataConsistency()

        # Create complex tensors
        predicted_img = torch.randn(1, 1, 16, 16, dtype=torch.complex64)
        measured_kspace = torch.randn(1, 1, 16, 16, dtype=torch.complex64)
        mask = torch.ones(1, 1, 16, 16)

        result = dc_layer(predicted_img, measured_kspace, mask)

        assert result is not None

    def test_forward_mixed_complex_and_real(self):
        """Test forward with mixed complex and real inputs."""
        dc_layer = NoiseSimulatingDataConsistency()

        # Complex predicted, real-imag measured
        predicted_img = torch.randn(1, 1, 16, 16, dtype=torch.complex64)
        measured_kspace = torch.randn(1, 1, 16, 16, 2)  # Real-imag format
        mask = torch.ones(1, 1, 16, 16)

        result = dc_layer(predicted_img, measured_kspace, mask)

        assert result is not None


class TestDataConsistencyLayerMaskHandling:
    """Test mask handling in data consistency."""

    def test_full_sampling_mask(self):
        """Test with fully sampled mask."""
        dc_layer = NoiseSimulatingDataConsistency()

        predicted_img = torch.randn(1, 1, 16, 16, dtype=torch.complex64)
        measured_kspace = torch.randn(1, 1, 16, 16, dtype=torch.complex64)
        mask = torch.ones(1, 1, 16, 16)  # Full sampling

        result = dc_layer(predicted_img, measured_kspace, mask)

        assert result is not None

    def test_zero_sampling_mask(self):
        """Test with zero sampling (no measured data)."""
        dc_layer = NoiseSimulatingDataConsistency()

        predicted_img = torch.randn(1, 1, 16, 16, dtype=torch.complex64)
        measured_kspace = torch.randn(1, 1, 16, 16, dtype=torch.complex64)
        mask = torch.zeros(1, 1, 16, 16)  # No sampling

        result = dc_layer(predicted_img, measured_kspace, mask)

        assert result is not None

    def test_partial_sampling_mask(self):
        """Test with partial sampling mask."""
        dc_layer = NoiseSimulatingDataConsistency()

        predicted_img = torch.randn(1, 1, 16, 16, dtype=torch.complex64)
        measured_kspace = torch.randn(1, 1, 16, 16, dtype=torch.complex64)
        mask = torch.ones(1, 1, 16, 16)
        mask[:, :, ::2, :] = 0  # Undersample by factor of 2

        result = dc_layer(predicted_img, measured_kspace, mask)

        assert result is not None

    def test_mask_broadcasting(self):
        """Test mask broadcasting to match k-space dimensions."""
        dc_layer = NoiseSimulatingDataConsistency()

        predicted_img = torch.randn(2, 1, 16, 16, dtype=torch.complex64)
        measured_kspace = torch.randn(2, 1, 16, 16, dtype=torch.complex64)
        mask = torch.ones(1, 1, 16, 16)  # Should broadcast

        result = dc_layer(predicted_img, measured_kspace, mask)

        assert result is not None


class TestDataConsistencyLayerConsistency:
    """Test data consistency enforcement."""

    def test_consistency_enforcement(self):
        """Test that measured k-space is enforced at sampled locations."""
        dc_layer = NoiseSimulatingDataConsistency()

        # Create image estimate
        predicted_img = torch.ones(1, 1, 16, 16, dtype=torch.complex64)

        # Create very different measured k-space
        measured_kspace = torch.zeros(1, 1, 16, 16, dtype=torch.complex64)

        # Full sampling mask - consistency should be enforced everywhere
        mask = torch.ones(1, 1, 16, 16)

        result = dc_layer(predicted_img, measured_kspace, mask)

        assert result is not None

    def test_selective_consistency(self):
        """Test consistency enforcement at masked locations only."""
        dc_layer = NoiseSimulatingDataConsistency()

        predicted_img = torch.ones(1, 1, 16, 16, dtype=torch.complex64)
        measured_kspace = torch.zeros(1, 1, 16, 16, dtype=torch.complex64)

        # Partial mask - consistency only where mask is 1
        mask = torch.zeros(1, 1, 16, 16)
        mask[:, :, :8, :] = 1  # Enforce consistency only in top half

        result = dc_layer(predicted_img, measured_kspace, mask)

        assert result is not None


class TestDataConsistencyLayerBatchProcessing:
    """Test batch processing."""

    def test_batch_forward(self):
        """Test forward with batch of samples."""
        dc_layer = NoiseSimulatingDataConsistency()

        batch_size = 4
        predicted_img = torch.randn(batch_size, 1, 16, 16, dtype=torch.complex64)
        measured_kspace = torch.randn(batch_size, 1, 16, 16, dtype=torch.complex64)
        mask = torch.ones(batch_size, 1, 16, 16)

        result = dc_layer(predicted_img, measured_kspace, mask)

        assert result.shape[0] == batch_size

    def test_batch_processing_preserves_batch_size(self):
        """Test that batch size is preserved."""
        dc_layer = NoiseSimulatingDataConsistency()

        for batch_size in [1, 2, 4, 8]:
            predicted_img = torch.randn(batch_size, 1, 16, 16, dtype=torch.complex64)
            measured_kspace = torch.randn(batch_size, 1, 16, 16, dtype=torch.complex64)
            mask = torch.ones(batch_size, 1, 16, 16)

            result = dc_layer(predicted_img, measured_kspace, mask)

            assert result.shape[0] == batch_size


class TestDataConsistencyLayerMultiCoil:
    """Test multi-coil data consistency."""

    def test_multicoil_forward(self):
        """Test forward with multi-coil data."""
        dc_layer = NoiseSimulatingDataConsistency()

        num_coils = 4
        predicted_img = torch.randn(1, num_coils, 16, 16, dtype=torch.complex64)
        measured_kspace = torch.randn(1, num_coils, 16, 16, dtype=torch.complex64)
        mask = torch.ones(1, 1, 16, 16)

        result = dc_layer(predicted_img, measured_kspace, mask)

        assert result.shape[1] == num_coils

    def test_multicoil_with_sensitivity_handling(self):
        """Test multi-coil processing."""
        dc_layer = NoiseSimulatingDataConsistency()

        num_coils = 8
        height, width = 32, 32

        predicted_img = torch.randn(1, num_coils, height, width, dtype=torch.complex64)
        measured_kspace = torch.randn(
            1, num_coils, height, width, dtype=torch.complex64
        )
        mask = torch.ones(1, 1, height, width)

        result = dc_layer(predicted_img, measured_kspace, mask)

        assert result.shape == predicted_img.shape


class TestDataConsistencyLayerGradientFlow:
    """Test gradient flow through data consistency."""

    def test_gradient_propagation(self):
        """Test that gradients propagate through DC layer."""
        dc_layer = NoiseSimulatingDataConsistency()

        predicted_img = torch.randn(
            1, 1, 16, 16, dtype=torch.complex64, requires_grad=True
        )
        measured_kspace = torch.randn(1, 1, 16, 16, dtype=torch.complex64)
        mask = torch.ones(1, 1, 16, 16)

        result = dc_layer(predicted_img, measured_kspace, mask)

        # Compute loss and backward
        if torch.is_complex(result):
            loss = torch.sum(torch.abs(result) ** 2)
        else:
            loss = torch.sum(result**2)

        loss.backward()

        assert predicted_img.grad is not None


class TestDataConsistencyLayerEdgeCases:
    """Test edge cases."""

    def test_small_image(self):
        """Test with very small image."""
        dc_layer = NoiseSimulatingDataConsistency()

        predicted_img = torch.randn(1, 1, 4, 4, dtype=torch.complex64)
        measured_kspace = torch.randn(1, 1, 4, 4, dtype=torch.complex64)
        mask = torch.ones(1, 1, 4, 4)

        result = dc_layer(predicted_img, measured_kspace, mask)

        assert result is not None

    def test_large_image(self):
        """Test with large image."""
        dc_layer = NoiseSimulatingDataConsistency()

        predicted_img = torch.randn(1, 1, 256, 256, dtype=torch.complex64)
        measured_kspace = torch.randn(1, 1, 256, 256, dtype=torch.complex64)
        mask = torch.ones(1, 1, 256, 256)

        result = dc_layer(predicted_img, measured_kspace, mask)

        assert result is not None


class TestAdaptiveDataConsistency:
    """Test AdaptiveDataConsistency layer."""

    def test_adaptive_dc_initialization(self):
        """Test initialization of adaptive DC layer."""
        adaptive_dc = AdaptiveDataConsistency()

        assert adaptive_dc is not None

    def test_adaptive_dc_forward(self):
        """Test forward pass of adaptive DC layer."""
        adaptive_dc = AdaptiveDataConsistency()

        predicted_img = torch.randn(1, 1, 16, 16, dtype=torch.complex64)
        measured_kspace = torch.randn(1, 1, 16, 16, dtype=torch.complex64)
        mask = torch.ones(1, 1, 16, 16)

        result = adaptive_dc(predicted_img, measured_kspace, mask)

        assert result is not None

    def test_adaptive_dc_learnable_params(self):
        """Test that adaptive DC exposes a parameters() iterator (is a Module)."""
        adaptive_dc = AdaptiveDataConsistency()

        # May or may not have parameters depending on implementation; just
        # verify it's a Module whose parameters() is iterable.
        assert isinstance(adaptive_dc, torch.nn.Module)
        assert isinstance(list(adaptive_dc.parameters()), list)

    def test_adaptive_dc_kspace_domain_no_double_fft(self):
        """K-space domain: prediction must NOT be FFT'd again.

        Regression test for the double-FFT bug where the DC layer applied
        fft2c to already-k-space data, causing ghost/doubling artifacts.
        """
        adaptive_dc = AdaptiveDataConsistency()
        adaptive_dc.eval()

        # Create k-space prediction (interleaved real/imag, 4 coils = 8 ch)
        B, C, H, W = 2, 8, 16, 16
        k_pred = torch.randn(B, C, H, W)
        k_measured = torch.randn(B, C, H, W)
        mask = torch.zeros(B, 1, H, W)
        mask[:, :, ::4, :] = 1.0  # Sample every 4th line

        result = adaptive_dc(k_pred, k_measured, mask, is_kspace_domain=True)

        assert (
            result.shape == k_pred.shape
        ), f"Shape mismatch: expected {k_pred.shape}, got {result.shape}"

        # At unsampled locations, output should equal the prediction
        mask_expanded = mask.expand_as(k_pred)
        unsampled = ~mask_expanded.bool()
        torch.testing.assert_close(
            result[unsampled],
            k_pred[unsampled],
            atol=1e-5,
            rtol=1e-5,
            msg="Unsampled k-space should be preserved from prediction",
        )

    def test_adaptive_dc_accepts_acs_mask_kwarg(self):
        """Verify acs_mask kwarg is accepted without TypeError."""
        adaptive_dc = AdaptiveDataConsistency()
        adaptive_dc.eval()

        k_pred = torch.randn(1, 2, 16, 16)
        k_measured = torch.randn(1, 2, 16, 16)
        mask = torch.ones(1, 1, 16, 16)
        acs_mask = torch.zeros(1, 1, 16, 16)
        acs_mask[:, :, 6:10, :] = 1.0

        # Should NOT raise TypeError
        result = adaptive_dc(
            k_pred,
            k_measured,
            mask,
            is_kspace_domain=True,
            acs_mask=acs_mask,
        )
        assert result is not None

    def test_adaptive_dc_kspace_complex_input(self):
        """K-space domain with complex tensors should stay in k-space."""
        adaptive_dc = AdaptiveDataConsistency()
        adaptive_dc.eval()

        k_pred = torch.randn(1, 4, 16, 16, dtype=torch.complex64)
        k_measured = torch.randn(1, 4, 16, 16, dtype=torch.complex64)
        mask = torch.ones(1, 1, 16, 16)

        result = adaptive_dc(k_pred, k_measured, mask, is_kspace_domain=True)

        assert torch.is_complex(result), "Output must remain complex"
        assert result.shape == k_pred.shape

    def test_adaptive_dc_kspace_gradient_flow(self):
        """Gradients must propagate through k-space domain path."""
        adaptive_dc = AdaptiveDataConsistency()
        adaptive_dc.train()

        k_pred = torch.randn(1, 4, 16, 16, requires_grad=True)
        k_measured = torch.randn(1, 4, 16, 16)
        mask = torch.ones(1, 1, 16, 16)

        result = adaptive_dc(k_pred, k_measured, mask, is_kspace_domain=True)
        loss = result.abs().sum()
        loss.backward()

        assert k_pred.grad is not None, "Gradients must flow to k_pred"


class TestDataConsistencyIntegration:
    """Integration tests for data consistency."""

    def test_dc_in_unrolled_network(self):
        """Test DC layer in unrolled network context."""

        # Create simple unrolled network with DC
        class UnrolledNetwork(torch.nn.Module):
            def __init__(self, num_unrolls=5):
                super().__init__()
                self.dc_layers = torch.nn.ModuleList(
                    [NoiseSimulatingDataConsistency() for _ in range(num_unrolls)]
                )
                self.image_updates = torch.nn.ModuleList(
                    [
                        torch.nn.Conv2d(1, 1, kernel_size=3, padding=1)
                        for _ in range(num_unrolls)
                    ]
                )

            def forward(self, image, measured_kspace, mask):
                for dc_layer, update in zip(
                    self.dc_layers, self.image_updates, strict=False
                ):
                    # Update image
                    image = update(image)

                    # Apply DC (would need to convert to complex)
                    # For test, just verify forward pass works
                return image

        network = UnrolledNetwork()

        image = torch.randn(1, 1, 16, 16)
        measured_kspace = torch.randn(1, 1, 16, 16, dtype=torch.complex64)
        mask = torch.ones(1, 1, 16, 16)

        result = network(image, measured_kspace, mask)

        assert result is not None


class TestDataConsistencyHelperWeightKnob:
    """The ``data_consistency()`` helper's ``weight`` must be a live soft-DC knob.

    The helper wraps ``SimpleDataConsistency``, whose default ``method='hard'``
    ignores ``weight``. It now passes ``method='soft'`` so ``weight`` is wired:
    ``weight=0`` is a pure passthrough and larger ``weight`` pulls the sampled
    lines toward the measurement (CLAUDE.md pitfall #15).
    """

    @staticmethod
    def _inputs():
        torch.manual_seed(1)
        x_img = torch.complex(torch.randn(1, 1, 8, 8), torch.randn(1, 1, 8, 8))
        k_measured = fft2c(
            torch.complex(torch.randn(1, 1, 8, 8), torch.randn(1, 1, 8, 8))
        )
        mask = torch.zeros(1, 1, 8, 8)
        mask[..., :4] = 1.0
        return x_img, k_measured, mask

    def test_weight_zero_is_passthrough(self):
        x_img, k_measured, mask = self._inputs()
        out = data_consistency(x_img, k_measured, mask, weight=0.0)
        assert torch.allclose(out.abs(), x_img.abs(), atol=1e-4)

    def test_larger_weight_moves_toward_measurement(self):
        x_img, k_measured, mask = self._inputs()
        mask_bool = mask.bool()
        out0 = data_consistency(x_img, k_measured, mask, weight=0.0)
        out_big = data_consistency(x_img, k_measured, mask, weight=1e3)
        k0 = fft2c(out0)[mask_bool]
        k_big = fft2c(out_big)[mask_bool]
        k_meas = k_measured[mask_bool]
        # weight=0 stays at the prediction; large weight is close to measurement.
        assert (k_big - k_meas).abs().mean() < (k0 - k_meas).abs().mean()


class TestValidDcMethodsSSOT:
    """``VALID_DC_METHODS`` is the single advertised set of DC methods.

    Both the model-internal DC builder (``KSpaceColdDiffusionGenerator``) and
    the reverse-diffusion sampler (``PhysicsInformedColdDiffusion``) validate
    against this one frozenset — so a method one accepts, the other can honour
    (the 2026-07-05 ``dc_method='adaptive'`` sampler crash was a divergence:
    the generator built ``AdaptiveDataConsistency`` while the sampler knew only
    hard/soft).
    """

    def test_is_frozenset(self):
        from spectramr.infrastructure.physics.data_consistency import VALID_DC_METHODS

        assert isinstance(VALID_DC_METHODS, frozenset)

    def test_contains_all_advertised_methods(self):
        from spectramr.infrastructure.physics.data_consistency import VALID_DC_METHODS

        assert {
            "hard",
            "soft",
            "noise_adjusted",
            "adaptive",
            "kan_adaptive",
            "target_aware_fsdc",
            "noise_adaptive",
            "wiener_gain",
        } <= VALID_DC_METHODS


class TestNoiseAdaptiveDataConsistency:
    """Physics-based Wiener/SNR DC: trust the measurement where SNR is high,
    denoise (keep the prediction) where noise dominates.

    ``lam(k) = |y(k)|^2 / (|y(k)|^2 + softplus(beta)*sigma^2)`` is a convex
    blend weight in ``[0, 1)``, so ``k_new`` never amplifies, and unsampled bins
    stay pure prediction (no measurement-independent blob).
    """

    def test_high_snr_bin_trusts_measurement(self):
        H = W = 16
        measured = torch.full((1, 1, H, W), 0.1, dtype=torch.cfloat)
        cy, cx = H // 2, W // 2
        measured[0, 0, cy, cx] = 100.0  # a very high-SNR bin at k-space centre
        mask = torch.ones(1, 1, H, W)
        k_pred = torch.zeros(1, 1, H, W, dtype=torch.cfloat)
        dc = NoiseAdaptiveDataConsistency(beta=1.0)
        out = dc(k_pred, measured_kspace=measured, mask=mask, is_kspace_domain=True)
        # lam -> 1 at the high-SNR bin: output tracks the measurement.
        assert torch.allclose(out[0, 0, cy, cx], measured[0, 0, cy, cx], rtol=1e-2)

    def test_pure_noise_bin_denoises(self):
        H = W = 16
        measured = torch.full((1, 1, H, W), 0.1, dtype=torch.cfloat)  # all noise-level
        mask = torch.ones(1, 1, H, W)
        k_pred = torch.full(
            (1, 1, H, W), 5.0, dtype=torch.cfloat
        )  # distinct prediction
        dc = NoiseAdaptiveDataConsistency(beta=3.0)
        out = dc(k_pred, measured_kspace=measured, mask=mask, is_kspace_domain=True)
        val = out[0, 0, 2, 2]  # a peripheral noise-dominated bin
        # Output is pulled toward the prediction, away from the noisy measurement.
        assert abs(val - k_pred[0, 0, 2, 2]) < abs(val - measured[0, 0, 2, 2])

    def test_convex_blend_boundedness(self):
        torch.manual_seed(0)
        H = W = 16
        measured = torch.randn(1, 1, H, W, dtype=torch.cfloat)
        k_pred = torch.randn(1, 1, H, W, dtype=torch.cfloat)
        mask = torch.ones(1, 1, H, W)
        dc = NoiseAdaptiveDataConsistency(beta=1.0)
        out = dc(k_pred, measured_kspace=measured, mask=mask, is_kspace_domain=True)
        bound = torch.maximum(measured.abs(), k_pred.abs()) + 1e-5
        assert bool((out.abs() <= bound).all())

    def test_unsampled_bins_are_pure_prediction(self):
        """Blob-safety: where mask==0 the output is exactly the prediction."""
        torch.manual_seed(1)
        H = W = 16
        measured = torch.randn(1, 1, H, W, dtype=torch.cfloat)
        k_pred = torch.randn(1, 1, H, W, dtype=torch.cfloat)
        mask = torch.zeros(1, 1, H, W)
        mask[..., :8] = 1.0
        dc = NoiseAdaptiveDataConsistency(beta=1.0)
        out = dc(k_pred, measured_kspace=measured, mask=mask, is_kspace_domain=True)
        uns = ~mask.bool().expand_as(out)
        assert torch.allclose(out[uns], k_pred[uns], atol=1e-6)

    def test_estimate_noise_variance_sane(self):
        """Self-contained sigma^2 estimate is within a small factor of the truth."""
        torch.manual_seed(2)
        H = W = 32
        sigma = 0.3
        noise = (torch.randn(1, 1, H, W) + 1j * torch.randn(1, 1, H, W)) * sigma
        mask = torch.ones(1, 1, H, W)
        dc = NoiseAdaptiveDataConsistency(beta=1.0)
        s2 = dc.estimate_noise_variance(noise.abs() ** 2, mask)
        expected = 2.0 * sigma**2  # E[|noise|^2] for complex Gaussian
        assert 0.25 * expected < float(s2.mean()) < 4.0 * expected

    def test_noise_sigma_override(self):
        H = W = 16
        measured = torch.full((1, 1, H, W), 1.0, dtype=torch.cfloat)
        k_pred = torch.zeros(1, 1, H, W, dtype=torch.cfloat)
        mask = torch.ones(1, 1, H, W)
        dc = NoiseAdaptiveDataConsistency(beta=1.0)
        out_hi = dc(k_pred, measured, mask, is_kspace_domain=True, noise_sigma=1000.0)
        out_lo = dc(k_pred, measured, mask, is_kspace_domain=True, noise_sigma=1e-6)
        # Large sigma -> lam~0 -> prediction (0); tiny sigma -> lam~1 -> measurement (1).
        assert float(out_hi.abs().mean()) < float(out_lo.abs().mean())
        assert torch.allclose(out_lo, measured, rtol=1e-2)

    def test_lambda_in_unit_interval_via_telemetry(self):
        torch.manual_seed(3)
        H = W = 16
        measured = torch.randn(1, 1, H, W, dtype=torch.cfloat) * 10.0
        k_pred = torch.randn(1, 1, H, W, dtype=torch.cfloat)
        mask = torch.ones(1, 1, H, W)
        dc = NoiseAdaptiveDataConsistency(beta=1.0)
        dc(k_pred, measured, mask, is_kspace_domain=True)
        stats = dc._last_trust_stats  # [center, periphery, mean, std]
        for i in range(3):
            assert 0.0 <= float(stats[i]) <= 1.0

    def test_interleaved_and_complex_io_agree(self):
        torch.manual_seed(4)
        H = W = 16
        measured_i = torch.randn(1, 2, H, W)  # 1 complex coil, real-interleaved
        k_pred_i = torch.randn(1, 2, H, W)
        mask = torch.ones(1, 1, H, W)
        dc = NoiseAdaptiveDataConsistency(beta=1.0)
        out_i = dc(k_pred_i, measured_i, mask, is_kspace_domain=True)
        assert out_i.shape == k_pred_i.shape and not torch.is_complex(out_i)

        measured_c = torch.complex(measured_i[:, 0::2], measured_i[:, 1::2])
        k_pred_c = torch.complex(k_pred_i[:, 0::2], k_pred_i[:, 1::2])
        out_c = dc(k_pred_c, measured_c, mask, is_kspace_domain=True)
        assert torch.is_complex(out_c) and out_c.shape == k_pred_c.shape

        out_i_as_c = torch.complex(out_i[:, 0::2], out_i[:, 1::2])
        assert torch.allclose(out_i_as_c, out_c, atol=1e-4)

    def test_signal_power_replaces_the_measured_power_in_the_weight(self):
        """A large declared signal power pulls a noise-level bin to the measurement."""
        H = W = 16
        measured = torch.full((1, 1, H, W), 0.1, dtype=torch.cfloat)
        k_pred = torch.full((1, 1, H, W), 5.0, dtype=torch.cfloat)
        mask = torch.ones(1, 1, H, W)
        dc = NoiseAdaptiveDataConsistency(beta=1.0)
        plain = dc(k_pred, measured, mask, is_kspace_domain=True, noise_sigma=0.1)
        loud = dc(
            k_pred,
            measured,
            mask,
            is_kspace_domain=True,
            noise_sigma=0.1,
            signal_power=torch.full((1, 1, H, W), 1e6),
        )
        assert float((loud - measured).abs().max()) < float((plain - measured).abs().max())
        assert torch.allclose(loud, measured, atol=1e-3)

    def test_the_trust_radius_keeps_measured_bins_in_the_noise_disc(self):
        torch.manual_seed(5)
        H = W = 16
        sigma, radius = 0.2, 3.0
        measured = torch.randn(1, 2, H, W, dtype=torch.cfloat)
        k_pred = 100.0 * torch.randn(1, 2, H, W, dtype=torch.cfloat)
        mask = torch.zeros(1, 1, H, W)
        mask[..., ::2, :] = 1.0
        dc = NoiseAdaptiveDataConsistency(beta=1.0)
        out = dc(
            k_pred, measured, mask, is_kspace_domain=True, noise_sigma=sigma, trust_radius=radius
        )
        held = mask.bool().expand_as(out)
        assert float((out - measured).abs()[held].max()) <= radius * sigma * (1 + 1e-5)
        assert torch.equal(out[~held], k_pred[~held])

    def test_the_new_arguments_default_to_the_wiener_blend(self):
        """Without them the layer is ``lam = |y|^2 / (|y|^2 + softplus(beta) sigma^2)``."""
        torch.manual_seed(6)
        measured = torch.randn(2, 2, 16, 16, dtype=torch.cfloat)
        k_pred = torch.randn(2, 2, 16, 16, dtype=torch.cfloat)
        mask = (torch.rand(2, 1, 16, 16) > 0.5).float()
        dc = NoiseAdaptiveDataConsistency(beta=1.0)
        out = dc(k_pred, measured, mask, is_kspace_domain=True, noise_sigma=0.5)
        power = (measured.abs() ** 2).mean(dim=1, keepdim=True)
        beta = torch.nn.functional.softplus(dc.beta)
        lam = power / (power + beta * 0.25 + dc.eps)
        blend = (1 - lam) * k_pred + lam * measured
        expected = torch.where(mask.bool().expand_as(k_pred), blend, k_pred)
        assert torch.allclose(out, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# noise_type: one owner, one policy (#1525)
#
# Three layers took this parameter and applied three different policies:
# NoiseSimulatingDataConsistency validated it, SimpleDataConsistency stored it unchecked,
# and HardDataConsistency accepted it and never stored it at all -- so an
# unsupported value degraded silently to Gaussian on two of the three.
# ---------------------------------------------------------------------------


class TestNoiseTypeIsValidatedEverywhere:
    """Every layer that ACCEPTS noise_type must also enforce it."""

    @staticmethod
    def _layers():
        from spectramr.infrastructure.physics.data_consistency import (
            HardDataConsistency,
            NoiseSimulatingDataConsistency,
            SimpleDataConsistency,
        )

        return [NoiseSimulatingDataConsistency, HardDataConsistency, SimpleDataConsistency]

    def test_gaussian_is_accepted_and_stored(self) -> None:
        for cls in self._layers():
            layer = cls(noise_type="GAUSSIAN")
            assert layer.noise_type == "gaussian", f"{cls.__name__} lost noise_type"

    def test_hard_dc_actually_stores_it(self) -> None:
        """It used to accept the parameter and drop it on the floor."""
        from spectramr.infrastructure.physics.data_consistency import HardDataConsistency

        assert hasattr(HardDataConsistency(), "noise_type")

    @pytest.mark.parametrize("bad", ["rician", "poisson", "none", "GAUSS"])
    def test_unsupported_noise_type_raises_on_every_layer(self, bad: str) -> None:
        """No silent fallback (non-negotiable 3). 'rician' was advertised, never built."""
        for cls in self._layers():
            with pytest.raises(ValueError, match="unsupported noise_type"):
                cls(noise_type=bad)

    def test_the_error_names_the_layer_that_rejected_it(self) -> None:
        from spectramr.infrastructure.physics.data_consistency import HardDataConsistency

        with pytest.raises(ValueError, match="HardDataConsistency"):
            HardDataConsistency(noise_type="rician")


class TestHardDataConsistencyTakesNoWeight:
    """The premise of the inert-by-method finding, pinned in the layer itself.

    If someone ever gives this constructor a ``weight``, ``dc_weight`` stops
    being inert under ``dc_method: hard`` and ``dc_settings.DCKnobReadership``
    becomes wrong -- silently, because both still produce a plausible number.
    """

    def test_constructor_has_no_weight_parameter(self) -> None:
        import inspect

        from spectramr.infrastructure.physics.data_consistency import HardDataConsistency

        params = set(inspect.signature(HardDataConsistency.__init__).parameters)
        assert "weight" not in params
        assert "lambda_init" not in params
        assert "beta" not in params

    def test_noise_levels_are_settable(self) -> None:
        from spectramr.infrastructure.physics.data_consistency import HardDataConsistency

        layer = HardDataConsistency(train_noise_level=0.07, eval_noise_level=0.03)
        assert layer.train_noise_level == 0.07
        assert layer.eval_noise_level == 0.03


# ---------------------------------------------------------------------------
# Noise estimate on centre-weighted masks (#2475)
#
# The estimator read a fixed outer annulus (r > 0.7 r_max) and fell back to every
# measured bin when it held fewer than 8, so on variable-density and gaussian masks
# it reported signal power as noise. It now ranks each sample's measured support.
# ---------------------------------------------------------------------------

_GRID = 256
_SIGMA = 0.05
_ACCELERATIONS = (2, 4, 8, 16, 32)


def _radius() -> torch.Tensor:
    k = torch.arange(_GRID) - _GRID // 2
    r = torch.sqrt(k[:, None] ** 2 + k[None, :] ** 2).float()
    return r / r.max()


def _density_mask(shape, accel: int, seed: int) -> torch.Tensor:
    """A random point mask of density ``shape(r / w)`` truncated at 0.5 r_max, mean ``1/accel``."""
    r = _radius()
    lo, hi = 1e-3, 10.0
    for _ in range(60):
        w = 0.5 * (lo + hi)
        density = torch.where(r <= 0.5, shape(r / w), torch.zeros_like(r)).clamp(0, 1)
        lo, hi = (w, hi) if float(density.mean()) < 1.0 / accel else (lo, w)
    keep = torch.rand(_GRID, _GRID, generator=torch.Generator().manual_seed(seed)) < density
    return keep.float().view(1, 1, _GRID, _GRID)


def _variable_density(accel: int) -> torch.Tensor:
    return _density_mask(lambda x: (1.0 - x).clamp_min(0) ** 4, accel, seed=accel)


def _gaussian(accel: int) -> torch.Tensor:
    return _density_mask(lambda x: torch.exp(-(x**2)), accel, seed=100 + accel)


def _central_band(accel: int) -> torch.Tensor:
    """Fully sampled central rows (a density_nested-like rung)."""
    mask = torch.zeros(1, 1, _GRID, _GRID)
    rows = max(1, _GRID // accel)
    start = _GRID // 2 - rows // 2
    mask[..., start : start + rows, :] = 1.0
    return mask


_MASKS = {"variable_density": _variable_density, "gaussian": _gaussian, "band": _central_band}


def _power() -> torch.Tensor:
    """``|y|^2`` of a low-frequency object plus white complex noise of variance ``_SIGMA^2``."""
    gen = torch.Generator().manual_seed(7)
    k = torch.arange(_GRID) - _GRID // 2
    r = torch.sqrt(k[:, None] ** 2 + k[None, :] ** 2).float()
    phase = torch.exp(1j * 6.283 * torch.rand(_GRID, _GRID, generator=gen))
    signal = 100.0 * torch.exp(-((r / 14.0) ** 2)) * phase
    noise = torch.complex(
        torch.randn(_GRID, _GRID, generator=gen), torch.randn(_GRID, _GRID, generator=gen)
    )
    y = signal + noise * (_SIGMA / 2**0.5)
    return (y.abs() ** 2).view(1, 1, _GRID, _GRID)


# Median of an exponential of mean sigma^2: what a median estimator converges to.
_NOISE_MEDIAN = _SIGMA**2 * math.log(2.0)


def _fixed_annulus_estimate(power: torch.Tensor, mask: torch.Tensor) -> tuple[int, float]:
    """The pre-#2475 estimator, kept as the planted violation: ``(annulus bins, sigma^2)``."""
    radius = _radius().view(1, 1, _GRID, _GRID)
    vals = power[(mask > 0) & (radius > 0.7)]
    count = vals.numel()
    if count < 8:
        vals = power[mask > 0]
    return count, float(vals.median())


class TestNoiseEstimateOnMeasuredSupport:
    @pytest.mark.parametrize("accel", _ACCELERATIONS)
    @pytest.mark.parametrize("kind", sorted(_MASKS))
    def test_estimate_tracks_the_noise_median(self, kind, accel):
        mask = _MASKS[kind](accel)
        dc = NoiseAdaptiveDataConsistency()
        region = dc.noise_region(mask)
        n = int(mask.sum())
        assert int(region.sum()) == max(1, -(-n // 10))
        assert bool((region <= (mask > 0)).all())
        sigma2 = float(dc.estimate_noise_variance(_power(), mask))
        assert abs(sigma2 / _NOISE_MEDIAN - 1.0) < 0.3

    @pytest.mark.parametrize("accel", (16, 32))
    @pytest.mark.parametrize("kind", ("variable_density", "gaussian"))
    def test_fixed_annulus_estimator_fails_where_the_new_one_holds(self, kind, accel):
        """Planted violation: the old annulus is empty and its fallback reads signal."""
        mask = _MASKS[kind](accel)
        power = _power()
        count, old = _fixed_annulus_estimate(power, mask)
        assert count == 0
        assert old > 5.0 * _NOISE_MEDIAN
        new = float(NoiseAdaptiveDataConsistency().estimate_noise_variance(power, mask))
        assert abs(new / _NOISE_MEDIAN - 1.0) < 0.3

    @pytest.mark.parametrize("accel", _ACCELERATIONS)
    @pytest.mark.parametrize("kind", ("variable_density", "gaussian"))
    def test_fixed_annulus_is_empty_at_every_rung(self, kind, accel):
        count, _ = _fixed_annulus_estimate(_power(), _MASKS[kind](accel))
        assert count == 0

    def test_zero_padded_edge_is_not_read_as_noise(self):
        """Planted violation: on zero-padded columns the outermost measured bins are the padding."""
        mask = _central_band(4)
        power = _power().clone()
        pad = 30
        power[..., :pad] = 1e-16  # what the log1p round trip leaves of an exact zero
        power[..., -pad:] = 1e-16
        dc = NoiseAdaptiveDataConsistency()
        naive = dc.noise_region(mask)
        assert float(power[naive].median()) < 1e-12
        support = (mask > 0) & (power > 1e-12)
        assert bool((dc.noise_region(support) <= support).all())
        sigma2 = float(dc.estimate_noise_variance(power, mask))
        assert abs(sigma2 / _NOISE_MEDIAN - 1.0) < 0.3

    def test_noise_region_takes_a_bool_or_float_support(self):
        mask = _gaussian(8)
        dc = NoiseAdaptiveDataConsistency()
        assert torch.equal(dc.noise_region(mask), dc.noise_region(mask > 0))

    def test_mixed_batch_matches_per_sample(self):
        masks = [_variable_density(32), _central_band(4), _gaussian(8)]
        power = _power().expand(3, -1, -1, -1)
        dc = NoiseAdaptiveDataConsistency()
        batched = dc.estimate_noise_variance(power, torch.cat(masks)).flatten()
        single = torch.stack(
            [dc.estimate_noise_variance(power[:1], m).flatten()[0] for m in masks]
        )
        assert torch.equal(batched, single)

    def test_a_sample_without_measured_bins_raises(self):
        mask = torch.cat([_central_band(4), torch.zeros(1, 1, _GRID, _GRID)])
        power = _power().expand(2, -1, -1, -1)
        with pytest.raises(ValueError, match=r"sample\(s\) \[1\] have no measured"):
            NoiseAdaptiveDataConsistency().estimate_noise_variance(power, mask)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_the_cuda_empty_support_check_does_not_sync(self):
        mask = _central_band(4).cuda()
        power = _power().cuda()
        dc = NoiseAdaptiveDataConsistency()
        dc.estimate_noise_variance(power, mask)  # builds the coordinate cache
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            dc.noise_region(mask)
        finally:
            torch.cuda.set_sync_debug_mode("default")

    def test_a_supplied_variance_skips_the_estimate(self, monkeypatch):
        dc = NoiseAdaptiveDataConsistency()
        y = torch.randn(1, 2, _GRID, _GRID, dtype=torch.complex64)
        mask = _central_band(4)
        monkeypatch.setattr(dc, "estimate_noise_variance", pytest.fail)
        sigma2 = torch.full((1, 1, 1, 1), 0.5)
        lam = dc._compute_lambda(y, mask, None, sigma2=sigma2)
        power = (y.abs() ** 2).mean(dim=1, keepdim=True)
        beta = torch.nn.functional.softplus(dc.beta)
        torch.testing.assert_close(lam, power / (power + beta * sigma2 + dc.eps))

    @pytest.mark.parametrize("frac", (0.0, -0.1, 1.5))
    def test_noise_support_frac_outside_unit_interval_raises(self, frac):
        with pytest.raises(ValueError, match="noise_support_frac"):
            NoiseAdaptiveDataConsistency(noise_support_frac=frac)


class TestNoiseAdjustedDataConsistency:
    """One Wiener trust per sample from its noise level (``noise_adjusted``).

    It shares ``noise_adaptive``'s noise estimate and blend, with the sample's mean
    measured power in place of each bin's, so every measured bin of a sample is trusted
    equally. Until 2026-10-09 the name was an alias of ``soft``.
    """

    @staticmethod
    def _peaked(h: int = 16) -> torch.Tensor:
        """A k-space whose centre bin is much louder than its periphery."""
        torch.manual_seed(0)
        k = 0.1 * torch.randn(1, 1, h, h, dtype=torch.cfloat)
        k[0, 0, h // 2, h // 2] = 50.0
        return k

    @staticmethod
    def _trust(dc, measured, mask) -> torch.Tensor:
        """The blend weight each measured bin got: out = (1 - lam) * 0 + lam * y."""
        out = dc(
            torch.zeros_like(measured), measured_kspace=measured, mask=mask, is_kspace_domain=True
        )
        return (out / measured)[mask.bool().expand_as(measured)].real

    def test_every_measured_bin_of_a_sample_gets_the_same_trust(self) -> None:
        measured, mask = self._peaked(), torch.ones(1, 1, 16, 16)
        lam = self._trust(NoiseAdjustedDataConsistency(beta=1.0), measured, mask)
        assert torch.allclose(lam, lam[0].expand_as(lam), atol=1e-5)

    def test_it_differs_from_the_per_bin_noise_adaptive_layer(self) -> None:
        """Planted: the alias it replaced (or noise_adaptive itself) would not be constant."""
        measured, mask = self._peaked(), torch.ones(1, 1, 16, 16)
        per_bin = self._trust(NoiseAdaptiveDataConsistency(beta=1.0), measured, mask)
        assert float(per_bin.max() - per_bin.min()) > 0.1

    def test_more_noise_means_less_trust(self) -> None:
        measured, mask = self._peaked(), torch.ones(1, 1, 16, 16)
        dc = NoiseAdjustedDataConsistency(beta=1.0)
        zero = torch.zeros_like(measured)
        kw = {"measured_kspace": measured, "mask": mask, "is_kspace_domain": True}
        quiet = dc(zero, noise_sigma=0.01, **kw)
        loud = dc(zero, noise_sigma=5.0, **kw)
        assert float((quiet / measured).real.mean()) > float((loud / measured).real.mean())

    def test_unsampled_bins_are_pure_prediction(self) -> None:
        torch.manual_seed(1)
        measured = torch.randn(1, 1, 16, 16, dtype=torch.cfloat)
        k_pred = torch.randn(1, 1, 16, 16, dtype=torch.cfloat)
        mask = torch.zeros(1, 1, 16, 16)
        mask[..., :8] = 1.0
        out = NoiseAdjustedDataConsistency()(
            k_pred, measured_kspace=measured, mask=mask, is_kspace_domain=True
        )
        assert torch.equal(out[..., 8:], k_pred[..., 8:])

    def test_padded_zeros_do_not_dilute_the_signal_power(self) -> None:
        measured, mask = self._peaked(), torch.ones(1, 1, 16, 16)
        dc = NoiseAdjustedDataConsistency()
        padded = measured.clone()
        padded[..., -4:] = 0.0
        unpadded = dc.mean_measured_power(measured[..., :-4], mask[..., :-4])
        assert torch.allclose(dc.mean_measured_power(padded, mask), unpadded)

    def test_a_caller_supplied_signal_power_is_refused(self) -> None:
        measured, mask = self._peaked(), torch.ones(1, 1, 16, 16)
        with pytest.raises(ValueError, match="sets its own signal power"):
            NoiseAdjustedDataConsistency()(
                torch.zeros_like(measured),
                measured_kspace=measured,
                mask=mask,
                is_kspace_domain=True,
                signal_power=measured.abs() ** 2,
            )

    def test_the_generator_builds_it_for_noise_adjusted_and_not_soft(self) -> None:
        """Wiring pin: the alias branch is gone (building the generator is a cpu_tests job)."""
        import inspect

        from spectramr.models.generators.kspace_cold_diffusion_generator import (
            KSpaceColdDiffusionGenerator,
        )

        src = inspect.getsource(KSpaceColdDiffusionGenerator)
        assert 'self.dc_method == "noise_adjusted":' in src
        assert "NoiseAdjustedDataConsistency(\n                beta=self.dc_weight, log_scaled=dc_log_scaled\n            )" in src
        assert 'or self.dc_method == "noise_adjusted"' not in src


class TestLinearDomainBlendGradients:
    """Backward through decompress -> blend -> recompress stays finite.

    The reverse-loop harness samples without gradients, so a NaN that only the
    backward pass produces is invisible to it. Compressed magnitudes span the
    realistic range up to near the decompression ceiling, where ``expm1`` is ~1e10.
    """

    @staticmethod
    def _operands(seed: int = 0):
        from spectramr.data.transforms.normalization import compress_kspace_log

        g = torch.Generator().manual_seed(seed)
        mag = torch.cat(
            [torch.rand(1, 2, 8, 6, generator=g) * 12.0, 20.0 + torch.rand(1, 2, 8, 2, generator=g) * 3.0],
            dim=-1,
        )
        phase = torch.rand(1, 2, 8, 8, generator=g) * 6.283
        z = torch.polar(torch.expm1(mag), phase)
        pred = compress_kspace_log(torch.stack([z.real, z.imag], 2).flatten(1, 2), channel_dim=1)
        meas = compress_kspace_log(
            torch.stack([z.real * 0.9, z.imag * 1.1], 2).flatten(1, 2), channel_dim=1
        )
        mask = torch.zeros(1, 1, 8, 8)
        mask[..., ::2] = 1.0
        return pred.detach().requires_grad_(True), meas, mask

    @pytest.mark.parametrize("kind", ["soft", "noise_adaptive", "noise_adjusted", "fsdc"])
    def test_gradient_is_finite(self, kind: str) -> None:
        from spectramr.infrastructure.physics.data_consistency import (
            NoiseAdaptiveDataConsistency,
            NoiseAdjustedDataConsistency,
            SoftDataConsistency,
            TargetAwareFSDC,
        )

        pred, meas, mask = self._operands()
        layer = {
            "soft": lambda: SoftDataConsistency(lambda_init=0.5, log_scaled=True),
            "noise_adaptive": lambda: NoiseAdaptiveDataConsistency(
                beta=0.5, learnable_beta=True, log_scaled=True
            ),
            "noise_adjusted": lambda: NoiseAdjustedDataConsistency(
                beta=0.5, learnable_beta=True, log_scaled=True
            ),
            "fsdc": lambda: TargetAwareFSDC(
                target_channels=4, center_fraction=0.25, log_scaled=True
            ),
        }[kind]()
        if kind == "fsdc":
            out = layer(pred, meas, mask)
        else:
            out = layer(pred, meas, mask, is_kspace_domain=True)
        assert torch.isfinite(out).all()
        out.square().sum().backward()
        assert torch.isfinite(pred.grad).all(), f"{kind}: non-finite input gradient"
        for name, p in layer.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"{kind}: non-finite grad on {name}"
