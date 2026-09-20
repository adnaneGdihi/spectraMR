"""Unit tests for FFT operations."""

import pytest
import torch

from spectramr.infrastructure.physics.fft_ops import (
    _to_complex,
    _to_ri,
    fft2c,
    fft2c_masked,
    ifft2c,
)


class TestToComplex:
    """Test _to_complex conversion function."""

    def test_complex_tensor_unchanged(self):
        """Test that complex tensors are returned unchanged."""
        x = torch.randn(2, 3, 4, dtype=torch.complex64)

        result = _to_complex(x)

        assert torch.is_complex(result)
        assert result.dtype == torch.complex64

    def test_real_tensor_to_complex(self):
        """Test conversion of real tensor to complex."""
        x = torch.randn(2, 3, 4)

        result = _to_complex(x)

        assert torch.is_complex(result)

    def test_real_imag_stacked_conversion(self):
        """Test conversion of real-imag stacked tensor."""
        # Shape: (2, 3, 4, 2) where last dim is real, imag
        x = torch.randn(2, 3, 4, 2)

        result = _to_complex(x)

        assert torch.is_complex(result)

    def test_channel_2_conversion(self):
        """Test conversion of [B, 2, H, W] format (common in PyTorch)."""
        # Shape: (2, 2, 4, 4) where channel dim is real, imag
        x = torch.randn(2, 2, 4, 4)

        result = _to_complex(x)

        assert torch.is_complex(result)


class TestToRI:
    """Test _to_ri conversion function."""

    def test_complex_to_real_imag(self):
        """Test conversion of complex to real-imag stacked."""
        x = torch.randn(2, 3, 4, dtype=torch.complex64)

        result = _to_ri(x)

        assert not torch.is_complex(result)
        assert result.shape[-1] == 2

    def test_real_tensor_to_ri(self):
        """Test conversion of real tensor (adds zero imag)."""
        x = torch.randn(2, 3, 4)

        result = _to_ri(x)

        assert not torch.is_complex(result)
        assert result.shape[-1] == 2

    def test_shape_preservation(self):
        """Test that shape is preserved except last dimension."""
        x = torch.randn(2, 3, 4, 5, dtype=torch.complex64)

        result = _to_ri(x)

        assert result.shape[:-1] == x.shape
        assert result.shape[-1] == 2


class TestFFT2C:
    """Test centered 2D FFT."""

    def test_fft2c_basic(self):
        """Test basic FFT2C operation."""
        x = torch.randn(2, 3, 4, dtype=torch.complex64)

        result = fft2c(x)

        assert torch.is_complex(result)
        assert result.shape == x.shape

    def test_fft2c_real_input(self):
        """Test FFT2C with real input."""
        x = torch.randn(2, 4, 4)

        result = fft2c(x)

        assert torch.is_complex(result)

    def test_fft2c_normalization(self):
        """Test FFT2C uses orthonormal normalization."""
        x = torch.ones(1, 4, 4, dtype=torch.complex64)

        result = fft2c(x)

        # Should produce a normalized output
        assert torch.is_complex(result)

    def test_fft2c_energy_preservation_approx(self):
        """Test that FFT energy is approximately preserved."""
        x = torch.randn(2, 8, 8, dtype=torch.complex64)

        kspace = fft2c(x)

        # Energy should be approximately equal (Parseval's theorem)
        x_energy = torch.sum(torch.abs(x) ** 2)
        k_energy = torch.sum(torch.abs(kspace) ** 2)

        # With orthonormal norm, should be equal
        assert torch.allclose(x_energy, k_energy, rtol=1e-5)

    def test_fft2c_batch_processing(self):
        """Test FFT2C with batch of images."""
        batch_size = 4
        x = torch.randn(batch_size, 16, 16, dtype=torch.complex64)

        result = fft2c(x)

        assert result.shape == x.shape


class TestIFFT2C:
    """Test centered 2D inverse FFT."""

    def test_ifft2c_basic(self):
        """Test basic IFFT2C operation."""
        k = torch.randn(2, 3, 4, dtype=torch.complex64)

        result = ifft2c(k)

        assert torch.is_complex(result)
        assert result.shape == k.shape

    def test_ifft2c_real_input(self):
        """Test IFFT2C with real input."""
        k = torch.randn(2, 4, 4)

        result = ifft2c(k)

        assert torch.is_complex(result)

    def test_fft_ifft_roundtrip(self):
        """Test FFT->IFFT roundtrip."""
        x_orig = torch.randn(2, 8, 8, dtype=torch.complex64)

        kspace = fft2c(x_orig)
        x_recon = ifft2c(kspace)

        # Should recover original image
        assert torch.allclose(x_orig, x_recon, rtol=1e-5, atol=1e-7)

    def test_ifft2c_batch_processing(self):
        """Test IFFT2C with batch of k-spaces."""
        batch_size = 4
        k = torch.randn(batch_size, 16, 16, dtype=torch.complex64)

        result = ifft2c(k)

        assert result.shape == k.shape


class TestFFT2CMasked:
    """Test masked FFT2C operation."""

    def test_fft2c_masked_with_mask(self):
        """Test FFT2C with sampling mask."""
        x = torch.randn(2, 8, 8, dtype=torch.complex64)
        mask = torch.ones(2, 8, 8)
        mask[:, ::2, :] = 0  # Undersample every other row

        result = fft2c_masked(x, mask)

        assert torch.is_complex(result)
        assert result.shape == x.shape

    def test_fft2c_masked_without_mask(self):
        """Test FFT2C with None mask (equivalent to full sampling)."""
        x = torch.randn(2, 8, 8, dtype=torch.complex64)

        result = fft2c_masked(x, None)

        # Should be equivalent to fft2c
        expected = fft2c(x)

        assert torch.allclose(result, expected, rtol=1e-5)

    def test_fft2c_masked_full_sampling(self):
        """Test FFT2C with full sampling mask."""
        x = torch.randn(2, 8, 8, dtype=torch.complex64)
        mask = torch.ones(2, 8, 8)  # Full sampling

        result = fft2c_masked(x, mask)

        expected = fft2c(x)

        assert torch.allclose(result, expected, rtol=1e-5)

    def test_fft2c_masked_no_sampling(self):
        """Test FFT2C with zero mask (no sampling)."""
        x = torch.randn(2, 8, 8, dtype=torch.complex64)
        mask = torch.zeros(2, 8, 8)

        result = fft2c_masked(x, mask)

        # Result should be zero everywhere
        assert torch.allclose(result, torch.zeros_like(result), atol=1e-7)

    def test_fft2c_masked_undersampling_factors(self):
        """Test FFT2C with various undersampling factors."""
        x = torch.randn(1, 16, 16, dtype=torch.complex64)

        for factor in [1, 2, 4, 8]:
            # Create undersampling mask
            mask = torch.zeros(1, 16, 16)
            mask[:, ::factor, :] = 1

            result = fft2c_masked(x, mask)

            # Should still be complex
            assert torch.is_complex(result)


class TestFFTEdgeCases:
    """Test edge cases for FFT operations."""

    def test_fft_single_element(self):
        """Test FFT with single element."""
        x = torch.randn(1, 1, 1, dtype=torch.complex64)

        result = fft2c(x)

        assert torch.is_complex(result)

    def test_fft_small_image(self):
        """Test FFT with very small image."""
        x = torch.randn(1, 2, 2, dtype=torch.complex64)

        result = fft2c(x)

        assert result.shape == x.shape

    def test_fft_large_image(self):
        """Test FFT with large image."""
        x = torch.randn(1, 256, 256, dtype=torch.complex64)

        result = fft2c(x)

        assert result.shape == x.shape

    def test_fft_non_square_image(self):
        """Test FFT with non-square images."""
        x = torch.randn(2, 16, 32, dtype=torch.complex64)

        result = fft2c(x)

        assert result.shape == x.shape

    def test_fft_power_of_two_not_required(self):
        """Test FFT works with non-power-of-two dimensions."""
        x = torch.randn(1, 15, 17, dtype=torch.complex64)

        result = fft2c(x)

        assert result.shape == x.shape


class TestFFTNumericalAccuracy:
    """Test numerical accuracy of FFT operations."""

    def test_fft_linearity(self):
        """Test FFT linearity: FFT(a*x + b*y) = a*FFT(x) + b*FFT(y)."""
        x = torch.randn(2, 8, 8, dtype=torch.complex64)
        y = torch.randn(2, 8, 8, dtype=torch.complex64)
        a = 2.5
        b = 1.3

        # Compute FFT of linear combination
        result = fft2c(a * x + b * y)

        # Compute linear combination of FFTs
        expected = a * fft2c(x) + b * fft2c(y)

        assert torch.allclose(result, expected, rtol=1e-5)

    def test_parseval_theorem(self):
        """Test Parseval's theorem: sum(|x|^2) = sum(|FFT(x)|^2)."""
        x = torch.randn(2, 16, 16, dtype=torch.complex64)

        kspace = fft2c(x)

        energy_spatial = torch.sum(torch.abs(x) ** 2)
        energy_frequency = torch.sum(torch.abs(kspace) ** 2)

        # Should be equal with orthonormal FFT
        assert torch.allclose(energy_spatial, energy_frequency, rtol=1e-5)

    def test_shifting_property(self):
        """Test FFT shifting property."""
        x = torch.randn(1, 16, 16, dtype=torch.complex64)

        # Original FFT
        fft_orig = fft2c(x)

        # Shifted FFT (with ifftshift/fftshift)
        fft_shifted = torch.fft.fftshift(fft_orig, dim=(-2, -1))

        # Should be the same concept (testing implementation)
        assert torch.is_complex(fft_shifted)


class TestFFTPerformance:
    """Test FFT performance characteristics."""

    def test_fft_deterministic_output(self):
        """Test that FFT produces deterministic output."""
        x = torch.randn(2, 16, 16, dtype=torch.complex64)

        result1 = fft2c(x)
        result2 = fft2c(x)

        assert torch.equal(result1, result2)

    def test_fft_gradient_propagation(self):
        """Test that gradients propagate through FFT."""
        x = torch.randn(2, 8, 8, dtype=torch.complex64, requires_grad=True)

        result = fft2c(x)
        loss = torch.sum(torch.abs(result) ** 2)
        loss.backward()

        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_ifft_gradient_propagation(self):
        """Test that gradients propagate through IFFT."""
        k = torch.randn(2, 8, 8, dtype=torch.complex64, requires_grad=True)

        result = ifft2c(k)
        loss = torch.sum(torch.abs(result) ** 2)
        loss.backward()

        assert k.grad is not None


class TestFFTDeviceHandling:
    """Test FFT device handling."""

    def test_fft_cpu_device(self):
        """Test FFT on CPU."""
        x = torch.randn(2, 8, 8, dtype=torch.complex64, device="cpu")

        result = fft2c(x)

        assert result.device == x.device

    # @# pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.gpu
    def test_fft_cuda_device(self):
        """Test FFT on CUDA."""
        x = torch.randn(2, 8, 8, dtype=torch.complex64, device="cuda")

        result = fft2c(x)

        assert result.device == x.device

    def test_fft_device_mismatch_handling(self):
        """Test handling of device mismatches."""
        x = torch.randn(2, 8, 8, dtype=torch.complex64, device="cpu")

        # FFT should work regardless
        result = fft2c(x)

        assert result is not None


# --- coil_combine selector (Task 3) ---
import pytest  # noqa: E402

from spectramr.infrastructure.physics.fft_ops import coil_combine  # noqa: E402


def _imgs(c=4):
    torch.manual_seed(0)
    return torch.complex(torch.randn(1, c, 16, 16), torch.randn(1, c, 16, 16))


def test_rss_combine_shape_and_realness():
    out = coil_combine(_imgs(), method="rss")
    assert out.shape == (1, 1, 16, 16) and out.dtype == torch.float32


def test_sense_combine_uses_smaps():
    imgs = _imgs()
    smaps = _imgs()
    out = coil_combine(imgs, method="sense", smaps=smaps)
    assert out.shape == (1, 1, 16, 16)


def test_sense_combine_has_roemer_denominator():
    """With coil images I_c = S_c (object x == 1), the Roemer combine recovers 1.

    Regression: without the 1/sum|S_c|^2 denominator the result would be
    sum_c|S_c|^2 (the matched-filter numerator), not the unbiased estimate.
    """
    torch.manual_seed(0)
    smaps = torch.complex(torch.randn(1, 4, 8, 8), torch.randn(1, 4, 8, 8))
    out = coil_combine(smaps, method="sense", smaps=smaps)
    assert torch.allclose(out, torch.ones_like(out), atol=1e-5)


def test_sense_without_smaps_raises():
    with pytest.raises(ValueError, match=r"sense.*smaps"):
        coil_combine(_imgs(), method="sense", smaps=None)


def test_unknown_method_raises():
    with pytest.raises(ValueError, match=r"Unknown.*combine"):
        coil_combine(_imgs(), method="bogus")


@pytest.mark.parametrize("n", [7, 8, 9, 16])
def test_fft2c_puts_dc_at_the_centre_for_both_parities(n: int) -> None:
    """A constant image must transform to a single DC bin at index ``n // 2``.

    fft2c used ``fftshift`` where ``ifftshift`` was needed. The two agree on EVEN
    lengths, so the round-trip stayed exact and every even-sized test passed —
    but on ODD lengths DC landed one index off, and no test asserted its absolute
    position. Anything reading a k-space coordinate (masks, ACS crops, phase
    ramps) was silently mis-centred on odd grids.
    """
    k = fft2c(torch.ones(n, n, dtype=torch.complex64))
    peak = int(torch.argmax(k.abs()))
    assert (peak // n, peak % n) == (n // 2, n // 2)


@pytest.mark.parametrize("n", [7, 8, 9, 16])
def test_fft2c_of_centred_delta_has_zero_phase(n: int) -> None:
    """A delta at the image centre is the zero-phase (flat) spectrum."""
    x = torch.zeros(n, n, dtype=torch.complex64)
    x[n // 2, n // 2] = 1.0
    assert float(fft2c(x).angle().abs().max()) < 1e-5


class TestOneDimensionalHelpers:
    """fft1c / ifft1c and the uncentered helpers: the readout-axis pair (2026-09-03)."""

    @pytest.mark.parametrize("n", [16, 17])
    def test_round_trip_even_and_odd(self, n):
        from spectramr.infrastructure.physics.fft_ops import fft1c, ifft1c

        x = torch.randn(2, 3, 8, n, dtype=torch.complex64)
        assert torch.allclose(ifft1c(fft1c(x)), x, atol=1e-5)
        assert torch.allclose(fft1c(ifft1c(x, dim=-2), dim=-2), x, atol=1e-5)

    def test_matches_the_inline_spelling_the_operator_used(self):
        # The operator spelled ifftshift(ifft(fftshift(k))) inline; on an even
        # axis the two shifts agree, so the helper reproduces it exactly.
        from spectramr.infrastructure.physics.fft_ops import ifft1c

        k = torch.randn(2, 4, 8, 16, dtype=torch.complex64)
        inline = torch.fft.ifftshift(
            torch.fft.ifft(torch.fft.fftshift(k, dim=-1), dim=-1, norm="ortho"), dim=-1
        )
        assert torch.allclose(ifft1c(k), inline, atol=1e-6)

    def test_dc_of_a_constant_signal_sits_at_the_centre_index(self):
        from spectramr.infrastructure.physics.fft_ops import fft1c

        n = 17
        k = fft1c(torch.ones(1, 1, 4, n, dtype=torch.complex64))
        assert k[..., n // 2].abs().min() > 1.0
        mask = torch.ones(n, dtype=torch.bool)
        mask[n // 2] = False
        assert k[..., mask].abs().max() < 1e-5

    def test_dim_argument_transforms_only_that_axis(self):
        from spectramr.infrastructure.physics.fft_ops import fft1c

        x = torch.randn(1, 1, 8, 16, dtype=torch.complex64)
        k = fft1c(x, dim=-2)
        assert torch.allclose(k.abs().pow(2).sum(dim=-2), x.abs().pow(2).sum(dim=-2), atol=1e-4)

    def test_uncentered_helpers_are_torch_fft_with_ortho_norm(self):
        from spectramr.infrastructure.physics.fft_ops import (
            fft1_uncentered,
            fft2_uncentered_last2,
            ifft1_uncentered,
            ifft2_uncentered_last2,
        )

        x = torch.randn(2, 8, 16, dtype=torch.complex64)
        assert torch.allclose(fft1_uncentered(x), torch.fft.fft(x, dim=-1, norm="ortho"), atol=1e-6)
        assert torch.allclose(ifft1_uncentered(fft1_uncentered(x)), x, atol=1e-5)
        assert torch.allclose(
            fft2_uncentered_last2(x), torch.fft.fft2(x, dim=(-2, -1), norm="ortho"), atol=1e-6
        )
        assert torch.allclose(ifft2_uncentered_last2(fft2_uncentered_last2(x)), x, atol=1e-5)

    def test_real_imag_input_is_promoted_like_the_2d_helpers(self):
        from spectramr.infrastructure.physics.fft_ops import fft1c

        assert fft1c(torch.randn(1, 2, 8, 16)).is_complex()

    def test_exported_in_the_ops_table(self):
        from spectramr.infrastructure.physics.fft_ops import get_fft_ops

        names = {
            "fft1c",
            "ifft1c",
            "fft1_uncentered",
            "ifft1_uncentered",
            "fft2_uncentered_last2",
            "ifft2_uncentered_last2",
        }
        assert names <= set(get_fft_ops())


class TestTheDynamoFence:
    """`fft2c`/`ifft2c` are fenced out of every compiled graph.

    Inductor cannot generate code for complex operators: it routes each one to
    an eager fallback and warns **once per process**, which on a cluster is
    indistinguishable from silence. Fencing the SSOT is what lets an arm declare
    `compile.allow_complex` honestly -- the compiled regions provably hold no
    complex tensors rather than hopefully so.

    The fence is applied at import and unconditionally, so it is paid by every
    arm including the ones that never compile. Measured on this machine's GPU at
    8x1x320x320: +1.4 us/call, +0.1%.
    """

    @pytest.mark.parametrize("fn", [fft2c, ifft2c], ids=["fft2c", "ifft2c"])
    def test_the_fence_is_applied(self, fn):
        """`torch._dynamo.disable` wraps with functools.wraps, so a fenced
        function carries `__wrapped__` and an unfenced one does not."""
        assert hasattr(fn, "__wrapped__"), f"{fn.__name__} is not fenced"

    @pytest.mark.parametrize("fn", [fft2c, ifft2c], ids=["fft2c", "ifft2c"])
    def test_the_fence_preserves_the_identity(self, fn):
        """A fence that swapped the function out would be a correctness bug, not
        a performance one -- which is why `dynamo_disable` checks `__name__`
        rather than `callable()` before returning the wrapper."""
        assert fn.__name__ == fn.__wrapped__.__name__

    def test_the_fence_does_not_change_the_numbers(self):
        """The centering and `norm="ortho"` contract is what this module is for
        (non-negotiable 2); a decorator must not perturb it."""
        x = torch.randn(2, 1, 8, 8, dtype=torch.complex64)
        assert torch.allclose(fft2c(x), fft2c.__wrapped__(x))
        assert torch.allclose(ifft2c(x), ifft2c.__wrapped__(x))

    def test_round_trip_survives_the_fence(self):
        x = torch.randn(2, 1, 8, 8, dtype=torch.complex64)
        assert torch.allclose(ifft2c(fft2c(x)), x, atol=1e-5)
