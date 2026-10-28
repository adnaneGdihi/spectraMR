"""Fourier-pair adapters (``fft_image_to_kspace`` / ``ifft_kspace_to_image``).

REGRESSION COHORT: the adapters used to pre-cast a real input with
``x.to(torch.complex64)`` before calling ``fft2c`` / ``ifft2c``. That naive cast
PRE-EMPTED the SSOT interleaved-aware ``_to_complex`` inside the centered FFT
ops: a 2-channel real_interleaved tensor (``[real, imag]`` of ONE complex coil)
became 2 zero-imaginary complex channels instead of one de-interleaved complex
channel. Downstream the VF strategy's ``_complex_to_real`` then expanded that to
4 real channels, crashing ``cross_attention_oracle_unet`` (built for
``in_channels=2``) at its first conv (cluster VF smoke 2026-06-16:
exp_vf_method_a / method_c / ib_infonce). The fix delegates coercion to
``fft2c`` / ``ifft2c``.
"""
from __future__ import annotations

import torch

from spectramr.data.adapters.fourier import (
    FFTImageToKspace,
    IFFTKspaceToImage,
    KSpaceToRealImagInterleave,
)
from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c


def test_ifft_adapter_deinterleaves_real_interleaved_to_one_complex_channel():
    """``[B, 2, H, W]`` real_interleaved → ``[B, 1, H, W]`` complex (NOT 2ch)."""
    b, h, w = 2, 16, 16
    ksp = torch.randn(b, 1, h, w, dtype=torch.complex64)
    # real_interleaved: ch0 = real, ch1 = imag of the single complex coil.
    ksp_real = torch.cat([ksp.real, ksp.imag], dim=1)
    assert ksp_real.shape[1] == 2 and not torch.is_complex(ksp_real)

    out = IFFTKspaceToImage()(ksp_real)

    assert torch.is_complex(out)
    assert out.shape == (b, 1, h, w)  # de-interleaved to 1 complex ch (was 2 — the bug)
    assert torch.allclose(out, ifft2c(ksp), atol=1e-4)


def test_fft_adapter_deinterleaves_real_interleaved_to_one_complex_channel():
    """Same contract for the forward FFT adapter."""
    b, h, w = 2, 16, 16
    img = torch.randn(b, 1, h, w, dtype=torch.complex64)
    img_real = torch.cat([img.real, img.imag], dim=1)

    out = FFTImageToKspace()(img_real)

    assert torch.is_complex(out)
    assert out.shape == (b, 1, h, w)
    assert torch.allclose(out, fft2c(img), atol=1e-4)


def test_ifft_adapter_preserves_channels_for_complex_input():
    """A genuinely-complex input is passed straight through (no doubling)."""
    x = torch.randn(2, 1, 16, 16, dtype=torch.complex64)
    out = IFFTKspaceToImage()(x)
    assert out.shape == (2, 1, 16, 16) and torch.is_complex(out)


def test_fourier_adapter_roundtrip_recovers_real_interleaved_image():
    """fft → ifft round-trips a real_interleaved image back to itself."""
    b, h, w = 2, 16, 16
    img = torch.randn(b, 1, h, w, dtype=torch.complex64)
    img_real = torch.cat([img.real, img.imag], dim=1)
    recon = IFFTKspaceToImage()(FFTImageToKspace()(img_real))
    assert torch.allclose(recon, img, atol=1e-4)


def test_kspace_to_real_imag_interleave_doubles_channels_in_re_im_order():
    z = torch.complex(torch.arange(4.0).view(1, 1, 2, 2), torch.ones(1, 1, 2, 2))
    out = KSpaceToRealImagInterleave()(z)
    assert out.shape == (1, 2, 2, 2)
    assert torch.equal(out[:, 0], z.real[:, 0])
    assert torch.equal(out[:, 1], z.imag[:, 0])


def test_kspace_to_real_imag_interleave_rejects_real_input():
    import pytest

    with pytest.raises(ValueError, match="complex k-space"):
        KSpaceToRealImagInterleave()(torch.zeros(1, 1, 2, 2))

