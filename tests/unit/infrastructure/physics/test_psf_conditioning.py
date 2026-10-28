"""Tests for the forward-PSF conditioning channel (VFU-01)."""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.psf_conditioning import otf_channel
from spectramr.infrastructure.physics.psf_estimation import gaussian_psf


def _kernel(sigma: float, k: int = 9, b: int = 1) -> torch.Tensor:
    return gaussian_psf(k, sigma).reshape(1, 1, 1, k, k).expand(b, -1, -1, -1, -1)


def test_delta_kernel_has_a_flat_unit_mtf() -> None:
    delta = torch.zeros(1, 1, 1, 5, 5)
    delta[..., 2, 2] = 1.0
    assert torch.allclose(otf_channel(delta, (16, 16)), torch.ones(1, 1, 16, 16), atol=1e-5)


def test_mtf_is_one_at_dc_and_falls_with_frequency() -> None:
    m = otf_channel(_kernel(1.0, b=2), (32, 32))
    assert m.shape == (2, 1, 32, 32)
    assert torch.allclose(m[:, 0, 16, 16], torch.ones(2), atol=1e-5)
    assert float(m[0, 0, 16, 0]) < float(m[0, 0, 16, 8]) < 1.0


def test_wider_blur_attenuates_more() -> None:
    narrow = otf_channel(_kernel(0.6), (32, 32))
    wide = otf_channel(_kernel(1.5), (32, 32))
    assert float(wide.mean()) < float(narrow.mean())


def test_refuses_a_spatially_varying_kernel_field() -> None:
    with pytest.raises(ValueError, match="ONE kernel per sample"):
        otf_channel(torch.ones(1, 2, 2, 3, 3) / 9.0, (16, 16))


def test_refuses_a_kernel_larger_than_the_grid() -> None:
    with pytest.raises(ValueError, match="does not fit"):
        otf_channel(_kernel(1.0, k=9), (8, 8))
