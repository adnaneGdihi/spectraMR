"""Slice-profile kernels and the along-slice convolution."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from spectramr.infrastructure.physics.degradations.blocks.slice_physics import (
    convolve_along_slices,
    slice_profile_kernel,
)


def _reference_sinc_gauss(fwhm_voxels: float, length: int) -> np.ndarray:
    # The formula slice_profile.py carried before the kernels moved here.
    half = (length - 1) / 2.0
    x = np.arange(length, dtype=np.float64) - half
    sigma = fwhm_voxels / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    k = np.sinc(x / max(fwhm_voxels, 1e-3)) * np.exp(-(x**2) / (2.0 * sigma**2))
    k = k - k.mean()
    k = k + abs(k.min())
    return k / k.sum()


def _reference_gauss(fwhm_voxels: float, length: int) -> np.ndarray:
    half = (length - 1) / 2.0
    x = np.arange(length, dtype=np.float64) - half
    sigma = fwhm_voxels / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    k = np.exp(-(x**2) / (2.0 * sigma**2))
    return k / k.sum()


@pytest.mark.parametrize(("fwhm", "length"), [(1.0, 3), (2.5, 11), (4.0, 21)])
def test_kernels_match_previous_formulas_bit_for_bit(fwhm: float, length: int) -> None:
    np.testing.assert_array_equal(
        slice_profile_kernel("sinc_gauss", fwhm, length), _reference_sinc_gauss(fwhm, length)
    )
    np.testing.assert_array_equal(
        slice_profile_kernel("gauss", fwhm, length), _reference_gauss(fwhm, length)
    )


def test_slr_kernel_is_the_fixed_33_tap_table() -> None:
    k = slice_profile_kernel("slr", 4.0, 21)
    assert k.shape == (33,) and k.dtype == np.float64
    np.testing.assert_array_equal(k, k[::-1])
    assert k[16] == 0.1908


def test_unknown_kind_raises() -> None:
    with pytest.raises(ValueError, match="kind"):
        slice_profile_kernel("box", 2.0, 5)  # type: ignore[arg-type]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.complex64])
def test_delta_kernel_is_identity(dtype: torch.dtype) -> None:
    x = torch.randn(2, 3, 5, 4, 6, dtype=dtype)
    out = convolve_along_slices(x, torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0]))
    assert out.dtype == dtype and out.shape == x.shape
    torch.testing.assert_close(out, x, rtol=0.0, atol=0.0)


def test_box_kernel_averages_neighbouring_slices_with_replicate_edges() -> None:
    x = torch.arange(5, dtype=torch.float64).view(1, 1, 5, 1, 1).expand(1, 1, 5, 2, 3)
    out = convolve_along_slices(x, torch.full((3,), 1.0 / 3.0))
    expected = torch.tensor([1 / 3, 1.0, 2.0, 3.0, 11 / 3], dtype=torch.float64)
    torch.testing.assert_close(out[0, 0, :, 0, 0], expected)
    torch.testing.assert_close(out, out[:, :, :, :1, :1].expand_as(out))


def test_even_kernel_keeps_shape() -> None:
    x = torch.randn(1, 1, 6, 2, 2)
    assert convolve_along_slices(x, torch.full((4,), 0.25)).shape == x.shape


def test_convolve_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match="B, C, D, H, W"):
        convolve_along_slices(torch.zeros(1, 4, 4, 4), torch.ones(3))
    with pytest.raises(ValueError, match="kernel"):
        convolve_along_slices(torch.zeros(1, 1, 4, 4, 4), torch.ones(3, 1))
