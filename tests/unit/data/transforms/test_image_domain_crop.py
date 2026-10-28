"""The fastMRI field of view is fixed in image space, not by slicing k-space."""

from __future__ import annotations

import pytest
import torch

from spectramr.data.transforms.image_domain_crop import (
    crop_or_pad_kspace_in_image_domain,
    validate_image_size,
)
from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c


def _kspace_of(image: torch.Tensor) -> torch.Tensor:
    """``(C, H, W, D)`` complex image -> k-space in the same layout."""
    return fft2c(image.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


def _image_of(kspace: torch.Tensor) -> torch.Tensor:
    return ifft2c(kspace.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


def _complex(*shape: int) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.complex(torch.randn(*shape), torch.randn(*shape))


def test_the_served_image_is_the_centre_of_the_oversampled_one() -> None:
    image = _complex(4, 40, 24, 2)

    out = crop_or_pad_kspace_in_image_domain(_kspace_of(image), (20, 16))

    assert out.shape == (4, 20, 16, 2)
    torch.testing.assert_close(_image_of(out), image[:, 10:30, 4:20, :], atol=1e-5, rtol=1e-5)


def test_phase_survives_the_crop() -> None:
    image = _complex(1, 16, 16, 1)

    out = _image_of(crop_or_pad_kspace_in_image_domain(_kspace_of(image), (8, 8)))

    torch.testing.assert_close(
        torch.angle(out), torch.angle(image[:, 4:12, 4:12]), atol=1e-4, rtol=0
    )


def test_a_narrow_axis_is_zero_padded_in_image_space() -> None:
    image = _complex(2, 12, 6, 1)

    out = _image_of(crop_or_pad_kspace_in_image_domain(_kspace_of(image), (8, 10)))

    assert out.shape == (2, 8, 10, 1)
    torch.testing.assert_close(out[:, :, 2:8], image[:, 2:10], atol=1e-5, rtol=1e-5)
    assert out[:, :, :2].abs().max() < 1e-5
    assert out[:, :, 8:].abs().max() < 1e-5


def test_matching_size_is_returned_unchanged() -> None:
    kspace = _complex(2, 8, 8, 1)

    assert crop_or_pad_kspace_in_image_domain(kspace, (8, 8)) is kspace


def test_a_kspace_domain_crop_is_not_what_is_served() -> None:
    """PLANTED VIOLATION: slicing the k-space array keeps an oversampled, blurred image."""
    image = _complex(1, 32, 32, 1)
    kspace = _kspace_of(image)

    kspace_sliced = kspace[:, 8:24, 8:24]
    served = crop_or_pad_kspace_in_image_domain(kspace, (16, 16))

    assert not torch.allclose(served, kspace_sliced, atol=1e-3)
    torch.testing.assert_close(_image_of(served), image[:, 8:24, 8:24], atol=1e-5, rtol=1e-5)


def test_real_interleaved_channels_are_refused() -> None:
    with pytest.raises(TypeError, match="complex"):
        crop_or_pad_kspace_in_image_domain(torch.randn(2, 8, 8, 1), (4, 4))


def test_a_tensor_without_the_torchio_layout_is_refused() -> None:
    with pytest.raises(ValueError, match="C, H, W, D"):
        crop_or_pad_kspace_in_image_domain(_complex(8, 8, 1), (4, 4))


@pytest.mark.parametrize("size", [(320,), (320, 0), (320, -1), (320.0, 320), (True, 320), "320"])
def test_a_malformed_size_raises(size: object) -> None:
    with pytest.raises(ValueError, match="image_domain_crop_or_pad"):
        validate_image_size(size)


# --- never-acquired lines stay exact zeros (the acquired-line rule reads them) -----
def _partially_acquired(height: int, width: int, acquired: slice) -> torch.Tensor:
    """``(C, H, W, D)`` k-space whose columns outside ``acquired`` are exact zeros."""
    kspace = _kspace_of(_complex(4, height, width, 2))
    keep = torch.zeros(width, dtype=torch.bool)
    keep[acquired] = True
    return kspace * keep[None, None, :, None]


def _acquired_columns(kspace: torch.Tensor) -> int:
    from spectramr.models.diffusion.acquired_line_budget import acquired_lines_of

    plane = kspace.permute(3, 0, 1, 2)  # (D, C, H, W): the model's plane-last layout
    return int(acquired_lines_of(plane.reshape(1, -1, *plane.shape[-2:]), -1).sum())


def test_a_readout_crop_keeps_the_unacquired_columns_exact_zeros() -> None:
    """fastMRI brain: 640 x 320 cropped on the readout only; 300 of 320 columns acquired."""
    kspace = _partially_acquired(64, 32, slice(1, 31))
    out = crop_or_pad_kspace_in_image_domain(kspace, (32, 32))
    assert _acquired_columns(out) == 30
    assert torch.equal(out[:, :, :1], torch.zeros_like(out[:, :, :1]))


def test_a_phase_encode_crop_maps_the_acquired_band_onto_the_new_grid() -> None:
    """fastMRI knee: 368 columns cropped to 320 in image space; the band rescales."""
    kspace = _partially_acquired(64, 46, slice(4, 42))  # 38 of 46 acquired
    out = crop_or_pad_kspace_in_image_domain(kspace, (32, 40))
    assert _acquired_columns(out) == 33  # 38 * 40 / 46, to the nearest line


def test_a_padded_axis_keeps_its_band_too() -> None:
    kspace = _partially_acquired(16, 24, slice(4, 20))  # 16 of 24
    out = crop_or_pad_kspace_in_image_domain(kspace, (16, 30))
    assert _acquired_columns(out) == 20  # 16 * 30 / 24


def test_planted_the_bare_round_trip_counts_every_column_as_acquired() -> None:
    """PLANTED: what the crop did before it restored the zeros."""
    kspace = _partially_acquired(64, 32, slice(1, 31))
    image = ifft2c(kspace.permute(0, 3, 1, 2))[..., 16:48, :]
    bare = fft2c(image).permute(0, 2, 3, 1)
    assert _acquired_columns(bare) == 32
