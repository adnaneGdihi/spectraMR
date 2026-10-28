"""3-D pull-back warp: identity, axis order, grid units, dtype and padding validation."""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.degradations.blocks.volume_warp import (
    device_constants,
    mm_to_px,
    warp_volume,
)

SHAPE = (4, 6, 8)


def _zeros(batch: int = 2, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.zeros(batch, 3, *SHAPE, dtype=dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128])
def test_zero_displacement_is_identity(dtype: torch.dtype) -> None:
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 3, *SHAPE, generator=g, dtype=dtype)
    real = torch.float64 if dtype in (torch.float64, torch.complex128) else torch.float32
    out = warp_volume(x, _zeros(dtype=real))
    assert out.dtype == dtype and out.shape == x.shape
    # grid_sample's (g + 1) / 2 * (N - 1) unnormalisation is exact only up to one rounding.
    atol = 1e-12 if real == torch.float64 else 1e-5
    torch.testing.assert_close(out, x, rtol=0.0, atol=atol)


def _stripe(axis: int) -> torch.Tensor:
    """A one-voxel stripe at index 2 along ``axis`` of a (1, 1, D, H, W) volume."""
    x = torch.zeros(1, 1, *SHAPE, dtype=torch.float64)
    index = [slice(None)] * 5
    index[axis] = 2
    x[tuple(index)] = 1.0
    return x


@pytest.mark.parametrize(("channel", "axis"), [(0, 2), (1, 3), (2, 4)])
def test_one_voxel_displacement_moves_only_its_axis(channel: int, axis: int) -> None:
    x = _stripe(axis)
    disp = torch.zeros(1, 3, *SHAPE, dtype=torch.float64)
    disp[:, channel] = 1.0
    out = warp_volume(x, disp)
    # Pull-back: out(r) = x(r + 1) puts the stripe at index 1 along the displaced axis only.
    expected = torch.roll(x, shifts=-1, dims=axis)
    torch.testing.assert_close(out, expected, rtol=0.0, atol=1e-12)


def test_displacement_on_other_axis_leaves_stripe_unchanged() -> None:
    # A W-stripe is invariant along D and H, so dz and dy shifts must not move it.
    x = _stripe(4)
    disp = torch.zeros(1, 3, *SHAPE, dtype=torch.float64)
    disp[:, 0] = 1.0
    disp[:, 1] = 1.0
    torch.testing.assert_close(warp_volume(x, disp), x, rtol=0.0, atol=1e-12)


def test_grid_channel_order_rejects_transposition() -> None:
    # A ramp along each axis distinguishes every channel permutation.
    d, h, w = torch.meshgrid(*(torch.arange(n, dtype=torch.float64) for n in SHAPE), indexing="ij")
    x = (100 * d + 10 * h + w)[None, None]
    disp = torch.zeros(1, 3, *SHAPE, dtype=torch.float64)
    disp[:, 0], disp[:, 1], disp[:, 2] = 0.25, 0.5, 0.75
    out = warp_volume(x, disp, padding_mode="border")
    interior = out[..., :-1, :-1, :-1]
    expected = x[..., :-1, :-1, :-1] + 100 * 0.25 + 10 * 0.5 + 0.75
    torch.testing.assert_close(interior, expected, rtol=0.0, atol=1e-9)


def test_complex_warp_matches_real_and_imag_warped_separately() -> None:
    g = torch.Generator().manual_seed(1)
    x = torch.randn(2, 2, *SHAPE, generator=g, dtype=torch.complex128)
    disp = torch.randn(2, 3, *SHAPE, generator=g, dtype=torch.float64)
    out = warp_volume(x, disp)
    expected = torch.complex(warp_volume(x.real, disp), warp_volume(x.imag, disp))
    torch.testing.assert_close(out, expected, rtol=0.0, atol=1e-12)


def test_axis_of_size_one_gets_zero_shift() -> None:
    x = torch.randn(1, 1, 1, 6, 8, dtype=torch.float64)
    disp = torch.zeros(1, 3, 1, 6, 8, dtype=torch.float64)
    disp[:, 0] = 3.0
    torch.testing.assert_close(warp_volume(x, disp), x, rtol=0.0, atol=1e-12)


def test_unknown_padding_mode_raises() -> None:
    x = torch.zeros(1, 1, *SHAPE)
    with pytest.raises(ValueError, match="padding_mode"):
        warp_volume(x, _zeros(1), padding_mode="wrap")


def test_bad_displacement_shape_raises() -> None:
    x = torch.zeros(2, 1, *SHAPE)
    with pytest.raises(ValueError, match="displacement"):
        warp_volume(x, torch.zeros(2, 2, *SHAPE))


def test_mm_to_px_divides_each_channel_by_its_voxel() -> None:
    disp = torch.ones(1, 3, *SHAPE)
    px = mm_to_px(disp, (3.0, 0.5, 2.0))
    assert torch.allclose(px[:, 0], torch.full_like(px[:, 0], 1 / 3))
    assert torch.allclose(px[:, 1], torch.full_like(px[:, 1], 2.0))
    assert torch.allclose(px[:, 2], torch.full_like(px[:, 2], 0.5))


def test_mm_to_px_rejects_non_positive_voxel() -> None:
    with pytest.raises(ValueError, match="voxel_mm"):
        mm_to_px(torch.ones(1, 3, *SHAPE), (1.0, 0.0, 1.0))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_device_constants_equal_the_host_tensor_bit_for_bit(dtype: torch.dtype) -> None:
    values = (5.0, 0.6875, 1.1, 0.0)
    got = device_constants(values, torch.device("cpu"), dtype)
    assert got.dtype == dtype and torch.equal(got, torch.tensor(values, dtype=dtype))
