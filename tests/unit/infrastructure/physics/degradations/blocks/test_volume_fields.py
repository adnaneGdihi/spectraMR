"""Per-sample smooth 3-D fields and the mm coordinate grid."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from spectramr.infrastructure.physics.degradations.blocks.volume_fields import (
    mm_grid,
    polynomial_field,
    smooth_random_field,
    upsampled_noise_field,
)

SHAPE = (4, 6, 8)
FIELDS: list[Callable[..., torch.Tensor]] = [
    polynomial_field,
    smooth_random_field,
    upsampled_noise_field,
]


def test_mm_grid_zero_at_centre_with_voxel_spacing() -> None:
    voxel = (3.0, 0.5, 2.0)
    grids = mm_grid(SHAPE, voxel)
    for axis, (grid, n, step) in enumerate(zip(grids, SHAPE, voxel, strict=True)):
        assert grid.shape == SHAPE
        line = grid.movedim(axis, 0)[:, 0, 0]
        assert line[n // 2] == 0.0
        torch.testing.assert_close(line.diff(), torch.full((n - 1,), step))


def test_mm_grid_normalised_without_voxel() -> None:
    z, y, x = mm_grid(SHAPE, None)
    torch.testing.assert_close(x[0, 0], 2 * (torch.arange(8.0) - 4) / 8)
    torch.testing.assert_close(z[:, 0, 0], 2 * (torch.arange(4.0) - 2) / 4)
    for grid in (z, y, x):
        assert grid.min() == -1.0 and grid.max() < 1.0


@pytest.mark.parametrize("field", FIELDS)
def test_field_shape_and_unit_peak(field: Callable[..., torch.Tensor]) -> None:
    out = field(3, SHAPE, generator=torch.Generator().manual_seed(0))
    assert out.shape == (3, 1, *SHAPE) and out.dtype == torch.float32
    peak = out.abs().amax(dim=(1, 2, 3, 4))
    torch.testing.assert_close(peak, torch.ones(3))


@pytest.mark.parametrize("field", FIELDS)
def test_field_is_batch_invariant(field: Callable[..., torch.Tensor]) -> None:
    alone = field(1, SHAPE, generator=torch.Generator().manual_seed(7))
    batch = field(3, SHAPE, generator=torch.Generator().manual_seed(7))
    torch.testing.assert_close(batch[:1], alone, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("field", FIELDS)
def test_field_differs_across_seeds(field: Callable[..., torch.Tensor]) -> None:
    a = field(1, SHAPE, generator=torch.Generator().manual_seed(1))
    b = field(1, SHAPE, generator=torch.Generator().manual_seed(2))
    assert (a - b).abs().max() > 1e-3


@pytest.mark.parametrize("field", FIELDS)
def test_field_dtype_follows_request(field: Callable[..., torch.Tensor]) -> None:
    out = field(2, SHAPE, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    assert out.dtype == torch.float64


def test_polynomial_field_is_zero_mean() -> None:
    out = polynomial_field(2, SHAPE, order=3, generator=torch.Generator().manual_seed(0))
    torch.testing.assert_close(out.mean(dim=(1, 2, 3, 4)), torch.zeros(2), atol=1e-6, rtol=0.0)


def test_polynomial_order_one_is_affine() -> None:
    out = polynomial_field(1, SHAPE, order=1, generator=torch.Generator().manual_seed(0))[0, 0]
    # An affine field has a constant first difference along every axis.
    for axis in range(3):
        diff = out.diff(dim=axis)
        torch.testing.assert_close(diff, diff.mean().expand_as(diff), atol=1e-6, rtol=0.0)


@pytest.mark.parametrize(
    ("field", "kwargs"),
    [
        (polynomial_field, {"order": 0}),
        (smooth_random_field, {"n_modes": 0}),
        (smooth_random_field, {"max_cycles": 0.0}),
        (upsampled_noise_field, {"coarse": (0, 2, 2)}),
    ],
)
def test_invalid_arguments_raise(field: Callable[..., torch.Tensor], kwargs: dict) -> None:
    with pytest.raises(ValueError):
        field(1, SHAPE, **kwargs)
