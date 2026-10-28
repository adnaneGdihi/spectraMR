"""per_slice adapter: exact fold round trip, realisation sharing, and kwarg folding."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from spectramr.infrastructure.physics.degradations.adapters import Fold, per_slice
from spectramr.infrastructure.physics.degradations.geometry import (
    AcquisitionGeometry,
    AcquisitionMode,
)
from spectramr.infrastructure.physics.per_sample_rng import per_sample_randn

GEOM = AcquisitionGeometry(mode=AcquisitionMode.MULTISLICE_2D, voxel_mm=None)
B, C, D, H, W = 2, 3, 4, 5, 6


def _identity(x: torch.Tensor, theta: float, *, seed: int, **_: Any) -> torch.Tensor:
    return x


def _channel_shared_noise(x: torch.Tensor, theta: float, *, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    noise = per_sample_randn(x.shape[0], 1, *x.shape[-2:], generator=g)
    return x + theta * noise


def _vol() -> torch.Tensor:
    return torch.randn(B, C, D, H, W, generator=torch.Generator().manual_seed(1))


def test_fold_values() -> None:
    assert Fold.INDEPENDENT.value == "batch" and Fold.SHARED.value == "channel"


@pytest.mark.parametrize("fold", list(Fold))
def test_identity_round_trip_bit_identical(fold: Fold) -> None:
    x = _vol()
    y = per_slice(_identity, fold)(x, 0.5, 0, GEOM)
    assert y.shape == x.shape and torch.equal(y, x)


def test_fold_layouts_seen_by_kernel() -> None:
    x = _vol()
    seen: dict[Fold, torch.Tensor] = {}

    def capture(fold: Fold):
        def kernel(x2d: torch.Tensor, theta: float, *, seed: int) -> torch.Tensor:
            seen[fold] = x2d
            return x2d

        return kernel

    for fold in Fold:
        per_slice(capture(fold), fold)(x, 0.0, 0, GEOM)
    assert torch.equal(seen[Fold.INDEPENDENT], x.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W))
    shared = seen[Fold.SHARED]
    assert shared.shape == (B, C * D, H, W)
    assert torch.equal(shared[:, 1 * D + 2], x[:, 1, 2])  # channel c, slice d -> c*D + d


def test_shared_gives_one_realisation_independent_gives_d() -> None:
    x = torch.zeros(B, C, D, H, W)
    shared = per_slice(_channel_shared_noise, Fold.SHARED)(x, 1.0, 7, GEOM)
    indep = per_slice(_channel_shared_noise, Fold.INDEPENDENT)(x, 1.0, 7, GEOM)
    for b in range(B):
        for d in range(1, D):
            assert torch.equal(shared[b, :, d], shared[b, :, 0])
            assert not torch.equal(indep[b, :, d], indep[b, :, 0])
    assert not torch.equal(shared[0], shared[1])


@pytest.mark.parametrize("fold", list(Fold))
@pytest.mark.parametrize("map_channels", [1, C])
def test_volume_kwargs_folded_like_x(fold: Fold, map_channels: int) -> None:
    x = _vol()
    vmap = torch.randn(B, map_channels, D, H, W)

    def kernel(x2d: torch.Tensor, theta: float, *, seed: int, field: torch.Tensor) -> torch.Tensor:
        return x2d + field

    y = per_slice(kernel, fold, volume_kwargs=("field",))(x, 0.0, 0, GEOM, field=vmap)
    assert torch.equal(y, x + vmap)


def test_non_tensor_kwargs_pass_through() -> None:
    def kernel(x2d: torch.Tensor, theta: float, *, seed: int, scale: float) -> torch.Tensor:
        return x2d * scale

    x = _vol()
    assert torch.equal(per_slice(kernel, Fold.INDEPENDENT)(x, 0.0, 0, GEOM, scale=2.0), x * 2.0)


def test_unknown_5d_kwarg_raises() -> None:
    with pytest.raises(ValueError, match="field"):
        per_slice(_identity, Fold.INDEPENDENT)(
            _vol(), 0.0, 0, GEOM, field=torch.zeros(B, 1, D, H, W)
        )


def test_named_volume_kwarg_wrong_rank_raises() -> None:
    impl = per_slice(_identity, Fold.SHARED, volume_kwargs=("field",))
    with pytest.raises(ValueError, match="field"):
        impl(_vol(), 0.0, 0, GEOM, field=torch.zeros(B, 1, H, W))


def test_named_volume_kwarg_wrong_shape_raises() -> None:
    impl = per_slice(_identity, Fold.SHARED, volume_kwargs=("field",))
    with pytest.raises(ValueError, match="field"):
        impl(_vol(), 0.0, 0, GEOM, field=torch.zeros(B, 2, D, H, W))


def test_4d_input_raises() -> None:
    with pytest.raises(ValueError, match="5-D"):
        per_slice(_identity, Fold.INDEPENDENT)(torch.zeros(B, C, H, W), 0.0, 0, GEOM)


def test_non_fold_raises() -> None:
    with pytest.raises(ValueError):
        per_slice(_identity, "batch")  # type: ignore[arg-type]
