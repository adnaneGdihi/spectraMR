"""SE3EquivarianceDefectLoss caches its fixed-seed rotations instead of redrawing them."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.losses.se3_equivariance_defect import SE3EquivarianceDefectLoss


@pytest.mark.parametrize("group", ["SO2", "SO3"])
def test_cached_rotations_equal_a_fresh_draw(group: str) -> None:
    loss_fn = SE3EquivarianceDefectLoss(group=group, num_rotations=5, seed=3)
    device = torch.device("cpu")
    first = loss_fn._sample_rotations(device, torch.float64)
    second = loss_fn._sample_rotations(device, torch.float64)
    assert first is second
    torch.testing.assert_close(first, loss_fn._draw_rotations(device, torch.float64))


@pytest.mark.parametrize("group", ["SO2", "SO3"])
def test_rotations_are_proper(group: str) -> None:
    rots = SE3EquivarianceDefectLoss(group=group, num_rotations=6)._sample_rotations(
        torch.device("cpu"), torch.float64
    )
    # SO3 is drawn in float32 before the cast, so orthogonality holds to float32 accuracy.
    eye = torch.eye(3, dtype=torch.float64).expand(6, 3, 3)
    torch.testing.assert_close(rots @ rots.transpose(-1, -2), eye, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(
        torch.linalg.det(rots), torch.ones(6, dtype=torch.float64), atol=1e-6, rtol=0.0
    )


def test_the_cache_is_keyed_by_dtype() -> None:
    loss_fn = SE3EquivarianceDefectLoss()
    device = torch.device("cpu")
    assert loss_fn._sample_rotations(device, torch.float32).dtype == torch.float32
    assert loss_fn._sample_rotations(device, torch.float64).dtype == torch.float64


def test_repeated_calls_give_the_same_defect() -> None:
    torch.manual_seed(0)
    weight = torch.randn(6, 6)

    def g(u: torch.Tensor) -> torch.Tensor:
        return u @ weight

    u = torch.randn(2, 4, 6)
    loss_fn = SE3EquivarianceDefectLoss(num_rotations=4)
    first = loss_fn(u, equivariant_fn=g)
    torch.testing.assert_close(loss_fn(u, equivariant_fn=g), first)
    assert first.item() > 0.0
