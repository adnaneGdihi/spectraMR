"""PrimaryIdealMembershipLoss clusters with a one-hot matmul, not boolean masks.

The reference is the per-cluster boolean-mask Lloyd loop the loss used before;
the vectorised form must land on the same centres and radii.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.losses.primary_ideal_membership_loss import PrimaryIdealMembershipLoss


def _masked_lloyd(
    points: torch.Tensor, k: int, iters: int = 5
) -> tuple[torch.Tensor, torch.Tensor]:
    n = points.shape[0]
    order = torch.argsort(points[:, 0])
    idx = torch.linspace(0, n - 1, k).round().long()
    centers = points[order[idx]].clone()
    for _ in range(iters):
        assign = torch.cdist(points, centers).argmin(dim=1)
        centers = torch.stack(
            [points[assign == c].mean(0) if (assign == c).any() else centers[c] for c in range(k)]
        )
    assign = torch.cdist(points, centers).argmin(dim=1)
    radii = []
    for c in range(k):
        members = points[assign == c]
        if members.numel():
            radii.append((members - centers[c]).pow(2).sum(-1).mean().clamp_min(1e-8).sqrt())
        else:
            radii.append(torch.tensor(1.0, dtype=points.dtype))
    return centers, torch.stack(radii)


@pytest.mark.parametrize("k", [1, 3, 5])
def test_vectorised_lloyd_matches_the_masked_loop(k: int) -> None:
    torch.manual_seed(k)
    points = torch.randn(400, 3, dtype=torch.float64)
    centers, radii = PrimaryIdealMembershipLoss._cluster(points, k)
    ref_centers, ref_radii = _masked_lloyd(points, k)
    torch.testing.assert_close(centers, ref_centers)
    torch.testing.assert_close(radii, ref_radii)


def test_empty_clusters_keep_their_centre_and_unit_radius() -> None:
    points = torch.zeros(10, 2, dtype=torch.float64)
    points[5:] = 1.0
    centers, radii = PrimaryIdealMembershipLoss._cluster(points, 4)
    ref_centers, ref_radii = _masked_lloyd(points, 4)
    torch.testing.assert_close(centers, ref_centers)
    torch.testing.assert_close(radii, ref_radii)
    assert (radii == 1.0).sum() == 2


def test_loss_is_finite_and_differentiable() -> None:
    pred = torch.rand(2, 3, 8, 8, requires_grad=True)
    loss = PrimaryIdealMembershipLoss()(pred, torch.rand(2, 3, 8, 8))
    loss.backward()
    assert torch.isfinite(loss)
    assert pred.grad is not None
    assert pred.grad.abs().sum() > 0
