"""Target score of ``SteinDiscrepancyLoss``: Woodbury solve vs the dense inverse."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.losses.stein_discrepancy_loss import SteinDiscrepancyLoss


def _dense_score(target: torch.Tensor, x: torch.Tensor, reg: float) -> torch.Tensor:
    n, d = target.shape
    mu = target.mean(dim=0, keepdim=True)
    centered = target - mu
    cov = centered.t() @ centered / max(n - 1, 1) + reg * torch.eye(d, dtype=target.dtype)
    return -(x - mu) @ torch.linalg.inv(cov)


@pytest.mark.parametrize(("n", "d", "reg"), [(2, 16, 1e-2), (4, 64, 1e-2), (8, 40, 1.0)])
def test_woodbury_score_matches_the_dense_precision(n: int, d: int, reg: float) -> None:
    torch.manual_seed(0)
    target = torch.randn(n, d, dtype=torch.float64)
    x = torch.randn(5, d, dtype=torch.float64)
    score_fn = SteinDiscrepancyLoss(target_score_reg=reg)._build_target_score_fn(target)
    torch.testing.assert_close(score_fn(x), _dense_score(target, x, reg), rtol=1e-9, atol=1e-9)


def test_identical_targets_fall_back_to_the_shrinkage_prior() -> None:
    """Zero empirical covariance leaves ``Sigma = reg * I``: score ``-(x - mu) / reg``."""
    target = torch.ones(3, 10)
    x = torch.randn(4, 10)
    score_fn = SteinDiscrepancyLoss(target_score_reg=0.5)._build_target_score_fn(target)
    torch.testing.assert_close(score_fn(x), -(x - 1.0) / 0.5)


def test_full_resolution_image_batch_is_tractable() -> None:
    """At 320 x 320 the dense ``(d, d)`` precision alone is ~42 GB per chunk."""
    torch.manual_seed(0)
    prediction = torch.randn(2, 1, 320, 320, requires_grad=True)
    target = torch.randn(2, 1, 320, 320)
    loss = SteinDiscrepancyLoss()(prediction, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert prediction.grad is not None and bool(torch.isfinite(prediction.grad).all())
