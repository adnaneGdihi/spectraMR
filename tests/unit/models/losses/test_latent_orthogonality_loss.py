"""``latent_orthogonality``: the z_p / z_a batch cross-covariance penalty.

Paired with ``src/spectramr/models/losses/latent_orthogonality_loss.py``.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.losses.latent_orthogonality_loss import (
    LatentOrthogonalityLoss,
    latent_cross_covariance,
)

_IMG = torch.zeros(1)


def _posterior(mu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return mu, torch.zeros_like(mu)


def test_decorrelated_blocks_cost_nothing() -> None:
    # Block a varies only along a sample direction orthogonal to block p's.
    mu = torch.tensor([[1.0, 1.0], [-1.0, 1.0], [1.0, -1.0], [-1.0, -1.0]])
    assert latent_cross_covariance(mu, 1) == pytest.approx(0.0, abs=1e-7)


def test_correlated_blocks_are_penalised() -> None:
    torch.manual_seed(0)
    p = torch.randn(8, 2)
    mu = torch.cat([p, 3.0 * p], dim=1)
    assert latent_cross_covariance(mu, 2) > 1.0


def test_the_loss_reads_the_posterior_mean_and_trains_it() -> None:
    torch.manual_seed(1)
    p = torch.randn(6, 3)
    mu = torch.cat([p, p], dim=1).requires_grad_(True)
    loss = LatentOrthogonalityLoss(split=3)(_IMG, _IMG, posterior=_posterior(mu))
    loss.backward()
    assert loss > 0 and mu.grad.abs().sum() > 0


@pytest.mark.parametrize(
    ("mu", "split", "match"),
    [
        (torch.randn(1, 4), 2, "batch of at least 2"),
        (torch.randn(4, 4), 0, r"must lie in \(0, 4\)"),
        (torch.randn(4, 4), 4, r"must lie in \(0, 4\)"),
    ],
)
def test_degenerate_inputs_raise(mu: torch.Tensor, split: int, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        latent_cross_covariance(mu, split)


def test_a_missing_posterior_raises() -> None:
    with pytest.raises(ValueError, match="needs the posterior"):
        LatentOrthogonalityLoss(split=2)(_IMG, _IMG)


def test_the_split_must_be_an_int() -> None:
    with pytest.raises(TypeError, match="must be an int"):
        LatentOrthogonalityLoss(split=2.5)  # type: ignore[arg-type]


def test_it_builds_from_the_registry_with_its_split() -> None:
    from spectramr.models.losses.registry import create_loss

    loss = create_loss("latent_orthogonality", split=7)
    assert isinstance(loss, LatentOrthogonalityLoss) and loss.split == 7
