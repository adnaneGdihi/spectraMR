"""Masks, the learned line prior and the rollouts behind IB active acquisition."""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.training.strategies.ib_acquisition_policy import (
    LinePrior,
    apply_line_mask,
    centre_mask,
    equispaced_line_mask,
    ib_rollout,
    line_budget,
    noise_signal_correlation,
    random_line_masks,
    variance_rollout,
)
from spectramr.infrastructure.training.strategies.ib_active_acquisition_strategy import (
    IBActiveAcquisitionStrategy,
)

W = 32


def test_budget_counts_the_centre_band_inside_the_total() -> None:
    assert line_budget(320, 4.0, 0.08) == (80, 26)
    with pytest.raises(ValueError, match="centre band"):
        line_budget(32, 8.0, 0.5)


def test_every_mask_family_keeps_the_centre_and_meets_the_budget() -> None:
    total, centre = line_budget(W, 4.0, 0.125)
    seed = centre_mask(W, centre)
    masks = random_line_masks(3, W, total, centre, torch.Generator().manual_seed(0))
    assert masks.shape == (3, W)
    assert (masks.sum(1) == total).all() and masks[:, seed].all()
    assert not torch.equal(masks[0], masks[1])
    equi = equispaced_line_mask(W, total, centre)
    assert int(equi.sum()) == total and equi[seed].all()


def test_masks_spend_the_budget_on_acquired_lines_only() -> None:
    total, centre = line_budget(W, 4.0, 0.125)
    support = torch.ones(2, W, dtype=torch.bool)
    support[:, :8] = False
    masks = random_line_masks(2, W, total, centre, torch.Generator().manual_seed(1), support)
    assert not masks[:, :8].any() and (masks.sum(1) == total).all()
    equi = equispaced_line_mask(W, total, centre, support[0])
    assert not equi[:8].any() and int(equi.sum()) == total


def test_apply_line_mask_zeroes_columns_on_the_phase_encode_dim() -> None:
    k = torch.ones(2, 2, 4, W)
    mask = torch.zeros(2, W, dtype=torch.bool)
    mask[0, 3] = True
    out = apply_line_mask(k, mask, pe_dim=3)
    assert out[0, :, :, 3].eq(1).all() and out[0].sum() == 2 * 4
    assert out[1].sum() == 0


def test_prior_reads_per_line_noise_from_the_outer_readout_band() -> None:
    torch.manual_seed(0)
    n_ro = 40
    scale = torch.linspace(0.5, 2.0, W)
    k = torch.randn(64, 2, n_ro, W) * scale
    prior = LinePrior()
    prior.update(k, pe_dim=3, readout_dim=2)
    cov, noise = prior.covariance_and_noise()
    assert cov.shape == (W, W) and noise.shape == (W,)
    # E|y|^2 = 2 * scale^2 for real + imag channels of unit-variance draws.
    assert torch.allclose(noise, 2 * scale.double() ** 2, rtol=0.35)
    restored = LinePrior()
    restored.load_state_dict(prior.state_dict(), torch.device("cpu"))
    assert torch.equal(restored.covariance_and_noise()[0], cov)


def _separable_kspace(n_ro: int, edge_tail: float, sigma: float) -> torch.Tensor:
    """``|y|^2 = f(kx) g(ky)`` with Gaussian g, plus white noise of variance ``2 sigma^2``."""
    kx = torch.linspace(-1, 1, n_ro).abs().unsqueeze(-1)
    ky = torch.linspace(-1, 1, W).unsqueeze(0)
    f = torch.exp(-kx / 0.08) + edge_tail * (kx > 0.9)
    amp = (f * torch.exp(-(ky**2) / 0.1)).sqrt() * 10.0
    return amp * torch.randn(128, 2, n_ro, W) + sigma * torch.randn(128, 2, n_ro, W)


def test_per_line_noise_does_not_track_line_power_when_the_edge_is_noise() -> None:
    torch.manual_seed(0)
    sigma = 0.1
    prior = LinePrior()
    prior.update(_separable_kspace(64, 0.0, sigma), pe_dim=3, readout_dim=2)
    cov, noise = prior.covariance_and_noise()
    assert torch.allclose(noise, torch.full_like(noise, 2 * sigma**2), rtol=0.15)
    assert abs(noise_signal_correlation(cov, noise)) < 0.5


def test_a_signal_tail_in_the_outer_band_shows_in_the_correlation() -> None:
    torch.manual_seed(0)
    prior = LinePrior()
    prior.update(_separable_kspace(64, 1e-2, 0.1), pe_dim=3, readout_dim=2)
    cov, noise = prior.covariance_and_noise()
    assert noise.max() > 3 * noise.min()
    assert noise_signal_correlation(cov, noise) > 0.9


def test_masks_are_built_on_the_support_device_without_a_host_copy() -> None:
    total, centre = line_budget(W, 4.0, 0.125)
    support = torch.ones(2, W, dtype=torch.bool, device="meta")
    masks = random_line_masks(2, W, total, centre, torch.Generator().manual_seed(0), support)
    assert masks.device.type == "meta" and masks.shape == (2, W)


def _prior(noise: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.arange(W, dtype=torch.float64)
    energy = torch.exp(-((idx - W / 2) ** 2) / 60.0) + 0.05 * (idx % 5)
    cov = torch.diag(energy) + 0.02 * torch.exp(-((idx[:, None] - idx[None]) ** 2) / 4.0)
    return cov, noise


def test_ib_rollout_follows_the_learned_prior_not_index_order() -> None:
    cov, noise = _prior(torch.full((W,), 0.01, dtype=torch.float64))
    total, centre = line_budget(W, 4.0, 0.125)
    seed = centre_mask(W, centre)
    mask = ib_rollout(cov, noise, seed, total, 1.0, IBActiveAcquisitionStrategy.select_next_line)
    assert int(mask.sum()) == total
    first_unobserved = (~seed).nonzero().squeeze(-1)[: total - centre]
    index_order = seed.clone()
    index_order[first_unobserved] = True
    assert not torch.equal(mask, index_order)


def test_beta_reorders_lines_whose_noise_differs() -> None:
    noise = torch.linspace(0.001, 0.5, W, dtype=torch.float64)
    cov, noise = _prior(noise)
    total, centre = line_budget(W, 4.0, 0.125)
    seed = centre_mask(W, centre)
    pick = IBActiveAcquisitionStrategy.select_next_line
    low = ib_rollout(cov, noise, seed, total, 0.0, pick)
    high = ib_rollout(cov, noise, seed, total, 5.0, pick)
    assert not torch.equal(low, high)


def test_variance_rollout_picks_the_largest_posterior_variance() -> None:
    cov, noise = _prior(torch.full((W,), 0.01, dtype=torch.float64))
    seed = centre_mask(W, 2)
    mask = variance_rollout(cov, noise, seed, 3)
    var = torch.linalg.inv(torch.linalg.inv(cov) + torch.diag(seed.double() / noise)).diagonal()
    assert mask[int(var.masked_fill(seed, -torch.inf).argmax())]
