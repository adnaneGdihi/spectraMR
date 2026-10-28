"""Unit tests for the voxelwise Rician Fisher-Rao geodesic loss.

The derivation is guarded three ways: the quadrature J(u) matches a Monte
Carlo estimate of the squared score, J has the Rayleigh (u^2) and Gaussian (1)
limits, and the loss reduces to the MSE at high SNR while departing from it
near the noise floor.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.losses.fisher_rao_geodesic_loss import (
    FisherRaoGeodesicLoss,
    rician_fisher_information,
)
from spectramr.models.losses.registry import LossRegistry, create_loss


class TestRegistration:
    def test_registered_under_canonical_name(self) -> None:
        assert "fisher_rao_geodesic" in LossRegistry.list_available()

    def test_aliases_resolve(self) -> None:
        for alias in ("fisher_rao_flow", "fr_geodesic", "information_geometry_geodesic"):
            loss = create_loss(alias, noise_sigma=0.02)
            assert isinstance(loss, FisherRaoGeodesicLoss)

    def test_domain_is_image(self) -> None:
        assert LossRegistry._loss_domains["fisher_rao_geodesic"]["domain"] == "image"


class TestValidation:
    @pytest.mark.parametrize("sigma", [None, 0.0, -0.1])
    def test_noise_sigma_is_required_and_positive(self, sigma: float | None) -> None:
        with pytest.raises(ValueError, match="noise_sigma"):
            FisherRaoGeodesicLoss(noise_sigma=sigma)

    def test_unknown_likelihood_raises(self) -> None:
        with pytest.raises(ValueError, match="likelihood"):
            FisherRaoGeodesicLoss(noise_sigma=0.02, likelihood="rice")

    def test_invalid_reduction_raises(self) -> None:
        with pytest.raises(ValueError, match="reduction"):
            FisherRaoGeodesicLoss(noise_sigma=0.02, reduction="median")

    def test_shape_mismatch_raises(self) -> None:
        loss = FisherRaoGeodesicLoss(noise_sigma=0.02)
        with pytest.raises(ValueError, match="shape"):
            loss(torch.zeros(1, 1, 8, 8), torch.zeros(1, 1, 4, 4))


class TestFisherInformation:
    def test_limits(self) -> None:
        j = rician_fisher_information(torch.tensor([0.0, 0.05, 32.0]))
        assert j[0].item() == pytest.approx(0.0, abs=1e-12)
        assert (j[1] / 0.05**2).item() == pytest.approx(1.0, abs=0.01)
        assert j[2].item() == pytest.approx(1.0, abs=1e-3)

    @pytest.mark.parametrize("u", [0.5, 1.5, 3.0])
    def test_matches_monte_carlo_score_variance(self, u: float) -> None:
        g = torch.Generator().manual_seed(0)
        n = 400_000
        re = u + torch.randn(n, generator=g, dtype=torch.float64)
        im = torch.randn(n, generator=g, dtype=torch.float64)
        r = (re**2 + im**2).sqrt()
        score = r * torch.special.i1e(r * u) / torch.special.i0e(r * u) - u
        mc = (score**2).mean().item()
        quad = rician_fisher_information(torch.tensor([u])).item()
        assert quad == pytest.approx(mc, rel=0.01)


class TestLoss:
    def test_zero_when_identical(self) -> None:
        x = torch.rand(2, 1, 16, 16)
        assert FisherRaoGeodesicLoss(noise_sigma=0.02)(x, x.clone()).item() == 0.0

    def test_high_snr_limit_is_mse(self) -> None:
        g = torch.Generator().manual_seed(1)
        target = 0.5 + 0.5 * torch.rand(1, 1, 32, 32, generator=g)
        pred = target + 0.05 * torch.randn(1, 1, 32, 32, generator=g)
        loss = FisherRaoGeodesicLoss(noise_sigma=0.005)(pred, target)
        mse = (pred - target).square().mean()
        assert loss.item() == pytest.approx(mse.item(), rel=1e-3)

    def test_gaussian_likelihood_is_exactly_mse(self) -> None:
        g = torch.Generator().manual_seed(2)
        target = torch.rand(1, 1, 16, 16, generator=g)
        pred = torch.rand(1, 1, 16, 16, generator=g)
        loss = FisherRaoGeodesicLoss(noise_sigma=0.05, likelihood="gaussian")(pred, target)
        assert loss.item() == pytest.approx((pred - target).square().mean().item(), rel=1e-4)

    def test_noise_floor_errors_are_discounted(self) -> None:
        sigma = 0.05
        rician = FisherRaoGeodesicLoss(noise_sigma=sigma)
        gaussian = FisherRaoGeodesicLoss(noise_sigma=sigma, likelihood="gaussian")
        dark_t, dark_p = torch.full((1, 1, 4, 4), 0.0), torch.full((1, 1, 4, 4), 0.5 * sigma)
        bright_t, bright_p = torch.full((1, 1, 4, 4), 0.8), torch.full((1, 1, 4, 4), 0.8 + 0.5 * sigma)
        assert rician(dark_p, dark_t) < 0.1 * gaussian(dark_p, dark_t)
        assert rician(bright_p, bright_t).item() == pytest.approx(
            gaussian(bright_p, bright_t).item(), rel=1e-2
        )

    def test_negative_prediction_is_pulled_back(self) -> None:
        loss = FisherRaoGeodesicLoss(noise_sigma=0.02)
        target = torch.full((1, 1, 4, 4), 0.3)
        pred = torch.full((1, 1, 4, 4), -0.3, requires_grad=True)
        loss(pred, target).backward()
        assert pred.grad is not None and (pred.grad < 0).all()

    def test_gradient_reaches_prediction_and_sum_reduction(self) -> None:
        target = torch.rand(1, 1, 8, 8)
        pred = torch.rand(1, 1, 8, 8, requires_grad=True)
        mean_loss = FisherRaoGeodesicLoss(noise_sigma=0.02)(pred, target)
        mean_loss.backward()
        assert pred.grad is not None and pred.grad.abs().sum() > 0
        sum_loss = FisherRaoGeodesicLoss(noise_sigma=0.02, reduction="sum")(pred, target)
        assert sum_loss.item() == pytest.approx(mean_loss.item() * 64, rel=1e-5)
