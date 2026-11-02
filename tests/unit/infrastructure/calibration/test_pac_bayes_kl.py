"""Tests for the R5 PAC-Bayes inputs: KL(Q||P) and the Monte-Carlo Gibbs risk."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from spectramr.infrastructure.calibration.pac_bayes_kl import (
    gibbs_per_image_loss,
    group_draws,
    isotropic_gaussian_kl,
    nmse_per_image,
    resolve_kl_estimator,
)


def test_equal_sigma_kl_is_scaled_squared_distance() -> None:
    q = {"w": torch.tensor([1.0, 2.0]), "b": torch.tensor([0.5])}
    p = {"w": torch.zeros(2), "b": torch.zeros(1), "running_mean": torch.ones(3)}
    kl, n, dist = isotropic_gaussian_kl(q, p, posterior_sigma=0.1, prior_sigma=0.1)
    assert kl == pytest.approx(5.25 / (2 * 0.01))
    assert n == 3 and dist == pytest.approx(math.sqrt(5.25))


def test_kl_adds_the_variance_mismatch_term() -> None:
    q = {"w": torch.zeros(4)}
    kl, _, _ = isotropic_gaussian_kl(q, {"w": torch.zeros(4)}, posterior_sigma=0.5, prior_sigma=1.0)
    r = 0.25
    assert kl == pytest.approx(4 * 0.5 * (r - 1 - math.log(r)))


@pytest.mark.parametrize(
    "prior", [{}, {"w": torch.zeros(3)}], ids=["missing_param", "shape_mismatch"]
)
def test_kl_refuses_a_partial_or_mismatched_prior(prior) -> None:
    with pytest.raises(ValueError, match="prior"):
        isotropic_gaussian_kl({"w": torch.zeros(2)}, prior, posterior_sigma=1.0, prior_sigma=1.0)


def test_unknown_estimator_raises() -> None:
    assert resolve_kl_estimator("isotropic_gaussian") is isotropic_gaussian_kl
    with pytest.raises(ValueError, match="unknown"):
        resolve_kl_estimator("laplace")


def test_nmse_per_image_and_shape_guard() -> None:
    t = torch.ones(2, 1, 4, 4)
    assert torch.allclose(nmse_per_image(t * 1.5, t), torch.full((2,), 0.25))
    with pytest.raises(ValueError, match="shape"):
        nmse_per_image(torch.ones(2, 2, 4, 4), t)


def _predict(batch, net):
    return net(batch["input"]), batch["target"]


def test_gibbs_loss_is_seeded_and_restores_weights() -> None:
    net = nn.Linear(4, 4)
    before = [p.detach().clone() for p in net.parameters()]
    batches = [{"input": torch.rand(3, 4), "target": torch.rand(3, 4) + 1.0}]
    kw = {
        "risk_loss": "nmse",
        "posterior_sigma": 0.05,
        "n_mc_samples": 4,
        "seed": 1,
        "loss_bound": 10.0,
        "out_of_range": "raise",
    }
    first = gibbs_per_image_loss(net, batches, _predict, **kw)
    again = gibbs_per_image_loss(net, batches, _predict, **kw)
    assert first.per_image == again.per_image and len(first.per_image) == 3
    assert first.mc_standard_error > 0.0
    assert all(torch.equal(a, b) for a, b in zip(before, net.parameters(), strict=True))


def test_gibbs_loss_truncates_only_when_declared() -> None:
    net = nn.Linear(4, 4)
    batches = [{"input": torch.rand(2, 4) * 100, "target": torch.full((2, 4), 0.01)}]
    kw = {
        "risk_loss": "nmse",
        "posterior_sigma": 0.01,
        "n_mc_samples": 1,
        "seed": 0,
        "loss_bound": 1.0,
    }
    with pytest.raises(ValueError, match="exceed"):
        gibbs_per_image_loss(net, batches, _predict, out_of_range="raise", **kw)
    out = gibbs_per_image_loss(net, batches, _predict, out_of_range="truncate", **kw)
    assert out.per_image == [1.0, 1.0] and out.n_truncated == 2


def test_group_draws_averages_per_subject() -> None:
    assert group_draws([0.1, 0.3, 0.5], ["a", "a", "b"]) == pytest.approx([0.2, 0.5])
    assert group_draws([0.1, 0.3], None) == [0.1, 0.3]
