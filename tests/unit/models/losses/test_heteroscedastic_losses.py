"""Tests for the heteroscedastic ULF loss (B-2.9)."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.losses.heteroscedastic_losses import (
    HeteroscedasticULFLoss,
    heteroscedastic_nll,
    heteroscedastic_ulf_loss,
)


def test_nll_minimised_at_correct_mean_and_variance() -> None:
    torch.manual_seed(0)
    target = torch.randn(4, 1, 8, 8)
    # correct mean, unit variance (logvar=0) vs wrong mean
    good = heteroscedastic_nll(target, torch.zeros_like(target), target)
    bad = heteroscedastic_nll(target + 1.0, torch.zeros_like(target), target)
    assert float(good) < float(bad)


def test_variance_prior_pulls_toward_sigma_target() -> None:
    mean = torch.zeros(2, 1, 8, 8)
    target = torch.zeros(2, 1, 8, 8)
    logvar = torch.full((2, 1, 8, 8), 2.0)  # var = e^2 ~ 7.4, far from sigma^2=0.04
    near = heteroscedastic_ulf_loss(
        mean, torch.full_like(logvar, -6.0), target, sigma_target=0.05,
        lambda_var_prior=1.0,
    )["loss_var_prior"]
    far = heteroscedastic_ulf_loss(
        mean, logvar, target, sigma_target=0.05, lambda_var_prior=1.0
    )["loss_var_prior"]
    assert float(near) < float(far)


def test_lambda_zero_disables_prior() -> None:
    out = heteroscedastic_ulf_loss(
        torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 4, 4),
        sigma_target=0.05, lambda_var_prior=0.0,
    )
    assert float(out["loss_var_prior"]) == 0.0


def test_mean_anchor_grounds_the_mean_regardless_of_variance() -> None:
    # Anti-degeneracy (#20): with a HUGE variance the NLL barely penalises a wrong mean,
    # so the bare NLL admits a lazy mean. The L1 mean-anchor makes a wrong mean cost more
    # than a correct one even at high variance.
    target = torch.zeros(2, 1, 8, 8)
    big_logvar = torch.full((2, 1, 8, 8), 2.0)  # clamped upper bound -> tiny inv_var
    good = heteroscedastic_ulf_loss(
        target, big_logvar, target, sigma_target=0.05, lambda_var_prior=0.0, lambda_mean=0.5,
    )
    bad = heteroscedastic_ulf_loss(
        target + 1.0, big_logvar, target, sigma_target=0.05, lambda_var_prior=0.0,
        lambda_mean=0.5,
    )
    assert "loss_mean_anchor" in good
    assert float(good["loss_total"]) < float(bad["loss_total"])


def test_lambda_mean_zero_omits_anchor() -> None:
    out = heteroscedastic_ulf_loss(
        torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 4, 4),
        sigma_target=0.05, lambda_var_prior=0.0, lambda_mean=0.0,
    )
    assert "loss_mean_anchor" not in out


def test_logvar_clamp_bounds_the_variance() -> None:
    # An enormous logvar is clamped to logvar_max before the NLL, so the loss is finite
    # and equals the loss computed with logvar pre-set to the clamp bound.
    mean = torch.zeros(1, 1, 4, 4)
    target = torch.ones(1, 1, 4, 4)
    huge = heteroscedastic_ulf_loss(
        mean, torch.full((1, 1, 4, 4), 1e6), target, sigma_target=0.05,
        lambda_var_prior=0.0, logvar_max=2.0,
    )["loss_nll"]
    at_bound = heteroscedastic_ulf_loss(
        mean, torch.full((1, 1, 4, 4), 2.0), target, sigma_target=0.05,
        lambda_var_prior=0.0, logvar_max=2.0,
    )["loss_nll"]
    assert torch.isfinite(huge)
    assert torch.allclose(huge, at_bound)


def test_per_sample_sigma_target_tensor() -> None:
    out = heteroscedastic_ulf_loss(
        torch.zeros(3, 1, 4, 4), torch.zeros(3, 1, 4, 4), torch.zeros(3, 1, 4, 4),
        sigma_target=torch.tensor([0.05, 0.1, 0.3]), lambda_var_prior=1.0,
    )
    assert torch.isfinite(out["loss_total"])


def test_registered_class_splits_two_channels() -> None:
    loss = HeteroscedasticULFLoss()
    pred = torch.cat([torch.zeros(2, 1, 8, 8), torch.zeros(2, 1, 8, 8)], dim=1)
    val = loss(prediction=pred, target=torch.zeros(2, 1, 8, 8), sigma_target=0.05)
    assert torch.isfinite(val)


def test_registered_via_framework_create_loss() -> None:
    from spectramr.models.losses import create_loss

    assert isinstance(create_loss("heteroscedastic_ulf"), HeteroscedasticULFLoss)


def test_single_channel_prediction_raises_instead_of_returning_nan() -> None:
    """A 1-channel prediction carries no variance head, so it must RAISE.

    Regression for cluster job 8004252. ``prediction[:, 1:2]`` on a C=1 tensor
    is an EMPTY tensor, so ``logvar`` was empty, every reduction over it was
    nan, and the loss reported nan with a zero gradient rather than rejecting
    the input. Same mechanism as the ``hyperelastic_jacobian`` and
    ``gradient_hardware_compliance`` regressions landed alongside it: an
    out-of-contract axis silently produces an empty slice, and the ``.mean()``
    of an empty tensor is nan (non-negotiable #3).
    """
    loss = HeteroscedasticULFLoss()
    with pytest.raises(ValueError, match="2-channel"):
        loss(prediction=torch.zeros(2, 1, 8, 8), target=torch.zeros(2, 1, 8, 8))


@pytest.mark.parametrize(
    ("key", "value"), [("sigma_target", "0.05"), ("lambda_var_prior", "0.1")]
)
def test_registered_loss_rejects_non_numeric_kwargs(key: str, value: str) -> None:
    prediction = torch.zeros(2, 2, 4, 4)
    target = torch.zeros(2, 1, 4, 4)
    with pytest.raises(TypeError, match=key):
        HeteroscedasticULFLoss()(prediction, target, **{key: value})
