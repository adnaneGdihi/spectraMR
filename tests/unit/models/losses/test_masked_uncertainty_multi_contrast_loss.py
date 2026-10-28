"""Unit tests for MaskedUncertaintyMultiContrastLoss (plan §5.3, P2)."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from spectramr.models.losses.masked_uncertainty_multi_contrast_loss import (  # noqa: E402
    MaskedUncertaintyMultiContrastLoss,
)


def _make_batch(
    batch: int = 4,
    channels: int = 1,
    spatial: tuple[int, ...] = (4, 8, 8),
    n_contrasts: int = 3,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    pred = torch.randn(batch, channels, *spatial, generator=g, requires_grad=True)
    target = torch.randn(batch, channels, *spatial, generator=g)
    mask = (torch.randn(batch, 1, *spatial, generator=g) > 0).to(torch.float32)
    cid = torch.arange(batch, dtype=torch.long) % n_contrasts
    return pred, target, mask, cid


def test_constructor_validates_n_contrasts() -> None:
    with pytest.raises(ValueError, match="n_contrasts"):
        MaskedUncertaintyMultiContrastLoss(n_contrasts=0)


def test_constructor_validates_reduction() -> None:
    with pytest.raises(ValueError, match="reduction"):
        MaskedUncertaintyMultiContrastLoss(n_contrasts=2, reduction="median")


def test_log_sigma_is_learnable_parameter() -> None:
    loss = MaskedUncertaintyMultiContrastLoss(n_contrasts=3)
    params = list(loss.parameters())
    assert len(params) == 1
    assert params[0].shape == (3,)
    assert params[0].requires_grad


def test_per_contrast_sigma_starts_at_one() -> None:
    loss = MaskedUncertaintyMultiContrastLoss(n_contrasts=3, init_log_sigma=0.0)
    sigmas = loss.per_contrast_sigma()
    assert torch.allclose(sigmas, torch.ones(3))


def test_forward_shape_validation() -> None:
    loss = MaskedUncertaintyMultiContrastLoss(n_contrasts=2)
    pred = torch.zeros(2, 1, 4, 4)
    target = torch.zeros(2, 1, 4, 8)
    with pytest.raises(ValueError, match="share shape"):
        loss(pred, target)


def test_forward_validates_contrast_id_range() -> None:
    loss = MaskedUncertaintyMultiContrastLoss(n_contrasts=2)
    pred = torch.zeros(2, 1, 4, 4, requires_grad=True)
    target = torch.zeros(2, 1, 4, 4)
    with pytest.raises(ValueError, match="contrast_id values"):
        loss(pred, target, contrast_id=torch.tensor([0, 5]))


def test_forward_returns_log_sigma_when_residual_is_zero() -> None:
    """With pred == target, the L1 term vanishes and only log_sigma remains."""
    loss = MaskedUncertaintyMultiContrastLoss(n_contrasts=3, init_log_sigma=0.0)
    pred = torch.full((3, 1, 4, 4, 4), 0.5, requires_grad=True)
    target = pred.detach().clone()
    cid = torch.tensor([0, 1, 2])
    val = loss(pred, target, contrast_id=cid)
    # mean over 3 contrasts of (0 + log(1)) = 0.
    assert torch.isclose(val, torch.tensor(0.0), atol=1e-6)


def test_forward_autograd_traces_through_log_sigma() -> None:
    loss = MaskedUncertaintyMultiContrastLoss(n_contrasts=3)
    pred, target, mask, cid = _make_batch(seed=1)
    val = loss(pred, target, mask=mask, contrast_id=cid)
    val.backward()
    assert loss.log_sigma.grad is not None
    assert torch.any(loss.log_sigma.grad != 0)
    # Pred gradient should also flow back.
    assert pred.grad is not None
    assert torch.any(pred.grad != 0)


def test_forward_increases_with_residual_magnitude() -> None:
    loss = MaskedUncertaintyMultiContrastLoss(n_contrasts=1, init_log_sigma=0.0)
    target = torch.zeros(2, 1, 4, 4, 4)
    cid = torch.zeros(2, dtype=torch.long)
    small = loss(0.1 * torch.ones_like(target), target, contrast_id=cid)
    big = loss(1.0 * torch.ones_like(target), target, contrast_id=cid)
    assert big.item() > small.item()


def test_mask_zero_gives_log_sigma_only() -> None:
    """With mask all-zero everywhere, residual is masked out -> only log_sigma."""
    loss = MaskedUncertaintyMultiContrastLoss(n_contrasts=2, init_log_sigma=0.0)
    pred = torch.randn(2, 1, 4, 4, 4, requires_grad=True)
    target = torch.zeros_like(pred)
    mask = torch.zeros(2, 1, 4, 4, 4)
    cid = torch.tensor([0, 1])
    val = loss(pred, target, mask=mask, contrast_id=cid)
    # log_sigma init is 0 for both contrasts; mean is 0.
    assert torch.isclose(val, torch.tensor(0.0), atol=1e-6)


def test_optimum_log_sigma_balances_precision_and_log_term() -> None:
    """At the optimum: 0.5 * (1/sigma^2) * L = 0.5  =>  sigma = sqrt(L)."""
    loss = MaskedUncertaintyMultiContrastLoss(n_contrasts=1, init_log_sigma=0.0)
    target = torch.zeros(1, 1, 4, 4, 4)
    pred = torch.full_like(target, 0.5)  # residual L = 0.5 exactly.
    cid = torch.zeros(1, dtype=torch.long)

    optim = torch.optim.SGD([loss.log_sigma], lr=0.1)
    for _ in range(500):
        optim.zero_grad()
        val = loss(pred, target, contrast_id=cid)
        val.backward()
        optim.step()
    sigma_opt = float(loss.log_sigma.detach().exp())
    # Optimum: d/dlog_sigma [0.5 * exp(-2 log_sigma) * L + log_sigma]
    #        = -L * exp(-2 log_sigma) + 1 = 0
    #        => sigma^2 = L  => sigma = sqrt(0.5)
    assert math.isclose(sigma_opt, math.sqrt(0.5), rel_tol=0.05)


def test_contrast_id_int_scalar_broadcasts() -> None:
    loss = MaskedUncertaintyMultiContrastLoss(n_contrasts=3)
    pred = torch.randn(4, 1, 8, 8, requires_grad=True)
    target = torch.randn_like(pred)
    val = loss(pred, target, contrast_id=2)  # int scalar
    val.backward()
    # Only log_sigma[2] should have a non-zero gradient.
    grad = loss.log_sigma.grad
    assert grad is not None
    assert grad[0].item() == 0.0
    assert grad[1].item() == 0.0
    assert grad[2].item() != 0.0


def test_default_contrast_id_collapses_to_single_sigma() -> None:
    loss = MaskedUncertaintyMultiContrastLoss(n_contrasts=2)
    pred = torch.randn(3, 1, 4, 4, requires_grad=True)
    target = torch.randn_like(pred)
    val = loss(pred, target, contrast_id=None)  # default
    val.backward()
    grad = loss.log_sigma.grad
    assert grad is not None
    assert grad[0].item() != 0.0
    # log_sigma[1] never participates -> zero grad.
    assert grad[1].item() == 0.0


def _reference_loop(
    loss: MaskedUncertaintyMultiContrastLoss,
    per_sample_l1: torch.Tensor,
    cid: torch.Tensor,
) -> torch.Tensor:
    """The per-contrast Python loop the scatter-add path replaces."""
    total = per_sample_l1.new_zeros(())
    present = 0
    for c in range(loss.n_contrasts):
        sel = cid == c
        if not bool(sel.any()):
            continue
        present += 1
        log_sigma_c = loss.log_sigma[c]
        precision = (-2.0 * log_sigma_c).exp().clamp_min(loss.eps)
        total = total + 0.5 * precision * per_sample_l1[sel].mean() + log_sigma_c
    return total / present if loss.reduction == "mean" else total


@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("cid_list", [[0, 1, 2, 0, 1], [2, 2, 2, 2, 2], [0, 2, 0, 2, 0]])
def test_scatter_add_path_matches_the_per_contrast_loop(
    reduction: str, cid_list: list[int]
) -> None:
    g = torch.Generator().manual_seed(3)
    pred = torch.randn(5, 1, 6, 6, generator=g, dtype=torch.float64, requires_grad=True)
    target = torch.randn(5, 1, 6, 6, generator=g, dtype=torch.float64)
    mask = (torch.randn(5, 1, 6, 6, generator=g) > 0).to(torch.float64)
    cid = torch.tensor(cid_list)
    loss = MaskedUncertaintyMultiContrastLoss(n_contrasts=3, reduction=reduction).double()
    with torch.no_grad():
        loss.log_sigma.copy_(torch.tensor([0.3, -0.2, 0.7], dtype=torch.float64))

    out = loss(pred, target, mask=mask, contrast_id=cid)
    grad_pred, grad_sigma = torch.autograd.grad(out, (pred, loss.log_sigma))

    per_sample = ((pred - target).abs() * mask).flatten(1).sum(1) / mask.flatten(1).sum(1)
    ref = _reference_loop(loss, per_sample, cid)
    ref_grad_pred, ref_grad_sigma = torch.autograd.grad(ref, (pred, loss.log_sigma))

    torch.testing.assert_close(out, ref)
    torch.testing.assert_close(grad_pred, ref_grad_pred)
    torch.testing.assert_close(grad_sigma, ref_grad_sigma)
