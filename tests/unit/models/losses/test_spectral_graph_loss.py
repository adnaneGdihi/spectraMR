"""Grad-flow unit tests for SpectralGraphLoss.

The canonical paired test file was missing (the review noted the spectral-graph
loss had no unit test at all). Pin registration and that the Dirichlet-energy
smoothness prior is genuinely differentiable — a zero / detached gradient would
make it inert regardless of the returned scalar.

Note (honesty): despite the "spectral" name this loss is the unnormalised
graph-Laplacian quadratic form ``xᵀLx`` over a k-NN graph — there is no
eigendecomposition and no normalised Laplacian; it is a smoothness prior, not a
spectral (eigenbasis) feature. The tests here reflect that (smoothness ordering
+ grad flow), not any spectral property.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from spectramr.models.losses.registry import list_available  # noqa: E402
from spectramr.models.losses.spectral_graph_loss import SpectralGraphLoss  # noqa: E402


def test_registered() -> None:
    assert "spectral_graph" in list_available()


def test_grad_flows_to_prediction() -> None:
    loss = SpectralGraphLoss(k=4)
    pred = torch.randn(1, 1, 8, 8, requires_grad=True)
    out = loss(pred)
    out.backward()
    assert pred.grad is not None
    assert torch.any(pred.grad != 0)


def test_target_agnostic_smoothness_prior() -> None:
    """The loss is a smoothness prior on pred; a smoother field must not cost
    more than a noisy one."""
    loss = SpectralGraphLoss(k=4)
    noisy = torch.randn(1, 1, 8, 8)
    smooth = torch.ones(1, 1, 8, 8)
    assert float(loss(smooth)) <= float(loss(noisy))


def test_block_path_matches_the_dense_path() -> None:
    """N > 4096 takes the row-block loop; it must equal the dense top-k energy."""
    torch.manual_seed(0)
    x = torch.rand(1, 4100, 2, dtype=torch.float64)
    loss_fn = SpectralGraphLoss(k=3)
    dense = torch.cdist(x, x).pow(2).topk(4, largest=False, dim=-1).values[:, :, 1:].sum()
    torch.testing.assert_close(loss_fn(x), dense)


def test_accepts_registry_kwargs_and_non_contiguous_input() -> None:
    x = torch.rand(1, 1, 8, 8).transpose(-1, -2)
    assert not x.is_contiguous()
    loss_fn = SpectralGraphLoss(k=2)
    torch.testing.assert_close(loss_fn(x, None, mask=None), loss_fn(x.contiguous()))
