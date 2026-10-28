"""``LPIPSLoss`` (``models/losses/lpips_loss.py``) over a stub ``lpips`` module."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import spectramr.models.losses.lpips_loss as lpips_mod


class _StubLPIPS(nn.Module):
    """Shaped like ``lpips.LPIPS``: trainable linear heads behind dropout."""

    def __init__(self, net: str = "alex", verbose: bool = False) -> None:
        super().__init__()
        self.lin = nn.Conv2d(3, 1, 1)
        self.drop = nn.Dropout(0.5)
        self.eval()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        diff = (a - b).expand(-1, 3, -1, -1)
        return self.lin(self.drop(diff)).pow(2).mean(dim=(1, 2, 3), keepdim=True)


@pytest.fixture
def stub_lpips(monkeypatch) -> None:
    monkeypatch.setattr(lpips_mod, "_lpips_available", SimpleNamespace(LPIPS=_StubLPIPS))


def test_train_mode_keeps_the_lpips_network_deterministic(stub_lpips) -> None:
    """``lpips.LPIPS`` has dropout on its heads; putting the owning module in
    train mode used to switch it on and make the loss a random variable."""
    loss_fn = lpips_mod.LPIPSLoss().train()
    assert loss_fn.training
    assert not loss_fn.loss_fn.training
    x, y = torch.rand(2, 1, 8, 8), torch.rand(2, 1, 8, 8)
    torch.manual_seed(0)
    first = loss_fn(x, y)
    torch.manual_seed(1)
    torch.testing.assert_close(loss_fn(x, y), first)


def test_lpips_weights_are_frozen_but_the_input_gets_a_gradient(stub_lpips) -> None:
    loss_fn = lpips_mod.LPIPSLoss()
    assert not any(p.requires_grad for p in loss_fn.parameters())
    x = torch.rand(2, 1, 8, 8, requires_grad=True)
    loss = loss_fn(x, torch.rand(2, 1, 8, 8))
    assert isinstance(loss, torch.Tensor)
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_factory_raises_rather_than_substituting_l1(monkeypatch) -> None:
    """An L1 stand-in trained a different objective under the LPIPS name."""
    monkeypatch.setattr(lpips_mod, "_lpips_available", None)
    with pytest.raises(ImportError, match="lpips"):
        lpips_mod.create_lpips_loss()
