"""``InfoNCECritic`` (``models/losses/infonce_critic.py``): configuration and the bank."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from spectramr.models.losses.infonce_critic import InfoNCECritic


def test_an_unknown_negative_pool_raises() -> None:
    """A misspelt pool used to fall through to in-batch negatives silently."""
    with pytest.raises(ValueError, match="negative_pool"):
        InfoNCECritic(negative_pool="momentum_bak", bank_size=8)  # type: ignore[arg-type]


def test_a_momentum_bank_without_capacity_raises() -> None:
    with pytest.raises(ValueError, match="bank_size"):
        InfoNCECritic(negative_pool="momentum_bank", bank_size=0)


def test_the_bank_is_a_fifo_of_the_projected_targets_without_host_reads(monkeypatch) -> None:
    """The ring write keeps its pointer on the host; ``Tensor.item`` is never called."""
    bank_size = 8
    critic = InfoNCECritic(
        critic_hidden_dim=0, feature_dim=4, negative_pool="momentum_bank", bank_size=bank_size
    )

    def _no_sync(self: torch.Tensor) -> float:
        raise AssertionError("InfoNCECritic read a device tensor back to the host")

    monkeypatch.setattr(torch.Tensor, "item", _no_sync)
    torch.manual_seed(0)
    ref = torch.zeros(bank_size, 4)
    ptr = 0
    for b in (3, 5, 8, 2, 11):
        z, y = torch.randn(b, 4), torch.randn(b, 4)
        critic(z, y)
        rows = F.normalize(y, dim=-1)[-bank_size:]
        for row in rows:
            ref[ptr] = row
            ptr = (ptr + 1) % bank_size
        assert critic._bank is not None and critic._bank_ptr is not None
        torch.testing.assert_close(critic._bank, ref)
        assert int(critic._bank_ptr[0]) == ptr


def test_in_batch_loss_is_symmetric_cross_entropy_on_the_diagonal() -> None:
    critic = InfoNCECritic(temperature=0.5, critic_hidden_dim=0)
    torch.manual_seed(0)
    z, y = torch.randn(4, 3), torch.randn(4, 3)
    zn, yn = F.normalize(z, dim=-1), F.normalize(y, dim=-1)
    labels = torch.arange(4)
    want = 0.5 * (
        F.cross_entropy(zn @ yn.T / 0.5, labels) + F.cross_entropy(yn @ zn.T / 0.5, labels)
    )
    torch.testing.assert_close(critic(z, y), want)
