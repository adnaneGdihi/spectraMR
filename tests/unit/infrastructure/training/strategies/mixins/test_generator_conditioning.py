"""conditioning_kwargs passes only the ids a generator's forward names, and refuses a gap."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from spectramr.infrastructure.training.strategies.mixins.generator_conditioning import (
    conditioning_kwargs,
)


class _Plain(nn.Module):
    def forward(self, x):
        return x


class _Vendor(nn.Module):
    def forward(self, x, *, contrast_idx=None, vendor_id=None):
        return x


_BATCH = {
    "input": torch.zeros(2),
    "contrast_idx": torch.tensor([0, 1]),
    "vendor_id": torch.tensor([1, 1]),
}


def test_an_unconditioned_generator_gets_nothing() -> None:
    assert conditioning_kwargs(_Plain(), _BATCH, "cpu") == {}


def test_named_ids_are_read_from_the_batch() -> None:
    out = conditioning_kwargs(_Vendor(), _BATCH, "cpu")
    assert set(out) == {"contrast_idx", "vendor_id"}
    assert out["vendor_id"].tolist() == [1, 1]


def test_a_named_id_the_batch_lacks_raises() -> None:
    with pytest.raises(ValueError, match="vendor_id"):
        conditioning_kwargs(_Vendor(), {"contrast_idx": torch.tensor([0])}, "cpu")
