"""Cycle-consistency and identity losses (``models/losses/translation_losses.py``)."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.losses.translation_losses import (
    CycleConsistencyLoss,
    IdentityPreservationLoss,
)


@pytest.mark.parametrize("cls", [CycleConsistencyLoss, IdentityPreservationLoss])
def test_an_unknown_norm_raises(cls: type) -> None:
    """IdentityPreservationLoss accepted any string and computed MSE for it."""
    with pytest.raises(ValueError, match="norm must be"):
        cls(norm="l3")


@pytest.mark.parametrize("cls", [CycleConsistencyLoss, IdentityPreservationLoss])
def test_l1_and_l2_norms(cls: type) -> None:
    pred, target = torch.zeros(4), torch.full((4,), 2.0)
    torch.testing.assert_close(cls(norm="l1")(pred, target), torch.tensor(2.0))
    torch.testing.assert_close(cls(norm="l2")(pred, target), torch.tensor(4.0))
