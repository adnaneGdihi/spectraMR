"""An unknown ``reduction`` must raise at construction, never return an unreduced tensor."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.losses.complex_losses import (
    ComplexL1Loss,
    ComplexMSELoss,
    FrequencyDomainLoss,
    LogSpectralPhaseLoss,
    SENSEAdjointL1Loss,
    WeightedKSpaceL1Loss,
)

LOSSES = [
    ComplexL1Loss,
    ComplexMSELoss,
    FrequencyDomainLoss,
    LogSpectralPhaseLoss,
    SENSEAdjointL1Loss,
    WeightedKSpaceL1Loss,
]


@pytest.mark.parametrize("cls", LOSSES)
def test_a_misspelt_reduction_raises(cls: type) -> None:
    """``reduction='meen'`` must not fall through to the unreduced branch."""
    with pytest.raises(ValueError, match="Unknown reduction"):
        cls(reduction="meen")


@pytest.mark.parametrize("cls", [c for c in LOSSES if c is not SENSEAdjointL1Loss])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_the_three_valid_reductions_still_work(cls: type, reduction: str) -> None:
    out = cls(reduction=reduction)(torch.randn(1, 2, 8, 8), torch.randn(1, 2, 8, 8))
    assert (out.dim() == 0) == (reduction != "none")
