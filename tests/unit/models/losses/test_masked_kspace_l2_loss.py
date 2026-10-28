"""``masked_kspace_l2``: scores the second repetition's lines and nothing else.

The claims that make the CoNEX arms comparable: a full mask reduces to the
unmasked ``l2`` exactly, an unselected line carries no gradient, and a missing
mask raises rather than scoring the whole residual (which would silently turn
every arm into the full-target bound).
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.losses.computers.unified_diffusion_reconstruction import (
    _call_safe_loss,
    _complex_safe_mse,
)
from spectramr.models.losses.masked_kspace_l2_loss import MaskedKSpaceL2Loss
from spectramr.models.losses.registry import create_loss, get_loss_capabilities

_B, _C2, _H, _W = 2, 8, 6, 10


def _pair(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(_B, _C2, _H, _W, generator=gen), torch.randn(_B, _C2, _H, _W, generator=gen)


def _line_mask(columns: list[int]) -> torch.Tensor:
    mask = torch.zeros(_B, 1, _H, _W)
    mask[..., columns] = 1.0
    return mask


def test_it_is_registered_as_a_kspace_loss() -> None:
    assert isinstance(create_loss("masked_kspace_l2"), MaskedKSpaceL2Loss)
    caps = get_loss_capabilities("masked_kspace_l2")
    assert getattr(caps.domain, "value", caps.domain) == "kspace"


def test_a_full_mask_is_the_unmasked_l2_exactly() -> None:
    pred, target = _pair()
    full = torch.ones(_B, 1, _H, _W)
    assert torch.allclose(
        MaskedKSpaceL2Loss()(pred, target, loss_mask=full), _complex_safe_mse(pred, target)
    )


def test_it_is_the_mean_over_the_selected_entries() -> None:
    pred, target = _pair()
    columns = [1, 4, 7]
    expected = (pred - target)[..., columns].pow(2).mean()
    loss = MaskedKSpaceL2Loss()(pred, target, loss_mask=_line_mask(columns))
    assert torch.allclose(loss, expected)


def test_an_unselected_line_carries_no_gradient() -> None:
    pred, target = _pair()
    pred.requires_grad_(True)
    MaskedKSpaceL2Loss()(pred, target, loss_mask=_line_mask([2, 3])).backward()
    grad = pred.grad.abs().sum(dim=(0, 1, 2))
    assert (grad[[2, 3]] > 0).all()
    assert (grad[[0, 1, 4, 5, 6, 7, 8, 9]] == 0).all()


def test_complex_and_interleaved_layouts_agree() -> None:
    pred, target = _pair()
    as_complex = [torch.complex(t[:, 0::2], t[:, 1::2]) for t in (pred, target)]
    mask = _line_mask([0, 5, 9])
    loss = MaskedKSpaceL2Loss()
    assert torch.allclose(loss(pred, target, loss_mask=mask), loss(*as_complex, loss_mask=mask))


class TestItRefuses:
    def test_a_missing_mask(self) -> None:
        pred, target = _pair()
        with pytest.raises(ValueError, match="requires loss_mask"):
            MaskedKSpaceL2Loss()(pred, target)

    def test_a_mask_that_does_not_broadcast(self) -> None:
        pred, target = _pair()
        with pytest.raises(ValueError, match="does not broadcast"):
            MaskedKSpaceL2Loss()(pred, target, loss_mask=torch.ones(_B, 1, _H, _W + 1))

    def test_a_complex_mask(self) -> None:
        pred, target = _pair()
        with pytest.raises(ValueError, match="real"):
            MaskedKSpaceL2Loss()(pred, target, loss_mask=torch.ones(_B, 1, _H, _W) + 0j)


class TestTheComputerHop:
    """``_call_safe_loss`` narrows kwargs to the signature: the mask must survive it."""

    def test_the_mask_is_delivered_by_name(self) -> None:
        pred, target = _pair()
        mask = _line_mask([3])
        via_hop = _call_safe_loss(
            MaskedKSpaceL2Loss(), pred, target, loss_mask=mask, coil_sensitivities=None
        )
        assert torch.allclose(via_hop, MaskedKSpaceL2Loss()(pred, target, loss_mask=mask))

    def test_without_the_mask_the_hop_still_raises(self) -> None:
        """A ValueError, the one exception class the recon computer re-raises."""
        pred, target = _pair()
        with pytest.raises(ValueError):
            _call_safe_loss(MaskedKSpaceL2Loss(), pred, target, coil_sensitivities=None)
