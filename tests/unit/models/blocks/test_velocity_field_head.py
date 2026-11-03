"""Tests for the registered ``velocity_field_head`` block."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.blocks import BLOCK_REGISTRY, create_block
from spectramr.models.blocks.velocity_field_head import VelocityFieldHead


def test_registered_and_resolved_through_create_block() -> None:
    assert BLOCK_REGISTRY["velocity_field_head"] is VelocityFieldHead
    head = create_block("velocity_field_head", in_channels=2)
    assert isinstance(head, VelocityFieldHead)


def test_output_shape_and_zero_init_is_identity_velocity() -> None:
    head = create_block("velocity_field_head", in_channels=4, hidden_channels=8)
    v = head(torch.randn(3, 4, 12, 10))
    assert v.shape == (3, 2, 12, 10)
    assert torch.count_nonzero(v) == 0


def test_gradient_reaches_the_zero_initialised_layer() -> None:
    head = create_block("velocity_field_head", in_channels=2)
    head(torch.randn(1, 2, 8, 8)).sum().backward()
    assert head.net[-1].weight.grad is not None
    assert head.net[-1].weight.grad.abs().sum() > 0


@pytest.mark.parametrize("bad", [0, 1, 3])
def test_odd_or_missing_pair_channels_raise(bad: int) -> None:
    with pytest.raises(ValueError, match="moving/fixed pair"):
        VelocityFieldHead(in_channels=bad)


def test_non_2d_input_raises() -> None:
    head = VelocityFieldHead(in_channels=2)
    with pytest.raises(ValueError, match="2-D"):
        head(torch.randn(1, 2, 4, 8, 8))
