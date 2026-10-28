"""Hyena filter helpers."""

import torch

from spectramr.models.blocks.hyena_block import _positional_encoding


def test_positional_encoding_is_cached_per_shape_and_device() -> None:
    a = _positional_encoding(16, 8, torch.device("cpu"))
    assert _positional_encoding(16, 8, torch.device("cpu")) is a
    assert _positional_encoding(17, 8, torch.device("cpu")) is not a


def test_positional_encoding_is_sinusoidal() -> None:
    pe = _positional_encoding(4, 6, torch.device("cpu"))
    assert pe.shape == (4, 6)
    assert torch.allclose(pe[0, 0::2], torch.zeros(3)) and torch.allclose(
        pe[0, 1::2], torch.ones(3)
    )
