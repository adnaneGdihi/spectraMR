"""``StructureTensorAttention``: the frozen-gamma knob makes gamma=0 an exact ablation.

Paired with ``src/spectramr/models/blocks/structure_tensor_attention.py``.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.blocks.structure_tensor_attention import StructureTensorAttention

_WINDOW = (1, 2, 2)


def _attn(**kw) -> StructureTensorAttention:
    torch.manual_seed(0)
    return StructureTensorAttention(8, window_size=_WINDOW, num_heads=2, **kw)


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(1)
    return torch.randn(3, 4, 8), torch.randn(3, 6, 4)


def test_a_trainable_gamma_is_a_parameter() -> None:
    attn = _attn(gamma=1.0)
    assert "gamma_param" in dict(attn.named_parameters())


def test_a_frozen_gamma_is_a_buffer_and_never_moves() -> None:
    attn = _attn(gamma=0.0, gamma_trainable=False)
    assert "gamma_param" not in dict(attn.named_parameters())
    assert "gamma_param" in dict(attn.named_buffers())
    x, s = _inputs()
    attn(x, s).sum().backward()
    assert float(attn.gamma_param) == 0.0


def test_frozen_zero_ignores_the_structure_tensor() -> None:
    """At gamma=0 frozen the block is plain window attention: S cannot change it."""
    attn = _attn(gamma=0.0, gamma_trainable=False)
    x, s = _inputs()
    with torch.no_grad():
        torch.testing.assert_close(attn(x, s), attn(x, s * 5.0 + 1.0))


def test_a_nonzero_gamma_reads_the_structure_tensor() -> None:
    attn = _attn(gamma=1.0, gamma_trainable=False)
    x, s = _inputs()
    with torch.no_grad():
        assert not torch.allclose(attn(x, s), attn(x, s * 5.0 + 1.0))


def test_a_non_bool_flag_raises() -> None:
    with pytest.raises(TypeError, match="gamma_trainable must be a bool"):
        _attn(gamma_trainable="false")
