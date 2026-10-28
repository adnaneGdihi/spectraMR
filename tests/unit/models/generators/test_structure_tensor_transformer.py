"""``StructureTensorTransformer`` threads ``gamma_trainable`` to every attention block.

Paired with ``src/spectramr/models/generators/structure_tensor_transformer.py``.
"""

from __future__ import annotations

import torch

from spectramr.models.blocks.structure_tensor_attention import StructureTensorAttention
from spectramr.models.generators.structure_tensor_transformer import StructureTensorTransformer


def _model(**kw) -> StructureTensorTransformer:
    torch.manual_seed(0)
    return StructureTensorTransformer(
        in_channels=1,
        out_channels=1,
        embed_dim=8,
        depths=(1, 1),
        num_heads=(1, 2),
        window_size=(1, 2, 2),
        **kw,
    )


def _attention_blocks(model: torch.nn.Module) -> list[StructureTensorAttention]:
    return [m for m in model.modules() if isinstance(m, StructureTensorAttention)]


def test_every_block_freezes_gamma_when_asked() -> None:
    blocks = _attention_blocks(_model(gamma=0.0, gamma_trainable=False))
    assert blocks
    assert all("gamma_param" in dict(b.named_buffers()) for b in blocks)


def test_the_default_keeps_gamma_learnable() -> None:
    blocks = _attention_blocks(_model(gamma=1.0))
    assert all("gamma_param" in dict(b.named_parameters()) for b in blocks)


def test_the_frozen_zero_control_changes_the_output() -> None:
    """Same seed and weights shapes: the control removes the modulation from the forward."""
    x = torch.rand(1, 1, 8, 8)
    with torch.no_grad():
        treated = _model(gamma=1.0)(x)
        control = _model(gamma=0.0, gamma_trainable=False)(x)
    assert treated.shape == control.shape == x.shape
    assert not torch.allclose(treated, control)
