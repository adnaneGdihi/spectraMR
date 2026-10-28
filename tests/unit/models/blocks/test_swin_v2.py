"""Unit tests for the Swin-V2 attention and block."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from spectramr.models.blocks.swin_v2 import (
    SwinV2Block,
    SwinV2WindowAttention,
    log_spaced_relative_coords,
)


def test_log_spaced_coords_are_bounded_odd_and_zero_at_centre():
    table = log_spaced_relative_coords(4)
    assert table.shape == (7, 7, 2)
    assert torch.isclose(table.abs().max(), torch.tensor(math.log2(9.0) / 3.0))
    assert torch.allclose(table[3, 3], torch.zeros(2))
    assert torch.allclose(table.flip(0).flip(1), -table)


def test_position_bias_lies_in_zero_sixteen():
    attn = SwinV2WindowAttention(dim=8, window_size=4, num_heads=2)
    bias = attn.position_bias()
    assert bias.shape == (1, 2, 16, 16)
    assert bias.min() > 0.0 and bias.max() < 16.0


def test_attention_uses_cosine_logits_at_the_learned_temperature():
    torch.manual_seed(0)
    attn = SwinV2WindowAttention(dim=8, window_size=2, num_heads=2).eval()
    with torch.no_grad():
        attn.logit_scale.fill_(math.log(30.0))
    x = torch.randn(3, 4, 8)
    q, k, v = attn.qkv(x).reshape(3, 4, 3, 2, 4).permute(2, 0, 3, 1, 4)
    logits = 30.0 * F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
    ref = torch.softmax(logits + attn.position_bias(), dim=-1) @ v
    ref = attn.proj(ref.transpose(1, 2).reshape(3, 4, 8))
    assert torch.allclose(attn(x), ref, atol=1e-5)


def test_temperature_is_clamped_at_log_100():
    attn = SwinV2WindowAttention(dim=4, window_size=2, num_heads=1).eval()
    x = torch.randn(1, 4, 4)
    with torch.no_grad():
        attn.logit_scale.fill_(math.log(100.0))
        at_cap = attn(x)
        attn.logit_scale.fill_(math.log(1e4))
        assert torch.allclose(attn(x), at_cap)


@pytest.mark.parametrize("shift", [0, 2])
@pytest.mark.parametrize("hw", [(8, 8), (6, 10)])
def test_block_keeps_token_shape_and_backpropagates(shift, hw):
    block = SwinV2Block(dim=8, num_heads=2, window_size=4, shift_size=shift)
    x = torch.randn(2, hw[0] * hw[1], 8, requires_grad=True)
    out = block(x, hw)
    assert out.shape == x.shape
    out.square().mean().backward()
    assert torch.isfinite(x.grad).all()


def test_block_normalises_after_the_branch_not_before(monkeypatch):
    block = SwinV2Block(dim=4, num_heads=1, window_size=2).eval()
    x = torch.randn(1, 4, 4)
    branch = torch.randn(1, 4, 4) * 50.0
    monkeypatch.setattr(block, "_attend", lambda *_: branch)
    h = x + block.norm1(branch)
    assert torch.allclose(block(x, (2, 2)), h + block.norm2(block.mlp(h)), atol=1e-6)


def test_block_rejects_bad_shift_and_bad_grid():
    with pytest.raises(ValueError, match="shift_size"):
        SwinV2Block(dim=4, num_heads=1, window_size=2, shift_size=2)
    with pytest.raises(ValueError, match="tokens"):
        SwinV2Block(dim=4, num_heads=1, window_size=2)(torch.randn(1, 5, 4), (2, 2))
