"""Plain multi-head attention modules match the explicit softmax(QK^T)V they replaced."""

import pytest
import torch

from spectramr.models.diffusion.diffusion_parts.vit_parts import Attention as VitAttention
from spectramr.models.gans.swin_transformer_kan import SimpleWindowAttention
from spectramr.models.gans.vit_kan_gan import Attention as VitKanAttention
from spectramr.models.mae.mae_generator import Attention as MaeAttention
from spectramr.models.transformer.components import MultiHeadSelfAttention


def _heads(m, x, heads):
    B, N, C = x.shape
    q, k, v = m.qkv(x).reshape(B, N, 3, heads, C // heads).permute(2, 0, 3, 1, 4)
    return q, k, v


def _softmax_av(q, k, v, scale, mask=None):
    s = q @ k.transpose(-2, -1) * scale
    if mask is not None:
        s = s + mask
    return s.softmax(-1) @ v


@pytest.mark.parametrize("cls", [VitAttention, MaeAttention])
def test_qkv_attention_matches_explicit(cls) -> None:
    torch.manual_seed(0)
    m = cls(dim=16, num_heads=2, qkv_bias=True).eval()
    x = torch.randn(2, 9, 16)
    q, k, v = _heads(m, x, 2)
    ref = m.proj(_softmax_av(q, k, v, m.scale).transpose(1, 2).reshape(2, 9, 16))
    torch.testing.assert_close(m(x), ref, rtol=1e-4, atol=1e-5)


def test_multihead_self_attention_matches_explicit() -> None:
    torch.manual_seed(0)
    m = MultiHeadSelfAttention(embed_dim=16, num_heads=2).eval()
    x = torch.randn(2, 9, 16)
    q, k, v = _heads(m, x, 2)
    ref = m.proj(_softmax_av(q, k, v, m.scale).transpose(1, 2).reshape(2, 9, 16))
    torch.testing.assert_close(m(x), ref, rtol=1e-4, atol=1e-5)


def test_vit_kan_attention_matches_explicit() -> None:
    torch.manual_seed(0)
    m = VitKanAttention(dim=16, heads=2, dim_head=8).eval()
    x = torch.randn(2, 9, 16)
    q, k, v = (t.reshape(2, 9, 2, 8).transpose(1, 2) for t in m.to_qkv(x).chunk(3, dim=-1))
    out = _softmax_av(q, k, v, m.scale).transpose(1, 2).reshape(2, 9, 16)
    torch.testing.assert_close(m(x), m.to_out(out), rtol=1e-4, atol=1e-5)


def test_swin_kan_window_attention_masked_matches_explicit() -> None:
    torch.manual_seed(0)
    m = SimpleWindowAttention(dim=16, heads=2, head_dim=8, window_size=3).eval()
    x = torch.randn(2, 9, 16)
    mask = torch.rand(9, 9) > 0.3
    mask.fill_diagonal_(True)
    q, k, v = (t.reshape(2, 9, 2, 8).transpose(1, 2) for t in m.to_qkv(x).chunk(3, dim=-1))
    s = q @ k.transpose(-1, -2) * m.scale
    s = s.masked_fill(~mask, -torch.finfo(s.dtype).max)
    out = (s.softmax(-1) @ v).transpose(1, 2).reshape(2, 9, 16)
    torch.testing.assert_close(m(x, mask), m.to_out(out), rtol=1e-4, atol=1e-5)


def _explicit_window_attention(m, x, mask):
    b_, N, C = x.shape
    nh = m.num_heads
    q, k, v = m.qkv(x).reshape(b_, N, 3, nh, C // nh).permute(2, 0, 3, 1, 4)
    attn = (q * m.scale) @ k.transpose(-2, -1)
    bias = m.relative_position_bias_table[m.relative_position_index.reshape(-1)].view(N, N, -1)
    attn = attn + bias.permute(2, 0, 1).unsqueeze(0)
    if mask is not None:
        n_win = mask.shape[0]
        attn = attn.view(b_ // n_win, n_win, nh, N, N) + mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, nh, N, N)
    out = (attn.softmax(-1) @ v).transpose(1, 2).reshape(b_, N, C)
    return m.proj(out)


@pytest.mark.parametrize("which", ["hybrid", "diffusion_swin"])
def test_relative_bias_window_attention_matches_explicit(which: str) -> None:
    if which == "hybrid":
        from spectramr.models.generators.hybrid_transformer_diffusion_generator import (
            WindowAttention,
        )
    else:
        from spectramr.models.diffusion.diffusion_parts.swin_transformer_parts import (
            WindowAttention,
        )
    torch.manual_seed(0)
    ws = (4, 4) if which == "diffusion_swin" else 4
    m = WindowAttention(dim=16, window_size=ws, num_heads=2).eval()
    torch.nn.init.normal_(m.relative_position_bias_table, std=0.5)
    x = torch.randn(6, 16, 16)
    shift_mask = torch.where(torch.rand(2, 16, 16) > 0.7, -100.0, 0.0)
    for mask in (None, shift_mask):
        torch.testing.assert_close(
            m(x, mask), _explicit_window_attention(m, x, mask), rtol=1e-4, atol=1e-5
        )


def test_linformer_attention_matches_explicit() -> None:
    from spectramr.models.generators.attention_variants import _LinformerAttention

    torch.manual_seed(0)
    m = _LinformerAttention(dim=16, seq_len=9, k=4, heads=2).eval()
    x = torch.randn(2, 9, 16)
    q, k, v = m.to_qkv(x).reshape(2, 9, 3, 2, 8).permute(2, 0, 3, 1, 4)
    k = m.proj_k(k.transpose(-1, -2)).transpose(-1, -2)
    v = m.proj_v(v.transpose(-1, -2)).transpose(-1, -2)
    out = _softmax_av(q, k, v, 8**-0.5).transpose(1, 2).reshape(2, 9, 16)
    torch.testing.assert_close(m(x), m.out(out), rtol=1e-4, atol=1e-5)


def test_diffusion_sr_attention_block_matches_explicit() -> None:
    from spectramr.models.generators.diffusion_sr_generator import DiffusionAttentionBlock

    torch.manual_seed(0)
    m = DiffusionAttentionBlock(16, num_heads=2).eval()
    x = torch.randn(2, 16, 4, 3)
    q, k, v = (t.reshape(2, 2, 8, 12).transpose(2, 3) for t in m.qkv(m.norm(x)).chunk(3, dim=1))
    out = _softmax_av(q, k, v, 8**-0.5).transpose(2, 3).reshape(2, 16, 4, 3)
    torch.testing.assert_close(m(x), x + m.proj(out), rtol=1e-4, atol=1e-5)


def test_fourier_kan_window_attention_matches_explicit() -> None:
    from spectramr.models.spectral.fourier_kan_swin import WindowAttention

    torch.manual_seed(0)
    m = WindowAttention(dim=16, window_size=4, num_heads=2).eval()
    x = torch.randn(3, 16, 16)
    q, k, v = m.qkv(x).reshape(3, 16, 3, 2, 8).permute(2, 0, 3, 1, 4)
    out = _softmax_av(q, k, v, m.scale).transpose(1, 2).reshape(3, 16, 16)
    torch.testing.assert_close(m(x), m.proj(out), rtol=1e-4, atol=1e-5)
