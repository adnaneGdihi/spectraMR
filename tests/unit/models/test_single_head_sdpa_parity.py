"""SDPA single-head attention matches the explicit bmm/softmax it replaced."""

import torch

from spectramr.models.blocks.block_registry import DiffusionAttentionStrategy
from spectramr.models.generators.gan_variants import _SelfAttention2d
from spectramr.models.generators.latent_diffusion_generator import AttnBlock


def _bmm_reference(q, k, v, scale):
    attn = torch.softmax(torch.bmm(q.transpose(1, 2), k) * scale, dim=-1)
    return torch.bmm(v, attn.transpose(1, 2))


def test_sagan_self_attention_matches_unscaled_bmm():
    torch.manual_seed(0)
    m = _SelfAttention2d(32)
    m.gamma.data.fill_(1.0)
    x = torch.randn(2, 32, 6, 5)
    q, k, v = (f(x).flatten(2) for f in (m.query, m.key, m.value))
    expected = x + _bmm_reference(q, k, v, 1.0).reshape(x.shape)
    torch.testing.assert_close(m(x), expected, rtol=1e-4, atol=1e-5)


def test_latent_diffusion_attn_block_matches_scaled_bmm():
    torch.manual_seed(0)
    m = AttnBlock(32, num_groups=8).eval()
    x = torch.randn(2, 32, 6, 5)
    h = m.norm(x)
    q, k, v = (f(h).flatten(2) for f in (m.q, m.k, m.v))
    ref = m.proj_out(_bmm_reference(q, k, v, 32**-0.5).reshape(x.shape))
    torch.testing.assert_close(m(x), x + ref, rtol=1e-4, atol=1e-5)


def test_diffusion_attention_strategy_matches_scaled_bmm():
    torch.manual_seed(0)
    m = DiffusionAttentionStrategy(32).eval()
    x = torch.randn(2, 32, 6, 5)
    q, k, v = m.qkv(m.norm(x)).chunk(3, dim=1)
    q, k, v = (t.flatten(2) for t in (q, k, v))
    ref = m.proj(_bmm_reference(q, k, v, 32**-0.5).reshape(x.shape))
    torch.testing.assert_close(m(x), x + ref, rtol=1e-4, atol=1e-5)
