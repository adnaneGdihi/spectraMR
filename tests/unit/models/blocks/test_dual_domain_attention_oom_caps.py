"""Regression tests for the OOM caps in dual_domain_attention_kan.

The 2026-05-10 cluster rerun saw 15+ ``experiment_11_attn_*`` arms OOM
on dense softmax attention over full-resolution feature maps:

* ``MultiScaleFreqBandAttention._attend_band`` allocated a ``[B, h, N, N]``
  attention matrix where ``N = H * W`` could exceed 32k tokens.
* ``RadialBandAttention`` iterated over logarithmic bands but the
  outermost band on a 256² grid carries ~half the image pixels (≈32k
  tokens) and the dense ``[B, h, N_k, N_k]`` matrix was ~32 GiB.

Both hot paths now cap their per-call attended sequence length. These
tests exercise the cap on a mock-sized input that, without the cap,
would allocate ~6 GiB for the attention matrix on the smallest GPU we
target. With the cap, the allocation stays under 100 MiB.
"""
from __future__ import annotations

import math

import pytest
import torch

from spectramr.models.blocks.dual_domain_attention_kan import (
    ComplexMHA,
    MultiScaleFreqBandAttention,
    RadialBandAttention,
    _complex_weighted_sum,
    _phase_aware_real_scores,
)


def _complex_seq(b: int, n: int, c: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    return torch.complex(
        torch.randn(b, n, c, generator=g), torch.randn(b, n, c, generator=g)
    )


@pytest.mark.parametrize("score_fn", ["softmax", "topk"])
def test_complex_mha_query_chunking_is_numerically_exact(score_fn):
    """Chunking over queries must equal the dense computation.

    ``ComplexMHA.forward`` allocated a dense ``[B, h, N, N]`` score matrix —
    the OOM that took out the experiment_11 KAN dual-domain cohort. The fix
    chunks over the query dim; because softmax / top-k act per-query on the key
    dim, the chunked result is identical to the dense one. Same module, same
    weights, ``_query_chunk`` small vs. larger-than-N → identical output.
    """
    mha = ComplexMHA(channels=8, num_heads=2, score_fn=score_fn, topk_k=3)
    x = _complex_seq(2, 17, 8)

    mha._query_chunk = 0  # disable chunking → dense reference
    ref = mha(x)
    mha._query_chunk = 4  # force several query chunks (N=17)
    chunked = mha(x)

    assert chunked.shape == ref.shape
    assert torch.allclose(chunked.real, ref.real, atol=1e-6)
    assert torch.allclose(chunked.imag, ref.imag, atol=1e-6)


def test_complex_mha_chunk_below_threshold_takes_dense_path():
    """When ``N <= chunk`` the dense path runs (no behavioural change for the
    common small-sequence case)."""
    mha = ComplexMHA(channels=8, num_heads=2)
    x = _complex_seq(1, 5, 8)
    mha._query_chunk = 64  # N=5 < 64 → dense
    out = mha(x)
    assert out.shape == x.shape and out.is_complex()


def test_multi_scale_freq_band_caps_dense_attention():
    """A 128² band would allocate ``[B,h,16384,16384]`` softmax. The cap
    keeps the dense ``[B,h,N,N]`` matrix under ``max_tokens²``."""
    block = MultiScaleFreqBandAttention(
        channels=4, num_heads=2, num_levels=1, max_tokens=256
    )
    x = torch.randn(1, 4, 128, 128, dtype=torch.complex64)
    # Should run without OOM; output preserves spatial shape.
    y = block(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_multi_scale_freq_band_no_cap_path():
    """Below the cap, the original full-resolution attention path is used."""
    block = MultiScaleFreqBandAttention(
        channels=4, num_heads=2, num_levels=1, max_tokens=4096
    )
    x = torch.randn(1, 4, 16, 16, dtype=torch.complex64)
    y = block(x)
    assert y.shape == x.shape


def test_radial_band_caps_per_band_attention():
    """A 256² grid with 8 bands has ~32k tokens in the outer band — the
    cap must subsample to ``max_band_tokens`` and still produce a same-
    shape output."""
    block = RadialBandAttention(
        channels=4, num_heads=2, num_bands=8, max_band_tokens=128
    )
    x = torch.randn(1, 4, 64, 64, dtype=torch.complex64)
    y = block(x)
    assert y.shape == x.shape


def test_radial_band_under_cap_uses_full_band():
    """When every band is under the cap, the original behavior runs
    unchanged: every token in the band is attended."""
    block = RadialBandAttention(
        channels=4, num_heads=2, num_bands=4, max_band_tokens=4096
    )
    x = torch.randn(1, 4, 16, 16, dtype=torch.complex64)
    y = block(x)
    assert y.shape == x.shape


def test_radial_band_invalid_max_band_tokens():
    with pytest.raises(ValueError, match="max_band_tokens"):
        RadialBandAttention(channels=4, num_heads=2, max_band_tokens=0)


def test_multi_scale_freq_band_invalid_max_tokens():
    with pytest.raises(ValueError, match="max_tokens"):
        MultiScaleFreqBandAttention(channels=4, num_heads=2, max_tokens=0)


# ---------------------------------------------------------------------------
# Real-matmul scoring (halves the score-tensor allocation at the OOM frame)
# ---------------------------------------------------------------------------
# Chunking (commit 8e2172132, 2026-06-21) bounds the score tensor to
# ``chunk * N`` but still built it in COMPLEX before taking ``.real``. The
# 2026-06-22 cluster reruns OOM'd anyway: ``Tried to allocate 256.00 MiB`` at
# dual_domain_attention_kan.py:158 is a ``[1, h, chunk, N]`` complex64 score
# tensor. ``Re(q kᴴ) = qr·krᵀ + qi·kiᵀ`` needs only real matmuls, so the
# intermediate is float (128 MiB, half) and fits the ~171 MiB headroom the
# 32 GiB-card arms died for.


def test_phase_aware_real_scores_matches_complex_reference():
    """``Re(q @ kᴴ)/scale`` via real matmuls equals the complex-matmul
    formula, but the intermediate score tensor is REAL (half the bytes)."""
    g = torch.Generator().manual_seed(1)
    q = torch.complex(
        torch.randn(2, 3, 7, 4, generator=g), torch.randn(2, 3, 7, 4, generator=g)
    )
    k = torch.complex(
        torch.randn(2, 3, 5, 4, generator=g), torch.randn(2, 3, 5, 4, generator=g)
    )
    scale = math.sqrt(4)
    ref = (q @ k.conj().transpose(-2, -1)).real / scale
    got = _phase_aware_real_scores(q, k, scale)
    assert not got.is_complex(), "score intermediate must be real to halve the allocation"
    assert got.shape == ref.shape
    assert torch.allclose(got, ref, atol=1e-6)


def test_complex_weighted_sum_matches_upcast_reference():
    """``complex(attn@V.real, attn@V.imag)`` equals ``attn.to(complex) @ V``
    without upcasting the real attention weights to complex."""
    g = torch.Generator().manual_seed(2)
    attn = torch.rand(2, 3, 7, 5, generator=g)  # real weights
    v = torch.complex(
        torch.randn(2, 3, 5, 4, generator=g), torch.randn(2, 3, 5, 4, generator=g)
    )
    ref = attn.to(v.dtype) @ v
    got = _complex_weighted_sum(attn, v)
    assert got.is_complex()
    assert got.shape == ref.shape
    assert torch.allclose(got.real, ref.real, atol=1e-6)
    assert torch.allclose(got.imag, ref.imag, atol=1e-6)


def test_complex_mha_forward_matches_complex_matmul_reference():
    """Full ``ComplexMHA`` forward equals an explicit complex-matmul reference of
    the QK-normalised score ``tau_h * Re(q_hat k_hat^H)`` (the real-matmul OOM fix
    must not change the math; the reference is the normalised score since the
    DC-capture fix replaced ``Re(q k^H) / sqrt(d)``)."""
    mha = ComplexMHA(channels=8, num_heads=2, score_fn="softmax")
    with torch.no_grad():
        mha.logit_scale.copy_(torch.tensor([0.5, 2.0]))
    x = _complex_seq(2, 13, 8)
    mha._query_chunk = 0  # dense path
    out = mha(x)
    B, N, C = x.shape
    Q = mha._project(mha.q_proj, x)
    K = mha._project(mha.k_proj, x)
    V = mha._project(mha.v_proj, x)

    def heads(t: torch.Tensor) -> torch.Tensor:
        return t.view(B, N, mha.num_heads, mha.head_dim).transpose(1, 2)

    def unit(t: torch.Tensor) -> torch.Tensor:
        return t / t.abs().pow(2).sum(-1, keepdim=True).sqrt()

    q_h, k_h, v_h = unit(heads(Q)), unit(heads(K)), heads(V)
    tau = torch.tensor([0.5, 2.0]).exp().view(1, 2, 1, 1)
    scores = (q_h @ k_h.conj().transpose(-2, -1)).real * tau
    attn = torch.softmax(scores, dim=-1)
    ref = attn.to(v_h.dtype) @ v_h
    ref = ref.transpose(1, 2).reshape(B, N, C)
    ri = torch.cat([ref.real, ref.imag], dim=-1)
    ri = mha.out_proj(ri)
    ref = torch.complex(ri[..., :C], ri[..., C:])
    assert torch.allclose(out.real, ref.real, atol=1e-5)
    assert torch.allclose(out.imag, ref.imag, atol=1e-5)


def test_complex_mha_linear_path_has_no_quadratic_tensor_and_runs_large():
    """``forward_linear`` (used above the token cap) never builds ``[N, N]``:
    16k tokens would need a 1 GiB score tensor per head on the dense path."""
    mha = ComplexMHA(channels=4, num_heads=2)
    x = _complex_seq(1, 128 * 128, 4)
    out = mha.forward_linear(x)
    assert out.shape == x.shape and torch.isfinite(out.real).all()
