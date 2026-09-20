"""KAN-Gated Dual-Domain Attention.

Composite attention block with three parallel branches mixed by
input/timestep-conditioned KAN gates:

* Image-domain complex multi-head self-attention
* K-space radial-band self-attention (logarithmically-spaced annular bands)
* Cross-domain attention (queries from image, keys/values from frequency)

The output is fused via per-branch sigmoid gates produced by small KAN
networks conditioned on the (phase-aware) global pooled features and an
optional time embedding.

All FFT operations route through `spectramr.infrastructure.physics.fft_ops` to
respect the centered, ortho-normalized convention enforced repo-wide.

Layout convention
-----------------
This block is wired into the kspace_cold_diffusion U-Net which uses
**interleaved** real/imag along the channel axis:
``x[:, 0::2]`` is real, ``x[:, 1::2]`` is imaginary, with ``2*C`` real
channels representing ``C`` complex channels. The block converts to/from
true complex tensors internally and returns interleaved.

Feature-domain contract (2026-07-03)
------------------------------------
The U-Net's feature maps are k-space under ``force_pure_kspace: true`` and
image otherwise — the blocks here no longer hard-assume image input. The
required ``feature_domain`` constructor kwarg ("kspace" | "image", derived
by ``FourierBridgeNetwork``, never a YAML knob of its own) dictates how the
image/spectrum views are derived; outputs are composed back in the input's
domain with the residual anchored on the untransformed native input. See
``spectramr.models.blocks.attention_domains`` (the SSOT) and
``docs/attention_wiring_audit.rst``.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c
from spectramr.models.blocks.attention_domains import (
    complex_to_interleaved as _complex_to_interleaved,
)
from spectramr.models.blocks.attention_domains import (
    interleaved_to_complex as _interleaved_to_complex,
)
from spectramr.models.blocks.attention_domains import validate_feature_domain
from spectramr.models.blocks.kan_layer import KANLayer

# Query-block size for :class:`ComplexMHA`. Above this sequence length the
# attention is computed in query chunks so the dense ``[B, h, N, N]`` score
# matrix is never materialised — softmax / top-k are per-query (last-dim), so
# chunking is numerically EXACT and only caps PEAK memory at O(chunk*N) instead
# of O(N*N). This is the OOM that took out the experiment_11 KAN dual-domain
# cohort (``ComplexMHA.forward`` allocating into a full 44 GiB GPU, 2026-06
# cluster rerun). Mirrors the ``max_band_tokens`` cap already on
# :class:`RadialBandAttention`. Per-instance override via ``self._query_chunk``.
_COMPLEX_MHA_QUERY_CHUNK = 2048

# NOTE: the interleaved<->complex layout helpers were promoted to
# ``spectramr.models.blocks.attention_domains`` (the feature-domain SSOT) on
# 2026-07-03 and are re-imported above under their historical private names.

# ---------------------------------------------------------------------------
# Phase-aware attention primitives (memory-bounded)
# ---------------------------------------------------------------------------
# The phase-aware score is ``Re(Q Kᴴ) / sqrt(d)``. Computing it as a complex
# matmul materialises a COMPLEX ``[..., N_q, N_k]`` tensor before ``.real`` is
# taken — twice the bytes of the real result. With query chunking already in
# place (commit 8e2172132) the surviving OOM on the experiment_11 KAN
# dual-domain cohort (2026-06-22 reruns, ``Tried to allocate 256.00 MiB`` at
# the score line) is exactly this complex intermediate. ``Re(q · conj(k))``
# summed over the feature dim equals ``qr·krᵀ + qi·kiᵀ``, so the score can be
# built from two REAL matmuls and the peak at the failing frame halves
# (256 MiB -> 128 MiB), which fits the ~171 MiB headroom the 32 GiB-card arms
# died for. Both helpers are numerically identical to the complex formulation.


def _phase_aware_real_scores(q: torch.Tensor, k: torch.Tensor, scale: float) -> torch.Tensor:
    """``Re(q @ kᴴ) / scale`` via real matmuls, never building a complex score.

    Args:
        q: ``[..., N_q, d]`` complex queries.
        k: ``[..., N_k, d]`` complex keys.
        scale: ``sqrt(head_dim)`` normaliser.

    Returns:
        ``[..., N_q, N_k]`` **real** score tensor, equal to
        ``(q @ k.conj().transpose(-2, -1)).real / scale``.
    """
    # Re(q · conj(k)) = q.real·k.real + q.imag·k.imag. ``.real``/``.imag`` and
    # ``.transpose`` are views, so no extra allocation beyond the two real
    # matmul outputs (each half the size of the complex product).
    kr_t = k.real.transpose(-2, -1)
    ki_t = k.imag.transpose(-2, -1)
    scores = torch.matmul(q.real, kr_t)
    scores = scores + torch.matmul(q.imag, ki_t)
    return scores / scale


def _complex_weighted_sum(attn: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """``attn @ v`` for real ``attn`` and complex ``v`` without upcasting attn.

    Equivalent to ``attn.to(v.dtype) @ v`` but keeps the (large)
    ``[..., N_q, N_k]`` weight tensor real instead of doubling it to complex
    before the matmul.
    """
    return torch.complex(torch.matmul(attn, v.real), torch.matmul(attn, v.imag))


# ---------------------------------------------------------------------------
# Complex multi-head self-attention
# ---------------------------------------------------------------------------


class ComplexMHA(nn.Module):
    """Complex multi-head self-attention with phase-aware scoring.

    Implements the score ``Re(Q K^H) / sqrt(d)`` where ``K^H`` is the
    conjugate transpose; the real-part inner product preserves phase
    information (it's ``|q||k|cos(arg q - arg k)``) while keeping the
    softmax real-valued and numerically stable.

    Args:
        channels: Complex channel count.
        num_heads: Number of attention heads. Must divide ``channels``.
        score_fn: One of:
            ``softmax`` — default dense softmax over all keys per query.
            ``topk`` — sparse: keep the ``topk_k`` largest scores per query,
                mask rest to -inf, softmax over surviving keys. Yields a
                sparse, interpretable attention pattern that aligns with
                compressed-sensing-style sparsity priors (plan §2.2).
        topk_k: Number of keys to keep per query when ``score_fn='topk'``.
            If 0, defaults to ``max(1, ceil(sqrt(N)))`` at forward time
            (sqrt-N is a common rule-of-thumb for sparse attention).
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        score_fn: str = "softmax",
        topk_k: int = 0,
    ) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by num_heads ({num_heads})")
        if score_fn not in ("softmax", "topk"):
            raise ValueError(f"score_fn must be 'softmax' or 'topk', got {score_fn!r}")
        if topk_k < 0:
            raise ValueError(f"topk_k must be >= 0, got {topk_k}")
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.channels = channels
        self.score_fn = score_fn
        self.topk_k = int(topk_k)
        # Real-valued projections applied separately to (real, imag) stacks.
        self.q_proj = nn.Linear(2 * channels, 2 * channels, bias=False)
        self.k_proj = nn.Linear(2 * channels, 2 * channels, bias=False)
        self.v_proj = nn.Linear(2 * channels, 2 * channels, bias=False)
        self.out_proj = nn.Linear(2 * channels, 2 * channels, bias=False)

    def _project(self, proj: nn.Linear, seq_complex: torch.Tensor) -> torch.Tensor:
        # seq_complex: [B, N, C] complex
        ri = torch.cat([seq_complex.real, seq_complex.imag], dim=-1)  # [B,N,2C]
        out = proj(ri)
        c = self.channels
        return torch.complex(out[..., :c], out[..., c:])

    def forward(self, seq_complex: torch.Tensor) -> torch.Tensor:
        """Args:
            seq_complex: ``[B, N, C]`` complex sequence.
        Returns:
            ``[B, N, C]`` complex sequence.
        """
        B, N, C = seq_complex.shape
        Q = self._project(self.q_proj, seq_complex)
        K = self._project(self.k_proj, seq_complex)
        V = self._project(self.v_proj, seq_complex)

        # Reshape to heads: [B, h, N, d]
        def _heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        Qh, Kh, Vh = _heads(Q), _heads(K), _heads(V)
        # Phase-aware score Re(Q Kᴴ)/sqrt(d) and the value aggregation are built
        # from REAL matmuls so the large [B, h, N_q, N_k] score/weight tensors
        # never go complex — this is the surviving 2026-06-22 KAN dual-domain
        # OOM at the score line (chunking alone left the intermediate complex).
        # See ``_phase_aware_real_scores`` / ``_complex_weighted_sum``.
        scale = math.sqrt(self.head_dim)
        chunk = int(getattr(self, "_query_chunk", _COMPLEX_MHA_QUERY_CHUNK))
        if chunk <= 0 or chunk >= N:
            scores = _phase_aware_real_scores(Qh, Kh, scale)
            attn = self._apply_score_fn(scores)
            out = _complex_weighted_sum(attn, Vh)  # [B, h, N, d] complex
        else:
            # Chunk over queries so the [B, h, N, N] score matrix is never fully
            # materialised. softmax / top-k act on the last (key) dim per query,
            # so the per-chunk result is identical to the dense computation.
            parts: list[torch.Tensor] = []
            for i in range(0, N, chunk):
                q_i = Qh[:, :, i : i + chunk]  # [B, h, c, d]
                scores_i = _phase_aware_real_scores(q_i, Kh, scale)  # [B,h,c,N] real
                attn_i = self._apply_score_fn(scores_i)
                parts.append(_complex_weighted_sum(attn_i, Vh))  # [B, h, c, d]
            out = torch.cat(parts, dim=2)  # [B, h, N, d]
        out = out.transpose(1, 2).reshape(B, N, C)
        ri = torch.cat([out.real, out.imag], dim=-1)
        ri = self.out_proj(ri)
        return torch.complex(ri[..., :C], ri[..., C:])

    def _apply_score_fn(self, scores: torch.Tensor) -> torch.Tensor:
        """Apply softmax or top-k masked softmax to attention scores.

        For top-k, we mask all but the largest k scores per query to -inf
        before softmax. The result is exactly k non-zero attention weights
        per query (sparse, autograd-compatible, no custom backward needed).
        """
        if self.score_fn == "softmax":
            return torch.softmax(scores, dim=-1)
        # topk path
        # scores shape: [B, h, N_q, N_k]
        N_k = scores.shape[-1]
        k = self.topk_k
        if k <= 0:
            # sqrt-N rule-of-thumb when not explicitly configured
            k = max(1, int(math.ceil(math.sqrt(N_k))))
        k = min(k, N_k)
        if k == N_k:
            return torch.softmax(scores, dim=-1)
        topk_vals, _ = scores.topk(k, dim=-1)
        # Threshold = the smallest of the top-k values per query
        threshold = topk_vals[..., -1:].detach()
        masked = scores.masked_fill(scores < threshold, float("-inf"))
        return torch.softmax(masked, dim=-1)


# ---------------------------------------------------------------------------
# K-space radial-band attention
# ---------------------------------------------------------------------------


class RadialBandAttention(nn.Module):
    """Self-attention restricted to concentric annular k-space bands.

    Attention is computed independently within each band but shares the
    Q/K/V projections across bands (so the model still learns a single
    representation, only the *interaction* is band-local). Bands are
    logarithmically spaced so the DC region is more finely resolved.

    Outer logarithmic bands on a 256² grid hold ~half the image's
    pixels (~32 k tokens). The dense ``[B, h, N, N]`` attention matrix
    for that single band is ~32 GiB — every ``experiment_11_attn_*``
    arm OOM'd here on the 2026-05-10 cluster rerun. ``max_band_tokens``
    caps each band's attended sequence at a fixed length by uniform
    stride subsampling. Unchosen positions inherit the mean of the
    attended outputs so the IFFT downstream doesn't see hard-zero
    frequencies in the un-attended k-space bins.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        num_bands: int = 8,
        beta: float = 2.0,
        score_fn: str = "softmax",
        topk_k: int = 0,
        max_band_tokens: int = 4096,
    ) -> None:
        super().__init__()
        if max_band_tokens < 1:
            raise ValueError(f"max_band_tokens must be >= 1, got {max_band_tokens}")
        self.attn = ComplexMHA(channels, num_heads, score_fn=score_fn, topk_k=topk_k)
        self.num_bands = int(num_bands)
        self.beta = float(beta)
        self.max_band_tokens = int(max_band_tokens)
        # Cached band partition keyed by (H, W, device); re-built on a shape
        # OR device change. ``_band_index`` is the [H,W] band-id map;
        # ``_band_sel`` is the per-band list of the FINAL selection indices
        # (already stride-subsampled, empty bands dropped) so the forward loop
        # only does cached ``index_select``/``index_copy_`` and never calls
        # ``.nonzero()``/``.numel()`` on live data (both force a CUDA sync —
        # ~64/forward here, every step, on all kan_dual_domain arms).
        self._cache_key: tuple[int, int, torch.device] | None = None
        self._band_sel: list[torch.Tensor] = []
        self.register_buffer("_band_index", torch.empty(0, dtype=torch.long), persistent=False)

    @torch.no_grad()
    def _build_band_index(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        ky = torch.linspace(-(H // 2), H - 1 - H // 2, H, device=device)
        kx = torch.linspace(-(W // 2), W - 1 - W // 2, W, device=device)
        ky_g, kx_g = torch.meshgrid(ky, kx, indexing="ij")
        r = torch.sqrt(ky_g**2 + kx_g**2)
        r_max = float(r.max().item())
        if r_max <= 0:
            return torch.zeros(H, W, device=device, dtype=torch.long)
        denom = self.beta if self.beta > 0 else 1.0
        edges = torch.tensor(
            [
                r_max * (math.exp(k / self.num_bands * math.log(1.0 + denom)) - 1.0) / denom
                for k in range(self.num_bands + 1)
            ],
            device=device,
        )
        # bucketize into [0, num_bands-1]
        band = torch.bucketize(r, edges[1:-1])
        return band.long()

    @torch.no_grad()
    def _build_band_sel(self, band_index: torch.Tensor) -> list[torch.Tensor]:
        """Precompute the per-band token-selection indices ONCE per shape.

        Mirrors the forward's selection exactly: the flat positions of each
        band, uniform-stride-subsampled to at most ``max_band_tokens`` (stride
        preserves frequency ordering so spatial coherence is kept). Empty bands
        are dropped. Doing the data-dependent ``.nonzero()``/subsample here (a
        single cached build per shape) keeps the CUDA-syncing ops off the hot
        forward path, which then only does cached ``index_select``/
        ``index_copy_``. Bands are disjoint and scatter to their own positions,
        so dropping empties / preserving k-order leaves the result unchanged.
        """
        band_flat = band_index.flatten()
        sel: list[torch.Tensor] = []
        for k in range(self.num_bands):
            idx = (band_flat == k).nonzero(as_tuple=False).flatten()
            n_k = idx.numel()
            if n_k == 0:
                continue
            if n_k > self.max_band_tokens:
                stride = max(1, n_k // self.max_band_tokens)
                sub_pos = torch.arange(0, n_k, stride, device=idx.device)[: self.max_band_tokens]
                idx = idx[sub_pos]
            sel.append(idx)
        return sel

    def forward(self, kspace_complex: torch.Tensor) -> torch.Tensor:
        """Args:
            kspace_complex: ``[B, C, H, W]`` complex k-space features.
        Returns:
            ``[B, C, H, W]`` complex k-space features.
        """
        B, C, H, W = kspace_complex.shape
        cache_key = (H, W, kspace_complex.device)
        if self._cache_key != cache_key:
            self._band_index = self._build_band_index(H, W, kspace_complex.device)
            self._band_sel = self._build_band_sel(self._band_index)
            self._cache_key = cache_key
        flat = kspace_complex.permute(0, 2, 3, 1).reshape(B, H * W, C)
        out_flat = flat.clone()  # default: identity (preserves un-attended bins)
        # Cached per-band selection indices (post stride-subsample, empty bands
        # dropped) — the forward does NO ``.nonzero()``/``.numel()`` on live
        # data, so no per-band CUDA sync. Un-attended positions keep their
        # original (identity) values from ``out_flat = flat.clone()``.
        for sel_idx in self._band_sel:
            band_seq = flat.index_select(1, sel_idx)  # [B, n_sel, C]
            band_out = self.attn(band_seq)
            out_flat.index_copy_(1, sel_idx, band_out)
        return out_flat.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()


# ---------------------------------------------------------------------------
# Cross-domain attention
# ---------------------------------------------------------------------------


class CrossDomainAttention(nn.Module):
    """Image-side queries against k-space keys/values.

    Output values come from the frequency domain, so we inverse-FFT
    before returning so the result is comparable to the image-domain branch.
    Q/K/V projections are independent (tying them would defeat the purpose).
    """

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.channels = channels
        self.q_proj = nn.Linear(2 * channels, 2 * channels, bias=False)
        self.k_proj = nn.Linear(2 * channels, 2 * channels, bias=False)
        self.v_proj = nn.Linear(2 * channels, 2 * channels, bias=False)
        self.out_proj = nn.Linear(2 * channels, 2 * channels, bias=False)

    def _project(self, proj: nn.Linear, seq_complex: torch.Tensor) -> torch.Tensor:
        ri = torch.cat([seq_complex.real, seq_complex.imag], dim=-1)
        out = proj(ri)
        c = self.channels
        return torch.complex(out[..., :c], out[..., c:])

    def forward(
        self,
        image_complex: torch.Tensor,
        kspace_complex: torch.Tensor,
    ) -> torch.Tensor:
        """Args:
            image_complex: ``[B, C, H, W]`` complex image-domain features.
            kspace_complex: ``[B, C, H, W]`` complex k-space features.
        Returns:
            ``[B, C, H, W]`` complex image-domain features.
        """
        B, C, H, W = image_complex.shape
        h_seq = image_complex.permute(0, 2, 3, 1).reshape(B, H * W, C)
        f_seq = kspace_complex.permute(0, 2, 3, 1).reshape(B, H * W, C)

        Q = self._project(self.q_proj, h_seq)
        K = self._project(self.k_proj, f_seq)
        V = self._project(self.v_proj, f_seq)

        def _heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)

        Qh, Kh, Vh = _heads(Q), _heads(K), _heads(V)
        # Real-matmul phase-aware scoring (see ComplexMHA): keeps the
        # [B, h, N, N] cross-attention score/weight tensors real instead of
        # complex, halving the peak at this full-resolution matmul.
        scores = _phase_aware_real_scores(Qh, Kh, math.sqrt(self.head_dim))
        attn = torch.softmax(scores, dim=-1)
        out = _complex_weighted_sum(attn, Vh)
        out = out.transpose(1, 2).reshape(B, H * W, C)
        ri = torch.cat([out.real, out.imag], dim=-1)
        ri = self.out_proj(ri)
        out_freq = torch.complex(ri[..., :C], ri[..., C:])
        out_freq = out_freq.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        # Values came from the frequency domain; bring back to image domain.
        return ifft2c(out_freq)


# ---------------------------------------------------------------------------
# Multi-scale frequency-band attention (Haar wavelet, plan §2.3)
# ---------------------------------------------------------------------------


def _haar_decompose_2d(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One level of 2D Haar wavelet decomposition.

    Args:
        x: ``[B, C, H, W]`` real or complex tensor. H and W must be even.

    Returns:
        Tuple ``(LL, LH, HL, HH)`` each of shape ``[B, C, H/2, W/2]``.
        LL = low-low (smooth/approximation), LH/HL/HH = high-frequency
        details in horizontal/vertical/diagonal directions respectively.
    """
    if x.shape[-2] % 2 or x.shape[-1] % 2:
        raise ValueError(f"Haar decomposition requires even spatial dims, got {x.shape}")
    # Per-axis pairwise sum/diff with sqrt(2) normalization (orthogonal Haar).
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    a = (x[..., 0::2, :] + x[..., 1::2, :]) * inv_sqrt2  # row-low
    d = (x[..., 0::2, :] - x[..., 1::2, :]) * inv_sqrt2  # row-high
    LL = (a[..., :, 0::2] + a[..., :, 1::2]) * inv_sqrt2
    LH = (a[..., :, 0::2] - a[..., :, 1::2]) * inv_sqrt2
    HL = (d[..., :, 0::2] + d[..., :, 1::2]) * inv_sqrt2
    HH = (d[..., :, 0::2] - d[..., :, 1::2]) * inv_sqrt2
    return LL, LH, HL, HH


def _haar_reconstruct_2d(LL, LH, HL, HH) -> torch.Tensor:
    """Inverse 2D Haar wavelet transform — left-inverse of ``_haar_decompose_2d``."""
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    # Reverse the column step
    a = torch.zeros_like(LL).repeat_interleave(2, dim=-1)
    d = torch.zeros_like(LL).repeat_interleave(2, dim=-1)
    a[..., :, 0::2] = (LL + LH) * inv_sqrt2
    a[..., :, 1::2] = (LL - LH) * inv_sqrt2
    d[..., :, 0::2] = (HL + HH) * inv_sqrt2
    d[..., :, 1::2] = (HL - HH) * inv_sqrt2
    # Reverse the row step
    out = torch.zeros(
        *LL.shape[:-2], LL.shape[-2] * 2, LL.shape[-1] * 2, dtype=LL.dtype, device=LL.device
    )
    out[..., 0::2, :] = (a + d) * inv_sqrt2
    out[..., 1::2, :] = (a - d) * inv_sqrt2
    return out


class WaveletFreqAttentionBlock(nn.Module):
    """Drop-in attention block for KSpaceDownsample/UpsampleBlock that uses
    Haar-wavelet multi-scale frequency-band attention internally.

    Same interleaved real I/O convention as ``KANGatedDualDomainAttention``
    so the dispatcher can swap it in without changing the call sites.

    The attended output is fused with the residual input by a learned per-batch
    gate whose parameterisation (``gate_type``) is either a KAN (two
    ``KANLayer``s) or a matched MLP. This is what makes the KAN/MLP ablation
    axis REAL for the wavelet cell: before, this block ignored ``gate_type``
    entirely, so ``attn_kan_wavelet`` and ``attn_mlp_wavelet`` were
    byte-identical (a self-declared ``needs_implementation`` facade — pitfall
    #16). The gate mirrors ``KANGatedDualDomainAttention._make_gate``.
    """

    def __init__(
        self,
        in_channels: int,
        num_heads: int = 4,
        num_levels: int = 1,
        score_fn: str = "softmax",
        topk_k: int = 0,
        gate_type: str = "kan",
        kan_grid_size: int = 5,
        kan_spline_order: int = 3,
        kan_hidden: int = 16,
        *,
        feature_domain: str,
    ) -> None:
        super().__init__()
        if in_channels % 2 != 0:
            raise ValueError(
                f"WaveletFreqAttentionBlock expects even (interleaved) channels, got {in_channels}"
            )
        # Haar DWT is a spatial (image-domain) prior — under a k-space input
        # the block derives the image view via ifft2c, attends there, and
        # conjugates the gated update back so output domain == input domain.
        self.feature_domain = validate_feature_domain(feature_domain)
        self.complex_channels = in_channels // 2
        nh = num_heads
        while nh > 1 and self.complex_channels % nh != 0:
            nh //= 2
        self.attn = MultiScaleFreqBandAttention(
            channels=self.complex_channels,
            num_heads=max(1, nh),
            num_levels=num_levels,
            score_fn=score_fn,
            topk_k=topk_k,
        )

        # KAN vs MLP fusion gate (the ablation axis). Reads a phase-aware GAP
        # ``[|h|; cos arg h; sin arg h]`` (3 * complex_channels) and emits a
        # per-batch scalar in (0, 1) blending attended output vs residual.
        self.gate_type = str(gate_type).lower()
        if self.gate_type not in ("kan", "mlp"):
            raise ValueError(f"gate_type must be 'kan' or 'mlp', got {gate_type!r}")
        gate_in = 3 * self.complex_channels
        if self.gate_type == "kan":
            self.gate: nn.Module = nn.Sequential(
                KANLayer(gate_in, kan_hidden, kan_grid_size, kan_spline_order),
                KANLayer(kan_hidden, 1, kan_grid_size, kan_spline_order),
            )
        else:
            # MLP ablation: matched depth & hidden width.
            self.gate = nn.Sequential(
                nn.Linear(gate_in, kan_hidden),
                nn.SiLU(),
                nn.Linear(kan_hidden, 1),
            )

    @staticmethod
    def _phase_aware_gap(h: torch.Tensor) -> torch.Tensor:
        """[mag, cos(arg), sin(arg)] global-average-pool of complex ``h`` -> [B, 3C]."""
        mag = h.abs().mean(dim=(-2, -1))
        denom = h.abs() + 1e-8
        cos_p = (h.real / denom).mean(dim=(-2, -1))
        sin_p = (h.imag / denom).mean(dim=(-2, -1))
        return torch.cat([mag, cos_p, sin_p], dim=-1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor | None = None) -> torch.Tensor:
        del t_emb  # not used; signature matches KANGatedDualDomainAttention
        h_native = _interleaved_to_complex(x)
        # Image view: the Haar DWT prior only makes sense on spatial features.
        h_img = ifft2c(h_native) if self.feature_domain == "kspace" else h_native
        out_complex = self.attn(h_img)
        # Gated residual fusion: g in (0,1) per batch. The KAN vs MLP gate is
        # the distinguishing parameterisation between attn_kan_wavelet and
        # attn_mlp_wavelet. Descriptor is always taken from the image view so
        # the gate semantics are mode-invariant.
        b = h_native.shape[0]
        g = torch.sigmoid(self.gate(self._phase_aware_gap(h_img))).view(b, 1, 1, 1)
        if self.feature_domain == "kspace":
            # Native-frame anchoring: the residual base is the UNTRANSFORMED
            # k-space input; only the gated update is conjugated back. Equals
            # fft2c(g*out + (1-g)*h_img) up to roundoff, but g=0 reduces to
            # the input bit-exactly (pitfall #20 guarantee preserved).
            fused = h_native + g * fft2c(out_complex - h_img)
        else:
            fused = g * out_complex + (1.0 - g) * h_native
        return _complex_to_interleaved(fused)


class MultiScaleFreqBandAttention(nn.Module):
    """Per-band attention on a Haar-wavelet decomposition of the input.

    Plan §2.3 — decomposes the feature map into multiple frequency bands
    via DWT (Haar basis), applies attention within each band independently,
    then reconstructs via inverse DWT. Each band carries a different scale
    of structure; attending separately encodes the inductive bias that
    "low-, mid-, and high-frequency bands carry different information that
    should be attended differently."

    Why Haar specifically: it's a one-line implementation (sum/diff with
    sqrt(2) normalization), no extra dependencies (PyWavelets, ptwt), and
    is exactly orthogonal so reconstruction is lossless on even grids.
    More sophisticated wavelet bases (db4, sym4) would require introducing
    a dependency on PyWavelets and add little for MRI features that are
    typically smoothly varying with sharp edges.

    Args:
        channels: Complex channel count.
        num_heads: Heads per band's ComplexMHA.
        num_levels: Wavelet decomposition depth (default 1 = 4 bands;
            level 2 = 7 bands by recursing into LL only).
        score_fn / topk_k: Forwarded to each band's ComplexMHA — sparsity
            in high-frequency bands aligns with natural-image priors.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        num_levels: int = 1,
        score_fn: str = "softmax",
        topk_k: int = 0,
        max_tokens: int = 4096,
    ) -> None:
        super().__init__()
        if num_levels < 1:
            raise ValueError(f"num_levels must be >= 1, got {num_levels}")
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
        self.channels = channels
        self.num_levels = int(num_levels)
        # OOM cap. The dense attention matrix is O(N²) in token count, so
        # attending directly over a 256×256 band (N=65536) requires ~16
        # GiB just for the [B,h,N,N] softmax tensor. The 2026-05-10 cluster
        # rerun OOM'd 15 ``experiment_11_attn_*`` arms on exactly this
        # path. We cap N at ``max_tokens`` by adaptive-pooling the band
        # before attention and bilinear-upsampling after — the attention
        # pattern is computed at coarser resolution but the residual
        # connection in the parent block preserves fine detail.
        self.max_tokens = int(max_tokens)
        # One ComplexMHA per band. Bands at the same level share an
        # attention block (low param count); levels each get their own
        # so the inductive bias of each scale stays distinct.
        # Bands per level: 4 at level 1, 3 new bands per additional level
        # (the LL child gets recursed into).
        self.attn_bands: nn.ModuleList = nn.ModuleList()
        for _ in range(num_levels):
            band_attn = ComplexMHA(channels, num_heads, score_fn=score_fn, topk_k=topk_k)
            self.attn_bands.append(band_attn)

    def _attend_band(self, band: torch.Tensor, attn: ComplexMHA) -> torch.Tensor:
        """Apply one ComplexMHA over the spatial dims of a single band.

        Caps the token count at ``self.max_tokens`` to avoid the
        quadratic dense-softmax OOM seen on 256×256 bands. Pool→attend→
        upsample is mathematically lossy but the parent block's
        residual carries the fine spatial detail.
        """
        B, C, H, W = band.shape
        if self.max_tokens < H * W:
            # Pick (h, w) that satisfies h*w <= max_tokens and preserves aspect
            ratio = (self.max_tokens / float(H * W)) ** 0.5
            new_h = max(1, int(H * ratio))
            new_w = max(1, int(W * ratio))
            # Complex adaptive pool: pool real and imag separately
            band_pooled = torch.complex(
                nn.functional.adaptive_avg_pool2d(band.real, (new_h, new_w)),
                nn.functional.adaptive_avg_pool2d(band.imag, (new_h, new_w)),
            )
            flat = band_pooled.permute(0, 2, 3, 1).reshape(B, new_h * new_w, C)
            out_flat = attn(flat)
            out_pooled = out_flat.reshape(B, new_h, new_w, C).permute(0, 3, 1, 2).contiguous()
            # Upsample back. Bilinear is fine for complex via
            # interpolate-on-stack (real and imag separately).
            up = torch.complex(
                nn.functional.interpolate(
                    out_pooled.real, size=(H, W), mode="bilinear", align_corners=False
                ),
                nn.functional.interpolate(
                    out_pooled.imag, size=(H, W), mode="bilinear", align_corners=False
                ),
            )
            return up.contiguous()
        flat = band.permute(0, 2, 3, 1).reshape(B, H * W, C)
        out_flat = attn(flat)
        return out_flat.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: ``[B, C, H, W]`` complex tensor. ``H`` and ``W`` must be
            divisible by ``2**num_levels`` (else falls back to 1 level
            of decomposition; raises if even one level isn't possible).

        Returns:
            ``[B, C, H, W]`` complex tensor of the same shape.
        """
        # Adapt num_levels to what spatial dims allow at runtime.
        H, W = x.shape[-2:]
        max_levels = 0
        h, w = H, W
        while h % 2 == 0 and w % 2 == 0 and max_levels < self.num_levels:
            h //= 2
            w //= 2
            max_levels += 1
        if max_levels == 0:
            # Spatial dims aren't even — no decomposition possible. Apply
            # attention at the input scale and return.
            return self._attend_band(x, self.attn_bands[0])

        # Recursive decomposition: each level decomposes the current LL
        # into 4 sub-bands and attends each. We accumulate (LH, HL, HH)
        # at each level (post-attention) for reconstruction.
        details: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        current = x
        for level in range(max_levels):
            LL, LH, HL, HH = _haar_decompose_2d(current)
            attn = self.attn_bands[level]
            # Apply attention to the three detail bands; the LL is
            # carried into the next level (or attended at the base if
            # this is the final level).
            LH_out = self._attend_band(LH, attn)
            HL_out = self._attend_band(HL, attn)
            HH_out = self._attend_band(HH, attn)
            details.append((LH_out, HL_out, HH_out))
            current = LL  # recurse into LL

        # Final LL gets attended at the deepest level.
        current = self._attend_band(current, self.attn_bands[max_levels - 1])

        # Reconstruct from the deepest level outward.
        for LH_out, HL_out, HH_out in reversed(details):
            current = _haar_reconstruct_2d(current, LH_out, HL_out, HH_out)
        return current


# ---------------------------------------------------------------------------
# KAN-gated composite block
# ---------------------------------------------------------------------------


class KANGatedDualDomainAttention(nn.Module):
    """Composite block: three attention branches fused by KAN gates.

    Inputs are interleaved real/imag (the convention of the kspace cold
    diffusion U-Net). Internally we work in true complex tensors.

    Feature-domain contract (2026-07-03): the required ``feature_domain``
    kwarg ("kspace" | "image", derived from ``force_pure_kspace`` by
    ``FourierBridgeNetwork``) declares which domain the input tensor is in.
    The image/k-space/cross views are derived accordingly, so the
    radial-band branch genuinely sees a k-space spectrum and the image
    branch a spatial map in BOTH modes; the output is composed back in the
    input's domain (output domain == input domain), with the residual
    anchored on the untransformed native input (pitfall #20 guarantee
    holds bit-exactly in both modes). ``feature_domain="image"`` is
    numerically identical to the historical behavior.
    """

    def __init__(
        self,
        in_channels: int,
        num_heads: int = 4,
        time_embedding_dim: int | None = None,
        num_bands: int = 8,
        radial_beta: float = 2.0,
        kan_grid_size: int = 5,
        kan_spline_order: int = 3,
        kan_hidden: int = 16,
        gate_type: str = "kan",
        disable_branches: tuple[str, ...] = (),
        max_dense_attn_tokens: int = 4096,
        kspace_score_fn: str = "softmax",
        kspace_topk_k: int = 0,
        condition_on_smaps: bool = False,
        smap_film_hidden: int = 32,
        *,
        feature_domain: str,
    ) -> None:
        super().__init__()
        if in_channels % 2 != 0:
            raise ValueError(
                f"KANGatedDualDomainAttention expects even (interleaved) channels, "
                f"got {in_channels}"
            )
        self.feature_domain = validate_feature_domain(feature_domain)
        self.complex_channels = in_channels // 2
        self.in_channels = in_channels
        self.time_embedding_dim = time_embedding_dim or 0

        # Make num_heads divide complex_channels; fall back to a smaller value
        # if the per-block channel count is too small (e.g. 4 channels with
        # num_heads=4 -> head_dim=1, OK; 2 channels with num_heads=4 -> bad).
        nh = num_heads
        while nh > 1 and self.complex_channels % nh != 0:
            nh //= 2
        self.num_heads = max(1, nh)

        self.attn_img = ComplexMHA(self.complex_channels, self.num_heads)
        # Sparsity-promoting attention is most physically motivated in
        # the k-space branch (compressed-sensing prior) — pass through.
        self.attn_kspace = RadialBandAttention(
            self.complex_channels,
            self.num_heads,
            num_bands=num_bands,
            beta=radial_beta,
            score_fn=kspace_score_fn,
            topk_k=kspace_topk_k,
        )
        self.attn_cross = CrossDomainAttention(self.complex_channels, self.num_heads)

        # Phase-aware GAP gives [|h|; cos arg h; sin arg h] per channel.
        gate_in = 3 * self.complex_channels + self.time_embedding_dim
        self.gate_type = str(gate_type).lower()
        if self.gate_type not in ("kan", "mlp"):
            raise ValueError(f"gate_type must be 'kan' or 'mlp', got {gate_type!r}")

        def _make_gate() -> nn.Module:
            if self.gate_type == "kan":
                return nn.Sequential(
                    KANLayer(gate_in, kan_hidden, kan_grid_size, kan_spline_order),
                    KANLayer(kan_hidden, 1, kan_grid_size, kan_spline_order),
                )
            # MLP ablation: matched depth & hidden width.
            return nn.Sequential(
                nn.Linear(gate_in, kan_hidden),
                nn.SiLU(),
                nn.Linear(kan_hidden, 1),
            )

        self.gate_img = _make_gate()
        self.gate_kspace = _make_gate()
        self.gate_cross = _make_gate()

        # Maximum sequence length for the dense (image, cross) attention
        # branches. When H*W exceeds this, the image and cross branches
        # operate on a downsampled feature map (adaptive avg pool ->
        # attend -> bilinear interpolate up). The radial-band branch is
        # always full resolution because it's already band-local.
        # Default 4096 = 64*64; safe for diffusion U-Net depths typical
        # for 256² inputs after 2 downsamples.
        if max_dense_attn_tokens <= 0:
            raise ValueError(f"max_dense_attn_tokens must be positive, got {max_dense_attn_tokens}")
        self.max_dense_attn_tokens = int(max_dense_attn_tokens)

        # Branch ablations: any name in {'image', 'kspace', 'cross'} silences
        # the corresponding branch by zeroing its gate at forward time.
        valid_disable = {"image", "kspace", "cross"}
        bad = set(disable_branches) - valid_disable
        if bad:
            raise ValueError(f"disable_branches must be a subset of {valid_disable}; got {bad}")
        self.disable_branches = frozenset(disable_branches)

        # Last-step running diagnostics for telemetry; not load-bearing.
        # Stored as (E[g_img], E[g_kspace], E[g_cross]) over the most recent batch.
        self.register_buffer("_last_gates", torch.zeros(3), persistent=False)

        # Optional S-map conditioning (plan §3.1). When enabled, the
        # k-space branch is FiLM-modulated by a projection of the
        # complex coil-sensitivity spatial-frequency profile. The maps
        # are pulled from a stash on the *parent generator* (see
        # ``KSpaceColdDiffusionGenerator.set_current_smaps``); the
        # attention block itself remains a pure nn.Module with an
        # unchanged forward signature so it stays composable.
        self.condition_on_smaps = bool(condition_on_smaps)
        if self.condition_on_smaps:
            # FiLM MLP: takes [|F(S)|, cos(arg F(S)), sin(arg F(S))] pooled
            # per coil-channel -> produces (gamma, beta) per complex channel.
            # Implemented as two real linears outputting 2*C reals each
            # which we re-pair as complex gamma/beta.
            self.smap_film_proj = nn.Sequential(
                nn.Linear(3, smap_film_hidden),
                nn.SiLU(),
                nn.Linear(smap_film_hidden, 4 * self.complex_channels),
            )
        else:
            self.smap_film_proj = None

    def _pull_smaps_from_context(
        self, batch_size: int, device: torch.device
    ) -> torch.Tensor | None:
        """Walk parent modules to find a generator with current S-maps stashed.

        The KSpaceColdDiffusionGenerator stashes ``self._current_smaps`` at
        the start of every forward (and clears it at the end). Walking up
        the parent chain at attention-time avoids needing to thread S-maps
        through every intermediate layer's forward signature. Returns None
        for a *legitimate absence* (no parent generator, or no maps stashed);
        RAISES on a *present-but-mismatched* batch dim, which is a real shape
        bug — silently dropping the maps there would no-op the FiLM
        conditioning and hide the defect (pitfall #9 no-silent-fallback).
        """
        # Walk via the module tree using the registered parent ref the
        # PyTorch nn.Module does NOT provide by default — so we use a
        # hand-maintained `_parent_generator` weakref set by the generator
        # at module-build time. If absent, return None.
        gen = getattr(self, "_parent_generator", None)
        if gen is None:
            return None
        gen = gen() if callable(gen) else gen  # weakref or direct
        if gen is None:
            return None
        smaps = getattr(gen, "_current_smaps", None)
        if smaps is None:
            return None
        if smaps.shape[0] != batch_size:
            raise ValueError(
                "smap-context batch mismatch: stashed sensitivity_maps have batch "
                f"{smaps.shape[0]} but this attention block was called with batch "
                f"{batch_size}. This is a wiring bug — smaps must match the feature "
                "batch for FiLM conditioning; refusing to silently drop them."
            )
        if smaps.device != device:
            smaps = smaps.to(device)
        return smaps

    def _pull_smap_feats_from_context(
        self, batch_size: int, device: torch.device
    ) -> torch.Tensor | None:
        """Pooled S-map frequency features, computed ONCE per generator forward.

        PERF (2026-07-01): every attention block in the U-Net (~8 for the
        default depth) used to independently run ``fft2c(smaps)`` + the
        phase-aware GAP on the SAME stashed tensor every forward. The pooled
        ``[B, 3]`` features depend only on the stash, so the first block to
        need them computes and caches them on the generator
        (``_current_smap_feats``, cleared whenever ``set_current_smaps`` is
        called); sibling blocks reuse. Each block still applies its own
        learnable ``smap_film_proj``. Standalone blocks (mock/absent parent
        stash) keep the compute-locally behaviour via the miss path.
        """
        gen = getattr(self, "_parent_generator", None)
        if gen is None:
            return None
        gen = gen() if callable(gen) else gen
        if gen is None:
            return None

        feats = getattr(gen, "_current_smap_feats", None)
        if feats is not None:
            if feats.shape[0] != batch_size:
                raise ValueError(
                    "smap-context batch mismatch: cached smap features have "
                    f"batch {feats.shape[0]} but this attention block was "
                    f"called with batch {batch_size} (wiring bug — refusing "
                    "to silently drop FiLM conditioning)."
                )
            return feats.to(device) if feats.device != device else feats

        smaps = self._pull_smaps_from_context(batch_size, device)
        if smaps is None:
            return None
        feats = self._smap_freq_feats(smaps)
        # Plain attribute (not a registered buffer) — FakeGen test doubles and
        # nn.Module generators both accept it; cleared by set_current_smaps.
        gen._current_smap_feats = feats
        return feats

    @staticmethod
    def _smap_freq_feats(smaps: torch.Tensor) -> torch.Tensor:
        """Pool complex S-maps to ``[B, 3]`` phase-aware frequency features."""
        # FFT to get the spatial-frequency profile of each coil.
        smap_freq = fft2c(smaps)
        # Phase-aware GAP: [|F(S)|, cos(arg F(S)), sin(arg F(S))] per coil
        denom = smap_freq.abs() + 1e-8
        feats = torch.stack(
            [
                smap_freq.abs().mean(dim=(-2, -1)),  # [B, C_coils]
                (smap_freq.real / denom).mean(dim=(-2, -1)),
                (smap_freq.imag / denom).mean(dim=(-2, -1)),
            ],
            dim=-1,
        )  # [B, C_coils, 3]
        # Average across coils so the FiLM is coil-count-agnostic.
        return feats.mean(dim=1)  # [B, 3]

    def _smap_film(
        self, feats: torch.Tensor, complex_channels: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute (gamma, beta) FiLM parameters from pooled S-map features.

        Args:
            feats: ``[B, 3]`` pooled features from :meth:`_smap_freq_feats`
                (shared across sibling blocks via the generator cache).
            complex_channels: Target complex-channel count for the FiLM output.

        Returns:
            ``(gamma, beta)`` each ``[B, complex_channels, 1, 1]`` complex.
        """
        film_out = self.smap_film_proj(feats)  # [B, 4*C_complex]
        # Split into (gamma_real, gamma_imag, beta_real, beta_imag) and pair
        gr, gi, br, bi = film_out.chunk(4, dim=-1)
        # Initialize to identity-ish: gamma ~= 1 + small, beta ~= 0
        gamma = torch.complex(1.0 + gr, gi).view(-1, complex_channels, 1, 1)
        beta = torch.complex(br, bi).view(-1, complex_channels, 1, 1)
        return gamma, beta

    @staticmethod
    def _phase_aware_gap(h: torch.Tensor) -> torch.Tensor:
        """Compute [mag, cos(arg), sin(arg)] global average pool of complex h."""
        mag = h.abs().mean(dim=(-2, -1))  # [B, C]
        denom = h.abs() + 1e-8
        # cos(arg), sin(arg) are h.real / |h|, h.imag / |h|
        cos_p = (h.real / denom).mean(dim=(-2, -1))
        sin_p = (h.imag / denom).mean(dim=(-2, -1))
        return torch.cat([mag, cos_p, sin_p], dim=-1)  # [B, 3C]

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Interleaved ``[B, 2C, H, W]`` real tensor.
            t_emb: Optional ``[B, time_embedding_dim]`` time embedding.

        Returns:
            ``[B, 2C, H, W]`` interleaved real tensor of the same shape.
        """
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"KANGatedDualDomainAttention expected {self.in_channels} channels, "
                f"got {x.shape[1]}"
            )
        B, _, H, W = x.shape
        h_native = _interleaved_to_complex(x)  # [B, C, H, W], in feature_domain
        # Derive the image view (h_complex) and spectrum view (h_freq) from
        # the DECLARED input domain. Under "image" this is the historical
        # path; under "kspace" the input already IS the spectrum, so the
        # image view comes from the inverse transform — every branch below
        # then operates on the physically-correct view.
        if self.feature_domain == "kspace":
            h_freq = h_native
            h_complex = ifft2c(h_native)
        else:
            h_complex = h_native
            h_freq = fft2c(h_complex)

        # Decide whether the dense (image, cross) branches operate at
        # full or reduced resolution. The radial-band branch always
        # operates at full resolution because per-band token counts are
        # naturally bounded by num_bands.
        n_full = H * W
        if n_full > self.max_dense_attn_tokens:
            side = max(1, int(round(self.max_dense_attn_tokens**0.5)))
            target_h = min(H, side)
            target_w = min(W, side)
        else:
            target_h, target_w = H, W
        downsampled = (target_h, target_w) != (H, W)

        # --- Branches ---
        # Image branch (optionally downsampled)
        if downsampled:
            # Adaptive pool real and imag channels separately, then re-pair.
            real_ds = torch.nn.functional.adaptive_avg_pool2d(h_complex.real, (target_h, target_w))
            imag_ds = torch.nn.functional.adaptive_avg_pool2d(h_complex.imag, (target_h, target_w))
            h_complex_ds = torch.complex(real_ds, imag_ds)
        else:
            h_complex_ds = h_complex

        h_seq = h_complex_ds.permute(0, 2, 3, 1).reshape(
            B, target_h * target_w, self.complex_channels
        )
        attn_i_seq = self.attn_img(h_seq)
        attn_i_ds = (
            attn_i_seq.reshape(B, target_h, target_w, self.complex_channels)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        if downsampled:
            attn_i = torch.complex(
                torch.nn.functional.interpolate(
                    attn_i_ds.real, size=(H, W), mode="bilinear", align_corners=False
                ),
                torch.nn.functional.interpolate(
                    attn_i_ds.imag, size=(H, W), mode="bilinear", align_corners=False
                ),
            )
        else:
            attn_i = attn_i_ds

        # K-space branch (full res, in freq domain). In image mode the branch
        # output is brought to the image frame for composition (historical
        # path); in kspace mode it already lives in the native frame — keep it
        # there and conjugate the OTHER branches at composition time instead.
        attn_k_freq = self.attn_kspace(h_freq)
        attn_k = attn_k_freq if self.feature_domain == "kspace" else ifft2c(attn_k_freq)

        # Cross-domain branch (downsampled if needed; uses image-domain queries
        # against frequency keys/values, both at the reduced resolution)
        if downsampled:
            real_freq_ds = torch.nn.functional.adaptive_avg_pool2d(
                h_freq.real, (target_h, target_w)
            )
            imag_freq_ds = torch.nn.functional.adaptive_avg_pool2d(
                h_freq.imag, (target_h, target_w)
            )
            h_freq_ds = torch.complex(real_freq_ds, imag_freq_ds)
            attn_x_ds = self.attn_cross(h_complex_ds, h_freq_ds)
            attn_x = torch.complex(
                torch.nn.functional.interpolate(
                    attn_x_ds.real, size=(H, W), mode="bilinear", align_corners=False
                ),
                torch.nn.functional.interpolate(
                    attn_x_ds.imag, size=(H, W), mode="bilinear", align_corners=False
                ),
            )
        else:
            attn_x = self.attn_cross(h_complex, h_freq)

        # --- Gates ---
        # Descriptor is ALWAYS the image view (h_complex), in both modes:
        # magnitude/phase pooling is physically meaningful on spatial maps,
        # and mode-invariant gates are what make the two modes exact
        # conjugates of each other.
        gap = self._phase_aware_gap(h_complex)  # [B, 3C]
        if self.time_embedding_dim > 0:
            if t_emb is None:
                # Strategy/configurations sometimes call attention without a
                # time embedding (e.g. validation hooks). Fall back to zeros
                # to keep the block usable; gates still depend on input.
                t_emb_used = gap.new_zeros(B, self.time_embedding_dim)
            else:
                if t_emb.dim() != 2 or t_emb.shape[-1] != self.time_embedding_dim:
                    raise ValueError(
                        f"t_emb expected shape [B, {self.time_embedding_dim}], "
                        f"got {tuple(t_emb.shape)}"
                    )
                t_emb_used = t_emb
            z = torch.cat([gap, t_emb_used], dim=-1)
        else:
            z = gap

        g1 = torch.sigmoid(self.gate_img(z)).view(B, 1, 1, 1)
        g2 = torch.sigmoid(self.gate_kspace(z)).view(B, 1, 1, 1)
        g3 = torch.sigmoid(self.gate_cross(z)).view(B, 1, 1, 1)
        # Telemetry records what the gate PRODUCED, which is read before the
        # ablation zeroes it: a disabled branch's applied gate is 0.0 by
        # construction and says nothing, while the value the gate wanted is
        # what makes the ablation readable.
        with torch.no_grad():
            self._last_gates.copy_(torch.stack([g1.mean(), g2.mean(), g3.mean()]).detach())

        if "image" in self.disable_branches:
            g1 = torch.zeros_like(g1)
        if "kspace" in self.disable_branches:
            g2 = torch.zeros_like(g2)
        if "cross" in self.disable_branches:
            g3 = torch.zeros_like(g3)

        # Optional S-map conditioning: FiLM-modulate the k-space branch
        # output by a projection of the per-channel coil-sensitivity
        # spatial-frequency profile. The pooled features are pulled from
        # (and cached on) the parent generator — computed once per generator
        # forward, reused by every sibling block (see the puller's docstring).
        if self.condition_on_smaps:
            smap_feats = self._pull_smap_feats_from_context(B, h_complex.device)
            if smap_feats is not None:
                gamma, beta = self._smap_film(smap_feats, self.complex_channels)
                # FiLM: y = gamma * x + beta (broadcast over spatial dims)
                # gamma/beta shape: [B, C, 1, 1] complex. Applied to attn_k in
                # whatever domain it is composed (image resp. freq): gamma
                # commutes with the FFT, beta does not — a benign learned-
                # parameterisation difference between modes, not an inversion.
                attn_k = attn_k * gamma + beta

        # Additive identity residual: this block REFINES the input feature map
        # rather than replacing it. Without the ``h_native +`` term the three
        # gated branches must reconstruct the ENTIRE feature map, and because
        # the gates are per-batch scalars from a spatially-collapsed global pool
        # (and KAN spline bases saturate to near-constants more readily than a
        # SiLU-MLP), when the gates shrink toward 0 the output loses all
        # measurement information -> a fixed, acceleration-independent DC blob
        # (pitfall #20). That is the measurement-independence collapse the L4
        # gate flagged across the experiment_11 KAN dual-domain cohort, and the
        # reason KAN-gated arms collapsed while their byte-identical MLP-gated
        # siblings survived. The residual makes that trivial solution
        # structurally impossible — the input (a function of the undersampled
        # k-space) always flows through — and matches every sibling attention
        # block (self: ``x + out``; sparse: ``attn_out + x``; channel:
        # multiplicative; PhaseSafeDual: ``x + gamma*out``; wavelet_freq:
        # ``g*out + (1-g)*h``). With all branches disabled it reduces to a
        # pure passthrough, i.e. identical to ``attention_type='none'``.
        #
        # Native-frame anchoring (2026-07-03 feature-domain change): the
        # residual base is the UNTRANSFORMED native input in BOTH modes — it
        # never passes through an FFT roundtrip, so the zero-gate identity is
        # bit-exact. In kspace mode the image-frame branches (attn_i, attn_x)
        # are conjugated into the native k frame with a single fft2c of their
        # gated sum (gates are scalars, so this equals conjugating each), and
        # attn_k is already native (attn_k_freq). fft2c(0) == 0, so disabled
        # branches contribute exactly nothing.
        if self.feature_domain == "kspace":
            fused = h_native + fft2c(g1 * attn_i + g3 * attn_x) + g2 * attn_k
        else:
            fused = h_native + g1 * attn_i + g2 * attn_k + g3 * attn_x
        return _complex_to_interleaved(fused)
