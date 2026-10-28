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
from spectramr.models.blocks.kspace_attention_kernels import (
    align_mask_to_grid,
    masked_linear_attention,
    stratified_conjugate_subsample,
)

# Query-block size for :class:`ComplexMHA`: above it the dense score matrix is
# built per query chunk, which is exact (softmax/top-k act per query) and caps
# peak memory at O(chunk * N) -- the experiment_11 KAN cohort OOM'd on the dense
# ``[B, h, N, N]`` tensor. Per-instance override via ``self._query_chunk``.
_COMPLEX_MHA_QUERY_CHUNK = 2048

# Per-head temperature for QK-normalised scores: init log(10), capped at
# log(100), the cosine-attention parameterisation of Swin-V2.
_LOGIT_SCALE_INIT = math.log(10.0)
_LOGIT_SCALE_MAX = math.log(100.0)

# Seed of the fixed per-band token subsample in :class:`RadialBandAttention`.
_BAND_SAMPLE_SEED = 0

# ---------------------------------------------------------------------------
# Phase-aware attention primitives (memory-bounded)
# ---------------------------------------------------------------------------
# ``Re(q . conj(k))`` summed over the feature dim equals ``qr.kr^T + qi.ki^T``,
# so scores are built from two REAL matmuls and the complex ``[..., N_q, N_k]``
# intermediate (twice the bytes, the 2026-06-22 OOM frame) is never formed.


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


def _qk_normalise(t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """L2-normalise complex ``[..., d]`` rows over their ``2d`` real features."""
    sq = t.real.square().sum(-1, keepdim=True) + t.imag.square().sum(-1, keepdim=True)
    return t * torch.rsqrt(sq.clamp_min(eps * eps))


def _elu_features(t: torch.Tensor) -> torch.Tensor:
    """Positive linear-attention features ``elu([Re t, Im t]) + 1``."""
    return nn.functional.elu(torch.cat([t.real, t.imag], dim=-1)) + 1.0


def _to_seq(x: torch.Tensor) -> torch.Tensor:
    """``[B, C, H, W]`` -> ``[B, H*W, C]``."""
    return x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, x.shape[1])


def _from_seq(seq: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """``[B, H*W, C]`` -> ``[B, C, H, W]``."""
    return seq.reshape(seq.shape[0], height, width, -1).permute(0, 3, 1, 2).contiguous()


def _token_grid(height: int, width: int, max_tokens: int) -> tuple[int, int]:
    """Largest ``(h, w) <= (height, width)`` near-square grid of ``<= max_tokens``."""
    if height * width <= max_tokens:
        return height, width
    side = max(1, round(max_tokens**0.5))
    return min(height, side), min(width, side)


# ---------------------------------------------------------------------------
# Complex multi-head attention
# ---------------------------------------------------------------------------


class ComplexMHA(nn.Module):
    """Complex multi-head attention with phase-aware, QK-normalised scores.

    The score is ``tau_h * Re(q_hat k_hat^H)``: q and k are L2-normalised per
    head over their ``2d`` real features and ``tau_h = exp(logit_scale_h)`` is a
    learned per-head temperature. The unnormalised ``Re(q k^H) / sqrt(d)`` is
    ``|q||k| cos(arg q - arg k)``, so it ranks keys by magnitude first, and on a
    k-space sequence the DC bin -- orders of magnitude above the rest under the
    power-law spectrum -- took every softmax row. Normalised, the score is a
    cosine and magnitude reaches the output only through the values.

    ``forward`` is dense softmax (or top-k) attention, optionally against a
    separate ``context`` sequence; ``forward_linear`` is the O(N) linear form on
    the same projections and temperature, with features ``elu(tau q_hat) + 1``,
    for token counts where an ``N x N`` matrix does not fit.

    Args:
        channels: Complex channel count.
        num_heads: Number of attention heads. Must divide ``channels``.
        score_fn: ``softmax`` (dense) or ``topk`` (keep the ``topk_k`` largest
            scores per query, a compressed-sensing-style sparsity prior).
        topk_k: Keys kept per query under ``topk``; 0 means ``ceil(sqrt(N))``.
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
        # Real-valued projections applied to the concatenated (real, imag) stack.
        self.q_proj = nn.Linear(2 * channels, 2 * channels, bias=False)
        self.k_proj = nn.Linear(2 * channels, 2 * channels, bias=False)
        self.v_proj = nn.Linear(2 * channels, 2 * channels, bias=False)
        self.out_proj = nn.Linear(2 * channels, 2 * channels, bias=False)
        self.logit_scale = nn.Parameter(torch.full((num_heads,), _LOGIT_SCALE_INIT))

    def _project(self, proj: nn.Linear, seq_complex: torch.Tensor) -> torch.Tensor:
        out = proj(torch.cat([seq_complex.real, seq_complex.imag], dim=-1))
        return torch.complex(out[..., : self.channels], out[..., self.channels :])

    def _heads(self, t: torch.Tensor) -> torch.Tensor:
        b, n, _ = t.shape
        return t.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

    def _temperature(self) -> torch.Tensor:
        return self.logit_scale.clamp(max=_LOGIT_SCALE_MAX).exp().view(1, -1, 1, 1)

    def _qkv(
        self, seq: torch.Tensor, context: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalised ``q_hat, k_hat`` and values, each ``[B, h, N, d]`` complex."""
        ctx = seq if context is None else context
        q = _qk_normalise(self._heads(self._project(self.q_proj, seq)))
        k = _qk_normalise(self._heads(self._project(self.k_proj, ctx)))
        return q, k, self._heads(self._project(self.v_proj, ctx))

    def _output(self, out: torch.Tensor) -> torch.Tensor:
        b, _, n, _ = out.shape
        out = out.transpose(1, 2).reshape(b, n, self.channels)
        ri = self.out_proj(torch.cat([out.real, out.imag], dim=-1))
        return torch.complex(ri[..., : self.channels], ri[..., self.channels :])

    def attention_weights(
        self, seq: torch.Tensor, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Dense ``[B, h, N_q, N_k]`` attention weights; unchunked, for inspection."""
        q, k, _ = self._qkv(seq, context)
        return self._apply_score_fn(_phase_aware_real_scores(q * self._temperature(), k, 1.0))

    def forward(
        self, seq_complex: torch.Tensor, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Dense attention of ``[B, N, C]`` queries over ``context`` (default: self)."""
        q, k, v = self._qkv(seq_complex, context)
        q = q * self._temperature()

        def attend(q_block: torch.Tensor) -> torch.Tensor:
            scores = _phase_aware_real_scores(q_block, k, 1.0)
            return _complex_weighted_sum(self._apply_score_fn(scores), v)

        n = q.shape[2]
        chunk = int(getattr(self, "_query_chunk", _COMPLEX_MHA_QUERY_CHUNK))
        if chunk <= 0 or chunk >= n:
            return self._output(attend(q))
        parts = [attend(q[:, :, i : i + chunk]) for i in range(0, n, chunk)]
        return self._output(torch.cat(parts, dim=2))

    def forward_linear(self, seq_complex: torch.Tensor) -> torch.Tensor:
        """O(N) linear self-attention of a ``[B, N, C]`` sequence; no ``N x N`` tensor."""
        q, k, v = self._qkv(seq_complex, None)
        tau = self._temperature()
        phi_q = _elu_features(q * tau).transpose(-2, -1)
        phi_k = _elu_features(k * tau).transpose(-2, -1)
        everywhere = phi_q.new_ones(phi_q.shape[0], phi_q.shape[-1])
        out = masked_linear_attention(phi_q, phi_k, v.transpose(-2, -1), everywhere)
        return self._output(out.transpose(-2, -1))

    def _apply_score_fn(self, scores: torch.Tensor) -> torch.Tensor:
        """Softmax, or softmax over the top-k scores per query (others -> -inf)."""
        if self.score_fn == "softmax":
            return torch.softmax(scores, dim=-1)
        n_k = scores.shape[-1]
        k = self.topk_k if self.topk_k > 0 else max(1, math.ceil(math.sqrt(n_k)))
        k = min(k, n_k)
        if k == n_k:
            return torch.softmax(scores, dim=-1)
        threshold = scores.topk(k, dim=-1).values[..., -1:].detach()
        return torch.softmax(scores.masked_fill(scores < threshold, float("-inf")), dim=-1)


# ---------------------------------------------------------------------------
# K-space radial-band attention
# ---------------------------------------------------------------------------


class RadialBandAttention(nn.Module):
    """Self-attention restricted to concentric annular k-space bands.

    Attention is computed independently within each band but shares the Q/K/V
    projections across bands. Bands are logarithmically spaced so the DC region
    is more finely resolved.

    Token cap. An outer band of a 256^2 grid holds ~32k tokens, whose dense
    attention matrix alone is ~32 GiB. A band above ``max_band_tokens`` attends
    a fixed subsample drawn over its WHOLE index set
    (:func:`~spectramr.models.blocks.kspace_attention_kernels.stratified_conjugate_subsample`):
    every row of the band is represented in both halves about ``N // 2``,
    columns are drawn uniformly rather than on a stride, and each selected bin
    arrives with its conjugate partner ``((N_h - i) % N_h, (N_w - j) % N_w)``.
    The subsample is seeded (``_BAND_SAMPLE_SEED`` + band) and cached per shape.
    Every bin of such a band that was not attended receives the band mean of the
    attended outputs, so the branch output is defined everywhere and no bin of a
    capped band falls back to the identity.
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
        # Per-shape cache built once on the host, so the forward only does
        # index_select/index_copy_ and never a data-dependent .nonzero() (a sync).
        # ``_band_slot`` maps each flat bin to its band's position in
        # ``_band_sel``; ``_unattended`` flags bins left out by the cap.
        self._cache_key: tuple[int, int, torch.device] | None = None
        self._band_sel: list[torch.Tensor] = []
        self._has_unattended = False
        self.register_buffer("_band_index", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("_band_slot", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("_unattended", torch.empty(0, dtype=torch.bool), persistent=False)

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
        return torch.bucketize(r, edges[1:-1]).long()

    @torch.no_grad()
    def _build_band_sel(self, band_index: torch.Tensor) -> list[torch.Tensor]:
        """Per-band attended indices (empty bands dropped) and the fill maps.

        A band under the cap attends every bin in raster order; a band over it
        attends the stratified, conjugate-paired subsample. Also sets
        ``_band_slot`` / ``_unattended`` / ``_has_unattended`` for the mean fill.
        """
        height, width = band_index.shape
        band_flat = band_index.flatten()
        slot = torch.zeros_like(band_flat)
        unattended = torch.zeros_like(band_flat, dtype=torch.bool)
        sel: list[torch.Tensor] = []
        for k in range(self.num_bands):
            idx = (band_flat == k).nonzero(as_tuple=False).flatten()
            if idx.numel() == 0:
                continue
            slot[idx] = len(sel)
            chosen = stratified_conjugate_subsample(
                idx, height, width, self.max_band_tokens, _BAND_SAMPLE_SEED + k
            )
            if chosen.numel() < idx.numel():
                unattended[idx] = True
                unattended[chosen] = False
            sel.append(chosen)
        self._band_slot, self._unattended = slot, unattended
        self._has_unattended = bool(unattended.any())
        return sel

    def forward(self, kspace_complex: torch.Tensor) -> torch.Tensor:
        """Args:
            kspace_complex: ``[B, C, H, W]`` complex k-space features.
        Returns:
            ``[B, C, H, W]`` complex k-space features.
        """
        H, W = kspace_complex.shape[-2:]
        cache_key = (H, W, kspace_complex.device)
        if self._cache_key != cache_key:
            self._band_index = self._build_band_index(H, W, kspace_complex.device)
            self._band_sel = self._build_band_sel(self._band_index)
            self._cache_key = cache_key
        flat = _to_seq(kspace_complex)
        out_flat = torch.zeros_like(flat)
        band_means: list[torch.Tensor] = []
        for sel_idx in self._band_sel:
            band_out = self.attn(flat.index_select(1, sel_idx))  # [B, n_sel, C]
            out_flat.index_copy_(1, sel_idx, band_out)
            band_means.append(band_out.mean(dim=1))
        if self._has_unattended:
            fill = torch.stack(band_means, dim=1).index_select(1, self._band_slot)
            out_flat = torch.where(self._unattended.view(1, -1, 1), fill, out_flat)
        return _from_seq(out_flat, H, W)


# ---------------------------------------------------------------------------
# Cross-domain attention
# ---------------------------------------------------------------------------


class CrossDomainAttention(nn.Module):
    """K-space queries against image-domain keys/values, written back in k-space.

    Pairing an image query at pixel ``p`` with k-space keys and writing the
    result into k-space bin ``p`` has no physical correspondence, so the roles
    are fixed by where each quantity lives. Queries are the k-space bins of a
    centre crop of at most ``max_tokens`` bins, taken with ``KSpaceCrop``'s
    arithmetic so every query keeps its frequency location. Keys and values are
    the image-domain features, average-pooled IN THE IMAGE DOMAIN to the same
    budget when the grid exceeds it (a low-pass of an image is legitimate; a
    pooled spectrum is a frequency rescaling). The result is a k-space update
    for exactly the queried bins and zero outside the crop: nothing in k-space
    is pooled, resampled, or brought through an image-domain upsample.
    """

    def __init__(self, channels: int, num_heads: int = 4, max_tokens: int = 4096) -> None:
        super().__init__()
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
        self.attn = ComplexMHA(channels, num_heads)
        self.max_tokens = int(max_tokens)

    def forward(self, kspace_complex: torch.Tensor, image_complex: torch.Tensor) -> torch.Tensor:
        """Args:
            kspace_complex: ``[B, C, H, W]`` complex k-space features (queries).
            image_complex: ``[B, C, H, W]`` complex image-domain features.
        Returns:
            ``[B, C, H, W]`` complex k-space update, zero outside the crop.
        """
        B, C, H, W = kspace_complex.shape
        th, tw = _token_grid(H, W, self.max_tokens)
        sh, sw = (H - th) // 2, (W - tw) // 2
        crop = kspace_complex[..., sh : sh + th, sw : sw + tw]
        if (th, tw) != (H, W):
            image_complex = torch.complex(
                nn.functional.adaptive_avg_pool2d(image_complex.real, (th, tw)),
                nn.functional.adaptive_avg_pool2d(image_complex.imag, (th, tw)),
            )
        out = _from_seq(self.attn(_to_seq(crop), context=_to_seq(image_complex)), th, tw)
        if (th, tw) == (H, W):
            return out
        update = kspace_complex.new_zeros(B, C, H, W)
        update[..., sh : sh + th, sw : sw + tw] = out
        return update


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
    ``KANLayer``s) or a matched MLP; that gate is the whole difference between
    ``attn_kan_wavelet`` and ``attn_mlp_wavelet`` (pitfall #16).

    Write gate. When the caller passes the acquisition ``mask`` (the generator
    does, via ``IdentityAtInitAttention``), the update is applied in k-space
    through ``1 - M`` on the centre-cropped level mask, so the acquired bins of
    the feature map leave the block unchanged. Without a mask the update covers
    the whole plane, as before; no support is inferred from the features.
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

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Gated wavelet-attention update of interleaved ``x`` ``[B, 2C, H, W]``.

        ``mask`` is the full-resolution acquisition mask ``[B, C', H', W']``;
        when given, the update reaches only the bins where it is 0.
        """
        del t_emb  # not used; signature matches KANGatedDualDomainAttention
        h_native = _interleaved_to_complex(x)
        # Image view: the Haar DWT prior only makes sense on spatial features.
        h_img = ifft2c(h_native) if self.feature_domain == "kspace" else h_native
        out_complex = self.attn(h_img)
        # Descriptor from the image view in both modes, so gate semantics are
        # mode-invariant.
        b = h_native.shape[0]
        g = torch.sigmoid(self.gate(self._phase_aware_gap(h_img))).view(b, 1, 1, 1)
        if mask is not None:
            observed = align_mask_to_grid(mask.to(x.dtype), x.shape[-2], x.shape[-1])
            update_k = (1.0 - observed) * fft2c(out_complex - h_img)
            update = update_k if self.feature_domain == "kspace" else ifft2c(update_k)
            fused = h_native + g * update
        elif self.feature_domain == "kspace":
            # Residual anchored on the UNTRANSFORMED input, so g = 0 returns it
            # bit-exactly (pitfall #20).
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
        # Dense attention is O(N^2): a 128^2 band alone needs a 16k x 16k score
        # matrix. Bands above ``max_tokens`` take the O(N) linear path of the
        # same ComplexMHA instead -- never a 2x2 average pool, which cancels
        # the alternating-sign detail coefficients the band exists to carry.
        self.max_tokens = int(max_tokens)
        # One ComplexMHA per level, shared by that level's bands, so each scale
        # keeps its own inductive bias at a low parameter count.
        self.attn_bands: nn.ModuleList = nn.ModuleList()
        for _ in range(num_levels):
            band_attn = ComplexMHA(channels, num_heads, score_fn=score_fn, topk_k=topk_k)
            self.attn_bands.append(band_attn)

    def _attend_band(self, band: torch.Tensor, attn: ComplexMHA) -> torch.Tensor:
        """Apply one ComplexMHA over every coefficient of a single band.

        Dense up to ``self.max_tokens`` coefficients, linear above it; the band
        is attended at full resolution either way.
        """
        _, _, H, W = band.shape
        seq = _to_seq(band)
        out = attn.forward_linear(seq) if self.max_tokens < H * W else attn(seq)
        return _from_seq(out, H, W)

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
    holds bit-exactly in both modes).

    Branches. The image branch is linear attention over every pixel, so its
    update reaches every frequency; the radial branch attends k-space within
    annuli; the cross branch attends centre-cropped k-space queries to
    image-domain keys/values and returns a k-space update on the cropped bins
    (:class:`CrossDomainAttention`).
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

        # Token budget of the cross branch's dense attention (k-space crop and
        # image pooling). The image branch is linear and the radial branch
        # band-capped, so neither reads it. 4096 = 64^2.
        if max_dense_attn_tokens <= 0:
            raise ValueError(f"max_dense_attn_tokens must be positive, got {max_dense_attn_tokens}")
        self.max_dense_attn_tokens = int(max_dense_attn_tokens)

        # Image branch: O(N) linear attention at full resolution (forward_linear).
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
        self.attn_cross = CrossDomainAttention(
            self.complex_channels, self.num_heads, max_tokens=self.max_dense_attn_tokens
        )

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

        # Image branch: linear attention over every pixel, no pooling, so the
        # update is not band-limited to the pooled grid's frequencies.
        attn_i = _from_seq(self.attn_img.forward_linear(_to_seq(h_complex)), H, W)

        # K-space branch (full res, in freq domain). In image mode its output is
        # brought to the image frame; in kspace mode it is already native.
        attn_k_freq = self.attn_kspace(h_freq)
        attn_k = attn_k_freq if self.feature_domain == "kspace" else ifft2c(attn_k_freq)

        # Cross branch: a k-space update on the centre-cropped query bins.
        attn_x_freq = self.attn_cross(h_freq, h_complex)

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

        # Additive identity residual: the block refines its input rather than
        # replacing it, so gates shrinking toward 0 (KAN splines saturate more
        # readily than an MLP) cannot collapse the output to a measurement-
        # independent DC blob (pitfall #20); with every branch disabled it is a
        # passthrough. The residual base is the UNTRANSFORMED native input and
        # only gated updates are FFT'd, so fft2c(0) == ifft2c(0) == 0 keeps the
        # zero-gate identity bit-exact in both modes.
        if self.feature_domain == "kspace":
            fused = h_native + fft2c(g1 * attn_i) + g2 * attn_k + g3 * attn_x_freq
        else:
            fused = h_native + g1 * attn_i + g2 * attn_k + ifft2c(g3 * attn_x_freq)
        return _complex_to_interleaved(fused)
