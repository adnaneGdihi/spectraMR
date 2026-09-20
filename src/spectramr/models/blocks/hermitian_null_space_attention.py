"""Phase-equivariant null-space attention with a conjugate-symmetry stream.

Two pieces of MRI physics that the rest of the ``complex_unet`` attention family
leaves on the table.

**Global phase.** The forward operator commutes with a global phase: multiply the
object by :math:`e^{i\\phi}` and every k-space sample rotates by the same
:math:`e^{i\\phi}`. A reconstruction network should therefore be *equivariant* --
:math:`f(e^{i\\phi}x) = e^{i\\phi}f(x)` -- and none of the family is, because they
apply real ``Conv2d``/``Linear`` and ``InstanceNorm``/``LayerNorm`` to the
interleaved Re/Im layout as if the two were unrelated channels. This block is
equivariant by construction: projections are ``ComplexConv2d``, normalisation is
``ComplexRMSNorm``, and the attention SCORES are built from
:math:`\\lvert\\langle q, w\\rangle\\rvert`, a complex inner product followed by a
magnitude. That magnitude is invariant to a shared rotation while still reading
the *relative* phase inside the feature vector -- so the weights are invariant,
the values rotate, and the output rotates with them.

**Conjugate symmetry.** A real-valued object has Hermitian k-space,
:math:`X(-k) = \\overline{X(k)}`; a globally phase-rotated one has
:math:`X(-k) = e^{2i\\phi}\\overline{X(k)}`. So an unobserved bin whose mirror WAS
acquired carries a strong prior, :math:`p(k) = \\hat{g}\\,\\overline{X(-k)}`, with
:math:`\\hat{g}` fitted by least squares over the observed mirror pairs. Fitting
it from the data rather than assuming :math:`\\hat{g} = 1` is what absorbs the
unknown :math:`e^{2i\\phi}` and the smooth background phase, and it keeps the
stream equivariant. This is Homodyne/POCS reconstruction expressed as an
attention stream, gated so it contributes only where a mirror exists.

**Measured constraint on that stream.** On this cohort's ``density_nested``
ladder at 256^2, the share of null bins whose mirror is observed is 43.8 % at
R = 2, 23.9 % at R = 3.8, 10.3 % at R = 8, 3.8 % at R = 16 and **1.2 % at
R = 32** -- the mask is not symmetric about k = 0, so the stream is real, but it
is strongest where the problem is easiest and nearly closed at high
acceleration. The ``mirror_observed`` gate closes smoothly, so this degrades
rather than misfires; do not read a null result at high R as a refutation of the
prior.

Everything else is inherited from :mod:`null_space_attention`: the same mandatory
mask, the same center-crop alignment, and the same ``1 - M`` write gate.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c
from spectramr.models.blocks.attention_domains import (
    complex_to_interleaved,
    interleaved_to_complex,
    validate_feature_domain,
)
from spectramr.models.blocks.null_space_attention import NullSpaceDualDomainAttention
from spectramr.models.layers.complex_conv import ComplexConv2d
from spectramr.models.layers.complex_norm import ComplexRMSNorm


def hermitian_mirror(x: torch.Tensor) -> torch.Tensor:
    """The :math:`k \\mapsto -k` partner on an ``fftshift``-ed even grid.

    ``flip`` alone maps index ``i`` to ``N-1-i``; the DC bin of a centred even
    grid sits at ``N//2``, so a ``roll`` by one puts ``i`` at ``N-i``, which is
    the actual Fourier partner. Getting this off by one silently pairs each bin
    with its neighbour's mirror and the prior turns into noise.
    """
    return torch.roll(torch.flip(x, dims=(-2, -1)), shifts=(1, 1), dims=(-2, -1))


class PhaseInvariantScoreAttention(nn.Module):
    """O(N) linear attention with invariant scores, complex values, masked keys.

    Features are ``elu(|<x, w>|) + 1``: a complex projection, then a magnitude.
    Positive (so the linear-attention denominator is well defined), invariant to
    a global phase (so the weights are), and still phase-aware, because the
    complex inner product reads relative phase inside the feature vector. The
    mask enters the key SUM, so restricting attention to the acquired bins costs
    nothing and no ``[L, L]`` matrix is formed.
    """

    def __init__(self, complex_channels: int, num_features: int = 32, eps: float = 1e-6):
        """__init__.

        Args:
            complex_channels (int): Complex channel count (half the interleaved).
            num_features (int): Kernel feature dimension of the score map.
            eps (float): Denominator floor; engages only where nothing is observed.
        """
        super().__init__()
        self.eps = eps
        self.norm = ComplexRMSNorm(complex_channels)
        self.q_proj = ComplexConv2d(complex_channels, num_features, 1, bias=False)
        self.k_proj = ComplexConv2d(complex_channels, num_features, 1, bias=False)
        self.v_proj = ComplexConv2d(complex_channels, complex_channels, 1, bias=False)
        self.out_proj = ComplexConv2d(complex_channels, complex_channels, 1, bias=False)

    def _features(self, projected: torch.Tensor) -> torch.Tensor:
        """``[B, 2F, H, W]`` interleaved -> ``[B, F, L]`` positive invariant features."""
        magnitude = interleaved_to_complex(projected).abs()
        return (F.elu(magnitude) + 1.0).flatten(-2, -1)

    def forward(self, x: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        """Attend from every bin to the observed bins.

        Args:
            x (torch.Tensor, shape (B, 2C, H, W)): Interleaved-real features.
            observed (torch.Tensor, shape (B, 1, H, W)): 1 where acquired.

        Returns:
            torch.Tensor: ``[B, 2C, H, W]``, projected output without a residual.
        """
        b, _, h, w = x.shape
        normed = self.norm(x)
        q = self._features(self.q_proj(normed))
        k = self._features(self.k_proj(normed)) * observed.reshape(b, 1, -1)
        v = interleaved_to_complex(self.v_proj(normed)).flatten(-2, -1)

        kv = torch.einsum("bfl,bcl->bfc", k.to(v.dtype), v)
        numerator = torch.einsum("bfl,bfc->bcl", q.to(v.dtype), kv)
        denominator = torch.einsum("bfl,bf->bl", q, k.sum(dim=-1)).clamp_min(self.eps)
        out = (numerator / denominator.unsqueeze(1)).reshape(b, -1, h, w)
        return self.out_proj(complex_to_interleaved(out))


class HermitianNullSpaceAttention(nn.Module):
    """Phase-equivariant null-space attention plus a conjugate-symmetry stream.

    Three contributions, all written through the ``1 - M`` gate so the observed
    support leaves the block bit-identical and the whole thing is an exact no-op
    on a fully sampled rung: the masked k-space stream, the image-domain stream
    (de-aliasing, where undersampling artefacts are spatially structured), and
    the Hermitian prior at null bins whose mirror was acquired.
    """

    def __init__(
        self,
        in_channels: int,
        num_features: int = 32,
        eps: float = 1e-6,
        *,
        feature_domain: str,
    ):
        """__init__.

        Args:
            in_channels (int): Interleaved-real channel count; must be even.
            num_features (int): Kernel feature dimension of both streams.
            eps (float): Denominator floor for the attention and the ``g`` fit.
            feature_domain (str): ``"kspace"`` or ``"image"``; raises otherwise.
        """
        super().__init__()
        if in_channels % 2 != 0:
            raise ValueError(f"expects even (interleaved) channels, got {in_channels}")
        self.feature_domain = validate_feature_domain(feature_domain)
        self.eps = eps
        complex_channels = in_channels // 2

        self.kspace_attn = PhaseInvariantScoreAttention(complex_channels, num_features, eps)
        self.image_attn = PhaseInvariantScoreAttention(complex_channels, num_features, eps)
        self.fuse = ComplexConv2d(2 * complex_channels, complex_channels, 1, bias=False)
        # Real per-channel gate: a magnitude scale commutes with a global phase, so
        # the conjugate stream stays equivariant however hard it is switched on.
        self.hermitian_gate = nn.Parameter(torch.zeros(1, complex_channels, 1, 1))

    def hermitian_prior(
        self, h_k: torch.Tensor, observed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Least-squares conjugate prediction and the bins it is defined on.

        Fits ``g`` per (batch, coil) over the bins where BOTH partners were
        acquired, minimising ``|X(k) - g*conj(X(-k))|^2``. Fitting rather than
        assuming ``g = 1`` is what makes this survive the unknown global phase
        and the smooth background phase a real acquisition carries.

        Args:
            h_k (torch.Tensor, shape (B, C, H, W)): Complex k-space features.
            observed (torch.Tensor, shape (B, 1, H, W)): 1 where acquired.

        Returns:
            tuple: ``(prediction, mirror_observed)`` -- the predicted complex
            value at every bin, and the 0/1 map of bins whose mirror is acquired.
        """
        mirrored = hermitian_mirror(h_k)
        mirror_observed = hermitian_mirror(observed)
        both = observed * mirror_observed
        numerator = (h_k * mirrored * both).sum(dim=(-2, -1), keepdim=True)
        denominator = (mirrored.abs().pow(2) * both).sum(dim=(-2, -1), keepdim=True)
        g = numerator / denominator.clamp_min(self.eps)
        return g * mirrored.conj(), mirror_observed

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Fill the null space from the measured bins and their conjugates.

        Args:
            x (torch.Tensor, shape (B, 2C, H, W)): Interleaved-real features in
                ``self.feature_domain``.
            mask (torch.Tensor, shape (B, C, H, W)): Full-resolution acquisition
                mask, 1 where acquired. MANDATORY (pitfall #9).

        Returns:
            torch.Tensor: Same shape and domain as ``x``.
        """
        if mask is None:
            raise ValueError(
                "HermitianNullSpaceAttention requires `mask`: both the null space "
                "and the conjugate-pair support are undefined without the "
                "acquisition mask, and inferring it from feature magnitude would "
                "make an unconditioned run look conditioned."
            )

        h = interleaved_to_complex(x)
        if self.feature_domain == "kspace":
            h_k, h_i = h, ifft2c(h)
        else:
            h_k, h_i = fft2c(h), h

        observed = NullSpaceDualDomainAttention.align_mask(
            mask.to(x.dtype), h_k.shape[-2], h_k.shape[-1]
        )

        x_k = complex_to_interleaved(h_k)
        kspace_branch = self.kspace_attn(x_k, observed)
        # The image stream attends over EVERY pixel. ``observed`` indexes k-space
        # bins, so on the ifft2c view it would restrict pixel keys to "pixels whose
        # index happens to match an acquired k-column" -- an arbitrary crop, not
        # physics. Undersampling artefacts are spread over the whole image, which is
        # the reason this branch exists at all.
        image_branch = complex_to_interleaved(
            fft2c(
                interleaved_to_complex(
                    self.image_attn(complex_to_interleaved(h_i), torch.ones_like(observed))
                )
            )
        )
        fused = interleaved_to_complex(self.fuse(torch.cat([kspace_branch, image_branch], dim=1)))

        prediction, mirror_observed = self.hermitian_prior(h_k, observed)
        fused = fused + self.hermitian_gate * mirror_observed * prediction

        out_k = complex_to_interleaved(h_k + (1.0 - observed) * fused)
        if self.feature_domain == "kspace":
            return out_k
        return complex_to_interleaved(ifft2c(interleaved_to_complex(out_k)))
