"""Dual-domain attention over the acceleration operator's null space.

The acceleration operator is :math:`A = M \\cdot F`: a Cartesian mask selects
which k-space lines the scanner acquired. Its range is the observed support
:math:`M`; its null space is everything supported on :math:`1 - M`, the bins no
measurement constrains. Data consistency owns the first, so the network's only
real degree of freedom is the second. This block attends *from* the measured
bins and writes *only* into the unmeasured ones.

Reading the mask is not a leak. :math:`M` is the acquisition trajectory, fixed
before the scan and consumed by every physics-driven reconstruction. The block
reads ``x`` -- which descends from the masked measurement -- and ``M``, and
nothing else: not the target, not the fully-sampled ``kspace`` alias.

What the ``1 - M`` gate does and does not claim. It guarantees the FEATURE
values at observed positions leave this block bit-identical. It does not pin the
measurement: ``initial_conv`` is a 3x3 circular convolution, so null-bin content
has already mixed into observed-bin features upstream, and hard DC is what holds
the measured k-space.

Positional bias. The k-space branch weights each observed key by
``(1 + |delta w|) ** (-s_h)``, the phase-encode column distance to the query,
with a learned per-head ``s_h = softplus(.)`` initialised to 1. The mask
acquires whole PE columns, so this is the GRAPPA-like preference for the nearest
acquired lines; being a smooth function of the distance it serves every encoder
width, where the levels are centre crops of k-space with unchanged ``delta k``.

Normalisation statistics run over the acquired bins only
(:class:`~spectramr.models.blocks.kspace_attention_kernels.MaskedInstanceNorm2d`),
so zero-filled bins no longer move the key statistics with the acceleration.

Structurally-zero bins are NOT excluded, and cannot be from these inputs. The
outer columns a scanner never acquires (about 61 of 256 on M4Raw) carry ``M = 0``
exactly like undersampled columns, so the block writes into them; telling the
two apart needs the target support, which the block must not read. Acquired
bins whose MEASURED value is zero are likewise invisible here: the block sees
features after ``initial_conv``, never the raw measurement.

Which mask, at reverse time. Training passes :math:`M_t`, the support of the
input it degraded. The reverse loop passes the ACQUISITION support, held
constant while the monotone-infill loop reveals bins above it -- the safe
direction: revealed bins are the network's own output, never treated as
measurements, and still refinable because they are inside :math:`1 - M`.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c
from spectramr.models.blocks.attention import LinearAttention
from spectramr.models.blocks.attention_domains import (
    complex_to_interleaved,
    interleaved_to_complex,
    validate_feature_domain,
)
from spectramr.models.blocks.kspace_attention_kernels import (
    MaskedInstanceNorm2d,
    align_mask_to_grid,
    masked_linear_attention,
    pe_biased_linear_attention,
    sample_density_attention,
)
from spectramr.models.layers.complex_conv import ComplexConv2d

__all__ = [
    "NullSpaceDualDomainAttention",
    "ObservedKeyLinearAttention",
    "masked_linear_attention",
    "sample_density_attention",
]

# softplus^{-1}(1): the PE-distance decay starts at (1 + |d|) ** -1, which
# already ranks a column 3 away above one 60 away by 15x.
PE_SLOPE_INIT = math.log(math.e - 1.0)


class ObservedKeyLinearAttention(nn.Module):
    """O(N) linear attention whose keys and values are restricted to observed bins.

    ``phi(x) = elu(x) + 1`` (Katharopoulos et al., 2020) is strictly positive, so
    the per-query denominator is positive whenever anything is observed and the
    implied attention rows sum to one. The mask enters that SUM rather than an
    ``[L, L]`` score matrix, and the PE-distance bias enters through per-column
    key sums, so no quadratic tensor is ever formed.

    Queries and keys read the masked-instance-normalised features. Values are a
    bias-free ``ComplexConv2d`` of the RAW features, so the output at a null bin
    is a convex combination of complex-linear images of the observed values.
    """

    def __init__(self, channels: int, num_heads: int = 4, eps: float = 1e-6):
        """__init__.

        Args:
            channels (int): Interleaved-real channel count; ``channels // 2``
                must divide by ``num_heads`` so a complex pair never straddles
                two heads (the weights are per head and must hit Re and Im alike).
            num_heads (int): Attention heads.
            eps (float): Denominator floor; keeps an empty support at 0, not NaN.
        """
        super().__init__()
        if channels % 2 != 0 or (channels // 2) % num_heads != 0:
            raise ValueError(
                f"channels={channels} must be even with channels // 2 divisible by "
                f"num_heads={num_heads}"
            )
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.eps = eps
        self.norm = MaskedInstanceNorm2d(channels)
        self.qk = nn.Conv2d(channels, channels * 2, 1)
        self.v_proj = ComplexConv2d(channels // 2, channels // 2, 1, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.pe_slope_raw = nn.Parameter(torch.full((num_heads,), PE_SLOPE_INIT))

    def pe_slope(self) -> torch.Tensor:
        """Per-head non-negative decay exponent of the PE-distance bias."""
        return F.softplus(self.pe_slope_raw)

    def forward(self, x: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        """Attend from every position to the observed positions only.

        Args:
            x (torch.Tensor, shape (B, C, H, W)): Interleaved k-space features.
            observed (torch.Tensor, shape (B, 1, H, W)): 1 where acquired.

        Returns:
            torch.Tensor: ``[B, C, H, W]``, the projected attention output (no
            residual -- the caller owns how it is written back).
        """
        b, c, h, w = x.shape
        shape = (b, self.num_heads, self.head_dim, h * w)
        q, k = self.qk(self.norm(x, observed)).chunk(2, dim=1)
        out = self.attend(
            F.elu(q.reshape(shape)) + 1.0,
            F.elu(k.reshape(shape)) + 1.0,
            self.v_proj(x).reshape(shape),
            observed,
        )
        return self.proj(out.reshape(b, c, h, w))

    def attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        observed: torch.Tensor,
    ) -> torch.Tensor:
        """Masked, PE-distance-biased linear attention on ``[B, heads, D, N]``.

        A separate seam because ``x`` feeds q, k and v alike -- only here can a
        test perturb the keys and values without also moving the queries.

        Args:
            q (torch.Tensor): Positive query features.
            k (torch.Tensor): Positive key features; zeroed off ``observed``.
            v (torch.Tensor): Values.
            observed (torch.Tensor, shape (B, 1, H, W)): 1 where acquired.

        Returns:
            torch.Tensor: ``[B, heads, head_dim, N]``.
        """
        return pe_biased_linear_attention(q, k, v, observed, self.pe_slope(), eps=self.eps)


class NullSpaceDualDomainAttention(nn.Module):
    """Fill the acceleration operator's null space from its range, in both domains.

    Two branches meet in the k-space frame: the k-space branch attends from every
    bin to the OBSERVED bins (:class:`ObservedKeyLinearAttention`), and the image
    branch runs ordinary linear attention on the image-domain view, where
    undersampling artefacts are spatially structured rather than scattered. Their
    fusion is written back gated by ``1 - M``, which makes the block exactly
    complementary to hard data consistency and an exact no-op wherever everything
    is acquired -- the ``R = 1`` rung at ``t = 0``, for instance.
    """

    def __init__(
        self,
        in_channels: int,
        num_heads: int = 4,
        eps: float = 1e-6,
        *,
        feature_domain: str,
    ):
        """__init__.

        Args:
            in_channels (int): Interleaved-real channel count; must be even.
            num_heads (int): Heads for both attention branches.
            eps (float): Denominator floor for the masked branch.
            feature_domain (str): ``"kspace"`` or ``"image"`` -- the domain of
                the incoming feature map, derived from ``force_pure_kspace`` by
                the caller. Decides which view needs an FFT and which the output
                is conjugated back into; raises on anything else.
        """
        super().__init__()
        if in_channels % 2 != 0:
            raise ValueError(
                f"NullSpaceDualDomainAttention expects even (interleaved) channels, "
                f"got {in_channels}"
            )
        self.feature_domain = validate_feature_domain(feature_domain)
        complex_channels = in_channels // 2

        self.kspace_attn = ObservedKeyLinearAttention(in_channels, num_heads=num_heads, eps=eps)
        # zero_init_output=False: IdentityAtInitAttention wraps this block and owns
        # identity-at-init; a second zero-init underneath it is an exact saddle (#471).
        self.image_attn = LinearAttention(
            in_channels, num_heads=num_heads, norm_type="instance", zero_init_output=False
        )
        # Complex 1x1 over the two branches stacked as 2C complex channels, so the
        # fusion itself does not break the phase equivariance the FFT routing keeps.
        self.fuse = ComplexConv2d(2 * complex_channels, complex_channels, 1, bias=False)

    @staticmethod
    def align_mask(mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Centre-crop a mask onto the feature grid; see :func:`align_mask_to_grid`."""
        return align_mask_to_grid(mask, height, width)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Attend null <- observed and write the result into the null space only.

        Args:
            x (torch.Tensor, shape (B, 2C, H, W)): Interleaved-real feature map in
                ``self.feature_domain``.
            mask (torch.Tensor, shape (B, C, H, W)): Full-resolution acquisition
                mask, 1 where acquired. MANDATORY: a null space is undefined
                without one, and inferring it from feature magnitude would make an
                unconditioned run indistinguishable from a conditioned one
                (pitfall #9).

        Returns:
            torch.Tensor: Same shape and same domain as ``x``.
        """
        if mask is None:
            raise ValueError(
                "NullSpaceDualDomainAttention requires `mask`: the null space of the "
                "acceleration operator is undefined without the acquisition support. "
                "The generator stashes it via set_current_mask() so the reverse "
                "sampler's bare (x, t) re-entry still carries it."
            )

        h = interleaved_to_complex(x)
        if self.feature_domain == "kspace":
            h_k, h_i = h, ifft2c(h)
        else:
            h_k, h_i = fft2c(h), h

        observed = self.align_mask(mask.to(x.dtype), h_k.shape[-2], h_k.shape[-1])

        x_k = complex_to_interleaved(h_k)
        kspace_branch = self.kspace_attn(x_k, observed)
        image_branch = complex_to_interleaved(
            fft2c(interleaved_to_complex(self.image_attn(complex_to_interleaved(h_i))))
        )

        fused = self.fuse(torch.cat([kspace_branch, image_branch], dim=1))
        # Broadcast over the interleaved channel axis: masking a complex coefficient
        # scales its real and imaginary parts by the same factor.
        out_k = x_k + (1.0 - observed) * fused

        if self.feature_domain == "kspace":
            return out_k
        return complex_to_interleaved(ifft2c(interleaved_to_complex(out_k)))
