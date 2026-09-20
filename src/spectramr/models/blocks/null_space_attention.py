"""Dual-domain attention over the acceleration operator's null space.

The acceleration operator is :math:`A = M \\cdot F`: a Cartesian mask selects
which k-space lines the scanner acquired. Its range is the observed support
:math:`M`; its null space is everything supported on :math:`1 - M`, the bins no
measurement constrains. Data consistency owns the first, so the network's only
real degree of freedom is the second. This block attends *from* the measured
bins and writes *only* into the unmeasured ones, which aims its whole capacity
at the part of the reconstruction that is actually learnable.

Reading the mask is not a leak. :math:`M` is the acquisition trajectory, fixed
before the scan and consumed by every physics-driven reconstruction, this
repository's own hard data consistency included. The block reads ``x`` -- which
descends from the masked measurement -- and ``M``, and nothing else: not the
target, not the fully-sampled ``kspace`` alias, not the pre-degradation source.

What the ``1 - M`` gate does and does not claim. It guarantees the FEATURE
values at observed positions leave this block bit-identical. It does not pin the
measurement: ``initial_conv`` is a 3x3 circular convolution, so null-bin content
has already mixed into observed-bin features upstream, and hard DC is what holds
the measured k-space. The k-space branch also carries no positional bias in this
version, so it is content-based mixing over observed keys -- neither a
neighbourhood interpolation nor GRAPPA.

Which mask, at reverse time. Training passes :math:`M_t`, the support of the
input it degraded, so mask and input support coincide exactly. The reverse loop
passes the ACQUISITION support, held constant while the monotone-infill loop
reveals bins above it -- so from step two on the input carries energy in bins
this block treats as null. That is the safe direction of the two: revealed bins
are the network's own output, and the block never attends to one as though it
were a measurement. It can still refine them, because they are inside
:math:`1 - M`.

Known R-dependence, v1. The k-space branch normalises (``InstanceNorm2d``) before
it restricts the keys, and unobserved bins are near-zero, so the normalisation
statistics move with how much of k-space was acquired. That is a property of the
mask, not of the target, so it is not a leak -- but it does mean the branch sees
slightly different key statistics at different rungs. Normalising over the
observed support alone is the fix if it ever matters.

Like the rest of the ``complex_unet`` attention family this operates on the
interleaved-real ``[B, 2C, H, W]`` layout with real-valued projections, so it is
not phase-equivariant; the FFT routing and the ``1 - M`` gate are.
"""

from __future__ import annotations

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
from spectramr.models.layers.complex_conv import ComplexConv2d


def masked_linear_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    observed: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Linear attention on ``[B, heads, head_dim, N]``, restricted to a support.

    Module level because two blocks need exactly this and the restriction IS the
    mechanism in both: duplicating it would let the grid and off-grid versions
    drift into computing different things under one name (non-negotiable 17).

    The mask enters the key SUM rather than an ``[N, N]`` score matrix, which is
    the only form that fits a 200k-sample trajectory or a 256^2 grid.

    Args:
        q: Positive query features.
        k: Positive key features; zeroed off ``observed`` here, so a caller
            cannot pass an unmasked key that survives.
        v: Values.
        observed: 1 where acquired; broadcast over the flat ``N`` axis.
        eps: Denominator floor.

    Returns:
        ``[B, heads, head_dim, N]``.
    """
    k = k * observed.reshape(observed.shape[0], 1, 1, -1)
    # The denominator stays REAL. It is the row normaliser -- and for an off-grid
    # trajectory it is also the learned density compensation factor -- so it must
    # be a positive magnitude, not something that can rotate the phase it divides.
    denominator = torch.einsum("bhdn,bhd->bhn", q, k.sum(dim=-1)).unsqueeze(2)
    if torch.is_complex(v):
        q, k = q.to(v.dtype), k.to(v.dtype)
    kv = torch.einsum("bhdn,bhen->bhde", k, v)
    numerator = torch.einsum("bhdn,bhde->bhen", q, kv)
    return numerator / denominator.clamp_min(eps).to(numerator.real.dtype)


def sample_density_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    observed: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Linear attention normalised over SAMPLES rather than over queries.

    :func:`masked_linear_attention` divides each query by the kernel mass it
    sees, which makes its output a weighted AVERAGE. Gridding is a
    density-compensated SUM -- the weight belongs to the sample, not to the
    grid point -- and the two are not the same operator at any kernel width.
    Measured with an exact Gaussian on a radial trajectory, the average form
    does not converge at any width while the sum form reaches 1.3x the fixed
    Kaiser-Bessel adjoint.

    ``c_j = m_j / sum_j' <phi(k_j), phi(k_j')> m_j'`` is one Pipe-Menon
    iteration. A duplicated sample raises the local density and so halves each
    copy's weight, which is what preserves the duplicate-invariance the
    per-query denominator provided.

    Args:
        q: ``[B, H, D, M]`` query features over ``M`` grid points.
        k: ``[B, H, D, N]`` key features over ``N`` samples.
        v: ``[B, H, C, N]`` sample values, real or complex.
        observed: ``[B, N]``, 1 where acquired. It enters the DENSITY sum too:
            a dropped spoke must not crowd its neighbours.
        eps: Floor on the density, for a sample no other sample reaches.
    """
    mask = observed.reshape(observed.shape[0], 1, 1, -1).to(k.dtype)
    k_masked = k * mask
    # Local sample density stays REAL -- it is a geometric property of the
    # trajectory, and a complex density would rotate the data it compensates.
    density = torch.einsum("bhdn,bhd->bhn", k_masked, k_masked.sum(dim=-1))
    weights = (mask.squeeze(2) / density.clamp_min(eps)).unsqueeze(2)
    if torch.is_complex(v):
        q, k_masked, weights = q.to(v.dtype), k_masked.to(v.dtype), weights.to(v.dtype)
    kv = torch.einsum("bhdn,bhen->bhde", k_masked * weights, v)
    return torch.einsum("bhdn,bhde->bhen", q, kv)


class ObservedKeyLinearAttention(nn.Module):
    """O(N) linear attention whose keys and values are restricted to observed bins.

    ``phi(x) = elu(x) + 1`` (Katharopoulos et al., 2020) is strictly positive, so
    the per-query denominator ``phi(q) . sum_j M_j phi(k_j)`` is positive whenever
    anything is observed and the implied attention rows sum to one. The mask
    enters that SUM rather than an ``[L, L]`` score matrix, so restricting the
    keys costs nothing and no quadratic tensor is ever formed -- the reason this
    is the only masked-attention form that fits 256^2 k-space.
    """

    def __init__(self, channels: int, num_heads: int = 4, eps: float = 1e-6):
        """__init__.

        Args:
            channels (int): Interleaved-real channel count; must divide by
                ``num_heads``.
            num_heads (int): Attention heads.
            eps (float): Denominator floor. Engages only when a crop observes
                nothing, which the central ACS band makes unreachable in
                practice; it keeps that case at 0 rather than NaN.
        """
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels={channels} must divide by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.eps = eps
        self.norm = nn.InstanceNorm2d(channels, affine=True)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        """Attend from every position to the observed positions only.

        Args:
            x (torch.Tensor, shape (B, C, H, W)): Feature map.
            observed (torch.Tensor, shape (B, 1, H, W)): 1 where the bin was
                acquired, 0 in the null space.

        Returns:
            torch.Tensor: ``[B, C, H, W]``, the projected attention output (no
            residual -- the caller owns how it is written back).
        """
        b, c, h, w = x.shape
        shape = (b, self.num_heads, self.head_dim, h * w)
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        out = self.attend(
            F.elu(q.reshape(shape)) + 1.0,
            F.elu(k.reshape(shape)) + 1.0,
            v.reshape(shape),
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
        """Masked linear attention on ``[B, heads, head_dim, N]`` features.

        A separate seam because the restriction is the whole mechanism and ``x``
        feeds q, k and v alike -- only here can a test perturb the keys and values
        without also moving the queries that read them.

        Args:
            q (torch.Tensor): Positive query features.
            k (torch.Tensor): Positive key features; zeroed off ``observed`` here,
                so a caller cannot pass an unmasked key that survives.
            v (torch.Tensor): Values.
            observed (torch.Tensor, shape (B, 1, H, W)): 1 where acquired.

        Returns:
            torch.Tensor: ``[B, heads, head_dim, N]``.
        """
        return masked_linear_attention(q, k, v, observed, eps=self.eps)


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
        """Center-crop an acquisition mask onto a ``[B, 1, height, width]`` feature grid.

        The encoder downsamples with ``KSpaceCrop``, which center-crops k-space, so
        the level-``l`` grid IS the center crop of the full grid and the mask must
        be sliced with the same arithmetic -- never interpolated, and never through
        ``KSpaceCrop`` itself, whose ``/ scale_factor`` would turn a binary mask
        into halves. Channels reduce by ``amin``: a bin counts as observed only
        where every channel observed it, which is the conservative direction under
        ``prior_channel_range`` and costs no host synchronisation.
        """
        if mask.dim() != 4:
            raise ValueError(f"mask must be [B, C, H, W]; got shape {tuple(mask.shape)}")
        full_h, full_w = mask.shape[-2], mask.shape[-1]
        if height > full_h or width > full_w:
            raise ValueError(
                f"feature grid ({height}, {width}) is larger than the mask "
                f"({full_h}, {full_w}); the mask must be the full-resolution one"
            )
        start_h, start_w = (full_h - height) // 2, (full_w - width) // 2
        cropped = mask[..., start_h : start_h + height, start_w : start_w + width]
        return (cropped.amin(dim=1, keepdim=True) > 0).to(mask.dtype)

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
