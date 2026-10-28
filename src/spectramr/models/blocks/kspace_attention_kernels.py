"""Shared kernels for the k-space attention blocks.

One home for the pieces several attention blocks compute identically, so the
grid, off-grid and wavelet variants cannot drift into computing different things
under one name (non-negotiable 17): support-restricted linear attention, its
phase-encode distance bias, normalisation over the acquired support, mask
alignment across the encoder's k-space crops, and the conjugate-pair-preserving
token subsample of :class:`~spectramr.models.blocks.dual_domain_attention_kan.RadialBandAttention`.

Layout facts these kernels rely on: the acquisition mask selects whole
phase-encode COLUMNS, so the PE axis is ``W`` (dim -1) and readout is ``H``;
every encoder level is a centre crop of k-space with unchanged ``delta k`` and DC
at ``N // 2``, so a column distance means the same physical ``delta k`` at every
level.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import torch
from torch import nn


def _float32_kernel(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """Run a linear-attention kernel in float32 and return the value dtype.

    The key sums grow as ``O(N)`` and pass the fp16 maximum near a 128^2 grid,
    and complex-half matmuls have no kernel, so under fp16 autocast these sums
    are computed with autocast off in float32 / complex64.
    """

    @functools.wraps(fn)
    def wrapper(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *args, **kwargs):
        out_dtype = v.dtype
        with torch.autocast(device_type=q.device.type, enabled=False):
            up = [t.to(torch.complex64) if t.is_complex() else t.float() for t in (q, k, v)]
            out = fn(*up, *args, **kwargs)
        return out.to(out_dtype)

    return wrapper


@_float32_kernel
def masked_linear_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    observed: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Linear attention on ``[B, heads, head_dim, N]``, restricted to a support.

    The mask enters the key SUM rather than an ``[N, N]`` score matrix, which is
    the only form that fits a 200k-sample trajectory or a 256^2 grid.

    Args:
        q: Positive query features.
        k: Positive key features; zeroed off ``observed`` here, so a caller
            cannot pass an unmasked key that survives.
        v: Values, real or complex.
        observed: 1 where acquired; broadcast over the flat ``N`` axis.
        eps: Denominator floor.

    Returns:
        ``[B, heads, head_dim_v, N]``.
    """
    k = k * observed.reshape(observed.shape[0], 1, 1, -1)
    # The denominator stays REAL: it is the row normaliser (and, off-grid, the
    # learned density compensation), so it must not rotate the phase it divides.
    denominator = torch.einsum("bhdn,bhd->bhn", q, k.sum(dim=-1)).unsqueeze(2)
    if torch.is_complex(v):
        q, k = q.to(v.dtype), k.to(v.dtype)
    kv = torch.einsum("bhdn,bhen->bhde", k, v)
    numerator = torch.einsum("bhdn,bhde->bhen", q, kv)
    return numerator / denominator.clamp_min(eps).to(numerator.real.dtype)


@_float32_kernel
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
    density-compensated SUM -- the weight belongs to the sample, not to the grid
    point. Measured with an exact Gaussian on a radial trajectory, the average
    form does not converge at any width while the sum form reaches 1.3x the fixed
    Kaiser-Bessel adjoint. ``c_j = m_j / sum_j' <phi(k_j), phi(k_j')> m_j'`` is
    one Pipe-Menon iteration; a duplicated sample halves each copy's weight.

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
    # Density is a REAL geometric property; a complex one would rotate the data.
    density = torch.einsum("bhdn,bhd->bhn", k_masked, k_masked.sum(dim=-1))
    weights = (mask.squeeze(2) / density.clamp_min(eps)).unsqueeze(2)
    if torch.is_complex(v):
        q, k_masked, weights = q.to(v.dtype), k_masked.to(v.dtype), weights.to(v.dtype)
    kv = torch.einsum("bhdn,bhen->bhde", k_masked * weights, v)
    return torch.einsum("bhdn,bhde->bhen", q, kv)


def pe_distance_kernel(width: int, slope: torch.Tensor) -> torch.Tensor:
    """``[heads, W, W]`` multiplicative bias ``(1 + |u - w|) ** (-slope_h)``.

    The exponential of the additive log-bias ``-slope * log(1 + |d|)``: smooth in
    the column distance, so the same parameters serve every encoder width.
    """
    pos = torch.arange(width, device=slope.device, dtype=slope.dtype)
    dist = (pos[:, None] - pos[None, :]).abs()
    return torch.exp(-slope.reshape(-1, 1, 1) * torch.log1p(dist))


@_float32_kernel
def pe_biased_linear_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    observed: torch.Tensor,
    slope: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """:func:`masked_linear_attention` with keys weighted by PE-column distance.

    The weight of key ``j`` for query ``i`` is
    ``phi(q_i) . phi(k_j) * (1 + |w_i - w_j|) ** (-slope)``, which prefers the
    nearest acquired lines the way GRAPPA does. Because the bias depends only on
    the column, the key sums are first reduced per column and then mixed by the
    ``[W, W]`` kernel -- exact, and O(N d e + W^2 d e) rather than O(N^2).

    Args:
        q, k: ``[B, heads, D, H*W]`` positive features (row-major flat grid).
        v: ``[B, heads, E, H*W]`` values, real or complex.
        observed: ``[B, 1, H, W]``, 1 where acquired.
        slope: ``[heads]`` non-negative decay exponents.
        eps: Denominator floor.
    """
    b, h, d, n = q.shape
    height, width = observed.shape[-2:]
    kernel = pe_distance_kernel(width, slope).to(q.dtype)
    q5 = q.reshape(b, h, d, height, width)
    k5 = (k * observed.reshape(observed.shape[0], 1, 1, -1)).reshape(b, h, d, height, width)
    v5 = v.reshape(b, h, -1, height, width)
    k_mix = torch.einsum("huw,bhdw->bhdu", kernel, k5.sum(dim=3))
    denominator = torch.einsum("bhdru,bhdu->bhru", q5, k_mix).unsqueeze(2)
    if torch.is_complex(v):
        q5, k5, kernel = q5.to(v.dtype), k5.to(v.dtype), kernel.to(v.dtype)
    kv = torch.einsum("huw,bhwde->bhude", kernel, torch.einsum("bhdrw,bherw->bhwde", k5, v5))
    numerator = torch.einsum("bhdru,bhude->bheru", q5, kv)
    out = numerator / denominator.clamp_min(eps).to(numerator.real.dtype)
    return out.reshape(b, h, -1, n)


def masked_complex_rms(
    x: torch.Tensor, observed: torch.Tensor, gain: torch.Tensor, eps: float
) -> torch.Tensor:
    """``ComplexRMSNorm`` whose power mean runs over the acquired bins only.

    Zero-filled bins would otherwise pull the scale down by the acceleration
    factor. One real scalar per sample keeps the normaliser phase-equivariant.
    """
    real, imag = x[:, 0::2], x[:, 1::2]
    power = ((real * real + imag * imag) * observed).sum(dim=(1, 2, 3), keepdim=True)
    count = observed.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1.0) * real.shape[1]
    scale = torch.rsqrt(power / count + eps) * gain.view(1, -1, 1, 1)
    return torch.stack([real * scale, imag * scale], dim=2).flatten(1, 2)


class MaskedInstanceNorm2d(nn.Module):
    """Affine instance norm whose statistics come from the acquired bins only.

    ``InstanceNorm2d`` over a zero-filled spectrum moves its mean and variance
    with the acceleration factor; restricting the moments to ``observed`` makes
    them a property of the measured data. Parameter names match
    ``nn.InstanceNorm2d(affine=True)``; an all-ones support reproduces it.
    """

    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        """Normalise ``[B, C, H, W]`` with moments over ``observed`` ``[B, 1, H, W]``."""
        count = observed.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
        mean = (x * observed).sum(dim=(-2, -1), keepdim=True) / count
        var = ((x - mean).square() * observed).sum(dim=(-2, -1), keepdim=True) / count
        y = (x - mean) * torch.rsqrt(var + self.eps)
        return y * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


def _crop_offset(full: int, size: int) -> int:
    """Start of the ``size`` window that the encoder's ``KSpaceCrop`` chain leaves.

    Each level crops ``n -> n // 2`` at ``(n - n // 2) // 2``; once an intermediate
    size is odd, the accumulated offset differs by a column from a one-step
    centre crop (W = 372 at 46 columns). A size the halving chain does not reach
    was cropped in one step, so it takes the one-step offset.
    """
    offset, n = 0, full
    while n > size:
        half = n // 2
        if half < size:
            return (full - size) // 2
        offset += (n - half) // 2
        n = half
    return offset if n == size else (full - size) // 2


def align_mask_to_grid(mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Centre-crop an acquisition mask onto a ``[B, 1, height, width]`` feature grid.

    The encoder downsamples with ``KSpaceCrop``, so the level grid IS the centre
    crop of the full grid and the mask is sliced with the same arithmetic --
    never interpolated, and never through ``KSpaceCrop`` itself, whose
    ``/ scale_factor`` would turn a binary mask into halves. Channels reduce by
    ``amin``: a bin is observed only where every channel observed it.
    """
    if mask.dim() != 4:
        raise ValueError(f"mask must be [B, C, H, W]; got shape {tuple(mask.shape)}")
    if height > mask.shape[-2] or width > mask.shape[-1]:
        raise ValueError(
            f"feature grid ({height}, {width}) is larger than the mask "
            f"({mask.shape[-2]}, {mask.shape[-1]}); the mask must be the full-resolution one"
        )
    cropped = crop_to_level_grid(mask, height, width)
    return (cropped.amin(dim=1, keepdim=True) > 0).to(mask.dtype)


def crop_to_level_grid(plane: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Slice a full-resolution ``[..., H, W]`` plane onto an encoder level's grid.

    The window is the one the ``KSpaceCrop`` chain leaves, and values are not
    rescaled: a mask or a coordinate keeps its full-grid meaning at every level.
    """
    full_h, full_w = plane.shape[-2], plane.shape[-1]
    if height > full_h or width > full_w:
        raise ValueError(
            f"feature grid ({height}, {width}) is larger than the plane "
            f"({full_h}, {full_w}); the plane must be the full-resolution one"
        )
    start_h, start_w = _crop_offset(full_h, height), _crop_offset(full_w, width)
    return plane[..., start_h : start_h + height, start_w : start_w + width]


def conjugate_partner(flat_idx: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Flat index of the ``k -> -k`` partner about DC at ``(H // 2, W // 2)``."""
    row, col = flat_idx // width, flat_idx % width
    return ((2 * (height // 2) - row) % height) * width + (2 * (width // 2) - col) % width


@torch.no_grad()
def stratified_conjugate_subsample(
    flat_idx: torch.Tensor, height: int, width: int, cap: int, seed: int
) -> torch.Tensor:
    """At most ``cap`` of ``flat_idx``, spread over every row, conjugate pairs kept.

    Pairs ``{i, partner(i)}`` are the sampling unit, so a selected bin always
    arrives with its Hermitian partner. Within each row the pairs are drawn in a
    seeded random order, and the global order sorts by *fractional rank within
    the row*, so every row that holds a pair is represented before any row gets
    a second one (needs ``cap >= 2 * rows``) and columns are drawn uniformly
    rather than on a stride. Deterministic for a given ``seed``; runs once per
    shape on the host. Returns sorted flat indices on ``flat_idx``'s device.
    """
    if flat_idx.numel() <= cap:
        return flat_idx
    device = flat_idx.device
    idx = flat_idx.cpu()
    reps = torch.unique(torch.minimum(idx, conjugate_partner(idx, height, width)))
    budget = max(1, cap // 2)
    if reps.numel() > budget:
        gen = torch.Generator().manual_seed(seed)
        shuffled = reps[torch.randperm(reps.numel(), generator=gen)]
        rows = shuffled // width
        order = torch.argsort(rows, stable=True)
        counts = torch.bincount(rows, minlength=height)
        starts = torch.cumsum(counts, 0) - counts
        rank = torch.empty_like(rows)
        rank[order] = torch.arange(rows.numel()) - starts[rows[order]]
        key = rank.double() / counts[rows].double()
        reps = shuffled[torch.argsort(key, stable=True)[:budget]]
    chosen = torch.cat([reps, conjugate_partner(reps, height, width)])
    return torch.unique(chosen).to(device)
