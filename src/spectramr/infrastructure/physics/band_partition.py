"""Spatial-frequency band partition keyed to what an acquisition actually resolved.

A super-resolution network is asked to produce detail the low-field acquisition
never measured. Nothing in a global PSNR/SSIM number distinguishes detail that
was *recovered* (present in the aliased multi-frame measurements, and unfolded)
from detail that was *fabricated* (a plausible prior, unconstrained by data).
Separating the two needs a frequency axis anchored to the acquisition, not to
the storage grid.

The anchor here is the normalised radial frequency

.. math::

    \\rho(\\mathbf{f}) = \\sqrt{\\sum_i \\left(f_i / f^{c}_i\\right)^2},
    \\qquad f^{c}_i = 1 / (2\\,\\Delta^{\\mathrm{eff}}_i),

where :math:`f_i` is in cycles/mm and :math:`\\Delta^{\\mathrm{eff}}_i` is the
resolution the *scanner* achieved on axis *i*. By construction:

* ``rho <= 1`` — inside the acquisition passband. Measured, up to noise.
* ``rho > 1``  — **super-Nyquist**. Not measured. Any content here is either
  unfolded from inter-frame aliasing or invented.

Normalising per axis makes the boundary exact under anisotropy: the ULF
protocol resolves 1.6 mm in-plane and 5.0 mm through-plane, so the true
passband is an ellipsoid and a scalar cutoff would misclassify whole bands on
the thin axis.

Two ways to state the passband, both explicit:

``voxel_mm`` + ``effective_voxel_mm``
    Real data. The volume is stored on the 3 T grid (0.22-0.49 mm) while the
    64 mT scanner resolved 1.6-1.7 mm, so grid spacing is NOT resolution.
``sr_scale``
    The synthetic decimation path, where frames are ``s``-fold pooled views of
    the HR grid. Then :math:`\\Delta^{\\mathrm{eff}} = s\\,\\Delta^{\\mathrm{grid}}`
    exactly and ``rho = 2 s |f|`` with ``f`` in cycles per HR pixel.
"""

from __future__ import annotations

from itertools import pairwise

import torch

from spectramr.infrastructure.physics.subpixel_registration import (
    centred_fft,
    centred_freqs,
    centred_ifft,
    spatial_dims,
)

__all__ = [
    "acquisition_rho",
    "band_edges",
    "band_masks",
    "band_partition",
    "band_transfer",
    "super_nyquist_band_indices",
]


def acquisition_rho(
    size: tuple[int, ...],
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    sr_scale: int | None = None,
    voxel_mm: tuple[float, ...] | None = None,
    effective_voxel_mm: tuple[float, ...] | None = None,
) -> torch.Tensor:
    """Radial frequency normalised so ``1.0`` is the acquisition Nyquist.

    Args:
        size: Spatial extent, 2-D ``(H, W)`` or 3-D ``(D, H, W)``.
        device: Where to build the map.
        dtype: Real floating dtype of the result.
        sr_scale: Decimation factor, for the synthetic path. Mutually exclusive
            with the millimetre pair.
        voxel_mm: Spacing of the STORED grid, per axis, in mm.
        effective_voxel_mm: Resolution the acquisition achieved, per axis, in mm.

    Returns:
        ``[*size]`` real tensor. ``rho <= 1`` is measured, ``rho > 1`` is
        super-Nyquist.

    Raises:
        ValueError: If neither or both parameterisations are given, if lengths
            disagree with ``size``, if any spacing is non-positive, or if the
            acquisition is claimed to out-resolve the grid it is stored on
            (``effective < grid``), which would make every band sub-Nyquist and
            the probe vacuous.
    """
    ndim = len(size)
    if ndim not in (2, 3):
        raise ValueError(f"size must be 2-D or 3-D, got {size}")

    mm_mode = voxel_mm is not None or effective_voxel_mm is not None
    if mm_mode == (sr_scale is not None):
        raise ValueError(
            "acquisition_rho needs exactly one parameterisation: either "
            "sr_scale (synthetic decimation) or voxel_mm + effective_voxel_mm "
            f"(real data). Got sr_scale={sr_scale}, voxel_mm={voxel_mm}, "
            f"effective_voxel_mm={effective_voxel_mm}."
        )

    if sr_scale is not None:
        if sr_scale < 1:
            raise ValueError(f"sr_scale must be >= 1, got {sr_scale}")
        grid = (1.0,) * ndim
        eff = (float(sr_scale),) * ndim
    else:
        if voxel_mm is None or effective_voxel_mm is None:
            raise ValueError(
                "millimetre mode needs BOTH voxel_mm (the stored grid) and "
                "effective_voxel_mm (what the scanner resolved). They are "
                "different numbers on resampled ULF data and assuming they "
                "are equal is what makes a marker invisible."
            )
        grid = tuple(float(v) for v in voxel_mm)
        eff = tuple(float(v) for v in effective_voxel_mm)
        if len(grid) != ndim or len(eff) != ndim:
            raise ValueError(
                f"voxel_mm {grid} and effective_voxel_mm {eff} must both have "
                f"{ndim} entries to match size {size}"
            )
        if any(v <= 0 for v in grid + eff):
            raise ValueError(f"spacings must be positive, got {grid}, {eff}")
        if any(e < g for e, g in zip(eff, grid, strict=True)):
            raise ValueError(
                f"effective_voxel_mm {eff} is finer than the stored grid {grid} "
                "on at least one axis. Then the whole grid lies inside the "
                "acquisition passband, there is no super-Nyquist band, and the "
                "probe measures nothing. Check which of the two is the native "
                "scanner resolution."
            )

    # f_i in cycles per grid pixel -> cycles/mm is f_i / grid_i; the axis
    # cutoff is 1 / (2 * eff_i) cycles/mm, so the normalised coordinate is
    # (f_i / grid_i) * 2 * eff_i.
    rho_sq = torch.zeros(size, device=device, dtype=dtype)
    for axis, n in enumerate(size):
        shape = [1] * ndim
        shape[axis] = n
        f = centred_freqs(n, torch.device(device), dtype).view(shape)
        rho_sq = rho_sq + (f * (2.0 * eff[axis] / grid[axis])) ** 2
    return rho_sq.sqrt()


def band_edges(n_sub: int = 2, n_super: int = 2, rho_max: float = 2.0) -> tuple[float, ...]:
    """Band boundaries in ``rho``, with an edge landing exactly on 1.0.

    The acquisition Nyquist is always a boundary, so no band ever straddles it
    and every band is unambiguously measured or unmeasured.

    Args:
        n_sub: Bands spanning ``[0, 1]``.
        n_super: Bands spanning ``(1, rho_max]``.
        rho_max: Outer edge, in units of the acquisition Nyquist.

    Returns:
        ``n_sub + n_super + 1`` monotone edges starting at 0.0.
    """
    if n_sub < 1 or n_super < 1:
        raise ValueError(
            f"need at least one band each side of Nyquist, got n_sub={n_sub}, "
            f"n_super={n_super}. The comparison between measured and "
            "unmeasured bands is the whole point of the partition."
        )
    if rho_max <= 1.0:
        raise ValueError(f"rho_max must exceed 1.0 to contain a super-Nyquist band, got {rho_max}")
    sub = [i / n_sub for i in range(n_sub + 1)]
    span = rho_max - 1.0
    sup = [1.0 + span * (i + 1) / n_super for i in range(n_super)]
    return tuple(sub + sup)


def super_nyquist_band_indices(edges: tuple[float, ...]) -> tuple[int, ...]:
    """Indices of the bands lying entirely above the acquisition Nyquist."""
    return tuple(i for i in range(len(edges) - 1) if edges[i] >= 1.0)


def band_masks(rho: torch.Tensor, edges: tuple[float, ...], *, min_bins: int = 16) -> torch.Tensor:
    """Half-open annular masks ``[edges[l], edges[l+1])`` over ``rho``.

    Args:
        rho: Normalised radial frequency from :func:`acquisition_rho`.
        edges: Monotone band boundaries.
        min_bins: Smallest population a band may have. A band with a handful of
            bins yields a correlation dominated by sampling noise, which reads
            as a real transfer measurement; raising is the honest outcome.

    Returns:
        ``[L, *rho.shape]`` float masks, one per band.

    Raises:
        ValueError: If ``edges`` is not strictly increasing, or a band holds
            fewer than ``min_bins`` frequency bins on this grid.
    """
    if len(edges) < 2:
        raise ValueError(f"need at least 2 edges to form a band, got {edges}")
    if any(b <= a for a, b in pairwise(edges)):
        raise ValueError(f"edges must be strictly increasing, got {edges}")

    masks = []
    for lo, hi in pairwise(edges):
        m = ((rho >= lo) & (rho < hi)).to(rho.dtype)
        count = int(m.sum().item())
        if count < min_bins:
            raise ValueError(
                f"band [{lo:.3f}, {hi:.3f}) holds {count} frequency bins on a "
                f"{tuple(rho.shape)} grid (minimum {min_bins}). The grid does "
                f"not reach rho={hi:.3f}: its maximum is {float(rho.max()):.3f}. "
                "Lower rho_max, or use a finer grid."
            )
        masks.append(m)
    return torch.stack(masks, dim=0)


def band_partition(x: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Split ``x`` into band-limited components.

    Args:
        x: ``[B, C, *spatial]``, real or complex.
        masks: ``[L, *spatial]`` from :func:`band_masks`.

    Returns:
        ``[B, C, L, *spatial]``. Real input returns the real part, and the
        components then sum back to ``x`` wherever the masks partition the
        whole grid. Differentiable in ``x``.
    """
    spatial_dims(x)  # rank guard: 2-D or 3-D spatial
    if masks.shape[1:] != x.shape[2:]:
        raise ValueError(
            f"masks {tuple(masks.shape)} do not match the spatial extent of x {tuple(x.shape)}"
        )
    was_real = not torch.is_complex(x)
    k = centred_fft(x).unsqueeze(2)  # [B, C, 1, *spatial]
    bands = centred_ifft((k * masks.to(k.dtype)).reshape(x.shape[0], -1, *x.shape[2:])).reshape(
        x.shape[0], x.shape[1], masks.shape[0], *x.shape[2:]
    )
    return bands.real if was_real else bands


def band_transfer(
    prediction: torch.Tensor,
    target: torch.Tensor,
    masks: torch.Tensor,
    *,
    support: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Per-band transfer gain: cosine similarity of the band components.

    For a band-pass component the mean is already zero, so the cosine
    similarity IS the Pearson correlation. It is scale-free, which matters: a
    network that reproduces a band's structure at half amplitude scores 1.0
    here and is penalised by the task loss instead, keeping the two claims
    separable.

    Args:
        prediction: ``[B, C, *spatial]`` model output.
        target: ``[B, C, *spatial]`` known reference.
        masks: ``[L, *spatial]`` band masks.
        support: Optional ``[B, 1, *spatial]`` weight restricting the
            comparison, e.g. the fiducial's own footprint. Broadcast over
            channels and bands.
        eps: Floor on the norms.

    Returns:
        ``[B, L]`` in ``[-1, 1]``, averaged over channels. Differentiable in
        ``prediction``.
    """
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction {tuple(prediction.shape)} and target {tuple(target.shape)} must match"
        )
    p = band_partition(prediction, masks)
    t = band_partition(target, masks)
    if support is not None:
        w = support.unsqueeze(2)  # [B, 1, 1, *spatial]
        p, t = p * w, t * w
    reduce = tuple(range(3, p.ndim))
    num = (p * t).sum(dim=reduce)
    den = p.pow(2).sum(dim=reduce).sqrt() * t.pow(2).sum(dim=reduce).sqrt()
    return (num / den.clamp(min=eps)).mean(dim=1)
