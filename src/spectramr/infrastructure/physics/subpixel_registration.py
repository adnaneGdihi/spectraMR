"""Differentiable sub-pixel translation and marker-anchored shift recovery.

Two primitives for multi-frame super-resolution:

``fourier_shift``
    Exact band-limited translation. A rigid shift is a pure linear phase ramp in
    k-space, so this is the physically correct sub-voxel dither operator. The
    ``grid_sample`` alternative low-pass filters by an amount that depends on the
    fractional part of the shift, destroying the high-frequency content that
    multi-frame SR exists to recover.

``estimate_subpixel_shifts``
    Wrap-free sub-pixel registration from the cross-power spectrum. Rather than
    unwrapping the phase or interpolating a correlation peak, it reads the shift
    off the phase difference between *adjacent* frequency bins, which is constant
    and small for any shift below the Nyquist limit.

Both operate on the centred FFT convention of ``fft_ops`` (DC at ``N // 2``) and
are differentiable end to end, so a network can be supervised through them.

References
----------
* B. S. Reddy, B. N. Chatterji, "An FFT-based technique for translation,
  rotation and scale-invariant image registration," IEEE TIP 5(8), 1996.
* H. Foroosh, J. B. Zerubia, M. Berthod, "Extension of phase correlation to
  subpixel registration," IEEE TIP 11(3), 2002.
"""

from __future__ import annotations

import math

import torch

from spectramr.infrastructure.physics.fft_ops import centred_freqs, fftnc, ifftnc, spatial_dims

__all__ = [
    "centred_fft",
    "centred_freqs",
    "centred_ifft",
    "estimate_subpixel_shifts",
    "fourier_shift",
    "spatial_dims",
]


# ``centred_freqs``, ``spatial_dims`` / ``centred_fft`` / ``centred_ifft`` are re-exports, not
# definitions: their owner is ``fft_ops``, the FFT SSOT (non-negotiable 2/17).
# They lived here while ``fft_ops`` exposed no CENTRED N-D pair; #1350 moved the
# implementation there verbatim, so this module no longer calls ``torch.fft``
# directly and its ``INTENTIONAL_EXEMPT`` entry was dropped with the move.
centred_fft = fftnc
centred_ifft = ifftnc


def _as_pixels(shifts: torch.Tensor, voxel_mm: tuple[float, ...] | None, ndim: int) -> torch.Tensor:
    """Convert millimetre shifts to pixels when a voxel size is supplied.

    ULF and HF volumes are strongly anisotropic (1.6 x 1.6 x 5.0 mm against
    0.49 x 0.49 x 1.0 mm), so a shift expressed in pixels means a different
    physical displacement on every axis and is not comparable across contrasts.
    Millimetres are the only unit in which a registration error is a single
    interpretable number.
    """
    if voxel_mm is None:
        return shifts
    vox = tuple(float(v) for v in voxel_mm)
    if len(vox) != ndim:
        raise ValueError(f"voxel_mm has {len(vox)} entries but data is {ndim}-D")
    if any(v <= 0 for v in vox):
        raise ValueError(f"voxel_mm must be positive, got {vox}")
    scale = torch.tensor(vox, device=shifts.device, dtype=shifts.dtype)
    return shifts / scale


def fourier_shift(
    x: torch.Tensor,
    shifts: torch.Tensor,
    *,
    voxel_mm: tuple[float, ...] | None = None,
) -> torch.Tensor:
    """Translate ``x`` by ``shifts`` via a k-space phase ramp.

    Args:
        x: ``[B, C, H, W]`` or ``[B, C, D, H, W]``, real or complex.
        shifts: ``[B, C, ndim]`` ordered like the spatial axes, so ``(dy, dx)``
            in 2-D and ``(dz, dy, dx)`` in 3-D. Positive moves content toward
            increasing index. Per-sample and per-channel, so a frame stack takes
            a different shift on every frame with no loop.
        voxel_mm: When given, ``shifts`` is read in MILLIMETRES and converted
            per axis. Required for anisotropic data, where one pixel means a
            different physical displacement on every axis.

    Returns:
        Same shape and dtype-family as ``x``. Real input returns the real part.
    """
    dims = spatial_dims(x)
    ndim = len(dims)
    if shifts.shape != (x.shape[0], x.shape[1], ndim):
        raise ValueError(
            f"shifts must be [B, C, {ndim}] matching x={tuple(x.shape)}, got {tuple(shifts.shape)}"
        )

    was_real = not torch.is_complex(x)
    shifts_px = _as_pixels(shifts.to(torch.float32), voxel_mm, ndim)

    # phase = -2*pi * sum_i f_i * d_i, each term broadcast along its own axis.
    phase = torch.zeros((*shifts.shape[:2], *x.shape[2:]), device=x.device, dtype=torch.float32)
    for axis, dim in enumerate(dims):
        n = x.shape[dim]
        shape = [1, 1] + [1] * ndim
        shape[2 + axis] = n
        freq = centred_freqs(n, x.device, torch.float32).view(shape)
        d = shifts_px[..., axis].view(*shifts.shape[:2], *([1] * ndim))
        phase = phase + freq * d
    phase = -2.0 * math.pi * phase

    # Not `torch.polar(torch.ones(()), phase)`: the scalar magnitude would be
    # allocated on CPU and cross-device against a CUDA phase.
    ramp = torch.complex(torch.cos(phase), torch.sin(phase))
    out = centred_ifft(centred_fft(x) * ramp)
    # Real input: the unpaired Nyquist bin of an even-length DFT has no conjugate
    # partner, so a fractional shift leaks a little signal into the imaginary
    # part that `.real` discards (~0.2% of dynamic range on band-limited MRI
    # content, and exactly zero for integer shifts). Still far below the
    # fraction-dependent blur a `grid_sample` resample would introduce.
    return out.real if was_real else out


def estimate_subpixel_shifts(
    reference: torch.Tensor,
    moving: torch.Tensor,
    *,
    eps: float = 1e-8,
    voxel_mm: tuple[float, ...] | None = None,
) -> torch.Tensor:
    """Recover the sub-pixel translation of each ``moving`` frame vs ``reference``.

    Inverts :func:`fourier_shift`: if ``moving[:, k] = fourier_shift(reference,
    d_k)`` then the returned ``[:, k]`` is ``d_k``.

    The cross-power spectrum ``R = F_moving * conj(F_reference)`` of a pure
    translation has unit magnitude and phase ``-2 pi (f_y d_y + f_x d_x)``. That
    phase wraps for shifts beyond one pixel, so we never read it directly:
    the product of adjacent bins ``R[y+1] conj(R[y])`` has phase
    ``-2 pi d_y / H`` for *every* bin, which is far inside ``(-pi, pi]`` for any
    ``|d_y| < H / 2``. Its weighted circular mean is the estimate; weighting by
    the cross-spectral magnitude keeps noise-only bins from voting.

    Args:
        reference: ``[B, 1, *spatial]`` real or complex.
        moving: ``[B, N, *spatial]`` real or complex.
        eps: Floor for the magnitude normalisation.
        voxel_mm: When given, the result is returned in MILLIMETRES rather than
            pixels. On anisotropic data (ULF is 1.6 x 1.6 x 5.0 mm) a pixel is a
            different displacement per axis, so a pixel-valued registration error
            is not a single interpretable number.

    Returns:
        ``[B, N, ndim]`` shifts ordered like the spatial axes, in pixels or in
        millimetres when ``voxel_mm`` is given. Differentiable in both inputs, so
        a shift-supervision loss reaches the frames that produced it.
    """
    dims = spatial_dims(moving)
    ndim = len(dims)
    if reference.ndim != moving.ndim:
        raise ValueError(
            f"reference {tuple(reference.shape)} and moving {tuple(moving.shape)} "
            "must have the same rank"
        )
    if reference.shape[1] != 1:
        raise ValueError(f"reference must have exactly 1 channel, got {reference.shape[1]}")
    if reference.shape[0] != moving.shape[0] or reference.shape[2:] != moving.shape[2:]:
        raise ValueError(
            f"reference {tuple(reference.shape)} and moving {tuple(moving.shape)} "
            "must share batch and spatial dims"
        )

    cross = centred_fft(moving) * centred_fft(reference).conj()
    weight = cross.abs()
    unit = cross / (weight + eps)

    # Adjacent-bin phase difference along each spatial axis, weighted by the
    # joint cross-spectral energy so noise-only bins do not vote.
    components = []
    for dim in dims:
        n = moving.shape[dim]
        hi = torch.narrow(unit, dim, 1, n - 1)
        lo = torch.narrow(unit, dim, 0, n - 1)
        w_hi = torch.narrow(weight, dim, 1, n - 1)
        w_lo = torch.narrow(weight, dim, 0, n - 1)
        acc = (hi * lo.conj() * w_hi * w_lo).sum(dim=dims)
        components.append(-torch.angle(acc) * n / (2.0 * math.pi))

    shifts_px = torch.stack(components, dim=-1)
    if voxel_mm is None:
        return shifts_px
    vox = tuple(float(v) for v in voxel_mm)
    if len(vox) != ndim:
        raise ValueError(f"voxel_mm has {len(vox)} entries but data is {ndim}-D")
    scale = torch.tensor(vox, device=shifts_px.device, dtype=shifts_px.dtype)
    return shifts_px * scale
