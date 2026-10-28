"""Native multislice and 3-D off-resonance: B0 (D10), B0 distortion (D10b), susceptibility (D17).

One 3-D field per subject on its ``(D, H, W)`` grid (see :func:`.fields_maps.field_coords`);
multislice slices are samples of it. Multislice adds the through-slice dephasing of a field
that changes across the slice thickness (Ruling 27): the signal of a slice whose phase varies
linearly by ``dphi`` across it is scaled by ``|sinc(dphi / 2 pi)|`` (normalised sinc).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from spectramr.infrastructure.physics.degradations._common import (
    _ensure_complex,
    _generator,
    readout_offset_px,
)
from spectramr.infrastructure.physics.degradations.blocks.volume_fields import (
    upsampled_noise_field,
)
from spectramr.infrastructure.physics.degradations.blocks.volume_warp import warp_volume
from spectramr.infrastructure.physics.degradations.geometry import (
    PE_DIM,
    RO_DIM,
    SLICE_DIM,
    AcquisitionGeometry,
    AcquisitionMode,
)
from spectramr.infrastructure.physics.degradations.volume.fields_maps import (
    check_volume,
    field_coords,
)
from spectramr.infrastructure.physics.dipole import dipole_kernel_centred
from spectramr.infrastructure.physics.fft_ops import fftnc, ifftnc
from spectramr.infrastructure.physics.per_sample_rng import per_sample_randn

__all__ = [
    "b0_geometric_volume",
    "b0_offset_hz",
    "b0_volume",
    "susceptibility_volume",
    "through_slice_factor",
]

_DIMS3 = (SLICE_DIM, PE_DIM, RO_DIM)
_B0_DIM = {"slice": SLICE_DIM, "pe": PE_DIM, "ro": RO_DIM}
# Per-sample draws of the perturbation grid come from their own stream, so the polynomial
# coefficients and the grid stay batch-invariant (per_sample_rng draws sample by sample).
_NOISE_STREAM = 1_000_003


def _is_multislice(geometry: AcquisitionGeometry) -> bool:
    return geometry.mode is AcquisitionMode.MULTISLICE_2D


def through_slice_factor(phase: Tensor, geometry: AcquisitionGeometry) -> Tensor:
    """``|sinc(dphi_thk / 2 pi)|`` for a ``[B, 1, D, H, W]`` phase (rad) sampled per slice.

    ``dphi_thk`` is the per-slice central difference of ``phase`` along D, scaled by
    ``slice_thickness_mm / voxel_mm[0]`` (1, contiguous slices, when either is unset).
    Beyond ``dphi_thk > 2 pi`` the sinc side lobes make the factor non-monotone (physical).
    """
    if phase.shape[SLICE_DIM] < 2:
        return torch.ones_like(phase)
    ratio = 1.0
    if geometry.slice_thickness_mm is not None and geometry.voxel_mm is not None:
        ratio = geometry.slice_thickness_mm / geometry.voxel_mm[0]
    dphi = torch.gradient(phase, dim=SLICE_DIM)[0] * ratio
    return torch.sinc(dphi / (2.0 * math.pi)).abs()


def _forward_diff_sq(field: Tensor, dims: tuple[int, ...]) -> Tensor:
    """Sum of squared forward differences along ``dims``, zero at each far edge (as in 2-D)."""
    total = torch.zeros_like(field)
    for d in dims:
        total = total + torch.diff(field, dim=d, append=field.narrow(d, -1, 1)).pow(2)
    return total


def b0_volume(
    x: Tensor,
    theta: float,
    seed: int,
    geometry: AcquisitionGeometry,
    strength: float | None = None,
) -> Tensor:
    """D10: phase ``exp(i pi f)`` of a random quadratic field plus gradient dropout.

    ``f`` is the 2-D polynomial extended with the z terms, coefficients ``N(0, strength^2)``.
    The dropout ``exp(-4 strength |grad f|^2 / max)`` uses the in-plane gradient in multislice
    (plus the through-slice factor) and the 3-D gradient in a volume.
    """
    check_volume(x, geometry, "b0")
    s = float(theta) if strength is None else float(strength)
    if s == 0.0:
        return x
    x = _ensure_complex(x)
    b = x.shape[0]
    z, y, xx = field_coords(tuple(x.shape[-3:]), geometry.voxel_mm, x.device)
    terms = (xx * xx, y * y, z * z, xx * y, xx * z, y * z, xx, y, z, torch.ones_like(z))
    coeffs = per_sample_randn(b, len(terms), generator=_generator(x.device, seed), device=x.device)
    coeffs = coeffs * s
    field = torch.zeros(b, 1, *x.shape[-3:], device=x.device)
    for t, term in enumerate(terms):
        field = field + coeffs[:, t].view(b, 1, 1, 1, 1) * term
    phase = math.pi * field
    grad_sq = _forward_diff_sq(field, _DIMS3[1:] if _is_multislice(geometry) else _DIMS3)
    grad_norm = grad_sq / grad_sq.amax(dim=_DIMS3, keepdim=True).clamp(min=1e-12)
    out = x * torch.exp(1j * phase) * torch.exp(-4.0 * s * grad_norm)
    if _is_multislice(geometry):
        out = out * through_slice_factor(phase, geometry)
    return out


def b0_offset_hz(
    batch: int,
    shape: tuple[int, int, int],
    strength: float,
    seed: int,
    voxel_mm: tuple[float, float, float] | None,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """The D10b off-resonance map in Hz, ``[B, 1, D, H, W]``, peak ``|dB0| <= 600 * strength``.

    The 2-D Halbach polynomial (Koolstra) with the z terms added at the same weights, scaled
    to 80 % of the peak, plus a low-frequency perturbation at 20 %.
    """
    z, y, x = field_coords(shape, voxel_mm, device, dtype)
    terms = (
        (x, 0.1), (y, 0.1), (z, 0.1),
        (x * x, 1.0), (y * y, 1.0), (z * z, 1.0),
        (x * y, 0.3), (x * z, 0.3), (y * z, 0.3),
    )  # fmt: skip
    coeffs = per_sample_randn(
        batch, len(terms), generator=_generator(device, seed), device=device, dtype=dtype
    )
    field = torch.zeros(batch, 1, *shape, device=device, dtype=dtype)
    for t, (term, weight) in enumerate(terms):
        field = field + coeffs[:, t].view(batch, 1, 1, 1, 1) * (weight * term)
    field = field - field.mean(dim=_DIMS3, keepdim=True)
    field = field / field.abs().amax(dim=_DIMS3, keepdim=True).clamp(min=1e-8)
    noise = upsampled_noise_field(
        batch,
        shape,
        coarse=tuple(max(2, n // 16) for n in shape),
        generator=_generator(device, seed + _NOISE_STREAM),
        device=device,
        dtype=dtype,
    )
    peak_hz = 600.0 * float(strength)
    return field * (0.8 * peak_hz) + noise * (0.2 * peak_hz)


def b0_geometric_volume(
    x: Tensor,
    theta: float,
    seed: int,
    geometry: AcquisitionGeometry,
    b0_strength: float | None = None,
    dwell_time: float = 5e-5,
    sequence: str = "spin_warp",
) -> Tensor:
    """D10b: spin-warp voxel shift ``dB0 / BW_px`` along readout only, ``BW_px = 1 / (W dwell)``."""
    check_volume(x, geometry, "b0_geometric_distortion")
    if sequence != "spin_warp":
        raise ValueError(
            f"b0_geometric_distortion on a volume models the spin-warp shift only; got {sequence!r}"
        )
    if dwell_time <= 0:
        raise ValueError(f"dwell_time must be > 0 s; got {dwell_time}")
    s = float(theta) if b0_strength is None else float(b0_strength)
    if s == 0.0:
        return x
    x = _ensure_complex(x)
    b, _, d, h, w = x.shape
    rdt = x.real.dtype
    hz = b0_offset_hz(b, (d, h, w), s, seed, geometry.voxel_mm, device=x.device, dtype=rdt)
    dx = readout_offset_px(hz, 1.0 / (w * dwell_time))
    zero = torch.zeros_like(dx)
    return warp_volume(x, torch.cat([zero, zero, dx], dim=1), padding_mode="border")


def susceptibility_volume(
    x: Tensor,
    theta: float,
    seed: int,
    geometry: AcquisitionGeometry,
    max_dropout_pct: float = 0.5,
) -> Tensor:
    """D17: dipole field of a smooth 3-D chi, ``B0`` along ``geometry.b0_axis``.

    ``dB0 = ifftnc(fftnc(chi) D(k))`` (standard sign ``1/3 - k_b0^2/k^2``), unit peak per
    sample; phase ``exp(2 pi i theta dB0)`` and dropout ``exp(-8 theta max_dropout dB0^2)`` as
    in 2-D, plus the multislice through-slice factor. The 2-D kernel keeps its own
    ``k_b0^2/k^2 - 1/3`` sign.
    """
    check_volume(x, geometry, "susceptibility")
    t = float(theta)
    if t <= 0.0:
        return x
    x = _ensure_complex(x)
    b = x.shape[0]
    shape = tuple(x.shape[-3:])
    rdt = x.real.dtype
    chi = upsampled_noise_field(
        b,
        shape,
        coarse=tuple(max(2, n // 16 + 1) for n in shape),
        generator=_generator(x.device, seed),
        device=x.device,
        dtype=rdt,
    )
    kernel = dipole_kernel_centred(
        shape, _B0_DIM[geometry.b0_axis], voxel_mm=geometry.voxel_mm, device=x.device, dtype=rdt
    )
    field = ifftnc(fftnc(torch.complex(chi, torch.zeros_like(chi)), _DIMS3) * kernel, _DIMS3)
    # A real kernel even in k on a real chi is Hermitian: the imaginary part is rounding only.
    peak = field.real.abs().amax()
    torch._assert_async(
        field.imag.abs().amax() <= 1e-4 * peak + 1e-12,
        "susceptibility: dipole field has a non-negligible imaginary part",
    )
    db0 = field.real / field.real.abs().amax(dim=_DIMS3, keepdim=True).clamp(min=1e-8)
    phase = (2.0 * math.pi * t) * db0
    keep = torch.exp(-8.0 * t * max_dropout_pct * db0.pow(2)).clamp(0.0, 1.0)
    out = x * torch.exp(1j * phase) * keep
    if _is_multislice(geometry):
        out = out * through_slice_factor(phase, geometry)
    return out
