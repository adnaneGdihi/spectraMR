"""metal_artifact: a metal sphere's dipole field as signal void, pile-up and GRE dephasing.

A sphere of radius ``a`` and susceptibility difference ``dchi`` in ``B0`` along ``w`` perturbs the
field outside it by the dipole ``dB = dchi B0 a^3 (3 cos^2 th - 1) / (3 r^3)`` (Schenck, Med Phys
23:815, 1996). Three effects follow: (1) no signal inside the sphere; (2) spin-warp readout
displacement ``df / BW_px`` with ``df = gamma dB``, applied by forward mapping so sources that
converge pile up (bright edge) and a stretch leaves a void; (3) for gradient-echo, intravoxel
dephasing ``|sinc(gamma d(dB)/dz thk TE / 2)|`` across the slice (Haacke et al., MRI: Physical
Principles and Sequence Design, 1999, ch. 25).

The sphere sits at a seeded 3-D position (a voxel of the central half plus a through-plane offset
z0 in [-a, a]) in every mode, so each slice has its own dipole and gradient. A 2-D slice is a
one-slice multislice stack, and both are slice-selective: the signal of a source is assigned to the
slice displaced by ``-thk df / BW_rf`` (the readout polarity; the potato-chip distortion that SEMAC
corrects, Lu et al., MRM 62:66, 2009), and what lands beyond the stack is lost. VOLUME_3D has no
slice selection. Readout displacements use the b0_geometric_distortion sign: content moves by
``-df/BW_px``. Signal pushed out of the FOV is dropped (the receiver's anti-alias filter).
"""

# ruff: noqa: N803 -- b0_T is the scanner profile field name
from __future__ import annotations

import math

import torch
from torch import Tensor

from spectramr.infrastructure.physics.degradations._common import _generator, readout_offset_px
from spectramr.infrastructure.physics.degradations.catalogue.motion_timing import (
    AXIS_INDEX,
    check_input,
    run_as_volume,
)
from spectramr.infrastructure.physics.degradations.conditions import b0_shift_sequence
from spectramr.infrastructure.physics.degradations.geometry import (
    AcquisitionGeometry,
    AcquisitionMode,
    voxel_mm_or_unit,
)
from spectramr.infrastructure.physics.digital_twin_simulator import chemical_shift_pixels
from spectramr.infrastructure.physics.per_sample_rng import per_sample_rand

__all__ = [
    "DELTA_CHI_PPM",
    "GRADIENT_ECHO_OF",
    "MAX_RADIUS_MM",
    "RF_BANDWIDTH_HZ",
    "degrade_metal_artifact",
    "dipole_offset_hz",
    "forward_map_readout",
    "metal_artifact_volume",
    "metal_condition",
    "slice_select_shift_mm",
    "sphere_centre_index",
    "sphere_z0_mm",
]

#: Sphere radius at theta = 1.
MAX_RADIUS_MM = 5.0
#: Titanium vs soft tissue is ~180 ppm (Schenck 1996, Table I); 200 ppm is the default.
DELTA_CHI_PPM = 200.0
#: Excitation bandwidth of the slice-selective pulse (multislice only); a typical 1 kHz.
RF_BANDWIDTH_HZ = 1000.0


def sphere_centre_index(
    batch: int, shape: tuple[int, int, int], seed: int, device: torch.device | None = None
) -> Tensor:
    """``[B, 3]`` voxel indices of the sphere centre (z, y, x): the central half of each axis."""
    device = torch.device("cpu") if device is None else device
    u = per_sample_rand(batch, 3, generator=_generator(device, seed), device=device)
    cols = []
    for a, n in enumerate(shape):
        quarter = n // 4
        cols.append((quarter + torch.floor(u[:, a] * (n - 2 * quarter))).clamp(max=n - quarter - 1))
    return torch.stack(cols, dim=1)


def sphere_z0_mm(batch: int, a_mm: float, seed: int, device: torch.device | None = None) -> Tensor:
    """``[B]`` through-plane offset of the sphere centre from its voxel, uniform in [-a, a]."""
    device = torch.device("cpu") if device is None else device
    u = per_sample_rand(batch, 1, generator=_generator(device, seed + 1), device=device)
    return (2.0 * u[:, 0] - 1.0) * a_mm


def dipole_offset_hz(
    rel_mm: tuple[Tensor, Tensor, Tensor],
    radius_mm: float,
    *,
    b0_axis: int,
    b0_T: float,
    delta_chi_ppm: float,
) -> tuple[Tensor, Tensor]:
    """Off-resonance ``df`` (Hz) and its slice-axis gradient (Hz/mm) at offsets ``rel_mm`` (z, y, x).

    ``df = gamma/2pi B0 dchi a^3 (3 w^2 / r^2 - 1) / (3 r^3)`` with ``w`` the offset along
    ``b0_axis`` and ``r`` clamped to ``a`` (the interior is void, never read). gamma/2pi B0 dchi
    comes from ``chemical_shift_pixels`` at 1 Hz/px, the one owner of gamma.
    """
    scale = chemical_shift_pixels(b0_T, 1.0, delta_chi_ppm) * radius_mm**3 / 3.0
    z = rel_mm[0]
    w = rel_mm[b0_axis]
    r2 = (rel_mm[0] ** 2 + rel_mm[1] ** 2 + rel_mm[2] ** 2).clamp_min(radius_mm**2)
    r = r2.sqrt()
    hz = scale * (3.0 * w**2 / r2 - 1.0) / r**3
    # d/dz of 3 w^2 / r^5 - 1 / r^3 (w = z when B0 is along the slice axis).
    dw = 6.0 * w / r**5 if b0_axis == 0 else torch.zeros_like(r)
    grad = scale * (dw - 15.0 * w**2 * z / r**7 + 3.0 * z / r**5)
    return hz, grad


def slice_select_shift_mm(hz: Tensor, thickness_mm: float, rf_bw_hz: float) -> Tensor:
    """Slice the signal of a source is assigned to: displaced by ``-thk df / BW_rf`` (Lu 2009).

    Same polarity as the readout shift, ``-readout_offset_px``: the excited locus moves by
    ``+thk df / BW_rf``.
    """
    return -thickness_mm * hz / rf_bw_hz


def forward_map_readout(x: Tensor, shift_px: Tensor, slice_shift: Tensor | None = None) -> Tensor:
    """Move each source by ``shift_px`` along readout (and ``slice_shift`` along D); sources add.

    Bilinear splatting conserves each source's signal, so the density of the result is the source
    density over the Jacobian: pile-up where sources converge, a void where they spread. Targets
    outside the FOV are dropped. On CUDA ``scatter_add_`` is not bit-deterministic.
    """
    d, w = x.shape[-3], x.shape[-1]
    pos = torch.arange(w, device=x.device, dtype=shift_px.dtype) + shift_px
    zpos = torch.arange(d, device=x.device, dtype=shift_px.dtype).view(-1, 1, 1) + torch.zeros_like(
        pos
    )
    if slice_shift is not None:
        zpos = zpos + slice_shift
    xi, zi = pos.floor(), zpos.floor()
    xf, zf = pos - xi, zpos - zi

    def flat(t: Tensor) -> Tensor:  # (B, C|1, D, H, W) -> (B, C|1, H, D*W)
        return t.permute(0, 1, 3, 2, 4).reshape(t.shape[0], t.shape[1], t.shape[3], -1)

    comps = (x.real, x.imag) if x.is_complex() else (x,)
    outs = [torch.zeros_like(flat(c)) for c in comps]
    z_corners = ((0.0, 1.0 - zf),) if slice_shift is None else ((0.0, 1.0 - zf), (1.0, zf))
    for dz, wz in z_corners:
        for dx, wx in ((0.0, 1.0 - xf), (1.0, xf)):
            ix, iz = xi + dx, zi + dz
            ok = (ix >= 0) & (ix <= w - 1) & (iz >= 0) & (iz <= d - 1)
            lin = flat((iz.clamp(0, d - 1) * w + ix.clamp(0, w - 1)).long())
            wt = torch.where(ok, wz * wx, torch.zeros_like(wz))
            for out, c in zip(outs, comps, strict=True):
                out.scatter_add_(-1, lin.expand(out.shape), flat(c * wt))
    h = x.shape[-2]
    back = [o.view(x.shape[0], x.shape[1], h, d, w).permute(0, 1, 3, 2, 4) for o in outs]
    return torch.complex(back[0], back[1]) if x.is_complex() else back[0]


def _validate(max_radius_mm, delta_chi_ppm, b0_T, bw, te_ms, rf_bw):
    for name, v in (
        ("max_radius_mm", max_radius_mm),
        ("b0_T", b0_T),
        ("readout_bw_hz_per_px", bw),
        ("te_ms", te_ms),
        ("rf_bandwidth_hz", rf_bw),
    ):
        if not v > 0:
            raise ValueError(f"metal_artifact: {name} must be > 0; got {v}")
    if not math.isfinite(delta_chi_ppm) or delta_chi_ppm == 0:
        raise ValueError(
            f"metal_artifact: delta_chi_ppm must be finite and non-zero; got {delta_chi_ppm}"
        )


def metal_artifact_volume(
    x: Tensor,
    theta: float,
    seed: int,
    geometry: AcquisitionGeometry,
    max_radius_mm: float = MAX_RADIUS_MM,
    delta_chi_ppm: float = DELTA_CHI_PPM,
    b0_T: float = 3.0,
    readout_bw_hz_per_px: float = 250.0,
    te_ms: float = 10.0,
    gradient_echo: bool = False,
    rf_bandwidth_hz: float = RF_BANDWIDTH_HZ,
) -> Tensor:
    """Metal-sphere artefact of ``x`` ``(B, C, D, H, W)`` in any mode (``D == 1``: a slice)."""
    check_input(x, geometry, "metal_artifact")
    _validate(max_radius_mm, delta_chi_ppm, b0_T, readout_bw_hz_per_px, te_ms, rf_bandwidth_hz)
    if geometry.mode is AcquisitionMode.SLICE_2D and geometry.b0_axis != "slice":
        raise ValueError(
            "metal_artifact: a slice_2d acquisition has B0 through-plane (b0_axis='slice')"
        )
    a = float(theta) * max_radius_mm
    if a <= 0.0:
        return x
    b, _, d, h, w = x.shape
    dev, dt = x.device, x.real.dtype
    voxel = voxel_mm_or_unit(geometry.voxel_mm)
    centre = sphere_centre_index(b, (d, h, w), seed, dev)
    rel = []
    for ax, n in enumerate((d, h, w)):
        pos = (torch.arange(n, device=dev, dtype=dt) - n // 2) * voxel[ax]
        off = (centre[:, ax].to(dt) - n // 2) * voxel[ax]
        if ax == 0:
            off = off + sphere_z0_mm(b, a, seed, dev).to(dt)
        view = [1, 1, 1, 1, 1]
        view[2 + ax] = n
        rel.append(pos.view(view) - off.view(b, 1, 1, 1, 1))
    hz, grad = dipole_offset_hz(
        tuple(rel), a, b0_axis=AXIS_INDEX[geometry.b0_axis], b0_T=b0_T, delta_chi_ppm=delta_chi_ppm
    )
    keep = ((rel[0] ** 2 + rel[1] ** 2 + rel[2] ** 2) >= a * a).to(dt)
    if gradient_echo:
        thk = geometry.slice_thickness_mm or voxel[0]
        keep = keep * torch.sinc(grad * thk * te_ms * 1e-3).abs()
    shift = -readout_offset_px(hz, readout_bw_hz_per_px)  # b0_geometric_distortion's polarity
    if geometry.mode is AcquisitionMode.VOLUME_3D:
        return forward_map_readout(x * keep, shift)
    thk = geometry.slice_thickness_mm or voxel[0]
    dz = slice_select_shift_mm(hz, thk, rf_bandwidth_hz) / voxel[0]
    return forward_map_readout(x * keep, shift, dz)


def degrade_metal_artifact(
    x: Tensor, theta: float, seed: int = 0, voxel_mm=None, **kwargs
) -> Tensor:
    """2-D kernel: ``x`` ``(B, C, H, W)`` as a one-slice volume."""
    return run_as_volume(metal_artifact_volume, x, theta, seed, voxel_mm, **kwargs)


#: Sequence -> does the signal dephase across the voxel (no refocusing pulse). bSSFP has none.
GRADIENT_ECHO_OF: dict[str, bool] = {
    "spin_echo": False,
    "tse": False,
    "ir_tse": False,
    "gre": True,
    "spoiled_gre": True,
    "mprage": True,
    "bssfp": True,
}


def metal_condition(profile, theta: float, geometry: AcquisitionGeometry, shape) -> dict:
    """B0, readout bandwidth, echo type and (when the profile has it) TE from the scanner."""
    # The readout shift exists only for spin-warp sequences: EPI and non-Cartesian raise here.
    if b0_shift_sequence(profile.sequence) != "spin_warp":
        raise ValueError(
            f"metal_artifact models the spin-warp readout shift only; sequence "
            f"{str(profile.sequence)!r} shifts along phase-encode"
        )
    out = {
        "b0_T": profile.b0_T,
        "readout_bw_hz_per_px": profile.readout_bw_hz_per_px,
        "gradient_echo": GRADIENT_ECHO_OF[profile.sequence],
    }
    if profile.te_ms is not None:
        out["te_ms"] = profile.te_ms
    return out
