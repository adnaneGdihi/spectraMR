"""SENSE aliasing, unfolding and g-factor on a (D, H, W) grid, with CAIPI-style sampling over (kz, ky).

Sampling the (ky, kz) index lattice ``L`` spanned by ``(R_y, delta)`` and ``(0, R_z)`` (Breuer et al.,
MRM 55:549, 2006, 2D-CAIPIRINHA; delta = 0 is plain SENSE) replicates the image on the dual lattice
(Poisson summation): voxel ``(z, y)`` aliases with ``(z + q D / R_z, y + (p - q delta / R_z) H / R_y)``
for ``p < R_y``, ``q < R_z``, wrapped on the FOV. The partners of a slice are therefore the slices
``z + q D / R_z`` that are excited together (simultaneous multi-slice, blipped CAIPI shifts them in-plane
by ``delta H / (R_y R_z)``) or the kz partitions of a 3-D scan. With ``E`` the ``C x n``
matrix of coil sensitivities at the ``n = R_y R_z`` aliased voxels, ``A = E^H Psi^-1 E``, the unfolded
voxels are ``A^-1 E^H Psi^-1 y`` and ``g_j = sqrt([A^-1]_jj [A]_jj)`` (Pruessmann et al., MRM 42:952,
1999, Eqs. 6 and 13). Matrices are tiny, so ``A`` is inverted with a relative diagonal load of 1e-6
(a numerical guard that caps g near 1e3 where the coils cannot separate a pair).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from spectramr.infrastructure.physics.degradations.volume.fields_maps import field_coords
from spectramr.infrastructure.physics.per_sample_rng import per_sample_rand

__all__ = [
    "AliasPlan",
    "alias_plan",
    "fold_members",
    "gfactor",
    "pad_to_plan",
    "scatter_members",
    "seeded_smooth_csm",
    "sense_unfold",
]

_LOAD = 1e-6


@dataclass(frozen=True, slots=True)
class AliasPlan:
    """The aliased voxel sets of a padded ``(D, H)`` plane: ``members[z0, y0, j]`` = flat ``z * H + y``."""

    members: Tensor  # [Dr, Hr, n] long
    shape: tuple[int, int]  # padded (D, H)

    @property
    def n(self) -> int:
        return self.members.shape[-1]


def _padded(n: int, r: int) -> int:
    return -(-n // r) * r


def alias_plan(
    shape: tuple[int, int], r_pe: int, r_slice: int, caipi_shift: int, device: torch.device
) -> AliasPlan:
    """Aliased sets for the ``(D, H)`` plane, padded up to multiples of ``(R_z, R_y)``."""
    d, h = _padded(shape[0], r_slice), _padded(shape[1], r_pe)
    dr, hr = d // r_slice, h // r_pe
    if caipi_shift and (hr * caipi_shift) % r_slice:
        raise ValueError(
            f"CAIPI shift {caipi_shift} needs (H / R_y) * shift divisible by R_z; "
            f"got H/R_y = {hr}, R_z = {r_slice}"
        )
    p = torch.arange(r_pe, device=device).view(1, 1, r_pe, 1)
    q = torch.arange(r_slice, device=device).view(1, 1, 1, r_slice)
    z0 = torch.arange(dr, device=device).view(dr, 1, 1, 1)
    y0 = torch.arange(hr, device=device).view(1, hr, 1, 1)
    z = z0 + q * dr
    y = (y0 + p * hr - q * (hr * caipi_shift // r_slice)) % h
    members = (z * h + y).expand(dr, hr, r_pe, r_slice).reshape(dr, hr, r_pe * r_slice)
    return AliasPlan(members, (d, h))


def pad_to_plan(t: Tensor, plan: AliasPlan, mode: str) -> Tensor:
    """Pad ``[B, C, D, H, W]`` along D and H to the plan's size: ``replicate`` (maps) or ``constant`` (images)."""
    d, h = plan.shape
    pad = (0, 0, 0, h - t.shape[-2], 0, d - t.shape[-3])
    if not any(pad):
        return t
    if t.is_complex():
        return torch.complex(F.pad(t.real, pad, mode=mode), F.pad(t.imag, pad, mode=mode))
    return F.pad(t, pad, mode=mode)


def fold_members(t: Tensor, plan: AliasPlan) -> Tensor:
    """``[B, C, D, H, W]`` -> ``[B, Dr, Hr, W, C, n]``: each channel's values on the aliased sets."""
    b, c, d, h, w = t.shape
    flat = t.reshape(b, c, d * h, w)[:, :, plan.members]  # [B, C, Dr, Hr, n, W]
    return flat.permute(0, 2, 3, 5, 1, 4)


def scatter_members(vals: Tensor, plan: AliasPlan) -> Tensor:
    """``[B, Dr, Hr, W, n]`` -> ``[B, 1, D, H, W]``; the sets partition the plane, so this is a bijection."""
    b, dr, hr, w, n = vals.shape
    d, h = plan.shape
    out = torch.zeros(b, d * h, w, dtype=vals.dtype, device=vals.device)
    out[:, plan.members.reshape(-1), :] = vals.permute(0, 1, 2, 4, 3).reshape(b, dr * hr * n, w)
    return out.view(b, 1, d, h, w)


def _normal_matrix(e: Tensor, psi_inv: Tensor | None) -> Tensor:
    eh = e.conj().transpose(-2, -1)
    return (eh @ e) if psi_inv is None else (eh @ psi_inv.to(e.dtype) @ e)


def _loaded_inverse(a: Tensor) -> Tensor:
    n = a.shape[-1]
    eye = torch.eye(n, dtype=a.dtype, device=a.device)
    load = _LOAD * torch.diagonal(a, dim1=-2, dim2=-1).real.mean(-1, keepdim=True)
    return torch.linalg.inv(a + load.unsqueeze(-1).to(a.dtype) * eye)


def _sensitivities(csm: Tensor, plan: AliasPlan) -> Tensor:
    """``E`` as ``[B, Dr, Hr, W, C, n]`` from maps ``[B, C, D, H, W]`` (already padded)."""
    return fold_members(csm, plan)


def gfactor(csm: Tensor, plan: AliasPlan, psi_inv: Tensor | None) -> Tensor:
    """``g`` on the padded plane, ``[B, 1, D, H, W]`` (Pruessmann 1999, Eq. 13)."""
    a = _normal_matrix(_sensitivities(csm, plan), psi_inv)
    ai = _loaded_inverse(a)
    diag = torch.diagonal(ai, dim1=-2, dim2=-1).real * torch.diagonal(a, dim1=-2, dim2=-1).real
    return scatter_members(diag.clamp_min(0.0).sqrt(), plan)


def sense_unfold(aliased: Tensor, csm: Tensor, plan: AliasPlan, psi_inv: Tensor | None) -> Tensor:
    """Unfold aliased coil data ``[B, Dr, Hr, W, C]`` with maps ``csm``: voxels ``[B, 1, D, H, W]``."""
    e = _sensitivities(csm, plan)
    a = _normal_matrix(e, psi_inv)
    rhs = e.conj().transpose(-2, -1)
    if psi_inv is not None:
        rhs = rhs @ psi_inv.to(e.dtype)
    m = (_loaded_inverse(a) @ rhs @ aliased.unsqueeze(-1)).squeeze(-1)
    return scatter_members(m, plan)


def seeded_smooth_csm(
    batch: int,
    n_coils: int,
    shape: tuple[int, int, int],
    voxel_mm: tuple[float, float, float] | None,
    *,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Seeded smooth coil maps ``[B, n_coils, D, H, W]``, sum-of-squares normalised to 1.

    Coil ``c`` is a Gaussian lobe (width 0.6-0.9 of the half-FOV) centred on a jittered Fibonacci-sphere
    point of radius 1.2 in coordinates normalised by the largest half-extent, times a smooth phase
    ``exp(i (pi t_c . r + phi_c))`` (receive-coil B1- is a smooth complex field; Roemer et al., MRM
    16:192, 1990). ``coil_sensitivity.create_synthetic_csm`` is unit modulus with a branch cut and no seed,
    which makes every g-factor degenerate, so it cannot stand in for a coil array here.
    """
    z, y, x = field_coords(shape, voxel_mm, device, dtype)
    i = torch.arange(n_coils, device=device, dtype=dtype) + 0.5
    pz = 1.0 - 2.0 * i / n_coils
    ring = (1.0 - pz**2).sqrt()
    ang = i * (3.141592653589793 * (3.0 - 5.0**0.5))
    base = torch.stack([pz, ring * torch.sin(ang), ring * torch.cos(ang)], dim=-1)  # [C, 3]
    u = per_sample_rand(batch, n_coils, 8, generator=generator, device=device, dtype=dtype)
    centre = 1.2 * (base + 0.15 * (2 * u[..., 0:3] - 1))  # [B, C, 3]
    width = 0.6 + 0.3 * u[..., 3]
    tilt = 2 * u[..., 4:7] - 1
    phi0 = 6.283185307179586 * u[..., 7]
    r2 = sum(
        (g - centre[..., k].view(batch, n_coils, 1, 1, 1)) ** 2 for k, g in enumerate((z, y, x))
    )
    amp = torch.exp(-r2 / (2 * width.view(batch, n_coils, 1, 1, 1) ** 2))
    phase = 3.141592653589793 * sum(
        tilt[..., k].view(batch, n_coils, 1, 1, 1) * g for k, g in enumerate((z, y, x))
    ) + phi0.view(batch, n_coils, 1, 1, 1)
    csm = torch.polar(amp, phase)
    rss = csm.abs().pow(2).sum(1, keepdim=True).sqrt().clamp_min(torch.finfo(dtype).tiny)
    return csm / rss
