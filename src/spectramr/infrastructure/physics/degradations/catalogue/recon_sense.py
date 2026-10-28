"""Parallel-imaging reconstruction axes: SENSE g-factor noise and residual aliasing.

``gfactor_noise`` adds the noise a SENSE reconstruction at acceleration ``R`` amplifies spatially,
``sigma(r) = sigma_0 g(r) sqrt(R)`` (Pruessmann et al., MRM 42:952, 1999, Eq. 13). ``pi_residual_aliasing``
unfolds the aliased coil images with miscalibrated maps ``S' = S (1 + eps)`` and returns the
difference to the exact unfolding, i.e. the residual replicas ``sum_k eps_k x(r + k FOV/R)`` at first
order. Both work on the (D, H) plane of ``(B, C, D, H, W)``: R_y along phase encode, R_z along
slices (multislice: simultaneous multi-slice) or kz (3-D), with CAIPI shift over the two.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from spectramr.infrastructure.physics.degradations._common import (
    _ensure_complex,
    _generator,
    _signal_level,
    derive_axis_seed,
)
from spectramr.infrastructure.physics.degradations.blocks.volume_fields import smooth_random_field
from spectramr.infrastructure.physics.degradations.catalogue.motion_timing import (
    check_input,
    run_as_volume,
)
from spectramr.infrastructure.physics.degradations.catalogue.sense_core import (
    AliasPlan,
    alias_plan,
    fold_members,
    gfactor,
    pad_to_plan,
    seeded_smooth_csm,
    sense_unfold,
)
from spectramr.infrastructure.physics.degradations.geometry import (
    AcquisitionGeometry,
    AcquisitionMode,
)
from spectramr.infrastructure.physics.per_sample_rng import per_sample_normal_pair

__all__ = [
    "DEFAULT_ACCELERATION",
    "DEFAULT_N_COILS",
    "SenseSetup",
    "gfactor_noise_2d",
    "gfactor_noise_volume",
    "pi_residual_2d",
    "pi_residual_volume",
    "residual_image",
    "sense_setup",
]

#: (R_pe, R_slice) and coil count used when neither kwargs nor a scanner profile give them.
DEFAULT_ACCELERATION = (2.0, 1.0)
DEFAULT_N_COILS = 8


class SenseSetup:
    """The aliasing plan, padded maps and ``Psi^-1`` of one call."""

    __slots__ = ("csm", "plan", "psi_inv", "r_total", "shape")

    def __init__(
        self, csm: Tensor, plan: AliasPlan, psi_inv: Tensor | None, shape: tuple[int, ...]
    ):
        self.csm, self.plan, self.psi_inv, self.shape = csm, plan, psi_inv, shape
        self.r_total = plan.n

    def crop(self, t: Tensor) -> Tensor:
        return t[..., : self.shape[-3], : self.shape[-2], :]


def _whole(r: float, what: str) -> int:
    if float(r) != int(r) or int(r) < 1:
        raise ValueError(f"{what} must be a whole number >= 1 to alias the FOV; got {r!r}")
    return int(r)


def _resolve_maps(
    coil_maps: Tensor, batch: int, n_coils: int, spatial: tuple[int, int, int], name: str
) -> Tensor:
    d, h, w = spatial
    if coil_maps.dim() == 4 and d == 1:
        coil_maps = coil_maps.unsqueeze(2)
    if not (
        coil_maps.dim() == 5
        and coil_maps.shape[0] in (1, batch)
        and tuple(coil_maps.shape[1:]) == (n_coils, d, h, w)
    ):
        raise ValueError(
            f"{name}: coil_maps must be (1|B, {n_coils}, [{d},] {h}, {w}); got {tuple(coil_maps.shape)}"
        )
    return _ensure_complex(coil_maps).expand(batch, -1, -1, -1, -1)


def _psi_inv(cov: object, n_coils: int, device: torch.device, name: str) -> Tensor | None:
    if cov is None:
        return None
    psi = torch.as_tensor(cov, dtype=torch.float64, device=device)
    if psi.shape != (n_coils, n_coils):
        raise ValueError(
            f"{name}: coil_covariance must be {n_coils} x {n_coils}; got {tuple(psi.shape)}"
        )
    return torch.linalg.inv(psi)


def sense_setup(
    like: Tensor,
    n_coils: int,
    geometry: AcquisitionGeometry,
    seed: int,
    *,
    name: str,
    acceleration: tuple[float, float],
    caipi_shift: int | None,
    coil_maps: Tensor | None,
    coil_covariance: object,
) -> SenseSetup:
    """Validate the acceleration and build maps (given, or seeded synthetic) for ``like``'s grid.

    ``like`` is ``[B, *, D, H, W]``; its channel count is not read (``n_coils`` is).
    """
    r_pe, r_sl = (_whole(r, f"{name}: acceleration") for r in acceleration)
    if geometry.mode is AcquisitionMode.SLICE_2D and r_sl != 1:
        raise ValueError(f"{name}: a 2-D slice has no slice/kz acceleration; got R_slice = {r_sl}")
    if n_coils < r_pe * r_sl:
        raise ValueError(
            f"{name}: SENSE at R = {r_pe}x{r_sl} needs at least {r_pe * r_sl} coils; got {n_coils}"
        )
    if caipi_shift is None:
        shift = 1 if r_pe > 1 and r_sl > 1 else 0
    elif isinstance(caipi_shift, int) and caipi_shift >= 0:
        shift = caipi_shift
    else:
        raise ValueError(f"{name}: caipi_shift must be an integer >= 0; got {caipi_shift!r}")
    b, spatial = like.shape[0], tuple(like.shape[-3:])
    plan = alias_plan(spatial[:2], r_pe, r_sl, shift, like.device)
    if coil_maps is None:
        gen = _generator(like.device, derive_axis_seed(seed, "csm"))
        csm = seeded_smooth_csm(
            b,
            n_coils,
            spatial,
            geometry.voxel_mm,
            generator=gen,
            device=like.device,
            dtype=like.real.dtype,
        )
    else:
        csm = _resolve_maps(coil_maps, b, n_coils, spatial, name).to(like.device)
    psi_inv = _psi_inv(coil_covariance, n_coils, like.device, name)
    return SenseSetup(pad_to_plan(csm, plan, "replicate"), plan, psi_inv, tuple(like.shape))


def gfactor_noise_volume(
    x: Tensor,
    theta: float,
    seed: int,
    geometry: AcquisitionGeometry,
    acceleration: tuple[float, float] = DEFAULT_ACCELERATION,
    n_coils: int | None = None,
    coil_maps: Tensor | None = None,
    coil_covariance: object = None,
    caipi_shift: int | None = None,
    sigma_max: float = 0.1,
) -> Tensor:
    """Complex noise of std ``theta sigma_max p99(|x|) g(r) sqrt(R)`` added to the image.

    ``E|n|^2 = sigma^2`` (real and imaginary parts ``sigma / sqrt 2``), drawn independently per voxel and
    channel: the spatial noise correlation between aliased voxels is dropped, the variance map is exact.
    ``n_coils`` (default 8) is the receive array (the image may be one combined channel); with ``coil_maps`` it must equal their coil count.
    """
    check_input(x, geometry, "gfactor_noise")
    if sigma_max < 0:
        raise ValueError(f"gfactor_noise: sigma_max must be >= 0; got {sigma_max}")
    x = _ensure_complex(x)
    if float(theta) <= 0.0:
        return x
    if coil_maps is None:
        n_coils = DEFAULT_N_COILS if n_coils is None else n_coils
    elif n_coils is not None and n_coils != coil_maps.shape[1]:
        raise ValueError(
            f"gfactor_noise: n_coils = {n_coils} (kwarg or scanner profile) disagrees with the "
            f"{coil_maps.shape[1]} coils of coil_maps"
        )
    else:
        n_coils = coil_maps.shape[1]
    st = sense_setup(
        x,
        n_coils,
        geometry,
        seed,
        name="gfactor_noise",
        acceleration=acceleration,
        caipi_shift=caipi_shift,
        coil_maps=coil_maps,
        coil_covariance=coil_covariance,
    )
    g = st.crop(gfactor(st.csm, st.plan, st.psi_inv)).to(x.real.dtype)
    sigma = float(theta) * sigma_max * _signal_level(x.abs()).view(-1, 1, 1, 1, 1)
    gen = _generator(x.device, derive_axis_seed(seed, "noise"))
    re, im = per_sample_normal_pair(*x.shape, generator=gen, device=x.device, dtype=x.real.dtype)
    return x + torch.complex(re, im) * (sigma * g * math.sqrt(st.r_total / 2.0))


def residual_image(x: Tensor, st: SenseSetup, eps: Tensor) -> Tensor:
    """Image-space residual ``S (m' - m)`` of unfolding ``x`` with ``S (1 + eps)`` instead of ``S``.

    ``m`` and ``m'`` are the exact and the miscalibrated unfolding (Pruessmann 1999, Eq. 6) of the data
    aliased with the true maps; ``eps`` is ``[B, C, D, H, W]`` on the unpadded grid.
    """
    aliased = fold_members(pad_to_plan(x, st.plan, "constant"), st.plan).sum(-1)
    wrong = st.csm * (1.0 + pad_to_plan(eps, st.plan, "replicate"))
    m = sense_unfold(aliased, wrong, st.plan, st.psi_inv) - sense_unfold(
        aliased, st.csm, st.plan, st.psi_inv
    )
    return st.crop(st.csm) * st.crop(m)


def pi_residual_volume(
    x: Tensor,
    theta: float,
    seed: int,
    geometry: AcquisitionGeometry,
    acceleration: tuple[float, float] = DEFAULT_ACCELERATION,
    coil_maps: Tensor | None = None,
    coil_covariance: object = None,
    caipi_shift: int | None = None,
    eps_max: float = 0.2,
) -> Tensor:
    """Coil images plus the residual of unfolding with maps off by a smooth ``eps``, ``max |eps| = theta eps_max``."""
    check_input(x, geometry, "pi_residual_aliasing")
    if eps_max < 0:
        raise ValueError(f"pi_residual_aliasing: eps_max must be >= 0; got {eps_max}")
    x = _ensure_complex(x)
    if float(theta) <= 0.0:
        return x
    b, c, d, h, w = x.shape
    st = sense_setup(
        x,
        c,
        geometry,
        seed,
        name="pi_residual_aliasing",
        acceleration=acceleration,
        caipi_shift=caipi_shift,
        coil_maps=coil_maps,
        coil_covariance=coil_covariance,
    )
    gen = _generator(x.device, derive_axis_seed(seed, "eps"))
    field = smooth_random_field(
        b * c, (d, h, w), generator=gen, device=x.device, dtype=x.real.dtype
    )
    eps = float(theta) * eps_max * field.view(b, c, d, h, w)
    return x + residual_image(x, st, eps)


def gfactor_noise_2d(x: Tensor, theta: float, seed: int = 0, voxel_mm=None, **kwargs) -> Tensor:
    return run_as_volume(gfactor_noise_volume, x, theta, seed, voxel_mm, **kwargs)


def pi_residual_2d(x: Tensor, theta: float, seed: int = 0, voxel_mm=None, **kwargs) -> Tensor:
    return run_as_volume(pi_residual_volume, x, theta, seed, voxel_mm, **kwargs)
