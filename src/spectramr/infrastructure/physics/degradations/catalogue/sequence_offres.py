"""offres_blur_noncartesian: off-resonance blur of a radial readout, by time-segmented NUFFT.

A spin at off-resonance ``df(r)`` Hz evolves ``exp(-i 2 pi df t)`` during the readout, so
``s(t) = sum_r m(r) exp(-i 2 pi df(r) t) exp(-i k(t) . r)`` (Sutton, Noll & Fessler, IEEE TMI 22:178, 2003;
Noll, Meyer, Pauly et al., IEEE TMI 10:629, 1991). The samples come from ``TimeSegmentedNUFFT`` (Fessler,
Lee, Olafsson & Noll, IEEE TSP 53:3393, 2005) on a golden-angle radial trajectory (Winkelmann et al.,
IEEE TMI 26:68, 2007) with ``t = 0`` at the spoke centre. The image is the gridding reconstruction
``A0^H W s`` that ignores ``df`` (``W`` the radial ramp density compensation), and only the change
off-resonance makes to it is applied: ``out = x + A0^H W (s_df - s_0) / g``, ``g`` the on-resonance point
response at the centre. The trajectory's own streaks and gridding error are therefore not added to the object.
A stack of stars (radial in plane, Cartesian kz) with fully sampled kz factorises per slice, because the kz
FFT and its inverse cancel; each slice then sees its own plane of the 3-D ``df`` field.
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch
from torch import Tensor

from spectramr.infrastructure.physics.degradations._common import (
    _ensure_complex,
    _generator,
    derive_axis_seed,
)
from spectramr.infrastructure.physics.degradations.blocks.volume_fields import smooth_random_field
from spectramr.infrastructure.physics.degradations.catalogue.motion_timing import (
    check_input,
    run_as_volume,
)
from spectramr.infrastructure.physics.degradations.geometry import AcquisitionGeometry
from spectramr.infrastructure.physics.gridding import RadialDCF
from spectramr.infrastructure.physics.nufft.time_segmented_nufft import TimeSegmentedNUFFT
from spectramr.infrastructure.physics.nufft_ops import GoldenAngleTrajectory

__all__ = [
    "DEFAULT_T_RO_MS",
    "MAX_CYCLES",
    "SEGMENTS_PER_CYCLE",
    "default_trajectory",
    "degrade_offres_blur_noncartesian",
    "n_segments",
    "offres_blur_volume",
]

#: Peak ``|df| * T_ro`` (phase cycles across the readout) at theta = 1.
MAX_CYCLES = 2.0
#: Declared readout duration; only ``df * T_ro`` matters to the image.
DEFAULT_T_RO_MS = 5.0
#: Time segments per phase cycle: linear interpolation between segments errs by about (phase step)^2 / 8,
#: under 2% for 16 segments per cycle across the readout.
SEGMENTS_PER_CYCLE = 16


def n_segments(cycles: float) -> int:
    """Time segments for a readout with ``cycles`` peak phase cycles."""
    return max(2, math.ceil(SEGMENTS_PER_CYCLE * cycles) + 1)


@lru_cache(maxsize=8)
def _golden(n_spokes: int, n_ro: int, im_size: tuple[int, int]) -> tuple[Tensor, Tensor]:
    traj = GoldenAngleTrajectory(im_size, n_spokes, n_ro).generate(1)[0]
    # t = 0 at the spoke centre; t = +-T_ro / 2 at the radius +-pi (in units of T_ro).
    return traj, (torch.linspace(-math.pi, math.pi, n_ro) / (2 * math.pi)).repeat(n_spokes)


def default_trajectory(
    h: int, w: int, n_spokes: int | None = None, samples_per_spoke: int | None = None
) -> tuple[Tensor, Tensor]:
    """Golden-angle radial ``([2, N] in [-pi, pi], [N] readout time in units of T_ro)``, ``t = 0`` at the centre."""
    n_ro = max(h, w) if samples_per_spoke is None else samples_per_spoke
    n_sp = math.ceil(math.pi / 2 * max(h, w)) if n_spokes is None else n_spokes
    if n_ro < 2 or n_sp < 1:
        raise ValueError(f"n_spokes >= 1 and samples_per_spoke >= 2 required, got {n_sp}, {n_ro}")
    return _golden(int(n_sp), int(n_ro), (h, w))


def _trajectory(
    h: int,
    w: int,
    t_ro_ms: float | None,
    trajectory: Tensor | None,
    readout_times_s: Tensor | None,
    n_spokes: int | None,
    samples_per_spoke: int | None,
) -> tuple[Tensor, Tensor, int, int, float]:
    """``(trajectory [2, N], readout times [N] s, n_spokes, samples_per_spoke, T_ro s)``.

    A supplied trajectory is validated and its ``T_ro`` is the span of ``readout_times_s``, so the
    declared peak |df| * T_ro is the realised one; ``t_ro_ms`` then must not be given.
    """
    if t_ro_ms is not None and (
        isinstance(t_ro_ms, bool) or not isinstance(t_ro_ms, int | float) or not t_ro_ms > 0
    ):
        raise ValueError(f"t_ro_ms must be positive, got {t_ro_ms!r}")
    sps = max(h, w) if samples_per_spoke is None else samples_per_spoke
    if trajectory is None:
        if readout_times_s is not None:
            raise ValueError("readout_times_s accompanies a supplied trajectory")
        traj, t_unit = default_trajectory(h, w, n_spokes, samples_per_spoke)
        t_ro_s = (DEFAULT_T_RO_MS if t_ro_ms is None else t_ro_ms) * 1e-3
        return traj, t_unit * t_ro_s, traj.shape[1] // sps, sps, t_ro_s
    if readout_times_s is None or n_spokes is not None:
        raise ValueError("a supplied trajectory needs readout_times_s (and no n_spokes)")
    if t_ro_ms is not None:
        raise ValueError("t_ro_ms is the span of readout_times_s for a supplied trajectory")
    traj, times = trajectory.float(), readout_times_s.float()
    if traj.dim() != 2 or traj.shape[0] != 2 or times.shape != traj.shape[1:]:
        raise ValueError(
            f"trajectory must be [2, N] with readout_times_s [N], got {tuple(traj.shape)}"
        )
    if traj.shape[1] % sps:
        raise ValueError(f"{traj.shape[1]} samples are not a whole number of {sps}-sample spokes")
    cpu_times = readout_times_s.detach().cpu().float()
    t_ro_s = float(cpu_times.max() - cpu_times.min())
    if not t_ro_s > 0:
        raise ValueError("readout_times_s must span a positive duration")
    return traj, times, traj.shape[1] // sps, sps, t_ro_s


def _unit_field(
    b0_map_hz: Tensor | None, shape: tuple[int, ...], seed: int, device: torch.device
) -> Tensor:
    """``[B, 1|C, D, H, W]`` off-resonance pattern of unit peak: the user's map, or a seeded smooth one."""
    b, c, d, h, w = shape
    if b0_map_hz is None:
        gen = _generator(device, derive_axis_seed(seed, "offres_b0"))
        return smooth_random_field(b, (d, h, w), generator=gen, device=device)
    m = b0_map_hz.to(device=device, dtype=torch.float32)
    if (
        m.dim() != 5
        or m.shape[0] not in (1, b)
        or m.shape[1] not in (1, c)
        or m.shape[2:] != shape[2:]
    ):
        raise ValueError(f"b0_map_hz must be (B|1, 1|C, {d}, {h}, {w}), got {tuple(m.shape)}")
    peak = m.abs().amax(dim=(1, 2, 3, 4), keepdim=True)
    # Asynchronous on a device (no host sync); raises RuntimeError for an all-zero map.
    torch._assert_async((peak > 0).all())
    return m / peak


def _gain(traj: Tensor, times: Tensor, h: int, w: int, n_sp: int, sps: int) -> float:
    """On-resonance point response at the image centre of ``A0^H W A0`` (CPU, a plain number).

    Costs one single-image NUFFT pair on every call; it is not cached.
    """
    plain = TimeSegmentedNUFFT((h, w), 1)
    delta = torch.zeros(1, 1, h, w, dtype=torch.complex64)
    delta[..., h // 2, w // 2] = 1.0
    zero = torch.zeros(1, 1, h, w)
    k = plain(delta, zero, traj, readout_times=times) * RadialDCF(n_sp, sps)(traj.T)
    return float(plain.adjoint(k, zero, traj, readout_times=times)[0, 0, h // 2, w // 2].real)


def offres_blur_volume(
    x: Tensor,
    theta: float,
    seed: int,
    geometry: AcquisitionGeometry,
    *,
    t_ro_ms: float | None = None,
    b0_map_hz: Tensor | None = None,
    trajectory: Tensor | None = None,
    readout_times_s: Tensor | None = None,
    n_spokes: int | None = None,
    samples_per_spoke: int | None = None,
) -> Tensor:
    check_input(x, geometry, "offres_blur_noncartesian")
    b, c, d, h, w = x.shape
    traj, times, n_sp, sps, t_ro_s = _trajectory(
        h, w, t_ro_ms, trajectory, readout_times_s, n_spokes, samples_per_spoke
    )
    unit = _unit_field(b0_map_hz, (b, c, d, h, w), seed, x.device)
    if theta == 0:
        return x
    dev = x.device
    traj, times = traj.to(dev), times.to(dev)
    dcf = RadialDCF(n_sp, sps)(traj.T)
    omega = 2 * math.pi * (MAX_CYCLES * theta / t_ro_s) * unit.expand(b, c, d, h, w)
    img = _ensure_complex(x).to(torch.complex64).reshape(-1, 1, h, w)
    omega = omega.reshape(-1, 1, h, w)
    seg, plain = (
        TimeSegmentedNUFFT((h, w), n_segments(MAX_CYCLES * theta)),
        TimeSegmentedNUFFT((h, w), 1),
    )

    def forward(op: TimeSegmentedNUFFT, image: Tensor, field: Tensor) -> Tensor:
        return op(image, field, traj, readout_times=times)

    def recon(k: Tensor) -> Tensor:
        zero = torch.zeros(k.shape[0], 1, h, w, device=dev)
        return plain.adjoint(k * dcf, zero, traj, readout_times=times)

    gain = _gain(traj.cpu(), times.cpu(), h, w, n_sp, sps)
    dk = forward(seg, img, omega) - forward(plain, img, torch.zeros_like(omega))
    return (img + recon(dk) / gain).reshape(b, c, d, h, w)


def degrade_offres_blur_noncartesian(
    x: Tensor, theta: float, seed: int = 0, *, b0_map_hz: Tensor | None = None, **kwargs
) -> Tensor:
    """2-D slice ``(B, C, H, W)``: one radial acquisition; ``b0_map_hz`` is ``(B|1, 1|C, H, W)``."""
    if b0_map_hz is not None:
        kwargs["b0_map_hz"] = b0_map_hz.unsqueeze(2)
    return run_as_volume(offres_blur_volume, x, theta, seed, **kwargs)
