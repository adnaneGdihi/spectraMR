"""Coil self-consistency projection between reverse calls (Algorithm E, SPIRiT).

Each receive coil sees the same object through a smooth sensitivity, so every
coil's k-space is a short convolution of all coils' neighbouring samples. Per
sample, a kernel ``G`` (ky 3 x kx 5 x coils) is calibrated by Tikhonov least
squares on the largest contiguous run of fully measured rows; the target
coil's own centre tap is excluded, so ``G = I`` is not a solution. On every
estimate a fixed number of gradient steps

.. math::

    k \\leftarrow k - \\mu\\,(1 - O) \\odot (G - I)^H (G - I)\\, k,
    \\qquad \\mu = 1 / \\lVert G - I \\rVert^2,

pulls the unmeasured bins toward consistency with their neighbours and leaves
the measured bins at their data-consistency values. With ``mu <= 1/L`` the
residual ``||(G - I) k||`` cannot increase, and the ``projection_residual``
invariant checks that it did not.

Energy: the projection moves values but bounds nothing, so the loop's bound
(``reverse_hooks.bound_estimate``) runs after it. Everything here is LINEAR
complex k-space.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from spectramr.models.diffusion.reverse_domain import linear_kspace, stacked_kspace
from spectramr.models.diffusion.reverse_envelope import per_coefficient_support
from spectramr.models.diffusion.reverse_integrity import check_monotone

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

    from spectramr.models.diffusion.kspace_process import PhysicsInformedColdDiffusion

#: Kernel extent (ky, kx); the padding keeps the plane's shape.
KERNEL = (3, 5)
_PAD = (KERNEL[0] // 2, KERNEL[1] // 2)
#: Tikhonov weight relative to the mean diagonal of the calibration Gram matrix.
_TIKHONOV = 1e-3
#: Gradient steps per estimate, and power iterations for ``||G - I||^2``.
PROJECTION_STEPS = 10
_POWER_STEPS = 20
#: Safety margin on the Lipschitz estimate, since power iteration approaches it from below.
_LIPSCHITZ_MARGIN = 1.1


def calibration_rows(measured_rows: Tensor) -> list[tuple[int, int]]:
    """Per sample, ``(start, stop)`` of the longest run of fully measured rows."""
    runs = []
    for row in measured_rows.tolist():
        best, start = (0, 0), None
        for h, on in enumerate([*row, False]):
            if on and start is None:
                start = h
            elif not on and start is not None:
                if h - start > best[1] - best[0]:
                    best = (start, h)
                start = None
        if best[1] - best[0] < KERNEL[0]:
            raise ValueError(
                f"reverse_projection='spirit' needs {KERNEL[0]} contiguous fully measured "
                f"rows to calibrate its kernel; a sample has {best[1] - best[0]}."
            )
        runs.append(best)
    return runs


def calibrate_kernel(core: Tensor) -> Tensor:
    """Kernel ``[nc, nc, 3, 5]`` fitted on a complex calibration block ``[nc, L, W]``."""
    nc, rows, cols = core.shape
    kh, kw = KERNEL
    out_h, out_w = rows - kh + 1, cols - kw + 1
    taps = [core[:, dy : dy + out_h, dx : dx + out_w] for dy in range(kh) for dx in range(kw)]
    sources = torch.stack(taps, dim=1).reshape(nc * kh * kw, -1).T
    targets = core[:, _PAD[0] : _PAD[0] + out_h, _PAD[1] : _PAD[1] + out_w].reshape(nc, -1)
    gram = sources.mH @ sources
    n = gram.shape[0]
    reg = _TIKHONOV * gram.diagonal().real.mean().clamp_min(1e-12)
    rhs = (sources.mH @ targets.T).T
    systems = (gram + reg * torch.eye(n, dtype=gram.dtype, device=gram.device)).expand(nc, n, n)
    systems = systems.clone()
    centre = torch.arange(nc, device=core.device) * kh * kw + _PAD[0] * kw + _PAD[1]
    coil = torch.arange(nc, device=core.device)
    systems[coil, centre, :] = 0
    systems[coil, :, centre] = 0
    systems[coil, centre, centre] = 1
    rhs = rhs.clone()
    rhs[coil, centre] = 0
    weights = torch.linalg.solve(systems, rhs.unsqueeze(-1)).squeeze(-1)
    return weights.reshape(nc, nc, kh, kw)


def _residual_op(
    kernels: Tensor, batch: int
) -> tuple[Callable[[Tensor], Tensor], Callable[[Tensor], Tensor]]:
    """``(G - I)`` and its adjoint over ``[B, nc, H, W]``, one kernel per sample."""
    weight = kernels.flatten(0, 1)

    def forward(k: Tensor) -> Tensor:
        z = F.conv2d(k.flatten(0, 1)[None], weight, padding=_PAD, groups=batch)
        return z.view_as(k) - k

    def adjoint(r: Tensor) -> Tensor:
        z = F.conv_transpose2d(r.flatten(0, 1)[None], weight.conj(), padding=_PAD, groups=batch)
        return z.view_as(r) - r

    return forward, adjoint


def _lipschitz(
    forward: Callable[[Tensor], Tensor], adjoint: Callable[[Tensor], Tensor], like: Tensor
) -> Tensor:
    """Per-sample upper estimate of ``||G - I||^2`` by power iteration, ``[B, 1, 1, 1]``."""
    gen = torch.Generator(device=like.device).manual_seed(0)
    v = torch.randn(like.shape, generator=gen, device=like.device, dtype=like.dtype)
    norm = torch.ones(like.shape[0], 1, 1, 1, device=like.device)
    for _ in range(_POWER_STEPS):
        w = adjoint(forward(v))
        norm = w.abs().pow(2).sum(dim=(1, 2, 3), keepdim=True).sqrt()
        v = w / norm.clamp_min(1e-12)
    return _LIPSCHITZ_MARGIN * norm.clamp_min(1e-12)


def _per_sample_norm(z: Tensor) -> Tensor:
    return z.abs().pow(2).sum(dim=(1, 2, 3)).sqrt()


def build_spirit(
    sampler: PhysicsInformedColdDiffusion, measurement: Tensor, obs: Tensor
) -> Callable[..., Tensor]:
    """Calibrate per-sample kernels on ``measurement`` and return the projector."""
    log_scaled = sampler.kspace_log_scaled
    with torch.autocast(device_type=measurement.device.type, enabled=False):
        y = linear_kspace(measurement.float(), log_scaled=log_scaled).to(torch.complex64)
        support = per_coefficient_support(obs, measurement) > 0
        measured_rows = support.all(dim=-1).all(dim=-2)
        runs = calibration_rows(measured_rows)
        kernels = torch.stack([calibrate_kernel(y[b, :, lo:hi]) for b, (lo, hi) in enumerate(runs)])
        forward, adjoint = _residual_op(kernels, y.shape[0])
        step = 1.0 / _lipschitz(forward, adjoint, y)
    free = (~support).expand(y.shape).to(y.real.dtype)

    def project(x0: Tensor, obs_now: Tensor, *, where: str) -> Tensor:
        with torch.autocast(device_type=x0.device.type, enabled=False):
            k = linear_kspace(x0.float(), log_scaled=log_scaled).to(torch.complex64)
            residuals = []
            for _ in range(PROJECTION_STEPS):
                r = forward(k)
                residuals.append(_per_sample_norm(r))
                k = k - step * free * adjoint(r)
            residuals.append(_per_sample_norm(forward(k)))
            check_monotone(sampler, residuals, invariant="projection_residual", where=where)
            out = stacked_kspace(k, x0, log_scaled=log_scaled)
        return x0 * obs_now + out * (1.0 - obs_now)

    return project


__all__ = ["KERNEL", "PROJECTION_STEPS", "build_spirit", "calibrate_kernel", "calibration_rows"]
