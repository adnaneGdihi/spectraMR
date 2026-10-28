"""Joint total variation along the phase-encode direction (Algorithm H).

Only ky is undersampled, so only the y direction of each coil image aliases,
and every coil image is the same anatomy times a smooth sensitivity: the coils
share their y-edges. Per estimate, ``ISTA_STEPS`` iterations of

.. math::

    X \\leftarrow \\operatorname{prox}_{\\lambda\\,\\mathrm{TV}_y}
        \\bigl(\\mathcal{F}^{-1}[\\,O\\,\\hat y + (1 - O)\\,\\mathcal{F}X\\,]\\bigr),
    \\qquad
    \\mathrm{TV}_y(X) = \\sum \\sqrt{\\textstyle\\sum_c |\\partial_y X_c|^2},

run on the coil images, where ``y_hat`` is the estimate's own measured bins after
data consistency (so a soft ``dc_method`` is honoured). The inner step is
the exact projection onto the data, i.e. an ISTA step of size 1 on
``1/2 ||O F X - y_hat||^2``, and the prox is ``PROX_STEPS`` Chambolle dual
iterations with the dual variable projected jointly over coils. ``lambda`` is
``TV_WEIGHT`` times the median joint y-gradient of the input, per sample.

Energy: ``TV(0) = 0`` is TV's minimum, so the prox fixes zero and, being
non-expansive, never raises the image norm; the data step only writes values
the loop has already bounded. The loop's bound still runs after the projection.

Monotonicity: the data term has a 1-Lipschitz gradient, so the ISTA majorizer
``Q(u; v) = 1/2 ||u - f||^2 + lambda TV_y(u) + const`` bounds the objective
``1/2 ||O F X - y_hat||^2 + lambda TV_y`` from above and touches it at ``v``.
A prox output is accepted per sample only if it lowers ``Q`` below its value at
the current iterate, so the inexact prox cannot raise the objective; refusals
are counted in ``reverse_stats["projection_rejected_steps"]``. The iterate is
held as ``anchor + r`` with ``anchor = F^-1 y_hat``, and every term is evaluated
from ``r`` and the prox correction rather than as a difference of k-space
arrays: a saturated estimate reaches ``|k| ~ 1e10``, where the complex64
round-off of ``F F^-1 y_hat - y_hat`` alone matched the whole objective. The
``projection_residual`` invariant checks the recorded objective is
non-increasing, which verifies the safeguard rather than trusting it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c
from spectramr.models.diffusion.reverse_domain import linear_kspace, stacked_kspace
from spectramr.models.diffusion.reverse_envelope import per_coefficient_support
from spectramr.models.diffusion.reverse_integrity import check_monotone

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

    from spectramr.models.diffusion.kspace_process import PhysicsInformedColdDiffusion

ISTA_STEPS = 3
PROX_STEPS = 30
#: ``lambda`` as a multiple of the per-sample median joint y-gradient.
TV_WEIGHT = 0.5
#: Round-off allowance on the objective between iterates.
OBJECTIVE_REL_TOL = 1e-5
#: Dual step ``tau``; ``||D||^2 <= 4`` for the forward difference.
_DUAL_STEP = 0.25


def grad_y(u: Tensor) -> Tensor:
    """Forward difference along y (dim -2), zero on the last row (Neumann)."""
    d = torch.zeros_like(u)
    d[..., :-1, :] = u[..., 1:, :] - u[..., :-1, :]
    return d


def grad_y_adjoint(p: Tensor) -> Tensor:
    """Adjoint of :func:`grad_y` (minus the discrete divergence)."""
    q = p.clone()
    q[..., -1, :] = 0
    out = -q
    out[..., 1:, :] += q[..., :-1, :]
    return out


def joint_magnitude(d: Tensor) -> Tensor:
    """``sqrt(sum_c |d_c|^2)`` per pixel, ``[B, 1, H, W]``."""
    return d.abs().pow(2).sum(dim=1, keepdim=True).sqrt()


def tv_y(u: Tensor) -> Tensor:
    """Joint y-TV per sample, ``[B]``."""
    return joint_magnitude(grad_y(u)).sum(dim=(1, 2, 3))


def prox_tv_y_correction(f: Tensor, lam: Tensor) -> Tensor:
    """``prox_tv_y(f, lam) - f``, kept apart so it is not rounded away against a large ``f``."""
    p = torch.zeros_like(f)
    safe = lam.clamp_min(1e-12)
    for _ in range(PROX_STEPS):
        u = f - lam * grad_y_adjoint(p)
        p = p + (_DUAL_STEP / safe) * grad_y(u)
        p = p / joint_magnitude(p).clamp_min(1.0)
    return -lam * grad_y_adjoint(p)


def prox_tv_y(f: Tensor, lam: Tensor) -> Tensor:
    """Approximate ``argmin_u 1/2 ||u - f||^2 + lam TV_y(u)``; ``lam`` is ``[B, 1, 1, 1]``."""
    return f + prox_tv_y_correction(f, lam)


def accept_prox(candidate: Tensor, current: Tensor) -> Tensor:
    """Per-sample acceptance of a prox output: its majorizer value is strictly lower."""
    return candidate < current


def build_joint_tv(
    sampler: PhysicsInformedColdDiffusion, measurement: Tensor, obs: Tensor
) -> Callable[..., Tensor]:
    """Return the per-estimate joint-TV projector for ``measurement``'s support."""
    log_scaled = sampler.kspace_log_scaled
    held = (per_coefficient_support(obs, measurement) > 0).to(torch.float32)

    def energy(z: Tensor) -> Tensor:
        return 0.5 * z.abs().pow(2).sum(dim=(1, 2, 3))

    def project(x0: Tensor, obs_now: Tensor, *, where: str) -> Tensor:
        with torch.autocast(device_type=x0.device.type, enabled=False):
            data = linear_kspace(x0.float(), log_scaled=log_scaled).to(torch.complex64)
            anchor = ifft2c(data)
            gradient = joint_magnitude(grad_y(anchor)).flatten(1)
            lam = TV_WEIGHT * gradient.median(dim=1).values.view(-1, 1, 1, 1)
            r = torch.zeros_like(anchor)
            tv_now = lam.flatten() * tv_y(anchor)
            values = [tv_now]
            rejected = torch.zeros((), dtype=torch.long, device=x0.device)
            for _ in range(ISTA_STEPS):
                g = ifft2c(fft2c(r) * held)
                f = anchor + (r - g)
                delta = prox_tv_y_correction(f, lam)
                tv_new = lam.flatten() * tv_y(f + delta)
                keep = accept_prox(energy(delta) + tv_new, energy(g) + tv_now)
                r = torch.where(keep.view(-1, 1, 1, 1), r - g + delta, r)
                tv_now = torch.where(keep, tv_new, tv_now)
                rejected = rejected + (~keep).sum()
                values.append(energy(fft2c(r) * held) + tv_now)
            previous = sampler.last_projection_rejected
            sampler.last_projection_rejected = rejected if previous is None else previous + rejected
            check_monotone(
                sampler,
                values,
                invariant="projection_residual",
                where=where,
                rel_tol=OBJECTIVE_REL_TOL,
            )
            out = stacked_kspace(data + fft2c(r), x0, log_scaled=log_scaled)
        return x0 * obs_now + out * (1.0 - obs_now)

    return project


__all__ = [
    "ISTA_STEPS",
    "PROX_STEPS",
    "TV_WEIGHT",
    "accept_prox",
    "build_joint_tv",
    "grad_y",
    "grad_y_adjoint",
    "prox_tv_y",
    "prox_tv_y_correction",
    "tv_y",
]
