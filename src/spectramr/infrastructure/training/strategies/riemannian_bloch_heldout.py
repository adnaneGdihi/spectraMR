"""Held-out q-map prediction for Riemannian Bloch diffusion (Idea 5).

The training objective is score matching on single manifold points, so it never
produces a parameter map. The held-out pass turns the learned field into one: each
voxel of a held-out ``[B, 3, *spatial]`` map is noised with the training forward
kernel at ``t_eval`` and integrated back to ``t_min`` along the probability-flow
ODE with geodesic-Euler steps. The result is a ``[B, 3, *spatial]`` map that
``geodesic_qmap_error`` grades against the clean one.

Only foreground voxels are noised, denoised and graded. Masked maps (the NIST
``MAP_MASKED`` stacks) carry a ``(0, 0, 0)`` background that lies outside the
manifold's support; scoring it would mostly measure clamped padding.

Sign convention: the strategy regresses ``u = -log_map(theta_t, theta_0) / t``,
the negative of the heat-kernel score, so the reverse step is ``exp(theta, -dt/2 * u)``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

from spectramr.infrastructure.physics.manifolds import BlochRelaxationManifold

#: ``field(theta [N, 3], t [N]) -> u [N, 3]``, the trained tangent regression.
TangentField = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class HeldoutProtocol:
    """How a held-out map is noised and denoised; logged once per run."""

    t_eval: float
    t_min: float
    step_size: float

    def __post_init__(self) -> None:
        if not 0.0 < self.t_min < self.t_eval:
            raise ValueError(
                f"held-out pass needs 0 < t_min < t_eval; got t_min={self.t_min}, "
                f"t_eval={self.t_eval}."
            )
        if self.step_size <= 0.0:
            raise ValueError(f"step_size must be positive; got {self.step_size}.")

    @classmethod
    def from_knobs(
        cls, eval_t: float | None, *, t_min: float, t_max: float, step_size: float
    ) -> HeldoutProtocol | None:
        """Read ``heldout_eval_t``: ``None`` when unset, raise above ``t_max``."""
        if eval_t is None:
            return None
        if float(eval_t) > t_max:
            raise ValueError(
                f"training.riemannian_bloch_diffusion.heldout_eval_t={eval_t} exceeds "
                f"t_max={t_max}: the field was never trained at that noise level."
            )
        return cls(t_eval=float(eval_t), t_min=t_min, step_size=step_size)

    @property
    def n_steps(self) -> int:
        return max(1, math.ceil((self.t_eval - self.t_min) / self.step_size))

    def describe(self) -> str:
        return (
            f"noise held-out maps at t={self.t_eval:g}, probability-flow ODE back to "
            f"t={self.t_min:g} in {self.n_steps} geodesic-Euler steps"
        )


def maps_to_points(maps: torch.Tensor) -> torch.Tensor:
    """``[B, 3, *spatial]`` -> ``[N, 3]``, one (M0, T1, T2) point per voxel."""
    if maps.dim() < 3 or maps.shape[1] != BlochRelaxationManifold.dim:
        raise ValueError(
            f"held-out q-map pass expects [B, 3, *spatial] maps (M0, T1, T2); got "
            f"{tuple(maps.shape)}."
        )
    return maps.movedim(1, -1).reshape(-1, BlochRelaxationManifold.dim)


def points_to_maps(points: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`maps_to_points` for a map shaped like ``like``."""
    spatial_last = like.movedim(1, -1).shape
    return points.reshape(spatial_last).movedim(-1, 1)


def noise_points(
    manifold: BlochRelaxationManifold,
    theta0: torch.Tensor,
    t: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """The training forward kernel at a fixed ``t``: a tangent draw advanced by ``exp``."""
    noise = torch.randn(theta0.shape, generator=generator, device=theta0.device)
    v = manifold.project_tangent(theta0, noise.to(theta0.dtype) * math.sqrt(t))
    return manifold.exp_map(theta0, v, n_steps=4)


def denoise_points(
    manifold: BlochRelaxationManifold,
    field: TangentField,
    theta_t: torch.Tensor,
    protocol: HeldoutProtocol,
) -> torch.Tensor:
    """Integrate the probability-flow ODE from ``t_eval`` down to ``t_min``."""
    n = protocol.n_steps
    dt = (protocol.t_eval - protocol.t_min) / n
    theta = theta_t
    t = torch.full((theta.shape[0],), protocol.t_eval, device=theta.device, dtype=theta.dtype)
    for _ in range(n):
        u = field(theta, t)
        theta = manifold.exp_map(theta, -0.5 * dt * u, n_steps=1)
        t = t - dt
    return theta


def foreground_mask(maps: torch.Tensor) -> torch.Tensor:
    """``[B, *spatial]`` bool: voxels with M0, T1 and T2 all strictly positive.

    The manifold's support is the positive octant, so a voxel with any
    non-positive coordinate is background padding, not tissue.
    """
    fg = (maps > 0).all(dim=1)
    if not bool(fg.any()):
        raise ValueError(
            f"held-out q-map batch {tuple(maps.shape)} has no foreground voxel "
            "(M0, T1, T2 all > 0): nothing to grade."
        )
    return fg


def heldout_prediction(
    manifold: BlochRelaxationManifold,
    field: TangentField,
    maps: torch.Tensor,
    protocol: HeldoutProtocol,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(prediction, noised, foreground)``.

    ``prediction`` and ``noised`` are shaped like ``maps``; only foreground voxels
    are noised and denoised, background voxels keep their target value.
    ``noised`` is the denoiser's own input, scored as the baseline the prediction
    must beat. ``foreground`` is the ``[B, *spatial]`` mask the metric grades on.
    """
    fg = foreground_mask(maps)
    theta0 = maps_to_points(maps)
    fg_points = fg.reshape(-1)
    theta_t = noise_points(manifold, theta0[fg_points], protocol.t_eval, generator)
    theta_hat = denoise_points(manifold, field, theta_t, protocol)
    pred, noised = theta0.clone(), theta0.clone()
    pred[fg_points] = theta_hat
    noised[fg_points] = theta_t
    return points_to_maps(pred, maps), points_to_maps(noised, maps), fg


__all__ = [
    "HeldoutProtocol",
    "TangentField",
    "denoise_points",
    "foreground_mask",
    "heldout_prediction",
    "maps_to_points",
    "noise_points",
    "points_to_maps",
]
