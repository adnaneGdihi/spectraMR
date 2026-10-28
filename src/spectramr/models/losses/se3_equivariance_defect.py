r"""SE(3) equivariance-defect monitoring loss (B-3 of the VF campaign).

Quantifies how far a map :math:`G` (the navigator's motion-latent estimator or
the full block) departs from being :math:`\mathrm{Ad}_g`-equivariant:

.. math::

    \mathcal{L}_{\mathrm{eq}}(G)
      \;=\; \mathbb{E}_{g}\,
      \bigl\| G(\mathrm{Ad}_g \cdot u) - \mathrm{Ad}_g \cdot G(u) \bigr\|_2 .

It is a **monitoring** loss: a perfectly equivariant block returns exactly zero
(up to floating-point), so it is safe to add at a small weight to surface
training-time symmetry breaking without distorting the primary objective. The
expectation is estimated by a small batch of random in-plane rotations
(SO(2) ⊂ SO(3)); set ``group="SO3"`` to sample full 3D rotations.

The loss does not require paired targets — it operates on a callable ``G`` and
a Lie-algebra trajectory ``u``. When used inside a composite-loss harness that
calls ``forward(prediction, target, **kwargs)``, pass the block as
``kwargs["equivariant_fn"]`` and the trajectory as ``kwargs["lie_input"]``
(falling back to ``prediction`` when omitted, so the registry can instantiate
and smoke-probe it without bespoke wiring).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

import torch
import torch.nn as nn

from spectramr.models.losses.registry import register_loss

_SUPPORTED_GROUPS = ("SO2", "SO3")


def _random_rotation_2d(theta: torch.Tensor) -> torch.Tensor:
    """Embed in-plane angles ``theta`` ``[K]`` as 3x3 rotations ``[K, 3, 3]``.

    Rotation about the z-axis (the slice normal): the in-plane (x, y) block
    rotates, z is fixed.
    """
    k = theta.shape[0]
    out = torch.eye(3, device=theta.device, dtype=theta.dtype).repeat(k, 1, 1)
    c, s = torch.cos(theta), torch.sin(theta)
    out[:, 0, 0] = c
    out[:, 0, 1] = -s
    out[:, 1, 0] = s
    out[:, 1, 1] = c
    return out


def _random_rotation_3d(n: int, device: torch.device, generator: torch.Generator) -> torch.Tensor:
    """Sample ``n`` uniform SO(3) rotation matrices ``[n, 3, 3]`` via QR.

    QR of a Gaussian matrix gives a Haar-random orthogonal matrix; we flip the
    sign of the last column where ``det < 0`` to land in SO(3).
    """
    a = torch.randn(n, 3, 3, device=device, generator=generator)
    qr = torch.linalg.qr(a)
    q: torch.Tensor = qr.Q
    r: torch.Tensor = qr.R
    # Make the decomposition unique / proper: fix signs from R's diagonal.
    diag_sign = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))
    diag_sign[diag_sign == 0] = 1.0
    q = q * diag_sign.unsqueeze(-2)
    det = torch.linalg.det(q)
    # Flip one column where det < 0 to enforce a proper rotation.
    flip = (det < 0).to(q.dtype).view(n, 1, 1)
    q = q.clone()
    q[:, :, -1] = q[:, :, -1] * (1.0 - 2.0 * flip.view(n, 1))
    return q


@register_loss(
    name="se3_equivariance_defect",
    aliases=["se3_equivariance", "equivariance_defect_se3"],
    domain="agnostic",
)
class SE3EquivarianceDefectLoss(nn.Module):
    r"""Monitoring loss :math:`\|G(\mathrm{Ad}_g u) - \mathrm{Ad}_g G(u)\|`.

    Args:
        num_rotations: Number of random group elements :math:`g` averaged.
        group: ``"SO2"`` (in-plane) or ``"SO3"`` (full 3D rotation).
        reduction: ``"mean"`` or ``"sum"`` over the sampled rotations.
        seed: Deterministic seed for the rotation sampler (exchangeability /
            reproducibility). A per-call ``torch.Generator`` is used so the
            loss does not perturb global RNG state.
        max_angle: Max in-plane angle (radians) for ``SO2`` sampling.
    """

    def __init__(
        self,
        num_rotations: int = 8,
        group: Literal["SO2", "SO3"] = "SO2",
        reduction: str = "mean",
        seed: int = 0,
        max_angle: float = math.pi,
    ) -> None:
        super().__init__()
        if group not in _SUPPORTED_GROUPS:
            # CLAUDE.md #9 — fail loudly on an unknown group.
            raise ValueError(
                f"SE3EquivarianceDefectLoss: unknown group {group!r}; "
                f"supported groups are {_SUPPORTED_GROUPS}."
            )
        if num_rotations < 1:
            raise ValueError("num_rotations must be >= 1.")
        if reduction not in ("mean", "sum"):
            raise ValueError(f"reduction must be mean/sum, got {reduction!r}.")
        self.num_rotations = int(num_rotations)
        self.group = group
        self.reduction = reduction
        self.seed = int(seed)
        self.max_angle = float(max_angle)
        # The sampler is re-seeded with a fixed seed, so every call draws the same
        # rotations; cache them per (device, dtype) instead of a CPU draw + copy each step.
        self._rotation_cache: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}

    @staticmethod
    def _apply_adjoint(u: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
        r"""Apply the pure-rotation adjoint ``Ad_{(R,0)}`` to ``u`` ``[B, L, 6]``.

        ``rot`` is ``[3, 3]``. :math:`\mathrm{Ad}_{(R,0)}(\omega, v)=(R\omega, Rv)`.
        """
        omega = u[..., :3]
        v = u[..., 3:]
        r_omega: torch.Tensor = torch.einsum("ij,blj->bli", rot, omega)
        r_v: torch.Tensor = torch.einsum("ij,blj->bli", rot, v)
        return torch.cat([r_omega, r_v], dim=-1)

    def _sample_rotations(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (device, dtype)
        cached = self._rotation_cache.get(key)
        if cached is None:
            cached = self._draw_rotations(device, dtype)
            self._rotation_cache[key] = cached
        return cached

    def _draw_rotations(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.seed)
        if self.group == "SO2":
            angles = (torch.rand(self.num_rotations, generator=gen) * 2.0 - 1.0) * self.max_angle
            return _random_rotation_2d(angles.to(device=device, dtype=dtype))
        # SO3 — sample on CPU then move (QR is deterministic given the seed).
        rots = _random_rotation_3d(self.num_rotations, torch.device("cpu"), gen)
        return rots.to(device=device, dtype=dtype)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        r"""Compute the mean equivariance defect of ``G`` over random rotations.

        Args:
            prediction: Used as the Lie-algebra input ``u`` ``[B, L, 6]`` when
                ``kwargs["lie_input"]`` is absent (lets the registry probe the
                loss with a single tensor).
            target: Unused (kept for the registry ``(pred, target)`` contract).
            **kwargs:
                equivariant_fn: callable ``G(u) -> [B, L, 6]`` to test. When
                    omitted, defaults to the identity (defect is then exactly 0,
                    which still exercises the full code path under smoke).
                lie_input: explicit ``[B, L, 6]`` trajectory ``u``.

        Returns:
            Scalar defect tensor (non-negative).
        """
        u = kwargs.get("lie_input", prediction)
        if not isinstance(u, torch.Tensor):
            raise TypeError("lie_input / prediction must be a torch.Tensor.")
        if u.dim() != 3 or u.shape[-1] != 6:
            raise ValueError(f"SE3 equivariance defect expects [B, L, 6]; got {tuple(u.shape)}.")

        def _identity(z: torch.Tensor) -> torch.Tensor:
            return z  # identity → zero defect (still exercises the code path)

        fn = kwargs.get("equivariant_fn")
        g: Callable[[torch.Tensor], torch.Tensor]
        if fn is None:
            g = _identity
        elif callable(fn):
            g = fn
        else:
            raise TypeError("equivariant_fn must be callable.")

        rots = self._sample_rotations(u.device, u.dtype)  # [K, 3, 3]

        g_u = g(u)  # G(u)
        defects = []
        for k in range(rots.shape[0]):
            r = rots[k]
            lhs = g(self._apply_adjoint(u, r))  # G(Ad_g u)
            rhs = self._apply_adjoint(g_u, r)  # Ad_g G(u)
            defects.append(torch.linalg.vector_norm(lhs - rhs))
        stacked = torch.stack(defects)
        return stacked.mean() if self.reduction == "mean" else stacked.sum()


__all__ = ["SE3EquivarianceDefectLoss"]
