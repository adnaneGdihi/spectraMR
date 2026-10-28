"""Optimal-transport-based loss functions.

Moved here from ``src/models/ot/optimal_transport.py`` 2026-05-14 per
``TODO/backlog_ssot_and_layering_cleanup.md`` Phase 5 + CLAUDE.md
canonical-home rule (pitfall #12: ``@register_loss`` lives in
``src/models/losses/``).

The OT primitives — ``sinkhorn_knopp`` (algorithm) and
``VelocityFieldNetwork`` (an nn.Module that parameterises the velocity
field for dynamic OT) — stay in ``src/models/ot/optimal_transport.py``
since they're not losses themselves but math/network building blocks.

References:
- Cuturi, M. (2013). Sinkhorn distances: Lightspeed computation of optimal transport.
- Benamou, J.D. & Brenier, Y. (2000). A computational fluid mechanics solution
  to the OT problem.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.losses.registry import register_loss
from spectramr.models.ot.optimal_transport import (
    VelocityFieldNetwork,
    sinkhorn_knopp,
)


@register_loss("sinkhorn", aliases=["wasserstein", "emd"])
class SinkhornDistance(nn.Module):
    """Sinkhorn (Entropic OT) Distance Loss.

    Computes the Wasserstein distance between two distributions
    represented as batches of samples, using the Sinkhorn algorithm.
    More efficient than exact EMD and supports gradients.

    Args:
        reg: Entropic regularization (higher = faster but less accurate).
        max_iter: Maximum Sinkhorn iterations.
        reduction: ``"mean"``, ``"sum"``, or ``"none"``.
    """

    def __init__(
        self,
        reg: float = 0.1,
        max_iter: int = 100,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.reg = reg
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        cost_fn: str = "l2",
    ) -> torch.Tensor:
        """Compute Sinkhorn distance between sample batches.

        Args:
            x: ``[B, N, D]`` or ``[B, C, H, W]`` samples from distribution P.
            y: ``[B, M, D]`` or ``[B, C, H, W]`` samples from distribution Q.
            cost_fn: ``"l1"``, ``"l2"``, or ``"cosine"``.
        """
        # Flatten spatial dimensions if image input.
        if x.dim() == 4:
            B, C, H, W = x.shape
            x = x.view(B, C, H * W).permute(0, 2, 1)  # [B, N, C]
            y = y.view(B, C, H, W).view(B, C, -1).permute(0, 2, 1)  # [B, M, C]

        B = x.shape[0]
        distances: list[torch.Tensor] = []

        for b in range(B):
            if cost_fn == "l2":
                cost = torch.cdist(x[b], y[b], p=2)
            elif cost_fn == "l1":
                cost = torch.cdist(x[b], y[b], p=1)
            elif cost_fn == "cosine":
                x_norm = F.normalize(x[b], dim=-1)
                y_norm = F.normalize(y[b], dim=-1)
                cost = 1 - x_norm @ y_norm.T
            else:
                raise ValueError(f"Unknown cost function: {cost_fn}")

            _, dist = sinkhorn_knopp(cost, self.reg, self.max_iter)
            distances.append(dist)

        stacked = torch.stack(distances)

        if self.reduction == "mean":
            return stacked.mean()
        if self.reduction == "sum":
            return stacked.sum()
        return stacked


@register_loss("dynamic_ot", aliases=["benamou_brenier", "ot_flow"])
class DynamicOTFlow(nn.Module):
    r"""Dynamic Optimal Transport (Benamou-Brenier).

    Learns a continuous flow from source to target distribution by
    minimising the kinetic energy of the transport:

    .. math::

        \min_{v}\ \int_0^1 \int |v(t,x)|^2 \rho(t,x)\, dx\, dt
        \quad\text{s.t.}\quad
        \partial_t \rho + \nabla\cdot(\rho v) = 0.

    Args:
        in_channels: Number of input channels.
        num_steps: Number of Euler integration steps.
        hidden_dim: Velocity-network hidden dimension.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_steps: int = 10,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.num_steps = num_steps
        self.velocity_net = VelocityFieldNetwork(in_channels, hidden_dim)

    def _euler_path(
        self,
        x0: torch.Tensor,
        t_start: float = 0.0,
        t_end: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Explicit-Euler transport of ``x0``: ``(endpoint, kinetic action)``."""
        dt = (t_end - t_start) / self.num_steps
        x = x0
        total_energy = x0.new_zeros(())
        for i in range(self.num_steps):
            t = torch.tensor(t_start + i * dt, device=x.device)
            v: torch.Tensor = self.velocity_net(x, t)
            total_energy = total_energy + dt * (v**2).mean()
            x = x + dt * v
        return x, total_energy

    def integrate(
        self,
        x0: torch.Tensor,
        t_start: float = 0.0,
        t_end: float = 1.0,
    ) -> torch.Tensor:
        """Integrate the ODE to transport ``x0`` from ``t_start`` to ``t_end``."""
        return self._euler_path(x0, t_start, t_end)[0]

    def compute_kinetic_energy(self, x0: torch.Tensor) -> torch.Tensor:
        """Compute the kinetic-energy action integral along the transport path."""
        return self._euler_path(x0)[1]

    def transport_loss(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        lambda_terminal: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(kinetic + lambda_terminal * terminal MSE, transported)`` from one Euler pass."""
        transported, kinetic_loss = self._euler_path(source)
        terminal_loss = F.mse_loss(transported, target)
        return kinetic_loss + lambda_terminal * terminal_loss, transported

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        lambda_terminal: float = 1.0,
    ) -> torch.Tensor:
        """Kinetic-energy action + terminal MSE matching."""
        return self.transport_loss(source, target, lambda_terminal)[0]


@register_loss("kidot", aliases=["knowledge_informed_ot"])
class KIDOTLoss(nn.Module):
    r"""Knowledge-Informed Dynamic Optimal Transport.

    Extends Dynamic OT with a physics-informed cost:

    .. math::

        c(x, y) = \| A(x) - y_{\text{meas}} \|^2
                + \lambda \| x - y \|_{\Sigma^{-1}}^2

    Args:
        forward_operator: Callable ``A(x)`` mapping image to measurements.
        lambda_physics: Weight for physics consistency.
        lambda_terminal: Weight for terminal matching.
    """

    def __init__(
        self,
        forward_operator: nn.Module | None = None,
        lambda_physics: float = 1.0,
        lambda_terminal: float = 1.0,
        in_channels: int = 1,
        num_steps: int = 10,
    ) -> None:
        super().__init__()
        self.forward_operator = forward_operator
        self.lambda_physics = lambda_physics
        self.lambda_terminal = lambda_terminal
        self.dynamic_ot = DynamicOTFlow(
            in_channels=in_channels,
            num_steps=num_steps,
        )

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        measurements: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Transport loss + optional physics consistency term."""
        transport_loss, transported = self.dynamic_ot.transport_loss(
            source, target, self.lambda_terminal
        )
        if self.forward_operator is not None and measurements is not None:
            predicted_meas = self.forward_operator(transported)
            physics_loss = F.mse_loss(predicted_meas, measurements)
            transport_loss = transport_loss + self.lambda_physics * physics_loss
        return transport_loss


__all__ = ["DynamicOTFlow", "KIDOTLoss", "SinkhornDistance"]
