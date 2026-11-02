"""SIREN INR for k-space coordinate networks.

Plan: TODO/backlog_ulf_radiologist_acceptance_roadmap.md §ULF-PR-12.

A small SIREN MLP with the Sitzmann-et-al weight initialisation:
  - first layer: ``U(−1/n, 1/n)`` scaled by ``ω0``
  - hidden layers: ``U(−sqrt(6/n)/ω, sqrt(6/n)/ω)``

Used by :mod:`coord_kspace_gen_strategy` and as a generic INR for any
(coords → values) regression in the codebase.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SIRENLayer(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, omega: float = 30.0, is_first: bool = False
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.omega = omega
        with torch.no_grad():
            if is_first:
                bound = 1.0 / in_features
            else:
                bound = math.sqrt(6.0 / in_features) / omega
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega * self.linear(x))


class SIRENNetwork(nn.Module):
    """N-layer SIREN: in_features → hidden* → out_features."""

    def __init__(
        self,
        in_features: int = 2,
        hidden_features: int = 256,
        hidden_layers: int = 4,
        out_features: int = 2,
        omega_first: float = 30.0,
        omega: float = 30.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            SIRENLayer(in_features, hidden_features, omega_first, is_first=True)
        ]
        for _ in range(hidden_layers):
            layers.append(SIRENLayer(hidden_features, hidden_features, omega))
        layers.append(nn.Linear(hidden_features, out_features))
        self.net = nn.Sequential(*layers)
        # Initialise final linear so first epoch of training is well-scaled.
        with torch.no_grad():
            bound = math.sqrt(6.0 / hidden_features) / omega
            self.net[-1].weight.uniform_(-bound, bound)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.net(coords)


def kspace_coordinate_grid(height: int, width: int, device: torch.device) -> torch.Tensor:
    """``[H*W, 2]`` k-space coordinates ``(k_x, k_y)`` on ``[-1, 1]``, row-major over ``(H, W)``.

    The one convention every SIREN-over-k-space consumer reads, so a field fitted by
    one strategy indexes the same frequencies as another.
    """
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, height, device=device),
        torch.linspace(-1, 1, width, device=device),
        indexing="ij",
    )
    return torch.stack([xx.flatten(), yy.flatten()], dim=-1)


__all__ = ["SIRENLayer", "SIRENNetwork", "kspace_coordinate_grid"]
