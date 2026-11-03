"""Beltrami-parameterised MRF pulse-train generator (MRF §4).

The design variable is a per-excitation Beltrami coefficient ``mu``; the emitted pulse is the
image of a constant base train under :func:`beltrami_pulse_curve`, so ``|mu| <= k_max`` bounds
the per-excitation deviation of flip angle and TR. The input batch only sets how many copies of
the single design are returned; the design does not depend on image content.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from spectramr.infrastructure.physics.acquisition.mrf_acquisition_manifold import (
    beltrami_pulse_curve,
)
from spectramr.models.registry import register_model


@register_model(
    "mrf_pulse_design",
    training_mode="crlb_mrf_pulse_design",
    spatial_dims=(1,),
    strict_kwargs=True,
)
class MRFPulseDesign(nn.Module):
    """Learnable ``[T, 2]`` (flip angle rad, TR ms) pulse train under a Beltrami bound."""

    def __init__(
        self,
        n_tr: int,
        alpha_init_rad: float,
        tr_init_ms: float,
        k_max: float = 0.9,
    ) -> None:
        super().__init__()
        if n_tr < 2:
            raise ValueError(f"n_tr must be >= 2; got {n_tr}")
        if not 0.0 < k_max < 1.0:
            raise ValueError(f"k_max must lie in (0, 1); got {k_max}")
        if alpha_init_rad <= 0 or tr_init_ms <= 0:
            raise ValueError("alpha_init_rad and tr_init_ms must be > 0")
        self.n_tr = int(n_tr)
        self.k_max = float(k_max)
        self.mu_raw = nn.Parameter(torch.zeros(self.n_tr, 2))
        self.register_buffer("base_alpha", torch.full((self.n_tr,), float(alpha_init_rad)))
        self.register_buffer("base_tr", torch.full((self.n_tr,), float(tr_init_ms)))

    def design(self) -> torch.Tensor:
        """The current ``[T, 2]`` pulse train."""
        mu = torch.complex(self.mu_raw[:, 0], self.mu_raw[:, 1])
        return beltrami_pulse_curve(
            mu, base_alpha=self.base_alpha, base_TR=self.base_tr, k_max=self.k_max
        )

    @property
    def min_tr_ms(self) -> float:
        """Smallest TR the Beltrami bound lets the design reach (``Im mu >= -k_max``)."""
        return float(self.base_tr.min()) * (1.0 - 0.5 * self.k_max)

    def initial_design(self) -> torch.Tensor:
        """The ``mu = 0`` base train the optimisation starts from."""
        return torch.stack([self.base_alpha, self.base_tr], dim=-1)

    def forward(self, x: torch.Tensor | None = None, *args: Any, **kwargs: Any) -> torch.Tensor:
        batch = x.shape[0] if torch.is_tensor(x) else 1
        return self.design().unsqueeze(0).expand(batch, -1, -1)


__all__ = ["MRFPulseDesign"]
