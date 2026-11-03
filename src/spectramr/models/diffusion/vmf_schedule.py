r"""Concentration schedule and forward process for von Mises-Fisher diffusion.

The latent is a field of unit vectors: at every spatial position the channel
vector :math:`z_0 \in S^{C-1}`. The forward process replaces Gaussian noising
with vMF noising about the clean direction,

.. math::

    q(z_t \mid z_0) = \mathrm{vMF}(z_t;\, z_0,\, \kappa_t),

with :math:`\kappa_t` decreasing from ``kappa_max`` (``t = 0``, nearly clean)
to ``kappa_min`` (``t = T - 1``, nearly uniform on the sphere). The terminal
prior is drawn exactly uniform. Sampling reuses the Wood rejection sampler of
:class:`~spectramr.models.blocks.vae.von_mises_fisher.VonMisesFisher`.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from spectramr.models.blocks.vae.von_mises_fisher import VonMisesFisher

#: The advertised ``kappa_schedule`` values; anything else raises.
KAPPA_SCHEDULES: tuple[str, ...] = ("log_linear", "linear")


def _kappa_table(timesteps: int, kappa_max: float, kappa_min: float, schedule: str) -> Tensor:
    """Descending concentrations, index ``t`` -> ``kappa_t``."""
    if schedule == "log_linear":
        return torch.logspace(math.log10(kappa_max), math.log10(kappa_min), timesteps)
    return torch.linspace(kappa_max, kappa_min, timesteps)


class VMFNoiseSchedule(nn.Module):
    """vMF forward process on the channel sphere of a ``(B, C, *spatial)`` latent.

    Args:
        timesteps: Number of noise levels ``T``.
        kappa_max: Concentration at ``t = 0``.
        kappa_min: Concentration at ``t = T - 1``; must be positive.
        schedule: One of :data:`KAPPA_SCHEDULES`.
    """

    def __init__(
        self,
        timesteps: int = 1000,
        kappa_max: float = 1.0e4,
        kappa_min: float = 1.0e-2,
        schedule: str = "log_linear",
    ) -> None:
        super().__init__()
        if schedule not in KAPPA_SCHEDULES:
            raise ValueError(
                f"Unknown kappa_schedule {schedule!r}; expected one of {list(KAPPA_SCHEDULES)}."
            )
        if int(timesteps) < 2:
            raise ValueError(f"vMF diffusion needs timesteps >= 2, got {timesteps}.")
        if not 0.0 < float(kappa_min) < float(kappa_max):
            raise ValueError(
                f"vMF diffusion needs 0 < kappa_min < kappa_max, got "
                f"kappa_min={kappa_min}, kappa_max={kappa_max}."
            )
        self.timesteps = int(timesteps)
        self.kappa_max = float(kappa_max)
        self.kappa_min = float(kappa_min)
        self.schedule = schedule
        table = _kappa_table(self.timesteps, self.kappa_max, self.kappa_min, schedule)
        self.register_buffer("kappas", table, persistent=False)

    def q_sample(self, z0: Tensor, t: Tensor) -> Tensor:
        r"""Draw :math:`z_t \sim \mathrm{vMF}(z_0, \kappa_t)` per position.

        Args:
            z0: Unit-norm (along dim 1) latent ``(B, C, *spatial)``.
            t: Integer timesteps ``(B,)``.
        """
        b, c = z0.shape[:2]
        spatial = z0.shape[2:]
        mu = z0.movedim(1, -1).reshape(-1, c)
        kappa = self.kappas.to(z0.dtype)[t].view(b, *([1] * len(spatial)))
        kappa = kappa.expand(b, *spatial).reshape(-1)
        zt = VonMisesFisher(mu, kappa, validate=False).rsample()
        return zt.reshape(b, *spatial, c).movedim(-1, 1)

    @staticmethod
    def prior_sample(
        shape: tuple[int, ...], device: torch.device, dtype: torch.dtype = torch.float32
    ) -> Tensor:
        """Exactly uniform unit vectors along dim 1 of ``shape``."""
        return F.normalize(torch.randn(shape, device=device, dtype=dtype), dim=1)

    def describe(self) -> dict[str, Any]:
        """Resolved schedule, for provenance."""
        return {
            "timesteps": self.timesteps,
            "kappa_max": self.kappa_max,
            "kappa_min": self.kappa_min,
            "kappa_schedule": self.schedule,
        }


__all__ = ["KAPPA_SCHEDULES", "VMFNoiseSchedule"]
