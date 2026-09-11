"""Monotone one-to-many field-ordered generator (MICCAI MRIxFields2026, B-2.6).

Renders the target image as a function of the continuous field coordinate
``tau = log10(B0)`` that is **monotone non-decreasing in the field by construction**:

    output(x, tau) = base(x) + sum_k softplus(coef_k(x)) * phi_k(tau)

where ``phi_k`` are FIXED monotone-non-decreasing sigmoid bases over the field range
and ``coef_k(x)`` are per-pixel coefficients. Non-negative coefficients (softplus) times
monotone bases give ``d/dtau output = sum_k softplus(coef_k) * phi_k'(tau) >= 0`` — a
cumulative-residual head with a monotone parameterisation. One source therefore renders
a field-ordered family (one-to-many) whose values never invert as the field rises.

SCOPE OF THE CLAIM (honest): the monotonicity is an architectural INDUCTIVE BIAS, not a
claim that pixel intensity is physically monotone in field. On per-image-normalised
([0,1]) MNI data the absolute "higher field is brighter" ordering is erased and
cross-field translation is a CONTRAST change (tissue-pair contrast can even invert with
field), so a per-pixel monotone-in-field prior may help (smooth field-ordered
regularisation) or hurt — that is exactly the empirical question the one-knob
``enforce_monotone`` ablation answers. ``enforce_monotone=False`` uses raw (signed)
coefficients (the unconstrained control). Magnitude-only (1->1); raises on complex.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from spectramr.models.registry import register_model


def _conv_block(cin: int, cout: int) -> nn.Sequential:
    g = min(8, cout)
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.GroupNorm(g, cout),
        nn.SiLU(),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.GroupNorm(g, cout),
        nn.SiLU(),
    )


@register_model(name="monotone_field_ordered_generator", training_mode="monotone_field")
class MonotoneFieldOrderedGenerator(nn.Module):
    """Field-ordered generator monotone in tau=log10(B0) by construction (B-2.6)."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        width: int = 48,
        n_bases: int = 4,
        tau_min: float = -1.0,  # log10(0.1 T)
        tau_max: float = 0.85,  # ~log10(7 T)
        enforce_monotone: bool = True,
    ) -> None:
        super().__init__()
        if in_channels != 1 or out_channels != 1:
            raise ValueError(
                "MonotoneFieldOrderedGenerator is magnitude-only (1->1); "
                f"got in={in_channels}, out={out_channels}."
            )
        if n_bases < 1:
            raise ValueError(f"n_bases must be >= 1; got {n_bases}")
        if not (tau_max > tau_min):
            # The monotone-by-construction guarantee requires increasing bases, i.e.
            # ramp_width > 0, i.e. tau_max > tau_min. A swapped/typo'd range would
            # silently INVERT the render (#9 silent invariant violation).
            raise ValueError(
                f"tau_max ({tau_max}) must be > tau_min ({tau_min}) so the field bases "
                "are monotone increasing."
            )
        self.n_bases = int(n_bases)
        self.enforce_monotone = bool(enforce_monotone)
        self.encoder = _conv_block(in_channels, width)
        self.base_head = nn.Conv2d(width, out_channels, 1)
        self.coef_head = nn.Conv2d(width, self.n_bases, 1)
        anchors = torch.linspace(tau_min, tau_max, self.n_bases)
        self.register_buffer("anchors", anchors)
        # Sigmoid steepness: one anchor-spacing (so adjacent bases overlap smoothly).
        self.ramp_width = float((tau_max - tau_min) / max(self.n_bases - 1, 1)) or 1.0

    def _phi(self, tau: torch.Tensor) -> torch.Tensor:
        """Monotone non-decreasing bases phi_k(tau) in (0,1): ``[B, n_bases]``."""
        t = tau.reshape(-1, 1).float()
        anchors = self.anchors.reshape(1, -1).to(t.device)  # device-safe (CUDA)
        return torch.sigmoid((t - anchors) / self.ramp_width)

    def forward(self, x: torch.Tensor, *, field_strength: torch.Tensor, **_: Any) -> torch.Tensor:
        if torch.is_complex(x):
            raise ValueError("MonotoneFieldOrderedGenerator expects magnitude input.")
        tau = torch.log10(field_strength.reshape(-1).float().clamp_min(1e-3))  # [B]
        h = self.encoder(x)
        base = self.base_head(h)  # [B, 1, H, W]
        coef = self.coef_head(h)  # [B, n_bases, H, W]
        if self.enforce_monotone:
            coef = F.softplus(coef)  # non-negative => monotone in tau
        phi = self._phi(tau).view(tau.shape[0], self.n_bases, 1, 1)
        incr = (coef * phi).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        return base + incr
