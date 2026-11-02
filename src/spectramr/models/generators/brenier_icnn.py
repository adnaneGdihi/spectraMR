"""Source-conditioned Brenier-form transport map (MICCAI MRIxFields2026, B-1.5).

A field-conditioned input-convex potential ``psi_b(x)``; its gradient has the FORM of a
Brenier map, so the synthesised 7T image ``T_b(x) = x + map_scale * grad_x psi_b(x)`` is a
convex-by-construction MONOTONE transport map (strictly monotone, near-identity at init). The
source field ``b`` conditions the potential (convexity-preserving gamma>=0/beta FiLM on the
convex path), so each source field has its own convex potential (Task 1); distinct from the
velocity ODE of B-3.1 and the learned latent render of B-3.8.

HONEST SCOPE: this is trained by paired L1 (the strategy), NOT the semi-dual OT objective.
The ARCHITECTURE guarantees a monotone (gradient-of-convex) map; it does NOT guarantee the
trained map is the W2-OPTIMAL Monge map between the source and 7T MARGINALS (that needs the OT
dual). And the map is curl-free (gradient of a scalar potential), which limits the
spatially-varying contrast changes it can express. We claim a convex-by-construction monotone
transport map, not measure/marginal optimality.

``model_kwargs.enforce_convexity=False`` drops the non-negative-weight constraint — the
potential is then a generic CNN whose gradient is NOT a guaranteed OT map (the one-knob
ablation, isolating the Brenier/convexity guarantee). Magnitude-only (1->1).

``model_kwargs.intensity_adaptation`` (``none`` | ``transfer``) lets the map adapt
INTENSITY/CONTRAST — the residual on co-registered, normalised source/7T pairs. ``none`` (default)
is the displacement map alone, which can realise an EXPANSIVE intensity change; ``transfer``
composes a field-conditioned monotone intensity transfer (``gain(b) < 1`` can COMPRESS contrast)
while staying a monotone Brenier map. See :class:`InputConvexNeuralNetwork` for the math and the
one-knob arm ``b15_brenier_intensity_transfer`` (vs ``b15_brenier_map``).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from spectramr.models.blocks.contrast_conditioning import (
    build_contrast_sequence,
    contrast_sequence_dim,
)
from spectramr.models.blocks.input_convex_neural_network import InputConvexNeuralNetwork
from spectramr.models.registry import register_model


@register_model(name="brenier_icnn", training_mode="brenier_synthesis")
class BrenierICNN(nn.Module):
    """Source-field-conditioned Brenier OT map to 7T (B-1.5)."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        hidden_channels: int = 32,
        n_layers: int = 3,
        map_scale_init: float = 0.1,
        enforce_convexity: bool = True,
        intensity_adaptation: str = "none",
        use_contrast_conditioning: bool = False,
        num_contrasts: int = 3,
    ) -> None:
        super().__init__()
        if in_channels != 1 or out_channels != 1:
            raise ValueError(
                f"BrenierICNN is magnitude-only (1->1); got in={in_channels}, out={out_channels}."
            )
        self.use_contrast_conditioning = bool(use_contrast_conditioning)
        self.num_contrasts = int(num_contrasts)
        # The convex potential conditions on the source field (field_dim=1); when contrast
        # conditioning is on, a per-sample contrast one-hot is appended (field_dim widens).
        # The field enters ONLY through the ICNN's gamma>=0 softplus FiLM, so convexity in x
        # is preserved for any field_dim (see InputConvexNeuralNetwork).
        field_dim = contrast_sequence_dim(1, self.num_contrasts, self.use_contrast_conditioning)
        # intensity_adaptation is validated inside the ICNN block (raises on an unknown value).
        self.icnn = InputConvexNeuralNetwork(
            in_channels=1,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            field_dim=field_dim,  # source field (+ contrast one-hot) conditions the potential
            map_scale_init=map_scale_init,
            enforce_convexity=enforce_convexity,
            intensity_adaptation=intensity_adaptation,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        field_strength: torch.Tensor,
        contrast_id: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        # field_strength REQUIRED (no default) so the Tier-2 probe injects it. It is the
        # SOURCE field that conditions the convex potential psi_b.
        if torch.is_complex(x):
            raise ValueError("BrenierICNN expects magnitude (real) input.")
        b = field_strength.reshape(-1, 1).float()
        # Append the contrast one-hot to the field-conditioning vector when on (#15/#9).
        field_cond = build_contrast_sequence(
            b,
            contrast_id,
            self.num_contrasts,
            self.use_contrast_conditioning,
            batch_size=x.shape[0],
            device=x.device,
            dtype=b.dtype,
        )
        return self.icnn.brenier_map(x, field_cond)  # T_b(x) = x + map_scale * grad psi_b(x)
