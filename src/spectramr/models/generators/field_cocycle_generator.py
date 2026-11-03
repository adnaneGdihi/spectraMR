"""Cocycle-consistent unified cross-field operator (MICCAI MRIxFields2026, idea 4.2).

A thin subclass of :class:`AnatomyFieldRenderer` that IS the shared-latent
factorisation of Theorem 6: ``G(x; s, t) = R(E(x, b_s), b_t)`` with a source-field-aware
encoder ``E = Phi`` (the universal field-canonicaliser) and a continuous-field
renderer ``R(., b_t) = Psi_t``. The single-model / anti-ensemble property required
by Task 3 is therefore *structural*, not argued — recorded here as the registry
attribute ``is_unified_single_model = True`` that the ``field_cocycle_single_model``
Tier-1 guard reads. Beyond the parent it exposes ``last_canonical_repr`` = ``Phi(x, b_s)``
so the Tier-2 mechanism-fires probe can assert the canonicaliser was populated.

No new architecture: reuse keeps the parent's ``FieldFiLMBlock`` conditioning and
its magnitude-only 1->1 contract (pitfall #12 — one home per component).
"""

from __future__ import annotations

from typing import Any

import torch

from spectramr.models.generators.cross_field_renderer import AnatomyFieldRenderer
from spectramr.models.registry import register_model


@register_model(
    name="field_cocycle_generator",
    training_mode="field_cocycle",
    supports_contrast_conditioning=True,
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    requires_paired_data=True,
)
class FieldCocycleGenerator(AnatomyFieldRenderer):
    """Encode-once / render-anywhere generator with the single-model contract.

    Identical forward behaviour to :class:`AnatomyFieldRenderer`; adds only the
    anti-ensemble registry attribute and the ``last_canonical_repr`` exposure
    contract used by the audit and by the cocycle metric.
    """

    #: Read by the ``field_cocycle_single_model`` Tier-1 guard: one conditioned
    #: generator realises every ordered field pair (no per-field routing submodule),
    #: which is exactly the Task-3 anti-ensemble requirement.
    is_unified_single_model: bool = True

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        latent_channels: int = 64,
        width: int = 64,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            latent_channels=latent_channels,
            width=width,
        )
        # Exposure contract: the most recent canonicaliser output Phi(x) = E(x).
        self.last_canonical_repr: torch.Tensor | None = None

    def encode(self, x: torch.Tensor, field_strength: torch.Tensor) -> torch.Tensor:
        """Canonicaliser ``Phi(x; b)`` of ``x`` at field ``b``; stamps ``last_canonical_repr``."""
        q = super().encode(x, field_strength)
        # Detached snapshot for exposure/audit (avoids retaining the graph across
        # the many encode calls a cocycle step makes).
        self.last_canonical_repr = q.detach()
        return q

    def forward(
        self,
        x: torch.Tensor,
        *,
        field_strength: torch.Tensor,
        field_strength_target: torch.Tensor,
        contrast_id: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        return super().forward(
            x,
            field_strength=field_strength,
            field_strength_target=field_strength_target,
            contrast_id=contrast_id,
            **kwargs,
        )
