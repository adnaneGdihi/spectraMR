"""Config for DisentangledVAETrainingStrategy (``config.training.disentangled_vae``).

``physics_source`` says where each row's physics vector comes from:

- ``fixed_t1w_64mt_3t``: the historical pair, T1w at 64 mT for the input and T1w at 3 T
  for the target, whatever the data holds.
- ``paired_field_stamps``: per row, from the loader's ``field_strength_source_t`` /
  ``field_strength_t`` (``data.expose.field_strength_pair`` on ``contrast_aware_paired``)
  and the declared ``data.input_contrast`` / ``data.target_contrast`` names. Validation
  adds the field-leak probe and the swap readings of ``core/metrics/disentanglement.py``.
- ``single_field``: one field, no physics axis. The style comes from the image, the
  translation and anatomy-consistency terms are off, and validation grades the
  self-reconstruction.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from spectramr.config.schemas.strictness import StrictSchema

PhysicsSource = Literal["fixed_t1w_64mt_3t", "paired_field_stamps", "single_field"]


class DisentangledVAEConfig(StrictSchema):
    """Physics-axis source for the disentangled VAE strategy."""

    physics_source: PhysicsSource = Field(
        default="fixed_t1w_64mt_3t",
        description=(
            "Where each row's physics vector comes from: the fixed T1w 64 mT/3 T pair, "
            "the loader's per-row field stamps, or none (single-field self-reconstruction)."
        ),
    )


__all__ = ["DisentangledVAEConfig", "PhysicsSource"]
