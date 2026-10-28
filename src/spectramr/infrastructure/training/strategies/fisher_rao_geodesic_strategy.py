"""Fisher-Rao geodesic cross-field translation training (MICCAI MRIxFields2026, B-3.4).

Trains a :class:`~spectramr.models.generators.fisher_rao_geodesic_net.FisherRaoGeodesicNet` to
translate any source field to any target field by per-pixel Fisher-Rao geodesic shooting on
the Bernoulli information manifold, via L1. The information geometry (the spherical geodesic
step) is STRUCTURAL on the model, so the one-knob ablation
(``model_kwargs.use_fisher_rao_geometry``) lives there. Any-to-any (Task 3): conditions on
BOTH source and target field.
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch
import torch.nn.functional as F

from .reconstruction import ReconstructionTrainingStrategy


def compute_fisher_rao_loss(
    model: Any, batch: dict[str, Any], *, lambda_l1: float = 1.0
) -> dict[str, torch.Tensor]:
    """L1 reconstruction of the target via the Fisher-Rao geodesic-shooting translator."""
    x = batch["input"]
    y = batch["target"]
    b_s = batch["field_strength"].reshape(-1).float()
    b_t = batch["field_strength_target"].reshape(-1).float()
    pred = model(
        x,
        field_strength=b_s,
        field_strength_target=b_t,
        contrast_id=batch.get("contrast_id"),
    )
    loss_l1 = F.l1_loss(pred, y)
    return {"loss_total": lambda_l1 * loss_l1, "loss_l1": loss_l1}


class FisherRaoGeodesicStrategy(ReconstructionTrainingStrategy):
    """Train the Fisher-Rao geodesic cross-field translator (B-3.4)."""

    #: Loss ownership (mrixfields review 2026-09-03): computes its objective inline and folds NOTHING else: any other declared image loss is a decoy.
    inline_losses: ClassVar[frozenset[str] | None] = frozenset({"l1"})
    folds_image_losses: ClassVar[bool | None] = False

    def _setup_strategy_specific_components(self) -> None:
        self._verify_strategy_config(expected_modes=("fisher_rao_geodesic", "reconstruction"))
        if getattr(self, "logging_service", None):
            self._log_config_features(self.logging_service)
        cfg = getattr(self.config.training, "fisher_rao_geodesic", None)
        self._fr_lambda_l1 = self._declared_inline_l1_weight()

    def _compute_losses_impl(
        self,
        input_batch: Any = None,
        target_batch: Any = None,
        epoch: int = 0,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        batch = self._resolve_legacy_batch(input_batch, kwargs)
        if batch is None or not hasattr(batch, "get"):  # dict OR TrainingBatch (canonical pipeline)
            raise ValueError(
                "FisherRaoGeodesicStrategy requires a mapping batch (dict/TrainingBatch) from the "
                f"mrixfields dataset; got {type(batch)!r}."
            )
        return compute_fisher_rao_loss(
            self.env.generator, batch, lambda_l1=getattr(self, "_fr_lambda_l1", 1.0)
        )

    def validation_step(
        self,
        input_batch: Any,
        target_batch: Any,
        field_strength: Any = None,
        field_strength_target: Any = None,
        contrast_id: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Fold the per-sample source + target fields + contrast id into batch_context."""
        bc = kwargs.pop("batch_context", None) or {}
        if field_strength is not None:
            bc["field_strength"] = field_strength
        if field_strength_target is not None:
            bc["field_strength_target"] = field_strength_target
        if contrast_id is not None:
            bc["contrast_id"] = contrast_id
        return super().validation_step(input_batch, target_batch, batch_context=bc, **kwargs)

    def _validation_forward(
        self, input_batch: Any, batch_context: dict[str, Any], **kwargs: Any
    ) -> torch.Tensor:
        """Validation = Fisher-Rao geodesic shooting from source to target field; clamp [0,1]."""
        x0 = batch_context.get("input", input_batch)
        b_s = batch_context.get("field_strength", kwargs.get("field_strength"))
        b_t = batch_context.get("field_strength_target", kwargs.get("field_strength_target"))
        if b_s is None or b_t is None:
            raise ValueError(
                "FisherRaoGeodesicStrategy validation requires per-sample 'field_strength' AND "
                "'field_strength_target'. Got "
                f"field_strength={b_s is not None}, field_strength_target={b_t is not None}."
            )
        b_s = torch.as_tensor(b_s, device=x0.device).reshape(-1).float()
        b_t = torch.as_tensor(b_t, device=x0.device).reshape(-1).float()
        return self.env.generator(
            x0,
            field_strength=b_s,
            field_strength_target=b_t,
            contrast_id=batch_context.get("contrast_id", kwargs.get("contrast_id")),
        ).clamp(0.0, 1.0)
