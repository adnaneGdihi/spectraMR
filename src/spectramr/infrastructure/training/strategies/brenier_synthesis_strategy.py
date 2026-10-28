"""Source-conditioned Brenier OT-map synthesis training (MICCAI MRIxFields2026, B-1.5).

Trains a :class:`~spectramr.models.generators.brenier_icnn.BrenierICNN` (a source-field-
conditioned input-convex potential whose gradient is the Brenier optimal-transport map) to
synthesise the 7T target from the source via L1. The convexity (Brenier guarantee) is
structural on the model, so the one-knob ablation (``model_kwargs.enforce_convexity``) lives
there. Conditions on the SOURCE field (varies on Task 1 -> genuinely load-bearing).
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch
import torch.nn.functional as F

from .reconstruction import ReconstructionTrainingStrategy


def compute_brenier_loss(
    model: Any, batch: dict[str, Any], *, lambda_l1: float = 1.0
) -> dict[str, torch.Tensor]:
    """L1 reconstruction of the 7T target via the source-conditioned Brenier map."""
    x = batch["input"]
    y = batch["target"]
    b_s = batch["field_strength"].reshape(-1).float()  # SOURCE field conditions psi_b
    pred = model(x, field_strength=b_s, contrast_id=batch.get("contrast_id"))
    loss_l1 = F.l1_loss(pred, y)
    return {"loss_total": lambda_l1 * loss_l1, "loss_l1": loss_l1}


class BrenierSynthesisStrategy(ReconstructionTrainingStrategy):
    """Train a source-conditioned Brenier OT map to 7T (B-1.5)."""

    #: Loss ownership (mrixfields review 2026-09-03): computes its objective inline and folds NOTHING else: any other declared image loss is a decoy.
    inline_losses: ClassVar[frozenset[str] | None] = frozenset({"l1"})
    folds_image_losses: ClassVar[bool | None] = False

    def _setup_strategy_specific_components(self) -> None:
        self._verify_strategy_config(expected_modes=("brenier_synthesis", "reconstruction"))
        if getattr(self, "logging_service", None):
            self._log_config_features(self.logging_service)
        cfg = getattr(self.config.training, "brenier_synthesis", None)
        self._br_lambda_l1 = self._declared_inline_l1_weight()

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
                "BrenierSynthesisStrategy requires a mapping batch (dict/TrainingBatch) from the "
                f"mrixfields dataset; got {type(batch)!r}."
            )
        return compute_brenier_loss(
            self.env.generator, batch, lambda_l1=getattr(self, "_br_lambda_l1", 1.0)
        )

    def validation_step(
        self,
        input_batch: Any,
        target_batch: Any,
        field_strength: Any = None,
        contrast_id: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Fold the per-sample SOURCE field + contrast id into batch_context (gated seam)."""
        bc = kwargs.pop("batch_context", None) or {}
        if field_strength is not None:
            bc["field_strength"] = field_strength
        if contrast_id is not None:
            bc["contrast_id"] = contrast_id
        return super().validation_step(input_batch, target_batch, batch_context=bc, **kwargs)

    def _validation_forward(
        self, input_batch: Any, batch_context: dict[str, Any], **kwargs: Any
    ) -> torch.Tensor:
        """Validation = apply the Brenier map to the source at its source field; clamp [0,1]."""
        x0 = batch_context.get("input", input_batch)
        b_s = batch_context.get("field_strength", kwargs.get("field_strength"))
        if b_s is None:
            raise ValueError(
                "BrenierSynthesisStrategy validation requires per-sample 'field_strength' "
                "(the source field). Got None."
            )
        b_s = torch.as_tensor(b_s, device=x0.device).reshape(-1).float()
        return self.env.generator(
            x0,
            field_strength=b_s,
            contrast_id=batch_context.get("contrast_id", kwargs.get("contrast_id")),
        ).clamp(0.0, 1.0)
