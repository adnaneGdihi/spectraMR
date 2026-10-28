"""McCann single-potential field-path training (MICCAI MRIxFields2026, B-3.9).

Trains a :class:`~spectramr.models.generators.mccann_geodesic_icnn.McCannGeodesicICNN` (one
global Brenier potential + a learnable monotone path function) to translate any source field
to any target field along a single Wasserstein-2 geodesic, via L1. The convexity (Brenier
guarantee) is structural on the model, so the one-knob ablation
(``model_kwargs.enforce_convexity``) lives there. Any-to-any (Task 3): conditions on BOTH the
source and target field so the McCann displacement increment ``t(b_t)-t(b_s)`` is exercised.
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch
import torch.nn.functional as F

from .reconstruction import ReconstructionTrainingStrategy


def compute_mccann_loss(
    model: Any, batch: dict[str, Any], *, lambda_l1: float = 1.0
) -> dict[str, torch.Tensor]:
    """L1 reconstruction of the target via the McCann displacement-interpolation map."""
    x = batch["input"]
    y = batch["target"]
    b_s = batch["field_strength"].reshape(-1).float()
    b_t = batch["field_strength_target"].reshape(-1).float()
    pred = model(x, field_strength=b_s, field_strength_target=b_t)
    loss_l1 = F.l1_loss(pred, y)
    return {"loss_total": lambda_l1 * loss_l1, "loss_l1": loss_l1}


class McCannFieldPathStrategy(ReconstructionTrainingStrategy):
    """Train the single-potential McCann geodesic field path (B-3.9)."""

    #: Loss ownership (mrixfields review 2026-09-03): computes its objective inline and folds NOTHING else: any other declared image loss is a decoy.
    inline_losses: ClassVar[frozenset[str] | None] = frozenset({"l1"})
    folds_image_losses: ClassVar[bool | None] = False

    def _setup_strategy_specific_components(self) -> None:
        self._verify_strategy_config(expected_modes=("mccann_field_path", "reconstruction"))
        if getattr(self, "logging_service", None):
            self._log_config_features(self.logging_service)
        cfg = getattr(self.config.training, "mccann_field_path", None)
        self._mc_lambda_l1 = self._declared_inline_l1_weight()

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
                "McCannFieldPathStrategy requires a mapping batch (dict/TrainingBatch) from the "
                f"mrixfields dataset; got {type(batch)!r}."
            )
        return compute_mccann_loss(
            self.env.generator, batch, lambda_l1=getattr(self, "_mc_lambda_l1", 1.0)
        )

    def validation_step(
        self,
        input_batch: Any,
        target_batch: Any,
        field_strength: Any = None,
        field_strength_target: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Fold the per-sample source + target fields into batch_context (gated seam)."""
        bc = kwargs.pop("batch_context", None) or {}
        if field_strength is not None:
            bc["field_strength"] = field_strength
        if field_strength_target is not None:
            bc["field_strength_target"] = field_strength_target
        return super().validation_step(input_batch, target_batch, batch_context=bc, **kwargs)

    def _validation_forward(
        self, input_batch: Any, batch_context: dict[str, Any], **kwargs: Any
    ) -> torch.Tensor:
        """Validation = McCann displacement interpolation from source to target field; clamp."""
        x0 = batch_context.get("input", input_batch)
        b_s = batch_context.get("field_strength", kwargs.get("field_strength"))
        b_t = batch_context.get("field_strength_target", kwargs.get("field_strength_target"))
        if b_s is None or b_t is None:
            raise ValueError(
                "McCannFieldPathStrategy validation requires per-sample 'field_strength' AND "
                "'field_strength_target'. Got "
                f"field_strength={b_s is not None}, field_strength_target={b_t is not None}."
            )
        b_s = torch.as_tensor(b_s, device=x0.device).reshape(-1).float()
        b_t = torch.as_tensor(b_t, device=x0.device).reshape(-1).float()
        return self.env.generator(x0, field_strength=b_s, field_strength_target=b_t).clamp(0.0, 1.0)
