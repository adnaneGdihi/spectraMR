"""Bloch quantitative-parameter-bottleneck training (MICCAI, B-1.8).

Trains :class:`BlochFieldBottleneck` to render the target at its field via L1; the
field-dependent rendering (Bottomley T1 dispersion + SPGR) is structural in the model,
so the one-knob ablation (``model_kwargs.use_field_dispersion``) lives on the model.
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch
import torch.nn.functional as F

from .reconstruction import ReconstructionTrainingStrategy


def compute_bloch_field_loss(
    model: Any, batch: dict[str, Any], *, lambda_l1: float = 1.0
) -> dict[str, torch.Tensor]:
    """L1 reconstruction of the target from the source via the Bloch-bottleneck render."""
    x = batch["input"]
    y = batch["target"]
    b_t = batch["field_strength_target"].reshape(-1).float()
    pred = model(x, field_strength=b_t)
    loss_l1 = F.l1_loss(pred, y)
    return {
        "loss_total": lambda_l1 * loss_l1,
        "loss_l1": loss_l1,
        # Grad-carrying pair for the loss-SSOT seam (popped in _compute_losses_impl).
        "prediction": pred,
        "target_image": y,
    }


class BlochFieldStrategy(ReconstructionTrainingStrategy):
    """Train a Bloch quantitative-parameter bottleneck for cross-field render (B-1.8)."""

    #: Loss ownership (mrixfields review 2026-09-03): folds the builder's other declared entries (calls the fold or the parent path).
    inline_losses: ClassVar[frozenset[str] | None] = frozenset({"l1"})
    folds_image_losses: ClassVar[bool | None] = True

    def _setup_strategy_specific_components(self) -> None:
        self._verify_strategy_config(expected_modes=("bloch_field", "reconstruction"))
        if getattr(self, "logging_service", None):
            self._log_config_features(self.logging_service)
        cfg = getattr(self.config.training, "bloch_field", None)
        self._bf_lambda_l1 = self._declared_inline_l1_weight()

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
                "BlochFieldStrategy requires a mapping batch (dict/TrainingBatch) from the mrixfields dataset; "
                f"got {type(batch)!r}."
            )
        out = compute_bloch_field_loss(
            self.env.generator, batch, lambda_l1=getattr(self, "_bf_lambda_l1", 1.0)
        )
        # Loss-SSOT seam: fold declarative image losses (hfen/ms_ssim) onto the inline
        # objective; the inline l1 is skipped by the seam (no double-count).
        pred = out.pop("prediction", None)
        target_image = out.pop("target_image", None)
        if pred is not None and target_image is not None:
            aux = self._apply_builder_image_losses(pred, target_image, out)
            if aux is not None:
                out["loss_total"] = out["loss_total"] + aux
            self._last_prediction = pred
            self._last_target = target_image
        return out

    def validation_step(
        self,
        input_batch: Any,
        target_batch: Any,
        field_strength_target: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Fold the per-sample target field into batch_context (gated seam)."""
        bc = kwargs.pop("batch_context", None) or {}
        if field_strength_target is not None:
            bc["field_strength_target"] = field_strength_target
        return super().validation_step(input_batch, target_batch, batch_context=bc, **kwargs)

    def _validation_forward(
        self, input_batch: Any, batch_context: dict[str, Any], **kwargs: Any
    ) -> torch.Tensor:
        """Validation = render the target image at the validation sample's field."""
        x0 = batch_context.get("input", input_batch)
        b_t = batch_context.get("field_strength_target", kwargs.get("field_strength_target"))
        if b_t is None:
            raise ValueError(
                "BlochFieldStrategy validation requires per-sample 'field_strength_target' "
                "(set data.expose_field_strength_target). Got field_strength_target=None."
            )
        b_t = torch.as_tensor(b_t, device=x0.device).reshape(-1).float()
        return self.env.generator(x0, field_strength=b_t)
