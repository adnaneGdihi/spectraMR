"""Field-conditioned INR super-resolution (MICCAI MRIxFields2026, B-2.8).

Trains a :class:`FieldConditionedINR` (SIREN coordinate-MLP) to render the high-field
target from the ULF source image conditioned on the continuous target field. The INR
is resolution-free; the field FiLM-modulates the render so the same anatomy maps to the
requested field. ``use_field_conditioning=False`` passes a constant (zero) field — the
one-knob ablation isolating the value of conditioning.

Pure helper :func:`compute_field_conditioned_inr_loss` holds the testable loss math;
the strategy wires it to ``self.env.generator`` and the mrixfields batch.
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch
import torch.nn.functional as F

from .reconstruction import ReconstructionTrainingStrategy


def _render(
    model: Any,
    x: torch.Tensor,
    b_t: torch.Tensor,
    use_field_conditioning: bool,
    contrast_id: torch.Tensor | None = None,
) -> torch.Tensor:
    """Render the target image; field-agnostic (zero field) when conditioning off."""
    fs = b_t.reshape(-1).float()
    if not use_field_conditioning:
        fs = torch.zeros_like(fs)
    return model(x, field_strength=fs, contrast_id=contrast_id)


def compute_field_conditioned_inr_loss(
    model: Any,
    batch: dict[str, Any],
    *,
    use_field_conditioning: bool = True,
    lambda_l1: float = 1.0,
) -> dict[str, torch.Tensor]:
    """L1 reconstruction of the high-field target from the source via the field-INR."""
    x = batch["input"]
    y = batch["target"]
    b_t = batch["field_strength_target"]
    pred = _render(model, x, b_t, use_field_conditioning, batch.get("contrast_id"))
    loss_l1 = F.l1_loss(pred, y)
    total = lambda_l1 * loss_l1
    return {
        "loss_total": total,
        "loss_l1": loss_l1,
        # Grad-carrying pair for the loss-SSOT seam (popped in _compute_losses_impl).
        "prediction": pred,
        "target_image": y,
    }


class FieldConditionedINRStrategy(ReconstructionTrainingStrategy):
    """Train a field-conditioned SIREN INR for cross-field SR (B-2.8)."""

    #: Loss ownership (mrixfields review 2026-09-03): folds the builder's other declared entries (calls the fold or the parent path).
    inline_losses: ClassVar[frozenset[str] | None] = frozenset({"l1"})
    folds_image_losses: ClassVar[bool | None] = True

    def _setup_strategy_specific_components(self) -> None:
        self._verify_strategy_config(expected_modes=("field_conditioned_inr", "reconstruction"))
        if getattr(self, "logging_service", None):
            self._log_config_features(self.logging_service)
        cfg = getattr(self.config.training, "field_conditioned_inr", None)
        self._inr_use_field = (
            bool(getattr(cfg, "use_field_conditioning", True)) if cfg is not None else True
        )
        self._inr_lambda_l1 = self._declared_inline_l1_weight()
        if getattr(self, "logging_service", None):
            self.logging_service.log_info(
                f"FieldConditionedINRStrategy: use_field_conditioning="
                f"{self._inr_use_field} lambda_l1={self._inr_lambda_l1}"
            )

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
                "FieldConditionedINRStrategy requires a mapping batch (dict/TrainingBatch) from the "
                f"mrixfields dataset; got {type(batch)!r}."
            )
        out = compute_field_conditioned_inr_loss(
            self.env.generator,
            batch,
            use_field_conditioning=getattr(self, "_inr_use_field", True),
            lambda_l1=getattr(self, "_inr_lambda_l1", 1.0),
        )
        # Loss-SSOT seam: fold declarative image losses (tv/log_spectral) onto the inline
        # objective; the inline l1 is skipped by the seam (no double-count). Here the folded
        # terms SUPPRESS the SIREN's over-textured stipple rather than sharpen (Regime-2).
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
        contrast_id: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Fold the per-sample target field + contrast id into batch_context (the gated
        seam only forwards fields the signature declares)."""
        bc = kwargs.pop("batch_context", None) or {}
        if field_strength_target is not None:
            bc["field_strength_target"] = field_strength_target
        if contrast_id is not None:
            bc["contrast_id"] = contrast_id
        return super().validation_step(input_batch, target_batch, batch_context=bc, **kwargs)

    def _validation_forward(
        self, input_batch: Any, batch_context: dict[str, Any], **kwargs: Any
    ) -> torch.Tensor:
        """Validation = render the target image at the validation sample's field."""
        x0 = batch_context.get("input", input_batch)
        b_t = batch_context.get("field_strength_target", kwargs.get("field_strength_target"))
        if b_t is None:
            raise ValueError(
                "FieldConditionedINRStrategy validation requires per-sample "
                "'field_strength_target' (set data.expose_field_strength_target on the "
                "mrixfields dataset). Got field_strength_target=None."
            )
        b_t = b_t if torch.is_tensor(b_t) else torch.as_tensor(b_t, device=x0.device)
        return _render(
            self.env.generator,
            x0,
            b_t,
            getattr(self, "_inr_use_field", True),
            batch_context.get("contrast_id", kwargs.get("contrast_id")),
        )
