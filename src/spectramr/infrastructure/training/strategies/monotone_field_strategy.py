"""Monotone field-ordered generator training (MICCAI MRIxFields2026, B-2.6).

Trains a :class:`MonotoneFieldOrderedGenerator` to render the target image at its field
via an L1 reconstruction loss, with an optional soft monotonicity penalty. The
field-monotonicity itself is STRUCTURAL (the model's softplus cumulative-residual head),
so the one-knob ablation lives on the model (``model_kwargs.enforce_monotone``).

Pure helper :func:`compute_monotone_field_loss` holds the testable loss math.
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch
import torch.nn.functional as F

from .reconstruction import ReconstructionTrainingStrategy


def compute_monotone_field_loss(
    model: Any,
    batch: dict[str, Any],
    *,
    lambda_l1: float = 1.0,
    lambda_monotone: float = 0.0,
) -> dict[str, torch.Tensor]:
    """L1 render loss + optional soft monotonicity penalty across the field axis."""
    x = batch["input"]
    y = batch["target"]
    b_t = batch["field_strength_target"].reshape(-1).float()
    pred = model(x, field_strength=b_t)
    loss_l1 = F.l1_loss(pred, y)
    total = lambda_l1 * loss_l1
    out: dict[str, torch.Tensor] = {"loss_l1": loss_l1}
    if lambda_monotone > 0.0:
        # Penalise field-inversion: rendering at a HIGHER field must not be smaller.
        # (Belt-and-suspenders; the softplus head already guarantees it structurally.)
        # Cap b_hi at the model's trained field range (10**tau_max) — NOT a hardcoded
        # literal — so the probe stays inside the bases' support for any field range.
        anchors = getattr(model, "anchors", None)
        ceiling = 10.0 ** float(anchors.max()) if anchors is not None else 7.0
        b_hi = (b_t * 1.5).clamp_max(ceiling)
        pred_hi = model(x, field_strength=b_hi)
        loss_mono = torch.mean(F.relu(pred - pred_hi))
        total = total + lambda_monotone * loss_mono
        out["loss_monotone"] = loss_mono
    out["loss_total"] = total
    # Grad-carrying pair for the loss-SSOT seam (popped in _compute_losses_impl).
    out["prediction"] = pred
    out["target_image"] = y
    return out


class MonotoneFieldStrategy(ReconstructionTrainingStrategy):
    """Train the monotone field-ordered generator (B-2.6)."""

    #: Loss ownership (mrixfields review 2026-09-03): folds the builder's other declared entries (calls the fold or the parent path).
    inline_losses: ClassVar[frozenset[str] | None] = frozenset({"l1"})
    folds_image_losses: ClassVar[bool | None] = True

    def _setup_strategy_specific_components(self) -> None:
        self._verify_strategy_config(expected_modes=("monotone_field", "reconstruction"))
        if getattr(self, "logging_service", None):
            self._log_config_features(self.logging_service)
        cfg = getattr(self.config.training, "monotone_field", None)
        self._mf_lambda_l1 = self._declared_inline_l1_weight()
        self._mf_lambda_monotone = (
            float(getattr(cfg, "lambda_monotone", 0.0)) if cfg is not None else 0.0
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
                "MonotoneFieldStrategy requires a mapping batch (dict/TrainingBatch) from the mrixfields "
                f"dataset; got {type(batch)!r}."
            )
        out = compute_monotone_field_loss(
            self.env.generator,
            batch,
            lambda_l1=getattr(self, "_mf_lambda_l1", 1.0),
            lambda_monotone=getattr(self, "_mf_lambda_monotone", 0.0),
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
                "MonotoneFieldStrategy validation requires per-sample "
                "'field_strength_target' (set data.expose_field_strength_target on the "
                "mrixfields dataset). Got field_strength_target=None."
            )
        # Move to the input device unconditionally (an already-tensor CPU field would
        # otherwise crash a CUDA model; the generator's anchors are device-safe too).
        b_t = torch.as_tensor(b_t, device=x0.device).reshape(-1).float()
        return self.env.generator(x0, field_strength=b_t)
