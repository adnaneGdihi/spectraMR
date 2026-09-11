"""Field-dependent spectral Wiener restoration training (MICCAI MRIxFields2026, B-2.7).

Trains a :class:`~spectramr.models.generators.field_wiener_net.FieldWienerNet` (UNet estimate +
field-conditioned empirical spectral Wiener gain) with L1. The Wiener noise level is structural on
the model, so the one-knob ablation (``model_kwargs.use_field_noise``) lives there. The headline
witness ``val_recoverable_band`` reports the recoverable frequency band, which the field-dependent
filter narrows at low field.
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch
import torch.nn.functional as F

from .reconstruction import ReconstructionTrainingStrategy


def compute_field_wiener_loss(
    model: Any, batch: dict[str, Any], *, lambda_l1: float = 1.0
) -> dict[str, torch.Tensor]:
    """L1 restoration via the field-conditioned spectral Wiener restorer."""
    x = batch["input"]
    y = batch["target"]
    if "field_strength" not in batch:
        raise ValueError(
            "FieldWienerStrategy requires batch['field_strength'] (source field, Tesla). Use "
            f"data.dataset_type='mrixfields'. Got batch keys: {sorted(batch.keys())}."
        )
    b_s = batch["field_strength"].reshape(-1).float()
    pred = model(x, field_strength=b_s)
    loss_l1 = F.l1_loss(pred, y)
    return {
        "loss_total": lambda_l1 * loss_l1,
        "loss_l1": loss_l1,
        # Grad-carrying pair for the loss-SSOT seam (popped in _compute_losses_impl).
        "prediction": pred,
        "target_image": y,
    }


class FieldWienerStrategy(ReconstructionTrainingStrategy):
    """Train the field-dependent spectral Wiener restorer (B-2.7)."""

    #: Loss ownership (mrixfields review 2026-09-03): folds the builder's other declared entries (calls the fold or the parent path).
    inline_losses: ClassVar[frozenset[str] | None] = frozenset({"l1"})
    folds_image_losses: ClassVar[bool | None] = True

    def _setup_strategy_specific_components(self) -> None:
        self._verify_strategy_config(expected_modes=("field_wiener", "reconstruction"))
        cfg = getattr(self.config.training, "field_wiener", None)
        self._fw_lambda_l1 = self._declared_inline_l1_weight()
        if getattr(self, "logging_service", None):
            self._log_config_features(self.logging_service)
            gen = getattr(self.env, "generator", None)
            self.logging_service.log_info(
                f"[B-2.7] field-dependent spectral Wiener: lambda_l1={self._fw_lambda_l1}, "
                f"use_field_noise={getattr(gen, 'use_field_noise', '?')} "
                "(lower field -> higher noise PSD -> narrower recoverable band)"
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
                "FieldWienerStrategy requires a mapping batch (dict/TrainingBatch); "
                f"got {type(batch)!r}."
            )
        out = compute_field_wiener_loss(
            self.env.generator, batch, lambda_l1=getattr(self, "_fw_lambda_l1", 1.0)
        )
        # Loss-SSOT seam: fold declarative image losses (hfen/ms_ssim/sobel_edge) onto
        # the inline objective; the inline l1 is skipped by the seam (no double-count).
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
        self, input_batch: Any, target_batch: Any, field_strength: Any = None, **kwargs: Any
    ) -> Any:
        bc = kwargs.pop("batch_context", None) or {}
        if field_strength is not None:
            bc["field_strength"] = field_strength
        metrics = super().validation_step(input_batch, target_batch, batch_context=bc, **kwargs)
        if isinstance(metrics, dict):
            metrics.update(self._score_wiener(input_batch, bc))
        return metrics

    def _score_wiener(self, input_batch: Any, batch_context: dict[str, Any]) -> dict[str, float]:
        """Witnesses: the recoverable band at the batch field + a field-sensitivity of the band.

        ``val_recoverable_band`` = mean radial frequency where the Wiener gain crosses 0.5 at the
        batch's field. ``val_band_field_sensitivity`` = (band at 7T) - (band at 0.1T) on a fixed
        estimate; >0 confirms the field-dependent Wiener narrows the band at low field (the
        mechanism fires), ~0 for the field-blind control. Self-gating; no new YAML knob.
        """
        x = batch_context.get("input", input_batch)
        b_s = batch_context.get("field_strength")
        gen = self.env.generator
        if not isinstance(x, torch.Tensor) or x.numel() == 0 or x.ndim < 2 or b_s is None:
            return {}
        if not hasattr(gen, "recoverable_band"):
            return {}
        with torch.no_grad():
            b_s = torch.as_tensor(b_s, device=x.device).reshape(-1).float()
            hw = (int(x.shape[-2]), int(x.shape[-1]))
            band = float(gen.recoverable_band(b_s, hw))
            hi = float(gen.recoverable_band(torch.tensor([7.0], device=x.device), hw))
            lo = float(gen.recoverable_band(torch.tensor([0.1], device=x.device), hw))
        return {"val_recoverable_band": band, "val_band_field_sensitivity": hi - lo}

    def _validation_forward(
        self, input_batch: Any, batch_context: dict[str, Any], **kwargs: Any
    ) -> torch.Tensor:
        x0 = batch_context.get("input", input_batch)
        b_s = batch_context.get("field_strength", kwargs.get("field_strength"))
        if b_s is None:
            raise ValueError("FieldWienerStrategy validation requires per-sample 'field_strength'.")
        b_s = torch.as_tensor(b_s, device=x0.device).reshape(-1).float()
        return self.env.generator(x0, field_strength=b_s).clamp(0.0, 1.0)
