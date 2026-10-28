"""Field-conditioned spectral transfer operator training (MICCAI, B-1.6/3.7).

Trains :class:`FieldConditionedFNO` to render the target image at its field via an L1
reconstruction loss; validation renders at the validation sample's target field. The
field-conditioning mechanism (the field-token spectral multiplier) lives on the MODEL,
so the one-knob ablation is ``model_kwargs.field_token_enable``.
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch
import torch.nn.functional as F

from .reconstruction import ReconstructionTrainingStrategy


def compute_field_fno_loss(
    model: Any, batch: dict[str, Any], *, lambda_l1: float = 1.0
) -> dict[str, torch.Tensor]:
    """L1 reconstruction of the target from the source via the field-conditioned FNO."""
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


class FieldFNOStrategy(ReconstructionTrainingStrategy):
    """Train a field-conditioned spectral transfer operator (B-1.6/3.7)."""

    #: Loss ownership (mrixfields review 2026-09-03): folds the builder's other declared entries (calls the fold or the parent path).
    inline_losses: ClassVar[frozenset[str] | None] = frozenset({"l1"})
    folds_image_losses: ClassVar[bool | None] = True

    def _setup_strategy_specific_components(self) -> None:
        self._verify_strategy_config(expected_modes=("field_fno", "reconstruction"))
        if getattr(self, "logging_service", None):
            self._log_config_features(self.logging_service)
        cfg = getattr(self.config.training, "field_fno", None)
        self._fno_lambda_l1 = self._declared_inline_l1_weight()

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
                "FieldFNOStrategy requires a mapping batch (dict/TrainingBatch) from the mrixfields dataset; "
                f"got {type(batch)!r}."
            )
        out = compute_field_fno_loss(
            self.env.generator, batch, lambda_l1=getattr(self, "_fno_lambda_l1", 1.0)
        )
        # Loss-SSOT seam: fold declarative image losses (hfen/ms_ssim sharpness) onto
        # the inline L1 so losses.image_losses is LIVE here. No-op when empty; the
        # inline l1 placeholder is skipped by the seam (no double-count).
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
        metrics = super().validation_step(input_batch, target_batch, batch_context=bc, **kwargs)
        if isinstance(metrics, dict):
            metrics.update(self._score_fno(input_batch, bc))
        return metrics

    def _score_fno(self, input_batch: Any, batch_context: dict[str, Any]) -> dict[str, float]:
        """Witness: relative output delta when the SAME input is rendered at 0.1 T vs 7 T.

        ``val_fno_field_sensitivity`` = ``mean|R(x, 7T) - R(x, 0.1T)|`` normalised by the mean
        output magnitude. The field-token spectral multiplier is zero-init = identity, so an
        INERT mechanism — the ``field_token_enable=False`` control, OR an as-yet-untrained
        multiplier — renders the two fields IDENTICALLY and the witness reads ``~0``. Once the
        multiplier trains away from identity the witness rises (``>0``), certifying *at run time*
        that the field-conditioning fired. This closes the #16 facade gap: a green smoke run on
        an untrained FNO is field-invariant *by construction*, so only this witness (not the
        smoke pass) distinguishes a live operator from the inert control. Self-gating; no new
        YAML knob (mirrors ``field_wiener_strategy._score_wiener``).
        """
        x = batch_context.get("input", input_batch)
        gen = self.env.generator
        if not isinstance(x, torch.Tensor) or x.numel() == 0 or x.dim() != 4:
            return {}
        with torch.no_grad():
            b = x.shape[0]
            hi = gen(x, field_strength=torch.full((b,), 7.0, device=x.device))
            lo = gen(x, field_strength=torch.full((b,), 0.1, device=x.device))
            denom = 0.5 * (hi.abs().mean() + lo.abs().mean()) + 1e-8
            sensitivity = float((hi - lo).abs().mean() / denom)
        return {"val_fno_field_sensitivity": sensitivity}

    def _validation_forward(
        self, input_batch: Any, batch_context: dict[str, Any], **kwargs: Any
    ) -> torch.Tensor:
        """Validation = render the target image at the validation sample's field."""
        x0 = batch_context.get("input", input_batch)
        b_t = batch_context.get("field_strength_target", kwargs.get("field_strength_target"))
        if b_t is None:
            raise ValueError(
                "FieldFNOStrategy validation requires per-sample 'field_strength_target' "
                "(set data.expose_field_strength_target). Got field_strength_target=None."
            )
        b_t = torch.as_tensor(b_t, device=x0.device).reshape(-1).float()
        return self.env.generator(x0, field_strength=b_t)
