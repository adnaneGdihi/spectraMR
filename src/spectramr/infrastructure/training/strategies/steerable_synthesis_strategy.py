"""Steerable C_4-equivariant cross-field synthesis training (MICCAI MRIxFields2026, B-1.4).

Trains a :class:`~spectramr.models.generators.steerable_field_unet.SteerableFieldUNet` to
synthesise the target image from the source via an L1 reconstruction loss, conditioned on
the SOURCE field strength. The rotation-equivariance is STRUCTURAL on the model (built from
the P4 block family), so the one-knob ablation (``model_kwargs.use_equivariance``) lives on
the model — flipping it to a matched-width plain-conv backbone isolates the equivariance
constraint (and its ``sqrt(|G|)`` generalisation benefit).

Conditioning on the SOURCE field (not the target): on Task 1 (any -> 7T) the target is fixed
at 7T, so a target-field FiLM would be inert; the source field varies per sample, so it is
genuinely load-bearing (avoids the fixed-target inert-knob trap).
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch
import torch.nn.functional as F

from .reconstruction import ReconstructionTrainingStrategy


def compute_steerable_synthesis_loss(
    model: Any, batch: dict[str, Any], *, lambda_l1: float = 1.0
) -> dict[str, torch.Tensor]:
    """L1 reconstruction of the target from the source via the equivariant synthesiser."""
    x = batch["input"]
    y = batch["target"]
    b_s = batch["field_strength"].reshape(-1).float()  # SOURCE field (varies on Task 1)
    pred = model(x, field_strength=b_s, contrast_id=batch.get("contrast_id"))
    loss_l1 = F.l1_loss(pred, y)
    return {"loss_total": lambda_l1 * loss_l1, "loss_l1": loss_l1}


class SteerableSynthesisStrategy(ReconstructionTrainingStrategy):
    """Train a steerable C_4-equivariant cross-field synthesiser (B-1.4)."""

    #: Loss ownership (mrixfields review 2026-09-03): computes its objective inline and folds NOTHING else: any other declared image loss is a decoy.
    inline_losses: ClassVar[frozenset[str] | None] = frozenset({"l1"})
    folds_image_losses: ClassVar[bool | None] = False

    def _setup_strategy_specific_components(self) -> None:
        self._verify_strategy_config(expected_modes=("steerable_synthesis", "reconstruction"))
        if getattr(self, "logging_service", None):
            self._log_config_features(self.logging_service)
        cfg = getattr(self.config.training, "steerable_synthesis", None)
        self._ss_lambda_l1 = self._declared_inline_l1_weight()

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
                "SteerableSynthesisStrategy requires a mapping batch (dict/TrainingBatch) from the "
                f"mrixfields dataset; got {type(batch)!r}."
            )
        return compute_steerable_synthesis_loss(
            self.env.generator, batch, lambda_l1=getattr(self, "_ss_lambda_l1", 1.0)
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
        """Validation = synthesise the target from the source at the source field."""
        x0 = batch_context.get("input", input_batch)
        b_s = batch_context.get("field_strength", kwargs.get("field_strength"))
        if b_s is None:
            raise ValueError(
                "SteerableSynthesisStrategy validation requires per-sample 'field_strength' "
                "(the source field; set on the mrixfields dataset). Got field_strength=None."
            )
        b_s = torch.as_tensor(b_s, device=x0.device).reshape(-1).float()
        # Clamp to the [0,1] target range before metric scoring: the generator output is
        # intentionally unbounded (raw, so training gradients flow above 1), but SSIM/PSNR
        # auto-detect data_range from the [0,1] target, so grading out-of-range pixels would
        # bias the metric AND the two arms have systematically different out-of-range stats
        # (#20). Clamp is pointwise -> equivariance-preserving. Training (L1) stays unclamped.
        return self.env.generator(
            x0,
            field_strength=b_s,
            contrast_id=batch_context.get("contrast_id", kwargs.get("contrast_id")),
        ).clamp(0.0, 1.0)
