"""Encode-once / render-anywhere cross-field translation (MICCAI MRIxFields2026).

Realises arms B-3.8 (any-to-any, Task 3) and B-1.9 (fixed target → 7T, Task 1) on
the :class:`AnatomyFieldRenderer` backbone. The pure helper
:func:`compute_cross_field_losses` holds the testable loss math; the strategy wires
it to ``self.env.generator`` and the framework loss plumbing.
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch
import torch.nn.functional as F

from spectramr.infrastructure.training.strategies.reconstruction import (
    ReconstructionTrainingStrategy,
)


def compute_cross_field_losses(
    generator: Any,
    batch: dict[str, Any],
    lambda_latent_cycle: float = 0.1,
    contrast_conditioning: bool = True,
) -> dict[str, torch.Tensor]:
    """Reconstruction (L1) + optional latent-cycle identifiability loss.

    Args:
        generator: an :class:`AnatomyFieldRenderer`-like module with
            ``encode``/``render``.
        batch: dict with ``input``, ``target``, ``field_strength_target`` and
            optionally ``contrast_id``.
        lambda_latent_cycle: weight of ``||E(R(q,b)) - q||_1`` (0 disables).
        contrast_conditioning: when False, the contrast id is NOT fed to the
            renderer FiLM (the ``cross_field.contrast_conditioning`` knob); the
            renderer then uses its zero/default contrast vector.

    Returns:
        dict with ``loss_total``, ``loss_recon``, ``loss_latent_cycle``.
    """
    x = batch["input"]
    y = batch["target"]
    b_t = batch["field_strength_target"]
    cid = batch.get("contrast_id") if contrast_conditioning else None
    q = generator.encode(x)
    x_hat = generator.render(q, b_t, cid)
    loss_recon = F.l1_loss(x_hat, y)
    if lambda_latent_cycle > 0.0:
        q_re = generator.encode(x_hat)
        loss_cycle = torch.mean(torch.abs(q_re - q))
    else:
        loss_cycle = x_hat.new_zeros(())
    total = loss_recon + lambda_latent_cycle * loss_cycle
    return {
        "loss_total": total,
        "loss_recon": loss_recon,
        "loss_latent_cycle": loss_cycle,
        # Grad-carrying pair for the loss-SSOT seam (popped in _compute_losses_impl).
        # Shared by b17/b19/b38: the seam skips the inline l1, so an arm that keeps the
        # bare ``[{l1, 1.0}]`` placeholder (b38) folds nothing and stays byte-identical;
        # only arms whose YAML adds hfen/ms_ssim (b17/b19) get the sharpness terms.
        "prediction": x_hat,
        "target_image": y,
    }


class CrossFieldTranslationStrategy(ReconstructionTrainingStrategy):
    """Train AnatomyFieldRenderer for cross-field translation (B-3.8 / B-1.9)."""

    #: Loss ownership (mrixfields review 2026-09-03): folds the builder's other declared entries (calls the fold or the parent path).
    inline_losses: ClassVar[frozenset[str] | None] = frozenset({"l1", "latent_cycle"})
    folds_image_losses: ClassVar[bool | None] = True

    def _setup_strategy_specific_components(self) -> None:
        # cross_field_translation reuses reconstruction plumbing; verify the mode
        # explicitly (the parent only accepts reconstruction-family modes).
        self._verify_strategy_config(expected_modes=("cross_field_translation", "reconstruction"))
        if getattr(self, "logging_service", None):
            self._log_config_features(self.logging_service)
        cfg = getattr(self.config.training, "cross_field", None)
        self._lambda_cycle = (
            float(getattr(cfg, "lambda_latent_cycle", 0.1)) if cfg is not None else 0.1
        )
        # contrast_conditioning knob (#15): gate the FiLM contrast id end-to-end so
        # the toggle is real (was advertised-but-unread). Stamped below for provenance.
        self._contrast_conditioning = (
            bool(getattr(cfg, "contrast_conditioning", True)) if cfg is not None else True
        )
        if getattr(self, "logging_service", None):
            self.logging_service.log_info(
                f"CrossFieldTranslationStrategy: lambda_latent_cycle="
                f"{self._lambda_cycle} contrast_conditioning="
                f"{self._contrast_conditioning}"
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
                "CrossFieldTranslationStrategy requires a mapping batch (dict/TrainingBatch) from the "
                f"mrixfields dataset; got {type(batch)!r}."
            )
        lam = getattr(self, "_lambda_cycle", 0.1)
        out = compute_cross_field_losses(
            self.env.generator,
            batch,
            lambda_latent_cycle=lam,
            contrast_conditioning=getattr(self, "_contrast_conditioning", True),
        )
        # Loss-SSOT seam: fold declarative image losses (hfen/ms_ssim) onto the inline
        # objective; the inline l1 is skipped by the seam (no double-count). Shared arm:
        # b38 keeps the bare l1 placeholder -> folds nothing; b17/b19 add sharpness.
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
        """Fold the per-sample target field + contrast into ``batch_context``.

        Declaring these makes the pipeline's gated validation seam forward them so
        :meth:`_validation_forward` can render at the validation sample's own target
        field (AnatomyFieldRenderer.forward requires ``field_strength``).
        """
        bc = kwargs.pop("batch_context", None) or {}
        if field_strength_target is not None:
            bc["field_strength_target"] = field_strength_target
        if contrast_id is not None:
            bc["contrast_id"] = contrast_id
        return super().validation_step(input_batch, target_batch, batch_context=bc, **kwargs)

    def _validation_forward(
        self, input_batch: Any, batch_context: dict[str, Any], **kwargs: Any
    ) -> torch.Tensor:
        """Validation prediction = render at the target field.

        The inherited reconstruction default calls ``generator(x)`` without
        ``field_strength``, which AnatomyFieldRenderer rejects; here we encode then
        render at the validation sample's target field (pitfall #18: the metric
        scores the actually-translated image).
        """
        x0 = batch_context.get("input", input_batch)
        b_t = batch_context.get("field_strength_target", kwargs.get("field_strength_target"))
        if b_t is None:
            raise ValueError(
                "CrossFieldTranslationStrategy validation requires per-sample "
                "'field_strength_target'. Set data.expose_field_strength_target on "
                "the mrixfields dataset. Got field_strength_target=None."
            )
        cid = (
            batch_context.get("contrast_id", kwargs.get("contrast_id"))
            if getattr(self, "_contrast_conditioning", True)
            else None
        )
        gen = self.env.generator
        return gen.render(gen.encode(x0), b_t, cid)
