"""Cartoon-texture safe-restoration training (MICCAI MRIxFields2026, B-2.5).

Trains a restorer (any image->image generator, e.g. ``configurable_unet``) with
``L1 + lambda_ct * cartoon_texture_safe`` — the BV/G decomposition loss
(:class:`~spectramr.models.losses.cartoon_texture_loss.CartoonTextureSafeLoss`) that matches the
CARTOON (structure) hard and the TEXTURE loosely, CONFINING hallucination to texture so the
restorer cannot invent false anatomy. The detail prior is parameter-free, so the one-knob
ablation (``lambda_ct=0``) is an EXACT param-matched L1-only control.
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch
import torch.nn.functional as F

from spectramr.models.losses.cartoon_texture_loss import CartoonTextureSafeLoss

from .reconstruction import ReconstructionTrainingStrategy


def compute_cartoon_texture_loss(
    model: Any,
    batch: dict[str, Any],
    ct_loss: CartoonTextureSafeLoss,
    *,
    lambda_l1: float = 1.0,
    lambda_ct: float = 0.5,
) -> dict[str, torch.Tensor]:
    """L1 to the clean target + the cartoon-texture safe (hallucination-confining) prior."""
    x = batch["input"]
    y = batch["target"]
    pred = model(x)
    if isinstance(pred, (tuple, list)):
        pred = pred[0]
    loss_l1 = F.l1_loss(pred, y)
    ct = ct_loss.compute(pred, y)
    total = lambda_l1 * loss_l1 + lambda_ct * ct["loss_total"]
    return {
        "loss_total": total,
        "loss_l1": loss_l1,
        "loss_cartoon": ct["loss_cartoon"].detach(),
        "loss_texture": ct["loss_texture"].detach(),
    }


class CartoonTextureSafeStrategy(ReconstructionTrainingStrategy):
    """Train a restorer with the cartoon-texture hallucination-confinement prior (B-2.5)."""

    #: Loss ownership (mrixfields review 2026-09-03): computes its objective inline and folds NOTHING else: any other declared image loss is a decoy.
    inline_losses: ClassVar[frozenset[str] | None] = frozenset({"l1"})
    folds_image_losses: ClassVar[bool | None] = False

    def _setup_strategy_specific_components(self) -> None:
        self._verify_strategy_config(expected_modes=("cartoon_texture_safe", "reconstruction"))
        cfg = getattr(self.config.training, "cartoon_texture_safe", None)

        def _g(name: str, default: float) -> float:
            return float(getattr(cfg, name, default)) if cfg is not None else default

        self._ct_lambda_l1 = self._declared_inline_l1_weight()
        self._ct_lambda_ct = _g("lambda_ct", 0.5)
        self._ct_loss = CartoonTextureSafeLoss(
            tv_weight=_g("tv_weight", 0.15),
            n_iter=int(_g("n_iter", 40)),
            cartoon_weight=_g("cartoon_weight", 1.0),
            texture_weight=_g("texture_weight", 0.3),
            texture_pool=int(_g("texture_pool", 4)),
        )
        if getattr(self, "logging_service", None):
            self._log_config_features(self.logging_service)
            ct = self._ct_loss
            # Stamp ALL resolved knobs into the run log (pitfall #15 obligation c).
            self.logging_service.log_info(
                f"[B-2.5] cartoon-texture safe restoration: lambda_l1={self._ct_lambda_l1}, "
                f"lambda_ct={self._ct_lambda_ct}, tv_weight={ct.tv_weight}, n_iter={ct.n_iter}, "
                f"cartoon_weight={ct.cartoon_weight}, texture_weight={ct.texture_weight}, "
                f"texture_pool={ct.texture_pool} (cartoon=hard structure match, texture=loose -> "
                "hallucination confined to texture)"
            )

    def _ct_module(self) -> CartoonTextureSafeLoss:
        ct = getattr(self, "_ct_loss", None)
        if ct is None:
            ct = CartoonTextureSafeLoss()
            self._ct_loss = ct
        return ct

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
                "CartoonTextureSafeStrategy requires a mapping batch (dict/TrainingBatch); "
                f"got {type(batch)!r}."
            )
        return compute_cartoon_texture_loss(
            self.env.generator,
            batch,
            self._ct_module(),
            lambda_l1=getattr(self, "_ct_lambda_l1", 1.0),
            lambda_ct=getattr(self, "_ct_lambda_ct", 0.5),
        )

    def validation_step(self, input_batch: Any, target_batch: Any, **kwargs: Any) -> Any:
        bc = kwargs.pop("batch_context", None) or {}
        metrics = super().validation_step(input_batch, target_batch, batch_context=bc, **kwargs)
        if isinstance(metrics, dict):
            metrics.update(self._score_cartoon(input_batch, target_batch, bc))
        return metrics

    def _score_cartoon(
        self, input_batch: Any, target_batch: Any, batch_context: dict[str, Any]
    ) -> dict[str, float]:
        """Safety witness ``val_cartoon_fidelity`` in [0,1] — agreement of the restored CARTOON
        (structure) with the target's. ~1 = structure faithfully restored (no hallucination); a
        drop flags that the restorer is inventing/distorting structure. Self-gating; no new knob.
        """
        x = batch_context.get("input", input_batch)
        if not isinstance(x, torch.Tensor) or x.numel() == 0 or x.ndim < 2:
            return {}
        if not isinstance(target_batch, torch.Tensor) or target_batch.numel() == 0:
            return {}
        with torch.no_grad():
            pred = self.env.generator(x)
            if isinstance(pred, (tuple, list)):
                pred = pred[0]
            pred = pred.clamp(0.0, 1.0)
            fid = float(self._ct_module().cartoon_fidelity(pred, target_batch))
        return {"val_cartoon_fidelity": fid}

    def _validation_forward(
        self, input_batch: Any, batch_context: dict[str, Any], **kwargs: Any
    ) -> torch.Tensor:
        x0 = batch_context.get("input", input_batch)
        pred = self.env.generator(x0)
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        return pred.clamp(0.0, 1.0)
