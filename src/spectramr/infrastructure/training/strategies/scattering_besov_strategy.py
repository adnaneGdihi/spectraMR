"""Scattering-Besov detail-prior training (MICCAI MRIxFields2026, B-1.3).

Trains a :class:`~spectramr.models.generators.scattering_field_translator.ScatteringFieldTranslator`
(source-field-conditioned ->7T) with ``L1 + lambda_scatter * scattering_besov`` — the wavelet
scattering + Besov detail prior (:class:`~spectramr.models.losses.scattering_besov_loss.\
ScatteringBesovLoss`) that drives high-frequency 7T detail synthesis L1 alone under-produces
(blur). The detail prior is parameter-free (fixed filterbank), so the one-knob ablation
(``lambda_scatter=0``) is an EXACT param-matched L1-only control.
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch
import torch.nn.functional as F

from spectramr.models.losses.scattering_besov_loss import ScatteringBesovLoss

from .reconstruction import ReconstructionTrainingStrategy


def compute_scattering_besov_loss(
    model: Any,
    batch: dict[str, Any],
    scatter_loss: ScatteringBesovLoss,
    *,
    lambda_l1: float = 1.0,
    lambda_scatter: float = 0.5,
) -> dict[str, torch.Tensor]:
    """L1 to the 7T target + the scattering-Besov detail prior."""
    x = batch["input"]
    y = batch["target"]
    if "field_strength" not in batch:
        # Fail loudly (not a cryptic KeyError) if the dataset does not emit the source field
        # this field-conditioned ->7T strategy needs (pitfall #9). The mrixfields dataset emits
        # it; the Tier-2 probe injects it (so a green --probe does NOT prove the real path works).
        raise ValueError(
            "ScatteringBesovStrategy requires batch['field_strength'] (the source field, Tesla). "
            "Use data.dataset_type='mrixfields' (which emits per-sample field_strength). "
            f"Got batch keys: {sorted(batch.keys())}."
        )
    b_s = batch["field_strength"].reshape(-1).float()
    pred = model(x, field_strength=b_s, contrast_id=batch.get("contrast_id"))
    loss_l1 = F.l1_loss(pred, y)
    sb = scatter_loss.compute(pred, y)
    total = lambda_l1 * loss_l1 + lambda_scatter * sb["loss_total"]
    return {
        "loss_total": total,
        "loss_l1": loss_l1,
        "loss_scatter": sb["loss_scatter"],
        "loss_besov": sb["loss_besov"],
        # Grad-carrying image pair so the strategy can fold builder-composed
        # declarative image losses (the loss-SSOT seam). Popped before the dict is
        # returned to the trainer, so these tensors never leak into the loss log.
        "prediction": pred,
        "target_image": y,
    }


class ScatteringBesovStrategy(ReconstructionTrainingStrategy):
    """Train the ->7T translator with the scattering-Besov detail prior (B-1.3)."""

    #: Loss ownership (mrixfields review 2026-09-03): folds the builder's other declared entries (calls the fold or the parent path).
    inline_losses: ClassVar[frozenset[str] | None] = frozenset({"l1", "scattering_besov"})
    folds_image_losses: ClassVar[bool | None] = True

    def _setup_strategy_specific_components(self) -> None:
        self._verify_strategy_config(expected_modes=("scattering_besov", "reconstruction"))
        cfg = getattr(self.config.training, "scattering_besov", None)

        def _g(name: str, default: float) -> float:
            return float(getattr(cfg, name, default)) if cfg is not None else default

        self._sb_lambda_l1 = self._declared_inline_l1_weight()
        self._sb_lambda_scatter = _g("lambda_scatter", 0.5)
        self._scatter_loss = ScatteringBesovLoss(
            n_scales=int(_g("n_scales", 3)),
            n_orientations=int(_g("n_orientations", 4)),
            besov_s=_g("besov_s", 1.0),
            scatter_weight=_g("scatter_weight", 1.0),
            besov_weight=_g("besov_weight", 1.0),
        )
        if getattr(self, "logging_service", None):
            self._log_config_features(self.logging_service)
            self.logging_service.log_info(
                f"[B-1.3] scattering-Besov detail prior: lambda_scatter={self._sb_lambda_scatter}, "
                f"n_scales={self._scatter_loss.n_scales}, n_orient={self._scatter_loss.n_orientations}, "
                f"besov_s={self._scatter_loss.besov_s} (filterbank is parameter-free)"
            )

    def _scatter_module(self) -> ScatteringBesovLoss:
        sl = getattr(self, "_scatter_loss", None)
        if sl is None:
            sl = ScatteringBesovLoss()
            self._scatter_loss = sl
        gen = getattr(self.env, "generator", None)
        if gen is not None:
            sl = sl.to(next(gen.parameters()).device)
        return sl

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
                "ScatteringBesovStrategy requires a mapping batch (dict/TrainingBatch) from the "
                f"mrixfields dataset; got {type(batch)!r}."
            )
        out = compute_scattering_besov_loss(
            self.env.generator,
            batch,
            self._scatter_module(),
            lambda_l1=getattr(self, "_sb_lambda_l1", 1.0),
            lambda_scatter=getattr(self, "_sb_lambda_scatter", 0.5),
        )
        # Loss-SSOT seam: fold any builder-composed declarative image losses
        # (losses.image_losses, e.g. hfen/ms_ssim sharpness terms) onto the inline
        # L1+scattering objective, so the declarative block is LIVE here rather than
        # a decoy. No-op when losses.image_losses is empty (env.losses falsy).
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
        field_strength: Any = None,
        contrast_id: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Fold the per-sample source field + contrast id into batch_context (gated seam)."""
        bc = kwargs.pop("batch_context", None) or {}
        if field_strength is not None:
            bc["field_strength"] = field_strength
        if contrast_id is not None:
            bc["contrast_id"] = contrast_id
        metrics = super().validation_step(input_batch, target_batch, batch_context=bc, **kwargs)
        if isinstance(metrics, dict):
            metrics.update(self._score_detail(input_batch, target_batch, bc))
        return metrics

    def _score_detail(
        self, input_batch: Any, target_batch: Any, batch_context: dict[str, Any]
    ) -> dict[str, float]:
        """Mechanism-fires monitor: how close is the prediction's DETAIL ENERGY to the target's?

        Emits ``val_detail_energy_ratio`` = mean wavelet-modulus energy of the prediction divided
        by the target's. ~1.0 means the detail prior is working (the synthesised 7T detail matches
        the target's richness); <<1.0 means the output is over-smooth/blurred (the prior has not
        fired). ``val_detail_energy_target`` is logged for provenance. Self-gating; no new knob.
        """
        x = batch_context.get("input", input_batch)
        if not isinstance(x, torch.Tensor) or x.numel() == 0 or x.ndim < 2:
            return {}
        b_s = batch_context.get("field_strength")
        if b_s is None:
            return {}
        sl = self._scatter_module()
        gen = self.env.generator
        with torch.no_grad():
            b_s = torch.as_tensor(b_s, device=x.device).reshape(-1).float()
            pred = gen(x, field_strength=b_s, contrast_id=batch_context.get("contrast_id")).clamp(
                0.0, 1.0
            )
            e_pred = float(sl.detail_energy(pred))
            out: dict[str, float] = {}
            if isinstance(target_batch, torch.Tensor) and target_batch.numel() > 0:
                e_tgt = float(sl.detail_energy(target_batch))
                out["val_detail_energy_target"] = e_tgt
                out["val_detail_energy_ratio"] = e_pred / (e_tgt + 1e-8)
            else:
                out["val_detail_energy_pred"] = e_pred
        return out

    def _validation_forward(
        self, input_batch: Any, batch_context: dict[str, Any], **kwargs: Any
    ) -> torch.Tensor:
        """Validation = ->7T render conditioned on the source field; clamp [0,1]."""
        x0 = batch_context.get("input", input_batch)
        b_s = batch_context.get("field_strength", kwargs.get("field_strength"))
        if b_s is None:
            raise ValueError(
                "ScatteringBesovStrategy validation requires per-sample 'field_strength' (source)."
            )
        b_s = torch.as_tensor(b_s, device=x0.device).reshape(-1).float()
        return self.env.generator(
            x0,
            field_strength=b_s,
            contrast_id=batch_context.get("contrast_id", kwargs.get("contrast_id")),
        ).clamp(0.0, 1.0)
