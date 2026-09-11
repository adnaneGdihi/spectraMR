"""Cross-field relaxometry inversion with Bloch resynthesis (MRIxFields2026, 2.1).

Task 1 (arbitrary field -> 7T) as a relaxometry problem: the
:class:`RelaxometryEncoder` estimates ``(rho, T1_ref, T2)`` + a dispersion exponent
``beta`` from the MULTI-CONTRAST source stack, transports ``T1`` across field by the
empirical power law, and re-evaluates the frozen SPGR signal operator at 7T; a
high-pass residual is confined to the opaque band. A reconstruction-family strategy
(single optimizer, like the sibling ``bloch_field``) that computes its objective
inline:

* paired L1 + folded LPIPS on the 7T synthesis;
* **source-consistency** ``||render(params, b_s) - y_s||_1`` (Proposition 3, the
  identify-then-resynthesise loop);
* **dispersion-prior** keeping ``beta in [0.3, 0.4]``;
* **segmentation-consistency** (reused ``segmentation_dice``) on a frozen differentiable
  segmenter — targets the challenge Dice / volume-consistency metrics.

Every inline weight is read from ``loop_state.loss_weight_overrides`` so a
``loss_schedule:`` block drives the curriculum (source-consistency first, then ramp
seg-consistency once the physics inversion is established).
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch
import torch.nn.functional as F
from torch import nn

from spectramr.infrastructure.training.strategies.loss_folding import scheduled_overrides
from spectramr.infrastructure.training.strategies.reconstruction import (
    ReconstructionTrainingStrategy,
)
from spectramr.models.losses.bloch_synth_losses import (
    BlochSourceConsistencyLoss,
    DispersionPriorLoss,
)
from spectramr.models.losses.dice_anatomy_loss import SegmentationDiceLoss

_SEGMENTER_BACKENDS = ("none", "label_dice")


class _IntensitySoftSegmenter(nn.Module):
    """Frozen, differentiable, contrast-agnostic soft segmenter (label_dice backend).

    Maps a magnitude image to ``K`` soft intensity classes via fixed Gaussian bumps at
    evenly-spaced centres in ``[0, 1]``: ``logits_k = -(x - c_k)^2 / temp``. No learned
    parameters (a fixed teacher), fully differentiable in the image — enough to make
    the seg-consistency term a genuine structural signal locally without a pretrained
    SynthSeg. On the cluster a real differentiable SynthSeg replaces it.
    """

    def __init__(self, n_classes: int = 6, temp: float = 0.02) -> None:
        super().__init__()
        if n_classes < 2:
            raise ValueError(f"n_classes must be >= 2; got {n_classes}.")
        self.temp = float(temp)
        centres = torch.linspace(0.0, 1.0, n_classes).view(1, n_classes, 1, 1)
        self.register_buffer("centres", centres)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return -((x - self.centres) ** 2) / self.temp


class BlochSynthesisStrategy(ReconstructionTrainingStrategy):
    """Relaxometry inversion + dispersion transport + SPGR resynthesis (idea 2.1)."""

    #: Loss ownership (mrixfields review 2026-09-03): folds the builder's other declared entries (calls the fold or the parent path).
    inline_losses: ClassVar[frozenset[str] | None] = frozenset({"l1", "segmentation_dice"})
    folds_image_losses: ClassVar[bool | None] = True

    _SUPPORTED_CONDITION_SOURCES = ("field_strength", "contrast_id")

    #: ``segmentation_dice`` is computed inline above (``w_seg * loss_seg``), so the
    #: fold must skip it. Declaring it on ``losses.image_losses`` is nonetheless
    #: required for the ``seg_consistency`` curriculum rule: the schedule controller
    #: resolves a rule's base weight through the loss-weight SSOT, which sees only
    #: ``losses.*`` — a target living solely in ``training.bloch_synth`` raises at the
    #: trigger iteration. ``bloch_source_consistency`` / ``dispersion_prior`` are not
    #: registered losses, so they can never reach ``env.losses`` and need no skip.

    def _setup_strategy_specific_components(self) -> None:
        self._verify_strategy_config(expected_modes=("bloch_synth", "reconstruction"))
        if getattr(self, "logging_service", None):
            self._log_config_features(self.logging_service)
        cfg = getattr(self.config.training, "bloch_synth", None)

        def _get(name: str, default: float) -> float:
            return float(getattr(cfg, name, default)) if cfg is not None else default

        self._source_consistency_weight = _get("source_consistency_weight", 1.0)
        self._seg_consistency_weight = _get("seg_consistency_weight", 0.5)
        self._dispersion_prior_weight = _get("dispersion_prior_weight", 0.1)
        self._residual_weight = _get("residual_weight", 0.25)
        lo, hi = 0.3, 0.4
        if cfg is not None and getattr(cfg, "dispersion_beta_bounds", None) is not None:
            lo, hi = float(cfg.dispersion_beta_bounds[0]), float(cfg.dispersion_beta_bounds[1])
        backend = str(getattr(cfg, "segmenter_backend", "label_dice") or "label_dice")
        if backend not in _SEGMENTER_BACKENDS:
            raise ValueError(
                f"bloch_synth: unsupported segmenter_backend={backend!r}; supported "
                f"{_SEGMENTER_BACKENDS} (a differentiable teacher; real SynthSeg is a "
                "cluster-side follow-up)."
            )
        self._segmenter_backend = backend
        # Move the frozen segmenter (its ``centres`` buffer) to the training device —
        # the strategy is not an nn.Module, so it is not swept there automatically, and
        # a CPU buffer vs CUDA image crashes the seg term on step 1 of any GPU run.
        self._segmenter = (
            _IntensitySoftSegmenter().to(self.device) if backend == "label_dice" else None
        )
        # Registered loss modules ARE the runtime path (weight applied by the strategy).
        self._source_loss = BlochSourceConsistencyLoss(weight=1.0)
        self._dispersion_prior = DispersionPriorLoss(weight=1.0, lo=lo, hi=hi)
        # require_segmenter follows availability: if we have one, demand it (no silent
        # no-op); if the arm turned seg off (weight 0), it is never called.
        self._seg_loss = SegmentationDiceLoss(require_segmenter=self._segmenter is not None)
        self.last_dispersion_beta: torch.Tensor | None = None

        if getattr(self, "logging_service", None):
            self.logging_service.log_info(
                "BlochSynthesisStrategy: w(src="
                f"{self._source_consistency_weight},seg={self._seg_consistency_weight},"
                f"disp={self._dispersion_prior_weight},resid={self._residual_weight}) "
                f"beta_bounds=[{lo},{hi}] segmenter={backend}"
            )

    def _scheduled(self) -> dict[str, float]:
        return scheduled_overrides(self)

    def _compute_losses_impl(
        self,
        input_batch: Any = None,
        target_batch: Any = None,
        epoch: int = 0,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        batch = self._resolve_legacy_batch(input_batch, kwargs)
        if batch is None or not hasattr(batch, "get"):
            raise ValueError(
                "BlochSynthesisStrategy requires a mapping batch (dict/TrainingBatch) "
                f"from the mrixfields multi_contrast dataset; got {type(batch)!r}."
            )
        x = batch["input"]  # [B, C_src, H, W] multi-contrast source stack
        y = batch["target"]  # [B, 1, H, W] target (7T)
        b_s = batch["field_strength"]
        b_t = batch.get("field_strength_target")
        if b_t is None:
            raise ValueError(
                "BlochSynthesisStrategy needs 'field_strength_target' (set "
                "data.expose_field_strength_target on the mrixfields dataset)."
            )
        gen = self.env.generator

        params = gen.predict_parameters(x)  # (rho, T1_ref, T2, beta), grad-carrying
        beta = params[3]
        y_det = gen.render(params, b_t)
        resid = gen.opaque_residual(y_det, x)
        y_hat = (y_det + self._residual_weight * resid).clamp(0.0, 1.0)
        y_src = gen.render(params, b_s)  # source-consistency (deterministic)

        # Source acquisition proxy: first source contrast channel (SPGR renders one
        # contrast). Documented reduction — the multi-contrast stack disambiguates the
        # inversion; the resynthesis is graded against the T1w channel.
        y_src_ref = x[:, 0:1]

        loss_paired = F.l1_loss(y_hat, y)
        loss_src = self._source_loss(y_src, y_src_ref)
        loss_disp = self._dispersion_prior(beta)

        sched = self._scheduled()
        w_src = sched.get("bloch_source_consistency", self._source_consistency_weight)
        w_disp = sched.get("dispersion_prior", self._dispersion_prior_weight)
        w_seg = sched.get(
            "seg_consistency",
            sched.get("segmentation_dice", self._seg_consistency_weight),
        )

        total = loss_paired + w_src * loss_src + w_disp * loss_disp
        out: dict[str, torch.Tensor] = {
            "loss_total": total,
            "loss_paired": loss_paired.detach(),
            "loss_bloch_source_consistency": loss_src.detach(),
            "loss_dispersion_prior": loss_disp.detach(),
        }
        if w_seg > 0:
            loss_seg = self._seg_loss(y_hat, y, context={"segmenter": self._segmenter})
            total = total + w_seg * loss_seg
            out["loss_total"] = total
            out["loss_seg_consistency"] = loss_seg.detach()

        # Fold declarative image losses (lpips/hfen/ms_ssim); inline l1 is skipped.
        aux = self._apply_builder_image_losses(y_hat, y, out)
        if aux is not None:
            out["loss_total"] = out["loss_total"] + aux

        with torch.no_grad():
            self.last_dispersion_beta = beta.detach().mean()
            out["dispersion_beta_mean"] = self.last_dispersion_beta
        self._last_prediction = y_hat
        self._last_target = y
        return out

    def validation_step(
        self,
        input_batch: Any,
        target_batch: Any,
        field_strength_target: Any = None,
        contrast_id: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Fold the per-sample target field into batch_context so the inherited
        reconstruction validation renders the RelaxometryEncoder at 7T (pitfall #18)."""
        bc = kwargs.pop("batch_context", None) or {}
        if field_strength_target is not None:
            bc["field_strength_target"] = field_strength_target
        if contrast_id is not None:
            bc["contrast_id"] = contrast_id
        return super().validation_step(input_batch, target_batch, batch_context=bc, **kwargs)

    def _validation_forward(
        self, input_batch: Any, batch_context: dict[str, Any], **kwargs: Any
    ) -> torch.Tensor:
        """Validation synthesis, blended at the SAME ``residual_weight`` as training.

        The inherited reconstruction validation calls ``generator.forward`` directly,
        which blends the opaque residual at weight 1.0 — but ``_compute_losses_impl``
        optimises ``y_det + residual_weight * resid`` (0.25 by default). Scoring
        val_synthseg_dice / SSIM and selecting the early-stopping checkpoint on a
        different image than the training objective mis-ranks checkpoints, so replicate
        the training blend exactly here.
        """
        x0 = batch_context.get("input", input_batch)
        b_t = batch_context.get("field_strength_target", kwargs.get("field_strength_target"))
        if b_t is None:
            raise ValueError(
                "BlochSynthesisStrategy validation requires per-sample "
                "'field_strength_target'. Set data.expose_field_strength_target on "
                "the mrixfields dataset."
            )
        b_t = b_t.to(x0.device)
        gen = self.env.generator
        params = gen.predict_parameters(x0)
        y_det = gen.render(params, b_t)
        resid = gen.opaque_residual(y_det, x0)
        return (y_det + self._residual_weight * resid).clamp(0.0, 1.0)
