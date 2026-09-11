"""Re-degradation-consistency test-time adaptation (MICCAI MRIxFields2026, B-2.10).

A field-conditioned ULF->HF translator is trained supervised (L1). At INFERENCE, the
model is adapted PER SAMPLE on a self-supervised re-degradation-consistency objective:
degrade the model's HF prediction back to ULF and require it to match the observed ULF
source, ``||A(G(src, b)) - src||`` with ``A = ulf_degrade`` — no ground truth needed.
The adapted model then renders the final estimate, after which the base weights are
restored (per-sample test-time training).

Distinct from B-2.1 (image-MAP) and B-2.2 (image-DPS), which adapt the IMAGE: here the
MODEL PARAMETERS are adapted at test time. ``adaptation_steps=0`` recovers plain
feed-forward — the one-knob ablation. Reuses ``ulf_degrade`` (the canonical HF->ULF
low-pass) and the registered field-conditioned translator (e.g. field_conditioned_inr).
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch
import torch.nn.functional as F

from .reconstruction import ReconstructionTrainingStrategy
from .ulf_map_strategy import ulf_degrade


def compute_redegrad_tta_train_loss(
    model: Any, batch: dict[str, Any], *, lambda_l1: float = 1.0
) -> dict[str, torch.Tensor]:
    """Supervised L1 base-model training (the prior the TTA adapts at test time)."""
    x = batch["input"]
    y = batch["target"]
    b_t = batch["field_strength_target"].reshape(-1).float()
    pred = model(x, field_strength=b_t, contrast_id=batch.get("contrast_id"))
    loss_l1 = F.l1_loss(pred, y)
    return {
        "loss_total": lambda_l1 * loss_l1,
        "loss_l1": loss_l1,
        # Grad-carrying pair for the loss-SSOT seam (popped in _compute_losses_impl).
        "prediction": pred,
        "target_image": y,
    }


def redegrad_tta_render(
    model: Any,
    source: torch.Tensor,
    b_t: torch.Tensor,
    *,
    adaptation_steps: int,
    adaptation_lr: float,
    consistency_weight: float,
    blur_cutoff: float,
    contrast_id: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-sample test-time adaptation then render, with rollback-to-best-residual.

    Save the model params, run ``adaptation_steps`` Adam steps minimising the
    self-supervised re-degradation loss ``||ulf_degrade(G(src,b)) - src||``, but render
    with the params of the step that achieved the LOWEST residual (the base/step-0
    params included). This guarantees the proxy never REGRESSES from a converged prior
    (Adam from a near-optimal init can otherwise overshoot and increase the residual),
    and falls back to plain feed-forward when no step improves. The base params are
    always restored afterwards (per-sample TTT, no cross-sample leakage).

    CAVEATS (honest scope): (1) the proxy constrains only the low-pass PASSBAND, so a
    lower residual does NOT guarantee a better SSIM-vs-HF-target — the ablation
    (adaptation_steps=0) is the detector of whether TTA helps. (2) ``ulf_degrade`` is a
    SYNTHETIC Gaussian low-pass surrogate for the true 0.1T->HF forward map. Forces
    ``enable_grad`` so it works under the validation ``no_grad`` context.
    """
    b_t = b_t.reshape(-1).float()
    if adaptation_steps <= 0:
        with torch.no_grad():
            return model(source, field_strength=b_t, contrast_id=contrast_id).clamp(0.0, 1.0)

    def _residual() -> float:
        with torch.no_grad():
            r = F.mse_loss(
                ulf_degrade(
                    model(source, field_strength=b_t, contrast_id=contrast_id), blur_cutoff
                ),
                source,
            )
        return float(r)

    saved = {k: v.detach().clone() for k, v in model.state_dict().items()}
    params = [p for p in model.parameters() if p.requires_grad]
    try:
        opt = torch.optim.Adam(params, lr=adaptation_lr)
        best_res = _residual()  # step-0 (base) residual
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        with torch.enable_grad():
            for _ in range(adaptation_steps):
                opt.zero_grad(set_to_none=True)
                pred = model(source, field_strength=b_t, contrast_id=contrast_id)
                loss = consistency_weight * F.mse_loss(ulf_degrade(pred, blur_cutoff), source)
                if not torch.isfinite(loss):  # divergence/NaN guard -> keep best so far
                    break
                loss.backward()
                opt.step()
                cur = _residual()
                if cur < best_res:  # keep only improving iterates (rollback otherwise)
                    best_res = cur
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)  # render with the lowest-residual params
        with torch.no_grad():
            out = model(source, field_strength=b_t, contrast_id=contrast_id).clamp(0.0, 1.0)
    finally:
        model.load_state_dict(saved)  # restore base (per-sample; no leakage)
    return out


class UlfReDegradationTTAStrategy(ReconstructionTrainingStrategy):
    """Train a ULF->HF translator; adapt per-sample at test time (B-2.10)."""

    #: Loss ownership (mrixfields review 2026-09-03): folds the builder's other declared entries (calls the fold or the parent path).
    inline_losses: ClassVar[frozenset[str] | None] = frozenset({"l1"})
    folds_image_losses: ClassVar[bool | None] = True

    def _setup_strategy_specific_components(self) -> None:
        self._verify_strategy_config(expected_modes=("ulf_redegrad_tta", "reconstruction"))
        if getattr(self, "logging_service", None):
            self._log_config_features(self.logging_service)
        cfg = getattr(self.config.training, "ulf_redegrad_tta", None)
        self._tta_steps = int(getattr(cfg, "adaptation_steps", 8)) if cfg else 8
        self._tta_lr = float(getattr(cfg, "adaptation_lr", 1e-4)) if cfg else 1e-4
        self._tta_weight = float(getattr(cfg, "consistency_weight", 1.0)) if cfg else 1.0
        self._tta_cutoff = float(getattr(cfg, "blur_cutoff", 0.5)) if cfg else 0.5
        self._tta_lambda_l1 = self._declared_inline_l1_weight()

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
                "UlfReDegradationTTAStrategy requires a mapping batch (dict/TrainingBatch) from the mrixfields "
                f"dataset; got {type(batch)!r}."
            )
        out = compute_redegrad_tta_train_loss(
            self.env.generator, batch, lambda_l1=getattr(self, "_tta_lambda_l1", 1.0)
        )
        # Loss-SSOT seam: fold declarative image losses (hfen/ms_ssim) onto the base-model
        # objective; the inline l1 is skipped by the seam (no double-count). The TTA proxy
        # is a separate test-time objective and is unaffected.
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
        """Fold the per-sample target field + contrast id into batch_context (gated seam)."""
        bc = kwargs.pop("batch_context", None) or {}
        if field_strength_target is not None:
            bc["field_strength_target"] = field_strength_target
        if contrast_id is not None:
            bc["contrast_id"] = contrast_id
        return super().validation_step(input_batch, target_batch, batch_context=bc, **kwargs)

    def _validation_forward(
        self, input_batch: Any, batch_context: dict[str, Any], **kwargs: Any
    ) -> torch.Tensor:
        """Validation = per-sample re-degradation TTA then render."""
        src = batch_context.get("input", input_batch)
        b_t = batch_context.get("field_strength_target", kwargs.get("field_strength_target"))
        if b_t is None:
            raise ValueError(
                "UlfReDegradationTTAStrategy validation requires per-sample "
                "'field_strength_target' (set data.expose_field_strength_target). Got "
                "field_strength_target=None."
            )
        b_t = torch.as_tensor(b_t, device=src.device)
        # Per-sample adaptation (chunk via val_chunk_size to bound the inner-loop graph).
        # Default to 1 (true per-sample TTT): val_chunk_size=0 would otherwise adapt the
        # whole batch jointly, weakening the per-sample contract this method is named for.
        val_cfg = getattr(self.config, "validation", None)
        chunk = int((val_cfg.loader.chunk_size if val_cfg else 0) or 0) or 1
        kw = {
            "adaptation_steps": getattr(self, "_tta_steps", 8),
            "adaptation_lr": getattr(self, "_tta_lr", 1e-4),
            "consistency_weight": getattr(self, "_tta_weight", 1.0),
            "blur_cutoff": getattr(self, "_tta_cutoff", 0.5),
        }
        gen = self.env.generator
        cid = batch_context.get("contrast_id", kwargs.get("contrast_id"))
        if chunk > 0 and src.shape[0] > chunk:
            outs = [
                redegrad_tta_render(
                    gen,
                    src[i : i + chunk],
                    b_t.reshape(-1)[i : i + chunk],
                    contrast_id=None if cid is None else cid.reshape(-1)[i : i + chunk],
                    **kw,
                )
                for i in range(0, src.shape[0], chunk)
            ]
            return torch.cat(outs, dim=0)
        return redegrad_tta_render(gen, src, b_t, contrast_id=cid, **kw)
