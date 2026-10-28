"""Continuous low-rank modulation training (MICCAI MRIxFields2026, B-3.6).

Trains a :class:`~spectramr.models.generators.lora_modulation_net.LoRAModulationNet` (a shared
conv backbone modulated by a continuous field-conditioned low-rank update) to translate any
source field to any target field, via L1. The low-rank modulation is STRUCTURAL on the model,
so the one-knob ablation (``model_kwargs.lora_rank``) lives there. Any-to-any (Task 3):
conditions on BOTH source and target field.
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch
import torch.nn.functional as F

from .reconstruction import ReconstructionTrainingStrategy


def compute_lora_modulation_loss(
    model: Any, batch: dict[str, Any], *, lambda_l1: float = 1.0
) -> dict[str, torch.Tensor]:
    """L1 reconstruction of the target via the low-rank-modulated shared backbone."""
    x = batch["input"]
    y = batch["target"]
    b_s = batch["field_strength"].reshape(-1).float()
    b_t = batch["field_strength_target"].reshape(-1).float()
    pred = model(x, field_strength=b_s, field_strength_target=b_t)
    loss_l1 = F.l1_loss(pred, y)
    return {"loss_total": lambda_l1 * loss_l1, "loss_l1": loss_l1}


class LoRAModulationStrategy(ReconstructionTrainingStrategy):
    """Train the continuous low-rank modulation cross-field translator (B-3.6)."""

    #: Loss ownership (mrixfields review 2026-09-03): computes its objective inline and folds NOTHING else: any other declared image loss is a decoy.
    inline_losses: ClassVar[frozenset[str] | None] = frozenset({"l1"})
    folds_image_losses: ClassVar[bool | None] = False

    def _setup_strategy_specific_components(self) -> None:
        self._verify_strategy_config(expected_modes=("lora_modulation", "reconstruction"))
        if getattr(self, "logging_service", None):
            self._log_config_features(self.logging_service)
            # Stamp the compliance bound r/max-rank(W) into the run log so the
            # headline "r << dim(W)" claim is recorded per run (pitfall #15 obl. c).
            gen = getattr(self.env, "generator", None)
            if gen is not None and hasattr(gen, "lora_rank_fraction"):
                frac = gen.lora_rank_fraction()
                self.logging_service.log_info(
                    f"[B-3.6] compliance bound r/max-rank(W)={frac:.4g} "
                    f"(lora_rank={getattr(gen, 'lora_rank', '?')}, width={getattr(gen, 'width', '?')}, "
                    f"freeze_field={getattr(gen, 'freeze_field', '?')})"
                )
        cfg = getattr(self.config.training, "lora_modulation", None)
        self._lm_lambda_l1 = self._declared_inline_l1_weight()

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
                "LoRAModulationStrategy requires a mapping batch (dict/TrainingBatch) from the "
                f"mrixfields dataset; got {type(batch)!r}."
            )
        return compute_lora_modulation_loss(
            self.env.generator, batch, lambda_l1=getattr(self, "_lm_lambda_l1", 1.0)
        )

    def validation_step(
        self,
        input_batch: Any,
        target_batch: Any,
        field_strength: Any = None,
        field_strength_target: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Fold the per-sample source + target fields into batch_context (gated seam)."""
        bc = kwargs.pop("batch_context", None) or {}
        if field_strength is not None:
            bc["field_strength"] = field_strength
        if field_strength_target is not None:
            bc["field_strength_target"] = field_strength_target
        metrics = super().validation_step(input_batch, target_batch, batch_context=bc, **kwargs)
        # Mechanism-fires monitor (pitfall #16): emit how much the output moves when ONLY
        # the field changes, so a SILENT collapse to field-blind surfaces in run artifacts.
        if isinstance(metrics, dict):
            metrics.update(self._score_field_sensitivity(input_batch, bc))
        return metrics

    def _score_field_sensitivity(
        self, input_batch: Any, batch_context: dict[str, Any]
    ) -> dict[str, float]:
        """Re-run the trained generator on a fixed input with two DISTINCT (b_s,b_t) pairs.

        Emits ``val_field_sensitivity`` = mean ``|out(f1) - out(f2)|`` / mean ``|out(f1)|``,
        the relative change in the output when only the conditioning field moves. For
        ``lora_rank > 0`` a value ~0 after warmup flags a silent collapse to field-blind
        (the headline "weights are field-modulated" claim becoming unfalsifiable); for the
        ``lora_rank=0`` control it is ~0 BY CONSTRUCTION — the documented field-blind floor,
        a reader can tell the two regimes apart. Self-gates on a real input tensor; no new
        YAML knob (mirrors the VF ``val_field_shift_mae`` seam).
        """
        x = batch_context.get("input", input_batch)
        if not isinstance(x, torch.Tensor) or x.numel() == 0 or x.ndim < 2:
            return {}
        gen = self.env.generator
        n = x.shape[0]
        dev = x.device

        def _full(v: float) -> torch.Tensor:
            return torch.full((n,), float(v), device=dev)

        with torch.no_grad():
            o1 = gen(x, field_strength=_full(0.05), field_strength_target=_full(3.0))
            o2 = gen(x, field_strength=_full(1.5), field_strength_target=_full(7.0))
        denom = o1.abs().mean().clamp_min(1e-8)
        sens = ((o1 - o2).abs().mean() / denom).detach()
        return {"val_field_sensitivity": float(sens)}

    def _validation_forward(
        self, input_batch: Any, batch_context: dict[str, Any], **kwargs: Any
    ) -> torch.Tensor:
        """Validation = render via the low-rank-modulated backbone (source->target); clamp."""
        x0 = batch_context.get("input", input_batch)
        b_s = batch_context.get("field_strength", kwargs.get("field_strength"))
        b_t = batch_context.get("field_strength_target", kwargs.get("field_strength_target"))
        if b_s is None or b_t is None:
            raise ValueError(
                "LoRAModulationStrategy validation requires per-sample 'field_strength' AND "
                "'field_strength_target'. Got "
                f"field_strength={b_s is not None}, field_strength_target={b_t is not None}."
            )
        b_s = torch.as_tensor(b_s, device=x0.device).reshape(-1).float()
        b_t = torch.as_tensor(b_t, device=x0.device).reshape(-1).float()
        return self.env.generator(x0, field_strength=b_s, field_strength_target=b_t).clamp(0.0, 1.0)
