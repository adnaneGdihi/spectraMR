"""SPECTRA test-time adaptation (TTA) training strategy (plan §D).

The *only* training-flavoured SPECTRA workstream. Full on-device training is a
major undertaking; the near-term, defensible use case is **lightweight
test-time adaptation for field/scanner shift**: freeze the pre-trained
backbone, adapt only the normalization / adapter parameters per subject with a
self-supervised k-space-consistency objective.

Why freeze the backbone (the load-bearing design choice)
--------------------------------------------------------
The inference→training memory multiplier is dominated by retained activations,
gradients, master weights and Adam moments — a 4–10×+ blow-up that a 2 MB GLB
cannot hold. Freezing the backbone collapses the gradient and moment terms from
``|W|`` to ``|W_adapter| ≪ |W|``, keeping the multiplier bounded so adaptation
fits the inference datapath. The ``tta_adapter_only`` Tier-1 check enforces this
at config time; this strategy enforces it at runtime via ``requires_grad``.

This is a host/co-sim prototype of the paradigm (plan §D recommends prototyping
TTA on host first, hardening to silicon only after the inference path is
measured and stable). It composes with the SPECTRA inference backend
(``backend_acceleration``) but does not require it to run.
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch

from spectramr.infrastructure.training.step_io import accepts_step_io
from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy
from spectramr.infrastructure.training.strategies.mixins.kspace import KspaceMixin
from spectramr.models.capabilities import StrategyCapabilities
from spectramr.models.losses.kspace_consistency_tta_loss import KspaceConsistencyTTALoss

# Parameter-name fragments whose modules carry the *adaptable* parameters under
# the freeze-the-backbone discipline: normalization layers (instance/group/batch
# norm affine params) and any module explicitly named "adapter". Everything else
# is frozen.
_ADAPTABLE_NAME_HINTS: tuple[str, ...] = (
    "norm",
    "bn",
    "gn",
    "instancenorm",
    "groupnorm",
    "adapter",
)


def _is_adaptable_param(name: str) -> bool:
    """A parameter is adaptable iff its qualified name names a norm/adapter module."""
    lname = name.lower()
    return any(hint in lname for hint in _ADAPTABLE_NAME_HINTS)


class SpectraTestTimeAdaptationStrategy(BaseTrainingStrategy, KspaceMixin):
    """Freeze-backbone, adapt-norm test-time adaptation with a self-supervised
    k-space-consistency objective.

    On construction the backbone is frozen except for normalization / adapter
    parameters (``requires_grad`` toggled per name). ``_compute_losses_impl``
    runs a forward pass and scores the reconstruction against the *measured*
    k-space via :class:`KspaceConsistencyTTALoss` — no ground-truth target is
    used, which is what makes this single-subject self-supervised.

    Config is read from ``training.spectra_tta`` (a :class:`TrainingConfigSpectraTTA`
    block); when absent, documented defaults apply so the audit/probe path still
    exercises the gradient.

    Args:
        env: The :class:`TrainingEnvironment` (carries ``.generator``).
        lambda_consistency: Weight on the k-space consistency term.
        freeze_backbone: When True (default), freeze every parameter that is not
            a norm/adapter parameter. Set False only to debug.
    """

    # Computes its own objective and reads no `losses:` block, so an empty builder stack
    # is the design (LossBuilder.validate) and any declared image loss is a decoy.
    inline_losses: ClassVar[frozenset[str] | None] = frozenset()
    folds_image_losses: ClassVar[bool | None] = False

    #: Validation keys this strategy's own ``validation_step`` writes (cohort
    #: review 2026-09-02, T0.5); ``validation.scoring.compute`` may name them.
    capabilities = StrategyCapabilities(emitted_metrics=frozenset({"val_kspace_consistency_tta"}))

    def __init__(
        self,
        env: Any | None = None,
        lambda_consistency: float = 1.0,
        freeze_backbone: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(env=env, **kwargs)
        self.lambda_consistency = float(lambda_consistency)
        self.freeze_backbone = bool(freeze_backbone)
        self.consistency_loss = KspaceConsistencyTTALoss(weight=self.lambda_consistency)
        # Read optional config overrides without hard-failing if absent. The
        # training.spectra_tta block arrives as a plain dict (Any passthrough on
        # TrainingStrategyConfigSchema, like tto) OR a coerced SpectraTTAConfig —
        # handle both shapes.
        cfg = getattr(self.env, "config", None) if self.env is not None else None
        # ``cfg`` can be None (no env) — single getattrs, no nested chain.
        _training = getattr(cfg, "training", None)
        tta_cfg = getattr(_training, "spectra_tta", None)
        if tta_cfg is not None:

            def _read(key: str, default: Any) -> Any:
                if isinstance(tta_cfg, dict):
                    return tta_cfg.get(key, default)
                return getattr(tta_cfg, key, default)

            self.lambda_consistency = float(_read("lambda_consistency", self.lambda_consistency))
            self.freeze_backbone = bool(_read("freeze_backbone", self.freeze_backbone))
            self.consistency_loss = KspaceConsistencyTTALoss(weight=self.lambda_consistency)
        if self.freeze_backbone:
            self.apply_backbone_freeze()

    # ------------------------------------------------------------------
    # Freeze discipline (the bounded-memory guarantee)
    # ------------------------------------------------------------------
    def apply_backbone_freeze(self) -> dict[str, int]:
        """Freeze every non-adaptable parameter; leave norm/adapter params trainable.

        Returns a small accounting dict (``trainable`` / ``frozen`` counts) so
        callers and tests can assert the freeze actually took effect.
        """
        gen = getattr(self.env, "generator", None) if self.env is not None else None
        trainable = frozen = 0
        if gen is None or not hasattr(gen, "named_parameters"):
            return {"trainable": 0, "frozen": 0}
        for name, p in gen.named_parameters():
            adapt = _is_adaptable_param(name)
            p.requires_grad_(adapt)
            if adapt:
                trainable += 1
            else:
                frozen += 1
        return {"trainable": trainable, "frozen": frozen}

    def adapter_parameters(self) -> list[torch.nn.Parameter]:
        """The trainable (norm/adapter) parameters — what the optimizer should see."""
        gen = getattr(self.env, "generator", None) if self.env is not None else None
        if gen is None or not hasattr(gen, "named_parameters"):
            return []
        return [p for n, p in gen.named_parameters() if _is_adaptable_param(n)]

    # ------------------------------------------------------------------
    # Self-supervised adaptation objective
    # ------------------------------------------------------------------
    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """One self-supervised TTA step.

        ``input_batch`` is the (undersampled) reconstruction input; the measured
        k-space used as the consistency target is taken from ``kwargs['kspace']``
        when supplied, otherwise derived from ``target_batch`` (treated as the
        measured signal). ``mask`` (sampling pattern) is passed through when
        present. Returns the required ``g_total_loss`` scalar.
        """
        predictions = self.env.generator(input_batch)

        measured_kspace = kwargs.get("kspace")
        if measured_kspace is None:
            # No explicit measured k-space provided: treat target_batch as the
            # measured signal in its native domain. Use the centered SSOT DFT
            # (``fft2c``) — matching the loss's centered forward and an externally-
            # supplied centered k-space/mask (CLAUDE.md pitfall #2, never raw fft2).
            from spectramr.infrastructure.physics.fft_ops import fft2c

            measured_kspace = fft2c(self.consistency_loss._to_complex(target_batch))

        dc = self.consistency_loss(predictions, measured_kspace, mask=kwargs.get("mask"))
        return {"kspace_consistency_tta": dc, "g_total_loss": dc}

    @accepts_step_io
    @torch.no_grad()
    def validation_step(self, batch: Any, **kwargs: Any) -> dict[str, float]:
        input_batch, target_batch = self._unpack_batch(batch)
        self.env.generator.eval()
        predictions = self.env.generator(input_batch)
        from spectramr.infrastructure.physics.fft_ops import fft2c

        measured_kspace = fft2c(self.consistency_loss._to_complex(target_batch))
        dc = self.consistency_loss(predictions, measured_kspace)
        return {"val_kspace_consistency_tta": float(dc.item())}
