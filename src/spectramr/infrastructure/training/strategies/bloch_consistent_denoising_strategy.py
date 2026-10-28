"""Bloch-Consistent Self-Supervised Denoising (ULF-PR-7 / II.4).

Plan: TODO/backlog_ulf_radiologist_acceptance_roadmap.md §ULF-PR-7.

Self-supervised denoising for ULF where ground-truth low-noise images
are unavailable. The trick: any clean image must satisfy the Bloch
signal-synthesis equation across multiple acquisitions of the *same*
subject with different (TR, TE, FA, B0). The strategy:

1. Forward the noisy multi-contrast input through a **parameter-map**
   model that emits a ``{'t1','t2','pd'}`` dict.
2. Use ``BlochSignalSynthesisConsistencyLoss`` to re-synthesise the
   per-contrast magnitudes from those maps and penalise the deviation
   from the observed contrasts (the maps must jointly explain every
   contrast, which is the self-supervision signal).

Data contract (all required when the Bloch objective runs):
``batch['observed_contrasts']`` ``[B, N_c, H, W]`` and
``batch['acquisition_params']`` (a list of per-contrast
``{TR_ms, TE_ms, FA_deg}``). If the generator returns a single tensor
(a plain image denoiser) the forward-synthesis loss cannot be computed
and the strategy fails loud (it does not silently degrade — NN#3).

The Noise2Noise decorrelation regulariser from the original plan needs
two independent noisy draws of the same subject; it is not wired yet
and is intentionally omitted rather than faked.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import torch

from spectramr.data.batch_types import read_batch_field

from .reconstruction import ReconstructionTrainingStrategy

logger = logging.getLogger(__name__)


class BlochConsistentDenoisingStrategy(ReconstructionTrainingStrategy):
    """Self-supervised Bloch-consistent denoiser."""

    #: Loss ownership (issue #1918). Bloch-consistency terms only; the hook overrides ReconstructionTrainingStrategy's
    #: without calling super() or the fold, so nothing from losses.image_losses runs.
    inline_losses: ClassVar[frozenset[str]] = frozenset()
    folds_image_losses: ClassVar[bool] = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        try:
            from spectramr.models.losses.bloch_signal_synthesis_consistency_loss import (
                BlochSignalSynthesisConsistencyLoss,
            )

            self._bloch = BlochSignalSynthesisConsistencyLoss().to(self.device)
        except ImportError as e:
            # NN#3 / pitfall #10: the strategy's defining physics objective is
            # the Bloch consistency term. Silently dropping it to ``None`` and
            # degrading to plain ReconstructionTrainingStrategy losses would be
            # a silent fallback that contradicts the strategy name and the
            # advertised training mode. Fail loudly instead.
            raise RuntimeError(
                "BlochConsistentDenoisingStrategy requires "
                "BlochSignalSynthesisConsistencyLoss but the import failed. "
                "Ensure spectramr.models.losses.bloch_signal_synthesis_consistency_loss "
                "is installed. (NN#3: no silent fallback to plain reconstruction.)"
            ) from e

    def _compute_losses_impl(
        self,
        input_batch: Any = None,
        target_batch: Any = None,
        epoch: int = 0,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        # F6/F8b: orchestrator-kwargs canonical signature. Resolve the batch
        # dict from kwargs and read the generator from ``self.env`` (the
        # StrategyContext dataclass has no ``generator`` field, so the old
        # ``self.context.generator`` was always None -> dead Bloch branch).
        from spectramr.models.losses.bloch_signal_synthesis_consistency_loss import (
            split_t1_t2_pd,
        )

        batch = self._resolve_legacy_batch(input_batch, kwargs)
        gen = getattr(self.env, "generator", None)
        if gen is None:
            raise RuntimeError(
                "bloch_consistent_denoising strategy: env.generator is None — "
                "the DI container did not produce a generator. Check the "
                "ModelBuilder wiring."
            )
        if self._bloch is None:
            # Dropping to the parent here trained a plain denoiser under a
            # Bloch-consistent arm name and reported success (audit A5/A7;
            # facade #16). The Bloch term IS the strategy.
            raise RuntimeError(
                "bloch_consistent_denoising: the Bloch signal-synthesis loss "
                "failed to construct, so the strategy's defining objective "
                "cannot be computed. Fix the loss wiring rather than running "
                "this arm as a plain denoiser."
            )

        # Guard BEFORE the generator call — see the twin fix in
        # multi_param_mapping_strategy. A dict batch missing the key passed
        # ``None`` into ``gen()``; a non-dict batch silently substituted the
        # whole batch object. The NotImplementedError below only fires once
        # ``gen`` has already returned, so it could never be reached from the
        # ``None`` path.
        observed = read_batch_field(batch, "observed_contrasts")
        acq = read_batch_field(batch, "acquisition_params")
        if observed is None:
            raise ValueError(
                "bloch_consistent_denoising requires batch['observed_contrasts'] "
                "of shape [B, N_c, H, W] to run the Bloch signal-synthesis "
                "consistency loss, but the batch supplies "
                f"{sorted(k for k in batch) if isinstance(batch, dict) else type(batch).__name__!r}. "
                "Supply a qMRI multi-contrast dataset that emits the key, or use "
                "a training_mode whose objective does not need it."
            )
        pred = gen(observed)

        # The Bloch signal-synthesis consistency loss is a *forward* model:
        # it maps predicted (T1, T2, PD) parameter maps + per-contrast
        # acquisition params back to multi-contrast magnitudes. A plain image
        # denoiser emits a single denoised tensor, not a parameter-map dict, so
        # the loss cannot be applied to it. Fail loud with an actionable message
        # rather than crash on the arity mismatch or silently drop the
        # strategy's defining objective (NN#3 / pitfall #10 / facade #16).
        if not isinstance(pred, dict):
            raise NotImplementedError(
                "BlochConsistentDenoisingStrategy needs the generator to emit a "
                "{'t1','t2','pd'} parameter-map dict, plus "
                "batch['observed_contrasts'] [B, N_c, H, W] and "
                "batch['acquisition_params'] (per-contrast TR/TE/FA), to compute "
                "the Bloch signal-synthesis consistency loss. The generator "
                "returned a single tensor (an image denoiser), which this "
                "forward-synthesis loss cannot consume. Use a parameter-map "
                "model (cf. OneShotMultiParameterStrategy) or a different "
                "training_mode."
            )
        if acq is None or not (isinstance(observed, torch.Tensor) and observed.dim() == 4):
            raise ValueError(
                "BlochConsistentDenoisingStrategy requires "
                "batch['acquisition_params'] (list of per-contrast "
                "{TR_ms, TE_ms, FA_deg}) and batch['observed_contrasts'] of "
                "shape [B, N_c, H, W]."
            )

        t1, t2, pd = split_t1_t2_pd(pred)
        bloch_l = self._bloch(t1, t2, pd, observed_contrasts=observed, acquisition_params=acq)
        # NOTE: the module docstring's Noise2Noise decorrelation regulariser
        # requires two independent noisy draws of the same subject, which the
        # batch does not currently supply — it is deliberately NOT faked here
        # (a zero-weighted or self-referential "n2n" would be a facade). The
        # objective is the Bloch self-consistency term until paired draws land.
        total = bloch_l
        return {
            "loss_total": total,
            "loss_bloch": bloch_l.detach(),
        }
