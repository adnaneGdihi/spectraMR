"""Karras EDM Training Strategy (PR-10 / M2).

Plan: TODO/backlog_paradigm_expansion_roadmap.md §PR-10.

Plugs the existing EDM noise-schedule + preconditioning primitives
(``spectramr.models.diffusion.edm_schedule``) into the diffusion training
loop so that ``training.diffusion.parameterization=edm`` selects the
Karras formulation instead of DDPM ε-prediction.

Reference: Karras et al. "Elucidating the Design Space of Diffusion-Based
Generative Models" (NeurIPS 2022).
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import torch

from spectramr.data.batch_types import read_batch_field

from .diffusion import DiffusionTrainingStrategy

logger = logging.getLogger(__name__)


class EDMTrainingStrategy(DiffusionTrainingStrategy):
    """Diffusion strategy with Karras EDM noise schedule + preconditioning."""

    #: Loss ownership (issue #1918). EDM preconditioned denoising score matching; computes no image loss and
    #: reaches neither the parent's builder path nor the computer.
    inline_losses: ClassVar[frozenset[str]] = frozenset()
    folds_image_losses: ClassVar[bool] = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        from spectramr.models.diffusion.edm_schedule import EDMNoiseSchedule

        self._edm_schedule = EDMNoiseSchedule()
        diff_cfg = getattr(self.config.training, "diffusion", None)
        if diff_cfg is not None:
            param = getattr(diff_cfg, "parameterization", None) or getattr(diff_cfg, "type", None)
            if param != "edm":
                logger.info(
                    "EDMTrainingStrategy active but training.diffusion.parameterization=%s — "
                    "the loop still uses Karras schedule and preconditioning by default.",
                    param,
                )

    def _compute_losses_impl(
        self,
        input_batch: Any = None,
        target_batch: Any = None,
        epoch: int = 0,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Sample sigma per Karras log-normal, apply preconditioning, regress."""
        batch = self._resolve_legacy_batch(input_batch, kwargs)
        gen = getattr(self.env, "generator", None)
        if gen is None:
            return {"loss_total": torch.tensor(0.0, device=self.device)}

        from spectramr.models.diffusion.edm_schedule import (
            edm_loss_weight,
            edm_preconditioning_coefficients,
        )

        # ``read_batch_field``, not ``isinstance(batch, dict)``. The batch the
        # loop delivers is a ``TrainingBatch`` dataclass, so the isinstance leg
        # is False and the ``else`` handed the WHOLE BATCH OBJECT on as if it
        # were the target tensor -- which surfaced as
        # ``'TrainingBatch' object has no attribute 'shape'`` / ``randn_like():
        # must be Tensor, not TrainingBatch``. ``TrainingBatch`` implements the
        # mapping protocol precisely so this accessor is shape-agnostic, and its
        # docstring names this pairing 'the single most repeated defect at this
        # seam'.
        clean = read_batch_field(batch, "target")
        if clean is None:
            return {"loss_total": torch.tensor(0.0, device=self.device)}

        b = clean.shape[0]
        sigma = self._edm_schedule.sample_sigma(b, device=self.device)
        noise = torch.randn_like(clean) * sigma.view(b, *([1] * (clean.dim() - 1)))
        noised = clean + noise

        c_skip, c_out, c_in, c_noise = edm_preconditioning_coefficients(sigma)
        c_skip = c_skip.view(b, *([1] * (clean.dim() - 1)))
        c_out = c_out.view(b, *([1] * (clean.dim() - 1)))
        c_in = c_in.view(b, *([1] * (clean.dim() - 1)))

        # Karras pred-x0 formulation: D(noised, sigma) = c_skip * noised + c_out * F(c_in * noised, c_noise)
        #
        # Hoisted out of the call so the snapshot can capture the tensor the
        # network ACTUALLY receives. ``noised`` alone would be a subtler version
        # of the same lie the contract exists to stop: the input is
        # PRECONDITIONED, and c_in varies by two orders of magnitude across the
        # sigma schedule, so the two do not even share a dynamic range.
        model_input = c_in * noised
        self._declare_model_input(
            {"model_input": model_input, "noised": noised, "target": clean},
            # Explicit and empty: none of these is k-space by NAME. The
            # canonical keys are unioned in from the config SSOT when the arm
            # declares k-space data -- hence the names.
            in_kspace_keys=set(),
            extra={
                "model_input_key": "model_input",
                "note": (
                    "EDM feeds the preconditioned c_in * noised, formed inside "
                    "this step; 'noised' (clean + sigma*eps) is shown beside it "
                    "for contrast and is NOT what the network sees. "
                    "'first_steps/input_prepared' is PRE-noising."
                ),
            },
        )
        f = gen(model_input, c_noise)
        denoised = c_skip * noised + c_out * f
        weight = edm_loss_weight(sigma).view(b, *([1] * (clean.dim() - 1)))
        loss = (weight * (denoised - clean).pow(2)).mean()
        return {"loss_total": loss, "loss_edm": loss.detach()}
