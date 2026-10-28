"""Scout-Conditioned Adaptive Sampling Strategy (SCAS — idea 5).

Plan: TODO/integration_plan_ulf_cheap_fast_mri.md §5.

A short scout pre-scan is fed into a hypernetwork that predicts a
sample-specific Bernoulli mask logit field.  The mask is sampled
through the relaxed-Bernoulli primitive and used to undersample the
main acquisition.

Audit promotion: previously this class only forwarded ``scout`` into
``_prepare_generator_inputs`` and never used the relaxed-Bernoulli
primitive its docstring claimed.  It now:

1. Builds a small ``ScoutHypernet`` MLP that maps a scout feature
   pooled from ``batch["scout"]`` to per-line mask logits.
2. Samples via :func:`gumbel_sigmoid` (straight-through) so gradients
   flow back into the hypernet weights.
3. Adds an :func:`expected_density` penalty pulling the average
   sampling rate toward the configured ``target_rate``.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from spectramr.infrastructure.optimization.relaxed_bernoulli import (
    expected_density,
    gumbel_sigmoid,
)

from .loupe_strategy import LOUPEStrategy

logger = logging.getLogger(__name__)


class ScoutHypernet(nn.Module):
    """Scout image → per-line mask logits.

    A tiny CNN→MLP that produces ``H`` logits when given a scout
    image of shape ``(B, C, H, W)``.  Output is *not* sigmoid-ed —
    the caller decides whether to relax-sample or threshold.
    """

    def __init__(self, in_channels: int = 1, hidden: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((None, 1)),  # collapse W
        )
        self.head = nn.Conv1d(hidden, 1, kernel_size=3, padding=1)

    def forward(self, scout: torch.Tensor) -> torch.Tensor:
        # scout: (B, C, H, W) → (B, hidden, H, 1) → (B, hidden, H) → (B, 1, H) → (B, H)
        f = self.encoder(scout).squeeze(-1)
        logits = self.head(f).squeeze(1)
        return logits


class SCASStrategy(LOUPEStrategy):
    """Scout-conditioned learnable mask via relaxed-Bernoulli sampling."""

    # parallel_compat: Its scout hypernet is a bare group on opt_g, so DDP never reduces it. Not
    # migrated: the scout path cannot run, since it indexes the parent's (image, kwargs) tuple.
    parallel_compat = frozenset()

    target_rate: float = 0.125
    tau: float = 0.5
    lambda_density: float = 1e-1

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._hypernet: ScoutHypernet | None = None

    def _ensure_hypernet(self, in_channels: int, device: torch.device) -> ScoutHypernet:
        if self._hypernet is None:
            self._hypernet = ScoutHypernet(in_channels=in_channels).to(device)
            opt = getattr(self.env, "opt_g", None)
            if opt is not None:
                opt.add_param_group({"params": list(self._hypernet.parameters()), "lr": 1e-4})
            logger.info("[scas] scout-hypernet attached (in_channels=%d)", in_channels)
        return self._hypernet

    def _prepare_generator_inputs(self, batch: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        inputs = super()._prepare_generator_inputs(batch, *args, **kwargs)
        if not isinstance(batch, dict) or "scout" not in batch:
            # SCAS *is* the scout-conditioned acquisition mechanism: without the
            # scout the hypernet is never built, never joins opt_g, and neither
            # scas_mask nor scas_logits is ever written -- so the density
            # penalty in _compute_losses_impl also silently no-ops and the arm
            # degenerates to plain LOUPE under the SCAS name (pitfall #16).
            # ScoutAcquisitionTransform is the documented producer and had no
            # way to be constructed until the transform registry landed.
            raise ValueError(
                "scas requires batch['scout'] (the low-resolution scout view) "
                "to condition the acquisition hypernet, but the batch supplies "
                f"{sorted(batch) if isinstance(batch, dict) else type(batch).__name__!r}. "
                "Declare the producer:\n"
                "  data:\n"
                "    processing:\n"
                "      transforms:\n"
                "        - name: scout_acquisition\n"
                "          kwargs: {scout_resolution: [32, 32]}\n"
                "Without it this arm is plain LOUPE, not SCAS."
            )

        scout = batch["scout"]
        h = self._ensure_hypernet(scout.shape[1], scout.device)
        logits = h(scout)
        # Straight-through relaxed sampling so gradients flow.
        mask = gumbel_sigmoid(logits, tau=self.tau, hard=True)
        # Cache for the loss term + downstream consumers.
        batch["scas_mask"] = mask
        batch["scas_logits"] = logits
        inputs["scout"] = scout
        inputs["acquisition_mask"] = mask
        return inputs

    def _compute_losses_impl(
        self,
        input_batch: Any = None,
        target_batch: Any = None,
        epoch: int = 0,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        # F6 (round 6 2026-05-17): orchestrator-kwargs canonical signature.
        # See TODO/audit/smoke_audit_20260516.md §F6.
        base = super()._compute_losses_impl(
            input_batch=input_batch, target_batch=target_batch, epoch=epoch, **kwargs
        )
        batch = self._resolve_legacy_batch(input_batch, kwargs)
        if not isinstance(batch, dict) or "scas_logits" not in batch:
            return base
        density = expected_density(batch["scas_logits"])
        density_loss = self.lambda_density * (density - self.target_rate).pow(2)
        base["loss_scas_density"] = density_loss
        for total_key in ("loss_total", "g_total_loss"):
            if total_key in base:
                base[total_key] = base[total_key] + density_loss
                break
        return base
