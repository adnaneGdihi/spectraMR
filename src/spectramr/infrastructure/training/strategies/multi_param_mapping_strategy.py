"""One-Shot Multi-Parameter Mapping Strategy (idea 10).

Plan: TODO/integration_plan_ulf_cheap_fast_mri.md §10.

Joint estimation of (T1, T2, PD, segmentation) from a single
multi-contrast acquisition. Uses Kendall-style learnable uncertainty
weighting + Bloch-self-consistency regulariser.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import torch

from spectramr.config.schemas.enums import Regime, Task
from spectramr.data.batch_types import read_batch_field
from spectramr.infrastructure.distributed.owned_modules import OWNED_MODULE, OwnedParameter
from spectramr.models.capabilities import StrategyCapabilities

from .reconstruction import ReconstructionTrainingStrategy

logger = logging.getLogger(__name__)


class OneShotMultiParameterStrategy(ReconstructionTrainingStrategy):
    """Reconstruction-derived strategy that emits multi-parameter heads."""

    # parallel_compat: The log-sigma parameter is an owned module, DDP-wrapped, read via call_owned.
    parallel_compat = frozenset({"replicated"})

    #: Quantitative MRI: joint (T1, T2, PD) estimation anchored by a Bloch
    #: signal-synthesis consistency loss that RAISES rather than degrade when the
    #: physics anchor cannot be fed.
    capabilities: ClassVar[StrategyCapabilities] = StrategyCapabilities(
        workflows=frozenset({Regime.QUANTITATIVE}),
        tasks=frozenset({Task.PARAMETER_MAPPING}),
    )

    @classmethod
    def owned_module_declarations(cls, config: Any) -> tuple[tuple[str, str], ...]:
        cfg = getattr(getattr(config, "training", None), "multi_parameter", None)
        weighted = getattr(cfg, "uncertainty_weighting", False)
        own = (("_log_sigmas", OWNED_MODULE),) if weighted else ()
        return (*super().owned_module_declarations(config), *own)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        cfg = getattr(self.config.training, "multi_parameter", None)
        if cfg is None:
            raise ValueError(
                "OneShotMultiParameterStrategy requires training.multi_parameter "
                "to be set in the YAML."
            )
        self._mp_cfg = cfg

        # Optional: learnable log-sigma per task for Kendall weighting.
        if cfg.uncertainty_weighting:
            n_tasks = len(cfg.parameters)
            self._log_sigmas = torch.nn.Parameter(torch.zeros(n_tasks, device=self.device))
            self.register_owned_module("_log_sigmas", OwnedParameter(self._log_sigmas))
            opt = getattr(self.env, "opt_g", None)
            if opt is not None:
                opt.add_param_group({"params": [self._log_sigmas], "lr": 1e-3})
        else:
            self._log_sigmas = None

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Override: route to multi-parameter heads + optional Bloch consistency.

        Signature aligned with the canonical contract (audit F1). The
        full batch dict is read from ``kwargs["batch"]`` (the convention
        already used by ``physics_driven_strategy``).
        """
        outputs: dict[str, torch.Tensor] = {}
        device = self.device
        # Prefer ``self.env.generator`` (the DI-injected one) — the
        # previous ``self.context.generator`` lookup hit a non-existent
        # attribute and silently returned a zero loss (audit F3).
        gen = self.env.generator
        if gen is None:
            raise RuntimeError(
                "multi_param_mapping strategy: env.generator is None — "
                "the DI container did not produce a generator. Check the "
                "ModelBuilder wiring."
            )

        # ``x or {}`` evaluates ``bool(x)``, which RAISES on a multi-element
        # tensor ("Boolean value of Tensor ... is ambiguous"). A tensor batch
        # therefore died here with an error naming neither the strategy nor the
        # missing contract. Same hazard the base orchestrator documents at
        # ``base.py:_resolve_legacy_batch``; use an explicit None test.
        batch = kwargs.get("batch")
        if batch is None:
            batch = {}

        # Forward pass — we expect the model to return a dict of
        # parameter maps (e.g. {"t1": ..., "t2": ..., "pd": ...}).
        # Guard BEFORE the generator call. This read used to substitute
        # ``input_batch`` for a non-dict batch and hand ``None`` straight to
        # ``gen()`` for a dict batch missing the key -- and the actionable
        # ValueError below is gated on ``bloch_consistency_lambda > 0``, so a
        # lambda-0 arm reached ``gen(None)`` and died on an opaque arity error
        # naming nothing. No dataset emits 'observed_contrasts' today.
        observed = read_batch_field(batch, "observed_contrasts")
        if observed is None:
            raise ValueError(
                "multi_param_mapping requires batch['observed_contrasts'] of "
                "shape [B, N_c, H, W] (the multi-contrast observations the "
                "parameter maps are fitted from), but the batch supplies "
                f"{sorted(k for k in batch) if isinstance(batch, dict) else type(batch).__name__!r}. "
                "Supply a qMRI multi-contrast dataset that emits the key; "
                "substituting the input tensor would fit T1/T2/PD against the "
                "wrong observations."
            )
        pred = gen(observed)
        if not isinstance(pred, dict):
            # Single-tensor backbones — fall back to base recon path.
            return super()._compute_losses_impl(
                input_batch=input_batch,
                target_batch=target_batch,
                epoch=epoch,
                **kwargs,
            )

        # Per-task losses + Kendall weighting.
        task_losses: dict[str, torch.Tensor] = {}
        present = [
            (i, name)
            for i, name in enumerate(self._mp_cfg.parameters)
            if name in pred and name in batch
        ]
        # One DDP forward per step, and only when its output reaches the loss.
        log_sigmas = (
            self.call_owned("_log_sigmas") if self._log_sigmas is not None and present else None
        )
        for i, name in present:
            task_loss = torch.nn.functional.l1_loss(pred[name], batch[name])
            if log_sigmas is not None:
                ls = log_sigmas[i]
                weighted = 0.5 * torch.exp(-2.0 * ls) * task_loss + ls
                task_losses[f"loss_{name}"] = weighted
            else:
                task_losses[f"loss_{name}"] = task_loss

        outputs.update(task_losses)
        outputs["loss_total"] = (
            sum(task_losses.values()) if task_losses else torch.tensor(0.0, device=device)
        )

        # Optional: Bloch self-consistency. The forward-synthesis loss consumes
        # the predicted (T1, T2, PD) maps, the multi-contrast observations, and
        # per-contrast acquisition params. If the user asked for it
        # (lambda > 0) but the data cannot supply those tensors, fail loud
        # rather than swallow the error to a warning (NN#3 / pitfall #10) — a
        # silently-dropped "physics anchor" is a facade (pitfall #16).
        if self._mp_cfg.bloch_consistency_lambda > 0.0:
            from spectramr.models.losses.bloch_signal_synthesis_consistency_loss import (
                BlochSignalSynthesisConsistencyLoss,
                split_t1_t2_pd,
            )

            acq = read_batch_field(batch, "acquisition_params")
            if acq is None or not (isinstance(observed, torch.Tensor) and observed.dim() == 4):
                raise ValueError(
                    "multi_param_mapping: bloch_consistency_lambda>0 requires "
                    "batch['acquisition_params'] (a list of per-contrast "
                    "{TR_ms, TE_ms, FA_deg}) and batch['observed_contrasts'] of "
                    "shape [B, N_c, H, W]. Supply a qMRI multi-contrast dataset, "
                    "or set training.multi_parameter.bloch_consistency_lambda: 0.0."
                )
            if not hasattr(self, "_bloch_loss"):
                self._bloch_loss = BlochSignalSynthesisConsistencyLoss().to(device)
            t1, t2, pd = split_t1_t2_pd(pred)
            bloch_l = self._bloch_loss(
                t1, t2, pd, observed_contrasts=observed, acquisition_params=acq
            )
            outputs["loss_bloch"] = self._mp_cfg.bloch_consistency_lambda * bloch_l
            outputs["loss_total"] = outputs["loss_total"] + outputs["loss_bloch"]

        return outputs
