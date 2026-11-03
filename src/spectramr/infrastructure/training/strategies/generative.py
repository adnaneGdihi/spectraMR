"""Generative (likelihood-based) training strategy.

Added 2026-05-22 for the Phase-3 normalising-flow models (``glow``,
``equivariant_flow``) re-implemented from the deletion ledger
(TODO/deleted_model_types_reimplementation_plan.md). Those models declare
``training_mode="generative"`` and are trained by maximum likelihood, a
paradigm the framework previously lacked.

The strategy is deliberately model-agnostic: it asks the model for a
likelihood in priority order
  1. ``model.log_prob(x)``        — exact-likelihood flows (Glow, equivariant)
  2. ``model.training_loss(x)``   — models that expose their own objective
  3. ``model(x) -> (z, log_det)`` — generic flow contract; standard-normal NLL
and only falls back to a reconstruction MSE if none of those are available.
There is no silent default to an unrelated objective (CLAUDE.md pitfall #9).
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn.functional as F

from spectramr.core.metrics.registry import MetricsRegistry

from .base import BaseTrainingStrategy

logger = logging.getLogger(__name__)

_NLL_METRIC = "nll_bits_per_dim"


class GenerativeTrainingStrategy(BaseTrainingStrategy):
    """Maximum-likelihood training for generative density models."""

    def __init__(self, env: Any = None, device: Any = None, **kwargs: Any) -> None:
        super().__init__(env=env, device=device, **kwargs)
        self._verify_strategy_config(expected_modes=("generative",))

    @staticmethod
    def _standard_normal_log_prob(z: torch.Tensor, log_det: torch.Tensor) -> torch.Tensor:
        """Per-sample ``log p(x) = log N(z; 0, I) + log|det J|`` in nats ``[B]``."""
        z_flat = z.reshape(z.shape[0], -1)
        d = z_flat.shape[1]
        log_pz = -0.5 * (z_flat**2).sum(dim=1) - 0.5 * d * math.log(2 * math.pi)
        return log_pz + log_det.reshape(z.shape[0])

    @classmethod
    def _standard_normal_nll(cls, z: torch.Tensor, log_det: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood under a standard-normal prior, averaged over the batch."""
        return -cls._standard_normal_log_prob(z, log_det).mean()

    @staticmethod
    def _density_input(input_batch: Any, target_batch: Any) -> torch.Tensor:
        """The tensor whose density is modelled; training and validation share it."""
        return target_batch if target_batch is not None else input_batch

    @staticmethod
    def _per_sample_log_prob(model: Any, x: torch.Tensor) -> torch.Tensor:
        """Per-sample ``log p(x)`` in nats ``[B]`` from ``log_prob`` or ``(z, log_det)``."""
        log_prob = getattr(model, "log_prob", None)
        if callable(log_prob):
            return log_prob(x).reshape(x.shape[0])
        out = model(x)
        if isinstance(out, tuple) and len(out) == 2:
            return GenerativeTrainingStrategy._standard_normal_log_prob(*out)
        raise TypeError(
            f"GenerativeTrainingStrategy validation needs an exact likelihood, but "
            f"{type(model).__name__} exposes neither log_prob(x) nor a (z, log_det) forward."
        )

    @torch.no_grad()
    def validation_step(
        self, input_batch: torch.Tensor, target_batch: torch.Tensor, **kwargs: Any
    ) -> dict[str, float]:
        """Score the model's own likelihood: ``val_loss`` (nats) and ``nll_bits_per_dim``.

        The registered ``nll_bits_per_dim`` metric reads ``preds`` as a per-element
        NLL in nats, so it receives ``-log p(x) / D`` broadcast over each sample's
        elements (the mean is unchanged). The flow latent ``z`` is never scored.
        """
        del kwargs  # batch_idx from the loop; the likelihood needs no context
        computer = self.validation_metrics_computer
        requested = computer.get_metric_names()
        likelihood_names = {_NLL_METRIC, *MetricsRegistry.get_aliases(_NLL_METRIC)}
        unsupported = sorted(set(requested) - likelihood_names)
        if unsupported:
            raise ValueError(
                f"GenerativeTrainingStrategy scores a density, not an image: "
                f"validation.scoring.compute names {unsupported}, which need a pixel "
                f"prediction. Use {_NLL_METRIC!r}."
            )
        model = self.generator_model
        x = self._density_input(input_batch, target_batch).to(self.device, non_blocking=True)
        was_training = model.training
        model.eval()
        try:
            log_px = self._per_sample_log_prob(model, x).float()
        finally:
            model.train(was_training)
        nll = -log_px
        metrics: dict[str, Any] = {"val_loss": nll.mean()}
        if requested:
            nats_per_dim = (nll / x[0].numel()).view(-1, *([1] * (x.ndim - 1))).expand_as(x)
            metrics.update(computer.compute(nats_per_dim, x))
        return self._convert_metrics_to_floats(metrics)

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        # Lazy fallback: only reach for ``self.generator_model`` (→
        # ``self.env.generator``) when the caller did NOT pass an explicit
        # model. ``dict.get`` evaluates its default eagerly, so the old form
        # touched ``self.env`` even on the explicit-model path — crashing any
        # caller that supplied ``model=`` without a full env.
        model = kwargs["model"] if "model" in kwargs else self.generator_model
        x = self._density_input(input_batch, target_batch)

        log_prob = getattr(model, "log_prob", None)
        if callable(log_prob):
            if getattr(model, "conditions_on_input", False):
                # A conditional density p(target | input): the input is its condition.
                if input_batch is None:
                    raise ValueError(f"{type(model).__name__} conditions on the input batch.")
                nll = -log_prob(x, condition=input_batch).mean()
            else:
                nll = -log_prob(x).mean()
            return {"g_total_loss": nll, "loss": nll, "loss_nll": nll}

        training_loss = getattr(model, "training_loss", None)
        if callable(training_loss):
            loss = training_loss(x)
            return {"g_total_loss": loss, "loss": loss, "loss_objective": loss}

        out = model(x)
        if isinstance(out, tuple) and len(out) == 2:
            z, log_det = out
            nll = self._standard_normal_nll(z, log_det)
            return {"g_total_loss": nll, "loss": nll, "loss_nll": nll}

        # Last resort: an autoencoder-style reconstruction objective. This is
        # explicit (not a silent fallback) and only reached when the model
        # exposes none of the likelihood contracts above.
        pred = out[0] if isinstance(out, tuple) else out
        mse = F.mse_loss(pred, x)
        logger.warning(
            "GenerativeTrainingStrategy: model %s exposes no log_prob / "
            "training_loss / (z, log_det) contract; falling back to "
            "reconstruction MSE.",
            type(model).__name__,
        )
        return {"g_total_loss": mse, "loss": mse, "loss_recon": mse}
