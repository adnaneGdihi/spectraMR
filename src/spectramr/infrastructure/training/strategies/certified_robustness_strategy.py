"""Certified-robust reconstruction training (A-8.3 CertRob, component C).

Extends :class:`AdversarialRobustnessStrategy` (PGD/FGSM adversarial training +
randomized-smoothing diagnostic) with a **certified global Lipschitz bound**: an
opt-in spectral penalty
(:class:`~spectramr.models.losses.lipschitz_penalty.LipschitzSpectralPenalty`)
that drives every weight layer's spectral norm toward ``target_sigma`` so the
provable Lipschitz upper bound
:math:`\\prod_i\\sigma_i` — certified at validation by the
``composed_spectral_norm_bound`` metric — shrinks. A reconstructor with a small
Lipschitz constant cannot amplify a worst-case input perturbation (Antun et al.,
PNAS 2020).

The headline one-knob ablation is ``training.certified_robustness.lambda_lipschitz``
(``0`` = adversarial-training-only control; ``>0`` = certified-robust treatment).
The adversarial knobs (epsilon, attack steps, clean/adv weights, ...) are read from
the ``certified_robustness`` block by the parent; this class reads the Lipschitz
knobs and stamps them with the live certified bound into run provenance.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, ClassVar

import torch

from spectramr.core.metrics.lipschitz_bound_metrics import (
    ComposedSpectralNormBound,
    MaxLayerSpectralNorm,
)
from spectramr.models.capabilities import StrategyCapabilities
from spectramr.models.losses.lipschitz_penalty import LipschitzSpectralPenalty

from .adversarial_robustness_strategy import AdversarialRobustnessStrategy


class CertifiedRobustnessStrategy(AdversarialRobustnessStrategy):
    """Adversarial training + opt-in certified Lipschitz-bound penalty (A-8.3)."""

    capabilities: ClassVar[StrategyCapabilities] = dataclasses.replace(
        AdversarialRobustnessStrategy.capabilities,
        emitted_metrics=AdversarialRobustnessStrategy.capabilities.emitted_metrics
        | {"val_composed_spectral_norm_bound", "val_max_layer_spectral_norm"},
    )
    reads_lipschitz_knobs: ClassVar[bool] = True

    def _setup_strategy_specific_components(self) -> None:
        super()._setup_strategy_specific_components()
        # The parent reads the adversarial knobs of this block; the Lipschitz ones are ours.
        cfg = getattr(self.config.training, "certified_robustness", None)

        # Lipschitz certification knobs (the A-8.3 one-knob).
        self._lambda_lipschitz = float(getattr(cfg, "lambda_lipschitz", 0.0)) if cfg else 0.0
        self._target_sigma = float(getattr(cfg, "target_sigma", 1.0)) if cfg else 1.0
        self._n_power_iters = int(getattr(cfg, "n_power_iterations", 1)) if cfg else 1
        self._lip_penalty = LipschitzSpectralPenalty(
            target_sigma=self._target_sigma, n_power_iterations=self._n_power_iters
        )
        self._bound_metric = ComposedSpectralNormBound()
        self._max_layer_metric = MaxLayerSpectralNorm()

        # Provenance stamp (#15): the resolved knobs + initial certified bound.
        if getattr(self, "logging_service", None):
            gen = getattr(self.env, "generator", None)
            bound = self._bound_metric(model=gen) if gen is not None else float("nan")
            self.logging_service.log_info(
                f"[A-8.3 CertRob] lambda_lipschitz={self._lambda_lipschitz}, "
                f"target_sigma={self._target_sigma}, n_power_iters={self._n_power_iters}, "
                f"epsilon={self.epsilon}, attack_norm={self.attack_norm}; "
                f"initial composed_spectral_norm_bound={bound:.4g}"
            )

    def _compute_losses_impl(
        self,
        input_batch: Any = None,
        target_batch: Any = None,
        epoch: int = 0,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        losses = super()._compute_losses_impl(
            input_batch=input_batch, target_batch=target_batch, epoch=epoch, **kwargs
        )
        gen = getattr(self.env, "generator", None)
        lam = getattr(self, "_lambda_lipschitz", 0.0)
        # Attach the penalty to the canonical grad-carrying total (``g_total_loss``,
        # the key base.py reads for backward) — NOT ``loss_total`` (which the parent
        # never emits). Require requires_grad so the penalty actually trains the
        # weights; otherwise the mechanism would be inert (#16).
        total = losses.get("g_total_loss") if isinstance(losses, dict) else None
        if (
            lam > 0.0
            and gen is not None
            and isinstance(total, torch.Tensor)
            and total.requires_grad
        ):
            penalty = self._lip_penalty(model=gen)
            losses["g_total_loss"] = total + lam * penalty
            losses["lipschitz_penalty"] = penalty.detach()
        return losses

    def validation_step(self, input_batch: Any, target_batch: Any, **kwargs: Any) -> Any:
        """Emit the global certified-bound witnesses alongside the usual metrics.

        Only finite witness values are stored: the bound metric returns ``nan``
        when no module is in scope, and a ``nan`` silently breaks early-stopping
        (``nan < best`` is always ``False`` -> the monitor never improves). The
        finite-guard keeps that failure mode out of the metrics dict (MEDIUM,
        A-8.3 review 2026-06-16).
        """
        metrics = super().validation_step(input_batch, target_batch, **kwargs)
        if isinstance(metrics, dict):
            gen = getattr(self.env, "generator", None)
            if gen is not None:
                bound = self._bound_metric(model=gen)
                max_layer = self._max_layer_metric(model=gen)
                if math.isfinite(bound):
                    metrics["val_composed_spectral_norm_bound"] = bound
                if math.isfinite(max_layer):
                    metrics["val_max_layer_spectral_norm"] = max_layer
        return metrics
