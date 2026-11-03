"""Adversarial Robustness Suite (PR-18 / L5).

Plan: TODO/backlog_paradigm_expansion_roadmap.md §PR-18.

Wired components:

- :meth:`pgd_attack` — projected-gradient-descent attack with
  configurable L∞/L2 norm constraint and step count.
- :meth:`fgsm_attack` — single-step FGSM.
- :meth:`certified_radius` — randomized-smoothing certificate, owned by the
  registered ``certified_radius`` metric.
- :meth:`_compute_losses_impl` — adversarial training: clean (small
  weight) + adversarial (large weight); ``adv_weight: 0`` is the clean-trained
  control and skips the attack.
- :meth:`validation_step` — the attack-time pass: the registered
  ``adversarial_psnr_drop`` PGD at the training budget scores clean and attacked
  PSNR on every validation batch, and ``certified_radius`` its smoothing radius.

Knobs come from ``training.certified_robustness`` when the block is present,
else the class defaults below.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable
from typing import Any, ClassVar

import torch
import torch.nn.functional as F

from spectramr.core.metrics.context import MetricContext
from spectramr.core.metrics.registry import MetricsRegistry
from spectramr.core.metrics.robustness_metrics import _psnr
from spectramr.data.batch_types import TrainingBatch
from spectramr.infrastructure.builders.leaf.adapter_builders import apply_pre_model
from spectramr.models.capabilities import StrategyCapabilities

from .mixins.utils import pick_present
from .mixins.validation_context import fold_depth
from .reconstruction import ReconstructionTrainingStrategy
from .validation_chunking import reconstruction_output

logger = logging.getLogger(__name__)


#: Keys the attack-time validation pass writes (strategy-owned, not registry names).
ATTACK_VALIDATION_KEYS: frozenset[str] = frozenset(
    {
        "val_attack_clean_psnr",
        "val_attack_psnr",
        "val_attack_psnr_drop",
        "val_attack_certified_radius",
    }
)
_ADVERSARIAL_KNOBS: tuple[str, ...] = (
    "epsilon",
    "attack_steps",
    "attack_norm",
    "clean_weight",
    "adv_weight",
    "sigma_smooth",
    "n_smooth_samples",
)
_LIPSCHITZ_KNOBS: frozenset[str] = frozenset(
    {"lambda_lipschitz", "target_sigma", "n_power_iterations"}
)


class AdversarialRobustnessStrategy(ReconstructionTrainingStrategy):
    """PGD/FGSM adversarial training + attack-time validation and smoothing certificate."""

    capabilities: ClassVar[StrategyCapabilities] = dataclasses.replace(
        ReconstructionTrainingStrategy.capabilities,
        emitted_metrics=ATTACK_VALIDATION_KEYS,
    )
    #: Whether this class reads the Lipschitz knobs of the shared config block.
    reads_lipschitz_knobs: ClassVar[bool] = False

    epsilon: float = 0.01
    attack_steps: int = 5
    attack_norm: str = "linf"  # "linf" | "l2"
    sigma_smooth: float = 0.25
    n_smooth_samples: int = 32
    clean_weight: float = 0.1
    adv_weight: float = 0.9

    def _setup_strategy_specific_components(self) -> None:
        super()._setup_strategy_specific_components()
        self._configure_adversarial_knobs(
            getattr(getattr(self.config, "training", None), "certified_robustness", None)
        )

    def _configure_adversarial_knobs(self, cfg: Any) -> None:
        """Read, validate and stamp the adversarial knobs (pitfall 15)."""
        if cfg is not None:
            unread = sorted(set(getattr(cfg, "model_fields_set", ())) & _LIPSCHITZ_KNOBS)
            if unread and not self.reads_lipschitz_knobs:
                raise ValueError(
                    f"training.certified_robustness sets {unread}, which "
                    f"{type(self).__name__} never reads; use training_mode: "
                    "certified_robustness for the Lipschitz penalty."
                )
            for knob in _ADVERSARIAL_KNOBS:
                setattr(self, knob, type(getattr(type(self), knob))(getattr(cfg, knob)))
        if self.attack_norm != "linf":
            raise ValueError(
                f"attack_norm={self.attack_norm!r}: the attack-time validation pass scores an "
                "L-inf PGD (adversarial_psnr_drop), so an L2-trained arm would be graded under "
                "a threat model it was not trained for. Use attack_norm: linf."
            )
        if self.clean_weight == 0.0 and self.adv_weight == 0.0:
            raise ValueError("clean_weight and adv_weight are both 0: the arm has no loss.")
        self.adversarial_provenance: dict[str, Any] = {
            k: getattr(self, k) for k in _ADVERSARIAL_KNOBS
        }
        logger.info("%s knobs: %s", type(self).__name__, self.adversarial_provenance)

    def fgsm_attack(
        self, x: torch.Tensor, loss_fn: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        x_adv = x.detach().clone().requires_grad_(True)
        loss = loss_fn(x_adv)
        grad = torch.autograd.grad(loss, x_adv, retain_graph=False)[0]
        if self.attack_norm == "linf":
            return (x + self.epsilon * grad.sign()).detach()
        # L2
        flat = grad.flatten(1)
        norm = flat.norm(dim=1, keepdim=True).clamp_min(1e-9)
        step = (grad / norm.view(-1, *([1] * (grad.dim() - 1)))) * self.epsilon
        return (x + step).detach()

    def pgd_attack(
        self, x: torch.Tensor, loss_fn: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        delta = torch.zeros_like(x, requires_grad=True)
        step_size = self.epsilon / max(1, self.attack_steps - 1)
        for _ in range(self.attack_steps):
            loss = loss_fn(x + delta)
            grad = torch.autograd.grad(loss, delta, retain_graph=False)[0]
            if self.attack_norm == "linf":
                delta = (delta + step_size * grad.sign()).clamp(-self.epsilon, self.epsilon)
            else:
                step = grad / grad.flatten(1).norm(dim=1).clamp_min(1e-9).view(
                    -1, *([1] * (grad.dim() - 1))
                )
                delta = delta + step_size * step
                # Project to L2 ball
                d_norm = delta.flatten(1).norm(dim=1).clamp_min(1e-9)
                proj = (self.epsilon / d_norm).clamp_max(1.0).view(-1, *([1] * (delta.dim() - 1)))
                delta = delta * proj
            delta = delta.detach().requires_grad_(True)
        return (x + delta).detach()

    def certified_radius(
        self, predict: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor
    ) -> torch.Tensor:
        """Per-sample randomized-smoothing radius, from the registered metric."""
        return self._radius_metric().per_sample(predict, x)

    def _radius_metric(self) -> Any:
        return MetricsRegistry.get(
            "certified_radius", sigma=self.sigma_smooth, n_samples=self.n_smooth_samples
        )

    @staticmethod
    def _attack_inputs(input_batch: Any, target_batch: Any, batch: Any) -> tuple[Any, Any]:
        """The tensor to attack and its target: the prepared tensors first, then the batch.

        ``train_step`` passes the device-moved, model-prepared tensors alongside the
        batch, which the training loop delivers as a ``TrainingBatch``, not a dict.
        """
        if isinstance(batch, TrainingBatch):
            return pick_present(input_batch, batch.input), pick_present(target_batch, batch.target)
        if isinstance(batch, dict):
            return (
                pick_present(input_batch, batch.get("kspace"), batch.get("input")),
                pick_present(target_batch, batch.get("target")),
            )
        return input_batch, target_batch

    @staticmethod
    def _with_input(batch: Any, adv_x: torch.Tensor) -> Any:
        """The same batch carrying the attacked input, in the batch's own container."""
        if isinstance(batch, TrainingBatch):
            return dataclasses.replace(batch, input=adv_x)
        if isinstance(batch, dict):
            adv_batch = dict(batch)
            adv_batch["kspace"] = adv_x
            if "input" in batch:
                adv_batch["input"] = adv_x
            return adv_batch
        return batch

    def _compute_losses_impl(
        self,
        input_batch: Any = None,
        target_batch: Any = None,
        epoch: int = 0,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        batch = self._resolve_legacy_batch(input_batch, kwargs)
        x, target = self._attack_inputs(input_batch, target_batch, batch)
        gen = getattr(self.env, "generator", None)
        if gen is None or x is None or target is None:
            raise ValueError(
                f"{type(self).__name__} needs a generator, an input and a target to train; got "
                f"generator={type(gen).__name__}, input={type(x).__name__}, "
                f"target={type(target).__name__}."
            )

        clean = super()._compute_losses_impl(
            input_batch=x, target_batch=target, epoch=epoch, **kwargs
        )
        # The clean-trained control: no attack, so no PGD forwards on the step.
        adv: dict[str, Any] = {}
        if self.adv_weight > 0.0:

            def loss_fn(z: torch.Tensor) -> torch.Tensor:
                return F.l1_loss(gen(z), target)

            adv_x = self.pgd_attack(x, loss_fn)
            adv_batch = self._with_input(batch, adv_x)
            adv = super()._compute_losses_impl(
                input_batch=adv_x,
                target_batch=target,
                epoch=epoch,
                **{**kwargs, "batch": adv_batch},
            )

        merged: dict[str, torch.Tensor] = {}
        # ``g_total_loss`` is the key base.py reads for backward; the parent also
        # writes a ``loss`` alias, never ``loss_total``.
        _total_keys = ("g_total_loss", "loss_total", "loss")
        clean_total = pick_present(*(clean.get(k) for k in _total_keys))
        adv_total = pick_present(*(adv.get(k) for k in _total_keys))
        if clean_total is not None and not adv:
            merged["g_total_loss"] = self.clean_weight * clean_total
        elif clean_total is not None and adv_total is not None:
            merged["g_total_loss"] = self.clean_weight * clean_total + self.adv_weight * adv_total
        _skip = {"g_total_loss", "loss_total", "loss"}
        for k, v in clean.items():
            if k not in _skip and isinstance(v, torch.Tensor):
                merged[f"clean_{k}"] = v.detach()
        for k, v in adv.items():
            if k not in _skip and isinstance(v, torch.Tensor):
                merged[f"adv_{k}"] = v.detach()
        return merged

    def validation_step(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        *,
        batch_data: Any = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """The usual metrics plus the attack-time pass on the same batch."""
        context = kwargs.get("batch_context")
        metrics = super().validation_step(
            input_batch, target_batch, batch_data=batch_data, **kwargs
        )
        if not isinstance(metrics, dict) or input_batch is None or target_batch is None:
            return metrics
        metrics.update(self._attack_validation_metrics(input_batch, target_batch, context))
        return metrics

    def _attack_validation_metrics(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        batch_context: dict[str, Any] | None,
    ) -> dict[str, float]:
        """Clean vs PGD-attacked PSNR at the training budget, and the smoothing radius.

        Both PSNRs come from one scorer against one target, so the drop is not
        mixed with the data-range handling of the registry's ``val_psnr``.
        """
        if self.validation_reads_context_measurement:
            raise ValueError(
                f"{type(self).__name__} reads its measurement from the batch context, so a "
                "PGD on the tensor handed to the forward would perturb an ignored input."
            )
        x, target = apply_pre_model(self.adapter_chains, input_batch, target_batch)
        ctx = batch_context or {"use_dc": False, "measured_kspace": None}

        def forward(z: torch.Tensor) -> torch.Tensor:
            return fold_depth(reconstruction_output(self._validation_forward(z, ctx)))[0]

        target = fold_depth(target)[0]
        with torch.no_grad():
            clean = forward(x).to(target.device)
        drop_metric = MetricsRegistry.get(
            "adversarial_psnr_drop", epsilon=self.epsilon, steps=self.attack_steps
        )
        drop = float(drop_metric(clean, target, model=forward, model_input=x))
        clean_psnr = _psnr(clean, target)
        radius = self._radius_metric()(context=MetricContext(reconstructor=forward), model_input=x)
        return {
            "val_attack_clean_psnr": clean_psnr,
            "val_attack_psnr": clean_psnr - drop,
            "val_attack_psnr_drop": drop,
            "val_attack_certified_radius": float(radius),
        }
