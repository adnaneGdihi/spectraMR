r"""Equivariance-Defect Conformal Calibration Strategy (B-1 / Idea 1).

A *degenerate* training strategy: it performs **no gradient updates**. It holds
a frozen, pretrained reconstructor :math:`G_\phi` and a calibration loader,
computes the equivariance-defect non-conformity score
(:mod:`spectramr.core.metrics.equivariance_defect`) per calibration sample over
``K`` random rotations, accumulates the empirical distribution, and emits the
split-conformal quantile :math:`q_\alpha` as a JSON artefact.

This is the reference-free *and* marker-free sibling of
:class:`~spectramr.infrastructure.training.strategies.conformal_calibration_strategy.ConformalCalibrationStrategy`
(which scores against a known marker subspace). They are deliberately distinct
classes.

Dispatch (lead wires this): add to
``spectramr/infrastructure/training/strategy_factory.py::STRATEGY_CLASS_PATHS``::

    "equivariance_conformal": (
        "spectramr.infrastructure.training.strategies.equivariance_conformal_strategy."
        "EquivarianceConformalCalibrationStrategy"
    )
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from spectramr.infrastructure.training.step_io import accepts_step_io
from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy
from spectramr.infrastructure.training.strategies.equivariance_conformal_scoring import (
    run_equivariance_calibration,
)
from spectramr.infrastructure.training.strategies.mixins.metrics_mixin import MetricsMixin
from spectramr.infrastructure.training.strategies.mixins.validation import ValidationMixin

logger = logging.getLogger(__name__)


class EquivarianceConformalCalibrationStrategy(BaseTrainingStrategy, MetricsMixin, ValidationMixin):
    """Calibration-only strategy; performs no gradient updates.

    The training loop is degenerate: :meth:`train_step` is a no-op and
    :meth:`_compute_losses_impl` returns a zero total loss so the base
    contract holds if the orchestrator ever calls it. The science happens in
    :meth:`run_calibration`, which loads the frozen reconstructor, computes the
    equivariance-defect scores, and writes the conformal quantile artefact.
    """

    # #183: the training step never scores train_* metrics; declare none.
    reports_train_metrics = False

    def _setup_strategy_specific_components(self) -> None:
        cfg = getattr(self.config.training, "equivariance_conformal", None)
        if cfg is None:
            # Tolerate absence so the class is importable / testable without the
            # sub-block; run_calibration() will raise if invoked without it.
            self._ec_cfg = None
            logger.warning(
                "EquivarianceConformalCalibrationStrategy initialised without a "
                "training.equivariance_conformal block — strategy is a no-op."
            )
            return
        # The mode-dispatch to the typed TrainingConfigEquivarianceConformal does
        # not fire for these arms (config.training loads as the generic
        # TrainingStrategyConfigSchema), so the sub-block arrives as a raw dict
        # whose ``.alpha``/``.num_rotations_K`` attribute access crashed with
        # "'dict' object has no attribute 'alpha'" (cluster smoke 20260605, only
        # reachable once the upstream 'conditioning' kwarg leak was fixed).
        # Coerce to the SSOT schema (extra='forbid' validates the block).
        if isinstance(cfg, dict):
            from spectramr.config.schemas.training.equivariance_conformal import (
                EquivarianceConformalConfig,
            )

            cfg = EquivarianceConformalConfig(**cfg)
        self._ec_cfg = cfg
        logger.info(
            "EquivarianceConformalCalibrationStrategy: alpha=%s K=%s group=%s",
            cfg.alpha,
            cfg.num_rotations_K,
            cfg.group,
        )

    # ------------------------------------------------------------------
    # Degenerate training loop — no parameter updates.
    # ------------------------------------------------------------------
    @accepts_step_io
    def train_step(
        self, batch: Any, epoch: int = 0, step: int = 0, **kwargs: Any
    ) -> list:  # pragma: no cover - integration-only
        """No-op train step (no optimizer steps); calibration runs in
        :meth:`run_calibration`. Returns an EMPTY step-config list — the old
        dict return hit KeyError('optimizer') in Trainer.execute_step."""
        return []

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:  # pragma: no cover - integration-only
        """Zero loss — the strategy never optimises anything."""
        ref = input_batch if torch.is_tensor(input_batch) else target_batch
        device = ref.device if torch.is_tensor(ref) else self.device
        return {"g_total_loss": torch.zeros((), device=device, requires_grad=False)}

    # ------------------------------------------------------------------
    # The calibration pass.
    # ------------------------------------------------------------------
    def _frozen_reconstructor(self) -> Callable[..., torch.Tensor]:
        """Return the frozen reconstructor callable ``G(y) -> image``.

        Prefers the generator wired onto the strategy context (loaded from the
        configured checkpoint by the model builder); falls back to the
        ``generator_model`` property. The returned callable detaches its output
        so no gradients leak into the (frozen) network.
        """
        _ctx = getattr(self, "context", None)
        model = getattr(_ctx, "generator", None)
        if model is None:
            model = self.generator_model
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        def _fn(y: torch.Tensor, csm: torch.Tensor | None = None) -> torch.Tensor:
            out = model(y) if csm is None else model(y, csm)
            return out.detach()

        return _fn

    @property
    def calibration_needs_test_loader(self) -> bool:
        """True when the arm declares the held-out cal-vs-test exchangeability split."""
        cfg = getattr(self, "_ec_cfg", None)
        return cfg is not None and cfg.exchangeability_split == "held_out"

    def _frozen_checkpoint(self) -> str:
        """The declared ``model.checkpoint_path`` of the frozen reconstructor.

        The model builder's warm-start loads it and raises when the file is missing;
        an undeclared path would leave a random-init network to calibrate (rule 3).
        """
        ckpt = getattr(getattr(self.config, "model", None), "checkpoint_path", None)
        if not ckpt:
            raise ValueError(
                "EquivarianceConformalCalibrationStrategy calibrates a FROZEN "
                "reconstructor: declare model.checkpoint_path (the trained arm's "
                "checkpoints/best.<format>). Scoring a random-init network is meaningless."
            )
        return str(ckpt)

    def run_calibration(self, dataloader: Any, test_dataloader: Any = None) -> dict[str, Any]:
        """Score ``dataloader`` (and ``test_dataloader`` under ``held_out``) and write the artefact.

        Raises:
            ValueError: the ``equivariance_conformal`` block or ``model.checkpoint_path``
                is absent, or ``held_out`` is declared without a test loader.
        """
        cfg = self._ec_cfg
        if cfg is None:
            raise ValueError("run_calibration requires a training.equivariance_conformal block.")
        ckpt = self._frozen_checkpoint()
        recon_fn = self._frozen_reconstructor()
        report = run_equivariance_calibration(
            recon_fn,
            dataloader,
            alpha=cfg.alpha,
            num_rotations=cfg.num_rotations_K,
            group=cfg.group,
            seed=cfg.seed,
            max_shift_px=cfg.max_shift_px,
            device=self.device,
            exchangeability_test=cfg.exchangeability_test,
            exchangeability_ks_pvalue_threshold=cfg.exchangeability_ks_pvalue_threshold,
            exchangeability_split=cfg.exchangeability_split,
            test_dataloader=test_dataloader,
        )
        report["checkpoint_path"] = ckpt
        self._write_artefact(getattr(cfg, "output_artefact", None), report)
        logger.info(
            "Equivariance-defect conformal quantile q_alpha=%.6g over %d samples (%s).",
            report["quantile"],
            report["n_calibration"],
            report["exchangeability_split"],
        )
        return report

    @staticmethod
    def _write_artefact(path: str | None, payload: dict[str, Any]) -> None:
        if not path:
            return
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, default=str))
        logger.info("Wrote equivariance-conformal artefact to %s", out)


__all__ = [
    "EquivarianceConformalCalibrationStrategy",
    "run_equivariance_calibration",
]
