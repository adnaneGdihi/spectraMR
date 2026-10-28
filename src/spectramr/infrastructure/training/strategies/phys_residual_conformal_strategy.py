r"""PR-CC calibration strategy: Physics-Residual Conformal hallucination certificate.

A degenerate, no-gradient calibration pass (mirrors B-1
``EquivarianceConformalCalibrationStrategy``). It freezes a tissue-parameter
estimator (a model exposing ``predict_parameters`` or a 3-channel ``forward``),
computes the physics-residual nonconformity score over a held-out calibration
split by re-rendering the observed contrast through the Bloch forward, and emits
the split-conformal quantile + empirical coverage, optionally with Mondrian
(per-:math:`B_0`) field stratification.

The science is the module-level :func:`run_phys_residual_calibration` (testable
without a training environment); the strategy wires the config + frozen model
into it. The post-training hook ``_maybe_run_calibration`` (pipelines/train.py)
auto-dispatches any strategy exposing ``run_calibration(dataloader)``.

Honest scope: the physics residual is a *meaningful* certificate only when the
(PD, T1, T2) parameters are at least weakly identified (quantitative / multi-
field-with-reference data). On single-field single-contrast magnitude (M4Raw
0.3T) the estimate is ill-posed and the residual certifies render-consistency,
not parameter identifiability (pitfall #19) -- the rigorous proof is the Tier-2
phantom probe.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import torch

from spectramr.infrastructure.calibration.phys_residual_certificate import (
    PhysResidualConformalCertificate,
)
from spectramr.infrastructure.training.step_io import accepts_step_io
from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy
from spectramr.infrastructure.training.strategies.mixins.metrics_mixin import MetricsMixin
from spectramr.infrastructure.training.strategies.mixins.validation import ValidationMixin

logger = logging.getLogger(__name__)

_INPUT_KEYS = ("input", "image", "kspace", "lr_image", "y")


def _extract_input_target_b0(
    batch: Any, default_b0: float
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Extract ``(model_input, observed_image, b0)`` from a batch.

    Accepts a dict (``input``/``image``/``kspace``/``y`` + ``target`` + optional
    ``field_strength``) or a 2-tuple ``(input, target)``. Raises otherwise (no
    silent fallback).
    """
    if isinstance(batch, dict):
        x = next((batch[k] for k in _INPUT_KEYS if k in batch), None)
        y = batch.get("target")
        if x is None or y is None:
            raise ValueError(
                f"PR-CC batch dict needs an input key {_INPUT_KEYS} and 'target'; "
                f"got keys {sorted(batch)}."
            )
        b0 = batch.get("field_strength", default_b0)
    elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
        x, y = batch[0], batch[1]
        b0 = default_b0
    else:
        raise ValueError(
            f"PR-CC cannot extract (input, target, b0) from batch of type {type(batch).__name__}."
        )
    if isinstance(b0, torch.Tensor):
        b0 = float(b0.flatten()[0].item())
    return x, y, float(b0)


def run_phys_residual_calibration(
    param_map_fn: Callable[[torch.Tensor], torch.Tensor],
    dataloader: Any,
    *,
    alpha: float,
    acquisition: dict[str, float],
    contrast_type: str,
    stratify_by: str,
    calibration_fraction: float,
    device: Any = "cpu",
    default_b0: float = 0.3,
) -> dict[str, Any]:
    """Run the PR-CC split-conformal calibration over a dataloader.

    Args:
        param_map_fn: Frozen estimator ``x -> (rho, T1, T2)`` map ``[B, 3, H, W]``.
        dataloader: Iterable of batches (dict or 2-tuple).
        alpha: Mis-coverage rate.
        acquisition: ``{"TR","TE","TI"}`` (ms).
        contrast_type: Contrast routed to the Bloch forward.
        stratify_by: ``"none"`` (marginal) or ``"b0"`` (Mondrian per-field).
        calibration_fraction: Fraction used as the calibration split.
        device: Compute device.
        default_b0: Field strength when a batch carries none.

    Returns:
        The certificate report dict.

    Raises:
        ValueError: empty loader.
    """
    units: list[tuple[torch.Tensor, torch.Tensor, float]] = []
    with torch.no_grad():
        for batch in dataloader:
            x, y, b0 = _extract_input_target_b0(batch, default_b0)
            x = x.to(device)
            y = y.to(device)
            params = param_map_fn(x).detach()
            units.append((params, y, b0))
    if not units:
        raise ValueError("PR-CC calibration loader is empty.")

    mondrian = stratify_by == "b0"
    if mondrian:
        by_field: dict[float, list] = defaultdict(list)
        for u in units:
            by_field[u[2]].append(u)
        calib_units, test_units = [], []
        for field_units in by_field.values():
            k = max(1, int(len(field_units) * calibration_fraction))
            calib_units += field_units[:k]
            test_units += field_units[k:]
    else:
        k = max(1, int(len(units) * calibration_fraction))
        calib_units, test_units = units[:k], units[k:]

    cert = PhysResidualConformalCertificate(
        alpha=alpha,
        acquisition=acquisition,
        contrast_type=contrast_type,
        stratify_by="b0" if mondrian else None,
    )
    cert.calibrate([(p, t, b0) for p, t, b0 in calib_units])

    eval_units = test_units if test_units else calib_units
    coverages: list[float] = []
    flagged: list[float] = []
    for p, t, b0 in eval_units:
        try:
            res = cert.certify(p, t, b0=b0)
        except KeyError:
            continue  # unseen field under Mondrian
        coverages.append(res.coverage)
        flagged.append(float(res.flagged.float().mean().item()))

    coverage = sum(coverages) / len(coverages) if coverages else float("nan")
    flagged_fraction = sum(flagged) / len(flagged) if flagged else float("nan")

    report: dict[str, Any] = {
        "score_fn": "physics_residual",
        "alpha": float(alpha),
        "n_calibration": len(calib_units),
        "stratify_by": stratify_by,
        "contrast_type": contrast_type,
        "acquisition": dict(acquisition),
        "coverage_at_alpha": coverage,
        "empirical_coverage": coverage,
        "flagged_fraction": flagged_fraction,
    }
    if mondrian:
        fields = sorted({b0 for _, _, b0 in calib_units})
        per_field = {str(v): cert.quantile(b0=v) for v in fields}
        report["per_field_quantile"] = per_field
        finite = [q for q in per_field.values() if q != float("inf")]
        report["quantile"] = min(finite) if finite else float("inf")
    else:
        report["per_field_quantile"] = {}
        report["quantile"] = cert.quantile()
    return report


class PhysResidualConformalStrategy(BaseTrainingStrategy, MetricsMixin, ValidationMixin):
    """No-gradient PR-CC calibration pass over a frozen parameter estimator."""

    # Computes its own objective and reads no `losses:` block, so an empty builder stack
    # is the design (LossBuilder.validate) and any declared image loss is a decoy.
    inline_losses: ClassVar[frozenset[str] | None] = frozenset()
    folds_image_losses: ClassVar[bool | None] = False

    def _setup_strategy_specific_components(self) -> None:
        cfg = getattr(self.config.training, "phys_residual_conformal", None)
        if cfg is None:
            self._prcc_cfg = None
            logger.warning(
                "PhysResidualConformalStrategy: no training.phys_residual_conformal "
                "block; run_calibration will raise."
            )
            return
        if isinstance(cfg, dict):
            # _MODE_DISPATCH may not fire -> coerce the raw dict to the SSOT schema.
            from spectramr.config.schemas.training.phys_residual_conformal import (
                PhysResidualConformalConfig,
            )

            cfg = PhysResidualConformalConfig(**cfg)
        self._prcc_cfg = cfg
        logger.info("PhysResidualConformalStrategy configured (alpha=%s).", cfg.alpha)

    @accepts_step_io
    def train_step(self, batch: Any, epoch: int = 0, step: int = 0, **kwargs: Any) -> list:
        # No optimizer step: an empty step-config list is the calibration contract.
        return []

    def _compute_losses_impl(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        # Defensive guard only (never called because train_step returns []).
        return {"g_total_loss": torch.zeros((), device=self.device, requires_grad=False)}

    def _frozen_param_map_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        ctx = getattr(self, "context", None)
        model = getattr(ctx, "generator", None)
        if model is None:
            model = self.generator_model
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        def _fn(x: torch.Tensor) -> torch.Tensor:
            if hasattr(model, "predict_parameters"):
                pd, t1, t2 = model.predict_parameters(x)
                return torch.cat([pd, t1, t2], dim=1).detach()
            out = model(x)
            if out.ndim >= 2 and out.shape[1] == 3:
                return out.detach()
            raise ValueError(
                "PR-CC base model must output (rho, T1, T2) parameter maps via "
                "predict_parameters() or a 3-channel forward; got output shape "
                f"{tuple(out.shape)}."
            )

        return _fn

    def run_calibration(self, dataloader: Any) -> dict[str, Any]:
        cfg = self._prcc_cfg
        if cfg is None:
            raise ValueError(
                "PhysResidualConformalStrategy.run_calibration requires the "
                "training.phys_residual_conformal config block."
            )
        param_fn = self._frozen_param_map_fn()
        default_b0 = float(
            (self.config.physics.field_strength if self.config.physics is not None else 0.3) or 0.3
        )
        report = run_phys_residual_calibration(
            param_fn,
            dataloader,
            alpha=cfg.alpha,
            acquisition={"TR": cfg.tr_ms, "TE": cfg.te_ms, "TI": cfg.ti_ms},
            contrast_type=cfg.contrast_type,
            stratify_by=cfg.stratify_by,
            calibration_fraction=cfg.calibration_fraction,
            device=self.device,
            default_b0=default_b0,
        )
        self._write_artefact(getattr(cfg, "output_artefact", None), report)
        return report

    @staticmethod
    def _write_artefact(path: str | None, payload: dict[str, Any]) -> None:
        if not path:
            return
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, default=str))


__all__ = [
    "PhysResidualConformalStrategy",
    "run_phys_residual_calibration",
]
