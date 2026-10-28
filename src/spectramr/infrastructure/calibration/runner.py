r"""Post-training conformal-calibration runner (PR-3 / H1, phase 2).

A small use-case class that:

1. Streams ``(prediction, target)`` pairs from a held-out calibration
   loader through a trained model.
2. Fits a :class:`ConformalCalibrator` on the resulting non-conformity
   scores.
3. Optionally evaluates the empirical coverage on a separate test
   loader.

Implements the "post-training phase" described in
``TODO/backlog_paradigm_expansion_roadmap.md`` PR-3.

Why a use case, not a full ``BaseTrainingStrategy``?
----------------------------------------------------
The plan describes the phase as "consuming an existing checkpoint" —
it has no optimiser, no loss, no train loop. Forcing it through the
training-strategy harness would bring AMP, EMA, and validation
machinery that the calibrator doesn't need. A small focused runner is
the right fit; integrating it into the existing pipeline is a
reporting-only concern (PR-3's reporting hook in
``tables/main_results.py``).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import torch

from spectramr.models.blocks.kspace_geometry_encoding import support_forward_kwargs

from .conformal import ConformalCalibrator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CalibrationReport:
    """Result emitted by :class:`ConformalCalibrationRunner`.

    Attributes:
        quantile: Fitted band radius (after calibration).
        n_calibration: Calibration sample count.
        alpha: Mis-coverage rate used.
        empirical_coverage: Test-set coverage if a test loader was
            provided to :py:meth:`run`; ``None`` otherwise.
        avg_set_size: Twice the band radius (matches the plan's
            ``avg_set_size`` reporting column).
    """

    quantile: float
    n_calibration: int
    alpha: float
    empirical_coverage: float | None
    avg_set_size: float


class ConformalCalibrationRunner:
    """Stream-mode conformal-calibration use case.

    Args:
        model: Trained reconstruction model in ``.eval()`` mode.
            The runner does **not** mutate weights.
        calibrator: Pre-built :class:`ConformalCalibrator`. The
            runner only ever calls :py:meth:`fit` /
            :py:meth:`empirical_coverage` on it.
        device: ``torch.device`` to run inference on. Defaults to CPU.
        unpack_batch: Optional callable that maps an arbitrary
            DataLoader batch object to ``(model_input, target)``.
            Default assumes a 2-tuple.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        calibrator: ConformalCalibrator,
        device: torch.device | str = "cpu",
        unpack_batch: Callable[[Any], tuple[Any, torch.Tensor]] | None = None,
    ) -> None:
        self._model = model
        self._calibrator = calibrator
        self._device = torch.device(device)
        self._unpack = unpack_batch or _default_unpack

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    @property
    def calibrator(self) -> ConformalCalibrator:
        """The (possibly fitted) calibrator."""
        return self._calibrator

    def run(
        self,
        calibration_loader: Iterable[Any],
        test_loader: Iterable[Any] | None = None,
    ) -> CalibrationReport:
        """Fit the calibrator on the calibration loader and report.

        Args:
            calibration_loader: Iterable yielding model-input / target
                batches. Must be **disjoint** from the validation
                split — the audit ladder enforces this.
            test_loader: Optional fresh test set used to compute the
                empirical coverage as a sanity check.

        Returns:
            :class:`CalibrationReport` with quantile, sample count, and
            optional empirical coverage.
        """
        self._model.eval()
        with torch.no_grad():
            self._calibrator.fit(self._yield_pairs(calibration_loader))

        emp_cov: float | None = None
        if test_loader is not None:
            emp_cov = self._compute_empirical_coverage(test_loader)

        return CalibrationReport(
            quantile=self._calibrator.quantile,
            n_calibration=self._calibrator.n_calibration,
            alpha=self._calibrator.alpha,
            empirical_coverage=emp_cov,
            avg_set_size=2.0 * self._calibrator.quantile,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _yield_pairs(self, loader: Iterable[Any]) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
        """Lazy generator of ``(prediction, target)`` pairs."""
        for batch in loader:
            inputs, target = self._unpack(batch)
            inputs = _to_device(inputs, self._device)
            target = target.to(self._device)
            pred = self._predict(inputs)
            yield pred, target

    def _predict(self, inputs: Any) -> Any:
        """``model(inputs)``, with the measurement's support for a geometry-plane model.

        Such a model raises without ``mask=``; the calibration batch carries only
        the zero-filled measurement, so its non-zero support is the observation mask.
        """
        if not isinstance(inputs, torch.Tensor):
            return self._model(inputs)
        return self._model(inputs, **support_forward_kwargs(self._model, inputs))

    def _compute_empirical_coverage(self, test_loader: Iterable[Any]) -> float:
        """Average per-batch coverage on the held-out test loader."""
        with torch.no_grad():
            coverages: list[float] = []
            for batch in test_loader:
                inputs, target = self._unpack(batch)
                inputs = _to_device(inputs, self._device)
                target = target.to(self._device)
                pred = self._predict(inputs)
                coverages.append(self._calibrator.empirical_coverage(pred, target))
        if not coverages:
            return 0.0
        return float(sum(coverages) / len(coverages))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_unpack(batch: Any) -> tuple[Any, torch.Tensor]:
    """Minimal unpacker: assumes ``batch == (input, target)``."""
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise ValueError(
        "Default unpacker received a batch that is not a (input, target) "
        "tuple. Pass a custom unpack_batch callable to the runner."
    )


def _to_device(x: Any, device: torch.device) -> Any:
    """Move tensors / dicts of tensors to ``device``; leave other objects intact."""
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, dict):
        return {k: _to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_to_device(v, device) for v in x)
    return x


__all__ = ["CalibrationReport", "ConformalCalibrationRunner"]
