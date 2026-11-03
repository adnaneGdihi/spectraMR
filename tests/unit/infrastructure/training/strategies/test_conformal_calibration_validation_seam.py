"""Regression: b17_dice_risk_calibration must survive validation to emit its certificate.

``ConformalCalibrationStrategy`` runs ``anatomy_field_renderer``, whose ``forward`` takes
``field_strength`` as a REQUIRED keyword-only argument. The pipeline's gated validation
seam forwards a batch key only when its name literally appears in the strategy's
``validation_step`` signature (``_VALIDATION_FORWARD_FIELDS`` in ``pipelines/train.py``).

The strategy did not override ``validation_step``, so it inherited a signature that names
neither ``field_strength_target`` nor ``contrast_id``. The seam dropped both, the renderer
was called bare, every validation batch raised TypeError, the loop hit "Validation produced
zero successful batches", and the run died BEFORE ``_maybe_run_calibration`` — so the
Dice-risk RCPS certificate, this arm's entire deliverable, was never written.
"""

from __future__ import annotations

import inspect

from spectramr.infrastructure.training.strategies.conformal_calibration_strategy import (
    ConformalCalibrationStrategy,
)
from spectramr.pipelines.train import _VALIDATION_FORWARD_FIELDS


class TestCalibrationValidationSeam:
    def test_validation_step_declares_the_field_kwargs_the_seam_gates_on(self) -> None:
        params = inspect.signature(
            ConformalCalibrationStrategy.validation_step
        ).parameters
        for key in ("field_strength", "field_strength_target", "contrast_id"):
            assert key in params, (
                f"{key!r} missing from validation_step: the seam only forwards a batch "
                "key whose name appears in the signature, so AnatomyFieldRenderer.forward "
                "would be called without field_strength and every val batch would raise."
            )

    def test_those_kwargs_are_actually_forwardable_by_the_seam(self) -> None:
        # Guard the other half of the contract: a name the seam does not know is inert.
        for key in ("field_strength", "field_strength_target", "contrast_id"):
            assert key in _VALIDATION_FORWARD_FIELDS

    def test_strategy_overrides_validation_step_rather_than_inheriting_it(self) -> None:
        assert "validation_step" in vars(ConformalCalibrationStrategy), (
            "ConformalCalibrationStrategy must define its own validation_step; "
            "inheriting one whose signature omits the field kwargs reintroduces the crash."
        )
