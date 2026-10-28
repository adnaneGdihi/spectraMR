"""Tests for ``EarlyStoppingConfigSchema``.

Targets ``spectramr.config.schemas.early_stopping``. Pydantic v2 schema for
early-stopping behaviour. Validates: documented defaults, ge/le
constraints, ``frozen=True`` immutability, ``extra='forbid'``.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from spectramr.config.schemas.early_stopping import EarlyStoppingConfigSchema
from spectramr.config.schemas.enums import MetricMode


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_default_construction_succeeds() -> None:
    """All fields have valid defaults — construction without args works."""
    cfg = EarlyStoppingConfigSchema()
    assert cfg.enabled is False
    assert cfg.patience == 10
    assert cfg.metric == "val_loss"
    assert cfg.mode == MetricMode.MIN
    assert cfg.min_delta == 0.0
    assert cfg.restore_best_weights is True


def test_explicit_values_override_defaults() -> None:
    """Explicit field values override the defaults."""
    cfg = EarlyStoppingConfigSchema(
        enabled=True, patience=5, metric="val_psnr", mode=MetricMode.MAX
    )
    assert cfg.enabled is True
    assert cfg.patience == 5
    assert cfg.metric == "val_psnr"
    assert cfg.mode == MetricMode.MAX


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------


def test_patience_must_be_positive() -> None:
    """``patience >= 1``."""
    with pytest.raises(ValidationError):
        EarlyStoppingConfigSchema(patience=0)


def test_min_delta_must_be_non_negative() -> None:
    """``min_delta >= 0``."""
    with pytest.raises(ValidationError):
        EarlyStoppingConfigSchema(min_delta=-0.001)


def test_check_interval_is_refused_by_name() -> None:
    """Planted: the removed knob is refused with its reason and the live key (#1555).

    A bare ``extra="forbid"`` error would also fail the load; this pins that the
    removal record's mount fires first and tells the author where the cadence lives.
    """
    with pytest.raises(ValidationError) as excinfo:
        EarlyStoppingConfigSchema(check_interval=5000)
    message = str(excinfo.value)
    assert "`early_stopping.check_interval` was removed" in message
    assert "`validation.schedule.interval_steps`" in message
    assert "check_interval" not in EarlyStoppingConfigSchema.model_fields


def test_patience_min_iterations_can_be_none() -> None:
    """``patience_min_iterations`` is optional."""
    cfg = EarlyStoppingConfigSchema(patience_min_iterations=None)
    assert cfg.patience_min_iterations is None


def test_patience_min_iterations_non_negative() -> None:
    """``patience_min_iterations >= 0``."""
    with pytest.raises(ValidationError):
        EarlyStoppingConfigSchema(patience_min_iterations=-1)


# ---------------------------------------------------------------------------
# Frozen + extra='forbid'
# ---------------------------------------------------------------------------


def test_schema_is_frozen() -> None:
    """``frozen=True`` blocks mutation after construction."""
    cfg = EarlyStoppingConfigSchema()
    with pytest.raises(ValidationError):
        cfg.patience = 99


def test_extra_fields_rejected() -> None:
    """``extra='forbid'`` rejects unknown fields."""
    with pytest.raises(ValidationError, match="extra"):
        EarlyStoppingConfigSchema(unknown_field=1)


# ---------------------------------------------------------------------------
# mode must not contradict the metric's own direction (metrics plan PR 2, 2.5)
#
# `mode: max` on a loss stops the run when the loss stops RISING and writes
# the best checkpoint at the WORST iterate. Nothing downstream notices: the
# comparison is well-defined, the wait counter advances, the file is written,
# every artifact reports success. The selected model is simply the wrong one --
# pitfall #16, decided by one enum.
# ---------------------------------------------------------------------------


class TestModeMatchesMetricDirection:
    @staticmethod
    def _build(metric: str, mode: str, *, enabled: bool = True):
        from spectramr.config.schemas.early_stopping import EarlyStoppingConfigSchema
        from spectramr.config.schemas.enums import MetricMode

        return EarlyStoppingConfigSchema(enabled=enabled, metric=metric, mode=MetricMode(mode))

    def test_a_disabled_block_is_not_policed(self) -> None:
        """A disabled block's ``mode`` selects nothing, so it cannot be wrong.

        The pairing below is the one ``test_a_contradicting_pair_raises`` refuses;
        the ONLY difference is ``enabled=False``. Asserting it here, in the schema's
        own suite, is the point: the schema is the owner that actually runs -- it
        validates at config load, before ``EarlyStoppingService`` is constructed --
        so its behaviour cannot be inferred from the service's tests.

        The two used to disagree, and silently. ``EarlyStoppingService`` has opened
        with ``if not self.enabled ... return`` since #712, but the schema gained its
        own copy of the rule later (6dbd12c7b) without that clause, and because the
        schema runs first the service's exemption was unreachable. Measured when the
        clause was added: 165 corpus arms disable early stopping and none contradicts
        its metric, so this settles which rule is real rather than changing any arm.
        """
        assert self._build("val_psnr", "min", enabled=False) is not None
        assert self._build("val_loss", "max", enabled=False) is not None

    @pytest.mark.parametrize(("metric", "mode"), [("val_loss", "min"), ("val_psnr", "max")])
    def test_an_agreeing_pair_is_accepted(self, metric: str, mode: str) -> None:
        assert self._build(metric, mode) is not None

    @pytest.mark.parametrize(("metric", "mode"), [("val_psnr", "min"), ("val_loss", "max")])
    def test_a_contradicting_pair_raises(self, metric: str, mode: str) -> None:
        with pytest.raises(ValueError, match="contradicts"):
            self._build(metric, mode)

    def test_the_message_names_the_fix(self) -> None:
        """A rejection that does not say which value to change costs a round trip."""
        with pytest.raises(ValueError) as excinfo:
            self._build("val_psnr", "min")
        message = str(excinfo.value)
        assert "mode: max" in message, message
        assert "higher is better" in message, message

    def test_an_unresolvable_metric_is_left_alone(self) -> None:
        """Not this rule's defect to report.

        ``metric_higher_is_better`` RAISES on an unknown key rather than
        guessing. Rejecting here too would report one problem as two: an
        unresolvable monitor is caught at the first validation event (#178),
        which owns that message and lists the keys the arm actually produces.
        """
        assert self._build("val_a_metric_that_does_not_exist", "max") is not None

    def test_the_direction_comes_from_the_shared_resolver(self) -> None:
        """Anti-vacuity: a second direction table would drift from the first.

        ``metric_higher_is_better`` is the one resolver already used by the
        metrics tracker, early stopping and the campaign ranker. This asserts the validator agrees with it rather than carrying
        its own opinion.
        """
        from spectramr.core.metrics.metric_directions import metric_higher_is_better

        for metric in ("val_loss", "val_psnr", "val_ssim", "val_lpips"):
            good = "max" if metric_higher_is_better(metric) else "min"
            bad = "min" if good == "max" else "max"
            assert self._build(metric, good) is not None
            with pytest.raises(ValueError):
                self._build(metric, bad)
