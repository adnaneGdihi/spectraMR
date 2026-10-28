"""Regression: LoggingService hardening (2026-06-11 infrastructure audit).

* The message throttle was a lifetime cap (never decays), so a recurring
  WARNING/ERROR (NaN loss, OOM, gradient overflow) was silenced after 3
  occurrences — the repo "warnings are not OK" anti-pattern. High-severity
  levels are now exempt; only INFO/DEBUG spam is rate-limited.
* Every ``log()`` call flushed all handlers, a redundant I/O stall on the
  per-iteration logging hot path. Flushing is now WARNING+ only.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from spectramr.infrastructure.services.logging_service import LoggingService


def _service_with_mock_logger():
    svc = LoggingService()
    svc._logger = MagicMock()
    svc._logger.handlers = [MagicMock()]
    return svc


def test_recurring_warning_is_never_throttled():
    svc = _service_with_mock_logger()
    for _ in range(6):
        svc.log_warning("recurring NaN loss")
    assert svc._logger.warning.call_count == 6  # not capped at 3


def test_info_spam_is_throttled():
    svc = _service_with_mock_logger()
    for _ in range(6):
        svc.log_info("noisy info")
    assert svc._logger.info.call_count == 3  # capped


def test_info_does_not_flush_but_warning_does():
    svc = _service_with_mock_logger()
    handler = svc._logger.handlers[0]
    svc.log_info("hot path line")
    assert handler.flush.call_count == 0  # no per-INFO flush
    svc.log_warning("important")
    assert handler.flush.call_count == 1  # WARNING flushed for durability


class TestThereIsExactlyOneTensorBoardWriter:
    """`ComprehensiveLoggingService` must not build a second `SummaryWriter`.

    It used to. That writer read `logging.tracking.tensorboard_dir` and wrote a
    different directory from the one `pipelines/train.py` uses -- and it never
    ran, because `bootstrap.py` resolves `ILoggingService` through
    `LoggingServiceFactory.create`, which returns a base `LoggingService`. So
    the knob the config documented steered a writer nobody had, while 21
    committed arms declared it (#928).

    Asserted against the SOURCE rather than an instance because the defect was
    never observable from an instance: the dead branch is behind
    `if logging_config:`, and the one construction path never passes it.
    """

    def test_the_service_module_constructs_no_summary_writer(self) -> None:
        import inspect

        from spectramr.infrastructure.services import logging_service

        source = inspect.getsource(logging_service)
        assert "SummaryWriter(" not in source, (
            "logging_service.py builds a SummaryWriter again — "
            "TensorBoardWriter is the single writer (#928)"
        )

    def test_the_single_writer_lives_in_its_own_module(self) -> None:
        """Anti-vacuity: the assertion above must not pass because the whole
        feature was deleted."""
        import inspect

        from spectramr.infrastructure.services import tensorboard_writer

        assert "SummaryWriter(" in inspect.getsource(tensorboard_writer)


class TestCriticalIsTheFifthRung:
    """#2254: ``critical`` was in ``_SUPPORTED_LEVELS`` and ``_level_map`` from the
    start, but no ``log_critical`` wrapper existed. The training loop's divergence
    tripwire called it anyway, so the one guard that stops a diverged run from
    corrupting weights raised ``AttributeError`` every time it was right."""

    def test_log_critical_exists_and_routes_to_the_critical_logger(self):
        svc = _service_with_mock_logger()
        svc.log_critical("DIVERGENCE DETECTED")
        svc._logger.critical.assert_called_once()
        assert svc._logger.critical.call_args[0][0] == "DIVERGENCE DETECTED"

    def test_critical_is_never_throttled(self):
        """Same exemption the other high-severity rungs get: a recurring divergence
        must not go quiet after three occurrences."""
        svc = _service_with_mock_logger()
        for _ in range(5):
            svc.log_critical("same message")
        assert svc._logger.critical.call_count == 5

    def test_critical_flushes_for_crash_durability(self):
        """A diverged run is about to stop; the record has to survive the stop."""
        svc = _service_with_mock_logger()
        svc.log_critical("boom")
        svc._logger.handlers[0].flush.assert_called()

    def test_the_wrapper_set_covers_every_supported_level(self):
        """The rung that was missing is the rung that was needed. Pin the whole
        ladder so the next level added to ``_SUPPORTED_LEVELS`` arrives with its
        wrapper instead of being discovered by a caller at runtime."""
        svc = LoggingService()
        for level in svc._SUPPORTED_LEVELS:
            assert callable(getattr(svc, f"log_{level}", None)), (
                f"_SUPPORTED_LEVELS advertises {level!r} with no log_{level} method"
            )
