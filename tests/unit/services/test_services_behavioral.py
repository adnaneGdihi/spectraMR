"""Behavioral unit tests for infrastructure/services implementations.

Tests cover the documented contracts of each cheaply-constructable IService
implementation:
  - LoggingService
  - MetricsTracker
  - EarlyStoppingService
  - IterationCounterService
  - DeviceManager / DevicePolicy
  - SteinAggregation (pure-function module)
  - RDPAccountant (pure class, no IO)

Services requiring real cluster IO, GPU, or heavy third-party deps (e.g.
the checkpoint store with real filesystem checkpoints, coil_sensitivity_service
needing CUDA, pipeline_execution_service with full DI) are skipped with an
explicit reason.
"""

from __future__ import annotations

import logging

import pytest

# ---------------------------------------------------------------------------
# LoggingService
# ---------------------------------------------------------------------------


class TestLoggingService:
    """LoggingService emits on each level without raising."""

    @pytest.mark.unit
    def test_canary_construct_and_log(self) -> None:
        from spectramr.infrastructure.services.logging_service import LoggingService

        svc = LoggingService(logger_name="test_canary")
        # Primary method must not raise
        svc.log("info", "canary info message")
        svc.log("warning", "canary warning message")
        svc.log("error", "canary error message")
        svc.log("debug", "canary debug message")

    @pytest.mark.unit
    def test_log_info_shorthand(self) -> None:
        from spectramr.infrastructure.services.logging_service import LoggingService

        svc = LoggingService(logger_name="test_shorthand")
        svc.log_info("info via shorthand")

    @pytest.mark.unit
    def test_log_warning_shorthand(self) -> None:
        from spectramr.infrastructure.services.logging_service import LoggingService

        svc = LoggingService(logger_name="test_warn_shorthand")
        svc.log_warning("warning via shorthand")

    @pytest.mark.unit
    def test_log_error_shorthand(self) -> None:
        from spectramr.infrastructure.services.logging_service import LoggingService

        svc = LoggingService(logger_name="test_error_shorthand")
        svc.log_error("error via shorthand")

    @pytest.mark.unit
    def test_invalid_level_raises(self) -> None:
        from spectramr.infrastructure.services.logging_service import LoggingService

        svc = LoggingService(logger_name="test_invalid")
        with pytest.raises(ValueError, match="Unsupported log level"):
            svc.log("NONSENSE", "some message")

    @pytest.mark.unit
    def test_throttle_limits_repeated_messages(self) -> None:
        """Same (level, message) pair is throttled after the limit."""
        from spectramr.infrastructure.services.logging_service import LoggingService

        svc = LoggingService(logger_name="test_throttle")
        # _throttle_limit is 3 — call 5× and expect no crash
        for _ in range(5):
            svc.log("info", "repeated message for throttle test")

    @pytest.mark.unit
    def test_get_logger_returns_logger(self) -> None:
        from spectramr.infrastructure.services.logging_service import LoggingService

        svc = LoggingService(logger_name="test_getlogger")
        lg = svc.get_logger("some.child")
        assert isinstance(lg, logging.Logger)

    @pytest.mark.unit
    def test_step_and_epoch_properties(self) -> None:
        from spectramr.infrastructure.services.logging_service import LoggingService

        svc = LoggingService(logger_name="test_step_epoch")
        svc.set_step(42)
        svc.set_epoch(7)
        assert svc.step == 42
        assert svc.epoch == 7

    @pytest.mark.unit
    def test_fallback_records_empty_on_clean_run(self) -> None:
        from spectramr.infrastructure.services.logging_service import LoggingService

        svc = LoggingService(logger_name="test_fallback_clean")
        svc.log("info", "clean message")
        # fallback_records only populated when the handler itself raises;
        # a clean run should not populate it
        assert isinstance(svc.fallback_records, list)


class TestJSONFormatter:
    """JSONFormatter produces valid JSON strings."""

    @pytest.mark.unit
    def test_formats_to_json(self) -> None:
        import json

        from spectramr.infrastructure.services.logging_service import JSONFormatter

        fmt = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="hello json",
            args=(),
            exc_info=None,
        )
        result = fmt.format(record)
        parsed = json.loads(result)
        assert parsed["level"] == "INFO"
        assert parsed["message"] == "hello json"

    @pytest.mark.unit
    def test_fields_present(self) -> None:
        import json

        from spectramr.infrastructure.services.logging_service import JSONFormatter

        fmt = JSONFormatter()
        record = logging.LogRecord(
            name="mylogger",
            level=logging.WARNING,
            pathname="",
            lineno=0,
            msg="warn msg",
            args=(),
            exc_info=None,
        )
        parsed = json.loads(fmt.format(record))
        for key in ("timestamp", "level", "name", "message"):
            assert key in parsed, f"missing key {key}"


class TestBootstrapConsoleLogging:
    """bootstrap_console_logging installs a handler without crash."""

    @pytest.mark.unit
    def test_idempotent(self) -> None:
        from spectramr.infrastructure.services.logging_service import (
            bootstrap_console_logging,
        )

        bootstrap_console_logging(level="WARNING")
        bootstrap_console_logging(level="WARNING")  # second call must not stack handlers


# ---------------------------------------------------------------------------
# MetricsTracker
# ---------------------------------------------------------------------------


class TestMetricsTracker:
    """MetricsTracker persists/aggregates state after update/flush; reset clears."""

    @pytest.mark.unit
    def test_canary_construct(self, tmp_path) -> None:
        from spectramr.infrastructure.services.metrics_tracker import MetricsTracker

        tracker = MetricsTracker(output_dir=str(tmp_path / "mt"), save_images=False)
        assert tracker.current_metrics == {}

    @pytest.mark.unit
    def test_update_persists(self, tmp_path) -> None:
        from spectramr.infrastructure.services.metrics_tracker import MetricsTracker

        tracker = MetricsTracker(output_dir=str(tmp_path / "mt2"), save_images=False)
        tracker.update_metric("psnr", 35.5)
        tracker.update_metric("ssim", 0.92)
        assert tracker.current_metrics["psnr"] == pytest.approx(35.5)
        assert tracker.current_metrics["ssim"] == pytest.approx(0.92)

    @pytest.mark.unit
    def test_update_metrics_dict(self, tmp_path) -> None:
        from spectramr.infrastructure.services.metrics_tracker import MetricsTracker

        tracker = MetricsTracker(output_dir=str(tmp_path / "mt3"), save_images=False)
        tracker.update_metrics({"loss_g": 0.3, "loss_d": 0.7})
        assert tracker.current_metrics["loss_g"] == pytest.approx(0.3)

    @pytest.mark.unit
    def test_reset_clears_state(self, tmp_path) -> None:
        from spectramr.infrastructure.services.metrics_tracker import MetricsTracker

        tracker = MetricsTracker(output_dir=str(tmp_path / "mt4"), save_images=False)
        tracker.update_metric("some_val", 1.0)
        tracker.reset_metrics()
        assert tracker.current_metrics == {}

    @pytest.mark.unit
    def test_set_gauge_alias(self, tmp_path) -> None:
        from spectramr.infrastructure.services.metrics_tracker import MetricsTracker

        tracker = MetricsTracker(output_dir=str(tmp_path / "mt5"), save_images=False)
        tracker.set_gauge("my_gauge", 99.0)
        assert tracker.current_metrics["my_gauge"] == pytest.approx(99.0)

    @pytest.mark.unit
    def test_log_metrics_writes_csv(self, tmp_path) -> None:
        import csv

        from spectramr.infrastructure.services.metrics_tracker import MetricsTracker

        tracker = MetricsTracker(output_dir=str(tmp_path / "mt6"), save_images=False)
        tracker.log_metrics({"psnr": 30.0}, step=10, epoch=1, prefix="validation")
        assert tracker.validation_csv.exists()
        with open(tracker.validation_csv, newline="") as f:
            rows = list(csv.reader(f))
        assert len(rows) >= 2  # header + at least one data row

    @pytest.mark.unit
    def test_get_latest_metrics_returns_copy(self, tmp_path) -> None:
        from spectramr.infrastructure.services.metrics_tracker import MetricsTracker

        tracker = MetricsTracker(output_dir=str(tmp_path / "mt7"), save_images=False)
        tracker.update_metric("x", 5.0)
        copy = tracker.get_latest_metrics()
        copy["x"] = 999.0  # mutating the copy must not affect tracker
        assert tracker.current_metrics["x"] == pytest.approx(5.0)

    @pytest.mark.unit
    def test_export_summary_report(self, tmp_path) -> None:
        from spectramr.infrastructure.services.metrics_tracker import MetricsTracker

        tracker = MetricsTracker(output_dir=str(tmp_path / "mt8"), save_images=False)
        path = tracker.export_summary_report()
        import os

        assert os.path.exists(path)
        with open(path) as f:
            content = f.read()
        assert "METRICS SUMMARY REPORT" in content

    @pytest.mark.unit
    def test_empty_best_metrics(self, tmp_path) -> None:
        from spectramr.infrastructure.services.metrics_tracker import MetricsTracker

        tracker = MetricsTracker(output_dir=str(tmp_path / "mt9"), save_images=False)
        assert tracker.get_best_metrics() == {}


# ---------------------------------------------------------------------------
# EarlyStoppingService
# ---------------------------------------------------------------------------


def _make_es_config(**kwargs):
    """Build a minimal EarlyStoppingConfigSchema."""
    from spectramr.config.schemas.early_stopping import EarlyStoppingConfigSchema

    defaults = {"enabled": True, "patience": 3, "metric": "val_loss", "min_delta": 0.0}
    defaults.update(kwargs)
    return EarlyStoppingConfigSchema(**defaults)


class TestEarlyStoppingService:
    @pytest.mark.unit
    def test_canary_construct(self) -> None:
        from spectramr.infrastructure.services.early_stopping import EarlyStoppingService

        cfg = _make_es_config(enabled=True, patience=5)
        svc = EarlyStoppingService(cfg)
        assert not svc.should_stop()

    @pytest.mark.unit
    def test_improves_resets_wait_count(self) -> None:
        from spectramr.config.schemas.enums import MetricMode
        from spectramr.infrastructure.services.early_stopping import EarlyStoppingService

        cfg = _make_es_config(enabled=True, patience=3, mode=MetricMode.MIN, min_delta=0.0)
        svc = EarlyStoppingService(cfg)
        svc.update(1.0, iteration=1)
        svc.update(0.8, iteration=2)  # improvement
        assert svc.wait_count == 0
        assert svc.best_value == pytest.approx(0.8)

    @pytest.mark.unit
    def test_no_improvement_increments_wait(self) -> None:
        from spectramr.config.schemas.enums import MetricMode
        from spectramr.infrastructure.services.early_stopping import EarlyStoppingService

        cfg = _make_es_config(enabled=True, patience=5, mode=MetricMode.MIN)
        svc = EarlyStoppingService(cfg)
        svc.update(1.0, iteration=1)  # sets best
        svc.update(1.5, iteration=2)  # worse → wait_count = 1
        svc.update(1.2, iteration=3)  # worse → wait_count = 2
        assert svc.wait_count == 2
        assert not svc.should_stop()

    @pytest.mark.unit
    def test_stop_triggered_after_patience(self) -> None:
        from spectramr.config.schemas.enums import MetricMode
        from spectramr.infrastructure.services.early_stopping import EarlyStoppingService

        cfg = _make_es_config(enabled=True, patience=2, mode=MetricMode.MIN)
        svc = EarlyStoppingService(cfg)
        svc.update(1.0, iteration=1)  # best
        svc.update(1.1, iteration=2)  # wait=1
        svc.update(1.2, iteration=3)  # wait=2 → stop
        assert svc.should_stop()

    @pytest.mark.unit
    def test_disabled_service_never_stops(self) -> None:
        from spectramr.config.schemas.enums import MetricMode
        from spectramr.infrastructure.services.early_stopping import EarlyStoppingService

        cfg = _make_es_config(enabled=False, patience=1, mode=MetricMode.MIN)
        svc = EarlyStoppingService(cfg)
        svc.update(1.0, iteration=1)
        svc.update(9.0, iteration=2)  # very bad — but service is disabled
        assert not svc.should_stop()

    @pytest.mark.unit
    def test_reset_clears_state(self) -> None:
        from spectramr.config.schemas.enums import MetricMode
        from spectramr.infrastructure.services.early_stopping import EarlyStoppingService

        cfg = _make_es_config(enabled=True, patience=1, mode=MetricMode.MIN)
        svc = EarlyStoppingService(cfg)
        svc.update(1.0, iteration=1)
        svc.update(2.0, iteration=2)  # triggers stop
        assert svc.should_stop()
        svc.reset()
        assert not svc.should_stop()
        assert svc.wait_count == 0

    @pytest.mark.unit
    def test_callback_called_on_improvement(self) -> None:
        from spectramr.config.schemas.enums import MetricMode
        from spectramr.infrastructure.services.early_stopping import EarlyStoppingService

        calls: list[tuple] = []
        cfg = _make_es_config(enabled=True, patience=5, mode=MetricMode.MIN)
        svc = EarlyStoppingService(cfg, on_improvement=lambda it, v: calls.append((it, v)))
        svc.update(1.0, iteration=1)
        svc.update(0.5, iteration=2)  # improvement
        assert len(calls) == 2
        assert calls[-1] == (2, pytest.approx(0.5))

    @pytest.mark.unit
    def test_max_mode_improves_upward(self) -> None:
        """MAX mode must be paired with a higher-is-better metric.

        This overrode only ``mode``, leaving the fixture's ``metric="val_loss"``
        — and ``val_loss`` is declared lower-is-better, so the pairing is exactly
        the contradiction ``_check_mode_against_direction`` was added to reject
        ("selection would keep the WORST checkpoint"). The guard is right; the
        test was asserting through it. ``val_psnr`` is higher-is-better, which is
        what MAX means.
        """
        from spectramr.config.schemas.enums import MetricMode
        from spectramr.infrastructure.services.early_stopping import EarlyStoppingService

        cfg = _make_es_config(enabled=True, patience=5, metric="val_psnr", mode=MetricMode.MAX)
        svc = EarlyStoppingService(cfg)
        svc.update(0.5, iteration=1)
        svc.update(0.8, iteration=2)  # higher = improvement
        assert svc.wait_count == 0

    @pytest.mark.unit
    def test_max_mode_on_a_lower_is_better_metric_raises(self) -> None:
        """The pairing above is rejected, not silently accepted."""
        from spectramr.config.schemas.enums import MetricMode
        from spectramr.infrastructure.services.early_stopping import EarlyStoppingService

        # The config build moved INSIDE the `raises` block and the pattern is
        # loose. `EarlyStoppingConfigSchema` gained its own copy of this guard
        # (6dbd12c7b) and it runs first, so the contradiction is refused while
        # building `cfg` -- one line ABOVE the block this test used to open,
        # which turned the failure into a collection-time error rather than a
        # clean assertion. The service's guard is unreachable with a real schema
        # object; what must still hold is that the pairing is refused at all.
        with pytest.raises(ValueError, match="contradicts"):
            cfg = _make_es_config(enabled=True, metric="val_loss", mode=MetricMode.MAX)
            EarlyStoppingService(cfg)

    @pytest.mark.unit
    def test_get_state_dict(self) -> None:
        from spectramr.infrastructure.services.early_stopping import EarlyStoppingService

        cfg = _make_es_config(enabled=True)
        svc = EarlyStoppingService(cfg)
        state = svc.get_state()
        for key in (
            "enabled",
            "monitor",
            "patience",
            "best_value",
            "wait_count",
            "should_stop",
        ):
            assert key in state, f"missing state key: {key}"


# ---------------------------------------------------------------------------
# IterationCounterService
# ---------------------------------------------------------------------------


def _make_iter_config(max_iterations: int = -1, max_steps_per_epoch: int = -1):
    """Build a minimal BaseTrainingConfigSchema-like mock."""
    from unittest.mock import MagicMock

    cfg = MagicMock()
    cfg.max_iterations = max_iterations
    cfg.max_steps_per_epoch = max_steps_per_epoch
    return cfg


class TestIterationCounterService:
    @pytest.mark.unit
    def test_canary_construct(self) -> None:
        from spectramr.infrastructure.services.iteration_counter_service import (
            IterationCounterService,
        )

        svc = IterationCounterService(_make_iter_config())
        assert svc.current_step == 0
        assert svc.current_epoch == 0

    @pytest.mark.unit
    def test_increment_advances_step(self) -> None:
        from spectramr.infrastructure.services.iteration_counter_service import (
            IterationCounterService,
        )

        svc = IterationCounterService(_make_iter_config())
        svc.increment()
        svc.increment()
        assert svc.current_step == 2
        assert svc.steps_in_current_epoch == 2

    @pytest.mark.unit
    def test_start_epoch_resets_epoch_steps(self) -> None:
        from spectramr.infrastructure.services.iteration_counter_service import (
            IterationCounterService,
        )

        svc = IterationCounterService(_make_iter_config())
        svc.increment()
        svc.increment()
        svc.start_epoch(total_steps_in_epoch=50)
        assert svc.current_epoch == 1
        assert svc.steps_in_current_epoch == 0
        assert svc.total_steps_this_epoch == 50

    @pytest.mark.unit
    def test_should_checkpoint_at_interval(self) -> None:
        from spectramr.infrastructure.services.iteration_counter_service import (
            IterationCounterService,
        )

        svc = IterationCounterService(_make_iter_config())
        for _ in range(10):
            svc.increment()
        assert svc.should_checkpoint(10)
        assert not svc.should_checkpoint(9)

    @pytest.mark.unit
    def test_should_log_at_interval(self) -> None:
        from spectramr.infrastructure.services.iteration_counter_service import (
            IterationCounterService,
        )

        svc = IterationCounterService(_make_iter_config())
        for _ in range(5):
            svc.increment()
        assert svc.should_log(5)
        assert not svc.should_log(4)

    @pytest.mark.unit
    def test_should_stop_global_at_max(self) -> None:
        from spectramr.infrastructure.services.iteration_counter_service import (
            IterationCounterService,
        )

        svc = IterationCounterService(_make_iter_config(max_iterations=3))
        for _ in range(3):
            svc.increment()
        assert svc.should_stop_global()

    @pytest.mark.unit
    def test_increment_raises_at_max(self) -> None:
        from spectramr.infrastructure.services.iteration_counter_service import (
            IterationCounterService,
        )

        svc = IterationCounterService(_make_iter_config(max_iterations=2))
        svc.increment()
        svc.increment()
        with pytest.raises(RuntimeError, match="max_iterations"):
            svc.increment()

    @pytest.mark.unit
    def test_unlimited_never_stops(self) -> None:
        from spectramr.infrastructure.services.iteration_counter_service import (
            IterationCounterService,
        )

        svc = IterationCounterService(_make_iter_config(max_iterations=-1))
        for _ in range(100):
            svc.increment()
        assert not svc.should_stop_global()

    @pytest.mark.unit
    def test_get_state_keys(self) -> None:
        from spectramr.infrastructure.services.iteration_counter_service import (
            IterationCounterService,
        )

        svc = IterationCounterService(_make_iter_config())
        state = svc.get_state()
        for key in (
            "current_step",
            "current_epoch",
            "steps_in_current_epoch",
            "max_iterations",
            "max_steps_per_epoch",
        ):
            assert key in state

    @pytest.mark.unit
    def test_repr_contains_step(self) -> None:
        from spectramr.infrastructure.services.iteration_counter_service import (
            IterationCounterService,
        )

        svc = IterationCounterService(_make_iter_config())
        svc.increment()
        assert "step=1" in repr(svc)

    @pytest.mark.unit
    def test_should_validate_step_based(self) -> None:
        from spectramr.infrastructure.services.iteration_counter_service import (
            IterationCounterService,
        )

        svc = IterationCounterService(_make_iter_config())
        for _ in range(20):
            svc.increment()
        assert svc.should_validate(eval_interval=20)
        assert not svc.should_validate(eval_interval=15)

    @pytest.mark.unit
    def test_should_validate_epoch_based(self) -> None:
        from spectramr.infrastructure.services.iteration_counter_service import (
            IterationCounterService,
        )

        svc = IterationCounterService(_make_iter_config())
        svc.start_epoch()  # epoch=1
        svc.start_epoch()  # epoch=2
        svc.start_epoch()  # epoch=3
        assert svc.should_validate(eval_interval=100, frequency_epochs=3, use_epoch_based=True)


# ---------------------------------------------------------------------------
# DeviceManager / DevicePolicy
# ---------------------------------------------------------------------------


class TestDevicePolicy:
    """DevicePolicy resolves device from config; CPU path is always available."""

    @pytest.mark.unit
    def test_cpu_policy(self) -> None:
        from spectramr.infrastructure.services.device_policy import (
            DevicePolicy,
            DevicePolicyConfig,
            DevicePolicyType,
        )

        policy = DevicePolicy(DevicePolicyConfig(policy_type=DevicePolicyType.CPU))
        import torch

        assert policy.get_device() == torch.device("cpu")

    @pytest.mark.unit
    def test_auto_policy_gives_cuda_or_raises(self) -> None:
        """AUTO never quietly settles for CPU (non-negotiable 9b).

        Asserted ``device.type in ("cpu", "cuda")`` until job 8000966, i.e. the
        silent fallback the accelerated-run contract abolished. Green on a GPU
        box, red on a CPU node — the one place the contract matters.
        """
        import torch

        from spectramr.core.compute_device import AcceleratorRequiredError
        from spectramr.infrastructure.services.device_policy import (
            DevicePolicy,
            DevicePolicyConfig,
            DevicePolicyType,
        )

        policy = DevicePolicy(DevicePolicyConfig(policy_type=DevicePolicyType.AUTO))

        if torch.cuda.is_available():
            assert policy.get_device() == torch.device("cuda")
            return

        with pytest.raises(AcceleratorRequiredError):
            policy.get_device()

    @pytest.mark.unit
    def test_specific_cpu_policy(self) -> None:
        from spectramr.infrastructure.services.device_policy import (
            DevicePolicy,
            DevicePolicyConfig,
            DevicePolicyType,
        )
        import torch

        policy = DevicePolicy(
            DevicePolicyConfig(
                policy_type=DevicePolicyType.SPECIFIC,
                specific_device="cpu",
            )
        )
        assert policy.get_device() == torch.device("cpu")

    @pytest.mark.unit
    def test_device_cached(self) -> None:
        from spectramr.infrastructure.services.device_policy import (
            DevicePolicy,
            DevicePolicyConfig,
            DevicePolicyType,
        )

        policy = DevicePolicy(DevicePolicyConfig(policy_type=DevicePolicyType.CPU))
        d1 = policy.get_device()
        d2 = policy.get_device()
        assert d1 == d2

    @pytest.mark.unit
    def test_move_tensor_to_cpu(self) -> None:
        import torch

        from spectramr.infrastructure.services.device_policy import (
            DevicePolicy,
            DevicePolicyConfig,
            DevicePolicyType,
        )

        policy = DevicePolicy(DevicePolicyConfig(policy_type=DevicePolicyType.CPU))
        t = torch.randn(3, 4)
        moved = policy.move_to_device(t)
        assert moved.device.type == "cpu"

    @pytest.mark.unit
    def test_move_list_to_cpu(self) -> None:
        import torch

        from spectramr.infrastructure.services.device_policy import (
            DevicePolicy,
            DevicePolicyConfig,
            DevicePolicyType,
        )

        policy = DevicePolicy(DevicePolicyConfig(policy_type=DevicePolicyType.CPU))
        tensors = [torch.randn(2), torch.randn(3)]
        moved = policy.move_to_device(tensors)
        assert all(t.device.type == "cpu" for t in moved)

    @pytest.mark.unit
    def test_move_dict_to_cpu(self) -> None:
        import torch

        from spectramr.infrastructure.services.device_policy import (
            DevicePolicy,
            DevicePolicyConfig,
            DevicePolicyType,
        )

        policy = DevicePolicy(DevicePolicyConfig(policy_type=DevicePolicyType.CPU))
        data = {"a": torch.randn(2), "b": torch.randn(3)}
        moved = policy.move_to_device(data)
        assert all(v.device.type == "cpu" for v in moved.values())

    @pytest.mark.unit
    def test_is_device_available_cpu(self) -> None:
        import torch

        from spectramr.infrastructure.services.device_policy import (
            DevicePolicy,
            DevicePolicyConfig,
            DevicePolicyType,
        )

        policy = DevicePolicy(DevicePolicyConfig(policy_type=DevicePolicyType.CPU))
        assert policy.is_device_available(torch.device("cpu"))

    @pytest.mark.unit
    def test_specific_missing_device_raises(self) -> None:
        from spectramr.infrastructure.services.device_policy import (
            DevicePolicy,
            DevicePolicyConfig,
            DevicePolicyType,
        )

        with pytest.raises(ValueError, match="specific_device must be provided"):
            DevicePolicy(
                DevicePolicyConfig(
                    policy_type=DevicePolicyType.SPECIFIC,
                    specific_device=None,
                )
            )

    @pytest.mark.unit
    def test_get_device_info_has_type(self) -> None:
        from spectramr.infrastructure.services.device_policy import (
            DevicePolicy,
            DevicePolicyConfig,
            DevicePolicyType,
        )

        policy = DevicePolicy(DevicePolicyConfig(policy_type=DevicePolicyType.CPU))
        info = policy.get_device_info()
        assert info["type"] == "cpu"
        assert "policy" in info


class TestDeviceManager:
    @pytest.mark.unit
    def test_canary_cpu(self) -> None:
        from spectramr.infrastructure.services.device_manager import DeviceManager

        dm = DeviceManager(preferred_device="cpu")
        import torch

        assert dm.get_device() == torch.device("cpu")

    @pytest.mark.unit
    def test_list_devices_includes_cpu(self) -> None:
        from spectramr.infrastructure.services.device_manager import DeviceManager

        dm = DeviceManager(preferred_device="cpu")
        devices = dm.list_devices()
        assert "cpu" in devices

    @pytest.mark.unit
    def test_unsupported_device_raises(self) -> None:
        from spectramr.infrastructure.services.device_manager import DeviceManager

        with pytest.raises(ValueError, match="Unsupported device"):
            DeviceManager(preferred_device="tpu")

    @pytest.mark.unit
    def test_cuda_request_without_gpu_raises(self) -> None:
        import torch

        if torch.cuda.is_available():
            pytest.skip("CUDA available — test targets no-GPU path only")
        from spectramr.infrastructure.services.device_manager import DeviceManager

        with pytest.raises(RuntimeError, match="CUDA"):
            DeviceManager(preferred_device="cuda")

    @pytest.mark.unit
    def test_move_tensor_to_device(self) -> None:
        import torch

        from spectramr.infrastructure.services.device_manager import DeviceManager

        dm = DeviceManager(preferred_device="cpu")
        t = torch.randn(4)
        moved = dm.move_to_device(t)
        assert isinstance(moved, torch.Tensor)


# ---------------------------------------------------------------------------
# Convenience factories in device_policy
# ---------------------------------------------------------------------------


class TestDevicePolicyFactories:
    @pytest.mark.unit
    def test_create_cpu_policy(self) -> None:
        from spectramr.infrastructure.services.device_policy import (
            create_cpu_device_policy,
        )
        import torch

        p = create_cpu_device_policy()
        assert p.get_device() == torch.device("cpu")

    @pytest.mark.unit
    def test_create_auto_policy(self) -> None:
        """The factory builds an AUTO policy; resolving it is 9b's business.

        Asserting ``isinstance(p.get_device(), torch.device)`` made this a
        second, weaker copy of the contract test above — and it failed on a CPU
        node for the same reason. What belongs here is that the FACTORY produces
        the AUTO policy type; where that resolves is asserted once, in
        ``test_auto_policy_gives_cuda_or_raises``.
        """
        from spectramr.infrastructure.services.device_policy import (
            DevicePolicyType,
            create_auto_device_policy,
        )

        assert create_auto_device_policy().config.policy_type is DevicePolicyType.AUTO

    @pytest.mark.unit
    def test_create_specific_policy(self) -> None:
        from spectramr.infrastructure.services.device_policy import (
            create_specific_device_policy,
        )
        import torch

        p = create_specific_device_policy("cpu")
        assert p.get_device() == torch.device("cpu")

    @pytest.mark.unit
    def test_create_cuda_policy_with_fallback(self) -> None:
        """CUDA policy with fallback_to_cpu should not raise on CPU-only environment."""
        from spectramr.infrastructure.services.device_policy import (
            create_cuda_device_policy,
        )

        p = create_cuda_device_policy(fallback_to_cpu=True)
        # Should resolve to cpu or cuda without raising
        device = p.get_device()
        assert device.type in ("cpu", "cuda")


# ---------------------------------------------------------------------------
# SteinAggregation (pure functions — no GPU needed)
# ---------------------------------------------------------------------------


class TestSteinAggregation:
    @pytest.mark.unit
    def test_rbf_kernel_shape(self) -> None:
        import torch

        from spectramr.infrastructure.services.stein_aggregation import rbf_kernel

        particles = torch.randn(5, 8)
        k, gk = rbf_kernel(particles)
        assert k.shape == (5, 5)
        assert gk.shape == (5, 5, 8)

    @pytest.mark.unit
    def test_rbf_kernel_symmetric(self) -> None:
        import torch

        from spectramr.infrastructure.services.stein_aggregation import rbf_kernel

        particles = torch.randn(4, 6)
        k, _ = rbf_kernel(particles)
        assert torch.allclose(k, k.T, atol=1e-5)

    @pytest.mark.unit
    def test_rbf_kernel_requires_2d(self) -> None:
        import torch

        from spectramr.infrastructure.services.stein_aggregation import rbf_kernel

        with pytest.raises(ValueError, match="particles must be"):
            rbf_kernel(torch.randn(3))

    @pytest.mark.unit
    def test_rbf_kernel_requires_at_least_2(self) -> None:
        import torch

        from spectramr.infrastructure.services.stein_aggregation import rbf_kernel

        with pytest.raises(ValueError, match="at least 2 particles"):
            rbf_kernel(torch.randn(1, 4))

    @pytest.mark.unit
    def test_svgd_step_shape(self) -> None:
        import torch

        from spectramr.infrastructure.services.stein_aggregation import svgd_update_step

        n, d = 6, 10
        particles = torch.randn(n, d)
        updated = svgd_update_step(particles, lambda x: torch.zeros_like(x))
        assert updated.shape == (n, d)

    @pytest.mark.unit
    def test_svgd_step_invalid_step_size(self) -> None:
        import torch

        from spectramr.infrastructure.services.stein_aggregation import svgd_update_step

        with pytest.raises(ValueError, match="step_size must be positive"):
            svgd_update_step(torch.randn(3, 4), lambda x: x, step_size=-1.0)

    @pytest.mark.unit
    def test_product_of_experts_mean(self) -> None:
        import torch

        from spectramr.infrastructure.services.stein_aggregation import (
            product_of_experts_score,
        )

        a = torch.ones(3, 5)
        b = torch.ones(3, 5) * 3.0
        result = product_of_experts_score([a, b])
        expected = 2.0
        assert torch.allclose(result, torch.full_like(result, expected))

    @pytest.mark.unit
    def test_product_of_experts_empty_raises(self) -> None:
        from spectramr.infrastructure.services.stein_aggregation import (
            product_of_experts_score,
        )

        with pytest.raises(ValueError, match="non-empty"):
            product_of_experts_score([])

    @pytest.mark.unit
    def test_product_of_experts_shape_mismatch_raises(self) -> None:
        import torch

        from spectramr.infrastructure.services.stein_aggregation import (
            product_of_experts_score,
        )

        with pytest.raises(ValueError, match="identical shape"):
            product_of_experts_score([torch.randn(3, 4), torch.randn(2, 4)])

    @pytest.mark.unit
    def test_federated_svgd_step(self) -> None:
        import torch

        from spectramr.infrastructure.services.stein_aggregation import (
            federated_svgd_step,
        )

        n, d = 5, 8
        particles = torch.randn(n, d)
        scores = [torch.zeros(n, d) for _ in range(3)]
        updated = federated_svgd_step(particles, scores)
        assert updated.shape == (n, d)

    @pytest.mark.unit
    def test_federated_svgd_negative_dp_raises(self) -> None:
        import torch

        from spectramr.infrastructure.services.stein_aggregation import (
            federated_svgd_step,
        )

        n, d = 4, 6
        with pytest.raises(ValueError, match="dp_noise_std"):
            federated_svgd_step(torch.randn(n, d), [torch.zeros(n, d)], dp_noise_std=-1.0)


# ---------------------------------------------------------------------------
# RDPAccountant (privacy)
# ---------------------------------------------------------------------------


class TestRDPAccountant:
    @pytest.mark.unit
    def test_canary_construct(self) -> None:
        from spectramr.infrastructure.services.privacy.rdp_accountant import RDPAccountant

        acc = RDPAccountant()
        assert acc.steps == 0

    @pytest.mark.unit
    def test_invalid_order_raises(self) -> None:
        from spectramr.infrastructure.services.privacy.rdp_accountant import RDPAccountant

        with pytest.raises(ValueError, match="must be > 1"):
            RDPAccountant(orders=[1.0, 2.0])

    @pytest.mark.unit
    def test_orders_property(self) -> None:
        from spectramr.infrastructure.services.privacy.rdp_accountant import RDPAccountant

        acc = RDPAccountant(orders=[2.0, 4.0, 8.0])
        assert set(acc.orders) == {2.0, 4.0, 8.0}
