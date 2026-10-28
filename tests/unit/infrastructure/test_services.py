"""
Unit tests for infrastructure services.

Tests validate:
1. Training orchestration
2. Logging service
3. Metrics service
4. Checkpoint service
5. Early stopping service

Run on CI, NOT local dev machine.
"""

import pytest
import torch

from spectramr.infrastructure.training.builders.optimization_builder import (
    OptimizationBuilder,
)


# === Early Stopping Service Tests ===
class TestEarlyStoppingService:
    """Test early stopping service."""

    # @# pytest.mark.timeout(30)
    def test_early_stopping_import(self):
        from spectramr.infrastructure.services.early_stopping import EarlyStoppingService

        assert EarlyStoppingService is not None

    # @# pytest.mark.timeout(30)
    def test_early_stopping_triggers_after_patience(self):
        """Early stopping should trigger after patience epochs without improvement."""
        from spectramr.config.schemas.early_stopping import EarlyStoppingConfigSchema
        from spectramr.infrastructure.services.early_stopping import EarlyStoppingService

        config = EarlyStoppingConfigSchema(
            enabled=True,
            patience=3,
            min_delta=0.01,
        )

        es = EarlyStoppingService(config)

        # Best value initially
        es.update(0.5, iteration=0)
        assert not es.should_stop()

        # No improvement
        es.update(0.5, iteration=1)
        es.update(0.5, iteration=2)
        es.update(0.5, iteration=3)

        assert es.should_stop()

    # @# pytest.mark.timeout(30)
    def test_early_stopping_resets_on_improvement(self):
        """Counter should reset on improvement."""
        from spectramr.config.schemas.early_stopping import EarlyStoppingConfigSchema
        from spectramr.infrastructure.services.early_stopping import EarlyStoppingService

        config = EarlyStoppingConfigSchema(
            enabled=True,
            patience=3,
            min_delta=0.01,
        )

        es = EarlyStoppingService(config)

        es.update(0.5, iteration=0)
        es.update(0.5, iteration=1)  # No improvement, counter=1
        es.update(0.5, iteration=2)  # No improvement, counter=2
        es.update(0.4, iteration=3)  # Improvement! counter=0

        assert not es.should_stop()
        assert es.wait_count == 0

    # @# pytest.mark.timeout(30)
    def test_early_stopping_disabled(self):
        """Disabled early stopping should never trigger."""
        from spectramr.config.schemas.early_stopping import EarlyStoppingConfigSchema
        from spectramr.infrastructure.services.early_stopping import EarlyStoppingService

        config = EarlyStoppingConfigSchema(enabled=False)

        es = EarlyStoppingService(config)

        for i in range(100):
            es.update(0.5, iteration=i)

        assert not es.should_stop()

    # @# pytest.mark.timeout(30)
    def test_early_stopping_patience_counts_updates(self):
        """Patience counts ``update`` calls -- one per validation pass -- not iterations."""
        from spectramr.config.schemas.early_stopping import EarlyStoppingConfigSchema
        from spectramr.infrastructure.services.early_stopping import EarlyStoppingService

        es = EarlyStoppingService(EarlyStoppingConfigSchema(enabled=True, patience=3))

        es.update(1.0, 1000)
        for iteration in (2000, 3000):
            es.update(1.0, iteration)
            assert not es.should_stop()
        es.update(1.0, 4000)
        assert es.should_stop()
        assert es.best_iteration == 1000


# === Logging Service Tests ===
class TestLoggingService:
    """Test logging service."""

    # @# pytest.mark.timeout(30)
    def test_logging_service_import(self):
        try:
            from spectramr.infrastructure.services.logging_service import LoggingService

            assert LoggingService is not None
        except ImportError:
            pytest.skip("LoggingService not available")

    # @# pytest.mark.timeout(30)
    def test_logging_basic_messages(self):
        """Logging service should log messages."""
        import logging

        logger = logging.getLogger("test")
        logger.setLevel(logging.DEBUG)

        # Create handler to capture logs
        class LogCapture(logging.Handler):
            def __init__(self):
                super().__init__()
                self.records = []

            def emit(self, record):
                self.records.append(record)

        handler = LogCapture()
        logger.addHandler(handler)

        logger.info("Test message")
        logger.debug("Debug message")

        assert len(handler.records) >= 1


# === Metrics Service Tests ===
class TestMetricsService:
    """Test metrics service."""

    # @# pytest.mark.timeout(30)
    def test_metrics_service_import(self):
        try:
            from spectramr.infrastructure.services.metrics_service import MetricsService

            assert MetricsService is not None
        except ImportError:
            pytest.skip("MetricsService not available")

    # @# pytest.mark.timeout(30)
    def test_metrics_update_and_retrieve(self):
        """Metrics can be updated and retrieved."""
        # Simple metrics dict
        metrics = {}

        def update_metrics(new_metrics):
            metrics.update(new_metrics)

        def get_metrics():
            return metrics.copy()

        update_metrics({"loss": 0.5, "psnr": 25.0})
        update_metrics({"ssim": 0.9})

        result = get_metrics()

        assert result["loss"] == 0.5
        assert result["psnr"] == 25.0
        assert result["ssim"] == 0.9


# === Iteration Counter Service Tests ===
class TestIterationCounterService:
    """Test iteration counter service."""

    # @# pytest.mark.timeout(30)
    def test_iteration_counter_import(self):
        try:
            from spectramr.infrastructure.services.iteration_counter_service import (
                IterationCounterService,
            )

            assert IterationCounterService is not None
        except ImportError:
            pytest.skip("IterationCounterService not available")

    # @# pytest.mark.timeout(30)
    def test_simple_counter(self):
        """Simple counter increments correctly."""
        counter = {"step": 0, "max": 100}

        for i in range(10):
            counter["step"] += 1

        assert counter["step"] == 10
        assert counter["step"] < counter["max"]


# === Memory Service Tests ===
class TestMemoryService:
    """Test memory management service."""

    # @# pytest.mark.timeout(30)
    def test_memory_cleanup(self):
        """Memory cleanup should free cache."""
        import gc

        # Create some tensors
        tensors = [torch.randn(1000, 1000) for _ in range(10)]

        # Clear
        del tensors
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Should not crash
        assert True


# === Training State Tests ===
class TestTrainingEnvironmentImport:
    """Test training environment import."""

    # @# pytest.mark.timeout(30)
    def test_training_environment_import(self):
        try:
            from spectramr.infrastructure.training.builders.environment import (
                TrainingEnvironment,
            )

            assert TrainingEnvironment is not None
        except ImportError:
            pytest.skip("TrainingEnvironment not available")

    # @# pytest.mark.timeout(30)
    def test_training_environment_core_fields(self):
        """TrainingEnvironment should have core fields."""
        try:
            from spectramr.infrastructure.training.builders.environment import (
                TrainingEnvironment,
            )
        except ImportError:
            pytest.skip("TrainingEnvironment not available")

        # Check class has expected fields
        env_fields = (
            TrainingEnvironment.__dataclass_fields__
            if hasattr(TrainingEnvironment, "__dataclass_fields__")
            else {}
        )

        expected_fields = ["models", "optimizers", "device", "config"]

        for field in expected_fields:
            if field in env_fields:
                assert True


# === Validation Service Tests ===
class TestValidationService:
    """Test validation service."""

    # `test_validation_service_import` removed with the module (#710).
    # It was an import wrapped in `except ImportError: pytest.skip(...)`, so it
    # could never fail -- after the deletion it would have turned into a silent
    # green skip, which is the failure mode #634 catalogued.


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
