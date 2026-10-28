"""Early stopping's improvement callback, the signal the best-checkpoint save keys on.

Verifies that the callback fires on an improvement and not otherwise. The best file
itself is tested in ``tests/unit/infrastructure/checkpointing/test_store.py``.
"""

from unittest.mock import MagicMock

from spectramr.config.schemas.early_stopping import EarlyStoppingConfigSchema
from spectramr.config.schemas.enums import MetricMode


class TestEarlyStoppingBestCheckpointIntegration:
    """Test early stopping service improvement callback behaviour."""

    def _make_service(self, mode=MetricMode.MIN, on_improvement=None, metric=None):
        """Helper to create an enabled EarlyStoppingService.

        The metric now tracks the mode. `val_loss` is declared lower-is-better, so
        pairing it with MAX is the contradiction #712's validator rejects -- these
        tests exercise MAX *mechanics*, and the metric name was incidental.
        """
        from spectramr.infrastructure.services.early_stopping import EarlyStoppingService

        if metric is None:
            metric = "val_psnr" if str(mode).lower().endswith("max") else "val_loss"
        config = EarlyStoppingConfigSchema(
            enabled=True,
            patience=5,
            metric=metric,
            mode=mode,
            min_delta=0.0,
        )
        return EarlyStoppingService(config, on_improvement=on_improvement)

    def test_on_improvement_callback_fires_on_better_metric(self):
        """Callback fires when metric improves (MIN mode: lower is better)."""
        callback = MagicMock()
        svc = self._make_service(mode=MetricMode.MIN, on_improvement=callback)

        svc.update(1.0, iteration=100)
        callback.assert_called_once_with(100, 1.0)

        callback.reset_mock()
        svc.update(0.5, iteration=200)
        callback.assert_called_once_with(200, 0.5)

    def test_on_improvement_callback_not_fired_on_worse_metric(self):
        """Callback does NOT fire when metric does not improve."""
        callback = MagicMock()
        svc = self._make_service(mode=MetricMode.MIN, on_improvement=callback)

        svc.update(1.0, iteration=100)
        callback.reset_mock()

        svc.update(1.5, iteration=200)  # Worse (higher in MIN mode)
        callback.assert_not_called()

    def test_wait_count_resets_on_improvement(self):
        """wait_count should be 0 after an improvement."""
        svc = self._make_service(mode=MetricMode.MIN)

        svc.update(1.0, iteration=100)
        assert svc.wait_count == 0

        svc.update(1.5, iteration=200)  # No improvement
        assert svc.wait_count == 1

        svc.update(0.8, iteration=300)  # Improvement!
        assert svc.wait_count == 0

    def test_on_improvement_callback_fires_max_mode(self):
        """Callback fires when metric improves (MAX mode: higher is better)."""
        callback = MagicMock()
        svc = self._make_service(mode=MetricMode.MAX, on_improvement=callback)

        svc.update(10.0, iteration=100)
        callback.assert_called_once_with(100, 10.0)

        callback.reset_mock()
        svc.update(15.0, iteration=200)  # Better (higher in MAX mode)
        callback.assert_called_once_with(200, 15.0)

        callback.reset_mock()
        svc.update(12.0, iteration=300)  # Worse
        callback.assert_not_called()
