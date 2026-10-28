"""Diffusion validation pays for images and diagnostics only where they are read.

``_compute_validation_metrics`` runs once per (batch, rung). On a batch that is
not the first, nothing consumes the TensorBoard panels or the INFO min/max
diagnostics, so computing them is pure host-sync and kernel waste.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock

import torch

import spectramr.core.metrics.evaluation_metrics  # noqa: F401  (registers the metrics)
from spectramr.config.schemas.data import DataProcessingConfigSchema
from spectramr.core.metrics.computer import ValidationMetricsComputer
from spectramr.core.metrics.types import ValidationMetricsConfig
from spectramr.infrastructure.training.strategies.diffusion import DiffusionTrainingStrategy
from spectramr.infrastructure.training.strategies.mixins.metrics_mixin import MetricsMixin

_TRANSFERS = ("item", "tolist", "cpu", "__bool__")


@contextmanager
def _count_host_transfers():
    """Count every Tensor method that blocks on the device."""
    counts = dict.fromkeys(_TRANSFERS, 0)
    originals = {name: getattr(torch.Tensor, name) for name in _TRANSFERS}

    def _wrap(name):
        def _counted(self, *args, **kwargs):
            counts[name] += 1
            return originals[name](self, *args, **kwargs)

        return _counted

    for name in _TRANSFERS:
        setattr(torch.Tensor, name, _wrap(name))
    try:
        yield counts
    finally:
        for name, fn in originals.items():
            setattr(torch.Tensor, name, fn)


def _strategy(*, writer: object | None = None) -> MagicMock:
    mock = MagicMock()
    mock.config = SimpleNamespace(
        data=SimpleNamespace(
            processing=DataProcessingConfigSchema(
                enable_kspace_normalization=False, enable_log_scaling=False
            )
        ),
        model=SimpleNamespace(input_type="image", model_type="diffusion"),
        validation=SimpleNamespace(
            scoring=SimpleNamespace(
                enable_image_metrics=True, domain="image", output_transform="none"
            )
        ),
        logging=SimpleNamespace(
            images=SimpleNamespace(
                log_validation=True, max_per_batch=2, log_difference=True, log_input=False
            )
        ),
    )
    ls = MagicMock()
    ls.logger.isEnabledFor = lambda level: level >= logging.INFO
    ls._writer = writer
    ls._enable_tensorboard = writer is not None
    mock.logging_service = ls
    mock._is_cold_diffusion = MagicMock(return_value=False)
    mock._metric_transform_views = MethodType(MetricsMixin._metric_transform_views, mock)
    mock._apply_metric_transforms = MethodType(MetricsMixin._apply_metric_transforms, mock)
    mock._measure_prediction_scale = lambda pred, target: {
        "pred_above_target_fraction": 0.0,
        "target_abs_max": 1.0,
        "pred_target_scale_ratio": 1.0,
    }
    mock._convert_metrics_to_floats = MetricsMixin._convert_metrics_to_floats
    mock._PRED_SCALE_WARN_FRACTION = DiffusionTrainingStrategy._PRED_SCALE_WARN_FRACTION
    mock._ZF_BASELINE_METRICS = DiffusionTrainingStrategy._ZF_BASELINE_METRICS
    mock._report_case_recorder = None
    mock._per_case_metric_sink = None
    mock.metrics_service = None
    mock.validation_step_count = 0
    mock._zf_measurement = None
    mock._log_validation_images_to_tensorboard = MethodType(
        DiffusionTrainingStrategy._log_validation_images_to_tensorboard, mock
    )
    mock.validation_metrics_computer = ValidationMetricsComputer(
        ValidationMetricsConfig.from_dict({"metrics": ["psnr"]}), device="cpu"
    )
    return mock


def _validate(mock: MagicMock, batch_idx: int) -> dict:
    torch.manual_seed(0)
    target = torch.rand(2, 1, 16, 16) + 0.5
    pred = target + 0.05 * torch.rand_like(target)
    return DiffusionTrainingStrategy._compute_validation_metrics(
        mock,
        pred,
        target,
        target.clone(),
        torch.zeros(2, dtype=torch.long),
        None,
        torch.ones(2),
        batch_idx=batch_idx,
    )


def test_non_first_batch_pays_only_the_range_and_metric_transfers() -> None:
    mock = _strategy()
    with _count_host_transfers() as counts:
        out = _validate(mock, batch_idx=1)

    # Remaining transfers, each named:
    #   item   x1: the dynamic data range handed to the metric computer;
    #   cpu    x2, tolist x2: two fused scalar transfers -- the metric computer's
    #          and the final ``_convert_metrics_to_floats`` (which now also carries
    #          val_timestep_mean / val_pred_mean / val_target_mean);
    #   __bool__ x0: the data-range branch is a ``torch.where``, not tensor-in-``if``.
    assert counts == {"item": 1, "tolist": 2, "cpu": 2, "__bool__": 0}, counts
    assert all(isinstance(v, float) for v in out.values())
    mock.logging_service.log_images_batch.assert_not_called()


def test_first_batch_with_a_writer_still_logs_the_panels() -> None:
    """Covers a state production does not currently reach: ``LoggingService``
    leaves ``_writer`` as ``None``, so the ``_panels`` path is only live for a caller
    that attaches a TensorBoard writer itself.
    """
    mock = _strategy(writer=object())
    _validate(mock, batch_idx=0)
    mock.logging_service.log_images_batch.assert_called_once()


def test_no_panel_work_without_a_tensorboard_writer() -> None:
    mock = _strategy(writer=None)
    mock.metrics_service = None
    images = MagicMock(side_effect=AssertionError("panels built with no consumer"))
    mock.logging_service.log_images_batch = images
    _validate(mock, batch_idx=0)
    images.assert_not_called()


def test_recorder_is_still_fed_on_every_batch(monkeypatch) -> None:
    from spectramr.infrastructure.training.strategies.mixins import metrics_mixin

    fed = []
    monkeypatch.setattr(metrics_mixin, "feed_report_case_recorder", lambda *a, **k: fed.append(k))
    mock = _strategy()
    mock._report_case_recorder = SimpleNamespace(enabled=True)
    _validate(mock, batch_idx=3)
    assert len(fed) == 1
    mock.logging_service.log_images_batch.assert_not_called()


def _verify_harness() -> MagicMock:
    mock = MagicMock()
    mock.logging_service.logger.isEnabledFor = lambda level: level >= logging.INFO
    return mock


def _masked_call(verify: bool):
    mock = _verify_harness()
    x = torch.rand(2, 1, 8, 8) + 0.5
    mask = (torch.rand(2, 1, 8, 8) > 0.5).float()
    with _count_host_transfers() as counts:
        DiffusionTrainingStrategy._apply_masking_and_verify(mock, x, mask, verify=verify)
    return mock, counts


def test_verify_false_has_no_host_transfer_and_no_log() -> None:
    mock, counts = _masked_call(verify=False)
    assert counts == dict.fromkeys(_TRANSFERS, 0), counts
    mock.logging_service.log_info.assert_not_called()
    mock.logging_service.log_warning.assert_not_called()


def test_verify_true_still_verifies_in_one_fused_transfer() -> None:
    mock, counts = _masked_call(verify=True)
    # One fuse_to_host: a single cpu() and a single tolist().
    assert counts["cpu"] == 1 and counts["tolist"] == 1 and counts["item"] == 0, counts
    mock.logging_service.log_info.assert_called_once()
