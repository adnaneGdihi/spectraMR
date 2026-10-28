"""Diffusion validation scores phase metrics on the complex image (#867).

``DiffusionTrainingStrategy._compute_validation_metrics`` runs its own metric
transform and computer call rather than the mixin's ``validation_step``. Before
it was routed through ``_metric_transform_views`` it handed the computer only the
``abs()`` of the SENSE adjoint, so a declared ``phase_mse`` met a real magnitude
image, raised ``PhaseBlindInputError``, and the re-raise dropped the arm's whole
validation metric dict. The planted case is a prediction that differs from its
target only by a global phase of pi/3: the right answer is (pi/3)^2.
"""

from __future__ import annotations

import math
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import spectramr.core.metrics.evaluation_metrics  # noqa: F401  (registers the metrics)
from spectramr.config.schemas.data import DataProcessingConfigSchema
from spectramr.core.metrics.complex_input import PhaseBlindInputError
from spectramr.core.metrics.computer import ValidationMetricsComputer
from spectramr.core.metrics.types import ValidationMetricsConfig
from spectramr.infrastructure.physics.fft_ops import fft2c
from spectramr.infrastructure.training.strategies.diffusion import DiffusionTrainingStrategy
from spectramr.infrastructure.training.strategies.mixins.metrics_mixin import MetricsMixin
from spectramr.infrastructure.training.utils.metric_views import MetricViews

SHIFT = math.pi / 3


def _phase_only_pair(coils: int = 4):
    """Multi-coil k-space whose SENSE-combined images differ only by a global phase."""
    torch.manual_seed(0)
    image = torch.complex(torch.rand(2, 1, 16, 16) + 0.5, torch.zeros(2, 1, 16, 16))
    phases = torch.rand(2, coils, 16, 16) * 2 * math.pi
    smaps = (torch.polar(torch.ones_like(phases), phases) / math.sqrt(coils)).to(torch.complex64)
    rotation = complex(math.cos(SHIFT), math.sin(SHIFT))
    return fft2c(image * rotation * smaps), fft2c(image * smaps), smaps


def _strategy(smaps: torch.Tensor | None) -> MagicMock:
    """A ``self`` that runs ``_compute_validation_metrics`` with the real transform hook."""
    mock = MagicMock()
    mock.config = SimpleNamespace(
        data=SimpleNamespace(
            processing=DataProcessingConfigSchema(
                enable_kspace_normalization=False, enable_log_scaling=False
            )
        ),
        model=SimpleNamespace(input_type="kspace", model_type="kspace_cold_diffusion"),
        validation=SimpleNamespace(
            scoring=SimpleNamespace(
                enable_image_metrics=True,
                domain="image",
                output_transform="ifft_sense_adjoint",
            )
        ),
    )
    mock._is_cold_diffusion = MagicMock(return_value=True)
    mock._metric_transform_views = MethodType(MetricsMixin._metric_transform_views, mock)
    mock._apply_metric_transforms = MethodType(MetricsMixin._apply_metric_transforms, mock)
    mock._select_batch_compatible_smaps = lambda batch: smaps
    mock._measure_prediction_scale = lambda pred, target: {
        "pred_above_target_fraction": 0.0,
        "target_abs_max": 1.0,
        "pred_target_scale_ratio": 1.0,
    }
    mock._convert_metrics_to_floats = lambda d: d
    mock._PRED_SCALE_WARN_FRACTION = DiffusionTrainingStrategy._PRED_SCALE_WARN_FRACTION
    mock._ZF_BASELINE_METRICS = DiffusionTrainingStrategy._ZF_BASELINE_METRICS
    mock.validation_metrics_computer = ValidationMetricsComputer(
        ValidationMetricsConfig.from_dict({"metrics": ["psnr", "phase_mse"]}), device="cpu"
    )
    mock._zf_measurement = None
    return mock


def _validate(mock: MagicMock, pred: torch.Tensor, target: torch.Tensor) -> dict:
    return DiffusionTrainingStrategy._compute_validation_metrics(
        mock,
        pred,
        target,
        target.clone(),
        torch.zeros(pred.shape[0], dtype=torch.long),
        None,
        torch.ones(pred.shape[0]),
    )


@pytest.mark.parametrize("with_smaps", [True, False], ids=["sense", "per_coil"])
def test_phase_only_error_is_measured_and_the_dict_survives(with_smaps: bool) -> None:
    pred_k, target_k, smaps = _phase_only_pair()
    out = _validate(_strategy(smaps if with_smaps else None), pred_k, target_k)

    assert float(out["val_phase_mse"]) == pytest.approx(SHIFT**2, rel=1e-3)
    # The magnitude metric is still scored on |A^H y|, where the pair is identical.
    assert "val_psnr" in out and "val_loss" in out


def test_magnitude_only_routing_is_red() -> None:
    """Planted: the pre-routing call shape -- magnitude pair only -- raises and loses the dict."""
    pred_k, target_k, smaps = _phase_only_pair()
    mock = _strategy(smaps)

    def _magnitude_only(pred, target, cfg):
        views = MetricsMixin._metric_transform_views(mock, pred, target, cfg)
        return MetricViews(views.pred, views.target)

    mock._metric_transform_views = _magnitude_only
    with pytest.raises(PhaseBlindInputError):
        _validate(mock, pred_k, target_k)
