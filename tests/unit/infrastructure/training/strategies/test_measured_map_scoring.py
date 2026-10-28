"""The ``validation.scoring.measurement_maps`` family: measured-map metrics beside the oracle ones.

Cold-diffusion validation calibrates coil maps from the fully sampled reference.
The family re-scores the same prediction with maps calibrated from the rung's
masked measurement. The two families must DIFFER at high R on a multi-coil
case; an estimator that secretly reads the reference makes them coincide, and
the planted tests below show the gap assertion going red on exactly that shape
(non-negotiable 15). CPU, synthetic tensors, no model build.
"""

from __future__ import annotations

import inspect
import math
import re
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from spectramr.core.metrics.computer import ValidationMetricsComputer
from spectramr.core.metrics.types import MetricSpec, ValidationMetricsConfig
from spectramr.infrastructure.physics.coil_sensitivity import (
    estimate_smaps_calibrated,
    estimate_smaps_from_measurement,
)
from spectramr.infrastructure.physics.fft_ops import fft2c
from spectramr.infrastructure.training.strategies import measured_map_scoring as mms
from spectramr.infrastructure.training.strategies.diffusion import DiffusionTrainingStrategy

SIZE = 64
#: The gap the measured family must open over the oracle one on this phantom.
#: The real estimator opens far more (see the fires-test); the planted ones close it.
MIN_GAP = 0.05
NAMES = ["psnr", "hfen", "robust_mri_psnr", "ndcr", "nse_hall", "fabrication_excess"]


def _config() -> SimpleNamespace:
    estimation = SimpleNamespace(
        method="power_iter", kernel_size=12, acs_size=24, eigen_threshold=None, maps_path=None
    )
    return SimpleNamespace(
        physics=SimpleNamespace(coil_processing=SimpleNamespace(estimation=estimation)),
        validation=SimpleNamespace(
            scoring=SimpleNamespace(
                enable_image_metrics=True,
                domain="image",
                output_transform="ifft_sense_adjoint",
                measurement_maps="metrics",
            )
        ),
        model=SimpleNamespace(model_type="kspace_cold_diffusion", input_type="kspace"),
    )


def _interleaved(k: torch.Tensor) -> torch.Tensor:
    """Complex ``[B,C,H,W]`` -> ``[B,2C,H,W]`` as ``(re0, im0, re1, im1, ...)``."""
    b, c, h, w = k.shape
    return torch.view_as_real(k).permute(0, 1, 4, 2, 3).reshape(b, 2 * c, h, w)


def _phantom(n_coils: int = 2):
    """``(target, prediction, masked measurement, mask)`` k-space at R=32 on 64 lines."""
    torch.manual_seed(0)
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, SIZE), torch.linspace(-1, 1, SIZE), indexing="ij")
    obj = ((xx / 0.7) ** 2 + (yy / 0.85) ** 2 < 1).float()
    obj = obj * (1.0 + 0.5 * torch.cos(6 * xx) * torch.sin(4 * yy))
    obj = obj.to(torch.complex64)
    centres = [(-0.9, 0.0), (0.9, 0.0), (0.0, 0.9), (0.0, -0.9)][:n_coils]
    maps = torch.stack(
        [
            torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 0.8)
            * torch.exp(1j * math.pi * (0.6 * i + 0.4) * (xx * (i + 1) - yy))
            for i, (cx, cy) in enumerate(centres)
        ]
    ).to(torch.complex64)
    target = fft2c(maps * obj).unsqueeze(0)
    # Per-coil errors: how they combine is exactly what the maps decide.
    noise = 0.05 * target.abs().mean() * torch.randn_like(target)
    prediction = target + noise
    mask = torch.zeros(1, 1, SIZE, SIZE)
    mask[..., SIZE // 2] = 1.0  # two phase-encode lines of 64: R = 32
    mask[..., SIZE // 2 + 7] = 1.0
    return target, prediction, target * mask, mask


def _host(oracle_maps: torch.Tensor | None) -> DiffusionTrainingStrategy:
    host = DiffusionTrainingStrategy.__new__(DiffusionTrainingStrategy)
    host.config = _config()
    host._current_smaps = oracle_maps
    host.logging_service = MagicMock()
    return host


def _computer() -> ValidationMetricsComputer:
    specs = [MetricSpec.from_name(n) for n in NAMES]
    return ValidationMetricsComputer(config=ValidationMetricsConfig(metrics=specs))


def _families(stash: mms.MeasuredMapStash) -> tuple[dict[str, float], dict[str, float]]:
    """``(oracle, measured)`` for one rung, through the production scoring calls."""
    target, prediction, measured, mask = _phantom()
    oracle_maps = estimate_smaps_calibrated(
        target, log_scaled=False, method="power_iter", acs_size=24, kernel_size=12
    )
    host, computer = _host(oracle_maps), _computer()
    t, p, z = _interleaved(target), _interleaved(prediction), _interleaved(measured)
    data_range = 1.0
    views = host._metric_transform_views(p, t, host.config.validation)
    oracle = dict(views.compute(computer, data_range=data_range, only=mms.MEASURED_IMAGE_METRICS))
    oracle.update(host._measurement_aware_metrics(p, t, z, computer, mask))
    family = mms.measured_map_family(
        host, stash, p, t, z, computer, mask, "ifft_sense_adjoint", data_range
    )
    suffix = mms.MEASURED_MAPS_SUFFIX
    measured_out = {
        k.removeprefix("val_").removesuffix(suffix): v
        for k, v in family.items()
        if k.endswith(suffix) and not k.startswith("val_zf_")
    }
    return oracle, measured_out


def _assert_families_differ(oracle: dict[str, float], measured: dict[str, float]) -> None:
    gaps = {k: abs(measured[k] - oracle[k]) / max(abs(oracle[k]), 1e-6) for k in ("psnr", "hfen")}
    assert max(gaps.values()) > MIN_GAP, (
        f"measured-map metrics equal the oracle ones ({gaps}); the estimator read "
        "something other than the masked measurement"
    )


def _real_stash() -> mms.MeasuredMapStash:
    _, _, measured, _ = _phantom()
    return mms.measured_map_stash(_interleaved(measured), config=_config(), log_scaled=False)


# --- the fires-test ------------------------------------------------------------
def test_measured_maps_differ_from_the_oracle_at_high_r() -> None:
    target, _, _, _ = _phantom()
    oracle_maps = estimate_smaps_calibrated(
        target, log_scaled=False, method="power_iter", acs_size=24, kernel_size=12
    )
    stash = _real_stash()
    assert stash.maps is not None and stash.skip_reason is None
    assert torch.isfinite(stash.maps.abs()).all()
    rel = (stash.maps - oracle_maps).abs().norm() / oracle_maps.abs().norm()
    assert rel > 0.1, f"two lines of 64 calibrate maps {rel:.3f} away from the reference's"


def test_the_two_families_differ_and_carry_every_declared_name() -> None:
    oracle, measured = _families(_real_stash())
    _assert_families_differ(oracle, measured)
    assert {"psnr", "hfen", "robust_mri_psnr", "ndcr", "nse_hall"} <= set(measured)
    assert all(math.isfinite(measured[k]) for k in ("psnr", "hfen", "ndcr")), measured


def test_the_family_scores_the_zero_filled_baseline_and_stamps_applied() -> None:
    target, prediction, measured, mask = _phantom()
    host = _host(None)
    family = mms.measured_map_family(
        host,
        _real_stash(),
        _interleaved(prediction),
        _interleaved(target),
        _interleaved(measured),
        _computer(),
        mask,
        "ifft_sense_adjoint",
        1.0,
    )
    assert float(family[mms.APPLIED_KEY]) == 1.0
    assert {"val_zf_psnr_measmaps", "val_zf_hfen_measmaps"} <= set(family)


# --- planted violations: the gap assertion must go red ------------------------
def test_planted_an_estimator_reading_the_target_closes_the_gap(monkeypatch) -> None:
    """The secretly-oracle estimator: the production stash, with the reference inside."""
    target, _, _, _ = _phantom()
    real = estimate_smaps_from_measurement

    def _reads_the_target(measurement, *, config, log_scaled):
        return real(target, config=config, log_scaled=log_scaled)

    monkeypatch.setattr(mms, "estimate_smaps_from_measurement", _reads_the_target)
    with pytest.raises(AssertionError, match="equal the oracle"):
        _assert_families_differ(*_families(_real_stash()))


def test_planted_an_estimator_reading_the_unmasked_input_closes_the_gap() -> None:
    """``input_batch`` is the dataloader's repetition BEFORE the rung mask."""
    target, _, _, _ = _phantom()
    repetition = target + 0.01 * target.abs().mean() * torch.randn_like(target)
    stash = mms.measured_map_stash(_interleaved(repetition), config=_config(), log_scaled=False)
    with pytest.raises(AssertionError, match="equal the oracle"):
        _assert_families_differ(*_families(stash))


def test_the_producer_calibrates_from_the_masked_measurement() -> None:
    """Wiring pin for the same shape as the planted test above, at the call site."""
    src = inspect.getsource(DiffusionTrainingStrategy._generate_validation_prediction)
    assert re.search(r"measured_map_stash\(\s*masked_input\b", src)
    assert not re.search(r"measured_map_stash\(\s*(input_batch|target_batch)\b", src)
    consumer = inspect.getsource(DiffusionTrainingStrategy._compute_validation_metrics)
    assert "self._rung_measured_maps = None" in consumer
    assert consumer.index("measured_map_family(") < consumer.index("metric_views.compute(")


# --- single coil ------------------------------------------------------------------
def test_a_single_coil_rung_is_skipped_with_a_declared_reason() -> None:
    target, prediction, measured, mask = _phantom(n_coils=1)
    stash = mms.measured_map_stash(_interleaved(measured), config=_config(), log_scaled=False)
    assert stash.maps is None and stash.skip_reason == mms.SINGLE_COIL_REASON
    host = _host(None)
    args = (_interleaved(prediction), _interleaved(target), _interleaved(measured))
    for _ in range(2):
        family = mms.measured_map_family(
            host, stash, *args, _computer(), mask, "ifft_sense_adjoint", 1.0
        )
        assert family == {mms.APPLIED_KEY: 0.0}
    host.logging_service.log_info.assert_called_once()
    host.logging_service.log_warning.assert_not_called()  # a warning fails the strict audit


def test_an_undeclared_family_and_a_mock_host_produce_nothing() -> None:
    target, prediction, measured, mask = _phantom()
    args = (_interleaved(prediction), _interleaved(target), _interleaved(measured))
    assert mms.measured_map_family(_host(None), None, *args, None, mask, "x", 1.0) == {}
    assert mms.measured_map_family(MagicMock(), MagicMock(), *args, None, mask, "x", 1.0) == {}


# --- key names ------------------------------------------------------------------
def test_every_key_resolves_through_the_witness_and_has_a_direction() -> None:
    import spectramr.core.metrics  # noqa: F401  -- populate the registry
    from spectramr.core.metrics.metric_directions import metric_higher_is_better
    from spectramr.core.metrics.registry import MetricsRegistry
    from spectramr.infrastructure.training.strategies.mixins.metrics_mixin import (
        universal_validation_keys,
    )
    from spectramr.infrastructure.validation.witness.checks.validation_metric_checks import (
        unresolved_metric_names,
    )

    emitted = DiffusionTrainingStrategy.capabilities.emitted_metrics
    universal = universal_validation_keys(DiffusionTrainingStrategy)
    bases = [*mms.MEASURED_IMAGE_METRICS, *DiffusionTrainingStrategy._MEASUREMENT_AWARE_METRICS]
    names = [f"val_{b}_measmaps{s}" for b in bases for s in ("", "_2x", "_8x", "_32x", "_mean")]
    names += ["val_zf_hfen_measmaps_32x", "val_hfen_measmaps_vs_zf_mean", mms.APPLIED_KEY]
    assert unresolved_metric_names(
        names, MetricsRegistry.is_registered, emitted, universal=universal
    ) == []
    assert metric_higher_is_better("val_psnr_measmaps_32x") is True
    assert metric_higher_is_better("val_hfen_measmaps_mean") is False
    assert metric_higher_is_better("val_hfen_measmaps_vs_zf_mean") is False


def test_planted_an_undeclared_spelling_does_not_resolve() -> None:
    import spectramr.core.metrics  # noqa: F401
    from spectramr.core.metrics.registry import MetricsRegistry
    from spectramr.infrastructure.training.strategies.mixins.metrics_mixin import (
        universal_validation_keys,
    )
    from spectramr.infrastructure.validation.witness.checks.validation_metric_checks import (
        unresolved_metric_names,
    )

    emitted = DiffusionTrainingStrategy.capabilities.emitted_metrics
    universal = universal_validation_keys(DiffusionTrainingStrategy)
    planted = ["val_hfen_measuredmaps_32x"]
    assert unresolved_metric_names(
        planted, MetricsRegistry.is_registered, emitted, universal=universal
    ) == planted


def test_the_cascade_stamps_the_measured_means_and_the_selector_counterpart() -> None:
    levels = [2, 8, 32]
    metrics: dict[str, float] = {}
    for r, (m, z) in zip(levels, [(1.0, 2.0), (2.0, 2.0), (3.0, 2.0)], strict=True):
        metrics[f"val_hfen_measmaps_{r}x"] = m
        metrics[f"val_zf_hfen_measmaps_{r}x"] = z
    bases = mms.measured_mean_bases(["hfen", "lpips"], ["ndcr"])
    assert bases == ["hfen_measmaps"]
    DiffusionTrainingStrategy._stamp_accel_mean(metrics, levels, bases)
    DiffusionTrainingStrategy._stamp_zf_normalised_mean(metrics, levels, metric="hfen_measmaps")
    assert metrics["val_hfen_measmaps_mean"] == pytest.approx(2.0)
    assert metrics["val_hfen_measmaps_vs_zf_mean"] == pytest.approx(1.0)
    src = inspect.getsource(DiffusionTrainingStrategy.validation_step)
    assert "measured_mean_bases(" in src and 'metric="hfen_measmaps"' in src


# --- the knob -------------------------------------------------------------------
def test_the_knob_vocabulary_is_closed_and_defaults_off() -> None:
    from pydantic import ValidationError

    from spectramr.config.schemas.validation import ValidationScoringConfigSchema

    assert ValidationScoringConfigSchema().measurement_maps == "off"
    assert ValidationScoringConfigSchema(measurement_maps="metrics").measurement_maps == "metrics"
    for bad in ("full", "on", "true"):
        with pytest.raises(ValidationError):
            ValidationScoringConfigSchema(measurement_maps=bad)


def test_declared_reads_only_metrics() -> None:
    assert mms.measured_maps_declared(_config()) is True
    off = _config()
    off.validation.scoring.measurement_maps = "off"
    assert mms.measured_maps_declared(off) is False
    assert mms.measured_maps_declared(SimpleNamespace(validation=None)) is False


def test_a_rung_with_no_acs_line_scores_without_raising_and_reports_unapplied() -> None:
    """Two lines, both outside the 24-line ACS window: the maps calibrate to zero."""
    target, prediction, _, _ = _phantom()
    mask = torch.zeros(1, 1, SIZE, SIZE)
    mask[..., 2] = 1.0
    mask[..., SIZE - 3] = 1.0
    measured = target * mask
    stash = mms.measured_map_stash(_interleaved(measured), config=_config(), log_scaled=False)
    assert stash.maps is not None and float(stash.maps.abs().max()) == 0.0
    family = mms.measured_map_family(
        _host(None),
        stash,
        _interleaved(prediction),
        _interleaved(target),
        _interleaved(measured),
        _computer(),
        mask,
        "ifft_sense_adjoint",
        1.0,
    )
    assert float(family[mms.APPLIED_KEY]) == 0.0
    assert "val_hfen_measmaps" in family  # the key set does not depend on the data


# --- observed on the production consumer (non-negotiable 16) ---------------------
def _consumer_host(oracle_maps: torch.Tensor, stash, mask, measured) -> MagicMock:
    from types import MethodType

    from spectramr.config.schemas.data import DataProcessingConfigSchema
    from spectramr.infrastructure.training.strategies.mixins.metrics_mixin import MetricsMixin

    mock = MagicMock()
    mock.config = _config()
    mock.config.data = SimpleNamespace(
        processing=DataProcessingConfigSchema(
            enable_kspace_normalization=False, enable_log_scaling=False
        )
    )
    mock._is_cold_diffusion = MagicMock(return_value=True)
    mock._metric_transform_views = MethodType(MetricsMixin._metric_transform_views, mock)
    mock._apply_metric_transforms = MethodType(MetricsMixin._apply_metric_transforms, mock)
    mock._measurement_aware_metrics = MethodType(
        DiffusionTrainingStrategy._measurement_aware_metrics, mock
    )
    mock._MEASUREMENT_AWARE_METRICS = DiffusionTrainingStrategy._MEASUREMENT_AWARE_METRICS
    mock._K_SPACE_METRICS = DiffusionTrainingStrategy._K_SPACE_METRICS
    mock._select_batch_compatible_smaps = lambda batch: oracle_maps
    mock._measure_prediction_scale = lambda pred, target: {
        "pred_above_target_fraction": 0.0,
        "target_abs_max": 1.0,
        "pred_target_scale_ratio": 1.0,
    }
    mock._convert_metrics_to_floats = lambda d: d
    mock._PRED_SCALE_WARN_FRACTION = DiffusionTrainingStrategy._PRED_SCALE_WARN_FRACTION
    mock._ZF_BASELINE_METRICS = DiffusionTrainingStrategy._ZF_BASELINE_METRICS
    mock.validation_metrics_computer = _computer()
    mock._zf_measurement = measured
    mock._rung_mask = mask
    mock._rung_acquired = None
    mock._ensemble_members = None
    mock._rung_measured_maps = stash
    return mock


def _consume(stash) -> dict:
    target, prediction, measured, mask = _phantom()
    oracle_maps = estimate_smaps_calibrated(
        target, log_scaled=False, method="power_iter", acs_size=24, kernel_size=12
    )
    t, p, z = _interleaved(target), _interleaved(prediction), _interleaved(measured)
    host = _consumer_host(oracle_maps, stash, mask, z)
    return DiffusionTrainingStrategy._compute_validation_metrics(
        host, p, t, t.clone(), torch.zeros(1, dtype=torch.long), None, torch.ones(1)
    )


def test_the_production_consumer_publishes_the_family_beside_unchanged_oracle_keys() -> None:
    with_family = _consume(_real_stash())
    without = _consume(None)
    for key in (
        "val_hfen_measmaps",
        "val_psnr_measmaps",
        "val_zf_hfen_measmaps",
        "val_ndcr_measmaps",
    ):
        assert key in with_family, key
    assert float(with_family[mms.APPLIED_KEY]) == 1.0
    assert not any("measmaps" in k for k in without)
    for key in ("val_hfen", "val_psnr", "val_zf_hfen", "val_ndcr"):
        assert float(with_family[key]) == float(without[key]), key
    assert abs(float(with_family["val_hfen_measmaps"]) - float(with_family["val_hfen"])) > 1e-4
