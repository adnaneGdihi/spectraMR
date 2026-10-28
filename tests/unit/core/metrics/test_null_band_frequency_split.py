"""Unmeasured-k-space error split into equal-count frequency bands."""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.core.metrics.context import MetricContext
from spectramr.core.metrics.null_band_frequency_split import BAND_METRICS, BANDS, band_errors
from spectramr.core.metrics.registry import MetricsRegistry
from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c

pytestmark = pytest.mark.unit

SIZE = 64


def _image() -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, SIZE), torch.linspace(-1, 1, SIZE), indexing="ij")
    img = ((yy / 0.8) ** 2 + (xx / 0.6) ** 2 < 1).float()
    img = img + 0.3 * torch.sign(torch.sin(25 * xx)) * (img > 0)
    return img.to(torch.complex64)[None, None]


def _central_rows(rows: int) -> torch.Tensor:
    mask = torch.zeros(1, 1, SIZE, SIZE)
    mask[..., SIZE // 2 - rows // 2 : SIZE // 2 + rows // 2, :] = 1.0
    return mask


def _variable_density() -> torch.Tensor:
    """Fully measured disk of radius 20: fixed shells inside it would all be empty."""
    yy, xx = torch.meshgrid(
        torch.arange(SIZE) - SIZE // 2, torch.arange(SIZE) - SIZE // 2, indexing="ij"
    )
    return (torch.sqrt(yy.float() ** 2 + xx.float() ** 2) < 20).float()[None, None]


def _score(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    ctx = MetricContext(mask=mask)
    return {name: cls()(pred, target, context=ctx) for name, cls in BAND_METRICS.items()}


def test_every_band_is_registered() -> None:
    for name, _ in BANDS:
        assert MetricsRegistry.is_registered(name)


@pytest.mark.parametrize("mask", [_central_rows(8), _central_rows(2), _variable_density()])
def test_exact_is_zero_and_zero_filled_is_one_on_every_mask(mask: torch.Tensor) -> None:
    """No band is ever empty or nan, whatever the mask leaves out."""
    target = _image()
    exact = _score(target, target, mask)
    zero_filled = _score(ifft2c(fft2c(target) * mask), target, mask)
    for name in BAND_METRICS:
        assert math.isfinite(exact[name]) and math.isfinite(zero_filled[name])
        assert exact[name] == pytest.approx(0.0, abs=1e-4)
        assert zero_filled[name] == pytest.approx(1.0, abs=1e-4)


def test_a_low_pass_reconstruction_fails_the_outer_band_first() -> None:
    """The cohort's shape: the inner missing content recovered, the outer blurred away."""
    target, mask = _image(), _central_rows(8)
    k = fft2c(target)
    yy, xx = torch.meshgrid(
        torch.arange(SIZE) - SIZE // 2, torch.arange(SIZE) - SIZE // 2, indexing="ij"
    )
    keep = (torch.sqrt(yy.float() ** 2 + xx.float() ** 2) < 14).to(torch.complex64)
    scores = _score(ifft2c(k * mask + k * (1 - mask) * keep), target, mask)
    assert scores["null_band_nrmse_outer"] == pytest.approx(1.0, abs=1e-4)
    assert (
        scores["null_band_nrmse_inner"]
        < scores["null_band_nrmse_middle"]
        <= scores["null_band_nrmse_outer"]
    )


def test_the_bands_split_the_unmeasured_bins_in_equal_thirds() -> None:
    """A uniform 50 % error everywhere reads 0.5 in every band: the bands partition the same bins."""
    mask = _central_rows(8)
    target = _image()
    errs = band_errors(target * 1.5, target, mask)
    assert all(e == pytest.approx(0.5, abs=1e-4) for e in errs)


def test_without_a_mask_the_metric_has_no_value() -> None:
    cls = BAND_METRICS["null_band_nrmse_outer"]
    assert math.isnan(cls()(_image(), _image(), context=MetricContext()))


def test_the_cold_diffusion_validation_seam_computes_every_band() -> None:
    from spectramr.infrastructure.training.strategies.diffusion import DiffusionTrainingStrategy

    for name, _ in BANDS:
        assert name in DiffusionTrainingStrategy._MEASUREMENT_AWARE_METRICS
