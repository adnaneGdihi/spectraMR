"""The percentile-scale contract and the one resolver that reads it (#2241).

Planted on the cluster shape: an RSS image whose 99th percentile the
``kspace_scale_domain: image`` normalization put at 1.0 and whose brightest voxels
reach 2.05. The peak test reports that batch not-applicable, which the metrics
computer records as NaN on every epoch; the declared contract resolves 1.0.
"""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.core.metrics.computer import create_validation_metrics_computer
from spectramr.core.metrics.data_range_contract import (
    DataRangeContractError,
    QuantileScaleContract,
)
from spectramr.core.metrics.evaluation_metrics import (
    PSNR,
    ClinicalSSIM,
    SSIMMetric,
    resolve_image_data_range,
)
from spectramr.core.metrics.outcome import MetricNotApplicableError

_P99 = QuantileScaleContract(quantile=0.99, source="test")


def _p99_image(peak: float = 2.05, scale: float = 1.0) -> torch.Tensor:
    """Disc at 1.0 over a zero background, with a 3x3 feature (0.2 % of voxels) at ``peak``."""
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, 64), torch.linspace(-1, 1, 64), indexing="ij")
    image = ((xx**2 + yy**2) < 0.6).float()
    image[20:23, 30:33] = peak
    return (image * scale).expand(2, 1, 64, 64).clone()


def test_the_planted_image_is_the_cluster_shape() -> None:
    image = _p99_image()
    assert float(torch.quantile(image.flatten(), 0.99)) == pytest.approx(1.0)
    assert float(image.max()) == pytest.approx(2.05)


class TestThePlantedViolation:
    def test_the_peak_test_reports_it_not_applicable(self) -> None:
        with pytest.raises(MetricNotApplicableError):
            resolve_image_data_range(_p99_image(), None, metric_name="psnr")

    def test_the_declared_contract_resolves_its_anchor(self) -> None:
        assert (
            resolve_image_data_range(_p99_image(), None, metric_name="psnr", contract=_P99) == 1.0
        )

    def test_below_the_peak_tolerance_both_agree(self) -> None:
        image = _p99_image(peak=1.5)
        assert resolve_image_data_range(image, None, metric_name="psnr") == (
            resolve_image_data_range(image, None, metric_name="psnr", contract=_P99)
        )


class TestAContradictedContractRaisesByName:
    def test_an_unapplied_scale_raises(self) -> None:
        with pytest.raises(DataRangeContractError, match=r"quantile 0\.99"):
            resolve_image_data_range(
                _p99_image(scale=2400.0), None, metric_name="psnr", contract=_P99
            )

    def test_it_is_not_the_not_applicable_the_computer_turns_into_nan(self) -> None:
        assert not issubclass(DataRangeContractError, MetricNotApplicableError)

    def test_the_computer_propagates_it_with_its_type(self) -> None:
        computer = create_validation_metrics_computer(["psnr"], scale_contract=_P99)
        image = _p99_image(scale=2400.0)
        with pytest.raises(DataRangeContractError):
            computer.compute(image * 0.9, image)


def test_a_declared_range_still_wins_over_the_contract() -> None:
    """The per-arm ``data_range`` declarations resolve exactly as before."""
    for declared in (1.0, 2.0):
        for image in (_p99_image(), _p99_image(scale=2400.0)):
            got = resolve_image_data_range(image, declared, metric_name="psnr", contract=_P99)
            assert got == declared


def test_a_signed_target_is_left_to_the_sign_test() -> None:
    """The contract describes a magnitude image; a signed tensor keeps today's answer."""
    signed = _p99_image(peak=0.9) * 2 - 1
    assert resolve_image_data_range(signed, None, metric_name="psnr", contract=_P99) == 2.0
    raw_kspace_like = torch.randn(1, 2, 16, 16) * 50
    with pytest.raises(MetricNotApplicableError):
        resolve_image_data_range(raw_kspace_like, None, metric_name="psnr", contract=_P99)


@pytest.mark.parametrize("quantile", [0.89, 1.0])
def test_a_quantile_outside_its_range_is_refused(quantile: float) -> None:
    with pytest.raises(ValueError, match="quantile"):
        QuantileScaleContract(quantile=quantile, source="test")


@pytest.mark.parametrize("quantile, bound", [(0.99, 100.0), (0.95, 20.0)])
def test_the_peak_bound_sits_where_the_docstring_says(quantile: float, bound: float) -> None:
    contract = QuantileScaleContract(quantile=quantile, source="test")
    assert contract.max_peak == pytest.approx(bound)
    inside = _p99_image(peak=0.99 * bound)
    assert resolve_image_data_range(inside, None, metric_name="psnr", contract=contract) == 1.0
    with pytest.raises(DataRangeContractError):
        resolve_image_data_range(
            _p99_image(peak=1.01 * bound), None, metric_name="psnr", contract=contract
        )


@pytest.mark.parametrize("metric", [PSNR, SSIMMetric, ClinicalSSIM], ids=lambda m: m.__name__)
def test_the_range_sensitive_metrics_read_the_contract(metric: type) -> None:
    target = _p99_image()
    pred = target + 0.01 * torch.randn(target.shape, generator=torch.Generator().manual_seed(0))
    with_contract = float(metric()(pred, target, scale_contract=_P99))
    declared = float(metric()(pred, target, data_range=1.0))
    assert math.isfinite(with_contract)
    assert with_contract == pytest.approx(declared, abs=1e-6)


def test_the_computer_hands_the_contract_to_psnr_and_ssim() -> None:
    target = _p99_image()
    pred = target + 0.01 * torch.randn(target.shape, generator=torch.Generator().manual_seed(0))
    before = create_validation_metrics_computer(["psnr", "ssim"]).compute(pred, target)
    after = create_validation_metrics_computer(["psnr", "ssim"], scale_contract=_P99).compute(
        pred, target
    )
    declared = create_validation_metrics_computer(["psnr", "ssim"], data_range=1.0).compute(
        pred, target
    )
    assert math.isnan(before["psnr"]) and math.isnan(before["ssim"]), before
    for key in ("psnr", "ssim"):
        assert after[key] == pytest.approx(declared[key], abs=1e-6), (key, after, declared)
