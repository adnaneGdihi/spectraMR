"""``requires_complex`` is declared on the registry and enforced at the computer (#867).

Before this change ``phase_mse`` and ``ipen`` handed a real magnitude image to
``fft2c`` and reported the k-space phase of ``|x|``; ``complex_hfen`` read
``.imag`` of a real tensor and crashed to NaN. Each test below plants one shape of
phase-blind input and asserts the refusal.
"""

from __future__ import annotations

import math

import pytest
import torch

import spectramr.core.metrics.evaluation_metrics  # noqa: F401  (registers the metrics)
from spectramr.core.metrics.complex_input import PhaseBlindInputError, as_phase_bearing
from spectramr.core.metrics.computer import ValidationMetricsComputer
from spectramr.core.metrics.registry import MetricsRegistry, register_metric
from spectramr.core.metrics.types import ValidationMetricsConfig

PHASE_METRICS = ("phase_mse", "ipen", "complex_hfen")


def _phase_shifted_pair(shift: float = math.pi / 3) -> tuple[torch.Tensor, torch.Tensor]:
    """Complex image and the same image rotated by a global phase ``shift``."""
    torch.manual_seed(0)
    magnitude = torch.rand(2, 1, 16, 16) + 0.5  # above phase_mse's 0.05 mask
    target = torch.complex(magnitude, torch.zeros_like(magnitude))
    pred = target * complex(math.cos(shift), math.sin(shift))
    return pred, target


def _computer(*names: str) -> ValidationMetricsComputer:
    return ValidationMetricsComputer(ValidationMetricsConfig.from_dict({"metrics": list(names)}))


class TestRegistryDeclaration:
    @pytest.mark.parametrize("name", PHASE_METRICS)
    def test_phase_metrics_declare_requires_complex(self, name: str) -> None:
        assert MetricsRegistry.requires_complex(name)

    @pytest.mark.parametrize("name", ["psnr", "ssim", "mae", "kspace_error"])
    def test_magnitude_metrics_do_not(self, name: str) -> None:
        assert not MetricsRegistry.requires_complex(name)

    def test_alias_and_unknown_name(self) -> None:
        assert MetricsRegistry.requires_complex("PhaseMSE")
        assert not MetricsRegistry.requires_complex("no_such_metric_867")

    def test_class_attribute_is_read_when_the_decorator_is_silent(self) -> None:
        snapshot = MetricsRegistry.snapshot()
        try:

            @register_metric("_probe_867_complex")
            class _Probe:
                requires_complex = True

            assert MetricsRegistry.requires_complex("_probe_867_complex")
        finally:
            MetricsRegistry.restore(snapshot)


class TestAsPhaseBearing:
    def test_complex_is_returned_unchanged(self) -> None:
        z = torch.complex(torch.randn(1, 3, 4, 4), torch.randn(1, 3, 4, 4))
        assert as_phase_bearing("phase_mse", z) is z

    def test_two_channel_real_imag_folds_to_complex(self) -> None:
        stacked = torch.randn(2, 2, 4, 4)
        out = as_phase_bearing("phase_mse", stacked)
        assert torch.is_complex(out) and out.shape == (2, 1, 4, 4)
        assert torch.equal(out.real[:, 0], stacked[:, 0])
        assert torch.equal(out.imag[:, 0], stacked[:, 1])

    @pytest.mark.parametrize(
        "shape", [(2, 1, 4, 4), (2, 3, 4, 4), (2, 4, 4, 4), (2, 4, 4)], ids=str
    )
    def test_real_tensor_without_phase_raises(self, shape: tuple[int, ...]) -> None:
        with pytest.raises(PhaseBlindInputError, match="requires_complex"):
            as_phase_bearing("phase_mse", torch.rand(shape))


class TestMetricsRefuseMagnitude:
    """The direct-call path: the metric itself no longer fft2c's a magnitude."""

    @pytest.mark.parametrize("name", PHASE_METRICS)
    def test_magnitude_input_raises(self, name: str) -> None:
        metric = MetricsRegistry.get(name, device="cpu")
        magnitude = torch.rand(2, 1, 16, 16) + 0.5
        with pytest.raises(PhaseBlindInputError):
            metric(magnitude, magnitude.clone())

    def test_phase_mse_measures_a_global_phase_shift(self) -> None:
        pred, target = _phase_shifted_pair()
        value = MetricsRegistry.get("phase_mse", device="cpu")(pred, target)
        assert float(value) == pytest.approx((math.pi / 3) ** 2, rel=1e-4)

    def test_phase_mse_on_multicoil_complex_does_not_misread_two_coils_as_real_imag(
        self,
    ) -> None:
        # Two complex coils used to hit the ``size(1) == 2`` real/imag branch first.
        pred, target = _phase_shifted_pair()
        pred2, target2 = pred.repeat(1, 2, 1, 1), target.repeat(1, 2, 1, 1)
        value = MetricsRegistry.get("phase_mse", device="cpu")(pred2, target2)
        assert float(value) == pytest.approx((math.pi / 3) ** 2, rel=1e-4)


class TestComputerRouting:
    def test_complex_pair_reaches_the_phase_metric_only(self) -> None:
        pred_c, target_c = _phase_shifted_pair()
        out = _computer("phase_mse", "mae").compute(
            pred_c.abs(), target_c.abs(), complex_pair=(pred_c, target_c)
        )
        assert out["phase_mse"] == pytest.approx((math.pi / 3) ** 2, rel=1e-4)
        assert out["mae"] == pytest.approx(0.0, abs=1e-6)  # magnitudes are identical

    def test_magnitude_metrics_ignore_the_complex_pair(self) -> None:
        pred_c, target_c = _phase_shifted_pair()
        pred_m, target_m = pred_c.abs() * 0.9, target_c.abs()
        with_pair = _computer("mae", "psnr").compute(
            pred_m, target_m, complex_pair=(pred_c, target_c)
        )
        without = _computer("mae", "psnr").compute(pred_m, target_m)
        assert with_pair == without

    def test_phase_metric_on_magnitude_raises_instead_of_nan(self) -> None:
        # The broad ``except Exception`` would record NaN; the guard sits outside it.
        magnitude = torch.rand(2, 1, 16, 16) + 0.5
        with pytest.raises(PhaseBlindInputError, match="phase_mse"):
            _computer("mae", "phase_mse").compute(magnitude, magnitude.clone())

    def test_real_complex_pair_is_refused_too(self) -> None:
        magnitude = torch.rand(2, 1, 16, 16) + 0.5
        with pytest.raises(PhaseBlindInputError, match="target"):
            _computer("phase_mse").compute(
                magnitude, magnitude, complex_pair=(_phase_shifted_pair()[0], magnitude)
            )

    def test_only_subset_skips_the_guard_for_unselected_phase_metrics(self) -> None:
        magnitude = torch.rand(2, 1, 16, 16) + 0.5
        out = _computer("mae", "phase_mse").compute(magnitude, magnitude, only=["mae"])
        assert set(out) == {"mae"}
