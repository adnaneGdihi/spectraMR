"""Phase metrics see the complex image the metric transform passes through (#867).

Driven through ``_compute_training_metrics`` -- the production entry point every
strategy's training step calls -- with a real ``ValidationMetricsComputer``.

The planted case: a prediction that differs from its target ONLY by a global phase
of pi/3. Under ``ifft_sense_adjoint`` the transform took ``abs()`` before any
metric ran, so ``phase_mse`` compared two identical magnitude images and logged
exactly 0. On origin/dev ``test_phase_only_error_is_measured`` fails with
``train_phase_mse == 0.0``.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

import spectramr.core.metrics.evaluation_metrics  # noqa: F401  (registers the metrics)
from spectramr.core.metrics.complex_input import PhaseBlindInputError
from spectramr.core.metrics.computer import ValidationMetricsComputer
from spectramr.core.metrics.types import ValidationMetricsConfig
from spectramr.infrastructure.physics.fft_ops import fft2c, sense_adjoint
from spectramr.infrastructure.training.strategies.mixins.metrics_mixin import MetricsMixin

SHIFT = math.pi / 3
MAGNITUDE_METRICS = ["mae", "mse", "psnr", "ssim"]


class _Host(MetricsMixin):
    """Bare strategy carrying only what ``_compute_training_metrics`` reads."""

    def __init__(self, transform: str | None, metric_names: list[str], smaps=None) -> None:
        self.config = SimpleNamespace(
            metrics=SimpleNamespace(
                enable_tracking=True, train_metric_interval=1, transform=transform
            ),
            training=SimpleNamespace(max_iterations=10),
        )
        self._training_computer = ValidationMetricsComputer(
            ValidationMetricsConfig.from_dict({"metrics": metric_names})
        )
        self._current_smaps = smaps

    def train_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
        return self._compute_training_metrics(pred, target, self.config, current_step=1)


def _multicoil_phase_only_pair(coils: int = 4):
    """Multi-coil k-space whose coil-combined images differ only by a global phase."""
    torch.manual_seed(0)
    image = torch.complex(torch.rand(2, 1, 16, 16) + 0.5, torch.zeros(2, 1, 16, 16))
    phases = torch.rand(2, coils, 16, 16) * 2 * math.pi
    smaps = (torch.polar(torch.ones_like(phases), phases) / math.sqrt(coils)).to(torch.complex64)
    rotation = complex(math.cos(SHIFT), math.sin(SHIFT))
    target_k = fft2c(image * smaps)
    pred_k = fft2c(image * rotation * smaps)
    return pred_k, target_k, smaps


class TestIfftSenseAdjoint:
    def test_phase_only_error_is_measured(self) -> None:
        pred_k, target_k, smaps = _multicoil_phase_only_pair()
        out = _Host("ifft_sense_adjoint", ["phase_mse"], smaps).train_metrics(pred_k, target_k)
        assert out["train_phase_mse"] == pytest.approx(SHIFT**2, rel=1e-3)

    def test_magnitude_metrics_are_numerically_unchanged(self) -> None:
        pred_k, target_k, smaps = _multicoil_phase_only_pair()
        pred_k = pred_k * 0.9  # give the magnitude metrics something to measure
        with_phase = _Host(
            "ifft_sense_adjoint", [*MAGNITUDE_METRICS, "phase_mse"], smaps
        ).train_metrics(pred_k, target_k)

        # Reference: the pre-#867 recipe, |A^H y| then the magnitude-only computer.
        reference = ValidationMetricsComputer(
            ValidationMetricsConfig.from_dict({"metrics": MAGNITUDE_METRICS})
        ).compute(
            sense_adjoint(pred_k, smaps=smaps).abs(), sense_adjoint(target_k, smaps=smaps).abs()
        )
        for name in MAGNITUDE_METRICS:
            assert with_phase[f"train_{name}"] == reference[name], name

    def test_without_smaps_phase_is_scored_per_coil(self) -> None:
        pred_k, target_k, _ = _multicoil_phase_only_pair()
        out = _Host("ifft_sense_adjoint", ["phase_mse"], None).train_metrics(pred_k, target_k)
        assert out["train_phase_mse"] == pytest.approx(SHIFT**2, rel=1e-3)


class TestOtherTransforms:
    def test_ifft_magnitude_single_coil(self) -> None:
        pred_k, target_k, _ = _multicoil_phase_only_pair(coils=1)
        stack = lambda z: torch.cat([z.real, z.imag], dim=1)  # noqa: E731  [R, I] layout
        out = _Host("ifft_magnitude", ["phase_mse"]).train_metrics(stack(pred_k), stack(target_k))
        assert out["train_phase_mse"] == pytest.approx(SHIFT**2, rel=1e-3)

    def test_magnitude_two_channel_image(self) -> None:
        pred_k, target_k, _ = _multicoil_phase_only_pair(coils=1)
        from spectramr.infrastructure.physics.fft_ops import ifft2c

        stack = lambda z: torch.cat([z.real, z.imag], dim=1)  # noqa: E731
        out = _Host("magnitude", ["phase_mse"]).train_metrics(
            stack(ifft2c(pred_k)), stack(ifft2c(target_k))
        )
        assert out["train_phase_mse"] == pytest.approx(SHIFT**2, rel=1e-3)


class TestGuard:
    """No complex intermediate exists: the phase metric must raise, not report."""

    def test_real_one_channel_without_transform_raises(self) -> None:
        image = torch.rand(2, 1, 16, 16) + 0.5
        with pytest.raises(PhaseBlindInputError, match="phase_mse"):
            _Host(None, ["mae", "phase_mse"]).train_metrics(image, image.clone())

    def test_magnitude_transform_against_a_magnitude_reference_raises(self) -> None:
        # 2-ch real/imag prediction but a 1-ch magnitude target: no target phase.
        pred = torch.randn(2, 2, 16, 16)
        target = torch.rand(2, 1, 16, 16) + 0.5
        with pytest.raises(PhaseBlindInputError, match="phase_mse"):
            _Host("magnitude", ["phase_mse"]).train_metrics(pred, target)

    def test_metric_set_without_phase_metrics_is_untouched(self) -> None:
        image = torch.rand(2, 1, 16, 16) + 0.5
        out = _Host(None, ["mae"]).train_metrics(image, image.clone())
        assert out == {"train_mae": pytest.approx(0.0, abs=1e-7)}
