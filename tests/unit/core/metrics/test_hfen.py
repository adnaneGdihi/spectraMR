"""Unit tests for HFEN (High-Frequency Error Norm) metric.

Tests the LoG-filtered edge-preservation metric that proves clinical
edge fidelity beyond what PSNR/SSIM can measure.
"""

import pytest
import torch

from spectramr.core.metrics.hfen import HFENMetric, _log_kernel
from spectramr.core.metrics.registry import compute_metric, get_metric


class TestLoGKernel:
    """Tests for the Laplacian-of-Gaussian kernel builder."""

    def test_output_shape(self):
        """Kernel should be (1, 1, size, size)."""
        k = _log_kernel(size=15, sigma=1.5)
        assert k.shape == (1, 1, 15, 15)

    def test_zero_mean(self):
        """LoG kernel should be zero-mean after normalisation."""
        k = _log_kernel(size=15, sigma=1.5)
        assert k.mean().abs() < 1e-5

    def test_unit_energy(self):
        """Kernel should have approximately unit norm."""
        k = _log_kernel(size=15, sigma=1.5)
        assert abs(k.norm().item() - 1.0) < 1e-4


class TestHFENMetric:
    """Tests for the HFEN metric."""

    def test_identical_inputs_zero_hfen(self):
        """HFEN should be zero for identical images."""
        hfen = HFENMetric()
        x = torch.randn(1, 1, 64, 64)
        score = hfen(x, x)
        assert score == pytest.approx(0.0, abs=1e-5)

    def test_blurred_image_higher_hfen(self):
        """Blurred prediction should have higher HFEN than sharp."""
        hfen = HFENMetric()

        # Sharp target with edges
        target = torch.zeros(1, 1, 64, 64)
        target[:, :, 20:44, 20:44] = 1.0  # Square with sharp edges

        # Sharp prediction (close to target)
        pred_sharp = target.clone()
        pred_sharp[:, :, 20:44, 20:44] += torch.randn(1, 1, 24, 24) * 0.01

        # Blurred prediction (smoothed edges → worse)
        pred_blurred = torch.nn.functional.avg_pool2d(
            target, kernel_size=5, stride=1, padding=2,
        )

        hfen_sharp = hfen(pred_sharp, target)
        hfen_blurred = hfen(pred_blurred, target)

        assert hfen_blurred > hfen_sharp

    def test_lower_is_better(self):
        """higher_is_better should be False."""
        hfen = HFENMetric()
        assert hfen.higher_is_better is False

    def test_name_property(self):
        """Name should be 'hfen'."""
        hfen = HFENMetric()
        assert hfen.name == "hfen"

    def test_complex_input(self):
        """Complex inputs should be handled via magnitude."""
        hfen = HFENMetric()
        pred = torch.complex(torch.randn(1, 1, 32, 32), torch.randn(1, 1, 32, 32))
        target = torch.complex(torch.randn(1, 1, 32, 32), torch.randn(1, 1, 32, 32))
        score = hfen(pred, target)
        assert isinstance(score, torch.Tensor) and score.dim() == 0
        assert score >= 0

    def test_registry_resolution(self):
        """HFEN should be resolvable via get_metric."""
        metric = get_metric("hfen")
        assert isinstance(metric, HFENMetric)

    def test_compute_metric_convenience(self):
        """compute_metric("hfen", ...) should work."""
        x = torch.randn(1, 1, 32, 32)
        score = compute_metric("hfen", x, x)
        assert score == pytest.approx(0.0, abs=1e-5)

    def test_registry_alias_resolution(self):
        """HFEN aliases should resolve."""
        for alias in ["HFEN", "HighFrequencyErrorNorm"]:
            metric = get_metric(alias)
            assert isinstance(metric, HFENMetric)

    def test_multichannel_input(self):
        """Multi-channel inputs should work."""
        hfen = HFENMetric()
        pred = torch.randn(2, 3, 32, 32)
        target = torch.randn(2, 3, 32, 32)
        score = hfen(pred, target)
        assert isinstance(score, torch.Tensor) and score.dim() == 0
        assert score >= 0
