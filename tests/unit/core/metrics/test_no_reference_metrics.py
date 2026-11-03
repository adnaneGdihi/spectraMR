"""Unit tests for no-reference and physics-based metrics."""

from __future__ import annotations

import math

import pytest
import torch

pytestmark = pytest.mark.unit

from spectramr.core.metrics.no_reference_metrics import (
    BRISQUE,
    NIQE,
    LaplacianVariance,
    TenengradVariance,
)
from spectramr.core.metrics.physics_nr_metrics import NEMASNR, GFactor
from spectramr.core.metrics.registry import MetricsRegistry


class TestTenengradVariance:
    """Tests for TenengradVariance metric."""

    def test_smoke(self) -> None:
        """Metric runs on a random tensor without error."""
        metric = TenengradVariance()
        img = torch.randn(1, 1, 64, 64)
        result = metric(img)
        assert isinstance(result, float)
        assert result >= 0.0

    def test_higher_for_sharp(self) -> None:
        """Sharp images should have higher Tenengrad than blurred."""
        metric = TenengradVariance()
        # Sharp: high-frequency checkerboard
        sharp = torch.zeros(1, 1, 64, 64)
        sharp[0, 0, ::2, ::2] = 1.0
        sharp[0, 0, 1::2, 1::2] = 1.0

        # Blurred: constant
        blurred = torch.ones(1, 1, 64, 64) * 0.5

        assert metric(sharp) > metric(blurred)

    def test_registered(self) -> None:
        """Metric should be in the registry."""
        assert MetricsRegistry.is_registered("tenengrad_variance")
        assert MetricsRegistry.is_registered("tenengrad")

    def test_2d_input(self) -> None:
        """Handles 2D input (no batch/channel dims)."""
        metric = TenengradVariance()
        result = metric(torch.randn(64, 64))
        assert isinstance(result, float)

    def test_properties(self) -> None:
        """Protocol properties are correct."""
        m = TenengradVariance()
        assert m.name == "tenengrad_variance"
        assert m.higher_is_better is True


class TestLaplacianVariance:
    """Tests for LaplacianVariance metric."""

    def test_smoke(self) -> None:
        metric = LaplacianVariance()
        result = metric(torch.randn(1, 1, 64, 64))
        assert isinstance(result, float)
        assert result >= 0.0

    def test_higher_for_edges(self) -> None:
        """Images with edges should have higher Laplacian variance."""
        metric = LaplacianVariance()
        edges = torch.zeros(1, 1, 64, 64)
        edges[0, 0, 30:34, :] = 1.0  # Horizontal edge

        smooth = torch.ones(1, 1, 64, 64) * 0.5

        assert metric(edges) > metric(smooth)

    def test_registered(self) -> None:
        assert MetricsRegistry.is_registered("laplacian_variance")
        assert MetricsRegistry.is_registered("lap_var")


class TestNIQE:
    """Tests for NIQE metric."""

    def test_smoke(self) -> None:
        metric = NIQE(patch_size=32, stride=16)
        result = metric(torch.randn(1, 1, 128, 128).abs())
        assert isinstance(result, float)
        assert result >= 0.0

    def test_lower_is_better(self) -> None:
        m = NIQE()
        assert m.higher_is_better is False

    def test_registered(self) -> None:
        assert MetricsRegistry.is_registered("niqe")
        assert MetricsRegistry.is_registered("NIQE")

    def test_small_image(self) -> None:
        """Small images should not crash."""
        metric = NIQE(patch_size=32, stride=16)
        result = metric(torch.randn(16, 16).abs())
        assert isinstance(result, float)


class TestBRISQUE:
    """Tests for BRISQUE metric."""

    def test_smoke(self) -> None:
        metric = BRISQUE()
        result = metric(torch.randn(1, 1, 128, 128).abs())
        assert isinstance(result, float)
        assert result >= 0.0

    def test_lower_is_better(self) -> None:
        m = BRISQUE()
        assert m.higher_is_better is False

    def test_registered(self) -> None:
        assert MetricsRegistry.is_registered("brisque")

    def test_complex_input(self) -> None:
        """Handles 2-channel complex input."""
        metric = BRISQUE()
        result = metric(torch.randn(1, 2, 64, 64))
        assert isinstance(result, float)


class TestBRISQUEKurtosisLive:
    """Regression: BRISQUE's standardised-4th-moment (kurtosis) slot is live.

    Previously the kurtosis feature was guarded by ``hasattr(flat, 'kurtosis')``,
    which is never True for a ``torch.Tensor``, so the slot was a dead constant
    ``0.0``. The fix computes ``((fc / fstd) ** 4).mean()`` explicitly. These
    tests prove that the BRISQUE feature vector (and hence the score) now
    responds to the tail behaviour of the input.
    """

    @staticmethod
    def _near_gaussian() -> torch.Tensor:
        """A near-Gaussian image (MSCN kurtosis close to 3)."""
        torch.manual_seed(20260604)
        return torch.randn(1, 1, 128, 128)

    @staticmethod
    def _heavy_tailed() -> torch.Tensor:
        """A heavy-tailed / peaky image (large MSCN kurtosis).

        Mostly near-zero pixels with a sparse set of extreme spikes — this
        drives the standardised 4th moment far above the Gaussian value while
        keeping variance and mean comparable in magnitude.
        """
        torch.manual_seed(20260604)
        img = 0.01 * torch.randn(1, 1, 128, 128)
        # Sparse extreme spikes => heavy tails.
        spike = torch.rand(1, 1, 128, 128) > 0.97
        img = img + spike.float() * 50.0
        return img

    def test_feature_vector_differs_for_tail_behaviour(self) -> None:
        """The 36-D NSS feature vector differs between Gaussian vs heavy-tailed.

        If the kurtosis slot were still a constant 0.0 this difference would
        only reflect the other moments; the kurtosis-slot assertion below
        pins the regression to the formerly-dead slot.
        """
        metric = BRISQUE()
        f_gauss = metric._extract_nss_features(self._near_gaussian().float())
        f_heavy = metric._extract_nss_features(self._heavy_tailed().float())

        assert f_gauss.shape == f_heavy.shape
        assert not torch.allclose(f_gauss, f_heavy)

    def test_kurtosis_slot_is_not_dead_constant(self) -> None:
        """The kurtosis slot (index 2) is non-zero and tail-dependent.

        Feature layout per scale: var(0), mean(1), kurtosis(2), then paired
        products. Slot 2 is the scale-0 standardised 4th moment. A near-Gaussian
        MSCN gives a value near the Gaussian kurtosis (~3); a heavy-tailed image
        gives a markedly larger value. A dead constant 0.0 could produce neither.
        """
        metric = BRISQUE()
        kurt_gauss = metric._extract_nss_features(self._near_gaussian().float())[2]
        kurt_heavy = metric._extract_nss_features(self._heavy_tailed().float())[2]

        # No longer the dead constant 0.0.
        assert float(kurt_gauss) > 0.0
        assert float(kurt_heavy) > 0.0
        # Heavy tails => strictly larger standardised 4th moment.
        assert float(kurt_heavy) > float(kurt_gauss)
        # Gaussian slot should sit in a sane neighbourhood of the true value 3.
        assert 2.0 < float(kurt_gauss) < 5.0

    def test_score_differs_for_tail_behaviour(self) -> None:
        """The scalar BRISQUE score (feature norm) differs across tail regimes."""
        metric = BRISQUE()
        s_gauss = metric(self._near_gaussian())
        s_heavy = metric(self._heavy_tailed())
        assert isinstance(s_gauss, float)
        assert isinstance(s_heavy, float)
        assert abs(s_heavy - s_gauss) > 1e-6


class TestNEMASNR:
    """Tests for NEMA SNR metric."""

    def test_smoke(self) -> None:
        metric = NEMASNR()
        result = metric(torch.randn(1, 1, 64, 64).abs() + 0.5)
        assert isinstance(result, float)
        assert result >= 0.0

    def test_high_snr_for_clean(self) -> None:
        """Clean image with noise in background should measure SNR."""
        metric = NEMASNR()
        # Phantom: bright center, faint noise in corners
        clean = torch.zeros(64, 64)
        clean[16:48, 16:48] = 1.0  # Signal ROI
        clean[:5, :5] = torch.randn(5, 5).abs() * 0.01  # Background noise

        noisy = clean.clone()
        noisy[:5, :5] = torch.randn(5, 5).abs() * 0.5  # Much louder background
        noisy[:5, -5:] = torch.randn(5, 5).abs() * 0.5
        noisy[-5:, :5] = torch.randn(5, 5).abs() * 0.5
        noisy[-5:, -5:] = torch.randn(5, 5).abs() * 0.5

        assert metric(clean) > metric(noisy)

    def test_registered(self) -> None:
        assert MetricsRegistry.is_registered("nema_snr")
        assert MetricsRegistry.is_registered("NemaSNR")

    def test_properties(self) -> None:
        m = NEMASNR()
        assert m.name == "nema_snr"
        assert m.higher_is_better is True


class TestGFactor:
    """Tests for G-Factor metric."""

    def test_smoke_no_smaps(self) -> None:
        """Without sensitivity maps the g-factor is unmeasured: NaN, not the ideal 1.0."""
        metric = GFactor()
        result = metric(torch.randn(1, 1, 64, 64))
        assert math.isnan(result)

    def test_registered(self) -> None:
        assert MetricsRegistry.is_registered("g_factor")
        assert MetricsRegistry.is_registered("GFactor")

    def test_properties(self) -> None:
        m = GFactor()
        assert m.name == "g_factor"
        assert m.higher_is_better is False
