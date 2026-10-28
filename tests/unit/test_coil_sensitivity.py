"""Unit tests for CoilSensitivityEstimationService.

Tests cover correctness of estimation methods, cache behaviour,
and SNR estimation against synthetic Biot-Savart-style coil data.
"""


import pytest
import torch

from spectramr.infrastructure.services.coil_sensitivity_service import (
    CoilSensitivityEstimationService,
)

# CoilSensitivityEstimationService is deprecated (superseded by estimate_smaps).
# Ignore the (spectramr.*-promoted-to-error) DeprecationWarning module-wide during
# the deprecation window; one test asserts it fires explicitly.
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def test_service_construction_emits_deprecation_warning():
    with pytest.warns(DeprecationWarning, match="estimate_smaps"):
        CoilSensitivityEstimationService(logging_service=None, enable_cache=False)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cache_dir(tmp_path):
    """Temporary cache directory for tests."""
    d = tmp_path / "coil_cache"
    d.mkdir()
    return d


@pytest.fixture
def service(cache_dir):
    """Service instance with caching enabled but no logging."""
    return CoilSensitivityEstimationService(
        logging_service=None,
        cache_dir=str(cache_dir),
        enable_cache=True,
    )


@pytest.fixture
def synthetic_multicoil_kspace():
    """Synthetic multi-coil k-space with known Biot-Savart-like sensitivities.

    Creates 4 coils with smooth sinusoidal sensitivity profiles, applies them
    to a simple test image, and transforms to k-space.
    """
    height, width = 64, 64
    n_coils = 4

    # Known sensitivity profiles (smooth sinusoids)
    y = torch.linspace(0, torch.pi, height)
    x = torch.linspace(0, torch.pi, width)
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    true_sens = torch.stack([
        torch.cos(yy) * torch.cos(xx),
        torch.sin(yy) * torch.cos(xx),
        torch.cos(yy) * torch.sin(xx),
        torch.sin(yy) * torch.sin(xx),
    ])  # [4, 64, 64]

    # Normalize per-pixel to unit RSS
    rss = torch.sqrt(torch.sum(true_sens ** 2, dim=0).clamp(min=1e-12))
    true_sens = true_sens / rss.unsqueeze(0)

    # Simple test phantom (disc)
    phantom = torch.zeros(height, width)
    center = height // 2
    for i in range(height):
        for j in range(width):
            if (i - center) ** 2 + (j - center) ** 2 < (height // 4) ** 2:
                phantom[i, j] = 1.0

    # Coil images = sensitivity * phantom
    coil_images = true_sens * phantom.unsqueeze(0)  # [4, 64, 64]

    # Convert to complex and FFT to k-space
    coil_images_complex = coil_images.to(torch.complex64)
    kspace = torch.fft.fftshift(
        torch.fft.fft2(
            torch.fft.ifftshift(coil_images_complex, dim=(-2, -1)),
            norm="ortho",
        ),
        dim=(-2, -1),
    )

    return kspace, true_sens


# ---------------------------------------------------------------------------
# Shape & normalization tests
# ---------------------------------------------------------------------------


class TestOutputShape:
    """Test that all methods produce correctly shaped, normalized maps."""

    def test_fallback_shape(self, service, synthetic_multicoil_kspace):
        kspace, _ = synthetic_multicoil_kspace
        result = service.estimate_sensitivity_maps(kspace, method="fallback")
        assert result.sensitivity_maps.shape == kspace.shape
        assert result.method == "fallback"

    def test_low_rank_shape(self, service, synthetic_multicoil_kspace):
        kspace, _ = synthetic_multicoil_kspace
        result = service.estimate_sensitivity_maps(kspace, method="low_rank")
        assert result.sensitivity_maps.shape == kspace.shape
        assert result.method == "low_rank"

    def test_unit_norm_per_pixel_fallback(self, service, synthetic_multicoil_kspace):
        """Sensitivity maps should have unit RSS norm at every pixel."""
        kspace, _ = synthetic_multicoil_kspace
        result = service.estimate_sensitivity_maps(kspace, method="fallback")
        norms = torch.sqrt(torch.sum(torch.abs(result.sensitivity_maps) ** 2, dim=0))
        # All non-zero pixels should have norm ≈ 1
        non_zero = norms > 0.01
        assert torch.allclose(norms[non_zero], torch.ones_like(norms[non_zero]), atol=1e-5)

    def test_unit_norm_per_pixel_low_rank(self, service, synthetic_multicoil_kspace):
        """Low-rank maps should also be unit-norm (after internal normalization)."""
        kspace, _ = synthetic_multicoil_kspace
        result = service.estimate_sensitivity_maps(kspace, method="low_rank")
        norms = torch.sqrt(torch.sum(torch.abs(result.sensitivity_maps) ** 2, dim=0))
        non_zero = norms > 0.01
        assert torch.allclose(norms[non_zero], torch.ones_like(norms[non_zero]), atol=1e-5)


# ---------------------------------------------------------------------------
# Correctness test
# ---------------------------------------------------------------------------


class TestLowRankCorrectness:
    """Verify low-rank estimation recovers known sensitivity profiles."""

    def test_recovery_correlation(self, service, synthetic_multicoil_kspace):
        """Estimated maps should produce similar spatial sensitivity profiles.

        We compare magnitude profiles because estimated maps may have
        arbitrary per-coil phase offsets vs ground truth.
        """
        kspace, true_sens = synthetic_multicoil_kspace
        result = service.estimate_sensitivity_maps(kspace, method="low_rank")
        estimated = result.sensitivity_maps

        # Strategy: compare RSS-weighted image reconstructions.
        # If sensitivity maps are correct, coil-combined images should match.
        # This is more robust than per-coil correlation.
        coil_images = torch.fft.fftshift(
            torch.fft.ifft2(
                torch.fft.ifftshift(kspace, dim=(-2, -1)),
                norm="ortho",
            ),
            dim=(-2, -1),
        )

        # Ground-truth SENSE-combined image
        gt_combined = torch.sum(coil_images * true_sens.conj(), dim=0).abs()
        # Estimated SENSE-combined image
        est_combined = torch.sum(coil_images * estimated.conj(), dim=0).abs()

        # Normalize both
        gt_flat = gt_combined.flatten().float()
        est_flat = est_combined.flatten().float()
        gt_norm = gt_flat - gt_flat.mean()
        est_norm = est_flat - est_flat.mean()

        corr = (gt_norm * est_norm).sum() / (
            gt_norm.norm() * est_norm.norm() + 1e-12
        )
        assert corr.item() > 0.5, (
            f"SENSE-combined image correlation {corr.item():.3f} too low — "
            f"estimated sensitivity maps do not recover the image."
        )


# ---------------------------------------------------------------------------
# SNR estimation
# ---------------------------------------------------------------------------


class TestSNREstimation:
    """Test that SNR estimation returns reasonable values."""

    def test_snr_returns_float(self, service, synthetic_multicoil_kspace):
        kspace, _ = synthetic_multicoil_kspace
        result = service.estimate_sensitivity_maps(kspace, method="fallback")
        assert result.snr_estimate is None or isinstance(result.snr_estimate, float)

    def test_snr_positive_for_signal(self, service, synthetic_multicoil_kspace):
        """Non-zero signal should produce positive SNR."""
        kspace, _ = synthetic_multicoil_kspace
        result = service.estimate_sensitivity_maps(kspace, method="fallback")
        if result.snr_estimate is not None:
            assert result.snr_estimate > 0


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


class TestCaching:
    """Test cache save/load round-trip."""

    def test_cache_hit(self, service, synthetic_multicoil_kspace, cache_dir):
        kspace, _ = synthetic_multicoil_kspace

        # First call — should compute
        result1 = service.estimate_sensitivity_maps(kspace, method="fallback")
        assert not result1.cached

        # Second call — should hit cache
        result2 = service.estimate_sensitivity_maps(kspace, method="fallback")
        assert result2.cached
        assert torch.allclose(
            result1.sensitivity_maps, result2.sensitivity_maps, atol=1e-6
        )

    def test_different_params_no_collision(self, service, synthetic_multicoil_kspace):
        """Different calibration_size should produce different cache keys."""
        kspace, _ = synthetic_multicoil_kspace
        r1 = service.estimate_sensitivity_maps(kspace, method="fallback", calibration_size=16)
        r2 = service.estimate_sensitivity_maps(kspace, method="fallback", calibration_size=32)
        # Both should compute (not cached from each other)
        assert not r2.cached or not torch.allclose(
            r1.sensitivity_maps, r2.sensitivity_maps
        )


    def test_cache_write_is_observed_on_disk(
        self, cache_dir, synthetic_multicoil_kspace
    ):
        """A miss writes ``<key>.pt``; the next call reads it back and nothing warns."""
        from unittest.mock import MagicMock

        logging_service = MagicMock()
        svc = CoilSensitivityEstimationService(
            logging_service=logging_service,
            cache_dir=str(cache_dir),
            enable_cache=True,
        )
        kspace, _ = synthetic_multicoil_kspace

        first = svc.estimate_sensitivity_maps(kspace, method="fallback")
        written = list(cache_dir.glob("*.pt"))
        second = svc.estimate_sensitivity_maps(kspace, method="fallback")

        assert not first.cached
        assert len(written) == 1
        assert not list(cache_dir.glob("*.tmp"))
        assert second.cached
        logging_service.log_warning.assert_not_called()

    def test_programming_error_in_cache_write_raises(
        self, service, synthetic_multicoil_kspace, monkeypatch
    ):
        """Only an I/O failure is downgraded to a warning (#2860).

        The cache used to call a method no checkpoint service implements, and a
        blanket ``except Exception`` turned the AttributeError into a warning.
        """

        def _broken_save(*_args, **_kwargs):
            raise AttributeError("save_payload_to_path")

        monkeypatch.setattr(torch, "save", _broken_save)
        kspace, _ = synthetic_multicoil_kspace
        with pytest.raises(AttributeError, match="save_payload_to_path"):
            service.estimate_sensitivity_maps(kspace, method="fallback")

    def test_disk_failure_in_cache_write_warns(
        self, cache_dir, synthetic_multicoil_kspace, monkeypatch
    ):
        """A full disk must not abort estimation; the result is still returned."""
        from unittest.mock import MagicMock

        logging_service = MagicMock()
        svc = CoilSensitivityEstimationService(
            logging_service=logging_service,
            cache_dir=str(cache_dir),
            enable_cache=True,
        )

        # The message torch.save raises when writing to /dev/full.
        def _full_disk(*_args, **_kwargs):
            raise RuntimeError("basic_ios::clear: iostream error")

        monkeypatch.setattr(torch, "save", _full_disk)
        kspace, _ = synthetic_multicoil_kspace
        result = svc.estimate_sensitivity_maps(kspace, method="fallback")

        assert result.sensitivity_maps.shape[0] == kspace.shape[0]
        logging_service.log_warning.assert_called_once()
        assert "Failed to save cache" in logging_service.log_warning.call_args.args[0]

    def test_checkpoint_service_is_not_a_constructor_argument(self, cache_dir):
        """The cache owns its own write; no checkpoint service is accepted."""
        with pytest.raises(TypeError, match="checkpoint_service"):
            CoilSensitivityEstimationService(
                logging_service=None,
                cache_dir=str(cache_dir),
                checkpoint_service=object(),
            )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and robustness."""

    def test_single_coil_input(self, service):
        """Single-coil (2D) input should work via unsqueeze."""
        kspace_2d = torch.randn(64, 64, dtype=torch.complex64)
        result = service.estimate_sensitivity_maps(kspace_2d, method="fallback")
        assert result.sensitivity_maps.shape == (1, 64, 64)

    def test_auto_method_selection_high_coil(self, service):
        """8+ coils should auto-select low_rank."""
        kspace = torch.randn(10, 32, 32, dtype=torch.complex64)
        result = service.estimate_sensitivity_maps(kspace, method="auto")
        assert result.method == "low_rank"

    def test_auto_method_selection_low_coil(self, service):
        """<4 coils should auto-select fallback."""
        kspace = torch.randn(2, 32, 32, dtype=torch.complex64)
        result = service.estimate_sensitivity_maps(kspace, method="auto")
        assert result.method == "fallback"
