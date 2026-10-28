"""``sense_psnr`` / ``sense_ssim`` and the combine they share (n2n PR 10).

The phantom is the rank-one coil model the SENSE combine inverts: a disk of
anatomy ``m`` seen through unit-norm complex maps that vanish outside the disk.
Every planted case states what the RSS image does on the same data, because the
reason for these columns is that the two combinations disagree.
"""

from __future__ import annotations

import math

import pytest
import torch

import spectramr.core.metrics.sense_image_metrics  # noqa: F401  (registers the metrics)
from spectramr.core.metrics.coil_manifold import as_interleaved, project_onto_manifold
from spectramr.core.metrics.computer import ValidationMetricsComputer
from spectramr.core.metrics.context import MetricContext
from spectramr.core.metrics.registry import MetricsRegistry
from spectramr.core.metrics.sense_image_metrics import (
    SenseMapsMissingError,
    sense_scoring_pair,
)
from spectramr.core.metrics.types import ValidationMetricsConfig
from spectramr.infrastructure.physics.fft_ops import fft2c
from spectramr.infrastructure.training.utils.metric_transform import (
    image_magnitude,
    kspace_rss_magnitude,
)

_N = 32
_C = 4


def _phantom(dtype: torch.dtype = torch.complex128):
    """``(m, maps, support)``: anatomy on a disk, unit-norm maps zero off it."""
    yy, xx = torch.meshgrid(torch.arange(_N), torch.arange(_N), indexing="ij")
    support = ((yy - _N / 2) ** 2 + (xx - _N / 2) ** 2) < (0.35 * _N) ** 2
    m = (0.5 + 0.4 * torch.cos(xx / 5.0)) * support
    phase = torch.stack([(c + 1) * (0.1 * xx + 0.05 * yy) for c in range(_C)])
    gain = torch.stack([1.0 + 0.5 * torch.sin((c + 1) * yy / 7.0) for c in range(_C)])
    maps = torch.polar(gain.double(), phase.double()).to(dtype)
    maps = maps / maps.abs().pow(2).sum(0, keepdim=True).sqrt()
    maps = maps * support
    return m.to(dtype)[None, None], maps[None], support


def _noise(seed: int, sigma: float) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    real = torch.randn(1, _C, _N, _N, generator=gen, dtype=torch.float64)
    imag = torch.randn(1, _C, _N, _N, generator=gen, dtype=torch.float64)
    return sigma * torch.complex(real, imag) / math.sqrt(2.0)


def _ctx(maps: torch.Tensor) -> MetricContext:
    return MetricContext(coil_maps=maps, signal_domain="kspace")


class TestTheTargetIsTheSenseCombination:
    def test_noise_free_it_is_m(self) -> None:
        m, maps, _ = _phantom()
        target_k = fft2c(maps * m)
        _, target = sense_scoring_pair(target_k, target_k, maps)
        torch.testing.assert_close(target, m, atol=1e-6, rtol=0)

    def test_noisy_its_background_is_zero_and_rss_keeps_a_floor(self) -> None:
        """RSS of a noisy reference keeps a chi floor off the support; SENSE does not."""
        m, maps, support = _phantom()
        target_k = fft2c(maps * m + _noise(seed=0, sigma=0.05))
        _, sense = sense_scoring_pair(target_k, target_k, maps)
        rss = kspace_rss_magnitude(target_k)
        background = ~support
        assert sense.abs()[0, 0][background].max() == 0.0
        assert rss[0, 0][background].mean() > 0.05  # sqrt(C) * sigma ~ 0.1

    def test_noisy_it_sits_closer_to_m_inside_the_support_than_rss(self) -> None:
        """One coil-equivalent noise (Rician) against C of them (chi_2C)."""
        m, maps, support = _phantom()
        target_k = fft2c(maps * m + _noise(seed=1, sigma=0.05))
        _, sense = sense_scoring_pair(target_k, target_k, maps)
        rss = kspace_rss_magnitude(target_k)
        truth = m.abs()[0, 0][support]
        sense_bias = (sense.abs()[0, 0][support] - truth).mean()
        rss_bias = (rss[0, 0][support] - truth).mean()
        assert abs(sense_bias) < rss_bias


class TestTheUnreachableReference:
    """#2509: an ``S x_hat`` output scored against the RSS of a reference that
    carries energy off the coil manifold. Planted with that energy in the null
    space only, so the SENSE reference is exactly ``m``."""

    def _reference(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        m, maps, _ = _phantom()
        noise = _noise(seed=2, sigma=0.05)
        off_manifold = noise - project_onto_manifold(noise, maps)
        return m, maps, fft2c(maps * m + off_manifold)

    def test_a_perfect_image_output_reaches_the_sense_reference(self) -> None:
        m, maps, target_k = self._reference()
        prediction = as_interleaved(m)  # EI/SSDU: one complex image
        psnr = MetricsRegistry.get("sense_psnr")(
            prediction, target_k, context=_ctx(maps), data_range=1.0
        )
        assert psnr.item() > 120.0

    def test_the_same_output_cannot_reach_the_rss_reference(self) -> None:
        m, _, target_k = self._reference()
        rss_psnr = MetricsRegistry.get("psnr")(
            image_magnitude(as_interleaved(m)), kspace_rss_magnitude(target_k), data_range=1.0
        )
        assert rss_psnr.item() < 40.0

    def test_a_kspace_output_is_combined_like_the_target(self) -> None:
        m, maps, target_k = (t.to(torch.complex64) for t in self._reference())
        ssim = MetricsRegistry.get("sense_ssim")(
            fft2c(maps * m), target_k, context=_ctx(maps), data_range=1.0
        )
        assert ssim.item() == pytest.approx(1.0, abs=1e-4)


class TestItRaisesInsteadOfFallingBack:
    def test_no_maps(self) -> None:
        m, maps, _ = _phantom()
        target_k = fft2c(maps * m)
        with pytest.raises(SenseMapsMissingError, match="sense_psnr"):
            MetricsRegistry.get("sense_psnr")(target_k, target_k, context=MetricContext())

    def test_magnitude_maps(self) -> None:
        m, maps, _ = _phantom()
        with pytest.raises(ValueError, match="COMPLEX"):
            sense_scoring_pair(fft2c(maps * m), fft2c(maps * m), maps.abs())

    def test_one_coil(self) -> None:
        m, maps, _ = _phantom()
        one = fft2c(maps[:, :1] * m)
        with pytest.raises(ValueError, match="one coil"):
            sense_scoring_pair(one, one, maps[:, :1])

    def test_a_prediction_with_the_wrong_coil_count(self) -> None:
        m, maps, _ = _phantom()
        target_k = fft2c(maps * m)
        with pytest.raises(ValueError, match="prediction has 2 coils"):
            sense_scoring_pair(target_k[:, :2], target_k, maps)


class TestTheComputerRoutesIt:
    def test_it_is_a_coil_map_context_metric(self) -> None:
        for name in ("sense_psnr", "sense_ssim"):
            assert MetricsRegistry.needs(name) == ("coil_maps",)
            assert MetricsRegistry.needs_context(name)
            assert MetricsRegistry.requires_reference(name)

    def test_a_declared_data_range_reaches_it(self) -> None:
        """Planted: a signature without ``data_range`` makes the computer report N/A."""
        m, maps, _ = _phantom(torch.complex64)
        target_k = fft2c(maps * m + _noise(seed=3, sigma=0.02).to(torch.complex64))
        prediction = fft2c(maps * m)
        computer = ValidationMetricsComputer(
            ValidationMetricsConfig.from_dict({"metrics": ["sense_psnr", "sense_ssim"]}),
            data_range=1.0,
        )
        out = computer.compute(prediction, target_k, context=_ctx(maps))
        assert not computer.last_not_applicable
        assert all(math.isfinite(value) for value in out.values()), out
        assert out["sense_psnr"] > 20.0
