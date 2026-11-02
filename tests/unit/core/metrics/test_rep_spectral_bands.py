"""Rep-BV by band on synthetic repetitions with a known anatomy and noise level.

The anatomy is white inside a disk, so every band carries signal, and the
reference noise is set comparable to the high band's signal, where the noise
correction decides the reading. The identity must keep every band (transfer 1)
at two noise levels and under noise confined to acquired columns; a Gaussian
low-pass must read back its own transfer; the bands must sum to the Rep-BV totals.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.core.metrics.coil_manifold import as_interleaved
from spectramr.core.metrics.context import MetricContext
from spectramr.core.metrics.metric_directions import metric_higher_is_better
from spectramr.core.metrics.registry import MetricsRegistry
from spectramr.core.metrics.rep_spectral_bands import BAND_NAMES, band_masks
from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c

_C, _N, _B = 4, 64, 8


def _randn(*shape: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.complex(torch.randn(*shape, generator=gen), torch.randn(*shape, generator=gen))


def _maps() -> torch.Tensor:
    """Unit-norm smooth maps on a disk, zero outside it (ESPIRiT's support)."""
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, _N), torch.linspace(-1, 1, _N), indexing="ij")
    centres = [(-0.7, -0.7), (0.7, -0.7), (-0.7, 0.7), (0.7, 0.7)]
    maps = torch.stack(
        [
            torch.exp(-((xx - a) ** 2 + (yy - b) ** 2)) * torch.exp(0.5j * (a * xx + b * yy))
            for a, b in centres
        ]
    ).to(torch.complex64)
    maps = maps / maps.abs().pow(2).sum(0, keepdim=True).sqrt()
    maps = maps * (xx.square() + yy.square() < 0.85).to(maps.dtype)
    return maps.unsqueeze(0).expand(_B, -1, -1, -1)


def _acquired_columns(fraction: float) -> torch.Tensor:
    """Centred phase-encode band, as M4Raw's 195 of 256 columns."""
    keep = round(_N * fraction)
    mask = torch.zeros(_N)
    mask[(_N - keep) // 2 : (_N - keep) // 2 + keep] = 1.0
    return mask.to(torch.complex64)


class _Scene:
    """``x = s m`` with ``m`` = 1 + white detail, and i.i.d. repetitions ``x + n_k``."""

    def __init__(self, sigma: float, acquired: float = 1.0) -> None:
        self.maps = _maps()
        self.columns = _acquired_columns(acquired)
        self.x = self._limit(self.maps * (1.0 + 0.7 * _randn(_B, 1, _N, _N, seed=0)))
        # CN(0, sigma^2) per coil, confined to the acquired columns like the anatomy.
        self.reps = [
            self.x + self._limit(sigma / 2**0.5 * _randn(_B, _C, _N, _N, seed=10 + k))
            for k in range(8)
        ]
        self.reference = torch.stack(self.reps[2:6]).mean(0)

    def _limit(self, images: torch.Tensor) -> torch.Tensor:
        return ifft2c(fft2c(images) * self.columns)

    @staticmethod
    def k(images: torch.Tensor) -> torch.Tensor:
        return as_interleaved(fft2c(images))

    def true_transfer(self, prediction: torch.Tensor) -> torch.Tensor:
        """Per band ``Re<f, x> / ||x||^2`` with the noise-free anatomy."""
        xk, fk = fft2c(self.x), fft2c(prediction)
        masks = band_masks(_N, _N, torch.device("cpu"), torch.float32)
        cross = torch.einsum("bhw,khw->k", (fk * xk.conj()).real.sum(1), masks)
        power = torch.einsum("bhw,khw->k", xk.abs().square().sum(1), masks)
        return cross / power


def _metric(name: str, prediction, target=None, **context) -> float:
    return MetricsRegistry.get(name)(prediction, target, context=MetricContext(**context)).item()


def _transfer(scene: _Scene, prediction: torch.Tensor) -> list[float]:
    target, maps = scene.k(scene.reference), scene.maps
    out = []
    for band in BAND_NAMES:
        cross = _metric(f"band_cross_power_{band}", scene.k(prediction), target)
        deficit = _metric(
            f"band_signal_deficit_{band}", scene.k(prediction), target, coil_maps=maps
        )
        out.append(cross / (cross + deficit))
    return out


class TestRegistration:
    @pytest.mark.parametrize("band", BAND_NAMES)
    @pytest.mark.parametrize(
        ("family", "needs", "higher"),
        [
            ("corrected_mse", ("coil_maps",), False),
            ("bias_squared", ("coil_maps", "pred_alt"), False),
            ("cross_power", (), True),
            ("signal_deficit", ("coil_maps",), False),
            ("input_cross_power", ("y_kspace",), True),
        ],
    )
    def test_declared_contract(self, family, needs, higher, band) -> None:
        name = f"band_{family}_{band}"
        assert MetricsRegistry.needs(name) == needs
        assert MetricsRegistry.needs_context(name) is True
        assert MetricsRegistry.requires_reference(name) is True
        assert metric_higher_is_better(name) is higher

    def test_the_bands_partition_the_grid(self) -> None:
        masks = band_masks(_N, _N + 6, torch.device("cpu"), torch.float32)
        assert torch.equal(masks.sum(0), torch.ones(_N, _N + 6))


class TestTheBandsSumToTheTotals:
    def test_corrected_mse(self) -> None:
        scene = _Scene(sigma=1.0)
        args = (scene.k(scene.reps[0]), scene.k(scene.reference))
        total = _metric("reference_noise_corrected_mse", *args, coil_maps=scene.maps)
        bands = sum(
            _metric(f"band_corrected_mse_{b}", *args, coil_maps=scene.maps) for b in BAND_NAMES
        )
        assert bands == pytest.approx(total, rel=1e-4)

    def test_bias_squared(self) -> None:
        scene = _Scene(sigma=1.0)
        args = (scene.k(scene.reps[0]), scene.k(scene.reference))
        context = {"coil_maps": scene.maps, "pred_alt": scene.k(scene.reps[1])}
        total = _metric("bias_squared", *args, **context)
        bands = sum(_metric(f"band_bias_squared_{b}", *args, **context) for b in BAND_NAMES)
        assert bands == pytest.approx(total, rel=1e-3, abs=1e-5)


class TestTheIdentityKeepsEveryBand:
    @pytest.mark.parametrize("sigma", [0.5, 1.0])
    def test_transfer_is_one_at_any_noise_level(self, sigma) -> None:
        scene = _Scene(sigma=sigma)
        assert _transfer(scene, scene.reps[0]) == pytest.approx([1.0, 1.0, 1.0], abs=0.08)

    def test_transfer_is_one_when_noise_fills_only_the_acquired_columns(self) -> None:
        scene = _Scene(sigma=1.0, acquired=195 / 256)
        assert _transfer(scene, scene.reps[0]) == pytest.approx([1.0, 1.0, 1.0], abs=0.08)

    def test_the_uncorrected_reference_power_would_read_blur(self) -> None:
        """Planted: dropping the noise term reads the identity as losing its high band."""
        scene = _Scene(sigma=1.0)
        target = scene.k(scene.reference)
        cross = _metric("band_cross_power_high", scene.k(scene.reps[0]), target)
        raw_power = (
            fft2c(scene.reference).abs().square().sum(1)
            * band_masks(_N, _N, torch.device("cpu"), torch.float32)[2]
        ).sum().item() / (_B * _N * _N)
        assert cross / raw_power < 0.8


class TestALowPassReadsBackItsTransfer:
    def test_gaussian_low_pass(self) -> None:
        scene = _Scene(sigma=1.0)
        yy, xx = torch.meshgrid(
            (torch.arange(_N) - _N // 2) / (_N / 2),
            (torch.arange(_N) - _N // 2) / (_N / 2),
            indexing="ij",
        )
        gain = torch.exp(-(yy.square() + xx.square()) / (2 * 0.3**2)).to(torch.complex64)
        blurred = ifft2c(fft2c(scene.reps[0]) * gain)
        expected = scene.true_transfer(blurred).tolist()
        assert expected[2] < 0.5 < expected[0]  # the filter does remove the high band
        assert _transfer(scene, blurred) == pytest.approx(expected, abs=0.05)

    def test_relative_to_the_input_without_a_noise_model(self) -> None:
        """cross / input cross needs no maps, and reads the same transfer."""
        scene = _Scene(sigma=1.0)
        gain = torch.exp(-torch.linspace(-1, 1, _N).square() / (2 * 0.3**2)).to(torch.complex64)
        blurred = ifft2c(fft2c(scene.reps[0]) * gain)  # a low-pass along readout only
        target, y = scene.k(scene.reference), fft2c(scene.reps[0])
        relative = [
            _metric(f"band_cross_power_{b}", scene.k(blurred), target)
            / _metric(f"band_input_cross_power_{b}", scene.k(blurred), target, y_kspace=y)
            for b in BAND_NAMES
        ]
        assert relative == pytest.approx(scene.true_transfer(blurred).tolist(), abs=0.05)

    def test_input_cross_power_agrees_with_the_corrected_reference_power(self) -> None:
        """The two signal estimates (no noise model, null-space noise) must coincide."""
        scene = _Scene(sigma=1.0, acquired=195 / 256)
        y, target = scene.k(scene.reps[0]), scene.k(scene.reference)
        for band in BAND_NAMES:
            signal = _metric(f"band_cross_power_{band}", y, target) + _metric(
                f"band_signal_deficit_{band}", y, target, coil_maps=scene.maps
            )
            free = _metric(
                f"band_input_cross_power_{band}", y, target, y_kspace=fft2c(scene.reps[0])
            )
            assert free == pytest.approx(signal, rel=0.08), band


class TestRaises:
    def test_missing_maps(self) -> None:
        scene = _Scene(sigma=1.0)
        with pytest.raises(ValueError, match="coil sensitivity maps"):
            _metric("band_signal_deficit_high", scene.k(scene.reps[0]), scene.k(scene.reference))

    def test_missing_second_repetition(self) -> None:
        scene = _Scene(sigma=1.0)
        with pytest.raises(ValueError, match="pred_alt"):
            _metric(
                "band_bias_squared_low",
                scene.k(scene.reps[0]),
                scene.k(scene.reference),
                coil_maps=scene.maps,
            )

    def test_missing_input(self) -> None:
        scene = _Scene(sigma=1.0)
        with pytest.raises(ValueError, match="y_kspace"):
            _metric("band_input_cross_power_low", scene.k(scene.reps[0]), scene.k(scene.reference))

    def test_an_image_domain_caller(self) -> None:
        scene = _Scene(sigma=1.0)
        with pytest.raises(ValueError, match="k-space"):
            _metric(
                "band_cross_power_mid",
                scene.k(scene.reps[0]),
                scene.k(scene.reference),
                signal_domain="image",
            )
