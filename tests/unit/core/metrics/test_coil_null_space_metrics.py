"""NuCert: the null-space residual certifies error without a reference, and without a sync.

The barrier loss this mirrors syncs to the host on every call (``float(total)``,
#2265). A metric that did the same would stall every validation batch, so the
no-sync claim is checked by making any sync an error, not by reading the code.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.core.metrics.coil_manifold import as_interleaved
from spectramr.core.metrics.computer import ValidationMetricsComputer
from spectramr.core.metrics.context import MetricContext
from spectramr.core.metrics.metric_directions import metric_higher_is_better
from spectramr.core.metrics.registry import MetricsRegistry
from spectramr.core.metrics.types import ValidationMetricsConfig
from spectramr.infrastructure.physics.fft_ops import fft2c

_C, _N = 4, 24
_NAMES = ("null_space_residual", "null_space_residual_fraction")


def _maps() -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, _N), torch.linspace(-1, 1, _N), indexing="ij")
    centres = [(-0.7, -0.7), (0.7, -0.7), (-0.7, 0.7), (0.7, 0.7)]
    maps = torch.stack(
        [
            torch.exp(-((xx - a) ** 2 + (yy - b) ** 2)) * torch.exp(1j * (a * xx - b * yy))
            for a, b in centres
        ]
    ).to(torch.complex64)
    return (maps / maps.abs().pow(2).sum(0, keepdim=True).sqrt()).unsqueeze(0)


def _randn(*shape: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.complex(torch.randn(*shape, generator=gen), torch.randn(*shape, generator=gen))


def _kspace(images: torch.Tensor) -> torch.Tensor:
    return as_interleaved(fft2c(images))


def _metric(name: str):
    return MetricsRegistry.get(name)


class TestRegistration:
    @pytest.mark.parametrize("name", _NAMES)
    def test_declared_contract(self, name: str) -> None:
        assert MetricsRegistry.is_registered(name)
        assert MetricsRegistry.requires_reference(name) is False
        assert MetricsRegistry.needs(name) == ("coil_maps",)
        assert metric_higher_is_better(name) is False


class TestValues:
    def test_zero_on_a_rank_one_prediction(self) -> None:
        maps = _maps()
        x = maps * _randn(1, 1, _N, _N, seed=0)
        ctx = MetricContext(coil_maps=maps)
        energy = x.abs().pow(2).sum(1).mean().item()
        assert _metric("null_space_residual")(_kspace(x), context=ctx).item() < 1e-6 * energy
        assert _metric("null_space_residual_fraction")(_kspace(x), context=ctx).item() < 1e-6

    def test_isotropic_noise_is_c_minus_one_over_c(self) -> None:
        noise = _randn(32, _C, _N, _N, seed=1)
        ctx = MetricContext(coil_maps=_maps().expand(32, -1, -1, -1))
        value = _metric("null_space_residual_fraction")(_kspace(noise), context=ctx)
        assert value.item() == pytest.approx((_C - 1) / _C, rel=0.02)

    def test_it_lower_bounds_the_error(self) -> None:
        """||P_perp f||^2 <= ||f - x||^2 for the truth, whatever f is."""
        maps = _maps()
        x = maps * _randn(1, 1, _N, _N, seed=2)
        f = x + 0.2 * _randn(1, _C, _N, _N, seed=3)
        residual = _metric("null_space_residual")(_kspace(f), context=MetricContext(coil_maps=maps))
        error = (f - x).abs().pow(2).sum(1).mean()
        assert residual.item() <= error.item()

    def test_complex_and_interleaved_predictions_agree(self) -> None:
        maps = _maps()
        k = fft2c(_randn(1, _C, _N, _N, seed=4))
        ctx = MetricContext(coil_maps=maps)
        metric = _metric("null_space_residual")
        assert torch.allclose(metric(k, context=ctx), metric(as_interleaved(k), context=ctx))

    def test_the_loose_kwarg_form_reaches_the_maps(self) -> None:
        k = _kspace(_randn(1, _C, _N, _N, seed=5))
        assert torch.isfinite(_metric("null_space_residual")(k, sensitivity_maps=_maps()))


class TestItRefuses:
    @pytest.mark.parametrize("name", _NAMES)
    def test_missing_maps(self, name: str) -> None:
        with pytest.raises(ValueError, match="none reached"):
            _metric(name)(_kspace(_randn(1, _C, _N, _N, seed=6)), context=MetricContext())

    @pytest.mark.parametrize("name", _NAMES)
    def test_real_maps(self, name: str) -> None:
        ctx = MetricContext(coil_maps=_maps().abs())
        with pytest.raises(ValueError, match="COMPLEX"):
            _metric(name)(_kspace(_randn(1, _C, _N, _N, seed=7)), context=ctx)

    def test_an_image_domain_caller(self) -> None:
        ctx = MetricContext(coil_maps=_maps(), signal_domain="image")
        with pytest.raises(ValueError, match="reads multi-coil k-space"):
            _metric("null_space_residual")(_kspace(_randn(1, _C, _N, _N, seed=8)), context=ctx)


class TestThroughTheComputer:
    def test_the_computer_forwards_the_context_and_returns_a_float(self) -> None:
        computer = ValidationMetricsComputer(
            ValidationMetricsConfig.from_dict({"metrics": list(_NAMES)})
        )
        k = _kspace(_randn(2, _C, _N, _N, seed=9))
        out = computer.compute(
            k, k, only=_NAMES, context=MetricContext(coil_maps=_maps().expand(2, -1, -1, -1))
        )
        assert set(out) == set(_NAMES)
        assert all(isinstance(v, float) and v > 0 for v in out.values())

    def test_without_the_context_the_computer_raises(self) -> None:
        """ValueError is the class the computer re-raises rather than NaN-ing."""
        computer = ValidationMetricsComputer(
            ValidationMetricsConfig.from_dict({"metrics": ["null_space_residual"]})
        )
        k = _kspace(_randn(1, _C, _N, _N, seed=10))
        with pytest.raises(ValueError, match="null_space_residual"):
            computer.compute(k, k)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="sync detection needs a CUDA device")
@pytest.mark.filterwarnings("ignore:Synchronization debug mode is a prototype")
@pytest.mark.parametrize("name", _NAMES)
def test_no_host_sync(name: str) -> None:
    """Planted: any device->host sync inside the metric raises under this mode."""
    device = torch.device("cuda")
    k = _kspace(_randn(1, _C, _N, _N, seed=11)).to(device)
    ctx = MetricContext(coil_maps=_maps().to(device))
    metric = _metric(name)
    metric(k, context=ctx)  # warm-up: first-call allocator work is not the metric's
    torch.cuda.set_sync_debug_mode("error")
    try:
        value = metric(k, context=ctx)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert value.device.type == "cuda" and value.ndim == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="sync detection needs a CUDA device")
@pytest.mark.filterwarnings("ignore:Synchronization debug mode is a prototype")
def test_the_sync_detector_fires_on_a_planted_sync() -> None:
    """Without this the no-sync test could be green because the mode detects nothing."""
    value = torch.ones(4, device="cuda").sum()
    torch.cuda.set_sync_debug_mode("error")
    try:
        with pytest.raises(RuntimeError):
            float(value)
    finally:
        torch.cuda.set_sync_debug_mode("default")
