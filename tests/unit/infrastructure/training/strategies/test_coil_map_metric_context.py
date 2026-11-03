"""``coil_map_metric_context``: maps + rate for the configured coil-map metrics."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from spectramr.core.metrics.computer import create_validation_metrics_computer
from spectramr.infrastructure.physics.coil_sensitivity import create_synthetic_csm
from spectramr.infrastructure.training.strategies import coil_map_metric_context as mod


def _coils(num_coils: int = 4) -> torch.Tensor:
    return create_synthetic_csm((1, 1, 32, 32), num_coils=num_coils).to(torch.complex64)


def _config(method: str | None = None) -> SimpleNamespace:
    estimation = None if method is None else SimpleNamespace(method=method)
    return SimpleNamespace(
        physics=SimpleNamespace(coil_processing=SimpleNamespace(estimation=estimation))
    )


def test_no_coil_map_metric_estimates_nothing(monkeypatch) -> None:
    def _boom(*_a, **_k):
        raise AssertionError("maps must not be estimated when no metric reads them")

    monkeypatch.setattr(mod, "estimate_smaps_calibrated", _boom)
    computer = create_validation_metrics_computer(["psnr"], device="cpu")
    assert mod.coil_map_metric_context(computer, _config(), _coils(), 2.0) is None
    assert mod.coil_map_metric_context(None, _config(), _coils(), 2.0) is None


def test_alias_spelling_selects_the_metric() -> None:
    spec = SimpleNamespace(name="GFactor", enabled=True)
    computer = SimpleNamespace(config=SimpleNamespace(metrics=[spec]))
    assert mod.configured_coil_map_metrics(computer) == ("g_factor",)
    spec.enabled = False
    assert mod.configured_coil_map_metrics(computer) == ()


@pytest.mark.parametrize(("method", "expected"), [(None, "espirit"), ("power_iter", "power_iter")])
def test_declared_method_is_read_and_espirit_is_the_default(monkeypatch, method, expected) -> None:
    seen: dict[str, object] = {}

    def _spy(kspace, *, log_scaled, method, **kw):
        seen.update(method=method, log_scaled=log_scaled)
        return torch.ones_like(kspace)

    monkeypatch.setattr(mod, "estimate_smaps_calibrated", _spy)
    computer = create_validation_metrics_computer(["g_factor"], device="cpu")
    ctx = mod.coil_map_metric_context(computer, _config(method), _coils(), 3.0)
    assert seen == {"method": expected, "log_scaled": False}
    assert ctx is not None and ctx.acceleration == 3.0 and ctx.coil_maps.shape == (1, 4, 32, 32)


@pytest.mark.parametrize(("coils", "rate"), [(1, 2.0), (4, 6.0)])
def test_unmeasurable_g_factor_raises(coils, rate) -> None:
    computer = create_validation_metrics_computer(["g_factor"], device="cpu")
    with pytest.raises(ValueError, match="underdetermined"):
        mod.coil_map_metric_context(computer, _config(), _coils(coils), rate)
