"""Nonlinear post-noise codes as Blackwell garblings: probe, relations, realiser."""

from __future__ import annotations

import pytest
import torch

from spectramr.core.metrics.meta_evaluation.blackwell.evaluate import blur_envelope, displays
from spectramr.infrastructure.physics.degradations import garbling_catalogue as gc
from spectramr.infrastructure.physics.fft_ops import fft2c

pytestmark = pytest.mark.unit
SHAPE = (24, 24)


def _contents() -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    return [torch.rand(SHAPE, generator=g) + 0.2 for _ in range(2)]


def _catalogue(**kw):
    cat = gc.garbling_catalogue(_contents(), **kw)
    return cat, {e.name: k for k, e in enumerate(cat.experiments)}


def test_probe_accepts_the_audited_codes_and_rejects_an_identity() -> None:
    for code in gc.GARBLING_CODES:
        assert gc.probe_garbling(code, 1.0, 0, SHAPE) > 0.0
    with pytest.raises(ValueError, match="unchanged"):
        gc.probe_garbling("adc_clipping", 0.0, 0, SHAPE)


def test_probe_rejects_a_nondeterministic_code(monkeypatch: pytest.MonkeyPatch) -> None:
    """Planted: a map that is not a fixed function of its input is not a garbling."""

    def noisy(code: str, x: torch.Tensor, theta: float, seed: int = 0) -> torch.Tensor:
        return x + torch.randn_like(x.real)

    monkeypatch.setattr(gc, "apply_degradation", noisy)
    with pytest.raises(ValueError, match="deterministic"):
        gc.probe_garbling("adc_clipping", 1.0, 0, SHAPE)


def test_garbling_relations() -> None:
    cat, ix = _catalogue(codes=("adc_clipping",), thetas=(1.0,))
    g = cat.graph
    clip_n, clip_t = ix["adc_clipping_1(noise_0.1)"], ix["adc_clipping_1(truncation_0.25)"]
    strict, unc = set(g.strict), set(g.uncertified)
    assert (ix["noise_0.1"], clip_n) in strict  # b >= K o b
    assert (ix["noise_0.05"], clip_n) in strict  # a >= b, so a >= K o b
    assert (ix["truncation_0.25"], clip_t) in strict
    assert tuple(sorted((ix["noise_0.15"], clip_n))) in unc  # noisier base: not certified
    assert tuple(sorted((clip_n, clip_t))) in unc  # K o a vs K o b: not certified


def test_realiser_matches_the_base_display_and_applies_the_code() -> None:
    cat, ix = _catalogue(codes=("adc_clipping",), thetas=(1.0,))
    realiser = cat.realiser(2, 0)
    x = _contents()[0]
    noise = torch.randn(
        (2, *SHAPE), dtype=torch.complex64, generator=torch.Generator().manual_seed(1)
    )
    env = blur_envelope(SHAPE, (0.0,), None)
    base = cat.experiments[ix["noise_0.1"]]
    assert torch.allclose(
        realiser(fft2c(x), base, noise, 1.0, env), displays(fft2c(x), base.base, noise, 1.0, env)
    )
    clipped = realiser(fft2c(x), cat.experiments[ix["adc_clipping_1(noise_0.1)"]], noise, 1.0, env)
    assert not torch.allclose(clipped, realiser(fft2c(x), base, noise, 1.0, env))


def test_unknown_codes_and_bases_raise() -> None:
    with pytest.raises(ValueError, match="audited garblings"):
        gc.garbling_catalogue(_contents(), codes=("rician",))
    with pytest.raises(ValueError, match="radial catalogue"):
        gc.garbling_catalogue(_contents(), garbled_bases=("blur_9",))
    with pytest.raises(ValueError, match="2-D"):
        gc.garbling_catalogue([torch.zeros(1, *SHAPE)])
