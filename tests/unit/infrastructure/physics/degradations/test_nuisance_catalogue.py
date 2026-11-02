"""Phase-only registry codes as Blackwell experiments by unitary absorption."""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.degradations import nuisance_catalogue as nc
from spectramr.infrastructure.physics.fft_ops import fft2c

pytestmark = pytest.mark.unit
SHAPE = (24, 24)


def _contents() -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    return [torch.rand(SHAPE, generator=g) + 0.2 for _ in range(2)]


def _catalogue(**kw):
    cat = nc.nuisance_catalogue(_contents(), **kw)
    return cat, {e.name: k for k, e in enumerate(cat.experiments)}


def test_probe_rejects_a_code_that_is_not_unit_modulus() -> None:
    with pytest.raises(ValueError, match="modulus"):
        nc.probe_unitary("chemical_shift", 0.5, 0, SHAPE)
    h = nc.probe_unitary("translation_drift", 0.5, 0, SHAPE)
    assert torch.allclose(h.abs(), torch.ones(SHAPE, dtype=torch.float64), atol=1e-4)


def test_seed_dependence_is_detected() -> None:
    cat, _ = _catalogue(codes=("random_motion", "translation_drift"), thetas=(0.5,))
    seeded = {e["name"]: e["seeded"] for e in cat.provenance["entries"]}
    assert seeded == {"random_motion_0.5": True, "translation_drift_0.5": False}


def test_absorption_rules() -> None:
    cat, ix = _catalogue(codes=("random_motion", "translation_drift"), thetas=(0.5,))
    g = cat.graph
    fixed, mix = ix["translation_drift_0.5"], ix["random_motion_0.5"]
    pair = tuple(sorted((ix["noise_0.1"], fixed)))
    assert pair in set(g.equivalent)  # a known shift is invertible
    assert (ix["noise_0.05"], mix) in set(g.strict)  # a mixture is a garbling of noise
    assert (ix["noise_0.1"], mix) in set(g.strict)
    assert (fixed, ix["noise_0.2"]) in set(g.strict)  # equivalent to sigma, beats noisier
    assert (fixed, mix) in set(g.strict)
    assert tuple(sorted((ix["noise_0.2"], mix))) in set(g.uncertified)


def test_realiser_draws_the_nuisance_per_draw_and_keeps_fixed_codes_fixed() -> None:
    cat, ix = _catalogue(codes=("random_motion", "translation_drift"), thetas=(1.0,))
    realiser = cat.realiser(2, 0)
    x = _contents()[0]
    zero = torch.zeros((2, *SHAPE), dtype=torch.complex64)
    one = torch.ones(1, *SHAPE)
    mixed = realiser(fft2c(x), cat.experiments[ix["random_motion_1"]], zero, 1.0, one)
    fixed = realiser(fft2c(x), cat.experiments[ix["translation_drift_1"]], zero, 1.0, one)
    assert not torch.allclose(mixed[0, 0], mixed[0, 1])
    assert torch.allclose(fixed[0, 0], fixed[0, 1])


def test_realiser_rechecks_every_seed_it_uses(monkeypatch: pytest.MonkeyPatch) -> None:
    cat, ix = _catalogue(codes=("random_motion",), thetas=(1.0,))
    calls: list[int] = []
    real = nc.probe_unitary

    def spy(code: str, theta: float, seed: int, shape: tuple[int, int]) -> torch.Tensor:
        calls.append(seed)
        return real(code, theta, seed, shape)

    monkeypatch.setattr(nc, "probe_unitary", spy)
    zero = torch.zeros((3, *SHAPE), dtype=torch.complex64)
    exp = cat.experiments[ix["random_motion_1"]]
    cat.realiser(3, 10)(fft2c(_contents()[0]), exp, zero, 1.0, torch.ones(1, *SHAPE))
    assert calls == [10, 11, 12]


def test_unknown_codes_and_non_2d_contents_raise() -> None:
    with pytest.raises(ValueError, match="phase-only"):
        nc.nuisance_catalogue(_contents(), codes=("chemical_shift",))
    with pytest.raises(ValueError, match="2-D"):
        nc.nuisance_catalogue([torch.zeros(1, *SHAPE)])
