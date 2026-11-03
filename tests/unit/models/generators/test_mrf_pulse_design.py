"""``mrf_pulse_design``: the Beltrami-bounded pulse-train generator of MRF §4."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.generators.mrf_pulse_design import MRFPulseDesign


def _model() -> MRFPulseDesign:
    return MRFPulseDesign(n_tr=8, alpha_init_rad=1.0, tr_init_ms=10.0, k_max=0.9)


def test_forward_broadcasts_one_design_over_the_batch() -> None:
    model = _model()
    out = model(torch.zeros(3, 1, 16))
    assert out.shape == (3, 8, 2)
    assert torch.equal(out[0], out[2])


def test_starts_at_the_initial_design_and_stays_inside_the_beltrami_bound() -> None:
    model = _model()
    assert torch.allclose(model.design(), model.initial_design())
    with torch.no_grad():
        model.mu_raw.fill_(50.0)
    alpha = model.design()[:, 0]
    assert (alpha <= 1.0 * (1 + 0.5 * 0.9) + 1e-5).all()


def test_design_is_trainable() -> None:
    model = _model()
    loss = model(torch.zeros(1, 4))[..., 0].pow(2).sum()
    loss.backward()
    assert model.mu_raw.grad is not None and torch.any(model.mu_raw.grad != 0)


@pytest.mark.parametrize(
    "kwargs",
    [{"n_tr": 1}, {"k_max": 1.0}, {"alpha_init_rad": 0.0}],
)
def test_invalid_construction_raises(kwargs: dict) -> None:
    base = {"n_tr": 8, "alpha_init_rad": 1.0, "tr_init_ms": 10.0}
    with pytest.raises(ValueError):
        MRFPulseDesign(**{**base, **kwargs})


def test_min_tr_bounds_every_reachable_design() -> None:
    model = MRFPulseDesign(n_tr=8, alpha_init_rad=1.0, tr_init_ms=10.0, k_max=0.9)
    with torch.no_grad():
        model.mu_raw.copy_(torch.tensor([0.0, -1e3]).expand(8, 2))
    assert model.min_tr_ms == pytest.approx(5.5)
    assert model.design()[:, 1].min().item() >= model.min_tr_ms - 1e-4
