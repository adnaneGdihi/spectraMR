"""Tests for the CycleBloch backward renders."""

import pytest
import torch

from spectramr.config.schemas.training.cycle_bloch import CycleBlochConfig
from spectramr.infrastructure.physics.biophysics.bloch import BlochSimulator
from spectramr.infrastructure.training.strategies.cycle_bloch_physics import (
    BACKWARD_MODELS,
    T1_RANGE_MS,
    T2_RANGE_MS,
    ParameterEstimator,
)


def _tissue():
    g = torch.Generator().manual_seed(0)
    shape = (2, 1, 8, 8)
    pd = (0.5 + torch.rand(shape, generator=g)).requires_grad_()
    t1 = (300.0 + 1500.0 * torch.rand(shape, generator=g)).requires_grad_()
    t2 = (40.0 + 100.0 * torch.rand(shape, generator=g)).requires_grad_()
    return pd, t1, t2


def _ulf(tissue_field):
    return CycleBlochConfig(
        backward_model="ulf_forward",
        ulf_tr_ms=500.0,
        ulf_te_ms=14.0,
        ulf_flip_deg=90.0,
        tissue_field_tesla=tissue_field,
    )


def _epg():
    return CycleBlochConfig(
        backward_model="epg_tse",
        ulf_tr_ms=500.0,
        ulf_te_ms=14.0,
        echo_spacing_ms=7.0,
        echo_train_length=4,
        refocus_deg=150.0,
    )


def test_bloch_fixed_is_the_historical_spin_echo():
    render, prov = BACKWARD_MODELS["bloch_fixed"](None, None)
    pd, t1, t2 = _tissue()
    expected = BlochSimulator.spin_echo(m0=pd, t1=t1, t2=t2, te=14.0, tr=500.0)
    torch.testing.assert_close(render(pd, t1, t2, None), expected)
    assert prov == {"tr_ms": 500.0, "te_ms": 14.0}


def test_ulf_forward_without_dispersion_equals_spin_echo():
    render, _ = BACKWARD_MODELS["ulf_forward"](_ulf(0.064), 0.064)
    pd, t1, t2 = _tissue()
    expected = BlochSimulator.spin_echo(m0=pd, t1=t1, t2=t2, te=14.0, tr=500.0)
    torch.testing.assert_close(render(pd, t1, t2, None), expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("name", sorted(BACKWARD_MODELS))
def test_every_render_backpropagates_finite_nonzero_gradients(name):
    cfg = {"ulf_forward": _ulf(3.0), "epg_tse": _epg()}.get(name)
    render, _ = BACKWARD_MODELS[name](cfg, 0.064)
    pd, t1, t2 = _tissue()
    render(pd, t1, t2, None).sum().backward()
    for x in (pd, t1, t2):
        assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


def test_epg_tse_render_equals_the_echo_of_the_declared_full_train():
    from spectramr.infrastructure.physics.tse_forward import tse_signal

    render, prov = BACKWARD_MODELS["epg_tse"](_epg(), None)
    pd, t1, t2 = _tissue()
    b1 = torch.full_like(pd, 0.9)
    seq = {"tr": 500.0, "te": 14.0, "ti": 0.0, "esp": 7.0, "etl": 4, "refocus_deg": 150.0}
    full = tse_signal(pd, t1, t2, family=0, b1_plus=b1, **seq)
    torch.testing.assert_close(render(pd, t1, t2, b1), full)
    assert prov["echo_train_length"] == 4


def test_ulf_forward_needs_a_declared_field():
    with pytest.raises(ValueError, match=r"physics\.field_strength"):
        BACKWARD_MODELS["ulf_forward"](_ulf(3.0), None)


_CONFIGS = {"bloch_fixed": lambda: None, "ulf_forward": lambda: _ulf(3.0), "epg_tse": _epg}


def _render(name, pd, t1, t2):
    render, _ = BACKWARD_MODELS[name](_CONFIGS[name](), 0.064)
    return render(pd, t1, t2, None)


@pytest.mark.parametrize("name", sorted(BACKWARD_MODELS))
def test_typical_tissue_in_ms_renders_an_order_one_signal(name):
    # Grey and white matter at 3 T: PD 0.8, T1 1000 ms, T2 80 ms.
    full = torch.full((1, 1, 4, 4), 1.0)
    out = _render(name, 0.8 * full, 1000.0 * full, 80.0 * full)
    assert out.min() > 0.05 and out.max() < 1.0


@pytest.mark.parametrize("with_b1", [False, True])
def test_estimator_maps_stay_in_physiological_ms_bounds(with_b1):
    est = ParameterEstimator(in_channels=1, with_b1=with_b1)
    for bias in (-50.0, 50.0):
        torch.nn.init.zeros_(est.net[-1].weight)
        torch.nn.init.constant_(est.net[-1].bias, bias)
        maps = est(torch.rand(1, 1, 4, 4))
        assert maps[:, 0].min() >= 0.0 and maps[:, 0].max() <= 1.0
        assert T1_RANGE_MS[0] <= maps[:, 1].min() and maps[:, 1].max() <= T1_RANGE_MS[1]
        assert T2_RANGE_MS[0] <= maps[:, 2].min() and maps[:, 2].max() <= T2_RANGE_MS[1]


@pytest.mark.parametrize("name", sorted(BACKWARD_MODELS))
def test_freshly_initialised_estimator_does_not_vanish_the_cycle_signal(name):
    torch.manual_seed(0)
    maps = ParameterEstimator(in_channels=1)(torch.rand(2, 1, 8, 8))
    out = _render(name, maps[:, 0:1], maps[:, 1:2], maps[:, 2:3])
    assert out.mean() > 1e-2


# Per-contrast sequence table: the Leiden 64 mT protocol (Table 1, Zenodo 10.5281/zenodo.20281403).
_MAP = {"T1w": 0, "T2w": 1, "FLAIR": 2}
_LEIDEN = {
    "T1w": {"tr_ms": 880.0, "te_ms": 5.03, "ti_ms": 340.0},
    "T2w": {"tr_ms": 2000.0, "te_ms": 194.8},
    "FLAIR": {"tr_ms": 3500.0, "te_ms": 175.1, "ti_ms": 1290.0},
}
_TRAIN = {
    "T1w": {"echo_spacing_ms": 5.03, "echo_train_length": 32},
    "T2w": {"echo_spacing_ms": 4.87, "echo_train_length": 80},
    "FLAIR": {"echo_spacing_ms": 5.15, "echo_train_length": 68},
}


def _table_ulf(tissue_field=0.064):
    seqs = {k: {**v, "flip_deg": 90.0} for k, v in _LEIDEN.items()}
    return CycleBlochConfig(
        backward_model="ulf_forward", tissue_field_tesla=tissue_field, sequences=seqs
    )


def _table_epg():
    seqs = {k: {**v, **_TRAIN[k], "refocus_deg": 180.0} for k, v in _LEIDEN.items()}
    return CycleBlochConfig(backward_model="epg_tse", sequences=seqs)


def test_table_renders_each_sample_with_its_own_contrast():
    from spectramr.infrastructure.training.strategies.cycle_bloch_physics import build_backward

    render, prov = build_backward(_table_ulf(), 0.064, _MAP)
    pd, t1, t2 = _tissue()
    out = render(pd, t1, t2, None, torch.tensor([1, 2]))
    t2w, _ = BACKWARD_MODELS["ulf_forward"](
        CycleBlochConfig(
            backward_model="ulf_forward",
            ulf_tr_ms=2000.0,
            ulf_te_ms=194.8,
            ulf_flip_deg=90.0,
            tissue_field_tesla=0.064,
        ),
        0.064,
    )
    torch.testing.assert_close(out[0:1], t2w(pd, t1, t2, None)[0:1], rtol=1e-4, atol=1e-6)
    assert not torch.allclose(out[1:2], t2w(pd, t1, t2, None)[1:2])
    assert prov["sequences"]["FLAIR"]["ti_ms"] == 1290.0


def test_ulf_forward_inversion_is_the_ideal_ir_spin_echo_without_dispersion():
    from spectramr.infrastructure.training.strategies.cycle_bloch_physics import build_backward

    render, _ = build_backward(_table_ulf(), 0.064, _MAP)
    pd, t1, t2 = _tissue()
    out = render(pd, t1, t2, None, torch.tensor([0, 2]))
    for row, (tr, te, ti) in enumerate([(880.0, 5.03, 340.0), (3500.0, 175.1, 1290.0)]):
        e = pd * (1 - 2 * torch.exp(-ti / t1) + torch.exp(-tr / t1)).abs() * torch.exp(-te / t2)
        torch.testing.assert_close(out[row], e[row], rtol=1e-4, atol=1e-6)


def test_epg_table_uses_the_inversion_and_the_train_echo():
    from spectramr.infrastructure.training.strategies.cycle_bloch_physics import build_backward

    render, prov = build_backward(_table_epg(), None, _MAP)
    pd, t1, t2 = _tissue()
    flair = render(pd, t1, t2, None, torch.tensor([2, 2]))
    t2w = render(pd, t1, t2, None, torch.tensor([1, 1]))
    assert torch.isfinite(flair).all() and not torch.allclose(flair, t2w)
    assert prov["sequences"]["T2w"]["echo_train_length"] == 80


def test_table_raises_on_a_contrast_map_mismatch_and_a_missing_id():
    from spectramr.infrastructure.training.strategies.cycle_bloch_physics import build_backward

    with pytest.raises(ValueError, match="contrast_map"):
        build_backward(_table_ulf(), 0.064, {"T1w": 0, "T2w": 1})
    render, _ = build_backward(_table_ulf(), 0.064, _MAP)
    pd, t1, t2 = _tissue()
    with pytest.raises(ValueError, match="no contrast_id"):
        render(pd, t1, t2, None)
    assert torch.isnan(render(pd, t1, t2, None, torch.tensor([1, 7]))[1]).all()
