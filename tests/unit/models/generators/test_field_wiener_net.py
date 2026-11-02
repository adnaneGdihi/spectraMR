"""Tests for FieldWienerNet (B-2.7 field-dependent spectral Wiener restorer)."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.generators.field_wiener_net import FieldWienerNet


def _net(**kw) -> FieldWienerNet:
    return FieldWienerNet(width=24, n_blocks=2, **kw)


_HW = (64, 64)


def test_forward_shape() -> None:
    out = _net()(torch.rand(2, 1, 32, 32), field_strength=torch.tensor([0.1, 3.0]), field_strength_target=None)
    assert out.shape == (2, 1, 32, 32)


def test_lower_field_narrows_recoverable_band() -> None:
    # THE oracle: a field-dependent Wiener raises the noise ratio (and narrows the recoverable
    # band, on the data-INDEPENDENT reference) as the field drops.
    m = _net(use_field_noise=True).eval()
    prev_band, prev_noise = None, None
    for b in (7.0, 3.0, 1.5, 0.1):
        with torch.no_grad():
            noise = float(m.noise_level(torch.tensor([b])))
            band = float(m.recoverable_band(torch.tensor([b]), _HW))
        if prev_band is not None:
            assert band <= prev_band + 1e-6  # band does not grow as field drops
            assert noise >= prev_noise - 1e-6  # noise does not shrink as field drops
        prev_band, prev_noise = band, noise
    # field-sensitivity is robust + positive (the mechanism fires; #16 guard), data-independent.
    with torch.no_grad():
        hi = float(m.recoverable_band(torch.tensor([7.0]), _HW))
        lo = float(m.recoverable_band(torch.tensor([0.1]), _HW))
    assert hi - lo > 1e-3


def test_field_slope_is_negative_by_construction() -> None:
    # noise_a can take ANY value; the field slope stays <= 0 so "lower field -> larger N" can never
    # invert during training.
    m = _net(use_field_noise=True)
    with torch.no_grad():
        m.noise_a.fill_(50.0)  # adversarial: large positive raw value
        n_low = float(m.noise_level(torch.tensor([0.1])))
        n_high = float(m.noise_level(torch.tensor([7.0])))
    assert n_low > n_high  # direction preserved despite a huge noise_a


def test_wiener_gain_is_scale_invariant() -> None:
    # The relative noise floor (N * mean signal power) makes the gain invariant to |x0| scale —
    # so the mechanism is not a near-no-op / kill-all depending on the spectrum magnitude (#16).
    torch.manual_seed(0)
    m = _net(use_field_noise=True).eval()
    x0 = torch.rand(1, 1, 32, 32)
    with torch.no_grad():
        g1 = m.wiener_gain(x0, torch.tensor([1.5]))
        g2 = m.wiener_gain(10.0 * x0, torch.tensor([1.5]))
    assert float((g1 - g2).abs().max()) < 1e-4


def test_field_blind_band_is_field_invariant() -> None:
    m = _net(use_field_noise=False).eval()
    with torch.no_grad():
        hi = float(m.recoverable_band(torch.tensor([7.0]), _HW))
        lo = float(m.recoverable_band(torch.tensor([0.1]), _HW))
    assert abs(hi - lo) < 1e-6  # field-blind -> constant band


def test_field_batch_guard() -> None:
    with pytest.raises(ValueError, match="field_strength batch"):
        _net()(torch.rand(2, 1, 16, 16), field_strength=torch.tensor([1.5]), field_strength_target=None)


def test_field_required_for_probe() -> None:
    import inspect

    params = inspect.signature(FieldWienerNet.forward).parameters
    assert params["field_strength"].default is inspect.Parameter.empty
    assert params["field_strength_target"].default is inspect.Parameter.empty


def test_target_field_conditions_the_estimate() -> None:
    # #196: on Task 3 one input is paired with several target fields; the net must see which.
    torch.manual_seed(0)
    m = _net(use_target_field=True).eval()
    for p in m.target_film.parameters():
        p.data += 0.1
    x, b_s = torch.rand(1, 1, 16, 16), torch.tensor([1.5])
    with torch.no_grad():
        lo = m(x, field_strength=b_s, field_strength_target=torch.tensor([0.1]))
        hi = m(x, field_strength=b_s, field_strength_target=torch.tensor([7.0]))
    assert not torch.allclose(lo, hi)


def test_target_conditioned_net_raises_without_target() -> None:
    with pytest.raises(ValueError, match="field_strength_target"):
        _net(use_target_field=True)(
            torch.rand(1, 1, 8, 8), field_strength=torch.tensor([1.5]), field_strength_target=None
        )
    with pytest.raises(ValueError, match="use_target_field"):
        FieldWienerNet(use_target_field="yes")  # type: ignore[arg-type]


def test_rejects_complex_and_multichannel_and_nonbool() -> None:
    with pytest.raises(ValueError, match="magnitude"):
        _net()(
            torch.rand(1, 1, 8, 8, dtype=torch.cfloat),
            field_strength=torch.tensor([1.5]),
            field_strength_target=None,
        )
    with pytest.raises(ValueError, match="magnitude-only"):
        FieldWienerNet(in_channels=2)
    with pytest.raises(ValueError, match="use_field_noise"):
        FieldWienerNet(use_field_noise="yes")  # type: ignore[arg-type]


def test_registered() -> None:
    from spectramr.models.init_registry import populate_model_registry
    from spectramr.models.registry import MODEL_REGISTRY

    populate_model_registry()
    assert "field_wiener_net" in MODEL_REGISTRY
