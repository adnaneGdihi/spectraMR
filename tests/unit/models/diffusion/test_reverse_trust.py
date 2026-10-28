"""Wiener data consistency with the fitted decay's signal power and a ``3 sigma`` disc (Algorithm G)."""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.data_consistency import (
    HardDataConsistency,
    NoiseAdaptiveDataConsistency,
)
from spectramr.models.diffusion.reverse_decay import fit_decay_envelope
from spectramr.models.diffusion.reverse_domain import linear_kspace, stacked_kspace
from spectramr.models.diffusion.reverse_trust import (
    TRUST_RADIUS,
    apply_fitted_trust,
    build_fitted_trust,
)
from tests.unit.models.diffusion import _reverse_phantom as P

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def setting():
    proc = P.process(8.0)
    target = P.kspace()
    mask = P.mask(proc)
    meas = target * mask
    decay = fit_decay_envelope(
        meas,
        mask,
        log_scaled=False,
        pooled=False,
        axis="row",
        noise=NoiseAdaptiveDataConsistency(),
    )
    return proc, target, meas, mask, decay


def _sampler(proc, target, *, log_scaled: bool = False):
    return P.sampler(
        P.Oracle(target, proc),
        "replace_freeze_dc",
        dc_method="noise_adaptive",
        decay_envelope="per_sample",
        dc_trust="fitted_decay",
        kspace_log_scaled=log_scaled,
    )


def test_the_signal_power_follows_the_law_off_centre_and_the_data_on_it(setting) -> None:
    _, _, meas, mask, decay = setting
    trust = build_fitted_trust(decay, meas, mask, log_scaled=False)
    centre = meas.shape[-2] // 2
    power = linear_kspace(meas, log_scaled=False).abs().pow(2).mean(dim=1, keepdim=True)
    assert trust.signal_power.shape == (meas.shape[0], 1, *meas.shape[-2:])
    assert torch.allclose(trust.signal_power[..., centre, :], power[..., centre, :])
    assert bool((trust.signal_power >= 0).all()) and bool(torch.isfinite(trust.signal_power).all())
    assert torch.allclose(trust.radius.flatten(), TRUST_RADIUS * decay.sigma2.sqrt())


def test_the_signal_power_falls_with_row_distance(setting) -> None:
    _, _, meas, mask, decay = setting
    trust = build_fitted_trust(decay, meas, mask, log_scaled=False)
    row_mean = trust.signal_power.mean(dim=-1)[:, 0]
    centre = meas.shape[-2] // 2
    outward = row_mean[:, centre + 1 :]
    assert bool((outward[:, 1:] <= outward[:, :-1] * (1 + 1e-5)).all())


def test_measured_bins_land_in_the_disc_and_unmeasured_bins_are_untouched(setting) -> None:
    proc, target, meas, mask, decay = setting
    trust = build_fitted_trust(decay, meas, mask, log_scaled=False)
    wild = 100.0 * target.flip(-1)
    out = apply_fitted_trust(_sampler(proc, target), trust, wild, meas, mask)
    measured = mask.bool().expand_as(linear_kspace(out, log_scaled=False))
    gap = (linear_kspace(out, log_scaled=False) - linear_kspace(meas, log_scaled=False)).abs()
    radius = trust.radius.expand_as(gap)
    assert bool((gap[measured] <= radius[measured] * (1 + 1e-4)).all())
    free = (mask == 0).expand_as(out)
    assert torch.equal(out[free], wild[free])


def test_fitted_trust_refuses_a_model_without_the_noise_adaptive_layer(setting) -> None:
    proc, target, meas, mask, decay = setting
    trust = build_fitted_trust(decay, meas, mask, log_scaled=False)
    sampler = _sampler(proc, target)
    sampler.model.dc_layer = HardDataConsistency()
    with pytest.raises(ValueError, match="NoiseAdaptiveDataConsistency"):
        apply_fitted_trust(sampler, trust, target, meas, mask)


def test_a_log_scaled_arm_blends_once_on_linear_kspace(setting) -> None:
    """PLANTED: converting in ``apply_fitted_trust`` as well as in the layer decompresses
    twice; the result must equal one linear-domain blend compressed once."""
    proc, target, meas, mask, _ = setting
    wild = 100.0 * target.flip(-1)
    comp = lambda x: stacked_kspace(linear_kspace(x, log_scaled=False), x, log_scaled=True)  # noqa: E731
    meas_c, wild_c = comp(meas), comp(wild)
    decay = fit_decay_envelope(
        meas_c, mask, log_scaled=True, pooled=False, axis="row",
        noise=NoiseAdaptiveDataConsistency(),
    )
    trust = build_fitted_trust(decay, meas_c, mask, log_scaled=True)
    support = mask.amax(dim=-3, keepdim=True)
    linear = NoiseAdaptiveDataConsistency()(
        wild, meas, support, is_kspace_domain=True,
        noise_sigma=trust.sigma, signal_power=trust.signal_power, trust_radius=TRUST_RADIUS,
    )
    expected = wild_c * (1.0 - mask) + comp(linear) * mask
    sampler = _sampler(proc, target, log_scaled=True)
    sampler.model.dc_layer = NoiseAdaptiveDataConsistency(log_scaled=True)
    out = apply_fitted_trust(sampler, trust, wild_c, meas_c, mask)
    assert torch.allclose(out, expected, rtol=1e-4, atol=1e-5)


def test_fitted_trust_refuses_a_layer_in_the_other_domain(setting) -> None:
    proc, target, meas, mask, decay = setting
    trust = build_fitted_trust(decay, meas, mask, log_scaled=False)
    sampler = _sampler(proc, target, log_scaled=True)
    with pytest.raises(ValueError, match="kspace_log_scaled=True"):
        apply_fitted_trust(sampler, trust, target, meas, mask)


def test_fitted_trust_refuses_a_radial_envelope(setting) -> None:
    """PLANTED: the separable line-by-across power has no meaning over shells of radial |k|."""
    _, _, meas, mask, _ = setting
    radial = fit_decay_envelope(
        meas,
        mask,
        log_scaled=False,
        pooled=False,
        axis="radial",
        noise=NoiseAdaptiveDataConsistency(),
    )
    with pytest.raises(ValueError, match="line envelope"):
        build_fitted_trust(radial, meas, mask, log_scaled=False)


def test_a_noise_floor_sample_trusts_the_prediction_inside_the_disc(setting) -> None:
    """No fitted row means no signal power off centre, so lambda = 0 there: the
    blend takes the prediction, still within 3 sigma of y, and stays finite."""
    proc, target, meas, mask, _ = setting
    noise = 1e-3 * torch.randn(meas[1:].shape, generator=torch.Generator().manual_seed(5))
    mixed = torch.cat([meas[:1], noise * mask[1:]])
    decay = fit_decay_envelope(
        mixed, mask, log_scaled=False, pooled=False, axis="row",
        noise=NoiseAdaptiveDataConsistency(),
    )
    assert decay.unfitted().tolist() == [False, True]
    trust = build_fitted_trust(decay, mixed, mask, log_scaled=False)
    assert bool(torch.isfinite(trust.signal_power).all())
    centre = mixed.shape[-2] // 2
    off_centre = torch.ones(mixed.shape[-2], dtype=torch.bool)
    off_centre[centre] = False
    assert bool((trust.signal_power[1, :, off_centre] == 0).all())
    out = apply_fitted_trust(_sampler(proc, target), trust, 100.0 * target.flip(-1), mixed, mask)
    assert bool(torch.isfinite(out).all())
    gap = (linear_kspace(out, log_scaled=False) - linear_kspace(mixed, log_scaled=False)).abs()
    measured = mask.expand_as(mixed)[:, ::2] > 0
    assert bool((gap[1][measured[1]] <= trust.radius[1].item() * (1 + 1e-4)).all())



def test_a_column_envelope_gives_the_transposed_row_trust(setting) -> None:
    """``S(|k_line|)^2 q(k_across)`` on a column mask is the row construction transposed."""
    _, _, meas, mask, _ = setting
    # 48x64: on a square plane an H/W swap inside the column path would pass.
    meas, mask = meas[..., 8:56, :], mask[..., 8:56, :]
    decay = fit_decay_envelope(
        meas, mask, log_scaled=False, pooled=False, axis="row", noise=NoiseAdaptiveDataConsistency()
    )
    meas_t, mask_t = meas.transpose(-1, -2), mask.transpose(-1, -2)
    column = fit_decay_envelope(
        meas_t,
        mask_t,
        log_scaled=False,
        pooled=False,
        axis="column",
        noise=NoiseAdaptiveDataConsistency(),
    )
    rows = build_fitted_trust(decay, meas, mask, log_scaled=False)
    cols = build_fitted_trust(column, meas_t, mask_t, log_scaled=False)
    assert torch.allclose(cols.signal_power, rows.signal_power.transpose(-1, -2), rtol=1e-4)
    assert torch.allclose(cols.sigma, rows.sigma)
