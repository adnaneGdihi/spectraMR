"""The power-law decay envelope (Algorithm F): its fit, how each sample resolves, and the bound it puts on bins."""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.data_consistency import NoiseAdaptiveDataConsistency
from spectramr.models.diffusion.kspace_process import paired_magnitude
from spectramr.infrastructure.physics.fft_ops import fft2c
from spectramr.models.diffusion.reverse_decay import (
    RESOLUTIONS,
    SLOPE_RANGE,
    bound_bins_by_decay,
    decay_ceiling,
    fit_decay_envelope,
    measured_power,
    row_index,
    tighten_ceiling,
    unmeasured_decay_bins,
)
from spectramr.models.diffusion.reverse_domain import stacked_kspace
from tests.unit.models.diffusion import _reverse_phantom as P

pytestmark = pytest.mark.unit

RATIO = 1.5


@pytest.fixture(scope="module")
def setting():
    proc = P.process(8.0)
    target = P.kspace()
    mask = P.mask(proc)
    return target, target * mask, mask


def _fit(meas, obs, *, pooled=False, axis="row"):
    return fit_decay_envelope(
        meas,
        obs,
        log_scaled=False,
        pooled=pooled,
        axis=axis,
        noise=NoiseAdaptiveDataConsistency(),
    )


def _rows_mask(rows, like):
    mask = torch.zeros(like.shape[0], 1, *like.shape[-2:])
    mask[..., rows, :] = 1.0
    return mask


def test_row_index_is_the_distance_to_the_centre_row() -> None:
    assert row_index(6, torch.device("cpu")).tolist() == [3.0, 2.0, 1.0, 0.0, 1.0, 2.0]


@pytest.mark.parametrize("pooled", [False, True])
def test_the_fit_recovers_a_decaying_slope(setting, pooled) -> None:
    _, meas, mask = setting
    env = _fit(meas, mask, pooled=pooled)
    assert bool(((env.slope > -1.6) & (env.slope < -1.2)).all()), env.slope
    assert bool((env.fitted_bins.sum(dim=1) >= 2).all())
    if pooled:
        assert torch.equal(env.slope, env.slope[:1].expand_as(env.slope))


def test_the_envelope_never_bounds_the_true_spectrum(setting) -> None:
    """At the loop's ratio neither the cap nor the row divisor touches the target."""
    target, meas, mask = setting
    env = _fit(meas, mask)
    cap = decay_ceiling(env, RATIO, log_scaled=False)
    rows = unmeasured_decay_bins(mask, target, env.axis)
    assert not bool(((paired_magnitude(target) > cap) & rows).any())
    out, divisor = bound_bins_by_decay(target, env, mask, ratio=RATIO, log_scaled=False)
    assert torch.equal(divisor, torch.ones_like(divisor))
    assert torch.allclose(out, target)


def test_the_row_divisor_only_removes_energy_and_only_from_unmeasured_rows(setting) -> None:
    target, meas, mask = setting
    env = _fit(meas, mask)
    loud = target * 50.0
    out, divisor = bound_bins_by_decay(loud, env, mask, ratio=RATIO, log_scaled=False)
    rows = unmeasured_decay_bins(mask, loud, env.axis).expand_as(paired_magnitude(loud))
    assert bool((divisor >= 1.0).all()) and float(divisor.max()) > 1.0
    assert bool((paired_magnitude(out) <= paired_magnitude(loud) * (1 + 1e-6)).all())
    assert torch.equal(paired_magnitude(out)[~rows], paired_magnitude(loud)[~rows])


def test_the_centre_row_is_never_bounded(setting) -> None:
    target, meas, mask = setting
    env = _fit(meas, mask)
    rms = env.bin_rms()
    assert bool(torch.isinf(rms[:, target.shape[-2] // 2]).all())
    assert bool(torch.isfinite(rms[:, : target.shape[-2] // 2]).all())


def test_tighten_ceiling_moves_only_unmeasured_outer_bins(setting) -> None:
    target, meas, mask = setting
    env = _fit(meas, mask)
    ceil = torch.full((target.shape[0], 1, 1, 1), 1e9)
    tight = tighten_ceiling(ceil, env, mask, target, ratio=RATIO, log_scaled=False)
    rows = unmeasured_decay_bins(mask, target, env.axis)
    assert bool((tight[~rows] == 1e9).all())
    assert bool((tight[rows] < 1e9).all())


def _band_spectrum(profile: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Two samples whose measured rows follow ``profile`` over ``|ky|``, plus the row mask."""
    gen = torch.Generator().manual_seed(0)
    noise = torch.complex(*torch.randn(2, 2, 2, 64, 64, generator=gen))
    band = (torch.arange(64) - 32).abs().lt(8).float().view(1, 1, 1, -1)
    k = 0.01 * noise + 10.0 * band * profile.view(1, 1, -1, 1)
    meas = stacked_kspace(k, torch.zeros(1), log_scaled=False)
    mask = _rows_mask([20, 26, 29, 31, 32, 33, 36, 44], meas)
    return meas * mask, mask


@pytest.mark.parametrize("shape", ["flat", "rising"])
def test_a_spectrum_that_does_not_decay_is_bounded_flat(shape) -> None:
    """A small object's spectrum is flat over the measured rows: no decay is assumed,
    and the bound sits at the loudest fitted row instead of refusing the sample."""
    rows = row_index(64, torch.device("cpu")).float()
    meas, mask = _band_spectrum(torch.ones(64) if shape == "flat" else 1.0 + rows)
    env = _fit(meas, mask)
    assert env.unfitted().all()
    assert (env.resolution == RESOLUTIONS.index("flat")).all()
    assert (env.slope == 0).all()
    power = measured_power(meas, mask, False)[1]
    measured = mask > 0
    row_power = env.axis.reduce_sum(power * measured) / env.axis.reduce_sum(measured).clamp_min(1)
    # E^2 = S_max^2 + sigma^2 >= P_row on every fitted row: no measured level is cut.
    assert bool((env.bin_rms() ** 2 >= row_power * (1 - 1e-5))[env.fitted_bins].all())


def test_a_steeper_than_admissible_spectrum_still_raises() -> None:
    """PLANTED: a bound steeper than ``SLOPE_RANGE[0]`` would damp true detail."""
    rows = row_index(64, torch.device("cpu")).float().clamp_min(1.0)
    meas, mask = _band_spectrum(rows**-4.0)
    with pytest.raises(ValueError, match="steeper than"):
        _fit(meas, mask)
    assert SLOPE_RANGE[0] < SLOPE_RANGE[1] < 0


def test_per_sample_with_one_distinct_row_distance_is_bounded_flat(setting) -> None:
    target, _, _ = setting
    mask = _rows_mask([31, 32, 33], target)
    env = _fit(target * mask, mask)
    assert (env.resolution == RESOLUTIONS.index("flat")).all()


def test_a_sample_with_no_off_centre_measurement_is_refused(setting) -> None:
    """Only the centre row measured: neither a decay nor a noise floor exists to read."""
    target, _, _ = setting
    mask = _rows_mask([32], target)
    with pytest.raises(ValueError, match="no bin at distance >= 1"):
        _fit(target * mask, mask, pooled=True)


def _noise_only(like: torch.Tensor) -> torch.Tensor:
    gen = torch.Generator().manual_seed(5)
    return 1e-3 * torch.randn(like.shape, generator=gen)


def test_a_noise_only_slice_gets_the_noise_floor_and_a_nonzero_cap(setting) -> None:
    """Nothing clears twice the noise power: ``E = sigma``, and the cap is not zero,
    which would erase every unmeasured coefficient of the slice."""
    target, _, mask = setting
    meas = _noise_only(target) * mask
    env = _fit(meas, mask)
    assert (env.resolution == RESOLUTIONS.index("noise_floor")).all()
    off_centre = env.bin_rms()[:, env.axis.coord >= 1]
    assert torch.allclose(off_centre, env.sigma2.sqrt()[:, None].expand_as(off_centre), rtol=1e-4)
    cap = decay_ceiling(env, RATIO, log_scaled=False)
    assert bool((cap[torch.isfinite(cap)] > 0).all())


def test_an_unfittable_sample_does_not_move_its_batch_mate(setting) -> None:
    """Per sample: one noise-only slice neither drops nor re-fits the slice beside it."""
    target, meas, mask = setting
    alone = _fit(meas[:1], mask[:1])
    mixed = _fit(torch.cat([meas[:1], _noise_only(target[1:]) * mask[1:]]), mask)
    expected = [RESOLUTIONS.index("fitted"), RESOLUTIONS.index("noise_floor")]
    assert mixed.resolution.tolist() == expected
    assert torch.allclose(mixed.slope[:1], alone.slope)
    assert torch.allclose(mixed.log_amp[:1], alone.log_amp)


def test_a_low_snr_small_object_at_the_top_rung_is_bounded_not_refused() -> None:
    """The shape that refused before: a small object at image SNR 5, 32x rung, rows."""
    proc = P.process(32.0)
    y, x = torch.meshgrid(torch.linspace(-1, 1, P.H), torch.linspace(-1, 1, P.W), indexing="ij")
    obj = (((y / 0.18) ** 2 + (x / 0.14) ** 2) < 1).to(torch.complex64)
    k = fft2c(obj.expand(1, P.COILS, -1, -1).contiguous())
    gen = torch.Generator().manual_seed(9)
    noise = torch.complex(torch.randn(k.shape, generator=gen), torch.randn(k.shape, generator=gen))
    k = k + 0.2 * noise / 2**0.5
    meas_full = stacked_kspace(k, torch.zeros(1), log_scaled=False)
    mask = P.mask(proc)[:1]
    env = _fit(meas_full * mask, mask)
    assert env.resolution.item() in (RESOLUTIONS.index("fitted"), RESOLUTIONS.index("flat"))


def test_the_row_axis_reproduces_the_row_only_fit(setting) -> None:
    """The bin generalization changes no arm that declares a row envelope."""
    _, meas, mask = setting
    env = _fit(meas, mask)
    phys, power, support = measured_power(meas, mask, False)
    sigma2 = NoiseAdaptiveDataConsistency().estimate_noise_variance(power, support.float()).flatten()
    count = support.sum(dim=-1)[:, 0].float()
    row_power = (power * support).sum(dim=-1)[:, 0] / count.clamp_min(1.0)
    ky = row_index(meas.shape[-2], meas.device)
    usable = (count > 0) & (ky >= 1) & (row_power > 2.0 * sigma2[:, None])
    peak = (phys * support).amax(dim=(-3, -1))
    crest = torch.where(usable, peak / row_power.clamp_min(1e-8).sqrt(), torch.zeros_like(peak))
    assert torch.equal(env.fitted_bins, usable)
    assert torch.equal(env.crest, crest.amax(dim=1))
    assert torch.equal(env.sigma2, sigma2)


def _point_mask(like: torch.Tensor, width: float = 0.25, seed: int = 0) -> torch.Tensor:
    """A gaussian 2-D point mask: some coefficients of every row, densest at the centre."""
    h, w = like.shape[-2:]
    ky = row_index(h, like.device).view(-1, 1)
    kx = row_index(w, like.device).view(1, -1)
    r = torch.sqrt(ky * ky + kx * kx)
    keep = torch.rand(h, w, generator=torch.Generator().manual_seed(seed))
    mask = (keep < torch.exp(-((r / (width * h)) ** 2))) | (r <= 3)
    return mask.float().expand(like.shape[0], 1, h, w).contiguous()


def _unmeasured_energy(x: torch.Tensor, mask: torch.Tensor) -> float:
    mag = paired_magnitude(x)
    return float((mag * mag * (1 - mask).expand_as(mag)).sum().sqrt())


def test_on_a_point_mask_the_radial_envelope_binds_where_the_row_envelope_does_not(
    setting,
) -> None:
    """PLANTED: a loud invented texture on a 2-D point mask, which every row partly measures."""
    target, _, _ = setting
    mask = _point_mask(target)
    phase = torch.rand(target.shape, generator=torch.Generator().manual_seed(3))
    loud = target * (1 + 3 * torch.cos(2 * torch.pi * phase))
    truth = _unmeasured_energy(target, mask)
    after = {}
    for axis in ("row", "radial"):
        env = _fit(target * mask, mask, axis=axis)
        out, _ = bound_bins_by_decay(loud, env, mask, ratio=1.3, log_scaled=False)
        kept, _ = bound_bins_by_decay(target, env, mask, ratio=1.3, log_scaled=False)
        after[axis] = _unmeasured_energy(out, mask) / truth
        assert _unmeasured_energy(kept, mask) / truth > 0.98
    assert _unmeasured_energy(loud, mask) / truth > 2.2
    assert after["row"] > 1.9
    assert after["radial"] < 1.5



def test_a_column_fit_is_the_row_fit_of_the_transposed_measurement(setting) -> None:
    """The envelope fits along the mask's line axis (#2160): on a column mask the
    bins are columns, and transposing the plane must give the same law."""
    _, meas, mask = setting
    # 48x64: on a square plane an H/W swap inside the column path would pass.
    meas, mask = meas[..., 8:56, :], mask[..., 8:56, :]
    rows = _fit(meas, mask)
    columns = _fit(meas.transpose(-1, -2), mask.transpose(-1, -2), axis="column")
    assert columns.axis.name == "column"
    assert torch.allclose(columns.slope, rows.slope, atol=1e-5)
    assert torch.allclose(columns.log_amp, rows.log_amp, atol=1e-5)
    assert torch.equal(columns.fitted_bins, rows.fitted_bins)
