"""Closed-form physics of gain drift, ADC clipping, ADC quantisation and coil dropout."""

from __future__ import annotations

import itertools
import math

import pytest
import torch

from spectramr.infrastructure.physics.degradations import (
    AcquisitionGeometry,
    AcquisitionMode,
    apply_degradation,
)
from spectramr.infrastructure.physics.degradations.catalogue.receive_adc import (
    adc_enob,
    apply_line_gain,
    drift_gain_map,
)
from spectramr.infrastructure.physics.degradations.catalogue.receive_common import (
    ENCODED_DIMS,
    acquisition_ranks,
    control_generator,
)
from spectramr.infrastructure.physics.degradations.geometry import LineOrder, SliceOrder
from spectramr.infrastructure.physics.fft_ops import fftnc, ifftnc
from spectramr.infrastructure.physics.per_sample_rng import per_sample_rand

M = AcquisitionMode
GEOMS = {
    "slice": None,
    "ms_interleaved": AcquisitionGeometry(mode=M.MULTISLICE_2D, voxel_mm=(5.0, 1.0, 1.0)),
    "ms_sequential": AcquisitionGeometry(
        mode=M.MULTISLICE_2D, voxel_mm=(5.0, 1.0, 1.0), slice_order=SliceOrder.SEQUENTIAL
    ),
    "vol_linear": AcquisitionGeometry(mode=M.VOLUME_3D, voxel_mm=(5.0, 1.0, 1.0)),
    "vol_centric": AcquisitionGeometry(
        mode=M.VOLUME_3D, voxel_mm=(5.0, 1.0, 1.0), line_order=LineOrder.CENTRIC
    ),
}


def _dims(geom):
    return ENCODED_DIMS[M.SLICE_2D if geom is None else geom.mode]


def _rand(geom, c=2, d=6, h=16, w=16, seed=1):
    shape = (1, c, h, w) if geom is None else (1, c, d, h, w)
    g = torch.Generator().manual_seed(seed)
    return torch.complex(torch.randn(shape, generator=g), torch.randn(shape, generator=g))


def _as5(t):
    return t.unsqueeze(2) if t.dim() == 4 else t


def _ranks(geom, x5):
    mode_geom = geom or AcquisitionGeometry(mode=M.SLICE_2D, voxel_mm=None)
    return acquisition_ranks(mode_geom, x5.shape[2], x5.shape[3], torch.device("cpu"))


# --- receiver_gain_drift ---------------------------------------------------------------------


def test_a_constant_gain_is_a_pure_scaling_and_no_drift_is_the_identity() -> None:
    x = _rand(GEOMS["vol_linear"])
    gain = torch.full((1, 1, 6, 16, 1), 1.07)
    torch.testing.assert_close(apply_line_gain(x, gain, (-3, -2, -1)), 1.07 * x, rtol=0, atol=2e-6)
    tau = torch.rand(6, 16)
    unit = drift_gain_map(tau, 0.0, torch.tensor([0.0]), torch.tensor([0.5]))
    assert torch.equal(unit, torch.ones_like(unit))
    torch.testing.assert_close(apply_line_gain(x, unit, (-3, -2, -1)), x, rtol=0, atol=2e-6)


def test_gain_map_exact_per_line_case() -> None:
    tau = torch.tensor([[0.0, 0.25, 0.5, 0.75]])  # H = 4: the centre line is index 2
    g = drift_gain_map(tau, 0.2, torch.tensor([1.0]), torch.tensor([0.5]))[0, 0, 0, :, 0]
    # raw: (1 + 0.2 tau) 2^[tau >= 0.5] = 1.0, 1.05, 2.2, 2.3; divided by the centre's 2.2
    torch.testing.assert_close(g, torch.tensor([1.0, 1.05, 2.2, 2.3]) / 2.2, rtol=0, atol=1e-6)
    assert float(g[2]) == 1.0


def test_gain_step_is_a_factor_of_two_at_full_severity() -> None:
    x = _rand(None, c=1, h=64, w=8)
    out = apply_degradation("receiver_gain_drift", x, 1.0, seed=2)
    ratio = (fftnc(out, (-2, -1)) / fftnc(x, (-2, -1))).real[0, 0, :, 0]
    # Drift is at most 5 %, so the two plateaus differ by 6 dB up to that drift.
    jump = float(ratio.max() / ratio.min())
    assert jump == pytest.approx(2.0, rel=0.1)


def test_gain_map_centre_reference_per_slice_and_per_volume() -> None:
    tau = torch.rand(4, 6)
    args = (0.05, torch.tensor([-1.0]), torch.tensor([0.4]))
    per_slice = drift_gain_map(tau, *args, per_slice=True)[0, 0, :, :, 0]
    volume = drift_gain_map(tau, *args, per_slice=False)[0, 0, :, :, 0]
    assert torch.equal(per_slice[:, 3], torch.ones(4))  # every slice's centre line
    assert float(volume[2, 3]) == 1.0  # the (kz, ky) centre only
    assert not torch.equal(volume[:, 3], torch.ones(4))


@pytest.mark.parametrize("geom", GEOMS.values(), ids=GEOMS.keys())
def test_gain_drift_is_the_centre_referenced_map_on_every_line(geom) -> None:
    theta, seed = 0.8, 6
    x = _rand(geom, c=2, d=6, h=16, w=16)
    out = apply_degradation("receiver_gain_drift", x, theta, seed=seed, geometry=geom)
    dims = _dims(geom)
    x5, out5 = _as5(x), _as5(out)
    ratio = fftnc(out5, dims) / fftnc(x5, dims)  # exactly the per-line gain
    gain = ratio.real[0, 0, :, :, 0]
    assert float(ratio.imag.abs().max()) < 1e-4
    torch.testing.assert_close(  # the same gain on every coil and every readout sample
        ratio.real, gain[None, None, :, :, None].expand_as(ratio.real), rtol=0, atol=2e-4
    )
    _, tau = _ranks(geom, x5)
    u = per_sample_rand(1, 2, generator=control_generator(torch.device("cpu"), seed))
    sign = torch.where(u[:, 0] < 0.5, -1.0, 1.0)
    per_slice = geom is None or geom.mode is not M.VOLUME_3D
    expected = drift_gain_map(tau, theta * 0.05, sign * theta, u[:, 1], per_slice=per_slice)
    torch.testing.assert_close(gain, expected[0, 0, :, :, 0], rtol=0, atol=3e-4)
    d, h = gain.shape
    centre = gain[:, h // 2] if per_slice else gain[d // 2, h // 2]
    assert float((centre - 1.0).abs().max()) < 3e-4  # the image scale is untouched


# --- adc_clipping ----------------------------------------------------------------------------


@pytest.mark.parametrize("geom", GEOMS.values(), ids=GEOMS.keys())
def test_adc_clipping_bounds_the_samples_and_leaves_the_rest(geom) -> None:
    theta, tol = 0.6, 2e-5
    x = torch.cat([_rand(geom, seed=2), 10.0 * _rand(geom, seed=3)], dim=0)
    out = apply_degradation("adc_clipping", x, theta, geometry=geom)
    dims = _dims(geom)
    k_in, k_out = fftnc(_as5(x), dims), fftnc(_as5(out), dims)
    peak = torch.maximum(
        k_in.real.abs().amax(dim=(1, 2, 3, 4)), k_in.imag.abs().amax(dim=(1, 2, 3, 4))
    )
    level = ((1.0 - 0.9 * theta) * peak).view(-1, 1, 1, 1, 1)
    for part_in, part_out in ((k_in.real, k_out.real), (k_in.imag, k_out.imag)):
        assert bool((part_out.abs() <= level * (1 + tol) + tol).all())
        below = part_in.abs() < level - 1e-3
        assert bool(below.any())
        assert float((part_out - part_in)[below].abs().max()) < 1e-3 * 1e-1
        assert bool((part_in.abs() > level + 1e-3).any())  # something was actually clipped
        assert float((part_out - part_in.clamp(-level, level)).abs().max()) < 3e-5 * float(
            peak.max()
        )


# --- adc_quantization ------------------------------------------------------------------------


def test_adc_enob_is_continuous() -> None:
    assert [adc_enob(t) for t in (0.0, 0.5, 1.0)] == [16.0, 10.0, 4.0]
    assert adc_enob(0.125) == pytest.approx(14.5)


@pytest.mark.parametrize("geom", GEOMS.values(), ids=GEOMS.keys())
@pytest.mark.parametrize("theta", [0.55, 0.9, 1.0])  # ENOB 9.4, 5.2, 4: non-integer included
def test_adc_quantisation_error_is_bounded_and_uniform(geom, theta) -> None:
    dims = _dims(geom)
    g = torch.Generator().manual_seed(11)
    shape = (1, 2, 64, 64) if geom is None else (1, 2, 8, 32, 32)
    k = torch.complex(
        torch.rand(shape, generator=g) * 2 - 1, torch.rand(shape, generator=g) * 2 - 1
    )
    x = ifftnc(_as5(k), dims)
    x = x.squeeze(2) if geom is None else x
    out = apply_degradation("adc_quantization", x, theta, geometry=geom)
    k_in, k_out = fftnc(_as5(x), dims), fftnc(_as5(out), dims)
    peak = float(torch.maximum(k_in.real.abs().max(), k_in.imag.abs().max()))
    step = peak / 2.0 ** (adc_enob(theta) - 1.0)
    err = torch.cat([(k_out - k_in).real.flatten(), (k_out - k_in).imag.flatten()])
    assert float(err.abs().max()) <= step / 2 * (1 + 1e-3) + 1e-5
    assert float(err.var()) == pytest.approx(step**2 / 12.0, rel=0.06)
    levels = k_out.real / step
    assert float((levels - levels.round()).abs().max()) < 2e-2  # on the mid-tread grid
    assert float(k_out.real.abs().min()) < step  # zero is a level (mid-tread, not mid-rise)


def test_adc_quantisation_effect_is_strictly_monotone_at_every_step() -> None:
    x = _rand(None, c=2, h=32, w=32)
    effect = [
        float((apply_degradation("adc_quantization", x, t) - x).abs().pow(2).sum())
        for t in torch.linspace(0.05, 1.0, 20).tolist()
    ]
    assert all(a < b for a, b in itertools.pairwise(effect)), effect


# --- coil_dropout ----------------------------------------------------------------------------


@pytest.mark.parametrize("geom", GEOMS.values(), ids=GEOMS.keys())
def test_full_dropout_zeroes_whole_coils_and_leaves_the_rest_exact(geom) -> None:
    x = _rand(geom, c=4)
    out = apply_degradation("coil_dropout", x, 1.0, seed=3, geometry=geom)
    dead = out.abs().flatten(2).amax(dim=2)[0] == 0
    assert int(dead.sum()) == 2  # fraction 0.5 of 4 coils
    assert torch.equal(out[:, ~dead], x[:, ~dead])
    assert (
        int(
            apply_degradation("coil_dropout", x, 0.3, seed=3, geometry=geom)
            .abs()
            .flatten(2)
            .amax(dim=2)
            .eq(0)
            .sum()
        )
        == 1
    )  # ceil(0.15 * 4)


@pytest.mark.parametrize("geom", GEOMS.values(), ids=GEOMS.keys())
def test_partial_dropout_zeroes_the_dead_coils_from_a_seeded_line(geom) -> None:
    x = _rand(geom, c=4, d=6, h=16)
    out = apply_degradation("coil_dropout", x, 1.0, seed=3, geometry=geom, mode="partial")
    dims = _dims(geom)
    x5, out5 = _as5(x), _as5(out)
    k_in, k_out = fftnc(x5, dims), fftnc(out5, dims)
    dead = (out5 - x5).abs().flatten(2).amax(dim=2)[0] > 1e-5
    assert int(dead.sum()) == 2
    assert torch.equal(out5[:, ~dead], x5[:, ~dead])
    zeroed = k_out[:, dead].abs().amax(dim=(1, -1))[0] < 1e-4  # [D, H]
    kept = (k_out[:, dead] - k_in[:, dead]).abs().amax(dim=(1, -1))[0] < 1e-4
    assert bool((zeroed ^ kept).all())
    rank, _ = _ranks(geom, x5)
    assert int(zeroed.sum()) >= 1
    assert int(rank[zeroed].min()) + int(zeroed.sum()) == int(rank[zeroed].max()) + 1
    assert int(rank[zeroed].max()) == rank.numel() - 1  # a suffix of the acquisition order


def test_dropout_set_is_nested_in_theta() -> None:
    x = _rand(None, c=6)
    sets = []
    for theta in (0.2, 0.6, 1.0):
        out = apply_degradation("coil_dropout", x, theta, seed=1)
        sets.append(out.abs().flatten(2).amax(dim=2)[0] == 0)
    assert bool((sets[0] <= sets[1]).all()) and bool((sets[1] <= sets[2]).all())
    assert [int(s.sum()) for s in sets] == [min(math.ceil(t * 0.5 * 6), 3) for t in (0.2, 0.6, 1.0)]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"mode": "half"}, "mode must be"),
        ({"max_fraction": 0.8}, "max_fraction"),
        ({"n_coils": 8}, "n_coils=8"),
    ],
)
def test_bad_dropout_knobs_raise_even_at_theta_zero(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        apply_degradation("coil_dropout", _rand(None, c=4), 0.0, **kwargs)


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [("receiver_gain_drift", {"drift_max": 0.0}), ("adc_clipping", {"depth": 1.0})],
)
def test_bad_amplitude_knobs_raise(name, kwargs) -> None:
    with pytest.raises(ValueError):
        apply_degradation(name, _rand(None), 0.5, **kwargs)


def test_full_dropout_returns_complex_for_real_input() -> None:
    out = apply_degradation("coil_dropout", torch.randn(1, 4, 8, 8), 0.5, seed=1)
    assert out.is_complex()
