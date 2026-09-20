"""Unit tests for the radial amplitude-transfer probe.

Every test here plants a model whose radial transfer is known in closed form and
asserts the probe reports it. The load-bearing ones are the **near misses**: a
uniform gain and a uniform attenuation are flat operators that a probe with a
broken normalisation reads as an amplifier and a filter, and a probe that pooled
the k-space corners into the last bin would read a shape that the operator does
not have. A detector is only a detector for the shape it has been watched fail
on (non-negotiable 15), so those shapes are committed rather than eyeballed.
"""

from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from spectramr.infrastructure.validation.spectral_transfer_probe import (
    RadialTransfer,
    measure_radial_transfer,
    radial_bins,
)

# Small grid, few bins, two draws: the whole file runs in about a second on CPU.
KW = {"size": 64, "channels": 4, "n_bins": 16, "repeats": 2, "seed": 0, "device": "cpu"}


class _Gain(nn.Module):
    """All-pass with a uniform gain -- flat transfer, non-unit scale."""

    def __init__(self, k: float) -> None:
        super().__init__()
        self.k = float(k)

    def forward(self, x, timesteps=None):
        return x * self.k


class _RadialShape(nn.Module):
    """Multiplies each coefficient by ``scale(r)`` on the centred frequency grid."""

    def __init__(self, inner: float, outer: float, cutoff: float = 0.5) -> None:
        super().__init__()
        self.inner, self.outer, self.cutoff = float(inner), float(outer), float(cutoff)

    def forward(self, x, timesteps=None):
        bins = 512
        index, _inside, _edges = radial_bins(x.shape[2], x.shape[3], bins, x.device)
        radius = (index.float() + 0.5) / bins
        gain = torch.where(radius <= self.cutoff, self.inner, self.outer)
        return x * gain.to(x.dtype)


class _PerCoefficientGain(nn.Module):
    """Deterministic gains that vary within a bin, so the reading tracks the draw."""

    def __init__(self, size: int) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(11)
        self.register_buffer("gain", torch.rand(size, size, generator=g))

    def forward(self, x, timesteps=None):
        return x * self.gain


class _NoTimestep(nn.Module):
    """A generator whose forward takes the batch only."""

    def forward(self, x):
        return x


class _RecordsTimestep(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen: list[int] = []

    def forward(self, x, timesteps=None):
        self.seen.append(int(timesteps[0]))
        return x


def _measure(model, **over):
    return measure_radial_transfer(model, **{**KW, **over})


# -- the flat reference --------------------------------------------------------
def test_identity_reads_flat_at_one():
    rt = _measure(_Gain(1.0))
    assert rt.dc_gain == pytest.approx(1.0, abs=1e-5)
    assert max(abs(v - 1.0) for v in rt.ratios) < 1e-5
    assert rt.outer_band_retention == pytest.approx(1.0, abs=1e-5)
    assert rt.outer_band_floor == pytest.approx(1.0, abs=1e-5)


# -- planted violation: a genuine architectural low-pass -----------------------
def test_hard_radial_low_pass_is_detected():
    rt = _measure(_RadialShape(inner=1.0, outer=0.0, cutoff=0.25))
    assert rt.outer_band_retention < 1e-9
    assert rt.outer_band_floor < 1e-9
    # The inner band is untouched: the finding is a SHAPE, not a dead model.
    inner = [r for r, c in zip(rt.ratios, rt.bin_centers, strict=True) if c < 0.25]
    assert min(inner) == pytest.approx(1.0, abs=1e-5)


def test_partial_roll_off_lands_between_the_extremes():
    rt = _measure(_RadialShape(inner=1.0, outer=0.3, cutoff=0.5))
    assert rt.outer_band_retention == pytest.approx(0.3, abs=1e-5)


# -- planted near-miss: a gain is not a filter ---------------------------------
def test_uniform_gain_is_not_mistaken_for_a_low_pass():
    rt = _measure(_Gain(3.0))
    assert max(abs(v - 3.0) for v in rt.ratios) < 1e-5  # the gain stays visible
    assert rt.outer_band_retention == pytest.approx(1.0, abs=1e-5)
    assert rt.outer_band_floor == pytest.approx(1.0, abs=1e-5)


def test_uniform_attenuation_is_not_mistaken_for_a_low_pass():
    rt = _measure(_Gain(0.1))
    assert max(abs(v - 0.1) for v in rt.ratios) < 1e-6
    assert rt.outer_band_retention == pytest.approx(1.0, abs=1e-5)


def test_dc_normalisation_is_the_step_that_separates_gain_from_filter():
    """The mutation this file exists to catch: drop ``/ dc_gain``.

    A probe that summarised the raw outer mean would rank a 3x all-pass as the
    least low-passing arm in a cohort by a factor of three, and a 0.1x all-pass
    as the most low-passing -- on two operators that are equally flat.
    """
    gain, attenuation = _measure(_Gain(3.0)), _measure(_Gain(0.1))
    raw_outer = [
        sum(r for r, c in zip(rt.ratios, rt.bin_centers, strict=True) if c > 0.5)
        / sum(1 for c in rt.bin_centers if c > 0.5)
        for rt in (gain, attenuation)
    ]
    assert raw_outer[0] == pytest.approx(3.0, abs=1e-5)
    assert raw_outer[1] == pytest.approx(0.1, abs=1e-6)
    assert gain.outer_band_retention == pytest.approx(attenuation.outer_band_retention, abs=1e-5)


def test_high_pass_reads_above_one_rather_than_as_a_low_pass():
    rt = _measure(_RadialShape(inner=0.5, outer=1.0, cutoff=0.5))
    assert rt.outer_band_retention == pytest.approx(2.0, abs=1e-4)


def test_a_model_that_annihilates_dc_raises_instead_of_dividing_by_zero():
    with pytest.raises(ValueError, match="cannot be normalised"):
        _measure(_RadialShape(inner=0.0, outer=1.0, cutoff=0.25))


# -- the grid the ratios are binned over ---------------------------------------
def test_radial_grid_is_centred_at_the_fftshift_dc():
    index, inside, edges = radial_bins(64, 64, 16, "cpu")
    assert index[32, 32].item() == 0  # DC sits at N // 2
    assert bool(inside[32, 32])
    assert edges[0].item() == pytest.approx(0.0)
    assert edges[-1].item() == pytest.approx(1.0)
    # The on-axis edge is exactly Nyquist; the corner is sqrt(2) away and outside.
    assert bool(inside[0, 32]) and bool(inside[32, 0])
    assert not bool(inside[0, 0])


def test_corner_coefficients_are_excluded_rather_than_pooled_into_the_last_bin():
    """Planted violation: pooling the corners inflates the outer bins' population.

    An annulus that exists only along the diagonals is a different population
    from the bins below it. If the exclusion were dropped, ``sum(counts)`` would
    equal ``H * W``.
    """
    rt = _measure(_Gain(1.0))
    assert sum(rt.counts) < 64 * 64
    assert sum(rt.counts) == pytest.approx(3.1416 / 4 * 64 * 64, rel=0.02)
    assert min(rt.counts) > 0


def test_bins_finer_than_the_grid_raise_rather_than_reporting_an_empty_bin():
    with pytest.raises(ValueError, match="empty"):
        _measure(_Gain(1.0), size=8, n_bins=64)


# -- calling convention --------------------------------------------------------
def test_timesteps_are_passed_when_the_forward_accepts_them():
    model = _RecordsTimestep()
    _measure(model, timestep=7, repeats=3)
    assert model.seen == [7, 7, 7]


def test_a_forward_without_timesteps_is_called_positionally():
    rt = _measure(_NoTimestep(), timestep=7)
    assert rt.outer_band_retention == pytest.approx(1.0, abs=1e-5)


def test_a_non_tensor_return_raises():
    class _Tuple(nn.Module):
        def forward(self, x, timesteps=None):
            return (x, x)

    with pytest.raises(TypeError, match="single output tensor"):
        _measure(_Tuple())


def test_trailing_dims_are_honoured():
    rt = _measure(_Gain(1.0), trailing=(1,))
    assert rt.shape == (1, 4, 64, 64, 1)
    assert rt.outer_band_retention == pytest.approx(1.0, abs=1e-5)


def test_odd_channel_width_raises():
    with pytest.raises(ValueError, match="even channel"):
        _measure(_Gain(1.0), channels=3)


def test_repeats_must_be_positive():
    with pytest.raises(ValueError, match="repeats"):
        _measure(_Gain(1.0), repeats=0)


def test_n_bins_below_two_raises():
    with pytest.raises(ValueError, match="n_bins"):
        _measure(_Gain(1.0), n_bins=1)


def test_the_model_is_left_in_the_mode_it_arrived_in():
    model = _Gain(1.0).train()
    _measure(model)
    assert model.training


def test_the_seed_pins_the_reading_against_the_ambient_rng():
    """Draws come from a dedicated generator, so ambient seeding cannot move them.

    The model has per-coefficient gains, which is what makes a bin's ratio a
    draw-weighted average and so makes the seed observable at all.
    """
    model = _PerCoefficientGain(64)
    torch.manual_seed(1234)
    first = _measure(model, seed=3)
    torch.manual_seed(4321)
    again = _measure(model, seed=3)
    other = _measure(model, seed=99)
    assert first.ratios == again.ratios
    assert other.ratios != first.ratios


def test_the_record_round_trips_through_json():
    rt = _measure(_Gain(1.0))
    payload = json.loads(json.dumps(rt.to_dict(), allow_nan=False))
    assert payload["repeats"] == 2
    assert payload["seed"] == 0
    assert len(payload["normalized"]) == 16
    assert isinstance(rt, RadialTransfer)


# -- the real backbone ---------------------------------------------------------
def test_complex_unet_rms_norm_scores_far_below_none():
    """``kspace_feature_norm: rms`` low-passes the untrained backbone; ``none`` does not.

    ``KSpacePad`` up-samples by zero-padding k-space, which leaves the map's
    energy unchanged while quadrupling its element count, so the global
    ``ComplexRMSNorm`` divisor falls by 2 and the trunk gains 2x over the flat
    full-resolution skip at every up-step. Both nets are built from the same
    seed, so the only difference between the two readings is that norm.
    """
    from spectramr.models.generators.complex_unet import ComplexUNet

    def build(norm: str) -> ComplexUNet:
        torch.manual_seed(0)
        return ComplexUNet(
            in_channels=8,
            out_channels=8,
            features=(8, 16, 32),
            feature_domain="kspace",
            attention_type="none",
            kspace_feature_norm=norm,
        )

    rms = _measure(build("rms"), channels=8, n_bins=12)
    none = _measure(build("none"), channels=8, n_bins=12)

    assert rms.outer_band_retention < 0.5
    assert none.outer_band_retention > 0.8
    assert rms.outer_band_retention < 0.6 * none.outer_band_retention
    # The roll-off is a shape, not an overall attenuation: DC survives in both.
    assert rms.normalized[0] == pytest.approx(1.0)
