"""The radial-annulus partition has one owner, and it is the probe's own.

A band mechanism graded by a band probe must partition identically or the two
numbers are not about the same thing. Three implementations of "annulus" already
existed; this pins that the probe reads the owner rather than keeping a fourth.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from spectramr.infrastructure.physics.radial_bands import (  # noqa: E402
    band_counts,
    band_reduce,
    radial_bins,
)


def test_the_probe_reads_this_owner_rather_than_its_own_copy():
    """One definition of an annulus, not two that agree today."""
    from spectramr.infrastructure.validation import spectral_transfer_probe as probe

    assert probe.radial_bins is radial_bins


def test_index_and_edges_have_the_declared_shapes():
    index, inside, edges = radial_bins(32, 32, 8, "cpu")
    assert index.shape == (32, 32)
    assert inside.shape == (32, 32)
    assert edges.shape == (9,)
    assert index.max().item() <= 7


def test_dc_sits_in_the_first_bin():
    """``r = 0`` is at index ``N // 2``; if that moves every band shifts."""
    index, _inside, _edges = radial_bins(64, 64, 8, "cpu")
    assert index[32, 32].item() == 0


def test_corners_are_outside_the_nyquist_disc():
    """Pooling the diagonals into the last bin mixes two populations."""
    _index, inside, _edges = radial_bins(32, 32, 8, "cpu")
    assert not bool(inside[0, 0])
    assert bool(inside[16, 16])


def test_the_disc_is_radially_symmetric():
    index, inside, _edges = radial_bins(64, 64, 8, "cpu")
    masked = torch.where(inside, index, torch.full_like(index, -1))
    assert torch.equal(masked, torch.flip(masked, dims=[0]).roll(1, dims=0))


@pytest.mark.parametrize("n_bins", [0, 1, -3])
def test_fewer_than_two_bins_raises(n_bins):
    """One annulus is a global scalar; nothing downstream would tell them apart."""
    with pytest.raises(ValueError, match="n_bins must be >= 2"):
        radial_bins(32, 32, n_bins, "cpu")


def test_band_counts_sum_to_the_disc():
    index, inside, _edges = radial_bins(32, 32, 8, "cpu")
    counts = band_counts(index, inside, 8)
    assert int(counts.sum()) == int(inside.sum())


def test_band_counts_expose_an_empty_annulus():
    """The failure a caller must check for: a band fitted on no bins."""
    index, inside, _edges = radial_bins(4, 4, 64, "cpu")
    assert int((band_counts(index, inside, 64) == 0).sum()) > 0


def test_band_reduce_sums_only_inside_the_disc():
    index, inside, _edges = radial_bins(16, 16, 4, "cpu")
    ones = torch.ones(16, 16)
    assert float(band_reduce(ones, index, inside, 4).sum()) == pytest.approx(float(inside.sum()))


def test_band_reduce_refuses_a_complex_field():
    """Complex values cancel across an annulus and report signal as near-zero."""
    index, inside, _edges = radial_bins(16, 16, 4, "cpu")
    with pytest.raises(ValueError, match="real per-bin quantity"):
        band_reduce(torch.randn(16, 16, dtype=torch.complex64), index, inside, 4)


def test_the_grid_matches_fftshifted_fftfreq():
    """The coordinate convention, pinned against the transform it imitates.

    Written from ``arange`` so this module is not a second FFT owner, which only
    holds if the two really agree.
    """
    n = 32
    expected = torch.fft.fftshift(torch.fft.fftfreq(n)) * 2
    got = (torch.arange(n, dtype=torch.float32) - n // 2) / (n / 2.0)
    assert torch.allclose(got, expected.float(), atol=1e-6)
