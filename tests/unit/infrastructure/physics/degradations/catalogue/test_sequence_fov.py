"""fov_foldover: the exact periodic sum over a reduced phase-encode FOV."""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.degradations import (
    DEGRADATION_REGISTRY,
    AcquisitionGeometry,
    AcquisitionMode,
    Realisation,
    apply_degradation,
)
from spectramr.infrastructure.physics.degradations.catalogue.sequence_fov import object_extent

from .invariants import GEOMETRIES, assert_axis_invariants

M = AcquisitionMode
NAME = "fov_foldover"
VOLUME = AcquisitionGeometry(mode=M.VOLUME_3D, voxel_mm=(5.0, 1.0, 1.0))


def _apply(x, theta, geometry=None, **kw):
    return apply_degradation(NAME, x, theta, seed=1, geometry=geometry, **kw)


@pytest.mark.parametrize("geom", GEOMETRIES)
def test_invariants(geom):
    assert_axis_invariants(NAME, GEOMETRIES[geom])


def test_declaration():
    spec = DEGRADATION_REGISTRY[NAME]
    p = spec.severity.primary
    assert (p.units, p.at_theta_min, p.at_theta_max) == (
        "fraction of the object's extent",
        0.0,
        0.5,
    )
    assert spec.requires == frozenset() and spec.condition is None
    assert spec.realisation[M.MULTISLICE_2D] is Realisation.PER_SLICE_EXACT
    assert spec.realisation[M.VOLUME_3D] is Realisation.NATIVE


def _blob(n, c, sigma=3.0):
    y = torch.arange(n, dtype=torch.float32) - n // 2
    return torch.exp(-((y - c) ** 2) / (2 * sigma**2)), y


def test_extent_is_the_projection_support_above_the_stated_fraction():
    x = torch.zeros(2, 1, 32, 8)
    x[0, :, 5:15] = 1.0
    x[1, :, 9:12] = 4.0
    x[1, :, 20] = 0.3  # a lone sample below 10% of the profile maximum
    centre, ext = object_extent(x, -2)
    assert ext.tolist() == [10.0, 3.0] and centre.tolist() == [9.5, 10.0]


def test_background_noise_at_four_percent_leaves_the_extent_within_two_pixels():
    n = 64
    yy, xx = torch.meshgrid(torch.arange(n) - n // 2, torch.arange(n) - n // 2, indexing="ij")
    disc = ((xx**2 + (yy - 5) ** 2) < 20**2).float()
    clean = object_extent(disc[None, None], -2)
    g = torch.Generator().manual_seed(0)
    noise = 0.04 * torch.randn(1, 1, n, n, 2, generator=g).norm(dim=-1)
    noisy = object_extent((disc + noise)[:, :], -2)
    assert abs(float(noisy[1] - clean[1])) <= 2 and abs(float(noisy[0] - clean[0])) <= 2


def test_blob_outside_the_window_reappears_at_y0_minus_fov_with_the_same_intensity():
    n, theta, c = 64, 1.0, 8.0
    g, y = _blob(n, c)
    x = g.view(1, 1, n, 1).expand(1, 1, n, 4).contiguous()
    centre, ext = object_extent(x, -2)
    fov = float(ext) * (1 - 0.5 * theta)
    assert fov != round(fov)  # a non-integer FOV'
    out = _apply(x, theta)[0, 0, :, 0]
    want = sum(_blob(n, c - k * fov)[0] for k in range(-2, 3))
    c0 = float(centre) - n // 2  # window centre relative to the matrix centre
    inside = (y >= c0 - fov / 2) & (y < c0 + fov / 2)
    torch.testing.assert_close(out[inside], want[inside], rtol=0, atol=2e-3)
    assert float(out[~inside].abs().max()) == 0.0
    assert float(out.max()) >= 1.0  # the unit blob plus the tails of its other replicas


def test_an_off_centre_object_never_vanishes_and_wraps_as_predicted():
    n, c = 128, 40.0
    g, y = _blob(n, c, sigma=4.0)
    x = g.view(1, 1, n, 1).expand(1, 1, n, 4).contiguous()
    centre, ext = object_extent(x, -2)
    assert float(centre) - n // 2 == pytest.approx(c, abs=0.5)
    for theta in (0.3, 1.0):
        out = _apply(x, theta)[0, 0, :, 0]
        fov = float(ext) * (1 - 0.5 * theta)
        want = sum(_blob(n, c - k * fov, sigma=4.0)[0] for k in range(-2, 3))
        c0 = float(centre) - n // 2
        inside = (y >= c0 - fov / 2) & (y < c0 + fov / 2)
        assert float(out.abs().max()) > 0.9
        torch.testing.assert_close(out[inside], want[inside], rtol=0, atol=2e-3)


def _rect(n=128):
    x = torch.zeros(1, 1, n, 16)
    x[..., 70:94, 3:13] = 1.0  # a compact object, off-centre along PE
    return x


def _gauss(n=128):
    g, _ = _blob(n, 20.0, sigma=4.0)
    return g.view(1, 1, n, 1).expand(1, 1, n, 16).contiguous()


def test_rectangle_effect_is_continuous_as_theta_goes_to_zero():
    x = _rect()
    big = float((_apply(x, 1.0) - x).norm())
    assert float((_apply(x, 1e-4) - x).norm()) < 1e-3 * big


def test_smooth_object_jumps_at_theta_zero_plus_by_the_wrapped_sub_threshold_tails():
    x = _gauss()  # sigma = 4 px Gaussian centred at +20 px, 128 x 16
    centre, ext = object_extent(x, -2)
    big = float((_apply(x, 1.0) - x).norm())
    out = _apply(x, 1e-4)
    jump = float((out - x).norm()) / big
    assert jump == pytest.approx(0.133, rel=0.2)  # measured; the notes quote 13%
    fov = float(ext) * (1 - 0.5e-4)
    inside = (torch.arange(x.shape[-2]) - float(centre) >= -fov / 2) & (
        torch.arange(x.shape[-2]) - float(centre) < fov / 2
    )
    assert float(out[0, 0, ~inside].abs().max()) == 0.0
    cropped = float(x[0, 0, ~inside].norm())  # energy beyond the 10% window
    wrapped = float((out - x)[0, 0, inside].norm())  # what came back in
    assert wrapped == pytest.approx(cropped, rel=0.05)  # wrapped in, not lost


def test_extent_is_per_sample_and_scale_free():
    big, _ = _blob(64, 0.0, sigma=6.0)
    small, _ = _blob(64, 0.0, sigma=2.0)
    x = torch.stack([big, small]).view(2, 1, 64, 1).expand(2, 1, 64, 4).contiguous()
    both = _apply(x, 1.0)
    for i in range(2):
        torch.testing.assert_close(both[i : i + 1], _apply(x[i : i + 1], 1.0), rtol=0, atol=1e-5)
        torch.testing.assert_close(
            _apply(1e3 * x[i : i + 1], 1.0), 1e3 * both[i : i + 1], rtol=1e-4, atol=1e-2
        )


def test_real_input_stays_real():
    assert not _apply(torch.rand(1, 1, 16, 16), 0.5).is_complex()


def test_kz_fold_uses_the_slab_extent_and_both_composes():
    d = 40
    g, _ = _blob(d, 6.0, sigma=2.0)
    x = g.view(1, 1, d, 1, 1).expand(1, 1, d, 6, 6).contiguous()
    sl = _apply(x, 1.0, VOLUME, axes="slice")
    centre, ext = object_extent(x, -3)
    fov = float(ext) * 0.5
    zz = torch.arange(d) - float(centre)
    assert float(sl[0, 0, :, 3, 3][(zz < -fov / 2) | (zz >= fov / 2)].abs().max()) == 0.0
    assert float(sl.abs().max()) > 0.5
    both = _apply(x, 1.0, VOLUME, axes="both")
    assert float(both.abs().max()) > 0.5 and not torch.equal(both, sl)


@pytest.mark.parametrize("geometry", [None, GEOMETRIES["ms_interleaved"]])
@pytest.mark.parametrize("axes", ["slice", "both"])
def test_kz_fold_is_refused_where_kz_is_not_encoded(geometry, axes):
    x = torch.zeros(1, 1, 8, 8) if geometry is None else torch.zeros(1, 1, 4, 8, 8)
    with pytest.raises(ValueError, match="kz"):
        _apply(x, 0.5, geometry, axes=axes)


def test_unknown_axes_raises():
    with pytest.raises(ValueError, match="axes"):
        _apply(torch.zeros(1, 1, 8, 8), 0.5, axes="ro")


def test_multislice_stack_shares_one_prescription():
    g = GEOMETRIES["ms_interleaved"]
    x = torch.zeros(1, 1, 3, 48, 16)
    x[:, :, 0, 10:30] = 1.0
    x[:, :, 1, 14:26] = 2.0
    x[:, :, 2, 18:22] = 3.0
    out = _apply(x, 1.0, g)
    support = [(out[0, 0, s] != 0).any(-1) for s in range(3)]
    assert torch.equal(support[0], support[1]) and torch.equal(support[0], support[2])
    _, ext = object_extent(x.reshape(1, 3, 48, 16), -2)  # the stack, slices as channels
    fov = float(ext) * 0.5
    assert abs(int(support[0].sum()) - fov) <= 2
