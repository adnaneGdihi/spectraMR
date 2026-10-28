"""gfactor_noise and pi_residual_aliasing: the g-factor noise law and the miscalibrated-unfolding residual."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
import torch

from spectramr.infrastructure.physics.degradations import (
    DEGRADATION_REGISTRY,
    AcquisitionGeometry,
    AcquisitionMode,
    MissingInput,
    Realisation,
    apply_degradation,
)
from spectramr.infrastructure.physics.degradations.catalogue.recon_sense import (
    gfactor_noise_volume,
    residual_image,
    sense_setup,
)
from spectramr.infrastructure.physics.degradations.catalogue.sense_core import gfactor
from spectramr.infrastructure.physics.scanner_presets import FASTMRI_3T, M4RAW_03T

from .invariants import GEOMETRIES, assert_axis_invariants

M = AcquisitionMode
SLICE = AcquisitionGeometry(mode=M.SLICE_2D, voxel_mm=None)
VOLUME = AcquisitionGeometry(mode=M.VOLUME_3D, voxel_mm=(5.0, 1.0, 1.0))


def _ones(c=4, d=1, h=64, w=64):
    return torch.ones(1, c, d, h, w, dtype=torch.complex64)


@pytest.mark.parametrize("name", ["gfactor_noise", "pi_residual_aliasing"])
@pytest.mark.parametrize("mode", list(M))
def test_every_mode_is_native_and_gated_on_acceleration(name, mode):
    spec = DEGRADATION_REGISTRY[name]
    assert spec.realisation[mode] is Realisation.NATIVE
    assert {str(g) for g in spec.active_when} == {"accelerated"}
    assert spec.condition is not None


def test_pi_residual_requires_coils_and_gfactor_does_not():
    assert {str(r) for r in DEGRADATION_REGISTRY["pi_residual_aliasing"].requires} == {"multicoil"}
    assert DEGRADATION_REGISTRY["gfactor_noise"].requires == frozenset()
    with pytest.raises(MissingInput):
        apply_degradation(
            "pi_residual_aliasing", torch.ones(1, 1, 16, 16, dtype=torch.complex64), 0.5
        )


@pytest.mark.parametrize("name", ["gfactor_noise", "pi_residual_aliasing"])
@pytest.mark.parametrize("geom", GEOMETRIES)
def test_invariants(name, geom):
    assert_axis_invariants(name, GEOMETRIES[geom])


def test_noise_variance_map_is_sigma_g_sqrt_r_squared():
    """E|n|^2 = (sigma_0 g sqrt(R))^2 per voxel (Pruessmann 1999): measured over 4 channels x 64^2."""
    x = _ones()
    out = gfactor_noise_volume(x, 1.0, 3, SLICE, acceleration=(2.0, 1.0), n_coils=8)
    st = sense_setup(
        x,
        8,
        SLICE,
        3,
        name="t",
        acceleration=(2.0, 1.0),
        caipi_shift=None,
        coil_maps=None,
        coil_covariance=None,
    )
    expect = (0.1 * st.crop(gfactor(st.csm, st.plan, None)) * math.sqrt(2)) ** 2
    ratio = (out - x).abs().pow(2).mean(1, keepdim=True) / expect
    assert abs(float(ratio.mean()) - 1.0) < 0.05
    top = expect > expect.median()  # the amplified half carries more noise than the rest
    assert float((out - x).abs().pow(2)[:, :, top[0, 0]].mean()) > float(
        (out - x).abs().pow(2)[:, :, ~top[0, 0]].mean()
    )


def test_unaccelerated_noise_is_plain_sigma_everywhere():
    x = _ones()
    n = (gfactor_noise_volume(x, 1.0, 3, SLICE, acceleration=(1.0, 1.0)) - x).abs().pow(2).mean()
    assert abs(float(n) / 0.1**2 - 1.0) < 0.05


def test_noise_is_referenced_to_the_signal_level():
    x = _ones()
    a = gfactor_noise_volume(x, 0.7, 2, SLICE) - x
    b = gfactor_noise_volume(50.0 * x, 0.7, 2, SLICE) - 50.0 * x
    torch.testing.assert_close(b, 50.0 * a, rtol=1e-4, atol=1e-4)


def test_profile_acceleration_and_coil_count_condition_the_noise():
    x = torch.ones(1, 1, 64, 64, dtype=torch.complex64)
    plain = apply_degradation("gfactor_noise", x, 1.0, seed=1, profile=M4RAW_03T) - x
    fast = apply_degradation("gfactor_noise", x, 1.0, seed=1, profile=FASTMRI_3T) - x
    assert abs(float(plain.abs().pow(2).mean()) / 0.1**2 - 1.0) < 0.1  # R = 1: g = 1
    assert float(fast.abs().pow(2).mean()) > 4.0 * float(plain.abs().pow(2).mean())  # >= R = 4
    with pytest.raises(ValueError, match="set by scanner profile"):
        apply_degradation("gfactor_noise", x, 1.0, profile=FASTMRI_3T, acceleration=(2.0, 1.0))


def test_profile_noise_covariance_changes_the_noise_map():
    cov = tuple(tuple(1.0 + 3.0 * float(i == j) * (1 + i) for j in range(4)) for i in range(4))
    cov = tuple(tuple(0.5 * (cov[i][j] + cov[j][i]) for j in range(4)) for i in range(4))
    prof = dataclasses.replace(M4RAW_03T, acceleration=(2.0, 1.0), coil_covariance=cov)
    base = dataclasses.replace(M4RAW_03T, acceleration=(2.0, 1.0))
    x = torch.ones(1, 1, 64, 64, dtype=torch.complex64)
    a = apply_degradation("gfactor_noise", x, 1.0, seed=1, profile=prof)
    b = apply_degradation("gfactor_noise", x, 1.0, seed=1, profile=base)
    assert not torch.allclose(a, b, atol=1e-3)


def test_coil_maps_kwarg_replaces_the_synthetic_maps():
    csm = torch.zeros(1, 2, 1, 16, 4, dtype=torch.complex64)
    csm[:, 0, :, :8], csm[:, 1, :, 8:] = 1.0, 1.0  # orthogonal supports: g = 1
    x = (
        torch.ones(1, 1, 1, 16, 4, dtype=torch.complex64)
        .expand(1, 1, 1, 16, 4)
        .repeat(1, 8, 1, 1, 1)
    )
    n = gfactor_noise_volume(x, 1.0, 0, SLICE, coil_maps=csm) - x
    assert abs(float(n.abs().pow(2).mean()) / (0.1**2 * 2) - 1.0) < 0.2  # sigma^2 R, R = 2


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"acceleration": (2.5, 1.0)}, "whole number"),
        ({"acceleration": (2.0, 2.0)}, "no slice/kz acceleration"),
        ({"acceleration": (4.0, 1.0), "n_coils": 2}, "needs at least 4 coils"),
        ({"caipi_shift": -1}, "caipi_shift"),
        ({"sigma_max": -0.1}, "sigma_max"),
        ({"coil_covariance": ((1.0, 0.0), (0.0, 1.0))}, "coil_covariance"),
        ({"coil_maps": torch.ones(1, 8, 3, 3)}, "coil_maps"),
        ({"coil_maps": torch.ones(1, 8, 64, 64, dtype=torch.complex64), "n_coils": 4}, "disagrees"),
    ],
)
def test_bad_knobs_raise(kwargs, match):
    with pytest.raises(ValueError, match=match):
        apply_degradation("gfactor_noise", _ones()[:, :1, 0], 0.5, **kwargs)


def test_slice_acceleration_aliases_slices_in_a_volume():
    x = _ones(c=2, d=8, h=16, w=16)
    out = gfactor_noise_volume(x, 1.0, 1, VOLUME, acceleration=(2.0, 2.0), n_coils=8)
    assert torch.isfinite(torch.view_as_real(out)).all()
    noise = (out - x).abs().pow(2).mean()
    assert float(noise) > 4.0 * 0.1**2  # sigma^2 g^2 R with g >= 1 and R = 4


def _pair_setup(eps_coil0):
    """Two coils, R = 2, one image column: E is 2 x 2 per aliased pair, so the unfolding is exact."""
    g = torch.Generator().manual_seed(2)
    csm = torch.complex(
        torch.randn(1, 2, 1, 4, 1, generator=g), torch.randn(1, 2, 1, 4, 1, generator=g)
    )
    csm = csm + torch.tensor([2.0, 0.0]).view(1, 2, 1, 1, 1)
    m = torch.zeros(1, 1, 1, 4, 1, dtype=torch.complex64)
    m[..., 2, :] = 1.0 + 0.5j  # object only in the second voxel of the pair (0, 2)
    x = csm * m
    st = sense_setup(
        x,
        2,
        SLICE,
        0,
        name="t",
        acceleration=(2.0, 1.0),
        caipi_shift=None,
        coil_maps=csm,
        coil_covariance=None,
    )
    eps = torch.zeros(1, 2, 1, 4, 1)
    eps[:, 0] = eps_coil0
    return x, csm, m, st, eps


def test_exact_maps_leave_no_residual():
    x, _, _, st, eps = _pair_setup(0.0)
    assert float(residual_image(x, st, eps).abs().max()) < 1e-5


def test_constant_error_in_coil_one_puts_the_analytic_replica_at_the_aliased_voxel():
    """m' - m = E^-1 (diag(1/(1+e), 1) - I) E m for the square 2 x 2 case: the replica at voxel 0."""
    e = 0.15
    x, csm, _m, st, eps = _pair_setup(e)
    res = residual_image(x, st, eps)
    emat = csm[0, :, 0, [0, 2], 0].numpy().astype(np.complex128)  # coils x (voxel 0, voxel 2)
    mvec = np.array([0.0, 1.0 + 0.5j])
    want = np.linalg.inv(emat) @ (np.diag([1 / (1 + e), 1.0]) - np.eye(2)) @ emat @ mvec
    got = (res / csm)[0, 0, 0, [0, 2], 0].numpy()  # S_c cancels: res_c = S_c (m' - m)
    np.testing.assert_allclose(got, want, rtol=2e-3, atol=1e-5)
    assert abs(want[0]) > 1e-3 and abs(want[1]) > 1e-3  # a genuine replica at +-FOV/2, plus a bias
    assert float(res[..., [1, 3], :].abs().max()) < 1e-5  # the other pair is empty


def test_a_common_gain_error_is_a_rescale_not_aliasing():
    x, _, _, st, _ = _pair_setup(0.0)
    e = 0.05
    res = residual_image(x, st, torch.full((1, 2, 1, 4, 1), e))
    torch.testing.assert_close(res, x * (1 / (1 + e) - 1), rtol=0, atol=1e-5)


def test_residual_of_a_one_coil_error_grows_linearly_at_first_order():
    x, _, _, st, _ = _pair_setup(0.0)
    one = [
        float(
            residual_image(
                x, st, torch.cat([torch.full((1, 1, 1, 4, 1), e), torch.zeros(1, 1, 1, 4, 1)], 1)
            )
            .abs()
            .sum()
        )
        for e in (0.01, 0.02)
    ]
    assert one[1] / one[0] == pytest.approx(2.0, rel=0.03)


def test_peak_map_error_is_theta_times_eps_max():
    """max |eps| = theta * 0.2: the residual of a doubled eps_max is (about) double at small theta."""
    x = torch.randn(1, 4, 1, 32, 8, dtype=torch.complex64)
    a = apply_degradation("pi_residual_aliasing", x[:, :, 0], 0.1, seed=2, eps_max=0.1) - x[:, :, 0]
    b = apply_degradation("pi_residual_aliasing", x[:, :, 0], 0.1, seed=2, eps_max=0.2) - x[:, :, 0]
    assert float(b.abs().sum() / a.abs().sum()) == pytest.approx(2.0, rel=0.05)


def test_profile_coil_count_conflicting_with_coil_maps_raises():
    x = torch.ones(1, 1, 64, 64, dtype=torch.complex64)
    maps = torch.ones(1, 8, 64, 64, dtype=torch.complex64)
    with pytest.raises(ValueError, match="disagrees"):
        apply_degradation("gfactor_noise", x, 1.0, profile=FASTMRI_3T, coil_maps=maps)
