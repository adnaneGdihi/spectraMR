"""Every receive-chain axis in every mode: the registry invariants, requirements and collinearity."""

from __future__ import annotations

import itertools

import pytest
import torch

from spectramr.infrastructure.physics.degradations import (
    DEGRADATION_REGISTRY,
    AcquisitionGeometry,
    AcquisitionMode,
    MissingInput,
    Realisation,
    Requirement,
    apply_degradation,
    is_known_collinear,
)
from spectramr.infrastructure.physics.degradations.registry import derive_axis_seed
from spectramr.infrastructure.physics.degradations.registry_queries import (
    requirement_free_degradations,
    slice_capable_degradations,
)

M = AcquisitionMode
AXES = (
    "correlated_coil_noise",
    "emi_burst",
    "receiver_gain_drift",
    "adc_clipping",
    "adc_quantization",
    "antialias_rolloff",
    "coil_dropout",
)
MULTICOIL = ("correlated_coil_noise", "coil_dropout")
STOCHASTIC = ("correlated_coil_noise", "emi_burst", "receiver_gain_drift", "coil_dropout")
GEOMS = {
    M.SLICE_2D: None,
    M.MULTISLICE_2D: AcquisitionGeometry(mode=M.MULTISLICE_2D, voxel_mm=(5.0, 1.0, 1.0)),
    M.VOLUME_3D: AcquisitionGeometry(mode=M.VOLUME_3D, voxel_mm=(5.0, 1.0, 1.0)),
}
CASES = list(itertools.product(AXES, M))
THETAS = (0.0, 0.3, 0.6, 1.0)


def _x(mode, b=1, c=4, seed=3):
    shape = (b, c, 16, 16) if mode is M.SLICE_2D else (b, c, 4, 16, 16)
    g = torch.Generator().manual_seed(seed)
    return torch.complex(torch.randn(shape, generator=g), torch.randn(shape, generator=g))


def _run(name, x, theta, mode, seed=1, **kw):
    return apply_degradation(name, x, theta, seed=seed, geometry=GEOMS[mode], **kw)


def _rel(name, x, y):
    return float((y - x).abs().pow(2).sum() / x.abs().pow(2).sum())


def test_every_axis_is_registered_native_in_every_mode() -> None:
    for name in AXES:
        spec = DEGRADATION_REGISTRY[name]
        assert spec.supported_modes == frozenset(M), name
        assert set(spec.realisation.values()) == {Realisation.NATIVE}, name
        assert spec.requires == (
            frozenset({Requirement.MULTICOIL}) if name in MULTICOIL else frozenset()
        )


@pytest.mark.parametrize(("name", "mode"), CASES)
def test_theta_zero_is_the_exact_identity(name, mode) -> None:
    x = _x(mode)
    assert torch.equal(_run(name, x, 0.0, mode), x)


@pytest.mark.parametrize(("name", "mode"), CASES)
def test_effect_grows_with_theta(name, mode) -> None:
    x = _x(mode)
    effect = [_rel(name, x, _run(name, x, t, mode)) for t in THETAS]
    assert effect[0] == 0.0
    assert all(a <= b for a, b in itertools.pairwise(effect)), effect
    assert effect[-1] > effect[1] > 0.0, effect


@pytest.mark.parametrize(("name", "mode"), CASES)
def test_batch_invariant(name, mode) -> None:
    x0 = _x(mode, seed=3)
    alone = _run(name, x0, 0.7, mode)
    beside = _run(name, torch.cat([x0, 10.0 * _x(mode, seed=4)]), 0.7, mode)
    torch.testing.assert_close(beside[:1], alone, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize(("name", "mode"), CASES)
def test_finite_shape_dtype_and_determinism(name, mode) -> None:
    x = _x(mode)
    a = _run(name, x, 0.8, mode, seed=5)
    assert a.shape == x.shape and a.is_complex() and bool(torch.isfinite(a).all())
    assert torch.equal(a, _run(name, x, 0.8, mode, seed=5))


@pytest.mark.parametrize(("name", "mode"), CASES)
def test_severity_is_scale_invariant(name, mode) -> None:
    x = _x(mode)
    ref = _rel(name, x, _run(name, x, 0.7, mode))
    for scale in (1e-4, 1e2):
        assert _rel(name, scale * x, _run(name, scale * x, 0.7, mode)) == pytest.approx(
            ref, rel=1e-3
        )


@pytest.mark.parametrize("mode", list(M))
@pytest.mark.parametrize("name", STOCHASTIC)
def test_a_different_seed_draws_a_different_realisation(name, mode) -> None:
    x = _x(mode)
    assert not torch.equal(_run(name, x, 0.8, mode, seed=1), _run(name, x, 0.8, mode, seed=2))


@pytest.mark.parametrize("mode", list(M))
@pytest.mark.parametrize("name", MULTICOIL)
def test_multicoil_axes_refuse_a_single_channel(name, mode) -> None:
    with pytest.raises(MissingInput) as err:
        _run(name, _x(mode, c=1), 0.5, mode)
    assert err.value.requirement is Requirement.MULTICOIL
    _run(name, _x(mode, c=2), 0.5, mode)


@pytest.mark.parametrize("mode", list(M))
@pytest.mark.parametrize("name", sorted(set(AXES) - set(MULTICOIL)))
def test_single_coil_axes_run_on_one_channel(name, mode) -> None:
    out = _run(name, _x(mode, c=1), 0.5, mode)
    assert bool(torch.isfinite(out).all())


def _phantom(h=64, w=64, coils=4):
    y, x = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij")
    img = torch.zeros(h, w)
    img[((x / 0.75) ** 2 + (y / 0.9) ** 2) < 1.0] = 0.4
    img[((x / 0.68) ** 2 + (y / 0.83) ** 2) < 1.0] = 0.8
    img[(((x - 0.1) / 0.1) ** 2 + ((y - 0.05) / 0.3) ** 2) < 1.0] = 0.25
    img[((x + 0.35) ** 2 + (y - 0.42) ** 2) < 0.09**2] = 1.0
    img = img * (0.8 + 0.2 * torch.cos(7 * x) * torch.sin(5 * y))
    phase = torch.exp(1j * (1.2 * x + 0.8 * y - 1.5 * (x**2 + y**2)))
    maps = torch.stack(
        [
            (0.5 + torch.exp(-((x - cx) ** 2 + (y - cy) ** 2))) * torch.exp(1j * 0.4 * c * x)
            for c, (cx, cy) in enumerate([(-0.7, -0.7), (0.7, -0.7), (-0.7, 0.7), (0.7, 0.7)])
        ][:coils]
    )
    return (maps * (img * phase)).unsqueeze(0).to(torch.complex64)


#: coil_dropout's default 'full' mode zeroes a coil, whose phase is undefined, so its phase
#: flag is measured in 'partial' mode (a coil that fails mid-scan).
AFFECTS_KW = {"coil_dropout": {"mode": "partial"}}


@pytest.mark.parametrize("name", AXES)
def test_declared_affects_matches_measurement(name) -> None:
    x = _phantom()
    out = apply_degradation(name, x, 1.0, seed=0, **AFFECTS_KW.get(name, {}))
    declared = DEGRADATION_REGISTRY[name].affects
    assert ((out.abs() - x.abs()).norm() / x.abs().norm() > 1e-4) == ("magnitude" in declared)
    w = x.abs() / x.abs().sum()
    dphi = torch.angle(out * torch.conj(x) / (x.abs() ** 2 + 1e-12))
    assert (float(torch.sqrt((w * dphi**2).sum())) > 1e-4) == ("phase" in declared)


def _residual(name, x):
    # Each axis gets its own seed, as every sweep does: two noise axes sharing seed 0 would
    # share one normal stream and look collinear for a reason that is not physical.
    seed = derive_axis_seed(0, name)
    r = (apply_degradation(name, x, 1.0, seed=seed).abs() - x.abs()).flatten()
    return r / r.norm().clamp(min=1e-9)


def test_no_multicoil_axis_is_collinear_with_any_other_axis_unless_declared() -> None:
    """|cos| of the theta = 1 magnitude residuals on a 4-coil stack, as the sim2rank guard does.

    The single-channel axes are covered by the guard in test_digital_twin_extensions.
    """
    x = _phantom()
    runnable = {*requirement_free_degradations(), *MULTICOIL}
    names = [n for n in slice_capable_degradations() if n in runnable]
    resid = {n: _residual(n, x) for n in names}
    offenders = [
        (a, b, float(resid[a].dot(resid[b]).abs()))
        for a, b in itertools.combinations(sorted(resid), 2)
        if (a in MULTICOIL or b in MULTICOIL)
        and not is_known_collinear(a, b)
        and float(resid[a].dot(resid[b]).abs()) > 0.9
    ]
    assert not offenders, offenders


def test_gain_drift_residual_is_modulation_not_a_global_rescale() -> None:
    """Six seeds on the shared guard phantom: |cos| of the residual with |x| stays small."""
    from tests.unit.infrastructure.physics.test_digital_twin_extensions import _brain

    x = _brain()
    scale = (x.abs() / x.abs().norm()).flatten()
    for seed in range(6):
        r = (apply_degradation("receiver_gain_drift", x, 1.0, seed=seed).abs() - x.abs()).flatten()
        assert float(r.dot(scale).abs() / r.norm()) < 0.6, seed
    assert not any(is_known_collinear("receiver_gain_drift", b) for b in DEGRADATION_REGISTRY)


@pytest.mark.parametrize("name", AXES)
def test_double_precision_input_is_accepted(name) -> None:
    x = _x(M.VOLUME_3D).to(torch.complex128)
    out = apply_degradation(name, x, 0.6, seed=2, geometry=GEOMS[M.VOLUME_3D])
    assert out.shape == x.shape and bool(torch.isfinite(out).all())
    real = torch.randn(1, 4, 16, 16, dtype=torch.float64)
    assert bool(torch.isfinite(apply_degradation(name, real, 0.6, seed=2)).all())
