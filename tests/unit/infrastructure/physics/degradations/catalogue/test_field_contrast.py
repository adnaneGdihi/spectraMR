"""field_contrast: the signal re-evaluated at a lower field from q-maps, via the ULF operator."""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch

from spectramr.infrastructure.physics.degradations import (
    DEGRADATION_REGISTRY,
    AcquisitionGeometry,
    AcquisitionMode,
    apply_degradation,
)
from spectramr.infrastructure.physics.degradations.catalogue import field_contrast as fc
from spectramr.infrastructure.physics.degradations.spec import MissingInput, Requirement
from spectramr.infrastructure.physics.scanner_presets import FASTMRI_3T, M4RAW_03T
from spectramr.infrastructure.physics.scanner_profile import Sequence
from tests.unit.infrastructure.physics.degradations.catalogue.invariants import (
    GEOMETRIES,
    assert_axis_invariants,
    phantom,
)

NAME = "field_contrast"
VOLUME = GEOMETRIES["volume"]


def qmaps(geometry: AcquisitionGeometry | None, *, batch: int = 1) -> dict[str, torch.Tensor]:
    """Smooth T1 / T2 (ms) with a zero (background) corner, shaped like ``phantom``."""
    shape = (32, 32) if geometry is None else (6, 32, 32)
    ax = [torch.linspace(0.0, 1.0, n) for n in shape]
    grid = torch.meshgrid(*ax, indexing="ij")
    s = sum(grid)
    t1 = 900.0 + 400.0 * torch.sin(3.0 * s)
    t2 = 80.0 + 30.0 * torch.cos(2.0 * s)
    t1[..., :3, :3] = 0.0
    full = (batch, 1, *shape)
    return {"t1": t1.expand(full).clone(), "t2": t2.expand(full).clone()}


def ernst(t1, t2, b0, *, b0_src, tr, te, alpha, beta=0.35, gamma=0.05):
    """Hand Ernst signal at field ``b0`` (PD = 1), float64."""
    t1b = t1.double() * (b0 / b0_src) ** beta
    t2b = t2.double() * (b0 / b0_src) ** gamma
    e1, e2 = torch.exp(-tr / t1b), torch.exp(-te / t2b)
    return math.sin(alpha) * (1 - e1) / (1 - math.cos(alpha) * e1) * e2


@pytest.mark.parametrize("geom", GEOMETRIES)
def test_axis_invariants(geom: str) -> None:
    g = GEOMETRIES[geom]
    assert_axis_invariants(NAME, g, qmaps=qmaps(g))


@pytest.mark.parametrize("geom", ["slice_2d", "volume"])
@pytest.mark.parametrize("theta", [0.37, 1.0])
def test_ratio_is_the_hand_ernst_ratio_at_the_target_field(geom: str, theta: float) -> None:
    g = GEOMETRIES[geom]
    x, q = phantom(g), qmaps(g)
    kw = {"tr_ms": 20.0, "te_ms": 2.0, "fa_deg": 15.0, "sequence": "spoiled_gre", "b0_T": 3.0}
    y = apply_degradation(NAME, x, theta, geometry=g, qmaps=q, **kw)
    alpha = math.radians(15.0)
    b_t = 3.0 * (0.05 / 3.0) ** theta
    hand = ernst(q["t1"], q["t2"], b_t, b0_src=3.0, tr=20.0, te=2.0, alpha=alpha) / ernst(
        q["t1"], q["t2"], 3.0, b0_src=3.0, tr=20.0, te=2.0, alpha=alpha
    )
    tissue = (q["t1"] > 0).expand_as(x.real)
    hand = torch.where(tissue, hand.expand_as(x.real), torch.ones_like(x.real)).float()
    ax = x.abs()
    support = ax > fc.SUPPORT_FRACTION * ax.amax()
    scale = (ax * support).sum() / (ax * hand * support).sum()  # mean-intensity match
    ratio = (y / x).real
    torch.testing.assert_close(ratio, hand * scale, rtol=2e-5, atol=0)
    # The contrast change is what remains: tissue brightens relative to the background.
    assert ratio[tissue].min() > ratio[~tissue].max()


@pytest.mark.parametrize("geom", ["slice_2d", "volume"])
def test_mean_intensity_over_the_support_is_unchanged(geom: str) -> None:
    g = GEOMETRIES[geom]
    x = torch.cat([phantom(g), 10.0 * phantom(g, seed=11)])
    y = apply_degradation(NAME, x, 1.0, geometry=g, qmaps=qmaps(g))
    ax, ay = x.abs().flatten(1), y.abs().flatten(1)
    support = ax > fc.SUPPORT_FRACTION * ax.amax(dim=1, keepdim=True)
    torch.testing.assert_close(
        (ay * support).sum(1) / support.sum(1),
        (ax * support).sum(1) / support.sum(1),
        rtol=1e-5,
        atol=0,
    )
    assert not torch.allclose(y.abs(), x.abs())


def test_target_field_is_log_linear_from_b0_to_0p05() -> None:
    assert fc.target_field(3.0, 0.0) == 3.0
    assert fc.target_field(3.0, 1.0) == pytest.approx(0.05)
    assert fc.target_field(3.0, 0.5) == pytest.approx(math.sqrt(3.0 * 0.05))
    assert fc.target_field(0.3, 1.0) == pytest.approx(0.05)


def test_phase_is_kept_and_only_magnitude_moves() -> None:
    x, q = phantom(VOLUME), qmaps(VOLUME)
    y = apply_degradation(NAME, x, 0.8, geometry=VOLUME, qmaps=q)
    torch.testing.assert_close(torch.angle(y), torch.angle(x), rtol=0, atol=1e-6)
    assert DEGRADATION_REGISTRY[NAME].affects == frozenset({"magnitude"})


def test_a_real_input_stays_real() -> None:
    x = phantom(None).real
    assert not apply_degradation(NAME, x, 0.5, qmaps=qmaps(None)).is_complex()


def test_one_slice_stack_equals_the_slice_kernel() -> None:
    x = phantom(None)
    q2 = qmaps(None)
    q5 = {k: v.unsqueeze(2) for k, v in q2.items()}
    g = dataclasses.replace(VOLUME)
    ref = apply_degradation(NAME, x, 0.6, qmaps=q2)
    out = apply_degradation(NAME, x.unsqueeze(2), 0.6, geometry=g, qmaps=q5)
    torch.testing.assert_close(out, ref.unsqueeze(2), rtol=0, atol=1e-6)
    ms = AcquisitionGeometry(mode=AcquisitionMode.MULTISLICE_2D, voxel_mm=(5.0, 1.0, 1.0))
    torch.testing.assert_close(
        apply_degradation(NAME, x.unsqueeze(2), 0.6, geometry=ms, qmaps=q5), out, rtol=0, atol=0
    )


def test_per_sample_maps_are_batch_aligned_in_a_volume() -> None:
    x = torch.cat([phantom(VOLUME), phantom(VOLUME, seed=9)])
    q = qmaps(VOLUME, batch=2)
    q["t1"][1] *= 1.7
    y = apply_degradation(NAME, x, 0.7, geometry=VOLUME, qmaps=q)
    alone = apply_degradation(
        NAME, x[1:], 0.7, geometry=VOLUME, qmaps={k: v[1:] for k, v in q.items()}
    )
    torch.testing.assert_close(y[1:], alone, rtol=0, atol=1e-6)
    assert not torch.allclose(
        y[0],
        apply_degradation(
            NAME, x[:1], 0.7, geometry=VOLUME, qmaps={k: v[1:] for k, v in q.items()}
        )[0],
    )


def test_it_delegates_to_the_ulf_operator_with_its_other_stages_off(monkeypatch) -> None:
    seen: list[dict] = []
    real = fc.DifferentiableULFForwardOperator

    def spy(**kw):
        seen.append(kw)
        return real(**kw)

    monkeypatch.setattr(fc, "DifferentiableULFForwardOperator", spy)
    apply_degradation(NAME, phantom(None), 1.0, qmaps=qmaps(None), tr_ms=100.0, te_ms=5.0)
    assert [s["b0_target"] for s in seen] == [3.0, pytest.approx(0.05)]
    for s in seen:
        assert s["b0_source"] == 3.0 and (s["tr"], s["te"]) == (100.0, 5.0)
        assert not any(s[k] for k in ("enable_noise", "enable_emi", "enable_gnl", "enable_maxwell"))


def test_operator_defaults_apply_when_sequence_times_are_not_given() -> None:
    x, q = phantom(None), qmaps(None)
    default = apply_degradation(NAME, x, 1.0, qmaps=q)
    explicit = apply_degradation(NAME, x, 1.0, qmaps=q, tr_ms=500.0, te_ms=14.0, fa_deg=90.0)
    assert torch.equal(default, explicit)


def test_the_axis_declares_qmaps_and_all_three_modes() -> None:
    spec = DEGRADATION_REGISTRY[NAME]
    assert spec.requires == frozenset({Requirement.QMAPS})
    assert spec.supported_modes == set(AcquisitionMode)
    assert spec.severity.params[0].units == "T"
    assert (spec.severity.params[0].at_theta_min, spec.severity.params[0].at_theta_max) == (
        3.0,
        0.05,
    )


@pytest.mark.parametrize("geom", GEOMETRIES)
def test_missing_qmaps_raises(geom: str) -> None:
    g = GEOMETRIES[geom]
    with pytest.raises(MissingInput):
        apply_degradation(NAME, phantom(g), 0.5, geometry=g)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"ti_ms": 100.0}, "ti_ms"),
        ({"sequence": "tse"}, "sequence"),
        ({"sequence": "spin_echo", "fa_deg": 60.0}, "90 degree"),
        ({"fa_deg": 200.0}, "fa_deg"),
        ({"tr_ms": 10.0, "te_ms": 12.0}, "te_ms"),
        ({"tr_ms": -1.0}, "tr_ms"),
        ({"b0_T": 0.05}, "b0_T"),
        ({"b0_T": float("nan")}, "b0_T"),
    ],
)
def test_bad_conditions_raise(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        apply_degradation(NAME, phantom(None), 0.5, qmaps=qmaps(None), **kwargs)


def test_bad_conditions_raise_even_at_theta_zero() -> None:
    with pytest.raises(ValueError, match="ti_ms"):
        apply_degradation(NAME, phantom(None), 0.0, qmaps=qmaps(None), ti_ms=1.0)


def test_bad_qmaps_raise() -> None:
    x, q = phantom(None), qmaps(None)
    with pytest.raises(ValueError, match="'t1' and 't2'"):
        apply_degradation(NAME, x, 0.5, qmaps={"t1": q["t1"]})
    with pytest.raises(ValueError, match="unknown qmaps keys"):
        apply_degradation(NAME, x, 0.5, qmaps={**q, "t2star": q["t2"]})
    with pytest.raises(ValueError, match="spatial shape"):
        apply_degradation(NAME, x, 0.5, qmaps={**q, "t2": q["t2"][..., :16]})


def test_a_profile_supplies_field_and_sequence() -> None:
    x, q = phantom(None), qmaps(None)
    via_profile = apply_degradation(NAME, x, 0.9, profile=M4RAW_03T, qmaps=q)
    explicit = apply_degradation(NAME, x, 0.9, qmaps=q, b0_T=0.3, sequence="spin_echo")
    assert torch.equal(via_profile, explicit)
    assert not torch.equal(
        via_profile, apply_degradation(NAME, x, 0.9, profile=FASTMRI_3T, qmaps=q)
    )


def test_a_profile_with_sequence_times_is_read() -> None:
    gre = dataclasses.replace(
        FASTMRI_3T, sequence=Sequence.SPOILED_GRE, tr_ms=30.0, te_ms=3.0, fa_deg=20.0
    )
    x, q = phantom(None), qmaps(None)
    out = apply_degradation(NAME, x, 1.0, profile=gre, qmaps=q)
    ref = apply_degradation(NAME, x, 1.0, qmaps=q, tr_ms=30.0, te_ms=3.0, fa_deg=20.0)
    assert torch.equal(out, ref)


@pytest.mark.parametrize(
    "change", [{"ti_ms": 400.0}, {"sequence": Sequence.TSE}, {"sequence": Sequence.MPRAGE}]
)
def test_a_profile_the_ernst_equation_cannot_describe_raises(change: dict) -> None:
    with pytest.raises(ValueError, match=r"ti_ms|sequence"):
        apply_degradation(
            NAME,
            phantom(None),
            0.5,
            profile=dataclasses.replace(FASTMRI_3T, **change),
            qmaps=qmaps(None),
        )


def test_a_caller_field_next_to_a_profile_raises() -> None:
    with pytest.raises(ValueError, match="b0_T"):
        apply_degradation(NAME, phantom(None), 0.5, profile=FASTMRI_3T, qmaps=qmaps(None), b0_T=3.0)


@pytest.mark.parametrize("knob", ["beta", "gamma_t2"])
@pytest.mark.parametrize("bad", [0.0, -0.1, float("nan"), float("inf")])
def test_dispersion_exponents_are_validated(knob: str, bad: float) -> None:
    with pytest.raises(ValueError, match=knob):
        apply_degradation(NAME, phantom(None), 0.5, qmaps=qmaps(None), **{knob: bad})


@pytest.mark.parametrize("knob", ["beta", "gamma_t2"])
def test_dispersion_exponents_are_read(knob: str) -> None:
    x, q = phantom(None), qmaps(None)
    base = apply_degradation(NAME, x, 1.0, qmaps=q)
    assert not torch.allclose(base, apply_degradation(NAME, x, 1.0, qmaps=q, **{knob: 0.9}))


def test_equal_fields_give_ratio_one_through_the_operator() -> None:
    q = qmaps(None)
    t1, t2 = q["t1"], q["t2"]
    args = (3.0, 3.0, 500.0, 14.0, math.pi / 2, 0.35, 0.05)
    ratio = fc._contrast_ratio(t1, t2, *args)
    torch.testing.assert_close(ratio, torch.ones_like(ratio), rtol=1e-6, atol=0)


def test_a_double_input_keeps_its_dtype_and_precision() -> None:
    x = phantom(None).to(torch.complex128)
    q = {k: v.double() for k, v in qmaps(None).items()}
    y = apply_degradation(NAME, x, 0.7, qmaps=q)
    assert y.dtype == torch.complex128


def test_theta_zero_returns_the_input_tensor_like_the_other_axes() -> None:
    x = phantom(None)
    assert apply_degradation(NAME, x, 0.0, qmaps=qmaps(None)) is x
