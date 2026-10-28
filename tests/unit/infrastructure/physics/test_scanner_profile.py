"""ScannerProfile: field- and sequence-dependent conditioning of the degradation registry.

A profile handed to ``apply_degradation`` conditions every spec that declares a
``condition``; 0.3 T and 3 T must give different outputs on each, the conditioning must
stay monotone in theta (the theta-invariance trap), and the axis set must follow the
protocol through spec metadata.
"""

from __future__ import annotations

import dataclasses
from dataclasses import replace
from itertools import pairwise

import pytest
import torch

from spectramr.infrastructure.physics.degradations import DEGRADATION_REGISTRY, apply_degradation
from spectramr.infrastructure.physics.degradations.declarations import DECLARED_PHYSICAL_FOV_MM
from spectramr.infrastructure.physics.degradations.geometry import (
    AcquisitionGeometry,
    AcquisitionMode,
)
from spectramr.infrastructure.physics.degradations.registry import check_requirements
from spectramr.infrastructure.physics.degradations.registry_queries import (
    slice_capable_degradations,
)
from spectramr.infrastructure.physics.degradations.spec import (
    ActivationGate,
    MissingInput,
    Requirement,
)
from tests.utils.degradation_inputs import apply_degradation as apply_with_inputs
from spectramr.infrastructure.physics.scanner_presets import (
    FASTMRI_3T,
    M4RAW_03T,
    PRESETS,
    resolve_profile,
)
from spectramr.infrastructure.physics.scanner_profile import ScannerProfile, Sequence, active_axes

CONDITIONED = (
    "b0",
    "b0_geometric_distortion",
    "chemical_shift",
    "concomitant_phase",
    "metal_artifact",
    "dielectric_b1",
    "field_contrast",
)
# TE is tuned to the fat frequency, so this axis' image is field-invariant: it is conditioned
# (B0, fat fraction, chi_ppm) but cannot join the low-versus-high-field tests above.
FIELD_INVARIANT_CONDITIONED = ("chemical_shift_phase",)
THETAS = (0.0, 0.3, 0.6, 1.0)
VOL = AcquisitionGeometry(mode=AcquisitionMode.VOLUME_3D, voxel_mm=(1.0, 1.0, 1.0))


def _x() -> torch.Tensor:
    g = torch.Generator().manual_seed(7)
    return torch.complex(
        torch.rand(1, 1, 32, 32, generator=g), torch.rand(1, 1, 32, 32, generator=g)
    )


def _at(b0: float, **over) -> ScannerProfile:
    return replace(FASTMRI_3T, name=f"b0_{b0}", b0_T=b0, **over)


def _out(name: str, theta: float, profile: ScannerProfile | None) -> torch.Tensor:
    # q-map axes get the golden gate's synthetic maps (the one place they are built).
    fov = DECLARED_PHYSICAL_FOV_MM.get(name)
    if fov is not None and profile is not None:  # measured on a phantom of its physical size
        voxel = (profile.geometry.voxel_mm[0], fov / 32, fov / 32)
        profile = replace(profile, geometry=replace(profile.geometry, voxel_mm=voxel))
    return apply_with_inputs(name, _x(), theta, seed=3, profile=profile)


def test_dielectric_effect_grows_with_the_field() -> None:
    # The calibrated standing wave varies on lambda = 3.3 m at 0.3 T and 0.33 m at 3 T.
    err = {
        b0: (_out("dielectric_b1", 1.0, _at(b0)).abs() - _x().abs()).pow(2).mean()
        for b0 in (0.3, 3.0)
    }
    assert float(err[0.3]) < 1e-6 * float(err[3.0])


# These read the profile's coil and parallel-imaging facts (n_coils, coil_covariance,
# acceleration), not a field-strength formula.
COIL_CONDITIONED = ("coil_dropout", "correlated_coil_noise", "gfactor_noise", "pi_residual_aliasing")
# These read the profile's sequence timing (TR, TE, flip angle, readout BW, ESP), not B0.
SEQUENCE_CONDITIONED = (
    "flow_ghosts",
    "flow_voids",
    "inflow_enhancement",
    "b0_drift",
    "mechanical_vibration",
    "imperfect_inversion",
    "sar_limited_flip",
    "slice_crosstalk",
    "spin_history",
)
# These read protocol facts other than the field: the sequence type, the fat fraction, or timing
# only for the sequence types that have it (so the default profile's timing is not read). Each
# axis' own catalogue test covers the facts it reads.
PROTOCOL_CONDITIONED = ("b1_transmit", "imperfect_fat_sat", "tse_t2_blur", "t1_shot_blur")
# Field-dependent like CONDITIONED, but volume-only, so the 2-D output tests above cannot run it:
# its condition is checked here and its output in catalogue/test_multislice_offres.py.
VOLUME_ONLY_FIELD_CONDITIONED = ("off_resonance_slice_shift",)


def test_conditioned_axes_are_exactly_the_physical_formula_ones() -> None:
    assert {n for n, s in DEGRADATION_REGISTRY.items() if s.condition is not None} == set(
        CONDITIONED
        + FIELD_INVARIANT_CONDITIONED
        + COIL_CONDITIONED
        + SEQUENCE_CONDITIONED
        + PROTOCOL_CONDITIONED
        + VOLUME_ONLY_FIELD_CONDITIONED
    )


@pytest.mark.parametrize("name", SEQUENCE_CONDITIONED)
def test_sequence_conditioned_axes_read_timing_not_field(name: str) -> None:
    cond, shape = DEGRADATION_REGISTRY[name].condition, _x().shape
    timing = {"tr_ms": 40.0, "te_ms": 10.0, "fa_deg": 20.0, "esp_ms": 0.5, "ti_ms": 500.0}
    low, high = (cond(_at(b0, **timing), 1.0, FASTMRI_3T.geometry, shape) for b0 in (0.3, 3.0))
    assert low == high
    retimed = {"tr_ms": 80.0, "te_ms": 20.0, "fa_deg": 40.0, "esp_ms": 1.0, "ti_ms": 900.0}
    other = _at(3.0, readout_bw_hz_per_px=500.0, **retimed)
    assert cond(other, 1.0, FASTMRI_3T.geometry, shape) != high


@pytest.mark.parametrize("name", PROTOCOL_CONDITIONED)
def test_protocol_conditioned_axes_do_not_read_the_field(name: str) -> None:
    cond, shape = DEGRADATION_REGISTRY[name].condition, _x().shape
    low, high = (cond(_at(b0), 1.0, FASTMRI_3T.geometry, shape) for b0 in (0.3, 3.0))
    assert low == high


@pytest.mark.parametrize("name", VOLUME_ONLY_FIELD_CONDITIONED)
def test_volume_only_field_conditioned_axes_read_the_field(name: str) -> None:
    assert name not in slice_capable_degradations()
    cond, shape = DEGRADATION_REGISTRY[name].condition, (1, 1, 4, 32, 32)
    low, high = (cond(_at(b0), 1.0, VOL, shape) for b0 in (0.3, 3.0))
    assert low != high


@pytest.mark.parametrize("name", CONDITIONED)
def test_low_and_high_field_differ(name: str) -> None:
    assert not torch.allclose(_out(name, 1.0, _at(0.3)), _out(name, 1.0, _at(3.0)), atol=1e-5)


@pytest.mark.parametrize("name", CONDITIONED)
@pytest.mark.parametrize("b0", [0.3, 3.0])
def test_conditioned_effect_is_monotone_in_theta(name: str, b0: float) -> None:
    # Magnitude error, as the sweep scores it; the phase-only axis is scored on the field.
    x, p = _x(), _at(b0)
    po = DEGRADATION_REGISTRY[name].phase_only
    ref = x if po else x.abs()
    errs = [
        ((_out(name, t, p) if po else _out(name, t, p).abs()) - ref).abs().pow(2).mean().item()
        for t in THETAS
    ]
    assert all(a < b for a, b in pairwise(errs)), errs


@pytest.mark.parametrize("name", CONDITIONED)
def test_theta_zero_is_identity_with_a_profile(name: str) -> None:
    # b0_geometric_distortion resamples on its grid even at zero shift: float-level error only.
    torch.testing.assert_close(_out(name, 0.0, _at(0.3)), _x(), atol=1e-5, rtol=0)


def test_b0_at_the_3t_reference_reproduces_the_default_ramp() -> None:
    for t in THETAS:
        torch.testing.assert_close(_out("b0", t, _at(3.0)), _out("b0", t, None), atol=1e-6, rtol=0)


def test_chemical_shift_scales_with_field_strength() -> None:
    assert _at(3.0).chemical_shift_max_px() == pytest.approx(
        10.0 * _at(0.3).chemical_shift_max_px(), rel=1e-6
    )


def test_conditions_are_theta_scaled_and_never_forward_raw_field() -> None:
    cond = DEGRADATION_REGISTRY["chemical_shift"].condition
    shape = _x().shape
    kw = cond(FASTMRI_3T, 0.5, FASTMRI_3T.geometry, shape)
    assert set(kw) == {"shift_pixels", "fat_fraction"}
    assert kw["shift_pixels"] == pytest.approx(0.5 * FASTMRI_3T.chemical_shift_max_px())
    geo = DEGRADATION_REGISTRY["b0_geometric_distortion"].condition(
        FASTMRI_3T, 1.0, FASTMRI_3T.geometry, shape
    )
    # dwell = 1 / (BW_px * W): the delegate's BW_FE = 1 / (W * dwell) is then the profile's BW.
    assert geo["dwell_time"] == pytest.approx(1.0 / (FASTMRI_3T.readout_bw_hz_per_px * 32))


def test_kwarg_overlap_with_a_condition_raises() -> None:
    with pytest.raises(ValueError, match="shift_pixels"):
        apply_degradation("chemical_shift", _x(), 0.5, profile=FASTMRI_3T, shift_pixels=1.0)


def test_geometry_and_profile_together_raise() -> None:
    with pytest.raises(ValueError, match="geometry"):
        apply_degradation("b0", _x(), 0.5, geometry=FASTMRI_3T.geometry, profile=FASTMRI_3T)


def test_profile_supplies_the_geometry() -> None:
    p = replace(FASTMRI_3T, geometry=VOL)
    out = apply_degradation("b0", torch.rand(1, 1, 4, 16, 16), 0.5, profile=p)
    assert out.shape == (1, 1, 4, 16, 16)


def test_sequence_epi_requirement_reads_the_profile() -> None:
    spec = replace(DEGRADATION_REGISTRY["gibbs"], requires=frozenset({Requirement.SEQUENCE_EPI}))
    x = _x()
    with pytest.raises(MissingInput, match="sequence_epi"):
        check_requirements(spec, x, {})
    with pytest.raises(MissingInput, match="sequence_epi"):
        check_requirements(spec, x, {}, profile=FASTMRI_3T)
    check_requirements(spec, x, {}, profile=replace(FASTMRI_3T, sequence=Sequence.EPI))


def test_nyquist_ghost_runs_without_a_profile_but_is_gated_to_epi() -> None:
    assert DEGRADATION_REGISTRY["nyquist_ghost"].requires == frozenset()
    assert ActivationGate.SEQUENCE_EPI in DEGRADATION_REGISTRY["nyquist_ghost"].active_when
    apply_degradation("nyquist_ghost", _x(), 0.5)
    epi = replace(FASTMRI_3T, sequence=Sequence.EPI)
    assert active_axes(FASTMRI_3T, ["nyquist_ghost"]) == []
    assert active_axes(epi, ["nyquist_ghost"]) == ["nyquist_ghost"]


def test_active_axes_gates_by_protocol() -> None:
    cands = ["complex_gaussian", "chemical_shift", "cartesian_undersamp", "nyquist_ghost", "b0"]
    m4 = active_axes(M4RAW_03T, cands)
    assert m4 == ["complex_gaussian", "chemical_shift", "b0"]
    assert "cartesian_undersamp" in active_axes(FASTMRI_3T, cands)
    assert "chemical_shift" not in active_axes(_at(3.0, fat_fraction=0.0), cands)


def test_active_axes_explicit_allow_list_overrides_and_unknown_raises() -> None:
    p = _at(3.0, active_axes=("b0",))
    assert active_axes(p, ["b0", "rician", "chemical_shift"]) == ["b0"]
    with pytest.raises(KeyError):
        active_axes(FASTMRI_3T, ["not_an_axis"])
    with pytest.raises(ValueError, match="active_axes"):
        _at(3.0, active_axes=("not_an_axis",))


@pytest.mark.parametrize(
    "over",
    [
        {"b0_T": 0.0},
        {"readout_bw_hz_per_px": -1.0},
        {"tr_ms": 0.0},
        {"te_ms": 50.0, "tr_ms": 40.0},
        {"fa_deg": 200.0},
        {"etl": 0},
        {"acceleration": (0.5, 1.0)},
        {"n_coils": 0},
        {"coil_covariance": ((1.0,),)},
        {"fat_fraction": 1.5},
        {"sequence": "epi"},
        {"geometry": None},
        {"slew": -1.0},
    ],
)
def test_invalid_values_raise(over: dict) -> None:
    with pytest.raises(ValueError):
        replace(FASTMRI_3T, **over)


def test_to_meta_covers_every_field() -> None:
    p = replace(FASTMRI_3T, coil_covariance=None, esp_ms=8.0, etl=4)
    meta = p.to_meta()
    assert set(meta) == {f.name for f in dataclasses.fields(ScannerProfile)}
    assert meta["sequence"] == "spin_echo"
    assert meta["geometry"] == FASTMRI_3T.geometry.to_meta()
    assert meta["acceleration"] == [4.0, 1.0]


def test_meta_sha256_identifies_every_field() -> None:
    # slew and active_axes are the fields a 16-key provenance cap used to cut.
    base = FASTMRI_3T.meta_sha256()
    assert base == replace(FASTMRI_3T).meta_sha256()
    assert replace(FASTMRI_3T, slew=150.0).meta_sha256() != base
    assert replace(FASTMRI_3T, active_axes=("gibbs",)).meta_sha256() != base


def test_presets_round_trip() -> None:
    for name, p in PRESETS.items():
        assert resolve_profile(name) is p
        assert p.to_meta()["name"] == name
    # Both are 2-D multislice protocols with 5 mm slices: voxel_mm is (slice, pe, ro).
    assert FASTMRI_3T.geometry.voxel_mm == (5.0, 0.6875, 0.6875)
    assert M4RAW_03T.geometry.voxel_mm == (5.0, 0.9375, 0.9375)
    with pytest.raises(KeyError):
        resolve_profile("siemens_7T")


def test_from_acquisition_config_overrides_the_preset_at_the_config_field():
    from spectramr.config.schemas.physics import DigitalTwinAcquisitionConfig
    from spectramr.infrastructure.physics.scanner_presets import FASTMRI_3T
    from spectramr.infrastructure.physics.scanner_profile import ScannerProfile

    acq = DigitalTwinAcquisitionConfig(
        scanner_profile="fastmri_3T", mode="multislice_2d", slice_order="sequential"
    )
    p = ScannerProfile.from_acquisition_config(acq, field_strength_t=3.0)
    assert p.geometry.slice_order.value == "sequential"
    assert p.geometry.voxel_mm == FASTMRI_3T.geometry.voxel_mm
    assert p.readout_bw_hz_per_px == FASTMRI_3T.readout_bw_hz_per_px


def test_from_acquisition_config_without_preset_is_a_2d_slice_by_default():
    from spectramr.config.schemas.physics import DigitalTwinAcquisitionConfig
    from spectramr.infrastructure.physics.scanner_profile import ScannerProfile, Sequence

    acq = DigitalTwinAcquisitionConfig(readout_bw_hz_per_px=300.0, sequence="epi")
    p = ScannerProfile.from_acquisition_config(acq, field_strength_t=0.55)
    assert (p.b0_T, p.sequence, p.geometry.mode.value) == (0.55, Sequence.EPI, "slice_2d")


@pytest.mark.parametrize("sequence", ["radial", "spiral"])
def test_b0_geometric_condition_rejects_non_cartesian_sequences(sequence) -> None:
    from spectramr.infrastructure.physics.scanner_profile import Sequence

    profile = replace(FASTMRI_3T, sequence=Sequence(sequence))
    with pytest.raises(ValueError, match="non-Cartesian"):
        apply_degradation("b0_geometric_distortion", _x(), 0.5, profile=profile)


def test_b0_geometric_condition_emits_the_shift_model() -> None:
    from spectramr.infrastructure.physics.degradations.conditions import b0_geometric_condition
    from spectramr.infrastructure.physics.scanner_profile import Sequence

    epi = replace(FASTMRI_3T, sequence=Sequence.EPI)
    assert b0_geometric_condition(epi, 1.0, epi.geometry, (1, 1, 32, 32))["sequence"] == "epi"
    assert (
        b0_geometric_condition(FASTMRI_3T, 1.0, FASTMRI_3T.geometry, (1, 1, 32, 32))["sequence"]
        == "spin_warp"
    )


# --- the field-invariant conditioned axes: TE is tuned to df, so only the TE sees B0 and chi_ppm ---
_FIELDS = (0.3, 1.5, 3.0)
_CHIS = (3.0, 3.5)


def _conditioned_te_ms(name: str, theta: float, profile: ScannerProfile) -> float:
    from spectramr.infrastructure.physics.degradations.catalogue.chemical_shift_phase import (
        swept_te_ms,
    )

    cond = DEGRADATION_REGISTRY[name].condition(profile, theta, profile.geometry, (1, 1, 32, 32))
    return {"chemical_shift_phase": lambda: swept_te_ms(theta, cond["b0_T"], cond["sigma_ppm"])}[
        name
    ]()


def _assert_field_invariant(name: str) -> None:
    ref = _out(name, 0.6, _at(3.0, chi_ppm=3.5))
    tes = set()
    for b0 in _FIELDS:
        for chi in _CHIS:
            p = _at(b0, chi_ppm=chi)
            torch.testing.assert_close(_out(name, 0.6, p), ref, atol=1e-5, rtol=0)
            tes.add(round(_conditioned_te_ms(name, 0.6, p), 6))
    assert len(tes) == len(_FIELDS) * len(_CHIS), "the conditioned TE must differ per field and ppm"


def test_field_invariant_tuple_is_the_guarded_set() -> None:
    assert set(FIELD_INVARIANT_CONDITIONED) == {"chemical_shift_phase"}


@pytest.mark.parametrize("name", FIELD_INVARIANT_CONDITIONED)
def test_field_invariant_axes_do_not_depend_on_field_or_ppm(name: str) -> None:
    _assert_field_invariant(name)


@pytest.mark.parametrize("name", FIELD_INVARIANT_CONDITIONED)
def test_field_invariant_axes_are_identity_at_theta_zero(name: str) -> None:
    for b0 in _FIELDS:
        assert torch.equal(_out(name, 0.0, _at(b0)), _x())


@pytest.mark.parametrize("name", FIELD_INVARIANT_CONDITIONED)
def test_field_invariant_axes_are_monotone_in_theta(name: str) -> None:
    x, p = _x(), _at(3.0)
    errs = [((_out(name, t, p).abs()) - x.abs()).abs().pow(2).mean().item() for t in THETAS]
    assert all(a < b for a, b in pairwise(errs)), errs


def test_a_field_dependent_axis_in_the_tuple_is_caught() -> None:
    # Planted: b0 is conditioned and field-dependent, so the guard must fail on it.
    with pytest.raises(AssertionError):
        _assert_field_invariant("b0")
