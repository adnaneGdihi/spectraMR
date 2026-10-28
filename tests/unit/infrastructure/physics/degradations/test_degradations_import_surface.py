"""The degradations package exports its public names, each the object from its defining module."""

from __future__ import annotations

import importlib

import pytest

import spectramr.infrastructure.physics.degradations as pkg
from spectramr.infrastructure.physics.degradations import registry
from spectramr.infrastructure.physics.degradations.families import noise

_PKG = "spectramr.infrastructure.physics.degradations"
_REG, _DECL = f"{_PKG}.registry", f"{_PKG}.declarations"
_QRY = f"{_PKG}.registry_queries"
_S2D = f"{_PKG}.slice2d"

# The public surface and each name's defining module.
PUBLIC: dict[str, str] = {
    "DECLARED_THETA_ZERO_PEDESTAL": _DECL,
    "DEGRADATION_REGISTRY": _REG,
    "KNOWN_COLLINEAR_PAIRS": _DECL,
    "MAGNITUDE": _DECL,
    "PHASE": _DECL,
    "DegradationSpec": f"{_PKG}.spec",
    "affected_components": _QRY,
    "apply_degradation": _REG,
    "derive_axis_seed": f"{_PKG}._common",
    "degrade_bias_field": f"{_S2D}.fields",
    "degrade_cartesian_undersampling": f"{_S2D}.sampling_cartesian",
    "degrade_coil_sensitivity_miscal": f"{_S2D}.fields",
    "degrade_complex_gaussian_noise": f"{_S2D}.noise",
    "degrade_concomitant_phase": f"{_S2D}.kspace_phase",
    "degrade_emi": "spectramr.infrastructure.physics.emi_degradation",
    "degrade_geometric_warp": f"{_S2D}.geometric",
    "degrade_iq_ghost": f"{_S2D}.kspace_phase",
    "degrade_nyquist_ghost": f"{_S2D}.kspace_phase",
    "degrade_partial_fourier": f"{_S2D}.sampling_cartesian",
    "degrade_pulsatile_motion": f"{_S2D}.motion",
    "degrade_radial_undersampling": f"{_S2D}.sampling_vd_radial",
    "degrade_random_motion": f"{_S2D}.motion",
    "degrade_resolution_snr": f"{_S2D}.blur",
    "degrade_rician_noise": f"{_S2D}.noise",
    "degrade_slice_thickness_pve": f"{_S2D}.blur",
    "degrade_stimulated_echo": f"{_S2D}.kspace_phase",
    "degrade_sudden_motion": f"{_S2D}.motion",
    "degrade_susceptibility": f"{_S2D}.fields",
    "degrade_t2star_blur": f"{_S2D}.blur",
    "degrade_translation_drift": f"{_S2D}.motion",
    "degrade_variable_density_2d": f"{_S2D}.sampling_vd_radial",
    "degrade_variable_density_cartesian": f"{_S2D}.sampling_cartesian",
    "degrade_variable_density_undersampling": f"{_S2D}.sampling_vd_radial",
    "get_degradation_origin": _QRY,
    "is_identity_at_theta_zero": _QRY,
    "is_known_collinear": _DECL,
    "is_phase_only": _QRY,
    "list_degradations": _QRY,
    "phase_only_degradations": _QRY,
    "requirement_free_degradations": _QRY,
    "slice_capable_degradations": _QRY,
}

ORDER = [
    "complex_gaussian", "rician", "emi", "cartesian_undersamp", "vd_undersamp", "vd_cartesian",
    "radial_undersamp", "rigid_motion", "pulsatile_motion", "translation_drift", "sudden_motion",
    "random_motion", "nyquist_ghost", "gibbs", "b0", "b0_geometric_distortion", "bias",
    "t2star_blur", "slice_pve", "eddy", "spike", "chemical_shift", "susceptibility", "coil_miscal",
    "geometric_warp", "gradient_nonlinearity", "magic_angle", "concomitant_phase",
    "partial_fourier", "iq_ghost", "stimulated_echo", "resolution_snr",
    "respiratory_motion", "local_intermittent_motion", "metal_artifact", "chemical_shift_phase",
    "flow_ghosts", "flow_voids", "inflow_enhancement",
    "b0_drift", "gradient_delay", "gradient_scaling_drift", "mechanical_vibration",
    "correlated_coil_noise", "emi_burst", "receiver_gain_drift", "adc_clipping", "adc_quantization",
    "antialias_rolloff", "coil_dropout",
    "b1_transmit", "dielectric_b1", "imperfect_fat_sat", "imperfect_inversion", "sar_limited_flip",
    "fov_foldover", "tse_t2_blur", "t1_shot_blur", "offres_blur_noncartesian",
    "slice_crosstalk", "through_plane_pve", "spin_history", "off_resonance_slice_shift",
    "gfactor_noise", "pi_residual_aliasing", "cs_recon_artifacts", "recon_apodization", "zero_fill_interp", "gradwarp_residual",
    "field_contrast",
]  # fmt: skip


def test_public_names_are_all_exported() -> None:
    assert set(PUBLIC) <= set(pkg.__all__)


@pytest.mark.parametrize(("name", "home"), PUBLIC.items())
def test_name_is_the_object_from_its_defining_module(name: str, home: str) -> None:
    assert getattr(pkg, name) is getattr(importlib.import_module(home), name)


def test_registry_order_is_preserved() -> None:
    assert list(pkg.DEGRADATION_REGISTRY) == ORDER
    assert pkg.list_degradations() == ORDER


def test_catalogue_codes_continue_the_d_sequence_in_registry_order() -> None:
    # The catalogue axes follow resolution_snr; their codes run on from D32 (emi) without a gap.
    catalogue = ORDER[ORDER.index("resolution_snr") + 1 :]
    codes = [pkg.DEGRADATION_REGISTRY[n].code for n in catalogue]
    assert codes == [f"D{33 + i}" for i in range(len(catalogue))]


def test_every_entry_exposes_its_2d_kernel() -> None:
    for name in pkg.slice_capable_degradations():
        assert callable(pkg.DEGRADATION_REGISTRY[name].fn), name


def test_a_volume_only_entry_has_no_2d_kernel() -> None:
    volume_only = set(pkg.DEGRADATION_REGISTRY) - set(pkg.slice_capable_degradations())
    assert volume_only
    for name in volume_only:
        with pytest.raises(pkg.UnsupportedAcquisitionMode):
            _ = pkg.DEGRADATION_REGISTRY[name].fn


def test_a_name_declared_twice_raises() -> None:
    with pytest.raises(ValueError, match="more than one family"):
        registry._build(noise.ENTRIES + noise.ENTRIES)
