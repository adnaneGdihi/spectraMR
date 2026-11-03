"""Declared endpoint kwargs: each names a kernel kwarg that really sets the primary axis endpoint."""

from __future__ import annotations

import inspect

import pytest
import torch

from spectramr.infrastructure.physics.degradations import (
    DEGRADATION_REGISTRY,
    AcquisitionGeometry,
    AcquisitionMode,
    Requirement,
)
from spectramr.infrastructure.physics.degradations.declarations import (
    PHYSICAL_OVERRIDE_KWARGS,
)
from tests.utils.degradation_inputs import apply_degradation

M = AcquisitionMode

# name -> (endpoint_kwargs, endpoint_scale). A new axis must opt in here.
DECLARED: dict[str, tuple[tuple[str | None, str | None], float]] = {
    "complex_gaussian": (("snr_max_db", "snr_min_db"), 1.0),
    "rician": ((None, "sigma_max"), 1.0),
    "emi": ((None, "max_amp_ratio"), 1.0),
    "cartesian_undersamp": ((None, "R_max"), 1.0),
    "vd_undersamp": ((None, "R_max"), 1.0),
    "vd_cartesian": ((None, "R_max"), 1.0),
    "radial_undersamp": ((None, "R_max"), 1.0),
    "partial_fourier": ((None, "min_fraction"), 1.0),
    "pulsatile_motion": ((None, "max_amplitude_px"), 1.0),
    "translation_drift": ((None, "max_drift_px"), 1.0),
    "sudden_motion": ((None, "max_shift_px"), 1.0),
    "random_motion": ((None, "max_shift_px"), 1.0),
    "nyquist_ghost": ((None, "max_phase_rad"), 1.0),
    "bias": ((None, "max_log_b"), 1.0),
    "t2star_blur": ((None, "max_readout_over_t2"), 1.0),
    "slice_pve": ((None, "max_kernel_px"), 4.0),
    "coil_miscal": ((None, "max_relative_error"), 1.0),
    "geometric_warp": ((None, "max_amplitude_px"), 1.0),
    "concomitant_phase": ((None, "max_gradient_mT_per_m"), 1.0),
    "iq_ghost": ((None, "max_dc_offset"), 1.0),
    "stimulated_echo": ((None, "max_amp"), 1.0),
    "resolution_snr": ((None, "fwhm_max_px"), 1.0),
    "respiratory_motion": ((None, "peak_si_mm"), 1.0),
    "local_intermittent_motion": ((None, "peak_mm"), 1.0),
    "metal_artifact": ((None, "max_radius_mm"), 1.0),
    "b0_drift": ((None, "max_offset_hz"), 1.0),
    "gradient_delay": ((None, "max_delay_dwell"), 1.0),
    "gradient_scaling_drift": ((None, "max_scale"), 1.0),
    "mechanical_vibration": ((None, "max_amplitude_mm"), 1.0),
    "correlated_coil_noise": ((None, "sigma_max"), 1.0),
    "emi_burst": ((None, "sigma_max"), 1.0),
    "receiver_gain_drift": ((None, "step_max_db"), 1.0),
    "coil_dropout": ((None, "max_fraction"), 1.0),
    "gfactor_noise": ((None, "sigma_max"), 1.0),
    "pi_residual_aliasing": ((None, "eps_max"), 1.0),
    "cs_recon_artifacts": ((None, "r_max"), 1.0),
    "zero_fill_interp": ((None, "max_fraction"), 1.0),
    "gradwarp_residual": ((None, "eps_max"), 1.0),
    "rigid_motion": ((None, "max_translation_px"), 1.0),
    "gibbs": ((None, "min_truncation_radius"), 1.0),
    "b0": ((None, "max_strength"), 1.0),
    "b0_geometric_distortion": ((None, "max_b0_strength"), 1.0),
    "eddy": ((None, "linear_phase_scale"), 1.0),
    "chemical_shift": ((None, "max_shift_pixels"), 1.0),
    "spike": ((None, "max_probability"), 1.0),
    "gradient_nonlinearity": ((None, "max_severity"), 1.0),
    "magic_angle": ((None, "t2_magic"), 1.0),
    "imperfect_fat_sat": ((None, "max_db0_hz"), 1.0),
    "imperfect_inversion": ((None, "max_inefficiency"), 1.0),
    "sar_limited_flip": ((None, "max_reduction_deg"), 1.0),
    "tse_t2_blur": ((None, "max_train_over_t2"), 1.0),
    "t1_shot_blur": ((None, "max_train_over_t1"), 1.0),
    "offres_blur_noncartesian": ((None, "max_cycles"), 1.0),
    "fov_foldover": ((None, "max_reduction"), 1.0),
    "flow_ghosts": ((None, "max_depth"), 1.0),
    "flow_voids": ((None, "max_velocity_cm_s"), 1.0),
    "inflow_enhancement": ((None, "max_velocity_cm_s"), 1.0),
    "adc_quantization": ((None, "min_enob"), 1.0),
    "antialias_rolloff": ((None, "max_inverse_cutoff"), 1.0),
    "slice_crosstalk": ((None, "max_sigma_per_thickness"), 1.0),
    "spin_history": ((None, "peak_dz_mm"), 1.0),
    "chemical_shift_phase": ((None, "max_te_sweep"), 1.0),
    "recon_apodization": ((None, "max_strength"), 1.0),
    "field_contrast": ((None, "floor_field_T"), 1.0),
    "adc_clipping": ((None, "min_clip_level"), 1.0),
}

# flow_voids saturates (washout = 0) for any velocity above 2 thk / TE, which 1 mm multislice voxels hit early.
# gibbs: 0.085 and half of it both keep only the DC sample of a 16-wide test grid.
_PROBE = {"flow_voids": 1.0, "gibbs": 0.5}

# Axes whose range cannot be extended honestly, and why. The reasons are hand-written; only
# the names are tested, as the complement of DECLARED in the registry.
RESTRICTED_TO_DECLARED_RANGE: dict[str, str] = {
    "b1_transmit": "mean calibration makes kappa - 1 nonlinear in the map peak",
    "dielectric_b1": "mean calibration makes kappa - 1 nonlinear in the standing-wave amplitude",
    "through_plane_pve": "the primary (slice-profile sigma) drives multislice only; 3-D uses the kz window",
    "off_resonance_slice_shift": "peak_hz kwarg not yet declared; outside this task's list",
    "susceptibility": "primary is a one_minus_exp co-parameter ramp, no linear endpoint",
}
_THETAS = (0.01, 0.3, 0.7, 1.0)

_CASES = [
    (name, end, kw)
    for name, (kws, _) in DECLARED.items()
    for end, kw in enumerate(kws)
    if kw is not None
]


def _spec(name: str):
    return DEGRADATION_REGISTRY[name]


def _input(name: str, mode: M, coils: int = 2) -> torch.Tensor:
    ch = coils if Requirement.MULTICOIL in _spec(name).requires else 1
    shape = (1, ch, 64, 64) if mode is M.SLICE_2D else (1, ch, 4, 16, 16)
    g = torch.Generator().manual_seed(3)
    return torch.complex(torch.randn(shape, generator=g), torch.randn(shape, generator=g))


def _geometry(mode: M) -> AcquisitionGeometry | None:
    if mode is M.SLICE_2D:
        return None
    return AcquisitionGeometry(mode=mode, voxel_mm=(1.0, 1.0, 1.0))


def _kernels(name: str):
    for impl in _spec(name).implementations.values():
        yield getattr(impl, "kernel", impl)


def test_declared_count() -> None:
    declared = {
        n: (s.severity.primary.endpoint_kwargs, s.severity.primary.endpoint_scale)
        for n, s in DEGRADATION_REGISTRY.items()
        if any(s.severity.primary.endpoint_kwargs)
    }
    assert declared == DECLARED
    assert len(DECLARED) == 65


def test_every_axis_is_declared_or_restricted() -> None:
    assert not set(DECLARED) & set(RESTRICTED_TO_DECLARED_RANGE)
    assert set(DECLARED) | set(RESTRICTED_TO_DECLARED_RANGE) == set(DEGRADATION_REGISTRY)


@pytest.mark.parametrize(("name", "end", "kw"), _CASES)
def test_declared_value_matches_kernel_default(name: str, end: int, kw: str) -> None:
    p = _spec(name).severity.primary
    value = (p.at_theta_min, p.at_theta_max)[end] * p.endpoint_scale
    for kernel in _kernels(name):
        param = inspect.signature(kernel).parameters.get(kw)
        if param is not None and param.default is not inspect.Parameter.empty:
            assert param.default == pytest.approx(value), kernel.__name__


@pytest.mark.parametrize(("name", "end", "kw"), _CASES)
def test_kwarg_at_declared_value_is_identity(name: str, end: int, kw: str) -> None:
    spec = _spec(name)
    p = spec.severity.primary
    value = (p.at_theta_min, p.at_theta_max)[end] * p.endpoint_scale
    for mode in sorted(spec.supported_modes, key=str):
        x = _input(name, mode)
        geom = _geometry(mode)
        for theta in _THETAS:
            base = apply_degradation(name, x, theta, seed=3, geometry=geom)
            via = apply_degradation(name, x, theta, seed=3, geometry=geom, **{kw: value})
            assert torch.equal(base, via), (name, mode, theta)


@pytest.mark.parametrize(("name", "end", "kw"), _CASES)
def test_kwarg_moves_the_axis(name: str, end: int, kw: str) -> None:
    # Runs in every mode the identity test does, so a kernel that swallows the kwarg cannot pass.
    spec = _spec(name)
    p = spec.severity.primary
    value = (p.at_theta_min, p.at_theta_max)[end] * p.endpoint_scale
    probe = value * 0.5 if value else 0.5
    probe = _PROBE.get(name, probe)
    if p.valid_min is not None and p.valid_min < value and probe <= p.valid_min:
        probe = (value + p.valid_min) / 2  # stay strictly inside the validity bounds
    for mode in sorted(spec.supported_modes, key=str):
        x = _input(name, mode, coils=4)  # coil_dropout counts whole coils, so 2 is too coarse
        geom = _geometry(mode)
        # Discrete effects (a spike, a k-space line, a coil) can round away at small theta, so the
        # strongest theta must move and one more must (a theta = 0 end has no say at theta = 1).
        strongest = 1.0 if end == 1 else 0.3
        moves = {}
        for theta in _THETAS:
            if end == 0 and theta == 1.0:
                continue
            base = apply_degradation(name, x, theta, seed=3, geometry=geom)
            moved = apply_degradation(name, x, theta, seed=3, geometry=geom, **{kw: probe})
            moves[theta] = not torch.equal(base, moved)
        assert moves[strongest] and sum(moves.values()) >= 2, (name, mode, moves)


@pytest.mark.parametrize("name", sorted(DECLARED))
def test_validity_bounds_contain_the_declared_range(name: str) -> None:
    p = _spec(name).severity.primary
    lo, hi = sorted((p.at_theta_min, p.at_theta_max))
    if p.valid_min is not None:
        assert p.valid_min <= lo
    if p.valid_max is not None:
        assert hi <= p.valid_max


def test_acceleration_axes_cannot_go_below_one() -> None:
    for name in ("cartesian_undersamp", "vd_undersamp", "vd_cartesian", "radial_undersamp"):
        assert _spec(name).severity.primary.valid_min == 1.0


def test_override_table_matches_kernels() -> None:
    assert {
        "gradient_nonlinearity": frozenset({"severity"}),
        "rigid_motion": frozenset({"max_translation", "max_rotation"}),
        "b0": frozenset({"strength"}),
        "b0_geometric_distortion": frozenset({"b0_strength"}),
        "chemical_shift": frozenset({"shift_pixels"}),
        "gibbs": frozenset({"truncation_fraction"}),
        "spike": frozenset({"probability"}),
        "eddy": frozenset({"strength"}),
    } == PHYSICAL_OVERRIDE_KWARGS
    for name, kws in PHYSICAL_OVERRIDE_KWARGS.items():
        for kw in kws:
            hit = False
            for kernel in _kernels(name):
                sig = inspect.signature(kernel).parameters
                src = inspect.getsource(inspect.getmodule(kernel))
                hit = hit or kw in sig or f'kw.get("{kw}"' in src
            assert hit, f"{name}: no kernel reads {kw!r}"


def _honest_pair(p, end: int) -> tuple[float, float, float, float]:
    """``(theta_ref, theta_alt, kwarg_alt, scale)``: the same physical value reached two ways.

    Preferred: the doubled extent at half theta. When that extent leaves the declared validity
    bounds (partial_fourier, coil_dropout, zero_fill_interp, adc_clipping), the in-range form:
    the default at theta = 0.5 against the kwarg set to that value at theta = 1.
    """
    lo, hi, scale = p.at_theta_min, p.at_theta_max, p.endpoint_scale
    if end == 0:
        return 0.5, 0.25, (p.value_at(0.5) - 0.25 * hi) / 0.75, scale
    alt = hi * hi / lo if p.curve == "geometric" else lo + 2.0 * (hi - lo)
    lo_b, hi_b = p.valid_min, p.valid_max
    if (lo_b is not None and alt < lo_b) or (hi_b is not None and alt > hi_b):
        return 0.5, 1.0, p.value_at(0.5), scale
    return 1.0, 0.5, alt, scale


# Primary endpoints that cannot be checked by trading theta: a co-varying parameter follows the
# native theta, so theta = 0.5 vs 1 differs by construction. strict xfail keeps the list honest.
COVARYING_UNVERIFIABLE = frozenset(
    {
        "rigid_motion",
        "iq_ghost",
        "stimulated_echo",
        "resolution_snr",
        "respiratory_motion",
        "local_intermittent_motion",
        "emi_burst",
        "receiver_gain_drift",
    }
)
_HONEST_CASES = [
    pytest.param(
        *c,
        marks=pytest.mark.xfail(strict=True, reason="co-varying parameter follows native theta"),
    )
    if c[0] in COVARYING_UNVERIFIABLE
    else c
    for c in _CASES
]


@pytest.mark.parametrize(("name", "end", "kw"), _HONEST_CASES)
def test_endpoint_kwarg_is_the_physical_endpoint(name: str, end: int, kw: str) -> None:
    # Doubling the extent and halving theta must land on the same physical value as the default
    # at theta = 1 (mirrored for a theta = 0 end): the kwarg ramps the primary linearly.
    spec = _spec(name)
    p = spec.severity.primary
    t_ref, t_alt, alt, scale = _honest_pair(p, end)
    for mode in sorted(spec.supported_modes, key=str):
        x = _input(name, mode)
        geom = _geometry(mode)
        ref = apply_degradation(name, x, t_ref, seed=3, geometry=geom)
        via = apply_degradation(name, x, t_alt, seed=3, geometry=geom, **{kw: alt * scale})
        assert torch.equal(ref, via), (name, mode)


# axis -> (override kwargs, endpoint kwarg that clashes with them, a non-default value)
_OVERRIDES = {
    "rigid_motion": ({"max_translation": 1.0}, "max_translation_px", 5.0),
    "gibbs": ({"truncation_fraction": 0.5}, "min_truncation_radius", 0.2),
    "b0": ({"strength": 0.5}, "max_strength", 2.0),
    "b0_geometric_distortion": ({"b0_strength": 0.5}, "max_b0_strength", 2.0),
    "eddy": ({"strength": 1.0}, "linear_phase_scale", 2.0),
    "chemical_shift": ({"shift_pixels": 2.0}, "max_shift_pixels", 7.0),
    "spike": ({"probability": 0.01}, "max_probability", 0.1),
    "gradient_nonlinearity": ({"severity": 1e-5}, "max_severity", 1e-4),
    "adc_clipping": ({"depth": 0.5}, "min_clip_level", 0.3),
}
_OVERRIDE_CASES = [(n, m) for n in _OVERRIDES for m in sorted(_spec(n).supported_modes, key=str)]


@pytest.mark.parametrize(("name", "mode"), _OVERRIDE_CASES)
def test_override_with_changed_endpoint_raises(name: str, mode: M) -> None:
    override, kw, value = _OVERRIDES[name]
    x, geom = _input(name, mode), _geometry(mode)
    # Volume gradient_nonlinearity rejects the severity override outright, before any endpoint check.
    with pytest.raises(ValueError, match=f"{kw}|severity"):
        apply_degradation(name, x, 0.7, seed=3, geometry=geom, **override, **{kw: value})


@pytest.mark.parametrize(("name", "mode"), _OVERRIDE_CASES)
def test_override_with_default_endpoint_still_works(name: str, mode: M) -> None:
    if name == "gradient_nonlinearity" and mode is not M.SLICE_2D:
        pytest.skip("volume gradient_nonlinearity rejects the severity override outright")
    override, _kw, _ = _OVERRIDES[name]
    x, geom = _input(name, mode), _geometry(mode)
    out = apply_degradation(name, x, 0.7, seed=3, geometry=geom, **override)
    assert torch.isfinite(out.abs()).all()


@pytest.mark.parametrize("mode", sorted(_spec("chemical_shift").supported_modes, key=str))
def test_chemical_shift_physical_branch_rejects_changed_endpoint(mode: M) -> None:
    x, geom = _input("chemical_shift", mode), _geometry(mode)
    kw = {"b0_T": 3.0, "bw_per_pixel_hz": 200.0}
    apply_degradation("chemical_shift", x, 0.7, seed=3, geometry=geom, **kw)
    with pytest.raises(ValueError, match="max_shift_pixels"):
        apply_degradation(
            "chemical_shift", x, 0.7, seed=3, geometry=geom, max_shift_pixels=7.0, **kw
        )
