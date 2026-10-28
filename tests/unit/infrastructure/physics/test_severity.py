"""Tests for the declared severity SSOT.

The point of SeveritySpec is that "what noise range did this sweep cover?" has an
answer that is not "read the degrader's function body". These tests pin the two
things that make the answer trustworthy: every degradation declares a range, and
the declaration matches what the degrader actually does.
"""

from __future__ import annotations

import itertools

import pytest
import torch

from spectramr.infrastructure.physics.degradations import DEGRADATION_REGISTRY, apply_degradation
from spectramr.infrastructure.physics.degradations.registry_queries import (
    requirement_free_degradations,
    slice_capable_degradations,
)
from spectramr.infrastructure.physics.severity import (
    PhysicalParam,
    SeverityDirection,
    SeveritySpec,
)


class TestPhysicalParam:
    def test_affine_interpolation(self) -> None:
        p = PhysicalParam("acceleration", "x", 1.0, 8.0)
        assert p.value_at(0.0) == pytest.approx(1.0)
        assert p.value_at(1.0) == pytest.approx(8.0)
        assert p.value_at(0.5) == pytest.approx(4.5)

    def test_a_decreasing_param_is_labelled_decreasing(self) -> None:
        """SNR runs 40 dB -> 2 dB: the physical MINIMUM is the WORST case.

        Reporting a bare min/max column would invert the reader's interpretation,
        which is why the fields are named by severity endpoint, not by magnitude.
        """
        snr = PhysicalParam("snr", "dB", 40.0, 2.0)
        assert snr.direction is SeverityDirection.DECREASING
        assert snr.value_at(1.0) == pytest.approx(2.0)  # worst
        assert snr.value_at(0.0) == pytest.approx(40.0)  # clean


class TestSeveritySpec:
    def test_theta_grid_spans_both_endpoints(self) -> None:
        spec = SeveritySpec((PhysicalParam("sigma", "px", 0.0, 4.0),), n_steps=5)
        grid = spec.theta_grid()
        assert len(grid) == 5
        assert grid[0] == pytest.approx(0.0)
        assert grid[-1] == pytest.approx(1.0)
        assert spec.step == pytest.approx(0.25)

    def test_physical_value_tracks_the_grid(self) -> None:
        spec = SeveritySpec((PhysicalParam("sigma", "px", 0.0, 4.0),), n_steps=5)
        assert [spec.physical_value(t) for t in spec.theta_grid()] == pytest.approx(
            [0.0, 1.0, 2.0, 3.0, 4.0]
        )

    def test_a_spec_with_no_params_raises(self) -> None:
        """An undeclared severity axis is exactly what this class exists to prevent."""
        with pytest.raises(ValueError, match="at least one physical parameter"):
            SeveritySpec(())

    def test_co_varying_params_are_declared_not_hidden(self) -> None:
        """rigid_motion ramps translation AND rotation. Both must be visible."""
        spec = SeveritySpec(
            (
                PhysicalParam("max_translation", "px", 0.0, 10.0),
                PhysicalParam("max_rotation", "rad", 0.0, 0.1),
            )
        )
        assert spec.primary.name == "max_translation"
        assert [p.name for p in spec.co_varying] == ["max_rotation"]
        assert spec.physical_value(1.0, "max_rotation") == pytest.approx(0.1)

    def test_an_undeclared_param_raises_rather_than_guessing(self) -> None:
        spec = SeveritySpec((PhysicalParam("sigma", "px", 0.0, 4.0),))
        with pytest.raises(KeyError, match="not a declared parameter"):
            spec.physical_value(0.5, "rotation")

    def test_degenerate_grid_raises(self) -> None:
        with pytest.raises(ValueError, match="n_steps must be >= 2"):
            SeveritySpec((PhysicalParam("sigma", "px", 0.0, 4.0),), n_steps=1)


class TestEveryDegradationDeclaresItsRange:
    """The coverage gate: no degradation may exist without a declared severity."""

    def test_all_registry_entries_have_a_severity_spec(self) -> None:
        from spectramr.infrastructure.physics.degradations import DEGRADATION_REGISTRY

        # No hard-coded size: the gate is that EVERY entry declares a range,
        # which must keep holding as axes are added (D28 resolution_snr,
        # D29-D31 motion types). Code uniqueness is pinned separately, by
        # test_digital_twin_waves.TestWave4_Registry.
        assert DEGRADATION_REGISTRY, "registry is empty"
        for name, spec in DEGRADATION_REGISTRY.items():
            assert isinstance(spec.severity, SeveritySpec), name
            assert spec.severity.primary.name, f"{name}: primary param has no name"
            assert spec.severity.primary.units, f"{name}: primary param has no units"

    def test_a_degradation_without_a_declared_severity_fails_at_import(self) -> None:
        """A family entry has no severity default -- you cannot add a degradation
        without writing down what its severity means physically."""
        from spectramr.config.schemas.enums import ArtifactOrigin
        from spectramr.infrastructure.physics.degradations.families import native_2d

        with pytest.raises(TypeError, match="severity"):
            native_2d(  # type: ignore[call-arg]
                code="D99",
                name="not_a_real_degradation",
                family="noise",
                fn=lambda x, theta, seed=0: x,
                description="bogus",
                affects=frozenset({"magnitude"}),
                origin=ArtifactOrigin.SCANNER,
            )

    def test_the_known_decreasing_axes_are_labelled_as_such(self) -> None:
        """These degradations ramp their physical parameter DOWN with severity.

        Mislabelling any of these silently inverts the reported range.
        """
        from spectramr.infrastructure.physics.degradations import DEGRADATION_REGISTRY

        decreasing = {
            name
            for name, s in DEGRADATION_REGISTRY.items()
            if s.severity.direction is SeverityDirection.DECREASING
        }
        # radial_undersamp was re-parameterised to acceleration R (1x -> 8x),
        # so it is now INCREASING, not DECREASING (#246).
        assert decreasing == {
            "complex_gaussian",  # SNR 40 dB -> 2 dB
            "gibbs",  # keeps 100% -> 8.5% of k-space radius (#246)
            "partial_fourier",  # keeps 100% -> 62.5% of ky
            "adc_clipping",  # clip level 100% -> 10% of the k-space peak
            "adc_quantization",  # 16 bit -> 4 bit
            "field_contrast",  # target field 3 T -> 0.05 T
        }


class TestDeclarationMatchesTheDegrader:
    """The anti-drift gate: a declared range that lies is worse than none.

    These re-derive the physical value from the degrader's own formula and check the
    spec agrees. If someone retunes a ramp without updating the spec, this fails.
    """

    def test_complex_gaussian_snr_matches_the_degrader_formula(self) -> None:
        from spectramr.infrastructure.physics.degradations import DEGRADATION_REGISTRY

        sev = DEGRADATION_REGISTRY["complex_gaussian"].severity
        # degrader: snr_db = snr_max_db + (snr_min_db - snr_max_db) * theta
        snr_max_db, snr_min_db = 40.0, 2.0
        for theta in (0.0, 0.25, 0.5, 1.0):
            expected = snr_max_db + (snr_min_db - snr_max_db) * theta
            assert sev.physical_value(theta) == pytest.approx(expected)

    def test_cartesian_undersampling_r_matches_the_degrader_formula(self) -> None:
        from spectramr.infrastructure.physics.degradations import DEGRADATION_REGISTRY

        sev = DEGRADATION_REGISTRY["cartesian_undersamp"].severity
        # degrader: R = 1.0 + (R_max - 1.0) * theta,  R_max = 8.0
        for theta in (0.0, 0.5, 1.0):
            assert sev.physical_value(theta) == pytest.approx(1.0 + 7.0 * theta)

    def test_partial_fourier_fraction_matches_the_degrader_formula(self) -> None:
        from spectramr.infrastructure.physics.degradations import DEGRADATION_REGISTRY

        sev = DEGRADATION_REGISTRY["partial_fourier"].severity
        # degrader: frac = 1.0 - (1.0 - min_fraction) * theta,  min_fraction = 0.625
        for theta in (0.0, 0.5, 1.0):
            assert sev.physical_value(theta) == pytest.approx(
                1.0 - (1.0 - 0.625) * theta
            )

    def test_gibbs_truncation_matches_the_degrader_formula(self) -> None:
        from spectramr.infrastructure.physics.degradations import DEGRADATION_REGISTRY

        sev = DEGRADATION_REGISTRY["gibbs"].severity
        # delegate: trunc = 1.0 - theta * 0.915  (theta=1 keeps 8.5% radius, #246)
        for theta in (0.0, 0.5, 1.0):
            assert sev.physical_value(theta) == pytest.approx(1.0 - theta * 0.915)


# ──────────────────────────────────────────────────────────────────────
# Behavioural conformance: the declared axis vs what the degrader DOES.
#
# The tests above pin the *declarations*. They cannot fail when a degrader's physics
# is wrong, because they never call one -- they re-type the literal from the signature
# and compare it to the spec. Transcription is not verification, and the gap let four
# broken axes into the SSOT:
#
#   partial_fourier    theta=0 returned a BLACK image. `mask[..., -cut:, :] = 0` with
#                      cut == 0 is `mask[..., 0:, :] = 0` (-0 == 0) -- the whole mask.
#                      FIXED here.
#   chemical_shift     the shifted fat was added on top of the FULL water signal, a
#                      (1 + fat_fraction) gain, so theta=0 came back 1.35x bright and
#                      rel-L2 DECREASED as severity rose: an inverted axis, on which
#                      every L2-family metric (PSNR/NMSE/MSE) scores backwards.
#                      FIXED here (water/fat partition).
#   concomitant_phase  a pure image-domain phase. |x * exp(i*phi)| == |x|, and the sweep
#                      grades .abs() -- so it is exactly zero-degradation at every theta.
#   magic_angle        likewise flat at every theta (T2 modulation, MSK-only).
#
# The last two are INERT: they contribute a flat axis on which every metric scores
# identically at every severity, so they cannot separate any metric from any other --
# pitfall #16 at the axis level. They are recorded as strict xfails below rather than
# skipped, so the day one of them starts doing something, this suite says so.
#
# Every one of these is a plausible-null: nothing crashes, the leaderboard just means
# nothing. These tests call the degraders.
# ──────────────────────────────────────────────────────────────────────

# These checks run on a single-coil 2-D slice; volume-only and multi-coil axes are covered by
# their own mode-rejection and coil-stack tests under degradations/catalogue/.
_ALL_DEGRADERS = sorted(
    set(slice_capable_degradations()) & set(requirement_free_degradations())
)

#: Axes that do not move the magnitude at all. Not a waiver -- a standing bug report.
#: See https://github.com/adnaneGdihi/spectramr/issues (inert severity axes).
_INERT_UNDER_MAGNITUDE = {
    "concomitant_phase": "pure image-domain phase; |x*exp(i.phi)| == |x|",
    "magic_angle": "T2 modulation (MSK-only); no effect on this phantom",
}

_THETA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def _phantom() -> torch.Tensor:
    torch.manual_seed(0)
    x = torch.zeros(1, 1, 128, 128, dtype=torch.complex64)
    x[..., 32:96, 32:96] = 1.0 + 0.3j
    return x


def _rel_l2(name: str, theta: float, x: torch.Tensor) -> float:
    """Degradation of the quantity the sweep actually grades (sim2rank: ``out.abs()``)."""
    y = apply_degradation(name, x, theta)
    y = y.abs() if y.is_complex() else y
    return float((y - x.abs()).norm() / x.abs().norm())


@pytest.mark.parametrize("name", _ALL_DEGRADERS)
def test_theta_zero_is_the_cleanest_point_on_the_axis(name: str) -> None:
    """theta=0 is the anchor every severity is measured from; nothing may be cleaner.

    Catches both an anchor that is itself destroyed (partial_fourier returned black)
    and an inverted axis (chemical_shift got *cleaner* as severity rose).

    Note this is deliberately weaker than "theta=0 is the identity": three axes carry a
    declared, honest floor at theta=0 -- complex_gaussian's 40 dB SNR ceiling,
    radial_undersamp's 200 spokes (below Nyquist for a 256 grid), gibbs's k-space radius
    window -- and those are properties of the declared range, not bugs in the ramp.
    """
    x = _phantom()
    curve = [_rel_l2(name, t, x) for t in _THETA_GRID]
    assert curve[0] <= min(curve) + 1e-9, (
        f"{name}: theta=0 (rel-L2 {curve[0]:.4f}) is not the cleanest point on the axis "
        f"-- curve {[f'{c:.4f}' for c in curve]}. The reference the metrics grade "
        "against is more degraded than a degraded sample."
    )


@pytest.mark.parametrize("name", _ALL_DEGRADERS)
def test_severity_is_monotone_in_theta(name: str) -> None:
    """Degradation must not DECREASE as theta rises, or the severity axis lies."""
    x = _phantom()
    curve = [_rel_l2(name, t, x) for t in _THETA_GRID]
    for i in range(len(curve) - 1):
        assert curve[i] <= curve[i + 1] + 1e-9, (
            f"{name}: rel-L2 falls from {curve[i]:.4f} (theta={_THETA_GRID[i]}) to "
            f"{curve[i + 1]:.4f} (theta={_THETA_GRID[i + 1]}). A metric that correctly "
            "tracks severity is penalised on a non-monotone axis."
        )


@pytest.mark.parametrize("name", _ALL_DEGRADERS)
def test_the_axis_is_visible_in_the_quantity_the_sweep_grades(name: str) -> None:
    """An axis invisible under .abs() cannot separate any metric from any other."""
    if name in _INERT_UNDER_MAGNITUDE:
        pytest.xfail(f"{name} is inert under magnitude: {_INERT_UNDER_MAGNITUDE[name]}")
    x = _phantom()
    rel = _rel_l2(name, 1.0, x)
    assert rel > 1e-4, (
        f"{name}: at theta=1 the magnitude is unchanged (rel-L2 {rel:.2e}), and the "
        "sweep grades magnitudes. This axis contributes a flat, zero-degradation column "
        "to the leaderboard."
    )


# ── severity ramp over diffusion timesteps ────────────────────────────────


class TestShapeSeverityRatio:
    """The counterpart of undersampling's acceleration_schedule, on severity."""

    def test_linear_is_the_identity_bit_for_bit(self):
        """The blocker property, and the reason this is safe to land.

        ``linear`` is the default; every existing digital-twin arm and every
        cold-diffusion arm routes through it. Computing ``ratio ** 1.0`` instead of
        returning ``ratio`` would be equal for most inputs and differ in the last
        bit for some, silently moving a corrupted volume with nothing to attribute
        the change to. Asserted on the hex repr, not with ``approx``.
        """
        from spectramr.infrastructure.physics.severity import shape_severity_ratio

        for r in (0.0, 0.1, 1 / 3, 0.53, 0.9999, 1.0):
            assert shape_severity_ratio(r, "linear").hex() == float(r).hex()

    @pytest.mark.parametrize(
        "schedule", ["linear", "polynomial", "power_law", "exponential", "step"]
    )
    def test_endpoints_are_anchored(self, schedule):
        """Whatever the shape, t=0 is clean and t=T is fully degraded. A schedule
        that moved the endpoints would silently rescale the declared range."""
        from spectramr.infrastructure.physics.severity import shape_severity_ratio

        assert shape_severity_ratio(0.0, schedule) == 0.0
        assert shape_severity_ratio(1.0, schedule) == 1.0

    @pytest.mark.parametrize(
        "schedule", ["linear", "polynomial", "power_law", "exponential", "step"]
    )
    def test_monotone_non_decreasing(self, schedule):
        from spectramr.infrastructure.physics.severity import shape_severity_ratio

        vals = [shape_severity_ratio(i / 20, schedule) for i in range(21)]
        assert all(a <= b + 1e-12 for a, b in itertools.pairwise(vals))

    def test_power_law_and_polynomial_agree(self):
        """Undersampling's own ramp handles 'power_law' and lets 'polynomial' fall
        through to linear, so a declared polynomial schedule is inert there. Both
        are ratio**power; this pins the honest reading rather than the drift."""
        from spectramr.infrastructure.physics.severity import shape_severity_ratio

        for r in (0.25, 0.5, 0.75):
            a = shape_severity_ratio(r, "power_law", 3.0)
            assert a == shape_severity_ratio(r, "polynomial", 3.0)
            assert a == pytest.approx(r**3.0)

    def test_step_is_binary_at_the_midpoint(self):
        """Undersampling's 'step' indexes an explicit acceleration_range LIST and
        falls back to base/max when none was declared. Severity has no list, only
        the (vmin, vmax) pair, so only that fallback carries over."""
        from spectramr.infrastructure.physics.severity import shape_severity_ratio

        assert shape_severity_ratio(0.49, "step") == 0.0
        assert shape_severity_ratio(0.5, "step") == 1.0

    def test_out_of_range_is_clamped_not_extrapolated(self):
        """A caller passing t > T should not silently receive a severity ABOVE the
        declared maximum -- the range is a declaration, not a suggestion."""
        from spectramr.infrastructure.physics.severity import shape_severity_ratio

        assert shape_severity_ratio(1.4, "linear") == 1.0
        assert shape_severity_ratio(-0.2, "linear") == 0.0

    def test_unknown_schedule_raises(self):
        from spectramr.infrastructure.physics.severity import shape_severity_ratio

        with pytest.raises(ValueError, match="unknown severity schedule"):
            shape_severity_ratio(0.5, "cosine")

    def test_non_positive_power_raises(self):
        from spectramr.infrastructure.physics.severity import shape_severity_ratio

        with pytest.raises(ValueError, match="power must be"):
            shape_severity_ratio(0.5, "power_law", 0.0)


class TestAgreesWithTheUndersamplingRamp:
    """The two ramps that share ``AccelerationSchedule`` must compute one curve.

    ``shape_severity_ratio`` is documented as "the counterpart of
    ``KSpaceAccelerator._normalized_progress``", and sharing the schedule
    vocabulary is only worth anything if the two agree about what each member
    means. They did not: undersampling let ``polynomial`` fall through to linear
    while severity computed ``ratio ** power``, so one arm ramping both got two
    different curves from one declared name (#789). These pin the agreement so it
    cannot drift apart again.
    """

    POWER = 3.0
    T = 9

    def _undersampling_curve(self, schedule: str) -> list[float]:
        from spectramr.infrastructure.physics.sampling import create_kspace_accelerator

        accel = create_kspace_accelerator(
            "random_cartesian",
            num_timesteps=self.T,
            max_acceleration=8.0,
            base_acceleration=1.0,
            center_fraction=0.08,
            acceleration_schedule=schedule,
            schedule_power=self.POWER,
            seed=42,
        )
        inner = getattr(accel, "accelerator", accel)
        return [inner._normalized_progress(t) for t in range(self.T)]

    @pytest.mark.parametrize("schedule", ["linear", "polynomial", "power_law", "exponential"])
    def test_curves_match(self, schedule: str) -> None:
        from spectramr.infrastructure.physics.severity import shape_severity_ratio

        severity = [
            shape_severity_ratio(t / (self.T - 1), schedule, self.POWER) for t in range(self.T)
        ]
        assert self._undersampling_curve(schedule) == pytest.approx(severity)

    def test_step_is_the_one_deliberate_divergence(self) -> None:
        """``step`` differs on purpose, and that difference is documented.

        Severity has only a ``(vmin, vmax)`` pair, so its ``step`` is binary at
        the midpoint. Undersampling's ``step`` indexes an explicit
        ``acceleration_range`` LIST, so its ramp must hand the consumer the raw
        ratio. Pinning this stops a future "consistency" cleanup from unifying
        them and silently re-indexing every step ladder in the corpus.
        """
        from spectramr.infrastructure.physics.severity import shape_severity_ratio

        assert shape_severity_ratio(0.75, "step") == 1.0
        assert self._undersampling_curve("step")[6] == pytest.approx(0.75)

    def test_the_two_vocabularies_are_the_same_size(self) -> None:
        """Neither ramp may quietly grow a member the other does not implement."""
        from spectramr.config.schemas.enums import AccelerationSchedule
        from spectramr.infrastructure.physics.sampling import _SUPPORTED_SCHEDULES

        assert {s.value for s in AccelerationSchedule} == _SUPPORTED_SCHEDULES
