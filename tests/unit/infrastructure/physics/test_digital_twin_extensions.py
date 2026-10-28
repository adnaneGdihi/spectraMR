"""Unit tests for the degradations package (D1-D20 with deterministic seed).

Pin two contracts that the meta-evaluation pipeline depends on:

1. **Determinism**: same ``(theta, seed)`` → identical output.
2. **Severity monotonicity proxy**: ``theta=0`` should approximately
   recover the input; ``theta=1`` should produce a strictly different
   output for every degradation.
"""
from __future__ import annotations

import math

import pytest
import torch

from spectramr.infrastructure.physics.degradations import (
    DEGRADATION_REGISTRY,
    apply_degradation,
    degrade_complex_gaussian_noise,
    degrade_radial_undersampling,
    degrade_susceptibility,
    list_degradations,
)
from spectramr.infrastructure.physics.degradations.registry_queries import (
    requirement_free_degradations,
    slice_capable_degradations,
)

# These harnesses feed a single-coil 2-D slice. Volume-only and multi-coil axes are covered by
# their own mode-rejection and coil-stack tests under degradations/catalogue/.
_SINGLE_COIL_SLICE_AXES = sorted(
    set(slice_capable_degradations()) & set(requirement_free_degradations())
)


def _make_phantom(H: int = 32, W: int = 32, complex_: bool = True) -> torch.Tensor:
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij"
    )
    p = ((x ** 2 + y ** 2) < 0.6).float().unsqueeze(0).unsqueeze(0)
    return torch.complex(p, torch.zeros_like(p)) if complex_ else p


class TestDegradationRegistry:
    def test_registry_covers_d1_to_d20(self):
        # D6 (rigid_motion duplicates D7) and D19 (paired data) are intentionally
        # delegated/excluded; we still expect ≥17 distinct registered axes.
        assert len(DEGRADATION_REGISTRY) >= 17
        assert len(list_degradations()) == len(DEGRADATION_REGISTRY)

    def test_unknown_degradation_raises(self):
        with pytest.raises(KeyError):
            apply_degradation("nonexistent", _make_phantom(), 0.5)


class TestDeterminism:
    """Same (theta, seed) MUST produce identical outputs."""

    @pytest.mark.parametrize(
        "name",
        [
            "complex_gaussian", "rician", "cartesian_undersamp", "vd_undersamp",
            "radial_undersamp", "pulsatile_motion", "nyquist_ghost",
            "t2star_blur", "slice_pve", "susceptibility", "coil_miscal",
            "geometric_warp", "bias", "gibbs", "b0", "eddy", "spike",
            "chemical_shift",
        ],
    )
    def test_same_seed_same_output(self, name):
        x = _make_phantom()
        out_a = apply_degradation(name, x, theta=0.7, seed=42)
        out_b = apply_degradation(name, x, theta=0.7, seed=42)
        assert torch.allclose(out_a, out_b), f"{name} is not deterministic"

    def test_different_seeds_differ(self):
        x = _make_phantom()
        a = apply_degradation("complex_gaussian", x, theta=0.5, seed=0)
        b = apply_degradation("complex_gaussian", x, theta=0.5, seed=1)
        assert not torch.allclose(a, b)


class TestSeverityProgression:
    """Output should grow more different from input as theta increases."""

    @pytest.mark.parametrize(
        "name",
        [
            "complex_gaussian", "rician", "vd_undersamp", "radial_undersamp",
            "pulsatile_motion", "nyquist_ghost", "t2star_blur", "slice_pve",
            "susceptibility", "coil_miscal", "geometric_warp", "bias",
        ],
    )
    def test_severity_increases_distance(self, name):
        x = _make_phantom()
        x_mag = x.abs()

        d0 = (apply_degradation(name, x, theta=0.0, seed=0).abs() - x_mag).pow(2).mean().item()
        d1 = (apply_degradation(name, x, theta=1.0, seed=0).abs() - x_mag).pow(2).mean().item()
        # theta=1 should be at least as different as theta=0
        # (theta=0 should approximately = identity for most ops)
        assert d1 >= d0, (
            f"{name}: distance at theta=1 ({d1:.4e}) should be ≥ "
            f"distance at theta=0 ({d0:.4e})"
        )


class TestShapePreservation:
    @pytest.mark.parametrize("name", _SINGLE_COIL_SLICE_AXES)
    def test_shape_preserved(self, name):
        x = _make_phantom(64, 64)
        out = apply_degradation(name, x, theta=0.5, seed=0)
        assert out.shape == x.shape


class TestSpecificDegradations:
    def test_complex_gaussian_increases_noise(self):
        x = _make_phantom()
        out = degrade_complex_gaussian_noise(x, theta=1.0, seed=0)
        # The residual (out - x) should have non-trivial energy at high theta
        residual = (out - x).abs()
        assert residual.var() > 1e-3, (
            f"Expected noisy residual to have variance > 1e-3, got {residual.var().item():.4e}"
        )

    def test_complex_gaussian_realises_declared_snr(self):
        # The complex noise has power 2*sigma^2, so sigma must carry 1/sqrt(2) of
        # the noise power to hit the DECLARED SNR. Without the fix the realised SNR
        # is 3.01 dB hotter than requested (critique physics 4.3).
        import math

        from spectramr.infrastructure.physics.fft_ops import fft2c

        # A large random phantom for a low-variance Monte-Carlo SNR estimate.
        x = torch.complex(torch.randn(1, 1, 96, 96), torch.randn(1, 1, 96, 96))
        target_db = 20.0
        # theta maps linearly snr_max(40)->snr_min(2); solve for theta giving 20 dB.
        theta = (40.0 - target_db) / (40.0 - 2.0)
        out = degrade_complex_gaussian_noise(x, theta=theta, seed=0)
        k_x = fft2c(x)
        noise_k = fft2c(out) - k_x  # out = ifft2c(k + noise) => fft2c(out)-k = noise
        signal_power = k_x.abs().pow(2).mean().item()
        noise_power = noise_k.abs().pow(2).mean().item()
        realised_db = 10.0 * math.log10(signal_power / noise_power)
        assert abs(realised_db - target_db) < 0.7, (
            f"realised SNR {realised_db:.2f} dB != declared {target_db} dB"
        )

    def test_radial_decreases_with_severity(self):
        x = _make_phantom()
        # More spokes (low theta) should preserve more energy than fewer spokes
        low = degrade_radial_undersampling(x, theta=0.1, seed=0)
        high = degrade_radial_undersampling(x, theta=0.95, seed=0)
        # Both lose energy vs original, but high-theta loses more
        e_orig = x.abs().pow(2).sum().item()
        e_low = low.abs().pow(2).sum().item()
        e_high = high.abs().pow(2).sum().item()
        assert e_high <= e_low <= e_orig + 1e-6

    def test_susceptibility_produces_dropout(self):
        x = _make_phantom()
        out = degrade_susceptibility(x, theta=1.0, seed=0)
        # Magnitude should be ≤ original magnitude (dropout)
        assert out.abs().max() <= x.abs().max() * 1.01


# ══════════════════════════════════════════════════════════════════════
# Degradation-simulator correctness (issues #223, #244, #245, #246, #247)
#
# The simulator is load-bearing: if an axis does not degrade, does not scale with
# severity, or corrupts its own clean anchor, then every "metric sensitivity to
# axis X" number the ranker reports is meaningless. These are the invariants that
# were violated, each pinned so it cannot silently regress.
# ══════════════════════════════════════════════════════════════════════

import itertools  # noqa: E402

from spectramr.infrastructure.physics.degradations import (  # noqa: E402
    DECLARED_THETA_ZERO_PEDESTAL,
    KNOWN_COLLINEAR_PAIRS,
    MAGNITUDE,
    PHASE,
    AcquisitionGeometry,
    AcquisitionMode,
    affected_components,
    is_identity_at_theta_zero,
    is_known_collinear,
    is_phase_only,
    phase_only_degradations,
)
from spectramr.infrastructure.physics.degradations.declarations import (  # noqa: E402
    DECLARED_PHYSICAL_FOV_MM,
)
from spectramr.infrastructure.physics.fft_ops import fft2c  # noqa: E402

_ALL_AXES = _SINGLE_COIL_SLICE_AXES


def _brain(H: int = 64, W: int = 64, textured: bool = True) -> torch.Tensor:
    """Asymmetric, textured, zero-background phantom.

    Zero background is essential: the #223 magic-angle no-op was only visible
    because the fibre patches landed in signal-free air. A phantom that fills the
    FOV would hide it. The texture matters for the orthogonality test — a smooth
    phantom flatters residual cosines.
    """
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij"
    )
    img = torch.zeros(H, W)
    img[((x / 0.75) ** 2 + (y / 0.9) ** 2) < 1.0] = 0.4
    img[((x / 0.68) ** 2 + (y / 0.83) ** 2) < 1.0] = 0.8
    img[(((x - 0.10) / 0.10) ** 2 + ((y - 0.05) / 0.30) ** 2) < 1.0] = 0.25
    img[(((x + 0.16) / 0.08) ** 2 + ((y + 0.02) / 0.26) ** 2) < 1.0] = 0.25
    for cx, cy, r, v in [
        (-0.35, 0.42, 0.09, 1.0),
        (0.41, -0.33, 0.07, 0.55),
        (-0.50, -0.20, 0.06, 0.90),
    ]:
        img[((x - cx) ** 2 + (y - cy) ** 2) < r**2] = v
    if textured:
        g = torch.Generator().manual_seed(7)
        img = img + torch.randn(H, W, generator=g) * 0.05 * (img > 0.05)
    return img.clamp(min=0).view(1, 1, H, W).to(torch.complex64)


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm())


def _as_complex(t: torch.Tensor) -> torch.Tensor:
    return t if t.is_complex() else torch.complex(t, torch.zeros_like(t))


def _mag(name: str, x: torch.Tensor, theta: float, seed: int = 0) -> torch.Tensor:
    # A declared FOV-dependent axis is measured on a phantom of its physical size.
    fov, geometry = DECLARED_PHYSICAL_FOV_MM.get(name), None
    if fov is not None:
        voxel = (5.0, fov / x.shape[-2], fov / x.shape[-1])
        geometry = AcquisitionGeometry(mode=AcquisitionMode.SLICE_2D, voxel_mm=voxel)
    out = apply_degradation(name, x, theta=theta, seed=seed, geometry=geometry)
    return _as_complex(out).abs()


class TestCleanAnchor:
    """#246 — ``D(x, theta=0)`` must be the identity, for EVERY registered axis.

    Every statistic in the sweep is anchored on the theta=0 baseline, so a
    degrader that corrupts its own clean anchor biases everything downstream.
    Two did:

    * ``gibbs`` compared an un-normalised k-space radius (max ``sqrt(2)``) against
      the retained fraction, so the nominal "no truncation" setting of 1.0 still
      zeroed the k-space corners — 22.8% of the samples — for a rel-L2 of 0.055.
    * ``radial_undersamp`` laid 200 golden-angle spokes on a 128^2 grid, which do
      not cover it: rel-L2 0.069.

    Both pedestals EXCEEDED the entire theta=1 distortion of ``partial_fourier``
    (0.088). This class is the guard that class of bug never returns unseen.
    """

    @pytest.mark.parametrize("name", _ALL_AXES)
    def test_theta_zero_is_identity(self, name: str) -> None:
        x = _brain()
        pedestal = _rel_l2(_mag(name, x, 0.0), x.abs())

        if is_identity_at_theta_zero(name):
            assert pedestal < 1e-4, (
                f"{name}: D(x, theta=0) is NOT the identity — rel-L2 {pedestal:.4f}. "
                "The clean anchor is pre-degraded; every statistic anchored on it "
                "is biased (#246)."
            )
        else:
            # The exception must be DECLARED and BOUNDED, never merely tolerated.
            bound = DECLARED_THETA_ZERO_PEDESTAL[name]
            assert pedestal < bound, (
                f"{name} declares a theta=0 pedestal of at most {bound}, measured "
                f"{pedestal:.4f}."
            )

    def test_only_complex_gaussian_declares_a_pedestal(self) -> None:
        # complex_gaussian's severity IS the SNR and its spec declares theta=0 to
        # be 40 dB — a clean scanner, not a noiseless one. Nothing else may claim
        # an exemption without a declared physical reason.
        assert set(DECLARED_THETA_ZERO_PEDESTAL) == {"complex_gaussian"}

    @pytest.mark.parametrize("name", _ALL_AXES)
    def test_severity_grows_with_theta(self, name: str) -> None:
        """An axis that does not scale with severity cannot rank anything."""
        x = _brain()
        # concomitant_phase is phase-only: score it on the complex tensor.
        if is_phase_only(name):
            ref = x
            vals = [
                _rel_l2(_as_complex(apply_degradation(name, x, theta=t, seed=0)), ref)
                for t in (0.25, 0.5, 1.0)
            ]
        else:
            ref = x.abs()
            vals = [_rel_l2(_mag(name, x, t), ref) for t in (0.25, 0.5, 1.0)]
        assert vals[0] < vals[1] < vals[2], f"{name} is not monotone in theta: {vals}"
        assert vals[-1] > 1e-3, f"{name} barely moves at theta=1: {vals[-1]:.2e}"


class TestMagicAngleSeedLottery:
    """#223 — ``magic_angle`` (D22) was an EXACT no-op at the pipeline's own seed.

    The collagen-fibre mask was two ellipses with centres drawn ~U[-1, 1]^2, i.e.
    anywhere in the FOV including the signal-free air around the anatomy. At
    seed 0 — the sweep's default — BOTH landed in the background, the modulation
    multiplied nothing, and ``torch.equal(D(x, theta), x)`` was True for every
    theta. It was a seed lottery: seeds 1/2/3/7 gave rel-L2 0.08/0.10/0.05/0.14,
    seed 0 gave exactly 0.0.

    Centres are now drawn from the SUPPORT of the image, so the lottery is
    structurally impossible. The parametrisation over seeds is the point — a
    single-seed test is what let this survive.
    """

    @pytest.mark.parametrize("seed", range(8))
    def test_fires_at_every_seed(self, seed: int) -> None:
        x = _brain()
        for theta in (0.25, 0.5, 1.0):
            out = apply_degradation("magic_angle", x, theta=theta, seed=seed)
            assert not torch.equal(out, x), (
                f"magic_angle is an EXACT no-op at seed={seed}, theta={theta} — "
                "the fibre patches missed the anatomy (#223)."
            )

    @pytest.mark.parametrize("seed", range(8))
    def test_monotone_and_material_at_every_seed(self, seed: int) -> None:
        x = _brain()
        vals = [_rel_l2(_mag("magic_angle", x, t, seed), x.abs()) for t in (0.25, 0.5, 1.0)]
        assert vals[0] < vals[1] < vals[2], f"seed {seed}: not monotone: {vals}"
        # Severity in family with the rest of the taxonomy (median axis ~0.40),
        # not the ~5x-weaker 0.05-of-FOV coverage it used to have.
        assert vals[-1] > 0.15, f"seed {seed}: magic_angle too weak at theta=1: {vals[-1]:.3f}"

    def test_theta_zero_still_identity(self) -> None:
        x = _brain()
        for seed in range(8):
            assert _rel_l2(_mag("magic_angle", x, 0.0, seed), x.abs()) < 1e-6

    def test_all_background_input_does_not_crash(self) -> None:
        """An empty support must fall back to the FOV, not divide by zero."""
        x = torch.zeros(1, 1, 32, 32, dtype=torch.complex64)
        out = apply_degradation("magic_angle", x, theta=1.0, seed=0)
        assert out.shape == x.shape
        assert torch.isfinite(out.abs()).all()


class TestPhaseOnlyRegistry:
    """#223 — the registry must declare which axes are pure-phase.

    ``concomitant_phase`` (D23) is the only pure-phase axis in the taxonomy: at
    theta=1 its magnitude rel-L2 is 4.6e-08 while its complex rel-L2 is 0.71. The
    sweep takes ``.abs()``, so it was handed to the metrics as a CONSTANT and
    every "sensitivity to concomitant phase" number was the sensitivity of a
    metric to nothing.
    """

    def test_concomitant_phase_is_the_only_phase_only_axis(self) -> None:
        assert phase_only_degradations() == ["concomitant_phase"]
        assert is_phase_only("concomitant_phase")

    def test_concomitant_phase_is_invisible_in_magnitude(self) -> None:
        x = _brain()
        mag_change = _rel_l2(_mag("concomitant_phase", x, 1.0), x.abs())
        cplx_change = _rel_l2(
            apply_degradation("concomitant_phase", x, theta=1.0, seed=0), x
        )
        assert mag_change < 1e-5, "the premise of the phase_only flag no longer holds"
        assert cplx_change > 0.1
        # The flag exists precisely so a caller does NOT magnitude-score this.
        assert is_phase_only("concomitant_phase")

    @pytest.mark.parametrize("name", _ALL_AXES)
    def test_declared_affects_matches_measurement(self, name: str) -> None:
        """Re-derive every ``affects`` flag from the operator itself.

        A declaration that nothing checks is exactly the facade this repo polices
        (pitfall #16). If a degrader is re-tuned so it stops moving phase (or
        starts), this fails rather than letting the flag go stale.
        """
        H = W = 64
        y, xx = torch.meshgrid(
            torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij"
        )
        # A real MRI slice is never phase-flat; give the input smooth phase so a
        # phase-preserving operator is distinguishable from a phase-destroying one.
        base = _brain(H, W).abs()
        x = (base * torch.exp(1j * (1.2 * xx + 0.8 * y - 1.5 * (xx**2 + y**2)))).to(
            torch.complex64
        )

        out = apply_degradation(name, x, theta=1.0, seed=0)
        declared = affected_components(name)

        mag_rel = _rel_l2(out.abs(), x.abs())
        assert (mag_rel > 1e-4) == (MAGNITUDE in declared), (
            f"{name}: declared affects={sorted(declared)} but magnitude rel-L2 "
            f"is {mag_rel:.2e}"
        )

        if not out.is_complex():
            # A real-valued output carries no phase at all — it cannot be routed
            # to a phase metric, so it must not claim to affect phase.
            assert declared == frozenset({MAGNITUDE}), (
                f"{name} returns a real image, so it can only be declared "
                f"magnitude-affecting; got {sorted(declared)}"
            )
            return

        w = x.abs() / x.abs().sum()
        dphi = torch.angle(out * torch.conj(x) / (x.abs() ** 2 + 1e-12))
        phase_rms = float(torch.sqrt((w * dphi**2).sum()))
        assert (phase_rms > 1e-4) == (PHASE in declared), (
            f"{name}: declared affects={sorted(declared)} but phase RMS is "
            f"{phase_rms:.2e} rad"
        )


class TestRealisedAcceleration:
    """#246 — the DECLARED acceleration must be the REALISED one.

    ``vd_undersamp`` kept each k-space sample with probability ``p / R``. With
    ``E[p] ~ 0.2`` for the centre-peaked density, the realised rate was ``R/E[p]``:
    a declared 1.0 -> 8.0x sweep in fact ran 6.0 -> 36.3x. The axis was therefore
    SATURATED at its first step (theta=0.05 already delivered 72% of the theta=1
    distortion; the usable range of the whole axis was 1.4x). ``cartesian_undersamp``
    and ``vd_cartesian`` had a milder version of the same disease — they budgeted
    the outer lines on top of the ACS instead of out of it, realising R = 5.3 at a
    declared R = 8.
    """

    R_AXES = ["cartesian_undersamp", "vd_undersamp", "vd_cartesian", "radial_undersamp"]

    @staticmethod
    def _kept_fraction(name: str, theta: float, seed: int = 0) -> float:
        # A dense complex-noise probe: no k-space cell is incidentally zero, so
        # the surviving support IS the sampling mask.
        g = torch.Generator().manual_seed(1234)
        probe = torch.complex(
            torch.randn(1, 1, 64, 64, generator=g),
            torch.randn(1, 1, 64, 64, generator=g),
        )
        k = fft2c(apply_degradation(name, probe, theta=theta, seed=seed))
        return float((k.abs() > 1e-6).float().mean())

    @pytest.mark.parametrize("name", R_AXES)
    @pytest.mark.parametrize("theta", [0.05, 0.5, 1.0])
    def test_realised_R_matches_declared_R(self, name: str, theta: float) -> None:
        declared = DEGRADATION_REGISTRY[name].severity.physical_value(theta)
        realised = 1.0 / max(self._kept_fraction(name, theta), 1e-9)
        assert realised == pytest.approx(declared, rel=0.12), (
            f"{name} at theta={theta}: declares R={declared:.2f} but realises "
            f"R={realised:.2f}. A severity axis whose physical parameter is not "
            "the one it advertises cannot be reported (#246)."
        )

    @pytest.mark.parametrize("theta", [0.05, 0.5, 1.0])
    def test_partial_fourier_realises_its_declared_retained_fraction(
        self, theta: float
    ) -> None:
        # D24's declared parameter is a retained ky fraction, not an R.
        declared = DEGRADATION_REGISTRY["partial_fourier"].severity.physical_value(theta)
        realised = self._kept_fraction("partial_fourier", theta)
        assert realised == pytest.approx(declared, rel=0.05)

    def test_vd_undersamp_is_not_saturated_at_the_first_step(self) -> None:
        """The regression that made D4 useless: 72% of the damage at theta=0.05."""
        x = _brain()
        first = _rel_l2(_mag("vd_undersamp", x, 0.05), x.abs())
        last = _rel_l2(_mag("vd_undersamp", x, 1.0), x.abs())
        assert first / last < 0.45, (
            f"vd_undersamp delivers {100 * first / last:.0f}% of its full distortion "
            "at the first severity step — the axis has no usable dynamic range."
        )


class TestAxisOrthogonality:
    """#245 — the axes are presented as orthogonal; the duplicates were not.

    ``t2star_blur`` (D12) and ``slice_profile`` (D27) were the SAME operator — an
    isotropic depthwise Gaussian blur — differing only in ``max_sigma_px``
    (4.0 vs 3.5). The cosine similarity of their theta=1 residuals was **0.998**.
    Two axes, one physics: double-counted in every per-family aggregation,
    inflating D in the Borda sum and the family count in the CDRS variance
    penalty. D27 is deleted.

    The residual non-orthogonal pairs are DECLARED in ``KNOWN_COLLINEAR_PAIRS``
    with their physical reason, so the double counting is visible to a reader of
    the results instead of buried in a test's skip list.
    """

    THRESHOLD = 0.9

    @staticmethod
    def _residuals(x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = {}
        for name in _ALL_AXES:
            r = (_mag(name, x, 1.0) - x.abs()).flatten()
            n = r.norm()
            out[name] = r / n if n > 1e-9 else r
        return out

    def test_no_undeclared_pair_is_collinear(self) -> None:
        x = _brain()
        resid = self._residuals(x)
        offenders = []
        for a, b in itertools.combinations(sorted(resid), 2):
            if is_known_collinear(a, b):
                continue
            cos = abs(float(torch.dot(resid[a], resid[b])))
            if cos > self.THRESHOLD:
                offenders.append((a, b, cos))
        assert not offenders, (
            "these axis pairs are near-duplicates but are not declared in "
            f"KNOWN_COLLINEAR_PAIRS: {offenders}. Either make them physically "
            "distinct, or delete one and declare the merge (#245)."
        )

    def test_the_duplicate_blur_axis_is_gone(self) -> None:
        assert "slice_profile" not in DEGRADATION_REGISTRY
        with pytest.raises(KeyError, match="Unknown degradation"):
            apply_degradation("slice_profile", _brain(), theta=1.0, seed=0)

    def test_every_declared_collinear_pair_is_registered_and_reasoned(self) -> None:
        for (a, b), reason in KNOWN_COLLINEAR_PAIRS.items():
            assert a in DEGRADATION_REGISTRY and b in DEGRADATION_REGISTRY
            assert (a, b) == tuple(sorted((a, b))), "keys must be sorted for lookup"
            assert len(reason) > 40, f"({a}, {b}) needs a real physical reason"


def test_registry_count_matches_d1_d31_taxonomy() -> None:
    """``DEGRADATION_REGISTRY`` is the count SSOT of the D-coded taxonomy. Pinned exactly here (co-located drift guard) so a
    docstring/count divergence — the stale ``D1-D20`` this replaced — fails at the
    module's own test, not only in the sim2rank canonical-counts guard."""
    assert len(DEGRADATION_REGISTRY) == 70


def test_resolution_snr_is_distinct_from_pure_blur_and_pure_noise() -> None:
    """resolution_snr (D28) couples BOTH effects on one theta: it adds noise
    (nonzero phase on a real input, unlike the pure blur t2star_blur) AND apodizes
    (less high-frequency energy than the pure-noise complex_gaussian at the same
    theta). This is the anti-facade check that the coupling actually fires."""
    import torch

    from spectramr.infrastructure.physics.degradations import (
        degrade_complex_gaussian_noise,
        degrade_resolution_snr,
        degrade_t2star_blur,
    )
    from spectramr.infrastructure.physics.fft_ops import fft2c

    torch.manual_seed(0)
    n = 64
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, n), torch.linspace(-1, 1, n), indexing="ij")
    x = ((xx**2 + yy**2) < 0.5).float()[None, None]

    def hf(t: torch.Tensor) -> float:
        k = fft2c(t.to(torch.complex64)).abs()
        return (k[..., : n // 4, :].mean() + k[..., 3 * n // 4 :, :].mean()).item()

    yr = degrade_resolution_snr(x, 0.6, seed=3)
    yb = degrade_t2star_blur(x.to(torch.complex64), 0.6)
    yn = degrade_complex_gaussian_noise(x, 0.6, seed=3)

    # Noise signature: an imaginary part (a pure blur leaves a real input real).
    def imag_ratio(t: torch.Tensor) -> float:
        return (t.imag.norm() / t.real.norm()).item()

    assert imag_ratio(yr) > 1e-2 > 1e-5 > imag_ratio(yb)
    # Apodization signature: less high-freq energy than pure noise (which only adds it).
    assert hf(yr.abs()) < hf(yn.abs())

def test_motion_types_are_distinct_schedules() -> None:
    """D29/D30/D31 are three DISTINCT motion TYPES, not aliases: at the same theta and
    seed they give materially different output (linear drift vs single mid-scan step vs
    i.i.d. jitter), each differs from rigid_motion (D6), and theta=0 is an exact
    identity for the deterministic-schedule axes (anti-facade: the new axes fire)."""
    import torch

    from spectramr.infrastructure.physics.degradations import apply_degradation

    torch.manual_seed(0)
    n = 48
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, n), torch.linspace(-1, 1, n), indexing="ij"
    )
    x = ((xx**2 + yy**2) < 0.5).float()[None, None]
    names = ("translation_drift", "sudden_motion", "random_motion", "rigid_motion")
    outs = {a: apply_degradation(a, x, 0.8, seed=1).abs() for a in names}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            assert not torch.allclose(outs[a], outs[b], atol=1e-3), f"{a} == {b}"
    for a in ("translation_drift", "sudden_motion", "random_motion"):
        assert torch.allclose(apply_degradation(a, x, 0.0, seed=1).abs(), x, atol=1e-5)

def test_advisor_major_axis_repairs() -> None:
    """Regression for the four major physics-compliance repairs (advisor review):
    D4b incoherent VD (not a low-pass block), D7 pulsatile enters k-space (PE ghost),
    D21 gradient nonlinearity moderate (not catastrophic), D25 iq DC-offset is a
    localized central-FOV spike (not a flat image-wide pedestal)."""
    import torch

    from spectramr.infrastructure.physics.degradations import apply_degradation

    torch.manual_seed(0)
    n = 64
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, n), torch.linspace(-1, 1, n), indexing="ij"
    )
    x = ((xx**2 + yy**2) < 0.5).float()[None, None]

    def mag(z):
        return z.abs() if torch.is_complex(z) else z

    # D4b: incoherent VD is no longer collinear with the low-pass gibbs axis.
    rv = (mag(apply_degradation("vd_cartesian", x, 0.9, seed=1)) - x).flatten()
    rg = (mag(apply_degradation("gibbs", x, 0.9, seed=1)) - x).flatten()
    cos = abs(float((rv @ rg) / (rv.norm() * rg.norm())))
    assert cos < 0.9, f"vd_cartesian still collinear with gibbs (|cos|={cos:.2f})"

    # D7: pulsatile now enters k-space -> nonzero phase on a real input.
    assert apply_degradation("pulsatile_motion", x, 0.7, seed=1).angle().abs().mean() > 1e-2

    # D21: gradient nonlinearity is moderate at theta=1 (was ~10-15x too strong).
    g1 = float(
        (mag(apply_degradation("gradient_nonlinearity", x, 1.0, seed=1)) - x).norm()
        / x.norm()
    )
    assert g1 < 0.2, f"gradient_nonlinearity still catastrophic (rel-L2={g1:.2f})"

    # D25: iq DC-offset is a LOCALIZED central spike, not a uniform pedestal.
    xf = torch.full((1, 1, n, n), 0.5)
    d = mag(apply_degradation("iq_ghost", xf, 0.9, seed=1)) - 0.5
    cy = cx = n // 2
    assert float(d[0, 0, cy, cx]) > float(d.median()) + 0.05, "no localized central spike"
    assert float(d[0, 0, cy, cx]) > float(d[0, 0, 2, 2]) + 0.05

def test_advisor_units_honesty_reconciliations() -> None:
    """Declared severity units/values now match what the operators realize (advisor):
    eddy declares BOTH its linear and quadratic phase scales (the prior 'rad per PE
    line' label was ~40x off), susceptibility's max_dropout matches its ~0.98 realized
    peak, and geometric_warp is labelled a per-mode scale rather than px."""
    import torch

    from spectramr.infrastructure.physics.degradations import (
        DEGRADATION_REGISTRY,
        apply_degradation,
    )

    eddy = DEGRADATION_REGISTRY["eddy"].severity
    assert [p.name for p in eddy.params] == ["linear_phase_scale", "quadratic_phase_scale"]
    assert "rad per PE line" not in {p.units for p in eddy.params}
    # geometric_warp is normalised so its declared px IS the realised peak displacement
    # (pinned in test_degradation_physics.py).
    gw = DEGRADATION_REGISTRY["geometric_warp"].severity.primary
    assert gw.units == "px" and gw.name == "peak_displacement"

    # susceptibility declared peak dropout matches the realized peak (was 0.5 vs ~0.98).
    declared = DEGRADATION_REGISTRY["susceptibility"].severity.params[1].at_theta_max
    assert declared == pytest.approx(1.0 - math.exp(-4.0), abs=1e-4)
    n = 64
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, n), torch.linspace(-1, 1, n), indexing="ij"
    )
    x = ((((xx * 0.7) ** 2 + (yy * 0.9) ** 2) < 1.0).float() * 0.8 + 0.2)[None, None]
    y = apply_degradation("susceptibility", x, 1.0, seed=1).abs()
    sig = x[0, 0] > 0.3
    realized = float(1.0 - (y[0, 0][sig] / x[0, 0][sig]).min())
    assert abs(realized - declared) < 0.1, f"declared {declared} vs realized {realized:.2f}"


# ──────────────────────────────────────────────────────────────────────
# D2 rician — sigma is a FRACTION of the signal, not an absolute intensity.
#
# sim2rank.py sweeps the RAW coil-combined magnitude (p99 ~ 1e-4 on fastMRI
# brain) and divides by p99 only AFTER the sweep. Every other noise axis
# derives sigma from the signal, so degrade-then-rescale == rescale-then-
# degrade for them. D2 used ``sigma = theta * 0.3`` literally, which put the
# noise ~4 orders of magnitude above the signal: psnr pinned to its -30 dB
# floor, ssim to 0.0 and nema_snr to a constant at EVERY severity level of the
# 2026-07-27 brain run, and the snapshot triptychs washed out to noise.
# ──────────────────────────────────────────────────────────────────────


class TestRicianScaleInvariance:
    @staticmethod
    def _nmse(clean: torch.Tensor, noisy: torch.Tensor) -> float:
        return float(((clean - noisy) ** 2).sum() / (clean**2).sum())

    @pytest.mark.parametrize("theta", [0.05, 0.5, 1.0])
    def test_severity_is_invariant_to_input_scale(self, theta: float) -> None:
        """The same theta must mean the same damage at any input scale."""
        from spectramr.infrastructure.physics.degradations import degrade_rician_noise

        base = _make_phantom(64, 64, complex_=False)
        base = base * torch.linspace(0.6, 1.0, 64).view(1, 1, 1, 64)  # texture
        ref = self._nmse(base, degrade_rician_noise(base, theta, seed=0))
        for scale in (1e-2, 1e-4, 1e2):
            scaled = base * scale
            got = self._nmse(scaled, degrade_rician_noise(scaled, theta, seed=0))
            assert got == pytest.approx(ref, rel=1e-6), (
                f"theta={theta} at input scale {scale:g} realised NMSE {got:.4g} "
                f"vs {ref:.4g} at unit scale — sigma is not signal-relative"
            )

    def test_severity_is_comparable_to_the_sibling_noise_axis(self) -> None:
        """D2 at theta=1 must land in the same regime as D1 at theta=1.

        Not an identity — they are different noise models — but a decade apart
        means the two noise axes are on incompatible severity yardsticks and no
        cross-axis aggregate over them is meaningful.
        """
        from spectramr.infrastructure.physics.degradations import degrade_rician_noise

        x = _make_phantom(64, 64, complex_=False)
        rician = self._nmse(x, degrade_rician_noise(x, 1.0, seed=0))
        cgauss = self._nmse(
            x, degrade_complex_gaussian_noise(x.to(torch.complex64), 1.0, seed=0).abs()
        )
        assert (
            0.1 < rician / cgauss < 10.0
        ), f"rician NMSE {rician:.4g} vs complex_gaussian {cgauss:.4g}"

    def test_declared_unit_matches_the_implementation(self) -> None:
        """The registry severity must not advertise a unit the operator does not use.

        The spec said "fraction of max signal" while the code applied an
        absolute intensity — an unwired declaration (pitfall #15).
        """
        from spectramr.infrastructure.physics.degradations import DEGRADATION_REGISTRY

        (param,) = DEGRADATION_REGISTRY["rician"].severity.params
        assert param.name == "sigma"
        assert "fraction" in param.units

        # And the operator honours it: at theta=1 the background (zero signal)
        # settles at the Rayleigh mean sigma*sqrt(pi/2) with sigma = 0.3 * p99.
        from spectramr.infrastructure.physics.degradations import degrade_rician_noise

        x = _make_phantom(64, 64, complex_=False)
        p99 = float(torch.quantile(x.reshape(-1), 0.99))
        out = degrade_rician_noise(x, 1.0, seed=0)
        background = out[x == 0]
        expected = 0.3 * p99 * (torch.pi / 2) ** 0.5
        assert float(background.mean()) == pytest.approx(expected, rel=0.1)

    def test_all_zero_input_does_not_divide_by_zero(self) -> None:
        from spectramr.infrastructure.physics.degradations import degrade_rician_noise

        out = degrade_rician_noise(torch.zeros(1, 1, 16, 16), 1.0, seed=0)
        assert torch.isfinite(out).all()


class TestDeriveAxisSeed:
    """``derive_axis_seed`` is the per-axis seed the digital twin's registry pass
    uses. Its contract: distinct per axis, stable across processes."""

    def test_distinct_axes_get_distinct_seeds(self) -> None:
        from spectramr.infrastructure.physics.degradations import (
            DEGRADATION_REGISTRY,
            derive_axis_seed,
        )

        seeds = {a: derive_axis_seed(42, a) for a in DEGRADATION_REGISTRY}
        assert len(set(seeds.values())) == len(seeds), (
            "two axes collided on one seed, so they would share an artefact "
            f"realisation: {seeds}"
        )

    def test_seed_is_stable_across_processes(self) -> None:
        """The reason it is ``zlib.crc32`` and not the builtin ``hash``: the
        latter is salted by PYTHONHASHSEED, which would put cross-process
        non-determinism back in through the side door."""
        import os
        import subprocess
        import sys

        snippet = (
            "from spectramr.infrastructure.physics.degradations import "
            "derive_axis_seed; print(derive_axis_seed(42, 'rician'))"
        )
        runs = {
            subprocess.run(
                [sys.executable, "-c", snippet],
                capture_output=True,
                text=True,
                check=True,
                env={"PYTHONHASHSEED": str(h), "PYTHONPATH": os.environ.get("PYTHONPATH", "")},
            ).stdout.strip()
            for h in (0, 1)
        }
        assert len(runs) == 1, f"seed varied with PYTHONHASHSEED: {runs}"

    def test_seed_does_not_depend_on_severity(self) -> None:
        """theta must stay the only free variable across a severity grid."""
        from spectramr.infrastructure.physics.degradations import derive_axis_seed

        assert derive_axis_seed(7, "spike") == derive_axis_seed(7, "spike")
        assert derive_axis_seed(7, "spike") != derive_axis_seed(8, "spike")
