"""Invariants every degradation axis must satisfy, checked on every registered axis at once.

These are *cross-axis* contracts, deliberately separate from the per-axis physics
assertions in ``tests/unit/infrastructure/physics/test_digital_twin_extensions.py``.
The distinction matters: a per-axis test asks "does gibbs ring?", these ask "is every
axis a well-formed severity coordinate?" — and the second question is what sim2rank
actually depends on, because it grades metrics by their *response* to theta.

The invariants, and the failure each one exists to catch:

1. **Finite everywhere.** A NaN anywhere in a degraded image poisons every metric
   computed from it, and a metric that returns NaN is recorded as ``never_computed``
   rather than as a crash — so the axis silently drops out of the ranking.

2. **Scale-invariant severity.** ``theta`` must mean the same damage regardless of the
   caller's units. D2 ``rician`` violated this until 2026-07-27 (``sigma = theta * 0.3``
   as a literal intensity): ``sim2rank.py`` sweeps the RAW coil-combined magnitude and
   divides by ``raw_p99`` only afterwards, so on fastMRI brain (p99 ~ 1e-4) the noise sat
   four orders of magnitude above the signal and the axis was destroyed at theta=0.05.
   Nothing crashed; ``psnr`` simply sat on its -30 dB floor at every severity level.
   **This is the test that would have caught it, and it catches the whole class.**

3. **Monotone in theta.** If severity does not increase with theta, ADR/SROCC/isotonic
   are all measuring noise, and a metric can score well by tracking the wrong direction.

4. **Materially degrading.** An axis that barely moves at theta=1 contributes a
   near-random column to every cross-axis mean (#223 — the magic-angle no-op).

Severity is measured as scale-free relative L2, ``||x - D(x)||^2 / ||x||^2``, which is
dimensionless by construction — so invariant 2 is a statement about the *operator*, not
about the units the assertion happens to use.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectramr.infrastructure.physics.degradations import (  # noqa: E402
    DEGRADATION_REGISTRY,
    AcquisitionGeometry,
    AcquisitionMode,
    apply_degradation,
    is_phase_only,
)
from spectramr.infrastructure.physics.degradations.declarations import (  # noqa: E402
    DECLARED_PHYSICAL_FOV_MM,
)
from spectramr.infrastructure.physics.degradations.registry_queries import (  # noqa: E402
    requirement_free_degradations,
    slice_capable_degradations,
)

# This harness feeds a single-coil 2-D phantom: volume-only and multi-coil axes are covered by
# their own mode-rejection and coil-stack tests under degradations/catalogue/.
ALL_AXES = sorted(
    set(slice_capable_degradations()) & set(requirement_free_degradations())
)

#: Severity grid sim2rank actually sweeps: ``theta in [1/T, 1]``.
THETA_GRID = np.linspace(0.125, 1.0, 8)

#: Input scales spanning the range a caller may plausibly hand in. 1e-4 is the real
#: p99 of a raw coil-combined fastMRI-brain magnitude; 1e2 is a un-normalised
#: integer-ish scanner scale. An operator that is a function of *severity* rather than
#: of *intensity* cannot tell them apart.
INPUT_SCALES = (1e-4, 1.0, 1e2)


def _phantom(h: int = 48, w: int = 48) -> torch.Tensor:
    """Asymmetric, textured, zero-background complex phantom, p99 ~ 1.

    Asymmetric so a transpose/flip bug cannot hide; textured so blur axes have
    something to remove; zero background so an additive axis is visible in air (the
    #223 magic-angle no-op was only detectable because its patches landed in air).
    """
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij"
    )
    disc = ((x - 0.1) ** 2 + (y * 1.3) ** 2 < 0.55).float()
    texture = (0.6 + 0.4 * torch.cos(6 * x) * torch.sin(5 * y)).abs()
    p = (disc * texture).unsqueeze(0).unsqueeze(0)
    return torch.complex(p, torch.zeros_like(p))


def _severity(
    clean: torch.Tensor, degraded: torch.Tensor, *, phase_only: bool
) -> float:
    """Scale-free relative L2 between clean and degraded.

    Phase-only axes (``concomitant_phase``) leave the magnitude untouched by
    construction, so measuring them on ``|x|`` would report a no-op and fail
    invariant 4 for the wrong reason. Compare the complex field for those.
    """
    a = clean if phase_only else clean.abs()
    b = degraded if phase_only else degraded.abs()
    denom = (a.abs() ** 2).sum().clamp(min=1e-30)
    return float(((a - b).abs() ** 2).sum() / denom)


def _geometry(axis: str, x: torch.Tensor) -> AcquisitionGeometry | None:
    """A declared FOV-dependent axis is measured on a phantom of its physical size."""
    fov = DECLARED_PHYSICAL_FOV_MM.get(axis)
    if fov is None:
        return None
    voxel = (5.0, fov / x.shape[-2], fov / x.shape[-1])
    return AcquisitionGeometry(mode=AcquisitionMode.SLICE_2D, voxel_mm=voxel)


def _severity_curve(axis: str, *, seed: int, scale: float = 1.0) -> list[float]:
    x = _phantom() * scale
    po, geometry = is_phase_only(axis), _geometry(axis, x)
    return [
        _severity(
            x,
            apply_degradation(axis, x.clone(), float(t), seed=seed, geometry=geometry),
            phase_only=po,
        )
        for t in THETA_GRID
    ]


# ──────────────────────────────────────────────────────────────────────
# 1. Finite everywhere
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("axis", ALL_AXES)
def test_output_is_finite_at_every_severity_and_seed(axis: str) -> None:
    """No NaN and no inf anywhere on the swept grid.

    A single non-finite voxel is enough: the metric computed from it returns NaN,
    ``classify_metric_health`` files that as ``never_computed``, and the axis leaves
    the ranking without anything having failed.
    """
    x = _phantom()
    for seed in (0, 7):
        for theta in THETA_GRID:
            out = apply_degradation(axis, x.clone(), float(theta), seed=seed)
            bad = (~torch.isfinite(out)).sum().item()
            assert bad == 0, (
                f"{axis} produced {bad} non-finite value(s) at theta={theta:.3f}, "
                f"seed={seed}"
            )


@pytest.mark.parametrize("axis", ALL_AXES)
def test_shape_and_device_are_preserved(axis: str) -> None:
    """A degradation is an endomorphism on the grid: same shape, same device."""
    x = _phantom()
    out = apply_degradation(axis, x.clone(), 0.5, seed=0)
    assert out.shape == x.shape, f"{axis}: {tuple(out.shape)} != {tuple(x.shape)}"
    assert out.device == x.device


#: The only operator defined ON the magnitude, so the only one allowed to return a
#: real tensor. Everything else must hand back a complex field.
_MAGNITUDE_VALUED_AXES = frozenset({"rician"})


@pytest.mark.parametrize("axis", ALL_AXES)
def test_only_a_declared_magnitude_operator_may_discard_phase(axis: str) -> None:
    """Returning a real tensor from a complex input silently destroys the phase.

    ``rician`` (D2) is defined as image-space Rician noise **on** ``|x|`` and is
    declared ``affected_components == {magnitude}``, so it is magnitude-valued by
    construction. Any other axis that returns real has dropped the phase, and the
    damage is invisible downstream: ``fft2c`` accepts a real even-channel stack and
    silently reinterprets it, so the k-space round-trip keeps working while the
    imaginary part is gone.

    Pinned as a set equality, not a per-axis exemption, so *adding* a phase-dropping
    axis fails here rather than being discovered in a reconstruction three layers up.
    """
    from spectramr.infrastructure.physics.degradations import MAGNITUDE, affected_components

    out = apply_degradation(axis, _phantom().clone(), 0.5, seed=0)
    if axis in _MAGNITUDE_VALUED_AXES:
        assert affected_components(axis) == frozenset({MAGNITUDE}), (
            f"{axis} is exempted as magnitude-valued but does not declare "
            f"affected_components == {{{MAGNITUDE}}}"
        )
        return
    assert out.is_complex(), (
        f"{axis} returned a real tensor for a complex input — the phase is gone. "
        "If that is intentional, declare it magnitude-only AND add it to "
        "_MAGNITUDE_VALUED_AXES with the reason."
    )


# ──────────────────────────────────────────────────────────────────────
# 2. Scale-invariant severity — the rician class of bug
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("axis", ALL_AXES)
def test_severity_is_invariant_to_input_scale(axis: str) -> None:
    """The same theta must mean the same damage at any input intensity.

    Severity here is scale-free by construction, so any drift across
    :data:`INPUT_SCALES` means the *operator* is reading an absolute intensity
    somewhere. Only a signal-relative parameterisation (D1: k-space RMS -> declared
    SNR) commutes with the p99 normalisation ``sim2rank.py`` applies AFTER the sweep.

    Regression for the 2026-07-27 D2 defect, stated generally so a new axis that
    hardcodes a sigma, a threshold, or an epsilon in absolute units fails here.
    """
    ref = _severity_curve(axis, seed=7, scale=1.0)
    for scale in INPUT_SCALES:
        if scale == 1.0:
            continue
        got = _severity_curve(axis, seed=7, scale=scale)
        assert np.allclose(got, ref, rtol=1e-3, atol=1e-12), (
            f"{axis}: realised severity changes with the input scale.\n"
            f"  at scale 1.0  : {np.round(ref, 6).tolist()}\n"
            f"  at scale {scale:g}: {np.round(got, 6).tolist()}\n"
            "The operator is reading an absolute intensity. sim2rank sweeps the RAW "
            "magnitude and normalises by p99 afterwards, so this axis will realise a "
            "different severity in the metric path than in the snapshot path."
        )


# ──────────────────────────────────────────────────────────────────────
# 3-4. Monotone, and actually degrading
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("axis", ALL_AXES)
def test_severity_increases_with_theta(axis: str) -> None:
    """Severity must track theta, or every downstream statistic is measuring noise.

    Spearman rather than strict step-wise monotonicity: a saturating axis is allowed
    to plateau, it is not allowed to wander.
    """
    from scipy.stats import spearmanr

    curve = _severity_curve(axis, seed=7)
    rho = float(spearmanr(THETA_GRID, curve).statistic)
    assert rho > 0.9, (
        f"{axis}: severity is not monotone in theta (Spearman rho={rho:.3f}); "
        f"curve={np.round(curve, 6).tolist()}"
    )


@pytest.mark.parametrize("axis", ALL_AXES)
def test_axis_materially_degrades_at_full_severity(axis: str) -> None:
    """theta=1 must do something. A near-no-op axis contributes a random column.

    The floor is deliberately loose (1e-4 relative L2): this catches an axis that has
    become inert (#223), not one that is merely gentle.
    """
    end = _severity_curve(axis, seed=7)[-1]
    assert end > 1e-4, (
        f"{axis}: relative L2 at theta=1 is only {end:.3g} — effectively a no-op, so "
        "every metric scores it at chance and it dilutes every cross-axis mean"
    )


# ──────────────────────────────────────────────────────────────────────
# Cross-axis: the severity yardstick
# ──────────────────────────────────────────────────────────────────────


def test_severity_yardstick_spread_is_pinned() -> None:
    """The axes are NOT severity-equalised, and this pins how unequal they are.

    ``compute_minimax_borda`` already warns that its worst-axis term "is only
    meaningful if the axes are severity-equalised. Otherwise every metric's worst axis
    is simply whichever axis was swept hardest" (#245). This test makes the size of
    that caveat a checked number instead of a comment: at theta=1 the realised severity
    currently spans ~2.0e3x, from ``gradient_nonlinearity`` (~1.0e-3) to ``spike``
    (~2.1). A change that widens it further is a real regression in the minimax term
    and must be an explicit decision, not a side effect.
    """
    ends = {a: _severity_curve(a, seed=7)[-1] for a in ALL_AXES}
    lo_axis, lo = min(ends.items(), key=lambda kv: kv[1])
    hi_axis, hi = max(ends.items(), key=lambda kv: kv[1])
    spread = hi / lo
    assert spread < 5e3, (
        f"severity spread at theta=1 widened to {spread:.0f}x "
        f"({lo_axis}={lo:.3g} .. {hi_axis}={hi:.3g}); the minimax worst-axis term now "
        "ranks the simulator's hand-tuned constants rather than the metrics (#245)"
    )


# ──────────────────────────────────────────────────────────────────────
# Cross-axis: the sweep must deliver the T levels it declares
# ──────────────────────────────────────────────────────────────────────


def _sweep(axis: str, *, grid: list[float] | None = None):
    """One axis swept by the real ``MetaDegradationSweep``, optionally on a
    hand-compressed severity grid (what a yardstick does to a strong axis)."""
    pytest.importorskip("scripts.sim2rank.degradation")  # not in the public export
    from scripts.sim2rank.degradation import MetaDegradationSweep

    sweep = MetaDegradationSweep(n_timesteps=20, device=torch.device("cpu"), seed=7)
    if grid is not None:
        sweep.severity_grid = grid
    return sweep, sweep.sweep_axis(_phantom(), axis)


@pytest.mark.parametrize("axis", sorted(set(ALL_AXES) - {"concomitant_phase"}))
def test_declared_grid_resolves_most_of_its_timesteps(axis: str) -> None:
    """On the DECLARED grid, an axis must render distinguishable levels.

    A timestep that is bit-identical to an earlier one is an exact tie in every metric
    trajectory at once. The rankers read the T levels as independent, so duplicates go
    straight into ADR's increment-perplexity as zero increments. Measured on real data
    at 256x256 all 30 default axes deliver 20/20; this phantom is 48x48, where the
    integer PE-line count ``round(H / R)`` legitimately repeats at high R, so the bar is
    the same 50% the ranking guard uses rather than a strict 20/20.
    """
    pytest.importorskip("scripts.sim2rank.degradation")  # not in the public export
    from scripts.sim2rank.degradation import _distinct_level_count

    _, mags = _sweep(axis)
    distinct = _distinct_level_count(mags)
    assert distinct >= 0.5 * len(mags), (
        f"{axis}: only {distinct} of {len(mags)} severity steps render a distinct "
        "image, so most timesteps carry no new information"
    )


def test_collapsed_grid_raises_at_the_sweep_not_in_a_ranker() -> None:
    """A compressed severity range must fail HERE, naming the axis (#236 regression).

    Reproduces the 2026-07-28 crash: ``cartesian_undersamp`` swept over
    theta <= 5e-4 realises R in [1.0002, 1.0035], and ``round(H / R)`` returns the full
    PE-line count for that whole band — 2 distinct images across 20 steps. Before this
    guard the sweep returned happily and the failure surfaced ~25 minutes later as
    ``DegenerateRankingError: Ranker 'Gen1_ADR[cartesian_undersamp]' ... only 4 distinct
    scores``, which names the ranker rather than the sweep that starved it.
    """
    pytest.importorskip("scripts.sim2rank.degradation")  # not in the public export
    from scripts.sim2rank.degradation import DegenerateSweepError

    grid = np.linspace(5e-4 / 20, 5e-4, 20).tolist()
    with pytest.raises(DegenerateSweepError) as excinfo:
        _sweep("cartesian_undersamp", grid=grid)
    msg = str(excinfo.value)
    assert "cartesian_undersamp" in msg
    assert "distinct degradation levels" in msg


def test_healthy_grid_does_not_raise() -> None:
    """The default path must stay silent — the guard may not cost anyone a run."""
    _, mags = _sweep("cartesian_undersamp")
    assert len(mags) == 20


def test_sweep_all_axes_reports_every_degenerate_axis_at_once() -> None:
    """One error listing all offenders, not one failed run per bad axis.

    A 30-axis sweep is ~10 minutes; raising on the first collapsed axis would make
    enumerating a yardstick's damage a five-run job.
    """
    pytest.importorskip("scripts.sim2rank.degradation")  # not in the public export
    from scripts.sim2rank.degradation import DegenerateSweepError, MetaDegradationSweep

    sweep = MetaDegradationSweep(
        n_timesteps=20,
        device=torch.device("cpu"),
        seed=7,
        axes=["cartesian_undersamp", "vd_cartesian", "complex_gaussian"],
    )
    sweep.severity_grid = np.linspace(5e-4 / 20, 5e-4, 20).tolist()
    with pytest.raises(DegenerateSweepError) as excinfo:
        sweep.sweep_all_axes(_phantom())
    msg = str(excinfo.value)
    assert "cartesian_undersamp" in msg and "vd_cartesian" in msg


@pytest.mark.parametrize("axis", ALL_AXES)
def test_declared_severity_params_resolve_and_are_finite(axis: str) -> None:
    """Every axis declares its physical range, and the declaration is usable.

    ``simulator_calibration`` and the severity-sweep figure both drive off
    ``DegradationSpec.severity``; an axis whose ``value_at`` returns NaN, or whose
    endpoints coincide, renders a flat curve that says nothing about the physics.
    """
    from spectramr.infrastructure.physics.degradations import DEGRADATION_REGISTRY

    spec = DEGRADATION_REGISTRY[axis].severity
    assert spec.params, f"{axis} declares no physical severity parameter"
    for param in spec.params:
        lo, hi = param.value_at(0.0), param.value_at(1.0)
        assert np.isfinite(lo) and np.isfinite(
            hi
        ), f"{axis}.{param.name}: non-finite endpoint ({lo}, {hi})"
        assert lo != hi, (
            f"{axis}.{param.name}: theta=0 and theta=1 both map to {lo} — the declared "
            "range is empty, so the axis advertises a knob it does not turn"
        )
        assert param.units, f"{axis}.{param.name} declares no units"


# ──────────────────────────────────────────────────────────────────────
# The sweep's clean end must BE the reference the metrics grade against
# ──────────────────────────────────────────────────────────────────────


def _coil_stack(h: int = 48, w: int = 48, n_coils: int = 4):
    """``(coil_images, smaps, x_true)`` with ``coil_images = smaps * x``.

    ``sum_c |S_c|^2`` is deliberately NOT 1, so RSS and SENSE disagree — which is
    the whole point. SENSE recovers ``x`` exactly; RSS returns
    ``|x| * sqrt(sum|S|^2)``.
    """
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij"
    )
    x = torch.complex(
        (((xx - 0.1) ** 2 + (yy * 1.3) ** 2) < 0.55).float(), torch.zeros(h, w)
    )
    maps = []
    for c in range(n_coils):
        cx, cy = (-0.6, 0.6)[c % 2], (-0.6, 0.6)[(c // 2) % 2]
        g = 0.4 + 0.9 * torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2))
        maps.append(torch.complex(g, 0.15 * g))
    smaps = torch.stack(maps).unsqueeze(0)
    coil_images = smaps * x.unsqueeze(0).unsqueeze(0)
    return coil_images, smaps, x.unsqueeze(0).unsqueeze(0)


def _sense_mag(coil_images: torch.Tensor, smaps: torch.Tensor) -> torch.Tensor:
    from spectramr.infrastructure.physics.coil_sensitivity import coil_combine_sense

    return coil_combine_sense(coil_images, smaps).abs().float()


def _reference_p99(coil_images: torch.Tensor, smaps: torch.Tensor) -> float:
    """The ``raw_p99`` scalar ``synthesize_pseudo_gt`` returns alongside the ref."""
    mag = _sense_mag(coil_images, smaps)
    return float(torch.quantile(mag.flatten(), 0.99).clamp(min=1e-8))


def _reference_mag(coil_images: torch.Tensor, smaps: torch.Tensor) -> torch.Tensor:
    """Exactly what ``synthesize_pseudo_gt`` builds: |SENSE| / its own p99."""
    return _sense_mag(coil_images, smaps) / _reference_p99(coil_images, smaps)


def test_sweep_clean_end_recovers_the_metric_reference() -> None:
    """The regression: a sweep that reconstructs differently from the reference puts
    a constant error under EVERY severity of EVERY axis.

    On the 2026-07-27 brain run the sweep combined coils by RSS while the reference
    was SENSE, so 29 of 30 axes started at MSE 0.610-0.622 — including ``magic_angle``,
    a no-op — and PSNR never exceeded 2.16 dB anywhere in the sweep. The theta=0
    anchor reported every axis clean because it compared the sweep against its OWN
    recon rather than against the reference.
    """
    pytest.importorskip("scripts.sim2rank.degradation")  # not in the public export
    from scripts.sim2rank.degradation import RegistryLeaderboardSweep, identity_floor

    coil_images, smaps, _ = _coil_stack()
    ref = _reference_mag(coil_images, smaps)
    scale = _reference_p99(coil_images, smaps)

    sweep = RegistryLeaderboardSweep(n_timesteps=8, device=torch.device("cpu"), seed=7)
    recovery = sweep.identity_recovery(
        coil_images, smaps, reference_mag=ref, scale=scale
    )
    bad = {a: v for a, v in recovery.items() if v > max(1e-3, identity_floor(a) * 1.5)}
    assert not bad, (
        f"{len(bad)} axes do not recover the metric reference at theta=0: "
        f"{dict(sorted(bad.items(), key=lambda kv: -kv[1])[:5])}. A nonzero value "
        "here is a pipeline offset that sits under every severity level."
    )


def test_an_rss_reference_is_detected_not_absorbed() -> None:
    """Anchoring on a mismatched (RSS) reference must FAIL loudly.

    This is the negative control for the test above: if the anchor cannot tell a
    SENSE sweep from an RSS reference, it would have passed the very run it was
    supposed to catch.
    """
    pytest.importorskip("scripts.sim2rank.degradation")  # not in the public export
    from scripts.sim2rank.degradation import RegistryLeaderboardSweep, identity_floor

    coil_images, smaps, _ = _coil_stack()
    rss = torch.sqrt((coil_images.abs() ** 2).sum(dim=1, keepdim=True)).float()
    rss = rss / torch.quantile(rss.flatten(), 0.99).clamp(min=1e-8)

    sweep = RegistryLeaderboardSweep(n_timesteps=8, device=torch.device("cpu"), seed=7)
    recovery = sweep.identity_recovery(coil_images, smaps, reference_mag=rss, scale=1.0)
    bad = [a for a, v in recovery.items() if v > max(1e-3, identity_floor(a) * 1.5)]
    assert len(bad) == len(recovery), (
        "an RSS reference against a SENSE sweep must fail on every axis; only "
        f"{len(bad)}/{len(recovery)} were flagged"
    )


def test_magnitude_valued_axis_is_applied_after_coil_combination() -> None:
    """``rician`` destroys per-coil phase, so it cannot be SENSE-combined afterwards.

    Applying it per-coil and then SENSE-combining a magnitude stack is a different
    operator from the reference's SENSE-combine-then-magnitude, and it left rician
    0.17 rel-L2 from the reference at theta=0 where it is algebraically the identity.
    RSS hid this because RSS reads only magnitudes.
    """
    pytest.importorskip("scripts.sim2rank.degradation")  # not in the public export
    from scripts.sim2rank.degradation import RegistryLeaderboardSweep

    coil_images, smaps, _ = _coil_stack()
    ref = _reference_mag(coil_images, smaps)
    scale = _reference_p99(coil_images, smaps)
    sweep = RegistryLeaderboardSweep(
        n_timesteps=8, device=torch.device("cpu"), seed=7, axes=["rician"]
    )
    assert sweep._is_magnitude_valued("rician")
    rec = sweep.identity_recovery(coil_images, smaps, reference_mag=ref, scale=scale)
    assert rec["rician"] < 1e-3, (
        f"rician is the identity at theta=0; rel-L2 {rec['rician']:.3e} means it was "
        "combined differently from the reference"
    )
