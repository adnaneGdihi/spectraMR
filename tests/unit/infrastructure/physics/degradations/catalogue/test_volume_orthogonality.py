"""The volume-only axes are not near-duplicates of any other axis of their mode (#245, in 5-D).

The 2-D orthogonality suite cannot see an axis without a slice_2d kernel, so this measures the
same quantity (|cos| of the unit-normalised theta=1 magnitude residuals) on deterministic
MULTISLICE_2D and VOLUME_3D phantoms, over three phantom seeds, against every single-channel
image-only axis of the mode. A pair at or above the threshold must be declared in
KNOWN_COLLINEAR_PAIRS, and a declared volume-only pair that no longer reaches it is stale.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import pytest
import torch

from spectramr.infrastructure.physics.degradations import (
    DEGRADATION_REGISTRY,
    KNOWN_COLLINEAR_PAIRS,
    AcquisitionGeometry,
    AcquisitionMode,
    apply_degradation,
    is_known_collinear,
    phase_only_degradations,
    slice_capable_degradations,
)
from spectramr.infrastructure.physics.degradations.registry_queries import (
    requirement_free_degradations,
)

M = AcquisitionMode
THRESHOLD = 0.9
SEEDS = (0, 1, 2)
VOXEL = (5.0, 1.0, 1.0)
GEOMETRIES = {
    m: AcquisitionGeometry(mode=m, voxel_mm=VOXEL) for m in (M.MULTISLICE_2D, M.VOLUME_3D)
}
VOLUME_ONLY = sorted(set(DEGRADATION_REGISTRY) - set(slice_capable_degradations()))
# Phase-only: its magnitude residual is zero by construction.
PHASE_ONLY = set(phase_only_degradations())


def _phantom(seed: int) -> torch.Tensor:
    """Textured, asymmetric, zero-background, phase-bearing: ``(1, 1, 8, 48, 48)``."""
    g = torch.Generator().manual_seed(seed)
    z, y, x = torch.meshgrid(*(torch.linspace(-1, 1, n) for n in (8, 48, 48)), indexing="ij")
    body = (((x - 0.05) / 0.8) ** 2 + (y / 0.9) ** 2 + (z / 1.2) ** 2 < 1).float()
    mag = body * (0.5 + 0.5 * torch.rand(8, 48, 48, generator=g))
    return (mag * torch.exp(1j * (1.2 * x + 0.8 * y))).to(torch.complex64)[None, None]


def residuals(mode: AcquisitionMode, seed: int) -> dict[str, torch.Tensor]:
    """Unit-normalised magnitude residual of every axis the mode supports; an axis whose
    residual is zero (other than the phase-only one) raises, so inertness cannot score 0."""
    x = _phantom(seed)
    out = {}
    for name in requirement_free_degradations():  # a one-channel phantom with no side inputs
        if mode not in DEGRADATION_REGISTRY[name].supported_modes or name in PHASE_ONLY:
            continue
        r = apply_degradation(name, x, 1.0, seed=0, geometry=GEOMETRIES[mode]).abs() - x.abs()
        r = r.flatten()
        norm = float(r.norm())
        assert norm > 1e-9, f"{name} has a zero magnitude residual in {mode.value}"
        out[name] = r / norm
    return out


def offenders(
    resid: Mapping[str, torch.Tensor],
    volume_only: Sequence[str],
    known: Callable[[str, str], bool] = is_known_collinear,
    threshold: float = THRESHOLD,
) -> list[tuple[str, str, float]]:
    """``(volume-only axis, other axis, |cos|)`` for every undeclared pair at or above threshold."""
    rows = []
    for a in volume_only:
        for b in resid:
            if a == b or (b in volume_only and b < a) or a not in resid:
                continue
            cos = abs(float(torch.dot(resid[a], resid[b])))
            if cos >= threshold and not known(a, b):
                rows.append((a, b, cos))
    return rows


def stale_declarations(
    best: Mapping[tuple[str, str], float],
    volume_only: Sequence[str],
    pairs: Mapping[tuple[str, str], str] = KNOWN_COLLINEAR_PAIRS,
    threshold: float = THRESHOLD,
) -> list[tuple[str, str]]:
    """Declared pairs with a volume-only member whose largest measured |cos| is below threshold.

    ``best`` maps an unordered pair (either order) to its largest |cos| over modes and seeds; a
    declared pair that was never measured is stale too, since nothing then justifies it.
    """
    measured = {tuple(sorted(k)): v for k, v in best.items()}
    return [
        pair
        for pair in sorted(pairs)
        if set(pair) & set(volume_only) and measured.get(tuple(sorted(pair)), 0.0) < threshold
    ]


def max_cosines(mode: AcquisitionMode) -> list[tuple[float, str, str]]:
    """The largest |cos| of each volume-only pair over the seeds, largest first."""
    best: dict[tuple[str, str], float] = {}
    for seed in SEEDS:
        resid = residuals(mode, seed)
        for a in VOLUME_ONLY:
            for b in resid:
                if a in resid and a != b and not (b in VOLUME_ONLY and b < a):
                    c = abs(float(torch.dot(resid[a], resid[b])))
                    best[a, b] = max(best.get((a, b), 0.0), c)
    return sorted(((c, a, b) for (a, b), c in best.items()), reverse=True)


def test_the_volume_only_axes_are_measured() -> None:
    # Non-vacuity: the exact volume-only set is pinned once, by the golden gate's exemption check.
    assert {
        "inflow_enhancement",
        "off_resonance_slice_shift",
        "slice_crosstalk",
        "spin_history",
        "through_plane_pve",
    } <= set(VOLUME_ONLY)


@pytest.mark.parametrize("mode", list(GEOMETRIES), ids=lambda m: m.value)
def test_no_undeclared_pair_is_collinear(mode) -> None:
    for seed in SEEDS:
        found = offenders(residuals(mode, seed), VOLUME_ONLY)
        assert not found, (
            f"{mode.value}, seed {seed}: near-duplicate pairs not declared in "
            f"KNOWN_COLLINEAR_PAIRS: {found}. Make them physically distinct or declare them."
        )


def test_the_detector_flags_planted_copies() -> None:
    resid = residuals(M.MULTISLICE_2D, 0)
    base = resid["gibbs"]
    noise = torch.randn(base.shape, generator=torch.Generator().manual_seed(5))
    noise = noise - torch.dot(noise, base) * base  # orthogonal to the base residual
    noise = noise / noise.norm()
    c = 0.3287  # tan(acos 0.95): |cos| of the near copy is 1/sqrt(1 + c^2) = 0.95
    near = base + c * noise
    planted = {**resid, "fake_exact": base.clone(), "fake_near": near / near.norm()}
    found = {(a, b): cos for a, b, cos in offenders(planted, ["fake_exact", "fake_near"])}
    assert found[("fake_exact", "gibbs")] == pytest.approx(1.0, abs=1e-5)
    assert found[("fake_near", "gibbs")] == pytest.approx(0.95, abs=1e-3)
    # A declared pair is not flagged, and a pair under the threshold is not either.
    declared = offenders(planted, ["fake_near"], known=lambda a, b: "gibbs" in (a, b))
    assert ("fake_near", "gibbs") not in {(a, b) for a, b, _ in declared}
    assert offenders(resid, ["slice_crosstalk"], threshold=1.01) == []


def test_a_dropped_pairing_filter_would_be_noticed() -> None:
    # offenders() must see the volume-only axes among the residuals at all, pair each with every
    # other axis once, and never with itself.
    resid = residuals(M.MULTISLICE_2D, 0)
    assert set(VOLUME_ONLY) <= set(resid)
    assert offenders(resid, ["not_an_axis"]) == []
    rows = {(a, b) for a, b, _ in offenders(resid, VOLUME_ONLY, threshold=0.0)}
    assert ("slice_crosstalk", "gibbs") in rows
    assert ("slice_crosstalk", "slice_crosstalk") not in rows
    pair = ("off_resonance_slice_shift", "slice_crosstalk")
    assert (pair in rows) != (pair[::-1] in rows)


def test_no_declared_volume_only_pair_is_stale() -> None:
    best = {(a, b): c for mode in GEOMETRIES for c, a, b in max_cosines(mode)}
    assert stale_declarations(best, VOLUME_ONLY) == []


def test_a_planted_stale_declaration_is_flagged() -> None:
    best = {("slice_crosstalk", "gibbs"): 0.4, ("iq_ghost", "slice_crosstalk"): 0.95}
    planted = {
        ("gibbs", "slice_crosstalk"): "planted: measured 0.4, below the threshold",
        ("iq_ghost", "slice_crosstalk"): "planted: measured 0.95, still collinear",
        ("never_measured", "slice_crosstalk"): "planted: no measurement justifies it",
        ("gibbs", "t2star_blur"): "a 2-D pair, not this suite's",
    }
    assert stale_declarations(best, ["slice_crosstalk"], pairs=planted) == [
        ("gibbs", "slice_crosstalk"),
        ("never_measured", "slice_crosstalk"),
    ]
