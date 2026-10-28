"""The four multislice axes' declarations: modes, excluded-mode reasons, shared pieces."""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.degradations import (
    DEGRADATION_REGISTRY,
    AcquisitionGeometry,
    AcquisitionMode,
    Realisation,
    UnsupportedAcquisitionMode,
    apply_degradation,
    slice_capable_degradations,
)
from spectramr.infrastructure.physics.degradations.catalogue.multislice_common import (
    slab_coverage,
)

M = AcquisitionMode
AXES = {
    "slice_crosstalk": {M.MULTISLICE_2D},
    "through_plane_pve": {M.MULTISLICE_2D, M.VOLUME_3D},
    "spin_history": {M.MULTISLICE_2D, M.VOLUME_3D},
    "off_resonance_slice_shift": {M.MULTISLICE_2D, M.VOLUME_3D},
}
GEOM = {
    m: AcquisitionGeometry(mode=m, voxel_mm=(5.0, 1.0, 1.0)) for m in (M.MULTISLICE_2D, M.VOLUME_3D)
}


@pytest.mark.parametrize(("name", "modes"), AXES.items())
def test_declared_modes_are_native_and_exclude_slice_2d(name, modes):
    spec = DEGRADATION_REGISTRY[name]
    assert spec.supported_modes == modes
    assert all(r is Realisation.NATIVE for r in spec.realisation.values())
    assert spec.requires == frozenset()
    assert name not in slice_capable_degradations()
    assert set(spec.unsupported_detail) == set(M) - modes


@pytest.mark.parametrize(
    ("name", "mode", "text"),
    [
        ("slice_crosstalk", M.SLICE_2D, "no neighbouring slice"),
        ("slice_crosstalk", M.VOLUME_3D, "excited once as a whole"),
        ("through_plane_pve", M.SLICE_2D, "slice_pve"),
        ("spin_history", M.SLICE_2D, "slice axis"),
        ("off_resonance_slice_shift", M.SLICE_2D, "slice axis"),
    ],
)
def test_an_excluded_mode_raises_with_its_reason(name, mode, text):
    shape = (1, 1, 8, 8) if mode is M.SLICE_2D else (1, 1, 4, 8, 8)
    geom = None if mode is M.SLICE_2D else GEOM[mode]
    with pytest.raises(UnsupportedAcquisitionMode, match=text) as err:
        apply_degradation(name, torch.ones(shape), 0.5, geometry=geom)
    assert err.value.mode is mode and err.value.supported == AXES[name]


def test_the_four_are_among_the_volume_only_axes():
    # The exact volume-only set is pinned once, by the golden gate's exemption check.
    assert set(AXES) <= set(DEGRADATION_REGISTRY) - set(slice_capable_degradations())


def test_slab_coverage_is_the_voxel_fraction_inside_the_moved_slab():
    zero = slab_coverage(torch.zeros(1, 1, 1, 1, 1), 6)
    assert torch.equal(zero, torch.ones_like(zero))
    up = slab_coverage(torch.full((1, 1, 1, 1, 1), 1.25), 6).flatten()
    assert up.tolist() == pytest.approx([0.0, 0.75, 1, 1, 1, 1])
    down = slab_coverage(torch.full((1, 1, 1, 1, 1), -0.5), 6).flatten()
    assert down.tolist() == pytest.approx([1, 1, 1, 1, 1, 0.5])


@pytest.mark.parametrize(("name", "modes"), AXES.items())
def test_declared_affects_matches_measurement_in_every_mode(name, modes):
    """Re-derive the ``affects`` flag as TestPhaseOnlyRegistry does, on a phase-bearing stack."""
    from .invariants import phantom

    declared = DEGRADATION_REGISTRY[name].affects
    mag = phase = 0.0
    for mode in modes:
        x = phantom(GEOM[mode])
        out = apply_degradation(name, x, 1.0, seed=0, geometry=GEOM[mode])
        mag = max(mag, float((out.abs() - x.abs()).abs().max()))
        w = x.abs() / x.abs().sum()
        dphi = torch.angle(out * torch.conj(x) / (x.abs() ** 2 + 1e-12))
        phase = max(phase, float(torch.sqrt((w * dphi**2).sum())))
    assert (mag > 1e-4) == ("magnitude" in declared)
    assert (phase > 1e-4) == ("phase" in declared)
