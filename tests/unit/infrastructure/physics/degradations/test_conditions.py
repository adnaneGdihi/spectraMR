"""Scanner conditions: theta-scaled kwargs, named as the delegates read them."""

from __future__ import annotations

import inspect

import pytest

from spectramr.infrastructure.physics.degradations import DEGRADATION_REGISTRY
from spectramr.infrastructure.physics.degradations.geometry import AcquisitionMode
from spectramr.infrastructure.physics.scanner_presets import FASTMRI_3T, M4RAW_03T

SHAPE = (1, 1, 32, 64)


@pytest.mark.parametrize(
    "name", [n for n, s in DEGRADATION_REGISTRY.items() if s.condition is not None]
)
def test_every_emitted_kwarg_is_read_by_every_volume_impl(name: str) -> None:
    # A kwarg the impl swallows through **kw would make the conditioning a facade.
    spec = DEGRADATION_REGISTRY[name]
    keys = set(spec.condition(FASTMRI_3T, 0.5, FASTMRI_3T.geometry, SHAPE))
    for mode, impl in spec.implementations.items():
        sig = inspect.signature(impl).parameters
        if mode is AcquisitionMode.SLICE_2D or any(p.kind is p.VAR_KEYWORD for p in sig.values()):
            continue  # **kw delegates are checked by output in test_scanner_profile.py
        params = set(sig)
        assert keys <= params, (name, mode, keys - params)


@pytest.mark.parametrize(
    ("name", "key"),
    [
        ("chemical_shift", "shift_pixels"),
        ("b0", "strength"),
        ("b0_geometric_distortion", "b0_strength"),
    ],
)
def test_theta_scaled_amount_grows_with_theta_and_field(name: str, key: str) -> None:
    cond = DEGRADATION_REGISTRY[name].condition
    lo, hi = (cond(FASTMRI_3T, t, FASTMRI_3T.geometry, SHAPE)[key] for t in (0.2, 0.8))
    assert 0.0 < lo < hi
    assert cond(M4RAW_03T, 0.8, M4RAW_03T.geometry, SHAPE)[key] < hi


def test_dwell_follows_the_readout_width() -> None:
    cond = DEGRADATION_REGISTRY["b0_geometric_distortion"].condition
    dwell = cond(FASTMRI_3T, 1.0, FASTMRI_3T.geometry, SHAPE)["dwell_time"]
    assert dwell == pytest.approx(1.0 / (250.0 * 64))


def test_concomitant_passes_the_profile_field() -> None:
    cond = DEGRADATION_REGISTRY["concomitant_phase"].condition
    assert cond(M4RAW_03T, 0.5, M4RAW_03T.geometry, SHAPE) == {"b0_T": 0.3}
