"""LocalSeverity: the user's physical range -> local theta -> native theta + endpoint kwargs."""

from __future__ import annotations

import itertools
import json

import pytest

from spectramr.config.schemas.enums import AccelerationSchedule
from spectramr.config.schemas.severity_axes import SeverityAxis
from spectramr.infrastructure.physics.degradations import DEGRADATION_REGISTRY
from spectramr.infrastructure.physics.scanner_presets import M4RAW_03T
from spectramr.infrastructure.physics.severity import severity_grid
from spectramr.infrastructure.physics.severity_local import (
    LocalSeverity,
    load_severity_axes_file,
    reject_profile_conflicts,
    resolve_axes,
)

ALL_SCHEDULES = tuple(AccelerationSchedule)


def _axis(start: float, end: float, schedule: str = "linear", **kw: object) -> SeverityAxis:
    return SeverityAxis(start=start, end=end, schedule=AccelerationSchedule(schedule), **kw)


@pytest.mark.parametrize("explicit", [False, True])
def test_passthrough_is_bit_identical(explicit: bool) -> None:
    primary = DEGRADATION_REGISTRY["cartesian_undersamp"].severity.primary
    setting = _axis(primary.at_theta_min, primary.at_theta_max) if explicit else None
    local = LocalSeverity.resolve("cartesian_undersamp", setting)
    assert local.is_passthrough
    assert local.degrader_kwargs() == {}
    for t in severity_grid(20):
        assert local.native_theta(t) is t
    assert local.thetas(20) == tuple(severity_grid(20))
    assert local.thetas(20, theta_min=0.05) == tuple(severity_grid(20, theta_min=0.05))


@pytest.mark.parametrize("schedule", ALL_SCHEDULES)
def test_endpoints_are_exact(schedule: AccelerationSchedule) -> None:
    for name, start, end in (
        ("cartesian_undersamp", 2.0, 16.0),
        ("complex_gaussian", 30.0, 20.0),
        ("field_contrast", 1.5, 0.01),
    ):
        local = LocalSeverity.resolve(name, _axis(start, end, schedule.value))
        assert local.physical(0.0) == start
        assert local.physical(1.0) == end


def test_log_doubles_acceleration() -> None:
    local = LocalSeverity.resolve("cartesian_undersamp", _axis(2.0, 16.0, "log", n_steps=4))
    physical = [local.physical(t) for t in local.thetas(8)]
    assert physical == pytest.approx([2.0, 4.0, 8.0, 16.0])
    assert physical[0] == 2.0 and physical[-1] == 16.0


def test_decreasing_snr_monotone() -> None:
    local = LocalSeverity.resolve("complex_gaussian", _axis(30.0, 20.0))
    grid = local.thetas(8)
    physical = [local.physical(t) for t in grid]
    native = [local.native_theta(t) for t in grid]
    assert all(a > b for a, b in itertools.pairwise(physical))
    assert local.physical(1.0) == 20.0
    assert all(a < b for a, b in itertools.pairwise(native))
    assert local.degrader_kwargs() == {}


def test_log_rejects_nonpositive_endpoint() -> None:
    with pytest.raises(ValueError, match=r"gibbs.*log.*> 0"):
        LocalSeverity.resolve("gibbs", _axis(0.0, 0.5, "log"))


def test_beyond_declared_without_kwarg_raises() -> None:
    primary = DEGRADATION_REGISTRY["b1_transmit"].severity.primary
    assert primary.endpoint_kwargs == (None, None)
    with pytest.raises(ValueError, match=r"b1_transmit.*declared"):
        LocalSeverity.resolve("b1_transmit", _axis(0.0, primary.at_theta_max * 2.0))


def test_beyond_declared_with_kwarg_sets_kwarg() -> None:
    local = LocalSeverity.resolve("cartesian_undersamp", _axis(2.0, 16.0))
    assert not local.is_passthrough
    assert local.degrader_kwargs() == {"R_max": 16.0}
    assert local.native_theta(0.0) == pytest.approx((2 - 1) / (16 - 1))
    assert local.native_theta(1.0) == 1.0


def test_endpoint_scale_applies_to_kwarg() -> None:
    primary = DEGRADATION_REGISTRY["slice_pve"].severity.primary
    local = LocalSeverity.resolve("slice_pve", _axis(0.0, 10.0))
    assert local.degrader_kwargs() == {"max_kernel_px": 10.0 * primary.endpoint_scale}


def test_non_extendable_start_raises() -> None:
    with pytest.raises(ValueError, match=r"field_contrast.*declared"):
        LocalSeverity.resolve("field_contrast", _axis(7.0, 0.05))


def test_valid_bounds_raise() -> None:
    with pytest.raises(ValueError, match="cartesian_undersamp"):
        LocalSeverity.resolve("cartesian_undersamp", _axis(0.5, 8.0))


def test_reversed_direction_raises() -> None:
    with pytest.raises(ValueError, match=r"cartesian_undersamp.*direction"):
        LocalSeverity.resolve("cartesian_undersamp", _axis(8.0, 2.0))


def test_unknown_axis_raises() -> None:
    with pytest.raises(KeyError):
        LocalSeverity.resolve("nope", None)


@pytest.mark.parametrize("schedule", ALL_SCHEDULES)
def test_native_theta_always_in_unit_interval(schedule: AccelerationSchedule) -> None:
    for name, start, end in (
        ("cartesian_undersamp", 2.0, 16.0),
        ("cartesian_undersamp", 2.0, 6.0),
        ("complex_gaussian", 60.0, 1.0),
        ("field_contrast", 3.0, 0.01),
        ("gibbs", 0.9, 0.05),
    ):
        local = LocalSeverity.resolve(name, _axis(start, end, schedule.value))
        for t in local.thetas(33):
            assert 0.0 <= local.native_theta(t) <= 1.0


def test_n_steps_override_and_zero_anchor() -> None:
    local = LocalSeverity.resolve("cartesian_undersamp", _axis(2.0, 6.0, n_steps=3))
    assert local.thetas(20) == tuple(severity_grid(3, theta_min=0.0))
    assert local.thetas(20)[0] == 0.0


def test_provenance_is_json_and_shows_coupling() -> None:
    local = LocalSeverity.resolve("rigid_motion", _axis(0.0, 20.0))
    prov = local.provenance()
    json.dumps(prov)
    assert prov["primary"] == "max_translation"
    assert prov["degrader_kwargs"] == {"max_translation_px": 20.0}
    assert prov["passthrough"] is False
    co = prov["co_varying"]
    assert isinstance(co, list) and co[0]["name"] == "max_rotation"
    assert co[0]["at_start"] == 0.0 and co[0]["at_end"] == pytest.approx(0.1)


def test_resolve_axes() -> None:
    names = ("cartesian_undersamp", "complex_gaussian")
    out = resolve_axes(names, {"cartesian_undersamp": _axis(2.0, 16.0)})
    assert list(out) == list(names)
    assert out["complex_gaussian"].is_passthrough
    assert not out["cartesian_undersamp"].is_passthrough
    with pytest.raises(ValueError, match="gibbs"):
        resolve_axes(names, {"gibbs": _axis(1.0, 0.5)})
    with pytest.raises(KeyError):
        resolve_axes(("nope",), {})


def test_declared_endpoints_on_geometric_primary_are_not_passthrough() -> None:
    # Linear in physical units differs from the declared geometric curve, so it is applied.
    local = LocalSeverity.resolve("field_contrast", _axis(3.0, 0.05))
    assert not local.is_passthrough
    assert local.physical(0.5) == pytest.approx(1.525)


def test_reject_profile_conflicts() -> None:
    # The profile's condition sets b0_strength, the physical parameter of the range.
    ranged = LocalSeverity.resolve("b0_geometric_distortion", _axis(0.0, 0.5))
    with pytest.raises(ValueError, match=r"scanner profile 'm4raw_0.3T'.*b0_strength"):
        reject_profile_conflicts(ranged, M4RAW_03T)
    # Without a range the profile conditions the degrader as before.
    reject_profile_conflicts(LocalSeverity.resolve("b0_geometric_distortion", None), M4RAW_03T)
    # A ranged axis whose profile condition touches nothing it owns is accepted.
    reject_profile_conflicts(LocalSeverity.resolve("rician", _axis(0.01, 0.08)), M4RAW_03T)


def test_load_severity_axes_file_checks_registered_names(tmp_path) -> None:
    p = tmp_path / "axes.yaml"
    p.write_text("axes:\n  rician: {start: 0.01, end: 0.08}\n")
    assert load_severity_axes_file(p)["rician"].end == 0.08
    p.write_text("axes:\n  not_a_degradation: {start: 0, end: 1}\n")
    with pytest.raises(ValueError, match=r"not_a_degradation.*not registered"):
        load_severity_axes_file(p)
