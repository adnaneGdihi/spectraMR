"""The degradation golden gate must match today's outputs and be able to fail."""

from __future__ import annotations

import dataclasses
import json

import pytest

from scripts.ci import check_degradation_golden as gate
from scripts.ci import degradation_golden_inputs as inputs
from spectramr.infrastructure.physics.degradations import DEGRADATION_REGISTRY
from spectramr.infrastructure.physics.degradations.geometry import (
    AcquisitionGeometry,
    AcquisitionMode,
)
from spectramr.infrastructure.physics.degradations.registry_queries import (
    slice_capable_degradations,
)
from spectramr.infrastructure.physics.degradations.spec import Requirement, SliceKernel


@pytest.fixture(scope="module")
def fixture() -> dict:
    return json.loads(gate.FIXTURE.read_text())


@pytest.fixture(scope="module")
def current() -> dict:
    return gate.compute_cases()


def _skip_if_foreign(fixture: dict) -> None:
    bad = gate.fingerprint_mismatches(fixture)
    if bad:
        pytest.skip("golden fixture is from another environment: " + "; ".join(bad))


def test_hashes_match_fixture(current, fixture) -> None:
    _skip_if_foreign(fixture)
    diff = gate.compare(current, fixture)
    assert diff == {"ADDED": [], "REMOVED": [], "CHANGED": []}


def test_every_registry_key_is_covered(fixture) -> None:
    covered = {k.split("|")[0].split("+")[0] for k in fixture["cases"]}
    assert not gate.uncovered_axes(DEGRADATION_REGISTRY, covered)
    assert {"simulator|default_nomotion_nob0_nob1", "tio_digital_twin_degradation"} <= set(
        fixture["cases"]
    )


def test_declared_exemptions_equal_the_registry_derived_sets() -> None:
    assert inputs.exemption_errors() == []
    assert inputs.registry_exemptions()["volume-only"] == frozenset(DEGRADATION_REGISTRY) - set(
        slice_capable_degradations()
    )


_KINDS = ("volume-only", "multicoil-only", "qmaps")
_CONSTANT = {
    "volume-only": "VOLUME_ONLY_AXES",
    "multicoil-only": "MULTICOIL_ONLY_AXES",
    "qmaps": "QMAPS_AXES",
}


@pytest.mark.parametrize("kind", _KINDS)
def test_planted_wrongly_exempted_axis_is_flagged(monkeypatch, kind: str) -> None:
    declared = inputs.declared_exemptions()[kind]
    monkeypatch.setattr(inputs, _CONSTANT[kind], declared | {"gibbs"})
    assert inputs.exemption_errors() == [f"{kind}: wrongly exempted: gibbs"]


@pytest.mark.parametrize("kind", _KINDS)
def test_planted_undeclared_axis_is_flagged(monkeypatch, kind: str) -> None:
    declared = sorted(inputs.declared_exemptions()[kind])
    monkeypatch.setattr(inputs, _CONSTANT[kind], frozenset(declared[1:]))
    assert inputs.exemption_errors() == [f"{kind}: undeclared: {declared[0]}"]


@pytest.mark.parametrize("kind", _KINDS)
def test_planted_registry_side_corruption_is_flagged(kind: str) -> None:
    required = inputs.registry_exemptions()
    required[kind] = required[kind] | {"ghost_axis"}
    assert inputs.exemption_errors(required=required) == [f"{kind}: undeclared: ghost_axis"]


def test_planted_axis_missing_from_the_fixture_is_reported() -> None:
    covered = {n for n in DEGRADATION_REGISTRY if n != "gibbs"}
    assert gate.uncovered_axes(DEGRADATION_REGISTRY, covered) == {"gibbs"}


def test_fingerprint_mismatch_raises(current, fixture) -> None:
    foreign = {**fixture, "fingerprint": {**fixture["fingerprint"], "torch_version": "0.0.0"}}
    with pytest.raises(RuntimeError, match="torch_version"):
        gate.compare(current, foreign)


def test_a_fixture_without_a_fingerprint_names_every_field() -> None:
    bad = gate.fingerprint_mismatches({"cases": {}})
    assert {b.split(":")[0] for b in bad} == set(gate.fingerprint())
    assert "torchkbnufft_version" in gate.fingerprint()


def test_fixture_has_no_error_entries(fixture) -> None:
    assert [k for k, v in fixture["cases"].items() if "error" in v] == []


def test_planted_perturbation_flags_exactly_that_operator(monkeypatch, fixture) -> None:
    _skip_if_foreign(fixture)
    spec = DEGRADATION_REGISTRY["gibbs"]
    original = spec.fn

    def perturbed(x, theta, **kw):
        return original(x, theta, **kw) * (1 + 1e-6)

    impls = {**spec.implementations, AcquisitionMode.SLICE_2D: SliceKernel(perturbed)}
    monkeypatch.setitem(
        DEGRADATION_REGISTRY, "gibbs", dataclasses.replace(spec, implementations=impls)
    )
    diff = gate.compare(gate.compute_cases(), fixture)
    assert diff["ADDED"] == []
    assert diff["REMOVED"] == []
    changed = {k.split("|")[0] for k in diff["CHANGED"]}
    assert changed == {"gibbs"}


def _fixture_of(cases: dict) -> dict:
    return {"fingerprint": gate.fingerprint(), "cases": cases}


def test_add_merges_only_the_new_cases() -> None:
    fixture = _fixture_of({"a": {"sha256": "1"}, "b": {"sha256": "2"}})
    current = {"a": {"sha256": "1"}, "b": {"sha256": "2"}, "c": {"sha256": "3"}}
    merged = gate.add_cases(current, fixture)
    assert merged["cases"] == current
    assert merged["fingerprint"] == fixture["fingerprint"]


@pytest.mark.parametrize(
    ("current", "label"),
    [
        ({"a": {"sha256": "1"}, "b": {"sha256": "changed"}, "c": {"sha256": "3"}}, "CHANGED"),
        ({"a": {"sha256": "1"}, "c": {"sha256": "3"}}, "REMOVED"),
    ],
)
def test_add_refuses_when_an_existing_case_moved(current: dict, label: str) -> None:
    fixture = _fixture_of({"a": {"sha256": "1"}, "b": {"sha256": "2"}})
    with pytest.raises(RuntimeError, match=label):
        gate.add_cases(current, fixture)


def test_add_refuses_a_foreign_fixture() -> None:
    fixture = _fixture_of({"a": {"sha256": "1"}})
    fixture["fingerprint"] = {**fixture["fingerprint"], "torch_version": "0.0.0"}
    with pytest.raises(RuntimeError, match="torch_version"):
        gate.add_cases({"a": {"sha256": "1"}, "c": {"sha256": "3"}}, fixture)


def test_main_add_writes_only_the_new_cases(monkeypatch, tmp_path) -> None:
    path = tmp_path / "golden.json"
    old = {"a": {"sha256": "1"}, "b": {"sha256": "2"}}
    path.write_text(json.dumps(_fixture_of(old), indent=1, sort_keys=True) + "\n")
    before = path.read_text().splitlines()
    monkeypatch.setattr(gate, "FIXTURE", path)
    monkeypatch.setattr(gate, "compute_cases", lambda: {**old, "c": {"sha256": "3"}})
    assert gate.main(["--add"]) == 0
    written = json.loads(path.read_text())
    assert written["cases"] == {**old, "c": {"sha256": "3"}}
    after = path.read_text().splitlines()
    assert [line for line in before if line not in after] == []  # additions only
    assert gate.main([]) == 0


def test_main_add_leaves_the_fixture_alone_when_a_case_moved(monkeypatch, tmp_path) -> None:
    path = tmp_path / "golden.json"
    path.write_text(json.dumps(_fixture_of({"a": {"sha256": "1"}}), indent=1, sort_keys=True))
    text = path.read_text()
    monkeypatch.setattr(gate, "FIXTURE", path)
    monkeypatch.setattr(gate, "compute_cases", lambda: {"a": {"sha256": "x"}, "c": {"sha256": "3"}})
    with pytest.raises(RuntimeError, match="CHANGED"):
        gate.main(["--add"])
    assert path.read_text() == text


def test_every_case_sits_on_a_shape_its_axis_is_hashed_on(current) -> None:
    assert inputs.misplaced_cases(current) == []
    assert {k.split("|")[0] for k in current} >= inputs.MULTICOIL_ONLY_AXES | inputs.QMAPS_AXES
    assert not {k.split("|")[0] for k in current} & inputs.VOLUME_ONLY_AXES


def test_planted_misplaced_cases_are_flagged() -> None:
    cases = {
        "coil_dropout|1x4x48x40|t1.0|s0": {},
        "coil_dropout|2x1x64x64|t1.0|s0": {},
        "inflow_enhancement|1x4x48x40|t1.0|s0": {},
        "gibbs|2x1x64x64|t1.0|s0": {},
        "simulator|default_nomotion_nob0_nob1": {},
    }
    assert inputs.misplaced_cases(cases) == [
        "coil_dropout|2x1x64x64|t1.0|s0",
        "inflow_enhancement|1x4x48x40|t1.0|s0",
    ]


def test_required_input_builders_match_the_registry_and_run_in_every_mode() -> None:
    import torch

    from spectramr.infrastructure.physics.degradations import apply_degradation

    inputs.validate_required_inputs(DEGRADATION_REGISTRY)
    assert inputs.QMAPS_AXES  # equal to the registry's q-map axes (exemption_errors)
    for name in sorted(inputs.QMAPS_AXES):
        for mode, x in (
            (AcquisitionMode.SLICE_2D, torch.ones(1, 2, 8, 8)),
            (AcquisitionMode.MULTISLICE_2D, torch.ones(1, 1, 3, 8, 8, dtype=torch.complex64)),
            (AcquisitionMode.VOLUME_3D, torch.ones(1, 1, 3, 8, 8)),
        ):
            geometry = None if x.dim() == 4 else AcquisitionGeometry(mode, voxel_mm=None)
            kw = inputs.required_kwargs(name, x)
            assert apply_degradation(name, x, 0.5, geometry=geometry, **kw).shape == x.shape
    assert inputs.required_kwargs("gibbs", torch.ones(1, 1, 8, 8)) == {}


def _with_requirement(name: str, req) -> dict:
    return {
        **DEGRADATION_REGISTRY,
        name: dataclasses.replace(DEGRADATION_REGISTRY[name], requires=frozenset({req})),
    }


def test_planted_requirement_no_kind_builds_is_red() -> None:
    specs = _with_requirement("gibbs", Requirement.VESSEL_MAP)
    with pytest.raises(RuntimeError, match=r"gibbs: builders leave"):
        inputs.validate_required_inputs(specs)


def test_planted_requirement_without_a_declared_kind_is_flagged(monkeypatch) -> None:
    planted = _with_requirement("gibbs", Requirement.VESSEL_MAP)["gibbs"]
    monkeypatch.setitem(DEGRADATION_REGISTRY, "gibbs", planted)
    assert inputs.exemption_errors() == ["requires vessel_map: undeclared: gibbs"]


def test_planted_builder_that_does_not_satisfy_requires_is_red() -> None:
    builders = {"qmaps": lambda x: {"qmap": {}}}
    with pytest.raises(RuntimeError, match="imperfect_inversion: builders leave"):
        inputs.validate_required_inputs(DEGRADATION_REGISTRY, builders)


def test_planted_builder_that_fails_only_on_the_5d_probe_is_red() -> None:
    real = inputs.KIND_INPUTS["qmaps"]
    builders = {"qmaps": lambda x: real(x) if x.dim() == 4 else {}}
    with pytest.raises(
        RuntimeError, match=r"imperfect_inversion: builders leave .* on \(1, 2, 3, 8, 8\)"
    ):
        inputs.validate_required_inputs(DEGRADATION_REGISTRY, builders)


def test_planted_stale_builder_or_axis_kwargs_is_red(monkeypatch) -> None:
    stale = {**inputs.KIND_INPUTS, "no_such_kind": lambda x: {}}
    with pytest.raises(RuntimeError, match="builder for a kind that is not declared: no_such_kind"):
        inputs.validate_required_inputs(DEGRADATION_REGISTRY, stale)
    monkeypatch.setitem(inputs.AXIS_KWARGS, "no_such_axis", {})
    with pytest.raises(RuntimeError, match="no_such_axis: AXIS_KWARGS for an axis that does not"):
        inputs.validate_required_inputs(DEGRADATION_REGISTRY)


def test_the_gate_main_path_refuses_a_stale_builder(monkeypatch) -> None:
    monkeypatch.setitem(inputs.AXIS_KWARGS, "no_such_axis", {})
    monkeypatch.setattr(gate, "compute_cases", lambda: pytest.fail("cases computed"))
    assert gate.main([]) == 1


def test_the_gate_main_path_refuses_a_wrong_exemption(monkeypatch) -> None:
    monkeypatch.setattr(inputs, "VOLUME_ONLY_AXES", inputs.VOLUME_ONLY_AXES | {"gibbs"})
    monkeypatch.setattr(gate, "compute_cases", lambda: pytest.fail("cases computed"))
    assert gate.main([]) == 1
