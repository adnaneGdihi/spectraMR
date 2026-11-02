"""SeverityAxis / SeverityAxesFile: the per-degradation physical-range schema."""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from spectramr.config.schemas.enums import AccelerationSchedule
from spectramr.config.schemas.severity_axes import SeverityAxesFile, SeverityAxis, load_file


def test_extra_key_rejected() -> None:
    with pytest.raises(ValidationError):
        SeverityAxis.model_validate({"start": 2.0, "end": 16.0, "stop": 3})
    with pytest.raises(ValidationError):
        SeverityAxesFile.model_validate({"axes": {}, "extra": 1})


def test_yaml_roundtrip() -> None:
    text = """
axes:
  cartesian_undersamp: {start: 2, end: 16, schedule: log, n_steps: 4}
  complex_gaussian: {start: 30, end: 20}
"""
    parsed = SeverityAxesFile.model_validate(yaml.safe_load(text))
    axis = parsed.axes["cartesian_undersamp"]
    assert axis.schedule is AccelerationSchedule.LOG
    assert (axis.start, axis.end, axis.n_steps) == (2.0, 16.0, 4)
    assert parsed.axes["complex_gaussian"].schedule is AccelerationSchedule.LINEAR
    dumped = yaml.safe_dump(parsed.model_dump(mode="json"))
    assert SeverityAxesFile.model_validate(yaml.safe_load(dumped)) == parsed


@pytest.mark.parametrize(
    "payload",
    [
        {"start": 2.0, "end": 2.0},
        {"start": 2.0, "end": 16.0, "n_steps": 1},
        {"start": 2.0, "end": 16.0, "power": 0.0},
        {"start": 2.0, "end": 16.0, "schedule": "bogus"},
        {"start": 2.0, "end": float("inf")},
        {"start": float("nan"), "end": 16.0},
        {"start": 2.0, "end": 16.0, "power": float("inf")},
    ],
)
def test_invalid_rejected(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        SeverityAxis.model_validate(payload)


def test_reject_severity_overlap_names_the_axis_and_owner() -> None:
    from spectramr.config.schemas.severity_axes import reject_severity_overlap

    reject_severity_overlap({"rician": 1}, degradation_ranges={"motion": 1})
    with pytest.raises(ValueError, match=r"\['rician'\].*degradation_ranges.*one owner"):
        reject_severity_overlap({"rician": 1}, degradation_ranges={"rician": 1})


def test_load_file(tmp_path) -> None:
    p = tmp_path / "axes.yaml"
    p.write_text("axes:\n  anything: {start: 2, end: 16, schedule: log}\n")
    assert load_file(p)["anything"].end == 16.0  # names are not checked here
    with pytest.raises(FileNotFoundError, match="no such file"):
        load_file(tmp_path / "absent.yaml")
    p.write_text("axes: [\n")
    with pytest.raises(ValueError):
        load_file(p)
    p.write_text("axes:\n  a: {start: 1, end: 2, bogus: 3}\n")
    with pytest.raises(ValueError):
        load_file(p)
