"""``line_axis_for``: the one owner of ``mask_direction`` -> ``line_axis`` (#2160)."""

from __future__ import annotations

import pytest

from spectramr.infrastructure.physics.line_axis import MEASURE_SCRIPT, line_axis_for


@pytest.mark.parametrize(
    ("direction", "axis", "expected"),
    [
        ("phase", -1, "x"),
        ("phase", -2, "y"),
        ("readout", -1, "y"),
        ("readout", -2, "x"),
    ],
)
def test_phase_and_readout_resolve_through_the_declared_axis(
    direction: str, axis: int, expected: str
) -> None:
    assert line_axis_for(direction, axis) == expected


@pytest.mark.parametrize("raw", ["x", "y"])
def test_raw_axis_names_pass_through_without_a_declaration(raw: str) -> None:
    assert line_axis_for(raw, None) == raw
    assert line_axis_for(raw, -1) == raw


@pytest.mark.parametrize("direction", ["phase", "readout"])
def test_a_physical_direction_without_an_axis_raises_naming_the_script(direction: str) -> None:
    """PLANTED: the literal ``phase -> 'y'`` this replaced made every M4Raw mask
    remove readout lines; an absent axis must never pick one."""
    with pytest.raises(ValueError, match=MEASURE_SCRIPT):
        line_axis_for(direction, None)


@pytest.mark.parametrize("axis", [0, 1, -3, True, "x"])
def test_an_unknown_axis_raises(axis: object) -> None:
    with pytest.raises(ValueError, match="phase_encode_axis"):
        line_axis_for("phase", axis)  # type: ignore[arg-type]


def test_an_unknown_direction_raises() -> None:
    with pytest.raises(ValueError, match="mask_direction"):
        line_axis_for("diagonal", -1)
