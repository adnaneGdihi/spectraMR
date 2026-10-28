"""``data.acquisition``: the dataset's phase-encode axis and acquired lines (#2160)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from spectramr.config.schemas.data import DataConfigSchema
from spectramr.config.schemas.data_acquisition import DataAcquisitionSchema

M4RAW = {"phase_encode_axis": -1, "acquired_pe_lines": {"FLAIR": 198, "T1": 195, "T2": 195}}


@pytest.mark.parametrize(
    "block",
    [
        M4RAW,
        {"phase_encode_axis": -1, "acquired_pe_lines": "measured"},
        {"phase_encode_axis": -2, "acquired_pe_lines": 195},
    ],
    ids=["per_contrast", "measured", "single_count"],
)
def test_the_three_declared_shapes_load(block: dict) -> None:
    data = DataConfigSchema(acquisition=block)
    assert data.acquisition == DataAcquisitionSchema(**block)


def test_absent_means_undeclared_not_a_default() -> None:
    """No default axis: a phase/readout mask then raises at build instead of guessing."""
    assert DataConfigSchema().acquisition is None


@pytest.mark.parametrize(
    "block",
    [
        {"phase_encode_axis": 0, "acquired_pe_lines": 195},
        {"phase_encode_axis": -3, "acquired_pe_lines": 195},
        {"phase_encode_axis": True, "acquired_pe_lines": 195},
        {"phase_encode_axis": "-1", "acquired_pe_lines": 195},
        {"phase_encode_axis": -1, "acquired_pe_lines": 0},
        {"phase_encode_axis": -1, "acquired_pe_lines": "measure"},
        {"phase_encode_axis": -1, "acquired_pe_lines": {}},
        {"phase_encode_axis": -1, "acquired_pe_lines": {"T1": 0}},
        {"phase_encode_axis": -1, "acquired_pe_lines": {" ": 195}},
        {"phase_encode_axis": -1, "acquired_pe_lines": 195.0},
        {"phase_encode_axis": -1},
        {"acquired_pe_lines": 195},
        {**M4RAW, "phase_encode_dim": -1},
    ],
    ids=[
        "axis_0",
        "axis_-3",
        "axis_bool",
        "axis_str",
        "count_0",
        "count_typo",
        "empty_mapping",
        "contrast_0",
        "blank_contrast",
        "count_float",
        "count_missing",
        "axis_missing",
        "unknown_key",
    ],
)
def test_an_illegal_block_raises(block: dict) -> None:
    with pytest.raises(ValidationError):
        DataAcquisitionSchema(**block)


def test_the_block_is_frozen() -> None:
    acquisition = DataAcquisitionSchema(**M4RAW)
    with pytest.raises(ValidationError):
        acquisition.phase_encode_axis = -2  # type: ignore[misc]
