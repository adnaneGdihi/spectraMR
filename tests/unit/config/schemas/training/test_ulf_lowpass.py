"""Tests for the shared ULF low-pass forward-operator block (training.ulf_lowpass)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from spectramr.config.schemas.training.ulf_lowpass import UlfLowpassConfig

_VOXELS = {
    "grid_voxel_mm": {"T1w": (0.49, 0.49)},
    "native_voxel_mm": {"T1w": (1.6, 1.6)},
}


@pytest.mark.unit
def test_constant_and_voxel_ratio_load() -> None:
    assert UlfLowpassConfig(cutoff_mode="constant", blur_cutoff=0.5).blur_cutoff == 0.5
    assert UlfLowpassConfig(cutoff_mode="voxel_ratio", **_VOXELS).native_voxel_mm == {
        "T1w": (1.6, 1.6)
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({}, "cutoff_mode"),
        ({"cutoff_mode": "psf"}, "cutoff_mode"),
        ({"cutoff_mode": "constant"}, "requires blur_cutoff"),
        ({"cutoff_mode": "constant", "blur_cutoff": 0.5, **_VOXELS}, "not read"),
        ({"cutoff_mode": "voxel_ratio", "blur_cutoff": 0.5, **_VOXELS}, "not read"),
        ({"cutoff_mode": "voxel_ratio", "grid_voxel_mm": {"T1w": (0.49, 0.49)}}, "requires"),
        (
            {
                "cutoff_mode": "voxel_ratio",
                "grid_voxel_mm": {"T1w": (0.49, 0.49)},
                "native_voxel_mm": {"T2w": (1.6, 1.6)},
            },
            "same contrasts",
        ),
        (
            {
                "cutoff_mode": "voxel_ratio",
                "grid_voxel_mm": {"T1w": (0.49, 0.0)},
                "native_voxel_mm": {"T1w": (1.6, 1.6)},
            },
            "positive",
        ),
    ],
)
def test_inconsistent_blocks_raise(kwargs: dict, match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        UlfLowpassConfig(**kwargs)


@pytest.mark.unit
def test_mounted_on_training() -> None:
    from spectramr.config.schemas.training.base import TrainingStrategyConfigSchema

    assert "ulf_lowpass" in TrainingStrategyConfigSchema.model_fields
