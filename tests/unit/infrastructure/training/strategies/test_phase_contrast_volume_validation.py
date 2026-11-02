"""FlowVolumeAccumulator: streaming per-subject volume metrics from validation slices."""

from __future__ import annotations

import pytest
import torch

from spectramr.config.schemas.data import DataConfigSchema
from spectramr.data.datasets.flow_synthetic_dataset import create_flow_synthetic_datasets
from spectramr.infrastructure.training.strategies.phase_contrast_volume_validation import (
    FlowVolumeAccumulator,
)

GRID = (12, 16, 14)


def _val_slices():
    cfg = DataConfigSchema(
        dataset_type="flow_synthetic",
        phase_contrast={"enabled": True, "venc": 1.5},
        flow_synthetic={
            "enabled": True,
            "train_subjects": 1,
            "val_subjects": 2,
            "grid": GRID,
            "voxel_size_mm": (1.0, 1.0, 1.0),
            "frames": 3,
            "coils": 2,
            "noise_std": 0.0,
            "center_fraction": 0.2,
            "acceleration": 2.0,
        },
    )
    return create_flow_synthetic_datasets(cfg, None, None)[1]


def _stack(chunk: list[dict], key: str) -> torch.Tensor:
    return torch.stack([i[key] for i in chunk])


def _feed(acc: FlowVolumeAccumulator, items: list[dict], velocity_offset: float = 0.0) -> None:
    for start in range(0, len(items), 4):
        chunk = items[start : start + 4]

        def stack(key: str, chunk: list[dict] = chunk) -> torch.Tensor:
            return _stack(chunk, key)

        acc.add(
            [i["subject_id"] for i in chunk],
            torch.tensor([i["slice_index"] for i in chunk]),
            torch.tensor([i["n_slices"] for i in chunk]),
            stack("voxel_size_mm"),
            {
                "velocity": stack("velocity") + velocity_offset,
                "velocity_true": stack("velocity"),
                "vessel_mask": stack("vessel_mask"),
                "inlet_mask": stack("inlet_mask"),
                "outlet_mask": stack("outlet_mask"),
            },
        )


def test_complete_subjects_are_scored_and_freed() -> None:
    val = _val_slices()
    acc = FlowVolumeAccumulator()
    _feed(acc, [val[i] for i in range(len(val))])
    out = acc.finalize()
    assert out["val_velocity_rmse_roi_volume"] == pytest.approx(0.0, abs=1e-6)
    assert out["val_mass_conservation"] < 10.0
    assert out["val_divergence_normalised"] < 0.2
    assert acc.finalize() == {}  # finalize resets


def test_a_velocity_offset_shows_in_the_volume_rmse() -> None:
    val = _val_slices()
    acc = FlowVolumeAccumulator()
    _feed(acc, [val[i] for i in range(len(val))], velocity_offset=0.1)
    assert acc.finalize()["val_velocity_rmse_roi_volume"] == pytest.approx(0.1, rel=1e-4)


def test_an_incomplete_subject_raises_at_finalize() -> None:
    val = _val_slices()
    acc = FlowVolumeAccumulator()
    _feed(acc, [val[i] for i in range(GRID[0] + 5)])  # subject 1 is cut after 5 slices
    with pytest.raises(RuntimeError, match=r"val-001.*5/12"):
        acc.finalize()


def test_reference_free_slices_are_not_scored() -> None:
    acc = FlowVolumeAccumulator()
    items = [_val_slices()[i] for i in range(GRID[0])]
    for item in items:
        acc.add(
            [item["subject_id"]],
            torch.tensor([item["slice_index"]]),
            torch.tensor([item["n_slices"]]),
            item["voxel_size_mm"][None],
            dict.fromkeys(
                ("velocity", "velocity_true", "vessel_mask", "inlet_mask", "outlet_mask")
            ),
        )
    assert acc.finalize() == {}
