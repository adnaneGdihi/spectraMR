"""The queue-free loader serves a self-supervised M4Raw batch the queue would (#3061)."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from spectramr.data.builders.torchio_queue_builder import TorchIOQueueBuilder, TorchIOQueueConfig
from spectramr.data.datasets.m4raw_dataset import M4RawRepetitionDataset
from spectramr.data.transforms.complementary_rep_masking import ComplementaryRepMasking
from tests.fixtures import m4raw_slice_volumes as vols


@pytest.fixture
def dataset(tmp_path):
    files = []
    for g, patient in enumerate(("2022061001", "2022061002")):
        files += vols.write_group(tmp_path / "data", patient, seed=g + 1)
    vols.write_maps(tmp_path / "maps", files)
    mask = ComplementaryRepMasking(
        acceleration=4.0, center_fraction=0.08, mode="full_target", pattern="poisson_disk"
    )
    return M4RawRepetitionDataset(
        files,
        transform=mask,
        single_contrast=True,
        coil_processing_mode="none",
        coil_maps_root=tmp_path / "maps",
        coil_maps_missing="raise",
        slice_level_records=True,
        target_mode="phase_aligned_mean",
        nex_target_exclude_input=True,
        emit_alternate_input=True,
    )


def _tensors(batch) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    return {k: (tuple(v.shape), v.dtype) for k, v in batch.items() if hasattr(v, "shape")}


@pytest.mark.parametrize("workers", [0, 2])
def test_plain_loader_batch_matches_the_queue_batch(dataset, workers):
    config = TorchIOQueueConfig(
        queue_length=4,
        samples_per_volume=1,
        patch_size=(vols.HEIGHT, vols.WIDTH, 1),
        batch_size=2,
        dataset_type="m4raw",
    )
    _, queue_loader = TorchIOQueueBuilder.build_train_queue(dataset, config)
    plain = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=workers,
        persistent_workers=workers > 0,
        collate_fn=queue_loader.collate_fn,
    )
    queued, served = _tensors(next(iter(queue_loader))), _tensors(next(iter(plain)))
    assert set(queued) - set(served) == {"location"}
    assert {k: served[k] for k in queued if k != "location"} == {
        k: v for k, v in queued.items() if k != "location"
    }
    assert {"input", "target", "input_alt", "loss_mask", "measured_kspace"} <= set(served)
