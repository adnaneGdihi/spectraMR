"""Validation scores no slice the image collate fabricated by padding depth.

fastMRI volumes differ in slice count, so a whole-volume validation batch is
zero-padded in depth to its longest volume. Flattened into slices, that padding
reached the reverse sampler as samples with no measured k-space bin.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.data.collation.strategies import ImageCollateStrategy
from spectramr.data.collation.unpad import (
    remove_padding,
    split_padded_batch,
    split_padded_batches,
)
from spectramr.infrastructure.training.validation_rows import expand_depth


def _volume(depth: int, seed: int) -> dict:
    gen = torch.Generator().manual_seed(seed)
    kspace = torch.randn(1, 6, 5, depth, generator=gen) + 0.5
    return {
        "input": kspace,
        "target": kspace.clone(),
        "kspace_scale": torch.tensor(float(seed)),
        "file_id": f"vol{seed}",
    }


def _collated(*depths: int) -> dict:
    return ImageCollateStrategy().collate([_volume(d, i + 1) for i, d in enumerate(depths)])


def _zero_slices(batch: dict) -> int:
    x = batch["input"]
    return int((x.abs().sum(dim=(1, 2, 3)) == 0).sum())


def test_the_collate_fabricates_zero_slices_for_unequal_depths() -> None:
    """The defect's precondition: without the split, 6 of 16 slices are padding."""
    batch = _collated(2, 8)
    assert batch["input"].shape == (2, 1, 6, 5, 8)
    assert _zero_slices(batch) == 6


def test_a_padded_batch_splits_into_samples_at_their_own_depth() -> None:
    parts = split_padded_batch(_collated(2, 8))
    assert [p["input"].shape[-1] for p in parts] == [2, 8]
    assert all(_zero_slices(p) == 0 for p in parts)
    assert [p["file_id"] for p in parts] == [["vol1"], ["vol2"]]
    assert [float(p["kspace_scale"]) for p in parts] == [1.0, 2.0]
    assert all(not any(k.endswith("_padding") for k in p) for p in parts)


def test_the_split_restores_exactly_what_was_collated() -> None:
    short, long_ = _volume(3, 1), _volume(8, 2)
    parts = split_padded_batch(ImageCollateStrategy().collate([short, long_]))
    assert torch.equal(parts[0]["input"][0], short["input"])
    assert torch.equal(parts[1]["target"][0], long_["target"])


def test_an_unpadded_batch_passes_through_whole() -> None:
    batch = _collated(8, 8)
    assert "input_padding" not in batch
    (only,) = split_padded_batch(batch)
    assert only["input"] is batch["input"]


def test_the_slice_route_sees_no_padded_slice() -> None:
    """``validation.per_batch_rows`` cuts depth after the split, never before."""
    slices = list(expand_depth(split_padded_batches([_collated(2, 8)])))
    assert len(slices) == 2 + 8
    assert all(float(s["input"].abs().sum()) > 0 for s in slices)


def test_non_mapping_batches_pass_through() -> None:
    sentinel = object()
    assert list(split_padded_batches([sentinel])) == [sentinel]


def test_a_record_for_another_rank_raises() -> None:
    with pytest.raises(ValueError, match="covers 2 axes"):
        remove_padding(torch.zeros(1, 4, 4), [(0, 0), (1, 1)])


def test_records_disagreeing_about_the_batch_size_raise() -> None:
    batch = _collated(2, 8)
    batch["target_padding"] = batch["target_padding"][:1]
    with pytest.raises(ValueError, match="disagree about the batch size"):
        split_padded_batch(batch)
