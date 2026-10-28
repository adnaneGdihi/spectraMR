"""Tests for per-sample fiducial realisations (VFU-02)."""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.marker_realisation import (
    marker_batch,
    realisation_seed,
    realisation_seeds,
    wrong_seed,
)
from spectramr.infrastructure.physics.multi_acquisition import MultiAcquisitionSimulator
from spectramr.infrastructure.physics.virtual_fiducial import VirtualFiducial

_GEOM = {"im_size": (32, 32), "spacing_px": (16.0, 16.0), "sigma_px": (2.0, 2.0), "jitter": 0.35}


@pytest.mark.parametrize("seed", [0, 7, 123456])
def test_batch_matches_the_virtual_fiducial_for_the_same_seed(seed: int) -> None:
    ref = VirtualFiducial(im_size=(32, 32), grid_spacing=16, sigma=2.0, jitter=0.35, seed=seed)
    got = marker_batch([seed], **_GEOM)
    assert torch.allclose(got, ref(1).real, atol=1e-6)


def test_train_and_validation_families_are_disjoint_by_parity() -> None:
    train = {realisation_seed(0, "train", e, o) for e in range(3) for o in range(200)}
    val = {realisation_seed(0, "val", 0, o) for o in range(200)}
    assert all(s % 2 == 0 for s in train) and all(s % 2 == 1 for s in val)
    assert not train & val
    assert len(train) == 600  # distinct per (epoch, ordinal)


def test_validation_seeds_ignore_the_epoch_and_are_deterministic() -> None:
    assert realisation_seeds(3, "val", 0, [0, 1]) == realisation_seeds(3, "val", 9, [0, 1])
    assert realisation_seed(3, "train", 1, 5) == realisation_seed(3, "train", 1, 5)
    assert realisation_seed(3, "train", 1, 5) != realisation_seed(3, "train", 2, 5)


def test_unknown_family_raises() -> None:
    with pytest.raises(ValueError, match="unknown marker family"):
        realisation_seed(0, "test", 0, 0)


def test_wrong_seed_is_a_different_marker_of_the_same_family() -> None:
    s = realisation_seed(0, "val", 0, 4)
    w = wrong_seed(s)
    assert w != s and w % 2 == s % 2
    a, b = marker_batch([s, w], **_GEOM)
    assert not torch.allclose(a, b)


def test_simulator_carries_each_sample_its_own_marker() -> None:
    sim = MultiAcquisitionSimulator(
        "subvoxel_sr", marker_enabled=True, marker_grid_spacing=16, marker_sigma=2.0
    )
    pd = torch.rand(2, 1, 32, 32)
    res = sim(pd, marker_seeds=(2, 4))
    assert res.marker_hr.shape == (2, 1, 32, 32)
    assert not torch.allclose(res.marker_hr[0], res.marker_hr[1])
    assert res.marker_seeds == (2, 4)


def test_fixed_seed_through_the_batch_path_reproduces_the_fixed_marker() -> None:
    sim = MultiAcquisitionSimulator(
        "subvoxel_sr", marker_enabled=True, marker_grid_spacing=16, marker_sigma=2.0
    )
    pd = torch.rand(2, 1, 32, 32)
    torch.manual_seed(0)
    fixed = sim(pd)
    torch.manual_seed(0)
    batch = sim(pd, marker_seeds=(0, 0))
    assert torch.allclose(fixed.marker_stack, batch.marker_stack, atol=1e-5)


def test_seed_count_must_match_the_batch() -> None:
    sim = MultiAcquisitionSimulator("subvoxel_sr", marker_enabled=True)
    with pytest.raises(ValueError, match="marker seeds for a batch of 2"):
        sim(torch.rand(2, 1, 32, 32), marker_seeds=(1,))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="sync-debug mode needs CUDA")
def test_batch_on_cuda_does_not_block_the_host() -> None:
    """Built every training step, so the jitter must reach the device without a sync."""
    marker_batch([1, 2], **_GEOM, device="cuda")
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        got = marker_batch([1, 2], **_GEOM, device="cuda")
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert torch.allclose(got.cpu(), marker_batch([1, 2], **_GEOM), atol=1e-6)
