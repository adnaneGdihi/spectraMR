"""Tests for qmap_scores: tissue reference masks and across-subject spread (exp 32)."""

from __future__ import annotations

import torch

from spectramr.infrastructure.training.strategies.qmap_scores import (
    coefficient_of_variation,
    reference_masks,
    white_matter_medians,
)


def _phantom():
    t1, t2, fl = torch.zeros(64, 64), torch.zeros(64, 64), torch.zeros(64, 64)
    t1[:, :32], t2[:, :32], fl[:, :32] = 30.0, 20.0, 20.0  # white matter
    t1[:, 32:], t2[:, 32:], fl[:, 32:] = 8.0, 100.0, 1.5  # CSF
    return t1, t2, fl, torch.ones(64, 64, dtype=torch.bool)


def test_masks_separate_csf_from_white_matter() -> None:
    masks = reference_masks(*_phantom())
    assert bool(masks["csf"][:, 40].all()) and not bool(masks["csf"][:, 10].any())
    assert bool(masks["wm"][:, 10].all())


def test_white_matter_medians_read_the_maps_inside_the_mask() -> None:
    t1, t2, fl, _ = _phantom()
    # Air fills most of a real slice; the brain rule (T1 above its median) relies on it.
    for img in (t1, t2, fl):
        img[:40] = 0.0
    z = torch.stack([t1, t2, fl]).unsqueeze(0)  # (1, 3, H, W)
    t1_map = torch.full((1, 1, 64, 64), 600.0)
    t1_map[..., 32:] = 4000.0  # CSF T1, which must not enter the white-matter median
    maps = {"t1": t1_map, "t2": torch.full((1, 1, 64, 64), 80.0)}
    out = white_matter_medians(maps, z)
    assert out["t1"].tolist() == [600.0] and out["t2"].tolist() == [80.0]


def test_a_constant_slice_gives_nan_medians_instead_of_raising() -> None:
    """No voxel lies above the median of a constant slice, so its brain mask is empty."""
    z = torch.full((1, 3, 16, 16), 5.0)
    maps = {"t1": torch.ones(1, 1, 16, 16), "t2": torch.ones(1, 1, 16, 16)}
    out = white_matter_medians(maps, z)
    assert torch.isnan(out["t1"]).all() and torch.isnan(out["t2"]).all()


def test_cv_is_zero_for_identical_subjects_and_positive_otherwise() -> None:
    assert float(coefficient_of_variation(torch.tensor([5.0, 5.0, 5.0]))) == 0.0
    assert float(coefficient_of_variation(torch.tensor([4.0, 6.0]))) > 0.0
