"""The shared image-domain patch mask (:mod:`patch_masking`)."""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.training.strategies.patch_masking import create_patch_mask


def test_seeded_stream_reproduces_and_samples_differ() -> None:
    a = create_patch_mask(
        3, 64, 64, 16, 0.75, torch.device("cpu"), torch.Generator().manual_seed(5)
    )
    b = create_patch_mask(
        3, 64, 64, 16, 0.75, torch.device("cpu"), torch.Generator().manual_seed(5)
    )
    assert torch.equal(a, b)
    assert not torch.equal(a[0], a[1])
    assert a.shape == (3, 1, 64, 64) and a.dtype == torch.bool
    assert int(a[0].sum()) == int(16 * 0.75) * 16 * 16


def test_non_divisible_extent_keeps_the_border_strip() -> None:
    mask = create_patch_mask(2, 40, 36, 16, 1.0 - 1e-9, torch.device("cpu"))
    assert mask.shape == (2, 1, 40, 36)
    assert not mask[..., 32:, :].any() and not mask[..., :, 32:].any()


@pytest.mark.parametrize("patch", [0, 65])
def test_patch_outside_the_image_raises(patch: int) -> None:
    with pytest.raises(ValueError, match="patch_size"):
        create_patch_mask(1, 64, 64, patch, 0.5, torch.device("cpu"))
