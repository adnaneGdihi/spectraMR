"""Tests for the k-t variable-density sampler."""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.kt_sampling import (
    KT_SAMPLING_PATTERNS,
    kt_sampling_pattern,
    kt_variable_density_mask,
)
from spectramr.infrastructure.training.strategies.robust_ssdu import split_acquired_mask


def _gen(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def test_every_frame_gets_exactly_the_budget_and_the_centre() -> None:
    mask = kt_variable_density_mask((4, 20, 40, 24), 8.0, generator=_gen())
    assert mask.shape == (4, 20, 40, 24) and mask.dtype == torch.bool
    per_frame = mask.flatten(start_dim=2).sum(dim=-1)
    assert (per_frame == round(40 * 24 / 8.0)).all()
    assert mask[..., 19:22, 11:13].all()


def test_frames_and_encodings_draw_different_patterns() -> None:
    mask = kt_variable_density_mask((4, 6, 32, 32), 6.0, generator=_gen())
    flat = mask.flatten(start_dim=2).float()
    assert not torch.equal(flat[0, 0], flat[0, 1])
    assert not torch.equal(flat[0, 0], flat[1, 0])


def test_density_falls_off_from_the_centre() -> None:
    mask = kt_variable_density_mask((200, 32, 32), 5.0, generator=_gen()).float().mean(dim=0)
    inner, outer = (
        mask[12:20, 12:20].mean(),
        torch.cat([mask[:4].flatten(), mask[-4:].flatten()]).mean(),
    )
    assert inner > 2 * outer


def test_line_masks_on_a_one_dimensional_grid() -> None:
    mask = kt_variable_density_mask((2, 10, 96), 4.0, grid_ndim=1, generator=_gen())
    assert (mask.sum(dim=-1) == 24).all()


def test_a_seeded_generator_reproduces_the_mask() -> None:
    a = kt_variable_density_mask((2, 3, 16, 16), 4.0, generator=_gen(7))
    b = kt_variable_density_mask((2, 3, 16, 16), 4.0, generator=_gen(7))
    assert torch.equal(a, b)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"acceleration": 0.5}, "acceleration"),
        ({"acceleration": 40.0, "center_fraction": 0.3}, "centre box"),
        ({"acceleration": 4.0, "grid_ndim": 3}, "grid_ndim"),
        ({"acceleration": 4.0, "sigma": 0.0}, "sigma"),
    ],
)
def test_invalid_requests_raise(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        kt_variable_density_mask((2, 2, 16, 16), **kwargs)


def test_pattern_names_resolve_and_stay_disjoint_from_the_2d_registry() -> None:
    from spectramr.infrastructure.physics.sampling_registry import SamplingPatternRegistry

    assert kt_sampling_pattern("kt_vd_gaussian") is kt_variable_density_mask
    assert not set(KT_SAMPLING_PATTERNS) & set(SamplingPatternRegistry.list_accepted())
    with pytest.raises(ValueError, match="unknown k-t sampling pattern"):
        kt_sampling_pattern("vista")


def test_the_ssdu_split_is_exact_per_frame_on_flattened_frames() -> None:
    """split_acquired_mask splits per leading row, so [E*T, H, W] splits each frame exactly."""
    mask = kt_variable_density_mask((4, 5, 24, 16), 4.0, generator=_gen())
    lam, theta = split_acquired_mask(mask.reshape(-1, 24, 16).float(), 0.4, _gen(1))
    lam, theta = lam.reshape(mask.shape), theta.reshape(mask.shape)
    assert torch.equal((lam + theta).bool(), mask)
    assert (lam * theta).sum() == 0
    assert (theta.flatten(start_dim=2).sum(dim=-1) == round(0.4 * 96)).all()
