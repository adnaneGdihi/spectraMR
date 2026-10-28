"""Unit tests for :mod:`spectramr.infrastructure.training.strategies.equivariant_imaging_band`."""

from __future__ import annotations

import pytest
import torch

from spectramr.core.metrics.coil_manifold import as_interleaved
from spectramr.infrastructure.physics.fft_ops import fft2c
from spectramr.infrastructure.physics.multicoil_ei_operator import interleaved_to_complex
from spectramr.infrastructure.training.strategies.equivariant_imaging_band import (
    band_project,
    batch_support,
    scored_anchor,
)

B, H, W = 2, 16, 16


def _support(width: int = 12) -> torch.Tensor:
    s = torch.zeros(B, 1, H, W, dtype=torch.bool)
    s[..., :width] = True
    return s


def test_band_projection_removes_exactly_the_out_of_band_energy():
    x = as_interleaved(torch.randn(B, 1, H, W, dtype=torch.complex64))
    support = _support()
    projected = band_project(x, support)
    k = fft2c(interleaved_to_complex(projected))
    assert torch.count_nonzero(k[~support.expand_as(k)].abs() > 1e-5) == 0
    torch.testing.assert_close(band_project(projected, support), projected, atol=1e-5, rtol=1e-5)


def test_the_anchor_scores_the_structural_zeros_against_zero():
    """y is zero off the support, so energy the network puts there is pure loss."""
    y = torch.zeros(B, 1, H, W, dtype=torch.complex64)
    pred = torch.zeros_like(y)
    pred[..., 12:] = 1.0  # unit energy in every out-of-band column
    off_band = ~_support()
    assert scored_anchor(pred * off_band, y, off_band, per_selected=True) == pytest.approx(1.0)
    assert scored_anchor(pred * off_band, y, off_band, per_selected=False) == pytest.approx(4 / W)


def test_per_selected_normalisation_does_not_move_with_the_mask_size():
    y = torch.zeros(B, 1, H, W, dtype=torch.complex64)
    pred = torch.ones_like(y)
    for width in (4, 8, 16):
        sel = _support(width)
        assert scored_anchor(pred * sel, y, sel, per_selected=True) == pytest.approx(1.0)


def test_batch_support_reads_the_collated_image_and_drops_the_depth_axis():
    mask = torch.ones(B, 1, H, W)
    raw = {"support_mask": {"data": _support().float()[..., None]}}
    assert torch.equal(batch_support(raw, mask), _support())


@pytest.mark.parametrize(
    ("batch", "match"),
    [({}, "complementary_rep_masking"), ({"support_mask": torch.ones(B, 1, H, W + 1)}, "plane")],
)
def test_batch_support_refuses_a_missing_or_misaligned_support(batch, match):
    with pytest.raises(ValueError, match=match):
        batch_support(batch, torch.ones(B, 1, H, W))
