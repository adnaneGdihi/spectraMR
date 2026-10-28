"""Shared pieces of the sequence-timing axes: line order from acquisition timing, and the k-space line filter."""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.degradations import (
    AcquisitionGeometry,
    AcquisitionMode,
    LineOrder,
)
from spectramr.infrastructure.physics.degradations.blocks.acquisition_timing import shot_index
from spectramr.infrastructure.physics.degradations.catalogue.sequence_common import (
    apply_line_weights,
    check_positive,
    ordered_line_times,
)
from spectramr.infrastructure.physics.fft_ops import fftnc

M = AcquisitionMode


def _rank(geometry, d, h, order=None):
    times, inner = ordered_line_times(geometry, d, h, order)
    return shot_index(times, times.numel()), inner


def test_a_slice_and_a_multislice_stack_are_timed_as_one_2d_k_space():
    for mode in (M.SLICE_2D, M.MULTISLICE_2D):
        rank, inner = _rank(AcquisitionGeometry(mode=mode, voxel_mm=None), 5, 8)
        assert torch.equal(rank, torch.arange(8)[None]) and inner == 8


def test_centric_orders_by_distance_from_the_centre_and_kz_inner_has_a_kz_inner_loop():
    rank, _ = _rank(AcquisitionGeometry(mode=M.SLICE_2D, voxel_mm=None), 1, 8, "centric")
    assert [int(rank[0, k]) for k in (4, 3, 5, 2, 6, 1, 7, 0)] == list(range(8))
    g = AcquisitionGeometry(mode=M.VOLUME_3D, voxel_mm=None, line_order=LineOrder.KZ_INNER)
    rank, inner = _rank(g, 3, 4)
    assert inner == 3 and int(rank[1, 2]) == 2 * 3 + 1


def test_unknown_order_and_non_positive_numbers_raise():
    g = AcquisitionGeometry(mode=M.SLICE_2D, voxel_mm=None)
    with pytest.raises(ValueError, match="order"):
        ordered_line_times(g, 1, 8, "radial")
    for bad in (0, -1.0, True, "3"):
        with pytest.raises(ValueError, match="knob"):
            check_positive("knob", bad)
    assert check_positive("knob", 3) == 3.0


@pytest.mark.parametrize(("mode", "dims"), [(M.SLICE_2D, (-2, -1)), (M.VOLUME_3D, (-3, -2, -1))])
def test_line_weights_scale_each_k_space_line(mode, dims):
    x = torch.randn(1, 2, 4, 6, 8, dtype=torch.complex64)
    w = torch.rand(4, 6) + 0.5 if mode is M.VOLUME_3D else torch.rand(1, 6) + 0.5
    out = apply_line_weights(x, w, mode)
    torch.testing.assert_close(
        fftnc(out, dims), fftnc(x, dims) * w.unsqueeze(-1), rtol=1e-4, atol=1e-4
    )
