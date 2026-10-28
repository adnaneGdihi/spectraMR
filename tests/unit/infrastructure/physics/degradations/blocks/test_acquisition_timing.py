"""Line-time formulas per acquisition mode and the shot partition."""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.degradations.blocks.acquisition_timing import (
    line_times,
    shot_index,
)
from spectramr.infrastructure.physics.degradations.geometry import (
    AcquisitionGeometry,
    AcquisitionMode,
    LineOrder,
    SliceOrder,
)

TR = 0.5


def _geom(mode, **kw):
    return AcquisitionGeometry(mode=mode, voxel_mm=(1.0, 1.0, 1.0), **kw)


def test_slice_2d():
    t = line_times(_geom(AcquisitionMode.SLICE_2D), 1, 4, tr_s=TR)
    assert torch.equal(t, torch.tensor([[0.0, 0.5, 1.0, 1.5]]))


def test_slice_2d_requires_single_slice():
    with pytest.raises(ValueError):
        line_times(_geom(AcquisitionMode.SLICE_2D), 2, 4, tr_s=TR)


def test_multislice_sequential():
    g = _geom(AcquisitionMode.MULTISLICE_2D, slice_order=SliceOrder.SEQUENTIAL)
    t = line_times(g, 3, 2, tr_s=TR)
    assert torch.equal(t, torch.tensor([[0.0, 0.5], [1.0, 1.5], [2.0, 2.5]]))


def test_multislice_interleaved_rank_order():
    g = _geom(AcquisitionMode.MULTISLICE_2D, slice_order=SliceOrder.INTERLEAVED)
    t = line_times(g, 5, 2, tr_s=1.0)
    # order 0,2,4,1,3 -> rank(d) = [0, 3, 1, 4, 2]; t = ky + rank/5
    rank = torch.tensor([0, 3, 1, 4, 2])[:, None] / 5
    expected = torch.arange(2)[None, :] + rank
    assert torch.allclose(t, expected.float(), atol=1e-6)
    assert (t[:, 0] < 1.0).all()  # every slice excited within the first TR


@pytest.mark.parametrize(
    ("order", "expected"),
    [
        (LineOrder.LINEAR, [[0, 1, 2], [3, 4, 5]]),
        (LineOrder.KY_INNER, [[0, 1, 2], [3, 4, 5]]),
        (LineOrder.KZ_INNER, [[0, 2, 4], [1, 3, 5]]),
    ],
)
def test_volume_orders(order, expected):
    t = line_times(_geom(AcquisitionMode.VOLUME_3D, line_order=order), 2, 3, tr_s=1.0)
    assert torch.equal(t, torch.tensor(expected, dtype=torch.float32))


def test_volume_centric_orders_by_radius_with_lexicographic_ties():
    t = line_times(_geom(AcquisitionMode.VOLUME_3D, line_order=LineOrder.CENTRIC), 3, 3, tr_s=1.0)
    # DC at (1, 1); the four edge-midpoints tie on radius, then corners.
    # ranks: 0 centre; 1..4 = (0,1),(1,0),(1,2),(2,1) lexicographic; 5..8 corners
    expected = torch.tensor([[5, 1, 6], [2, 0, 3], [7, 4, 8]], dtype=torch.float32)
    assert torch.equal(t, expected)


@pytest.mark.parametrize(
    "mode,kw,d,h",
    [
        (AcquisitionMode.SLICE_2D, {}, 1, 7),
        (AcquisitionMode.MULTISLICE_2D, {"slice_order": SliceOrder.INTERLEAVED}, 6, 5),
        (AcquisitionMode.MULTISLICE_2D, {"slice_order": SliceOrder.SEQUENTIAL}, 6, 5),
        (AcquisitionMode.VOLUME_3D, {"line_order": LineOrder.LINEAR}, 4, 6),
        (AcquisitionMode.VOLUME_3D, {"line_order": LineOrder.KZ_INNER}, 4, 6),
        (AcquisitionMode.VOLUME_3D, {"line_order": LineOrder.CENTRIC}, 4, 6),
    ],
)
def test_times_are_distinct_and_cover_each_line_once(mode, kw, d, h):
    t = line_times(_geom(mode, **kw), d, h, tr_s=1.0)
    assert t.shape == (d, h)
    assert torch.unique(t).numel() == d * h
    assert t.min() == 0


@pytest.mark.parametrize("tr", [0.0, -1.0])
def test_nonpositive_tr_raises(tr):
    with pytest.raises(ValueError):
        line_times(_geom(AcquisitionMode.SLICE_2D), 1, 4, tr_s=tr)


def test_shot_index_equal_counts_in_time_order():
    t = line_times(_geom(AcquisitionMode.VOLUME_3D, line_order=LineOrder.KZ_INNER), 4, 6, tr_s=1.0)
    s = shot_index(t, 4)
    assert s.shape == t.shape and s.dtype == torch.long
    assert torch.bincount(s.flatten()).tolist() == [6, 6, 6, 6]
    flat_s = s.flatten()[torch.argsort(t.flatten())]
    assert (flat_s[1:] >= flat_s[:-1]).all()


def test_shot_index_one_shot_and_every_line():
    t = torch.tensor([[2.0, 0.0], [3.0, 1.0]])
    assert (shot_index(t, 1) == 0).all()
    assert torch.equal(shot_index(t, 4), torch.tensor([[2, 0], [3, 1]]))


@pytest.mark.parametrize("n", [0, 5])
def test_shot_index_bad_count_raises(n):
    with pytest.raises(ValueError):
        shot_index(torch.zeros(2, 2), n)


def test_volume_centric_non_square_exact_ties():
    t = line_times(_geom(AcquisitionMode.VOLUME_3D, line_order=LineOrder.CENTRIC), 4, 6, tr_s=1.0)
    # Key 36*dz^2 + 16*dy^2 with DC at (2, 3); includes the cross-axis tie (0,3) vs (2,0).
    expected = torch.tensor(
        [
            [23, 21, 17, 15, 18, 22],
            [19, 11, 5, 3, 6, 12],
            [16, 9, 1, 0, 2, 10],
            [20, 13, 7, 4, 8, 14],
        ],
        dtype=torch.float32,
    )
    assert torch.equal(t, expected)
