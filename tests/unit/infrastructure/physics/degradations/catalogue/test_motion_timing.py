"""The shared 5-D path of the catalogue motion axes."""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.degradations import AcquisitionGeometry, AcquisitionMode
from spectramr.infrastructure.physics.degradations.catalogue.motion_timing import (
    check_input,
    encoded_axes,
    run_as_volume,
    slice_geometry,
)
from spectramr.infrastructure.physics.degradations.volume.motion_schedules import (
    ENCODING,
    ENCODING_DIMS,
)

M = AcquisitionMode


def test_every_mode_has_an_encoding():
    assert set(ENCODING) == set(ENCODING_DIMS) == set(M)
    assert ENCODING_DIMS[M.SLICE_2D] == ENCODING_DIMS[M.MULTISLICE_2D] == (-2, -1)
    assert ENCODING_DIMS[M.VOLUME_3D] == (-3, -2, -1)


def test_only_3d_encodes_the_slice_axis():
    assert "slice" in encoded_axes(M.VOLUME_3D)
    assert "slice" not in encoded_axes(M.MULTISLICE_2D) | encoded_axes(M.SLICE_2D)


def test_run_as_volume_passes_a_one_slice_slice_geometry():
    seen = {}

    def impl(x, theta, seed, geometry, **kw):
        seen.update(shape=tuple(x.shape), mode=geometry.mode, kw=kw)
        return x * 2

    x = torch.ones(2, 3, 8, 5)
    out = run_as_volume(impl, x, 0.5, 1, a=1)
    assert torch.equal(out, x * 2)
    assert seen == {"shape": (2, 3, 1, 8, 5), "mode": M.SLICE_2D, "kw": {"a": 1}}


def test_run_as_volume_rejects_a_volume():
    with pytest.raises(ValueError, match="B, C, H, W"):
        run_as_volume(lambda *a, **k: a[0], torch.ones(1, 1, 2, 4, 4), 0.5, 0)


def test_check_input_rejects_bad_shapes():
    with pytest.raises(ValueError, match="B, C, D, H, W"):
        check_input(torch.ones(1, 1, 4, 4), slice_geometry(), "op")
    with pytest.raises(ValueError, match="one slice"):
        check_input(torch.ones(1, 1, 2, 4, 4), slice_geometry(), "op")
    check_input(torch.ones(1, 1, 2, 4, 4), AcquisitionGeometry(M.VOLUME_3D, None), "op")
