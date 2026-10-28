"""Planted checks for :mod:`spectramr.infrastructure.physics.scanner_frame`.

Pairs with ``src/spectramr/infrastructure/physics/scanner_frame.py``. The seam this module
exists for is exercised through a real NIfTI header: the voxel SimpleITK places at ``p`` is the
voxel nibabel places at ``LPS_TO_RAS @ p``, which is what makes the registration's outputs and
the unwarp transform's evaluation points agree.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from spectramr.infrastructure.physics import scanner_frame as sf
from tests.utils.hardware_warp_synthetic_sessions import rotation_matrix

sitk = pytest.importorskip("SimpleITK")

pytestmark = [pytest.mark.unit]


def _image(origin=(10.0, 20.0, 30.0), spacing=(1.5, 2.0, 5.0), direction=None):
    image = sitk.Image(4, 5, 6, sitk.sitkFloat32)
    image.SetOrigin(origin)
    image.SetSpacing(spacing)
    if direction is not None:
        image.SetDirection(np.asarray(direction, dtype=np.float64).reshape(-1).tolist())
    return image


def test_fov_center_is_the_physical_point_of_the_half_index() -> None:
    image = _image(direction=rotation_matrix(np.array([0.0, 0.0, np.pi / 2])))
    expected = image.TransformContinuousIndexToPhysicalPoint((1.5, 2.0, 2.5))
    assert sf.fov_center_lps(image) == pytest.approx(np.asarray(expected), abs=1e-9)


def test_lps_to_ras_maps_simpleitk_positions_onto_the_nibabel_affine(tmp_path: Path) -> None:
    image = sitk.GetImageFromArray(np.random.default_rng(0).random((6, 5, 4)).astype(np.float32))
    image.SetOrigin((10.0, 20.0, 30.0))
    image.SetSpacing((1.5, 2.0, 5.0))
    path = tmp_path / "t.nii.gz"
    sitk.WriteImage(image, str(path))
    affine = nib.load(path).affine
    for index in ((0, 0, 0), (2, 3, 1), (3, 4, 5)):
        lps = np.asarray(sitk.ReadImage(str(path)).TransformIndexToPhysicalPoint(index))
        ras = affine[:3, :3] @ np.asarray(index, dtype=np.float64) + affine[:3, 3]
        assert sf.LPS_TO_RAS @ lps == pytest.approx(ras, abs=1e-6)
        assert not np.allclose(lps, ras), "the two frames differ; a test that passes both is empty"


def test_vectors_and_matrices_transform_as_a_proper_rotation() -> None:
    rotvec = np.array([0.02, -0.03, 0.05])
    p = torch.as_tensor(sf.LPS_TO_RAS)
    m_lps = torch.as_tensor(rotation_matrix(rotvec))
    m_ras = sf.matrix_to_ras(m_lps)
    assert torch.allclose(m_ras, torch.as_tensor(rotation_matrix(sf.LPS_TO_RAS @ rotvec)))
    v = torch.tensor([[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]], dtype=torch.float64)
    assert torch.allclose(sf.vectors_to_ras(v), v @ p.T)
    assert torch.allclose(sf.vectors_to_ras(sf.vectors_to_ras(v)), v), "P is an involution"


class TestSharedFrame:
    def test_identical_headers_share_the_fov_centre(self) -> None:
        a, b = _image(), _image()
        assert sf.require_shared_frame(a, b, tolerance_mm=1.0) == pytest.approx(
            sf.fov_center_lps(a)
        )

    def test_a_moved_fov_is_refused_unless_within_tolerance(self) -> None:
        a, b = _image(), _image(origin=(10.0, 25.0, 30.0))
        with pytest.raises(ValueError, match="do not share a frame"):
            sf.require_shared_frame(a, b, tolerance_mm=1.0)
        assert sf.require_shared_frame(a, b, tolerance_mm=6.0) == pytest.approx(
            (sf.fov_center_lps(a) + sf.fov_center_lps(b)) / 2
        )

    def test_different_direction_cosines_are_refused(self) -> None:
        a = _image()
        b = _image(direction=rotation_matrix(np.array([0.0, 0.0, 1e-3])))
        with pytest.raises(ValueError, match="do not share a frame"):
            sf.require_shared_frame(a, b, tolerance_mm=100.0)
