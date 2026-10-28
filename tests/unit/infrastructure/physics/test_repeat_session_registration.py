"""Planted checks for :mod:`spectramr.infrastructure.physics.repeat_session_registration`.

Pairs with ``src/spectramr/infrastructure/physics/repeat_session_registration.py``. The
generator conversion is checked against the closed form; the rigid fit must recover a planted
pose; and the whole chain (two sessions -> rigid -> Demons -> residual -> calibration) must
return the planted warp with the *theory's* sign and in the *contract's* frame. The fixture
plants ``v`` as true-to-observed in the synthetic LPS frame; the artifact is observed-to-true in
RAS about the FOV centre, so the fitted field mapped back through ``LPS_TO_RAS`` must be the
negative of the planted one, and the comparison without that map must come out different. A
corner-origin header, as a Swoop export has it, must give the same answer: that is the shape
the first run on real data failed on (a ball about the header origin caught a sliver of head).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from spectramr.infrastructure.physics import repeat_session_registration as rsr
from spectramr.infrastructure.physics.hardware_warp_calibration import calibrate_hardware_warp
from spectramr.infrastructure.physics.scanner_frame import LPS_TO_RAS
from tests.utils.hardware_warp_synthetic_sessions import SyntheticSessions, rotation_matrix

sitk = pytest.importorskip("SimpleITK")

pytestmark = [pytest.mark.unit]

POSES = [
    (np.zeros(3), np.zeros(3)),
    (np.array([0.0, 0.0, 0.05]), np.array([2.0, -1.0, 0.5])),
    (
        np.array([0.05 * np.sin(np.radians(60)), 0.0, 0.05 * np.cos(np.radians(60))]),
        np.array([-1.0, 1.5, -0.5]),
    ),
]


@pytest.fixture(scope="module")
def sessions() -> tuple[SyntheticSessions, list]:
    synth = SyntheticSessions(n_voxels=36)
    return synth, [synth.sitk_image(synth.session_array(*pose)) for pose in POSES]


def _corner_origin(image):
    """The same volume with the header origin at the FOV corner, as a Swoop export has it."""
    out = sitk.Image(image)
    out.SetOrigin((0.0, 0.0, 0.0))
    return out


def _calibrate(synth: SyntheticSessions, images: list, **kwargs) -> dict:
    pose_set = {"unit_id": "syn", "software_version": "0", "ball_radius_mm": synth.ball_radius_mm}
    pose_set["pairs"], residuals = [], []
    for a, b in [(0, 1), (0, 2)]:
        reg = rsr.register_session_pair(
            images[a],
            images[b],
            ball_radius_mm=synth.ball_radius_mm,
            demons_iterations=150,
            demons_sigma=1.5,
            **kwargs,
        )
        assert reg.n_foreground >= reg.n_in_ball >= reg.points.shape[0] > 500
        pose_set["center_mm"] = reg.center.tolist()
        pose_set["pairs"].append(
            {
                "subject_id": "syn",
                "sessions": [f"s{a}", f"s{b}"],
                "omega": reg.omega.tolist(),
                "translation": reg.translation.tolist(),
                "residual_path": None,
            }
        )
        residuals.append((reg.points, reg.residual))
    return calibrate_hardware_warp(
        pose_set,
        residuals,
        max_degree=3,
        min_degree=2,
        ball_radius_mm=synth.ball_radius_mm,
        basis_points_per_axis=9,
        unit_id="syn",
        software_version="0",
    )


def _agreement(synth: SyntheticSessions, artifact: dict) -> tuple[float, float, float]:
    """``(cosine, norm ratio, cosine without the frame map)`` of fitted against planted ``v``.

    The planted field lives in the synthetic (LPS, content-centred) frame; the artifact is RAS
    about the FOV centre, which is the content centre, so ``v_lps(u) = P v_ras(P u)``.
    """
    grid = torch.as_tensor(synth.grid, dtype=torch.float64)
    points = grid[grid.norm(dim=1) < 0.8 * synth.ball_radius_mm][::7]
    coeffs = torch.tensor(artifact["coeffs"], dtype=torch.float64)
    p = torch.as_tensor(LPS_TO_RAS)
    fitted = synth.canonical.synthesize(coeffs, points @ p.T) @ p.T
    naive = synth.canonical.synthesize(coeffs, points)
    planted = torch.as_tensor(synth.v(points.numpy()))
    cosine = lambda a, b: float((a * b).sum() / (a.norm() * b.norm()))  # noqa: E731
    return cosine(fitted, planted), float(fitted.norm() / planted.norm()), cosine(naive, planted)


class TestGeometry:
    def test_generator_from_rigid_matches_the_closed_form(self) -> None:
        rotvec = np.array([0.02, -0.03, 0.05])
        c, tau = np.array([5.0, -7.0, 3.0]), np.array([1.5, 0.5, -2.0])
        matrix = rotation_matrix(rotvec)
        tx = sitk.Euler3DTransform()
        tx.SetCenter(c.tolist())
        tx.SetMatrix(matrix.reshape(-1).tolist())
        tx.SetTranslation(tau.tolist())
        omega, translation, got_matrix = rsr.generator_from_rigid(tx)
        assert omega.numpy() == pytest.approx(rotvec, abs=1e-9)
        assert translation.numpy() == pytest.approx(c - matrix @ c + tau, abs=1e-9)
        assert got_matrix.numpy() == pytest.approx(matrix, abs=1e-9)
        x = np.array([10.0, 20.0, -30.0])
        mapped = np.asarray(tx.TransformPoint(x.tolist()))
        assert mapped == pytest.approx(matrix @ x + translation.numpy(), abs=1e-9)
        # About another centre the same map reads y -> M y + t' in y = x - centre.
        center = np.array([90.0, -40.0, 12.0])
        _, t_centred, _ = rsr.generator_from_rigid(tx, center)
        assert mapped - center == pytest.approx(matrix @ (x - center) + t_centred.numpy(), abs=1e-9)

    def test_physical_points_follow_the_header(self) -> None:
        image = sitk.Image(3, 4, 5, sitk.sitkFloat32)
        image.SetSpacing((2.0, 3.0, 4.0))
        image.SetOrigin((-1.0, 5.0, 10.0))
        points = rsr.physical_points(image)
        assert points.shape == (60, 3)
        assert points.tolist() == [
            list(image.TransformIndexToPhysicalPoint((i, j, k)))
            for i in range(3)
            for j in range(4)
            for k in range(5)
        ]

    def test_sample_residual_rotates_the_field_and_respects_mask_ball_and_centre(self) -> None:
        arr = np.tile(np.array([1.0, 0.0, 0.0]), (6, 6, 6, 1))
        field = sitk.GetImageFromArray(arr, isVector=True)
        field.SetSpacing((10.0,) * 3)
        field.SetOrigin((-25.0,) * 3)
        mask = np.zeros((6, 6, 6), dtype=bool)
        mask[1:5, 1:5, 1:5] = True
        matrix = torch.as_tensor(rotation_matrix(np.array([0.0, 0.0, np.pi / 2])))
        points, residual, n_in_ball = rsr.sample_residual(
            field, matrix, mask=mask, ball_radius_mm=30.0
        )
        assert points.shape == residual.shape and 0 < points.shape[0] == n_in_ball
        assert bool((points.norm(dim=1) < 30.0).all())
        assert torch.allclose(
            residual, torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64).expand_as(residual)
        )
        capped, _, n_before = rsr.sample_residual(
            field, matrix, mask=mask, ball_radius_mm=30.0, max_points=7
        )
        assert capped.shape[0] == 7 and n_before == n_in_ball
        # Points come back relative to the centre, and the ball is placed around it.
        center = np.array([15.0, 15.0, 15.0])
        shifted, _, _ = rsr.sample_residual(
            field, matrix, mask=mask, ball_radius_mm=30.0, center=center
        )
        assert bool((shifted.norm(dim=1) < 30.0).all())
        assert bool(((shifted + torch.as_tensor(center)) > -30.0).all())
        with pytest.raises(ValueError, match="inside the ball"):
            rsr.sample_residual(
                field, matrix, mask=mask, ball_radius_mm=30.0, center=np.array([200.0, 0.0, 0.0])
            )


class TestRegistration:
    def test_rigid_fit_recovers_the_planted_pose(self, sessions) -> None:
        synth, images = sessions
        moving = rsr.match_intensities(images[1], images[0])
        omega, translation, _ = rsr.generator_from_rigid(rsr.rigid_transform(images[0], moving))
        omega_true, t_true = synth.true_generator(POSES[0], POSES[1])
        assert omega.numpy() == pytest.approx(omega_true, abs=0.012)
        assert translation.numpy() == pytest.approx(t_true, abs=0.6)

    def test_calibration_recovers_the_planted_warp_with_the_theory_sign_in_ras(
        self, sessions
    ) -> None:
        synth, images = sessions
        artifact = _calibrate(synth, images)
        assert artifact["center_mm"] == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)
        cosine, ratio, naive = _agreement(synth, artifact)
        assert cosine < -0.5, f"fitted v must be the negative of the planted field; {cosine:.3f}"
        assert 0.4 < ratio < 2.5
        assert abs(naive - cosine) > 0.05, "the LPS->RAS map must be load-bearing for this field"

    def test_corner_origin_header_is_centred_on_the_fov(self, sessions) -> None:
        synth, images = sessions
        corner = [_corner_origin(image) for image in images]
        half = (synth.n - 1) / 2 * synth.spacing
        artifact = _calibrate(synth, corner)
        assert artifact["center_mm"] == pytest.approx([-half, -half, half], abs=1e-6)
        cosine, ratio, _ = _agreement(synth, artifact)
        assert cosine < -0.5 and 0.4 < ratio < 2.5


class TestGuards:
    def test_missing_simpleitk_is_a_named_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(rsr, "HAVE_SITK", False)
        with pytest.raises(ImportError, match="SimpleITK"):
            rsr.require_sitk()
        with pytest.raises(ImportError, match="SimpleITK"):
            rsr.foreground_mask(None)

    def test_unknown_metric_raises(self, sessions) -> None:
        _, images = sessions
        with pytest.raises(ValueError, match="metric"):
            rsr.rigid_transform(images[0], images[1], metric="ncc")

    def test_sessions_in_different_frames_are_refused(self, sessions) -> None:
        _, images = sessions
        moved = sitk.Image(images[1])
        moved.SetOrigin(
            tuple(o + d for o, d in zip(images[1].GetOrigin(), (5.0, 0.0, 0.0), strict=True))
        )
        with pytest.raises(ValueError, match="do not share a frame"):
            rsr.register_session_pair(images[0], moved, ball_radius_mm=100.0)

    def test_empty_ball_raises(self) -> None:
        field = sitk.GetImageFromArray(np.zeros((4, 4, 4, 3)), isVector=True)
        field.SetOrigin((500.0,) * 3)
        with pytest.raises(ValueError, match="inside the ball"):
            rsr.sample_residual(
                field,
                torch.eye(3, dtype=torch.float64),
                mask=np.ones((4, 4, 4), dtype=bool),
                ball_radius_mm=10.0,
            )
