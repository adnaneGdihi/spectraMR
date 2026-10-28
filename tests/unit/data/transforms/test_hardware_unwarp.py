"""Unit tests for :mod:`spectramr.data.transforms.hardware_unwarp`.

The image is a linear ramp in world coordinates, so trilinear resampling is exact and an
axis-order mistake in the ``grid_sample`` grid would show up as a wrong slope; an identity
artifact must return the input unchanged; a planted degree-2 warp of about 1.5 mm must read
the ramp at the observed position ``x`` solving ``x + v(x) = r`` (the artifact's ``v`` is
observed-to-true, so the exact inverse is the pull-back), with the Jacobian factor
``1 / det(I + grad v(x))``. The convention test plants the warp on an observed image and
asserts that the transform gives the anatomy back -- the direction that the registration
module's end-to-end test fixes on the estimator side.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torchio as tio

from spectramr.data.hardware_warp_artifacts import identity_artifact, load_artifact, write_artifact
from spectramr.data.transforms.hardware_unwarp import HardwareUnwarp
from spectramr.data.transforms.registry import TRANSFORM_REGISTRY, build_transform
from spectramr.infrastructure.physics.hardware_warp_commutator import (
    HarmonicGradientBasis,
    ball_points,
)

pytestmark = [pytest.mark.unit]

KNOBS = {
    "min_degree": 2,
    "max_degree": 3,
    "ball_radius_mm": 100.0,
    "basis_points_per_axis": 9,
    "unit_id": "swoop-01",
    "software_version": "1.2.3",
}
KNOBS_NO_LABELS = {k: v for k, v in KNOBS.items() if k not in ("unit_id", "software_version")}
SHAPE = (24, 16, 12)
AFFINE = torch.tensor(
    [[1.0, 0.0, 0.0, -10.0], [0.0, 2.0, 0.0, 5.0], [0.0, 0.0, 3.0, 7.0], [0.0, 0.0, 0.0, 1.0]],
    dtype=torch.float64,
)
SLOPE = torch.tensor([0.05, -0.03, 0.02], dtype=torch.float64)
INTERCEPT = 1.0
R0_VOXEL = (12, 8, 6)


def _world(voxel: tuple[int, int, int]) -> torch.Tensor:
    return AFFINE[:3, :3] @ torch.tensor(voxel, dtype=torch.float64) + AFFINE[:3, 3]


def _ramp() -> torch.Tensor:
    idx = torch.stack(torch.meshgrid(*[torch.arange(n) for n in SHAPE], indexing="ij"), -1)
    world = idx.to(torch.float64) @ AFFINE[:3, :3].T + AFFINE[:3, 3]
    return (world @ SLOPE + INTERCEPT).to(torch.float32).unsqueeze(0)


def _subject(data: torch.Tensor | None = None) -> tio.Subject:
    return tio.Subject(
        input=tio.ScalarImage(tensor=_ramp() if data is None else data, affine=AFFINE.numpy())
    )


def _planted_artifact(tmp_path: Path) -> tuple[Path, torch.Tensor, HarmonicGradientBasis]:
    """A degree-2 warp scaled to 1.5 mm at the test voxel; returns (path, coeffs, basis)."""
    basis = HarmonicGradientBasis(
        ball_points(KNOBS["basis_points_per_axis"], KNOBS["ball_radius_mm"]), 3, 2
    )
    coeffs = torch.zeros(basis.n_basis, dtype=torch.float64)
    coeffs[basis.block_index(2)] = torch.tensor([0.7, -0.4, 0.2, 0.5, -0.3], dtype=torch.float64)
    v0 = basis.synthesize(coeffs, _world(R0_VOXEL).unsqueeze(0))[0]
    coeffs = coeffs * (1.5 / float(v0.norm()))
    artifact = identity_artifact(**KNOBS)
    artifact["coeffs"] = coeffs.tolist()
    return write_artifact(artifact, tmp_path / "planted.json"), coeffs, basis


def test_identity_artifact_returns_the_input_unchanged(tmp_path: Path) -> None:
    path = write_artifact(identity_artifact(**KNOBS), tmp_path / "identity.json")
    subject = _subject()
    before = subject["input"].data.clone()
    out = HardwareUnwarp(artifact_path=path)(subject)
    assert torch.allclose(out["input"].data, before, atol=1e-6, rtol=0.0)
    assert torch.count_nonzero(out["hardware_warp_displacement"].data) == 0


def test_identity_with_jacobian_returns_the_input_unchanged(tmp_path: Path) -> None:
    path = write_artifact(identity_artifact(**KNOBS), tmp_path / "identity.json")
    subject = _subject()
    before = subject["input"].data.clone()
    out = HardwareUnwarp(artifact_path=path, apply_jacobian=True)(subject)
    assert torch.allclose(out["input"].data, before, atol=1e-6, rtol=0.0)


def _observed(basis: HarmonicGradientBasis, coeffs: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Independent solve of ``x + v(x) = r`` (eight iterations, far past convergence)."""
    x = r.clone()
    for _ in range(8):
        x = r - basis.synthesize(coeffs, x.unsqueeze(0))[0]
    return x


def test_planted_warp_reads_the_observed_position(tmp_path: Path) -> None:
    path, coeffs, basis = _planted_artifact(tmp_path)
    r0 = _world(R0_VOXEL)
    v0 = basis.synthesize(coeffs, r0.unsqueeze(0))[0]
    assert 1.0 <= float(v0.norm()) <= 2.0
    x0 = _observed(basis, coeffs, r0)
    assert float((x0 + basis.synthesize(coeffs, x0.unsqueeze(0))[0] - r0).norm()) < 1e-9
    out = HardwareUnwarp(artifact_path=path)(_subject())
    expected = float(x0 @ SLOPE + INTERCEPT)
    assert out["input"].data[0][R0_VOXEL].item() == pytest.approx(expected, abs=1e-5)
    # Reading at r + v(r) -- the other sign -- is measurably different at this voxel.
    assert abs(expected - float((r0 + v0) @ SLOPE + INTERCEPT)) > 1e-3
    stored = out["hardware_warp_displacement"].data[:, *R0_VOXEL].to(torch.float64)
    assert torch.allclose(stored, x0 - r0, atol=1e-5)


def test_convention_unwarping_an_observed_image_returns_the_anatomy(tmp_path: Path) -> None:
    """The observed voxel at x holds tissue from x + v(x); after the transform, voxel r holds A(r)."""
    path, coeffs, basis = _planted_artifact(tmp_path)
    idx = torch.stack(torch.meshgrid(*[torch.arange(n) for n in SHAPE], indexing="ij"), -1)
    world = idx.to(torch.float64).reshape(-1, 3) @ AFFINE[:3, :3].T + AFFINE[:3, 3]
    observed = ((world + basis.synthesize(coeffs, world)) @ SLOPE + INTERCEPT).reshape(SHAPE)
    out = HardwareUnwarp(artifact_path=path)(_subject(observed.to(torch.float32).unsqueeze(0)))
    anatomy = (world @ SLOPE + INTERCEPT).reshape(SHAPE)
    interior = (slice(2, -2),) * 3
    assert torch.allclose(
        out["input"].data[0][interior].to(torch.float64), anatomy[interior], atol=2e-3
    )


def test_jacobian_divides_by_the_volume_element_ratio(tmp_path: Path) -> None:
    path, coeffs, basis = _planted_artifact(tmp_path)
    r0 = _world(R0_VOXEL)
    x0 = _observed(basis, coeffs, r0)
    eps = 1e-3
    grad = torch.stack(
        [
            (
                basis.synthesize(coeffs, (x0 + eps * e).unsqueeze(0))[0]
                - basis.synthesize(coeffs, (x0 - eps * e).unsqueeze(0))[0]
            )
            / (2 * eps)
            for e in torch.eye(3, dtype=torch.float64)
        ],
        dim=-1,
    )
    det = float(torch.linalg.det(torch.eye(3, dtype=torch.float64) + grad))
    out = HardwareUnwarp(artifact_path=path, apply_jacobian=True)(_subject())
    expected = float(x0 @ SLOPE + INTERCEPT) / det
    assert det != pytest.approx(1.0, abs=1e-3)
    assert out["input"].data[0][R0_VOXEL].item() == pytest.approx(expected, abs=1e-5)


def test_complex_image_is_resampled_per_component(tmp_path: Path) -> None:
    path, coeffs, basis = _planted_artifact(tmp_path)
    ramp = _ramp()
    data = torch.complex(ramp, 2.0 * ramp)
    out = HardwareUnwarp(artifact_path=path)(_subject(data))
    expected = float(_observed(basis, coeffs, _world(R0_VOXEL)) @ SLOPE + INTERCEPT)
    value = out["input"].data[0][R0_VOXEL]
    assert value.real.item() == pytest.approx(expected, abs=1e-5)
    assert value.imag.item() == pytest.approx(2.0 * expected, abs=1e-5)


def test_registered_and_buildable_from_the_registry(tmp_path: Path) -> None:
    path = write_artifact(identity_artifact(**KNOBS), tmp_path / "identity.json")
    entry = TRANSFORM_REGISTRY["hardware_unwarp"]
    assert entry.cls is HardwareUnwarp
    assert entry.produces == ("hardware_warp_displacement",)
    built = build_transform("hardware_unwarp", artifact_path=str(path))
    assert built.unit_id == KNOBS["unit_id"]


class TestRefusals:
    def test_missing_artifact_raises_at_construction(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            HardwareUnwarp(artifact_path=tmp_path / "absent.json")

    def test_malformed_artifact_raises_at_construction(self, tmp_path: Path) -> None:
        artifact = identity_artifact(**KNOBS)
        del artifact["degrees"]
        path = tmp_path / "bad.json"
        path.write_text(__import__("json").dumps(artifact))
        with pytest.raises(ValueError, match="missing keys"):
            HardwareUnwarp(artifact_path=path)

    def test_missing_image_key_raises(self, tmp_path: Path) -> None:
        path = write_artifact(identity_artifact(**KNOBS), tmp_path / "identity.json")
        subject = tio.Subject(other=tio.ScalarImage(tensor=_ramp(), affine=AFFINE.numpy()))
        with pytest.raises(KeyError, match="no 'input' image"):
            HardwareUnwarp(artifact_path=path)(subject)

    def test_two_dimensional_volume_raises(self, tmp_path: Path) -> None:
        path = write_artifact(identity_artifact(**KNOBS), tmp_path / "identity.json")
        subject = _subject(torch.rand(1, 8, 8, 1))
        with pytest.raises(ValueError, match="3-D resampling"):
            HardwareUnwarp(artifact_path=path)(subject)


class TestCentreAndHeaderFrame:
    """The artifact is RAS about ``center_mm``; a NIfTI header read by TorchIO must land there."""

    def test_centre_shift_is_a_translation_of_the_field(self, tmp_path: Path) -> None:
        path, _coeffs, _ = _planted_artifact(tmp_path)
        centre = torch.tensor([7.0, -11.0, 4.0], dtype=torch.float64)
        artifact = load_artifact(path)
        artifact["center_mm"] = centre.tolist()
        centred = write_artifact(artifact, tmp_path / "centred.json")
        shifted_affine = AFFINE.clone()
        shifted_affine[:3, 3] -= centre
        field_centred, _ = HardwareUnwarp(artifact_path=centred).displacement_mm(SHAPE, AFFINE)
        field_origin, _ = HardwareUnwarp(artifact_path=path).displacement_mm(SHAPE, shifted_affine)
        assert torch.allclose(field_centred, field_origin, atol=1e-9)
        assert float(field_centred.abs().max()) > 0.5, "the planted field must not be trivial here"

    @staticmethod
    def _sessions_and_expected(n_voxels: int = 36):
        from tests.utils.hardware_warp_synthetic_sessions import SyntheticSessions

        synth = SyntheticSessions(n_voxels=n_voxels)
        poses = [
            (np.zeros(3), np.zeros(3)),
            (np.array([0.0, 0.0, 0.05]), np.array([2.0, -1.0, 0.5])),
            (np.array([0.043, 0.0, 0.025]), np.array([-1.0, 1.5, -0.5])),
        ]
        images = []
        for pose in poses:
            image = synth.sitk_image(synth.session_array(*pose))
            image.SetOrigin((0.0, 0.0, 0.0))  # a Swoop export puts the origin at the FOV corner
            images.append(image)
        return synth, poses, images

    @staticmethod
    def _errors(synth, poses, nifti: Path, artifact_path: Path) -> tuple[float, float]:
        subject = tio.Subject(input=tio.ScalarImage(nifti))
        observed = subject["input"].data[0].numpy().copy()
        got = HardwareUnwarp(artifact_path=artifact_path)(subject)["input"].data[0].numpy()
        expected = synth.unwarped_array(*poses[1])
        radius = np.linalg.norm(synth.grid, axis=1).reshape(expected.shape)
        interior = (expected > 0.3) & (radius < 60.0)
        before = float(np.abs(observed - expected)[interior].mean())
        after = float(np.abs(got - expected)[interior].mean())
        return before, after

    def test_exact_artifact_in_ras_unwarps_a_corner_origin_nifti(self, tmp_path: Path) -> None:
        """The planted warp, converted to the contract's frame by hand, must undo the session."""
        sitk = pytest.importorskip("SimpleITK")
        from spectramr.infrastructure.physics.scanner_frame import LPS_TO_RAS

        synth, poses, images = self._sessions_and_expected()
        p = torch.as_tensor(LPS_TO_RAS)
        canonical = synth.canonical
        # v_ras(y) = P v_lps(P y), projected onto the canonical basis on its own orthonormal cloud.
        field_ras = (
            synth.canonical.synthesize(synth.coeffs_true_to_observed, canonical.points @ p.T) @ p.T
        )
        coeffs_ras = torch.einsum("ni,jni->j", field_ras, canonical.fields)
        artifact = identity_artifact(**KNOBS)
        artifact["coeffs"] = (-coeffs_ras).tolist()  # observed-to-true is minus true-to-observed
        half = (synth.n - 1) / 2 * synth.spacing
        artifact["center_mm"] = (p @ torch.tensor([half, half, half], dtype=torch.float64)).tolist()
        path = write_artifact(artifact, tmp_path / "exact.json")
        nifti = tmp_path / "ses-02.nii.gz"
        sitk.WriteImage(images[1], str(nifti))
        before, after = self._errors(synth, poses, nifti, path)
        # The floor is trilinear sampling of the observed volume at r + v(r) in the synthetic
        # frame itself: what remains after the exact correction is interpolation, not the warp.
        from scipy.ndimage import map_coordinates

        observed = synth.session_array(*poses[1])
        voxel = ((synth.grid + synth.v(synth.grid) - synth.origin) / synth.spacing).T
        floor = map_coordinates(observed, voxel, order=1, mode="nearest").reshape(observed.shape)
        expected = synth.unwarped_array(*poses[1])
        radius = np.linalg.norm(synth.grid, axis=1).reshape(expected.shape)
        interior = (expected > 0.3) & (radius < 60.0)
        floor_error = float(np.abs(floor - expected)[interior].mean())
        assert after < 1.3 * floor_error < before, (
            f"{before:.4f} -> {after:.4f}, floor {floor_error:.4f}"
        )
        # Both halves of the frame are load-bearing: dropping the parity map or centring on the
        # header origin each leaves the image worse than it was.
        naive = dict(artifact, coeffs=(-synth.coeffs_true_to_observed).tolist())
        _, after_naive = self._errors(
            synth, poses, nifti, write_artifact(naive, tmp_path / "naive.json")
        )
        uncentred = dict(artifact, center_mm=[0.0, 0.0, 0.0])
        _, after_uncentred = self._errors(
            synth, poses, nifti, write_artifact(uncentred, tmp_path / "uncentred.json")
        )
        assert min(after_naive, after_uncentred) > before

    def test_registered_artifact_unwarps_the_session_it_was_calibrated_from(
        self, tmp_path: Path
    ) -> None:
        """Two sessions -> SimpleITK registration -> artifact -> TorchIO unwarp, one chain."""
        sitk = pytest.importorskip("SimpleITK")
        from spectramr.infrastructure.physics import repeat_session_registration as rsr
        from spectramr.infrastructure.physics.hardware_warp_calibration import (
            calibrate_hardware_warp,
        )

        synth, poses, images = self._sessions_and_expected()
        pose_set = {"unit_id": "syn", "software_version": "0", "ball_radius_mm": 100.0, "pairs": []}
        residuals = []
        for a, b in [(0, 1), (0, 2), (1, 2)]:
            reg = rsr.register_session_pair(
                images[a], images[b], ball_radius_mm=100.0, demons_iterations=150, demons_sigma=1.5
            )
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
        artifact = calibrate_hardware_warp(
            pose_set, residuals, unit_id="syn", software_version="0", **KNOBS_NO_LABELS
        )
        path = write_artifact(artifact, tmp_path / "fitted.json")
        negated = dict(artifact, coeffs=[-c for c in artifact["coeffs"]])
        wrong = write_artifact(negated, tmp_path / "negated.json")
        nifti = tmp_path / "ses-02.nii.gz"
        sitk.WriteImage(images[1], str(nifti))
        before, after = self._errors(synth, poses, nifti, path)
        _, anti = self._errors(synth, poses, nifti, wrong)
        # The sign survives the whole chain. Error reduction itself is not asserted: on these
        # 5 mm synthetic voxels the Demons input, not the estimator, limits the fit (the analytic
        # residual is recovered exactly), which the registration suite is the place to watch.
        assert after < anti, f"before {before:.4f}, after {after:.4f}, anti {anti:.4f}"
        # Through the header, the transform must apply the fitted field exactly as a direct
        # evaluation in the synthetic frame does.
        from scipy.ndimage import map_coordinates

        from spectramr.infrastructure.physics.scanner_frame import LPS_TO_RAS

        p = torch.as_tensor(LPS_TO_RAS)
        coeffs = torch.tensor(artifact["coeffs"], dtype=torch.float64)
        basis = HarmonicGradientBasis(ball_points(KNOBS["basis_points_per_axis"], 100.0), 3, 2)
        v = lambda u: (basis.synthesize(coeffs, torch.as_tensor(u) @ p.T) @ p.T).numpy()  # noqa: E731
        observed_at = synth.grid.copy()
        for _ in range(6):
            observed_at = synth.grid - v(observed_at)
        observed = synth.session_array(*poses[1])
        direct = map_coordinates(
            observed, ((observed_at - synth.origin) / synth.spacing).T, order=1, mode="nearest"
        ).reshape(observed.shape)
        got = HardwareUnwarp(artifact_path=path)(tio.Subject(input=tio.ScalarImage(nifti)))
        got = got["input"].data[0].numpy()
        expected = synth.unwarped_array(*poses[1])
        radius = np.linalg.norm(synth.grid, axis=1).reshape(expected.shape)
        interior = (expected > 0.3) & (radius < 60.0)
        assert float(np.abs(got - direct)[interior].max()) < 1e-3
