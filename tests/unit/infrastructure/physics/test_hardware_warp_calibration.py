"""Unit tests for :mod:`spectramr.infrastructure.physics.hardware_warp_calibration`.

A warp is planted in the canonical basis, its commutator residuals are synthesized with
``adjoint_action`` for six random rigid motions on a cloud that is NOT the canonical ball grid,
and the calibration must return the planted coefficients in canonical coordinates together with
every key of the artifact contract. The refusals are planted one shape at a time.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from spectramr.infrastructure.physics.hardware_warp_calibration import (
    calibrate_hardware_warp,
    concatenate_residuals,
    pose_set_generators,
    project_onto_basis,
)
from spectramr.infrastructure.physics.hardware_warp_commutator import HarmonicGradientBasis
from tests.utils.hardware_warp_synthesis import synthesize_planted_pose_set

pytestmark = [pytest.mark.unit, pytest.mark.physics]

RADIUS = 100.0
N_PER_AXIS = 11
KNOBS = {
    "max_degree": 3,
    "min_degree": 2,
    "ball_radius_mm": RADIUS,
    "basis_points_per_axis": N_PER_AXIS,
    "unit_id": "swoop-01",
    "software_version": "1.2.3",
}
ARTIFACT_KEYS = {
    "unit_id",
    "software_version",
    "min_degree",
    "max_degree",
    "ball_radius_mm",
    "center_mm",
    "basis_points_per_axis",
    "coeffs",
    "degrees",
    "covariance",
    "constant_per_degree",
    "n_pairs",
}


@pytest.fixture(scope="module")
def planted() -> dict[str, Any]:
    return synthesize_planted_pose_set(**KNOBS)


class TestRecovery:
    def test_planted_warp_is_recovered_in_canonical_coordinates(self, planted) -> None:
        artifact = calibrate_hardware_warp(planted["pose_set"], planted["residuals"], **KNOBS)
        truth = planted["truth"]
        coeffs = torch.tensor(artifact["coeffs"], dtype=torch.float64)
        assert torch.allclose(coeffs, truth, rtol=0.0, atol=5e-2 * float(truth.abs().max()))
        assert set(artifact) == ARTIFACT_KEYS
        assert artifact["degrees"] == planted["canonical"].degrees.tolist()
        assert artifact["n_pairs"] == 6
        assert artifact["covariance"] is None
        assert set(artifact["constant_per_degree"]) == {"2", "3"}
        assert all(v > 0.0 for v in artifact["constant_per_degree"].values())
        for key in ("unit_id", "software_version", "min_degree", "max_degree"):
            assert artifact[key] == KNOBS[key]
        assert artifact["basis_points_per_axis"] == N_PER_AXIS

    def test_covariance_transforms_with_the_projection(self, planted) -> None:
        artifact = calibrate_hardware_warp(
            planted["pose_set"], planted["residuals"], noise_std=1e-3, **KNOBS
        )
        cov = torch.tensor(artifact["covariance"], dtype=torch.float64)
        n = len(artifact["coeffs"])
        assert cov.shape == (n, n)
        assert torch.allclose(cov, cov.T, atol=1e-12)
        assert torch.all(torch.diagonal(cov) > 0)

    def test_projection_round_trips(self, planted) -> None:
        canonical = planted["canonical"]
        points = planted["residuals"][0][0]
        fitted = HarmonicGradientBasis(points, max_degree=3, min_degree=2)
        forward = project_onto_basis(canonical, fitted)
        back = project_onto_basis(fitted, canonical)
        truth = planted["truth"]
        assert torch.allclose(back @ (forward @ truth), truth, atol=1e-8)

    def test_pure_translation_pair_is_a_valid_generator(self) -> None:
        pose_set = {"pairs": [{"omega": [0.0, 0.0, 0.0], "translation": [1.0, 0.0, 0.0]}]}
        (omega, t), *_ = pose_set_generators(pose_set)
        assert float(omega.norm()) == 0.0 and float(t[0]) == 1.0


class TestRefusals:
    def test_residual_count_mismatch_raises(self, planted) -> None:
        with pytest.raises(ValueError, match="every pair needs exactly one"):
            calibrate_hardware_warp(planted["pose_set"], planted["residuals"][:3], **KNOBS)

    def test_nan_residual_raises(self, planted) -> None:
        points, residual = planted["residuals"][0]
        bad = residual.clone()
        bad[0, 0] = float("nan")
        residuals = [(points, bad), *planted["residuals"][1:]]
        with pytest.raises(ValueError, match="non-finite"):
            calibrate_hardware_warp(planted["pose_set"], residuals, **KNOBS)

    def test_pairs_on_different_clouds_are_fitted_together(self, planted) -> None:
        """Every real pair is another head, so no two clouds coincide; the fit must not care."""
        canonical, truth = planted["canonical"], planted["truth"]
        residuals = []
        for k, (omega, t) in enumerate(pose_set_generators(planted["pose_set"])):
            index = torch.arange(k % 3, canonical.points.shape[0], 3)
            field = torch.einsum("b,bni->ni", truth, canonical.adjoint_action(omega, t, index))
            residuals.append((canonical.points[index], field))
        assert not torch.equal(residuals[0][0], residuals[1][0])
        points, stacked, slices = concatenate_residuals(residuals)
        assert points.shape == stacked.shape and slices[0] == slice(0, residuals[0][0].shape[0])
        assert slices[-1].stop == points.shape[0]
        artifact = calibrate_hardware_warp(planted["pose_set"], residuals, **KNOBS)
        coeffs = torch.tensor(artifact["coeffs"], dtype=torch.float64)
        assert torch.allclose(coeffs, truth, rtol=0.0, atol=5e-2 * float(truth.abs().max()))

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same shape"):
            concatenate_residuals([(torch.zeros(4, 3), torch.zeros(5, 3))])

    def test_unit_or_version_mismatch_raises(self, planted) -> None:
        knobs = dict(KNOBS, unit_id="swoop-02")
        with pytest.raises(ValueError, match="specific to both"):
            calibrate_hardware_warp(planted["pose_set"], planted["residuals"], **knobs)

    def test_radius_mismatch_raises(self, planted) -> None:
        knobs = dict(KNOBS, ball_radius_mm=RADIUS * 1.5)
        with pytest.raises(ValueError, match="ball_radius_mm"):
            calibrate_hardware_warp(planted["pose_set"], planted["residuals"], **knobs)

    def test_too_few_points_raise(self, planted) -> None:
        one_pair = dict(planted["pose_set"], pairs=planted["pose_set"]["pairs"][:1])
        points, residual = planted["residuals"][0]
        with pytest.raises(ValueError, match="cannot determine"):
            calibrate_hardware_warp(one_pair, [(points[:2], residual[:2])], **KNOBS)
