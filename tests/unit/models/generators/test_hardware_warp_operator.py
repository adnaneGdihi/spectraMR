"""Unit tests for ``hardware_warp_operator``, the calibrated warp registered as a model.

Registration is observed through the populated registry rather than assumed (non-negotiable
16), the harness's optimizer is built over the operator's parameters exactly as the trainer
does, and the field it evaluates is compared with the basis it was fitted in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from spectramr.data.hardware_warp_artifacts import identity_artifact, write_artifact
from spectramr.infrastructure.physics.hardware_warp_calibration import calibrate_hardware_warp
from spectramr.models.generators.hardware_warp_operator import HardwareWarpOperator
from tests.utils.hardware_warp_synthesis import synthesize_planted_pose_set

pytestmark = [pytest.mark.unit]

KNOBS: dict[str, Any] = {
    "max_degree": 3,
    "min_degree": 2,
    "ball_radius_mm": 100.0,
    "basis_points_per_axis": 11,
    "unit_id": "swoop-01",
    "software_version": "1.2.3",
}


@pytest.fixture(scope="module")
def fitted() -> tuple[dict[str, Any], dict[str, Any]]:
    planted = synthesize_planted_pose_set(**KNOBS)
    artifact = calibrate_hardware_warp(planted["pose_set"], planted["residuals"], **KNOBS)
    return planted, artifact


class TestRegistration:
    def test_registered_under_the_calibration_mode(self) -> None:
        from spectramr.models.init_registry import populate_model_registry
        from spectramr.models.registry import get_model_class

        populate_model_registry()
        assert get_model_class("hardware_warp_operator") is HardwareWarpOperator

    def test_declares_the_whole_probe_opt_out(self) -> None:
        assert "forward" in HardwareWarpOperator.synthetic_forward_probe_skip

    def test_accepts_the_factory_kwargs_and_builds_an_optimizer(self) -> None:
        op = HardwareWarpOperator(in_channels=1, out_channels=1, base_channels=32, depth=4)
        assert not op.calibrated
        assert op.get_parameter_count() == 0
        optimizer = torch.optim.AdamW(op.parameters(), lr=1e-4)
        torch.zeros((), requires_grad=True).backward()
        optimizer.step()


class TestField:
    def test_uncalibrated_operator_refuses_to_evaluate(self) -> None:
        with pytest.raises(RuntimeError, match="not calibrated"):
            HardwareWarpOperator()(torch.zeros(4, 3))

    def test_loaded_fit_reproduces_the_basis_field(self, fitted) -> None:
        planted, artifact = fitted
        op = HardwareWarpOperator()
        optimizer = torch.optim.AdamW(op.parameters(), lr=1e-4)
        op.load_artifact_record(artifact)
        assert op.calibrated
        points = planted["canonical"].points[::7]
        expected = planted["canonical"].synthesize(
            torch.tensor(artifact["coeffs"], dtype=torch.float64), points
        )
        out = op(points.to(torch.float32))
        assert out.dtype == torch.float32 and out.shape == points.shape
        assert torch.allclose(out.to(torch.float64), expected, atol=1e-6)
        # The optimizer still holds the parameter the fit was written into.
        assert optimizer.param_groups[0]["params"][0] is op.coeffs
        assert op.coeffs.shape == (len(artifact["coeffs"]),)

    def test_centre_is_subtracted_before_evaluation(self, fitted) -> None:
        planted, artifact = fitted
        shifted = {**artifact, "center_mm": [10.0, -20.0, 5.0]}
        op = HardwareWarpOperator()
        op.load_artifact_record(shifted)
        points = planted["canonical"].points[::9]
        centre = torch.tensor(shifted["center_mm"], dtype=torch.float64)
        expected = planted["canonical"].synthesize(
            torch.tensor(artifact["coeffs"], dtype=torch.float64), points
        )
        assert torch.allclose(op(points + centre), expected, atol=1e-6)

    def test_batched_point_clouds_keep_their_shape(self, fitted) -> None:
        _, artifact = fitted
        op = HardwareWarpOperator()
        op.load_artifact_record(artifact)
        assert op(torch.zeros(2, 5, 3)).shape == (2, 5, 3)

    def test_image_batches_are_refused(self, fitted) -> None:
        _, artifact = fitted
        op = HardwareWarpOperator()
        op.load_artifact_record(artifact)
        with pytest.raises(TypeError, match="hardware_unwarp"):
            op(torch.zeros(1, 1, 8, 8))


class TestArtifactPath:
    def test_constructs_from_an_artifact_on_disk(self, tmp_path: Path) -> None:
        path = write_artifact(identity_artifact(**KNOBS), tmp_path / "identity.json")
        op = HardwareWarpOperator(artifact_path=str(path))
        assert op.calibrated and op.artifact is not None
        assert torch.equal(op(torch.ones(3, 3)), torch.zeros(3, 3))

    def test_missing_artifact_raises_at_construction(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            HardwareWarpOperator(artifact_path=str(tmp_path / "absent.json"))
