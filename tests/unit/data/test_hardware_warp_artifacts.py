"""Unit tests for :mod:`spectramr.data.hardware_warp_artifacts`.

Round trips through ``tmp_path`` for both records, and one planted defect per clause of the two
contracts: a missing key, a wrong type, a pair without a residual, a residual file that is
missing or malformed, a NaN coefficient, degrees out of block order.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from spectramr.data.hardware_warp_artifacts import (
    ARTIFACT_KEYS,
    identity_artifact,
    load_artifact,
    load_pair_residuals,
    load_pose_set,
    validate_artifact,
    write_artifact,
    write_pose_set,
)

pytestmark = [pytest.mark.unit]

IDENTITY_KNOBS = {
    "min_degree": 2,
    "max_degree": 3,
    "ball_radius_mm": 100.0,
    "basis_points_per_axis": 9,
    "unit_id": "swoop-01",
    "software_version": "1.2.3",
}


def _pose_set(residual_path: str | None = "pair_0.pt") -> dict[str, Any]:
    return {
        "unit_id": "swoop-01",
        "software_version": "1.2.3",
        "ball_radius_mm": 100.0,
        "center_mm": [-93.0, -102.5, 87.5],
        "pairs": [
            {
                "subject_id": "sub-01",
                "sessions": ["ses-01", "ses-02"],
                "omega": [0.0, 0.0, 0.05],
                "translation": [1.0, -2.0, 0.5],
                "residual_path": residual_path,
            }
        ],
    }


def _write_pose_set(tmp_path: Path, doc: dict[str, Any]) -> Path:
    path = tmp_path / "pose_set.json"
    path.write_text(json.dumps(doc))
    return path


class TestPoseSet:
    def test_round_trip(self, tmp_path: Path) -> None:
        loaded = load_pose_set(_write_pose_set(tmp_path, _pose_set()))
        assert loaded == _pose_set()

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_pose_set(tmp_path / "absent.json")

    def test_not_json_raises_naming_the_path(self, tmp_path: Path) -> None:
        broken = tmp_path / "broken.json"
        broken.write_text("{not json", encoding="utf-8")
        with pytest.raises(ValueError, match="not readable JSON") as info:
            load_pose_set(broken)
        assert str(broken) in str(info.value)

    @pytest.mark.parametrize(
        "mutate, match",
        [
            (lambda d: d.pop("ball_radius_mm"), "missing keys"),
            (lambda d: d.pop("center_mm"), "missing keys"),
            (lambda d: d.__setitem__("center_mm", [0.0, float("nan"), 0.0]), "non-finite"),
            (lambda d: d.__setitem__("center_mm", [0.0, 0.0]), "list of 3"),
            (lambda d: d.__setitem__("extra", 1), "unknown keys"),
            (lambda d: d.__setitem__("unit_id", ""), "non-empty string"),
            (lambda d: d.__setitem__("pairs", []), "non-empty list"),
            (lambda d: d["pairs"][0].pop("omega"), "missing keys"),
            (lambda d: d["pairs"][0].__setitem__("omega", [0.0, 0.0]), "list of 3"),
            (lambda d: d["pairs"][0].__setitem__("sessions", ["ses-01"]), "exactly two"),
            (lambda d: d["pairs"][0].__setitem__("translation", [0.0, "x", 0.0]), "non-numeric"),
            (lambda d: d["pairs"][0].__setitem__("omega", [float("nan"), 0.0, 0.0]), "non-finite"),
        ],
    )
    def test_contract_violations_raise(self, tmp_path: Path, mutate, match: str) -> None:
        doc = _pose_set()
        mutate(doc)
        with pytest.raises(ValueError, match=match):
            load_pose_set(_write_pose_set(tmp_path, doc))


class TestWritePoseSet:
    def test_round_trips_through_the_loader(self, tmp_path: Path) -> None:
        path = write_pose_set(_pose_set(), tmp_path / "nested" / "pose_set.json")
        assert path.is_file() and load_pose_set(path) == _pose_set()

    def test_refuses_an_invalid_record_and_writes_nothing(self, tmp_path: Path) -> None:
        doc = _pose_set()
        doc["pairs"][0]["omega"] = [0.0, 0.0]
        target = tmp_path / "bad.json"
        with pytest.raises(ValueError, match="list of 3"):
            write_pose_set(doc, target)
        assert not target.exists()


class TestPairResiduals:
    def test_loads_points_and_residual_as_float64(self, tmp_path: Path) -> None:
        points = torch.randn(10, 3)
        residual = torch.randn(10, 3)
        torch.save({"points": points, "residual": residual}, tmp_path / "pair_0.pt")
        [(p, r)] = load_pair_residuals(_pose_set(), tmp_path)
        assert p.dtype == torch.float64 and r.dtype == torch.float64
        assert torch.allclose(p, points.double()) and torch.allclose(r, residual.double())

    def test_null_residual_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="no residual_path"):
            load_pair_residuals(_pose_set(residual_path=None), tmp_path)

    def test_missing_residual_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="residual not found"):
            load_pair_residuals(_pose_set(), tmp_path)

    def test_malformed_payload_raises(self, tmp_path: Path) -> None:
        torch.save({"points": torch.zeros(4, 3)}, tmp_path / "pair_0.pt")
        with pytest.raises(ValueError, match="exactly 'points' and 'residual'"):
            load_pair_residuals(_pose_set(), tmp_path)
        torch.save(
            {"points": torch.zeros(4, 3), "residual": torch.zeros(5, 3)}, tmp_path / "pair_0.pt"
        )
        with pytest.raises(ValueError, match="same shape"):
            load_pair_residuals(_pose_set(), tmp_path)
        bad = torch.zeros(4, 3)
        bad[1, 2] = float("inf")
        torch.save({"points": torch.zeros(4, 3), "residual": bad}, tmp_path / "pair_0.pt")
        with pytest.raises(ValueError, match="non-finite"):
            load_pair_residuals(_pose_set(), tmp_path)


class TestArtifact:
    def test_identity_round_trip(self, tmp_path: Path) -> None:
        artifact = identity_artifact(**IDENTITY_KNOBS)
        assert set(artifact) == ARTIFACT_KEYS
        assert artifact["degrees"] == [2] * 5 + [3] * 7
        path = write_artifact(artifact, tmp_path / "nested" / "artifact.json")
        assert load_artifact(path) == artifact

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_artifact(tmp_path / "absent.json")

    def test_nan_coefficient_is_refused_before_writing(self, tmp_path: Path) -> None:
        artifact = identity_artifact(**IDENTITY_KNOBS)
        artifact["coeffs"][0] = float("nan")
        with pytest.raises(ValueError, match="non-finite"):
            write_artifact(artifact, tmp_path / "artifact.json")
        assert not (tmp_path / "artifact.json").exists()

    @pytest.mark.parametrize(
        "mutate, match",
        [
            (lambda a: a.pop("n_pairs"), "missing keys"),
            (lambda a: a.pop("center_mm"), "missing keys"),
            (lambda a: a.__setitem__("center_mm", [1.0, 2.0]), "list of 3"),
            (lambda a: a.__setitem__("degrees", [3] * 7 + [2] * 5), "canonical block order"),
            (lambda a: a.__setitem__("coeffs", [0.0] * 11), "list of 12"),
            (lambda a: a.__setitem__("covariance", [[0.0] * 12] * 11), "12x12"),
            (lambda a: a.__setitem__("constant_per_degree", {"2": 1.0}), "exactly the degrees"),
            (lambda a: a.__setitem__("constant_per_degree", {"2": -1.0, "3": 1.0}), "negative"),
            (lambda a: a.__setitem__("basis_points_per_axis", 1), ">= 2"),
            (lambda a: a.__setitem__("max_degree", 1), ">= 2"),
        ],
    )
    def test_contract_violations_raise(self, mutate, match: str) -> None:
        artifact = identity_artifact(**IDENTITY_KNOBS)
        mutate(artifact)
        with pytest.raises(ValueError, match=match):
            validate_artifact(artifact)

    def test_covariance_matrix_round_trips(self, tmp_path: Path) -> None:
        artifact = identity_artifact(**IDENTITY_KNOBS)
        artifact["covariance"] = torch.eye(12).tolist()
        path = write_artifact(artifact, tmp_path / "artifact.json")
        assert load_artifact(path)["covariance"] == artifact["covariance"]
