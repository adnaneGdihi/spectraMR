"""Unit tests for :mod:`spectramr.data.hardware_warp_pose_set_builder`.

A temporary Leiden-style tree carries three subjects: one with four sessions, one with two, one
with a single session (never paired); the sidecar of the first carries the unit and software
version. Discovery, both pairing schemes, provenance resolution and its refusal are checked
without registration; the build itself runs on two synthetic sessions and must produce files
that the artifact owner's ``load_pose_set`` and ``load_pair_residuals`` accept, and that the
CLI's ``--dry-run`` lists.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from spectramr.data import hardware_warp_pose_set_builder as builder
from spectramr.data.hardware_warp_artifacts import (
    load_artifact,
    load_pair_residuals,
    load_pose_set,
)
from spectramr.infrastructure.training.strategies.hardware_warp_calibration_strategy import (
    HardwareWarpKnobs,
    run_hardware_warp_calibration,
)

sitk = pytest.importorskip("SimpleITK")

pytestmark = [pytest.mark.unit]

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "preprocessing" / "build_hardware_warp_pose_set.py"


def _write_nifti(path: Path, image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(path))


def _tree(tmp_path: Path, synth, poses: dict[str, list[tuple]]) -> Path:
    raw = tmp_path / "raw"
    for k, (subject, subject_poses) in enumerate(poses.items()):
        for s, pose in enumerate(subject_poses, start=1):
            name = f"{subject}_ses-{s:02d}_run-1_T1w.nii.gz"
            _write_nifti(
                raw / "64mT_data" / subject / f"ses-{s:02d}" / "anat" / name,
                synth.sitk_image(synth.session_array(*pose)),
            )
            if k == 0 and s == 1:
                sidecar = (
                    raw
                    / "64mT_data"
                    / subject
                    / f"ses-{s:02d}"
                    / "anat"
                    / name.replace(".nii.gz", ".json")
                )
                sidecar.write_text(
                    json.dumps({"DeviceSerialNumber": "SW-0042", "SoftwareVersions": "8.7.1"})
                )
    return raw


@pytest.fixture(scope="module")
def synth():
    from tests.utils.hardware_warp_synthetic_sessions import SyntheticSessions

    return SyntheticSessions(n_voxels=24)


@pytest.fixture(scope="module")
def raw_tree(tmp_path_factory, synth) -> Path:
    z = np.zeros(3)
    poses = {
        "sub-0001": [
            (z, z),
            (np.array([0, 0, 0.05]), np.array([2.0, -1.0, 0.5])),
            (z, z),
            (np.array([0.04, 0, 0.02]), z),
        ],
        "sub-0002": [(z, z), (np.array([0.03, 0.03, 0.0]), np.array([-1.0, 1.0, 0.0]))],
        "sub-0003": [(z, z)],
    }
    return _tree(tmp_path_factory.mktemp("leiden"), synth, poses)


class TestDiscoveryAndPairing:
    def test_only_subjects_with_repeats_are_kept_in_session_order(self, raw_tree: Path) -> None:
        found = builder.discover_repeat_sessions(raw_tree)
        assert list(found) == ["sub-0001", "sub-0002"]
        assert [s.session for s in found["sub-0001"]] == ["01", "02", "03", "04"]
        assert found["sub-0001"][0].path.name == "sub-0001_ses-01_run-1_T1w.nii.gz"

    def test_missing_root_raises_naming_both_spellings(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="tried '64mT_data', '64mT data'"):
            builder.discover_repeat_sessions(tmp_path)

    def test_archive_spelling_is_found_without_a_flag(self, raw_tree: Path, tmp_path: Path) -> None:
        archive = tmp_path / "Data"
        archive.mkdir()
        (archive / "64mT data").symlink_to(raw_tree / "64mT_data", target_is_directory=True)
        assert builder.resolve_lowfield_root(archive).name == "64mT data"
        assert list(builder.discover_repeat_sessions(archive)) == ["sub-0001", "sub-0002"]

    def test_explicit_name_is_tried_alone(self, raw_tree: Path) -> None:
        with pytest.raises(FileNotFoundError, match=r"tried 'lowfield'$"):
            builder.discover_repeat_sessions(raw_tree, lowfield_dir="lowfield")
        assert builder.resolve_lowfield_root(raw_tree, "64mT_data") == raw_tree / "64mT_data"

    def test_disjoint_and_all_schemes(self, raw_tree: Path) -> None:
        sessions = builder.discover_repeat_sessions(raw_tree)["sub-0001"]
        disjoint = builder.pair_sessions(sessions)
        assert [(p.fixed.session, p.moving.session) for p in disjoint] == [
            ("01", "02"),
            ("03", "04"),
        ]
        every = builder.pair_sessions(sessions, scheme="all")
        assert len(every) == 6 and every[0].stem == "sub-0001_ses-01_ses-02"
        with pytest.raises(ValueError, match="scheme"):
            builder.pair_sessions(sessions, scheme="ring")

    def test_sidecar_provenance(self, raw_tree: Path) -> None:
        sessions = builder.discover_repeat_sessions(raw_tree)
        assert builder.read_sidecar_provenance(sessions["sub-0001"][0].path) == {
            "unit_id": "SW-0042",
            "software_version": "8.7.1",
        }
        assert builder.read_sidecar_provenance(sessions["sub-0002"][0].path) == {
            "unit_id": None,
            "software_version": None,
        }


class TestBuild:
    def test_build_writes_a_pose_set_the_owner_accepts(
        self, raw_tree: Path, tmp_path: Path, synth
    ) -> None:
        # (01, 02) and (02, 03): the second pair's fixed session is a rotated head, so the two
        # pairs are sampled on different clouds, as every real pair is.
        every = builder.pair_sessions(
            builder.discover_repeat_sessions(raw_tree)["sub-0001"], scheme="all"
        )
        pairs = [every[0], every[3]]
        assert pairs[1].fixed.session == "02" and pairs[0].fixed.path != pairs[1].fixed.path
        path, pose_set = builder.build_pose_set(
            pairs,
            tmp_path / "hw",
            ball_radius_mm=synth.ball_radius_mm,
            registration_kwargs={
                "rigid_iterations": 60,
                "demons_iterations": 20,
                "max_points": 500,
            },
        )
        loaded = load_pose_set(path)
        assert (
            loaded == pose_set
            and loaded["unit_id"] == "SW-0042"
            and loaded["software_version"] == "8.7.1"
        )
        assert loaded["pairs"][0]["residual_path"] == "residuals/sub-0001_ses-01_ses-02.pt"
        assert loaded["center_mm"] == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)
        residuals = load_pair_residuals(loaded, path.parent)
        assert len(residuals) == 2
        for points, residual in residuals:
            assert points.shape == residual.shape and 0 < points.shape[0] <= 500
        assert not torch.equal(residuals[0][0], residuals[1][0]), "two heads, two clouds"
        omega = np.array(loaded["pairs"][0]["omega"])
        assert omega == pytest.approx([0.0, 0.0, 0.05], abs=0.02)
        # The production path consumes exactly what the builder wrote: two pairs sampled on
        # different clouds must fit as one least-squares problem and write a valid artifact.
        artifact_path = tmp_path / "hw" / "artifact.json"
        run_hardware_warp_calibration(
            HardwareWarpKnobs(
                max_degree=3,
                min_degree=2,
                ball_radius_mm=synth.ball_radius_mm,
                basis_points_per_axis=9,
                identifiability_floor=1e-9,
                pose_set_path=path,
                artifact_path=artifact_path,
                readout_term="off",
                unit_id="SW-0042",
                software_version="8.7.1",
            )
        )
        artifact = load_artifact(artifact_path)
        assert artifact["n_pairs"] == 2 and artifact["center_mm"] == loaded["center_mm"]

    def test_provenance_override_and_refusal(self, raw_tree: Path, tmp_path: Path, synth) -> None:
        pairs = builder.pair_sessions(builder.discover_repeat_sessions(raw_tree)["sub-0002"])
        with pytest.raises(ValueError, match="unit_id"):
            builder.build_pose_set(pairs, tmp_path / "no", ball_radius_mm=synth.ball_radius_mm)
        path, pose_set = builder.build_pose_set(
            pairs,
            tmp_path / "ok",
            ball_radius_mm=synth.ball_radius_mm,
            unit_id="bench",
            software_version="dev",
            registration_kwargs={
                "rigid_iterations": 30,
                "demons_iterations": 10,
                "max_points": 200,
            },
        )
        assert pose_set["unit_id"] == "bench" and path.is_file()

    def test_empty_pairs_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="empty"):
            builder.build_pose_set(
                [], tmp_path, ball_radius_mm=100.0, unit_id="u", software_version="v"
            )

    def test_cli_dry_run_lists_the_pairs(self, raw_tree: Path) -> None:
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--raw-root", str(raw_tree), "--dry-run"],
            capture_output=True,
            text=True,
            check=False,
            env={
                "PYTHONPATH": str(REPO / "src"),
                "SPECTRAMR_SUPPRESS_CLINICAL_WARNING": "1",
                "PATH": "",
            },
        )
        assert result.returncode == 0, result.stderr
        assert "2 subject(s) with repeats, 3 pair(s)" in result.stdout
        assert "sub-0001_ses-03_ses-04" in result.stdout

    def test_cli_lowfield_dir_flag_names_what_it_tried(self, raw_tree: Path) -> None:
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--raw-root", str(raw_tree), "--lowfield-dir", "nope"],
            capture_output=True,
            text=True,
            check=False,
            env={
                "PYTHONPATH": str(REPO / "src"),
                "SPECTRAMR_SUPPRESS_CLINICAL_WARNING": "1",
                "PATH": "",
            },
        )
        assert result.returncode != 0
        assert "tried 'nope'" in result.stderr and "64mT_data" not in result.stderr
