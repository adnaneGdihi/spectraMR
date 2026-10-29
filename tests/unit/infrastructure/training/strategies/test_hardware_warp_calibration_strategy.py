"""Unit tests for the ``hardware_warp_calibration`` strategy.

The orchestration is a module-level function driven from a stand-in config (the real schema
lives on a sibling branch), so the artifact, the floor refusal and the missing-pose-set refusal
are exercised end to end here; the class is checked the way the sibling
``test_quality_matching_strategy`` checks its strategy -- by the source of its hooks, because
constructing the real strategy needs the full DI/model harness. The factory key is asserted to
resolve to this class, so registration is observed rather than assumed (non-negotiable 16).
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from spectramr.data.hardware_warp_artifacts import ARTIFACT_KEYS, load_artifact
from spectramr.infrastructure.training.strategies.hardware_warp_calibration_strategy import (
    HardwareWarpCalibrationStrategy,
    HardwareWarpKnobs,
    require_hardware_warp_config,
    run_hardware_warp_calibration,
)
from tests.utils.hardware_warp_synthesis import write_planted_pose_set

pytestmark = [pytest.mark.unit]


def _config(
    tmp_path: Path, split: dict[str, Any] | None = None, **overrides: Any
) -> SimpleNamespace:
    pose_set_path, _ = write_planted_pose_set(tmp_path / "pose", n_subjects=3)
    training = {
        "training_mode": "hardware_warp_calibration",
        "max_degree": 3,
        "min_degree": 2,
        "ball_radius_mm": 100.0,
        "basis_points_per_axis": 11,
        "identifiability_floor": 1e-6,
        "pose_set_path": str(pose_set_path),
        "artifact_path": str(tmp_path / "out" / "hardware_warp.json"),
        "readout_term": "off",
        "unit_id": "swoop-01",
        "software_version": "1.2.3",
    }
    training.update(overrides)
    config = SimpleNamespace(training=SimpleNamespace(**training))
    if split is not None:
        defaults = {"type": "manifest", "holdout_subject": None, "loso_fold": None}
        config.data = SimpleNamespace(split=SimpleNamespace(**{**defaults, **split}))
    return config


class TestOrchestration:
    def test_writes_an_artifact_with_every_contract_key(self, tmp_path: Path) -> None:
        knobs = require_hardware_warp_config(_config(tmp_path))
        assert isinstance(knobs, HardwareWarpKnobs)
        artifact = run_hardware_warp_calibration(knobs)
        assert set(artifact) == ARTIFACT_KEYS
        assert artifact["n_pairs"] == 6
        assert load_artifact(knobs.artifact_path) == artifact
        assert all(
            v >= knobs.identifiability_floor for v in artifact["constant_per_degree"].values()
        )

    def test_constant_below_floor_raises_and_writes_nothing(self, tmp_path: Path) -> None:
        knobs = require_hardware_warp_config(_config(tmp_path, identifiability_floor=1e9))
        with pytest.raises(ValueError, match="below the floor"):
            run_hardware_warp_calibration(knobs)
        assert not knobs.artifact_path.exists()

    def test_missing_pose_set_raises_at_config_time(self, tmp_path: Path) -> None:
        config = _config(tmp_path, pose_set_path=str(tmp_path / "absent.json"))
        with pytest.raises(FileNotFoundError, match="pose_set_path"):
            require_hardware_warp_config(config)

    def test_unimplemented_readout_term_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="readout_term"):
            require_hardware_warp_config(_config(tmp_path, readout_term="within_session"))

    def test_missing_knob_raises_naming_the_field(self, tmp_path: Path) -> None:
        config = _config(tmp_path)
        del config.training.unit_id
        with pytest.raises(ValueError, match=r"training\.unit_id"):
            require_hardware_warp_config(config)

    @pytest.mark.parametrize(
        "override, match",
        [
            ({"min_degree": 4}, "min_degree <= training.max_degree"),
            ({"ball_radius_mm": 0.0}, "ball_radius_mm"),
            ({"basis_points_per_axis": 1}, "basis_points_per_axis"),
            ({"identifiability_floor": 0.0}, "identifiability_floor"),
            ({"software_version": ""}, "software_version"),
        ],
    )
    def test_invalid_knob_raises(self, tmp_path: Path, override: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            require_hardware_warp_config(_config(tmp_path, **override))

    def test_unit_mismatch_with_pose_set_raises(self, tmp_path: Path) -> None:
        knobs = require_hardware_warp_config(_config(tmp_path, unit_id="swoop-99"))
        with pytest.raises(ValueError, match="specific to both"):
            run_hardware_warp_calibration(knobs)


class TestHoldOneSubjectOut:
    def test_no_split_block_and_a_manifest_split_fit_every_pair(self, tmp_path: Path) -> None:
        assert require_hardware_warp_config(_config(tmp_path)).holdout_subject is None
        knobs = require_hardware_warp_config(_config(tmp_path, split={"type": "manifest"}))
        assert knobs.holdout_subject is None
        artifact = run_hardware_warp_calibration(knobs)
        assert artifact["n_pairs"] == 6 and artifact["held_out"] is None

    def test_loso_subject_holds_the_named_subject_out_and_reports_it(self, tmp_path: Path) -> None:
        split = {"type": "loso_subject", "holdout_subject": "sub-02"}
        knobs = require_hardware_warp_config(_config(tmp_path, split=split))
        assert knobs.holdout_subject == "sub-02"
        artifact = run_hardware_warp_calibration(knobs)
        assert artifact["n_pairs"] == 4
        held = artifact["held_out"]
        assert held["subject_id"] == "sub-02" and held["n_pairs"] == 2
        assert held["rms_after_mm"] < 1e-9 * held["rms_before_mm"]
        assert load_artifact(knobs.artifact_path)["held_out"] == held

    def test_unknown_holdout_subject_raises_before_writing(self, tmp_path: Path) -> None:
        split = {"type": "loso_subject", "holdout_subject": "sub-99"}
        knobs = require_hardware_warp_config(_config(tmp_path, split=split))
        with pytest.raises(ValueError, match="sub-99"):
            run_hardware_warp_calibration(knobs)
        assert not knobs.artifact_path.exists()

    def test_loso_fold_is_refused_because_it_indexes_the_manifest(self, tmp_path: Path) -> None:
        split = {"type": "loso_subject", "loso_fold": 1}
        with pytest.raises(ValueError, match="loso_fold"):
            require_hardware_warp_config(_config(tmp_path, split=split))

    def test_loso_subject_without_an_id_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="holdout_subject"):
            require_hardware_warp_config(_config(tmp_path, split={"type": "loso_subject"}))


class TestStrategyWiring:
    def test_factory_key_resolves_to_this_class(self) -> None:
        from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory

        path = TrainingStrategyFactory.STRATEGY_CLASS_PATHS["hardware_warp_calibration"]
        module_path, class_name = path.rsplit(".", 1)
        assert getattr(importlib.import_module(module_path), class_name) is (
            HardwareWarpCalibrationStrategy
        )

    def test_hooks_drive_the_orchestration(self) -> None:
        setup = inspect.getsource(
            HardwareWarpCalibrationStrategy._setup_strategy_specific_components
        )
        assert "require_hardware_warp_config(" in setup
        assert "isinstance(generator, HardwareWarpOperator)" in setup
        # The fit and its load into the operator happen at construction, not on a training
        # step: the substrate loader may yield no batch, and a step that never comes must not
        # mean an artifact that is never written.
        assert "run_hardware_warp_calibration(self._knobs)" in setup
        assert "load_artifact_record(self._artifact)" in setup
        step = inspect.getsource(HardwareWarpCalibrationStrategy._compute_losses_impl)
        assert "run_hardware_warp_calibration(" not in step
        assert "_zero_loss()" in step

    def test_declares_loss_ownership(self) -> None:
        assert HardwareWarpCalibrationStrategy.__dict__["inline_losses"] == frozenset()
        assert HardwareWarpCalibrationStrategy.__dict__["folds_image_losses"] is False
