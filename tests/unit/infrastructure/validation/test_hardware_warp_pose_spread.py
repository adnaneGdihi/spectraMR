"""Planted violations for the ``hardware_warp_pose_spread`` Tier-1 check.

Pairs with ``src/spectramr/infrastructure/validation/hardware_warp_pose_spread.py``. Each case
is one shape the check claims to catch (non-negotiable 15): two rotation axes at 60 degrees
pass; rotations about one axis, and a single pair, turn it red; a missing or malformed pose
set is a failed verdict naming the defect; a floor raised above the constants turns it red
(the floor is read); an arm of another ``training_mode`` is not applicable. The configs are
real ``TrainingSettings`` built through ``settings_from_dict``, so the knobs travel the same
``extra="allow"`` route a YAML takes, and the verdict is read through the production
``ConfigHealthChecker`` method rather than the module alone. The pose-set contract itself is
owned and tested by ``spectramr.data.hardware_warp_artifacts``; what is planted here is that
each of its refusals reaches the verdict as a failure, not the wording of the refusal.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from spectramr.infrastructure.validation.config_health_checker import ConfigHealthChecker
from spectramr.infrastructure.validation.hardware_warp_pose_spread import (
    FIX_HINT,
    TRAINING_MODE,
    YAML_KEYS,
)

pytestmark = [pytest.mark.unit]

FIXTURES = Path(__file__).resolve().parents[4] / "tests" / "fixtures" / "hardware_warp"
CHECK_NAME = "hardware_warp_pose_spread"
Z_AXIS = [0.0, 0.0, 0.05]
AXIS_60 = [0.05 * math.sin(math.radians(60.0)), 0.0, 0.05 * math.cos(math.radians(60.0))]
NO_SHIFT = 0.0


def _parallel_pairs(shift_mm: float) -> list[dict]:
    """Three pairs rotated about z only, each translated by ``shift_mm`` along a different axis."""
    return [
        _pair("sub-01", Z_AXIS, (shift_mm, 0.0, 0.0)),
        _pair("sub-02", [0.0, 0.0, -0.03], (0.0, shift_mm, 0.0)),
        _pair("sub-03", [0.0, 0.0, 0.04], (0.0, 0.0, shift_mm)),
    ]


def _pair(subject: str, omega: list[float], translation=(0.5, -0.3, 0.2)) -> dict:
    return {
        "subject_id": subject,
        "sessions": ["ses-01", "ses-02"],
        "omega": list(omega),
        "translation": list(translation),
        "residual_path": None,
    }


def _write_pose_set(path: Path, pairs: list[dict], **overrides) -> Path:
    doc = {
        "unit_id": "hyperfine_swoop_lumc",
        "software_version": "unknown",
        "ball_radius_mm": 100.0,
        "center_mm": [0.0, 0.0, 0.0],
        "pairs": pairs,
    }
    doc.update(overrides)
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def _settings(pose_set_path: Path | str, training_mode: str = TRAINING_MODE, **knobs):
    from spectramr.config.schemas.base import CANONICAL_CONFIG_VERSION
    from spectramr.config.settings import TrainingSettings

    training = {"training_mode": training_mode}
    if training_mode == TRAINING_MODE:
        training.update(
            {
                "pose_set_path": str(pose_set_path),
                "artifact_path": "experiments/results/hwc/artifact.pt",
                "unit_id": "hyperfine_swoop_lumc",
                "software_version": "unknown",
                "basis_points_per_axis": 7,
            }
        )
    training.update(knobs)
    return TrainingSettings.settings_from_dict(
        {
            "config_version": CANONICAL_CONFIG_VERSION,
            "data": {
                "train_path": "/tmp/train",
                "val_path": "/tmp/val",
                "batch_size": 2,
                "dataset_type": "mrixfields",
            },
            "optimization": {"learning_rate": 1e-4},
            "logging": {},
            "model": {"model_type": "unet"},
            "training": training,
        }
    )


def _check(settings):
    return ConfigHealthChecker().check_hardware_warp_pose_spread(settings)


class TestPolarity:
    def test_two_non_parallel_axes_pass(self, tmp_path: Path) -> None:
        pose_set = _write_pose_set(
            tmp_path / "ok.json", [_pair("sub-01", Z_AXIS), _pair("sub-01", AXIS_60)]
        )
        result = _check(_settings(pose_set))
        assert result.passed, result.message
        assert result.check_name == CHECK_NAME
        assert "l=2" in result.message and "l=4" in result.message
        assert "1.000e-04" in result.message  # the floor is stated beside the constants

    def test_parallel_axes_turn_red(self, tmp_path: Path) -> None:
        """Rotations about one axis leave the axisymmetric harmonic of every degree unobserved."""
        pose_set = _write_pose_set(tmp_path / "parallel.json", _parallel_pairs(NO_SHIFT))
        result = _check(_settings(pose_set))
        assert not result.passed
        assert result.severity == "error"
        assert result.fix_hint == FIX_HINT
        assert result.yaml_keys == YAML_KEYS
        assert "[2, 3, 4]" in result.message and "1.000e-04" in result.message

    def test_translations_are_observations_too(self, tmp_path: Path) -> None:
        """``ad_xi v`` carries ``-H_Phi t``: a head shift observes the warp's Jacobian along it.

        Parallel axes are therefore degenerate only up to the translations, and the constant
        the check reports grows with ``|t| / R``. Certifying the rotation axes alone would
        disagree with the design ``estimate_warp`` solves, so the check keeps the term and
        the floor decides: 0.1 mm shifts on a 100 mm ball stay red, 5 mm ones do not.
        """
        small = _check(_settings(_write_pose_set(tmp_path / "s.json", _parallel_pairs(0.1))))
        large = _check(_settings(_write_pose_set(tmp_path / "l.json", _parallel_pairs(5.0))))
        assert not small.passed and small.severity == "error"
        assert large.passed, large.message

    def test_single_pair_turns_red(self, tmp_path: Path) -> None:
        result = _check(
            _settings(_write_pose_set(tmp_path / "one.json", [_pair("sub-01", Z_AXIS)]))
        )
        assert not result.passed and result.severity == "error"
        assert "1 session pair(s)" in result.message

    def test_floor_is_read(self, tmp_path: Path) -> None:
        pose_set = _write_pose_set(
            tmp_path / "ok.json", [_pair("sub-01", Z_AXIS), _pair("sub-01", AXIS_60)]
        )
        assert _check(_settings(pose_set)).passed
        result = _check(_settings(pose_set, identifiability_floor=1.0))
        assert not result.passed and "1.000e+00" in result.message

    def test_degree_band_is_read(self, tmp_path: Path) -> None:
        pose_set = _write_pose_set(
            tmp_path / "ok.json", [_pair("sub-01", Z_AXIS), _pair("sub-01", AXIS_60)]
        )
        result = _check(_settings(pose_set, max_degree=2))
        assert result.passed
        assert "l=2" in result.message and "l=3" not in result.message

    def test_other_training_mode_is_not_applicable(self) -> None:
        result = _check(_settings("unused.json", training_mode="reconstruction"))
        assert result.passed and result.severity == "info"
        assert "not applicable" in result.message


class TestNeverSilent:
    def test_missing_file_fails_naming_the_path(self, tmp_path: Path) -> None:
        missing = tmp_path / "absent.json"
        result = _check(_settings(missing))
        assert not result.passed and result.severity == "error"
        assert str(missing) in result.message
        assert result.yaml_keys == ["training.pose_set_path"]

    def test_not_json_fails(self, tmp_path: Path) -> None:
        broken = tmp_path / "broken.json"
        broken.write_text("{not json", encoding="utf-8")
        result = _check(_settings(broken))
        assert not result.passed and "not readable JSON" in result.message

    def test_pair_missing_omega_fails_naming_the_pair(self, tmp_path: Path) -> None:
        pair = _pair("sub-01", Z_AXIS)
        del pair["omega"]
        result = _check(_settings(_write_pose_set(tmp_path / "p.json", [pair])))
        assert not result.passed
        assert "pairs[0]" in result.message and "omega" in result.message

    def test_pure_translation_pair_is_read_and_alone_stays_red(self, tmp_path: Path) -> None:
        """omega = 0 is legal: it observes ``-H_Phi t`` and leaves the harmonics independent
        of the shift direction unobserved, so on its own it cannot clear the floor."""
        pose_set = _write_pose_set(
            tmp_path / "shift.json", [_pair("sub-01", [0.0, 0.0, 0.0], (5.0, 0.0, 0.0))]
        )
        result = _check(_settings(pose_set))
        assert not result.passed and result.severity == "error"
        assert "1 session pair(s)" in result.message and "[2, 3, 4]" in result.message

    def test_missing_required_knob_fails_naming_it(self, tmp_path: Path) -> None:
        pose_set = _write_pose_set(tmp_path / "ok.json", [_pair("sub-01", Z_AXIS)])
        result = _check(_settings(pose_set, unit_id=None))
        assert not result.passed and result.severity == "error"
        assert "unit_id" in result.message


class TestContractDefectsReachTheVerdict:
    """Every refusal of the data owner is a failed verdict on ``training.pose_set_path``."""

    @pytest.mark.parametrize(
        ("overrides", "needle"),
        [
            ({"pairs": []}, "non-empty list"),
            ({"pairs": "not-a-list"}, "non-empty list"),
            ({"unit_id": None, "software_version": None}, "missing keys"),
            ({"extra_key": 1}, "unknown keys"),
        ],
    )
    def test_top_level_defects(self, tmp_path: Path, overrides: dict, needle: str) -> None:
        doc = _write_pose_set(tmp_path / "d.json", [_pair("sub-01", Z_AXIS)])
        raw = json.loads(doc.read_text())
        for key, value in overrides.items():
            if value is None:
                raw.pop(key)
            else:
                raw[key] = value
        doc.write_text(json.dumps(raw))
        result = _check(_settings(doc))
        assert not result.passed and result.severity == "error"
        assert "cannot be certified" in result.message and needle in result.message
        assert result.yaml_keys == ["training.pose_set_path"]

    @pytest.mark.parametrize(
        ("omega", "needle"),
        [
            ([0.0, 0.05], "list of 3"),
            (["a", 0.0, 0.0], "non-numeric"),
            ([math.nan, 0, 0], "non-finite"),
        ],
    )
    def test_bad_omega(self, tmp_path: Path, omega: list, needle: str) -> None:
        path = _write_pose_set(tmp_path / "o.json", [_pair("sub-01", omega)])
        result = _check(_settings(path))
        assert not result.passed and result.severity == "error"
        assert "pairs[0].omega" in result.message and needle in result.message


class TestCommittedFixtures:
    """The fixtures the exp_ulf_09 arms point at carry the polarity the arms claim."""

    def test_two_axes_fixture_passes(self) -> None:
        result = _check(_settings(FIXTURES / "pose_set_two_axes.json"))
        assert result.passed, result.message

    def test_single_axis_fixture_is_the_planted_violation(self) -> None:
        result = _check(_settings(FIXTURES / "pose_set_single_axis.json"))
        assert not result.passed and result.severity == "error"

    def test_wired_into_run_all_checks(self) -> None:
        report = ConfigHealthChecker().run_all_checks(
            _settings(FIXTURES / "pose_set_single_axis.json")
        )
        names = [r.check_name for r in report.errors]
        assert CHECK_NAME in names, names


class TestCommittedArms:
    """The two exp_ulf_09 arms, loaded the way ``spectramr audit`` loads them.

    The arm is the planted pair for this gate: the main arm must certify and the single-pose
    ablation must be refused. Loading through ``from_yaml`` is what caught a bare
    ``readout_term: off`` parsing as the YAML boolean ``False``.
    """

    ARMS = Path(__file__).resolve().parents[4] / "experiments" / "inprogress" / "ulf_physics"

    def _arm(self, name: str, monkeypatch: pytest.MonkeyPatch):
        from spectramr.config.settings import TrainingSettings

        monkeypatch.chdir(self.ARMS.parents[2])  # pose_set_path is repo-root relative
        return TrainingSettings.from_yaml(str(self.ARMS / name))

    def test_main_arm_is_certified(self, monkeypatch: pytest.MonkeyPatch) -> None:
        result = _check(self._arm("exp_ulf_09_hardware_warp_from_repeats.yaml", monkeypatch))
        assert result.passed, result.message

    def test_single_pose_ablation_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        result = _check(self._arm("exp_ulf_09_ablate_single_pose.yaml", monkeypatch))
        assert not result.passed and result.severity == "error"
        assert "[2, 3, 4]" in result.message
