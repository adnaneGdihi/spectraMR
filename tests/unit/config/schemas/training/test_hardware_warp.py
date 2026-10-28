"""Unit tests for :mod:`spectramr.config.schemas.training.hardware_warp`.

The schema is the one owner of the calibration knobs' defaults and bounds, so the tests pin
what the Tier-1 check and the strategy will read: the defaults, every bound, the degree-band
validator, the strictness (frozen, unknown key raises), the ``training_mode`` dispatch through
the same ``create_training_config`` loader the other paradigms use, and the typed read of the
knobs off a loaded ``training:`` block that carries them as extras.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from spectramr.config.schemas.training import _MODE_DISPATCH, create_training_config
from spectramr.config.schemas.training.base import (
    BaseTrainingConfigSchema,
    TrainingStrategyConfigSchema,
)
from spectramr.config.schemas.training.hardware_warp import (
    TrainingConfigHardwareWarpCalibration,
)

pytestmark = [pytest.mark.unit]

REQUIRED = {
    "pose_set_path": "tests/fixtures/hardware_warp/pose_set_two_axes.json",
    "artifact_path": "experiments/results/hwc/artifact.pt",
    "unit_id": "hyperfine_swoop_lumc",
    "software_version": "unknown",
}


class TestDefaultsAndBounds:
    def test_defaults(self) -> None:
        cfg = TrainingConfigHardwareWarpCalibration(**REQUIRED)
        assert (cfg.max_degree, cfg.min_degree) == (4, 2)
        assert cfg.ball_radius_mm == 100.0
        assert cfg.basis_points_per_axis == 9
        assert cfg.identifiability_floor == pytest.approx(1e-4)
        assert cfg.readout_term == "off"
        assert isinstance(cfg.pose_set_path, Path)
        assert isinstance(cfg.artifact_path, Path)

    @pytest.mark.parametrize("name", sorted(REQUIRED))
    def test_required_knobs_are_required(self, name: str) -> None:
        with pytest.raises(ValidationError, match=name):
            TrainingConfigHardwareWarpCalibration(
                **{k: v for k, v in REQUIRED.items() if k != name}
            )

    @pytest.mark.parametrize(
        "override",
        [
            {"max_degree": 1},
            {"max_degree": 9},
            {"min_degree": 0},
            {"ball_radius_mm": 0.0},
            {"basis_points_per_axis": 4},
            {"basis_points_per_axis": 42},
            {"identifiability_floor": 0.0},
            {"readout_term": "within_session"},
            {"unit_id": ""},
            {"software_version": ""},
        ],
        ids=lambda o: "=".join(map(str, next(iter(o.items())))),
    )
    def test_out_of_range_values_raise(self, override: dict) -> None:
        with pytest.raises(ValidationError, match=next(iter(override))):
            TrainingConfigHardwareWarpCalibration(**{**REQUIRED, **override})

    def test_min_degree_above_max_degree_raises(self) -> None:
        with pytest.raises(ValidationError, match=r"min_degree .* must not exceed max_degree"):
            TrainingConfigHardwareWarpCalibration(**REQUIRED, min_degree=5, max_degree=4)

    def test_equal_degrees_are_a_single_block(self) -> None:
        cfg = TrainingConfigHardwareWarpCalibration(**REQUIRED, min_degree=3, max_degree=3)
        assert cfg.min_degree == cfg.max_degree == 3


class TestStrictness:
    def test_unknown_key_raises(self) -> None:
        with pytest.raises(ValidationError, match="max_degre"):
            TrainingConfigHardwareWarpCalibration(**REQUIRED, max_degre=3)

    def test_frozen(self) -> None:
        cfg = TrainingConfigHardwareWarpCalibration(**REQUIRED)
        with pytest.raises(ValidationError):
            cfg.max_degree = 5  # type: ignore[misc]

    def test_paradigm_fields_are_exactly_the_ten_knobs(self) -> None:
        assert TrainingConfigHardwareWarpCalibration.paradigm_field_names() == frozenset(
            {
                "max_degree",
                "min_degree",
                "ball_radius_mm",
                "basis_points_per_axis",
                "identifiability_floor",
                "pose_set_path",
                "artifact_path",
                "readout_term",
                "unit_id",
                "software_version",
            }
        )
        assert not TrainingConfigHardwareWarpCalibration.paradigm_field_names() & frozenset(
            BaseTrainingConfigSchema.model_fields
        )


class TestDispatch:
    def test_mode_is_in_the_dispatch_table(self) -> None:
        assert (
            _MODE_DISPATCH["hardware_warp_calibration"] == "TrainingConfigHardwareWarpCalibration"
        )

    def test_create_training_config_resolves_the_mode(self) -> None:
        cfg = create_training_config(
            {"training_mode": "hardware_warp_calibration", **REQUIRED, "max_degree": 3}
        )
        assert isinstance(cfg, TrainingConfigHardwareWarpCalibration)
        assert cfg.max_degree == 3

    def test_create_training_config_rejects_a_bad_knob(self) -> None:
        with pytest.raises(ValidationError, match="identifiability_floor"):
            create_training_config(
                {
                    "training_mode": "hardware_warp_calibration",
                    **REQUIRED,
                    "identifiability_floor": -1.0,
                }
            )


class TestFromTrainingBlock:
    """The loaded ``training:`` block holds the knobs as extras; the schema types them."""

    def test_extras_become_typed_knobs(self) -> None:
        block = TrainingStrategyConfigSchema(
            training_mode="hardware_warp_calibration",
            max_degree=3,
            identifiability_floor=2e-3,
            **REQUIRED,
        )
        cfg = TrainingConfigHardwareWarpCalibration.from_training_block(block)
        assert cfg.max_degree == 3
        assert cfg.identifiability_floor == pytest.approx(2e-3)
        assert cfg.basis_points_per_axis == 9  # undeclared -> schema default
        assert cfg.pose_set_path == Path(REQUIRED["pose_set_path"])

    def test_missing_required_knob_raises_naming_it(self) -> None:
        block = TrainingStrategyConfigSchema(
            training_mode="hardware_warp_calibration",
            **{k: v for k, v in REQUIRED.items() if k != "unit_id"},
        )
        with pytest.raises(ValidationError, match="unit_id"):
            TrainingConfigHardwareWarpCalibration.from_training_block(block)

    def test_none_counts_as_undeclared(self) -> None:
        block = TrainingStrategyConfigSchema(
            training_mode="hardware_warp_calibration", max_degree=None, **REQUIRED
        )
        assert TrainingConfigHardwareWarpCalibration.from_training_block(block).max_degree == 4
