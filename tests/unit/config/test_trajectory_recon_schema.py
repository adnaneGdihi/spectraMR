"""trajectory_recon training-mode wiring (vf_35 Phase 3b).

The new paradigm needs four things wired: a frozen config block, a mount point on
the training schema, an entry in the valid-modes allow-list + constraints, and a
strategy_class path. These tests pin all four.
"""
from __future__ import annotations

import pytest


def test_trajectory_recon_config_fields_and_validation():
    from spectramr.config.schemas.training.strategy_knobs_2026_06 import (
        TrainingConfigTrajectoryRecon,
    )

    c = TrainingConfigTrajectoryRecon(read_measured_trajectory=True, girf_param_dim=128)
    assert c.read_measured_trajectory is True
    assert c.trajectory_units == "cycles_per_fov"  # default
    assert c.girf_param_dim == 128
    assert c.self_supervised_sharpness is False
    assert c.sharpness_weight == 0.0
    # invalid units rejected (Literal)
    with pytest.raises(Exception):
        TrainingConfigTrajectoryRecon(trajectory_units="furlongs")


def test_trajectory_recon_mounted_on_training_schema():
    from spectramr.config.schemas.training.base import TrainingStrategyConfigSchema

    s = TrainingStrategyConfigSchema(trajectory_recon={"read_measured_trajectory": True})
    assert s.trajectory_recon is not None
    assert s.trajectory_recon.read_measured_trajectory is True


def test_trajectory_recon_in_valid_modes_constraints_and_strategy_paths():
    from spectramr.config.validation_constants import (
        VALID_TRAINING_MODES,
    )
    from spectramr.infrastructure.training.strategy_factory import (
        TrainingStrategyFactory,
    )

    paths = TrainingStrategyFactory.STRATEGY_CLASS_PATHS
    assert "trajectory_recon" in VALID_TRAINING_MODES
    assert "trajectory_recon" in paths
    assert paths["trajectory_recon"].startswith("spectramr.")
