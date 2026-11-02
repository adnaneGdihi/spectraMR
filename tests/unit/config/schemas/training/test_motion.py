"""MotionConfig: the single-task control knob."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from spectramr.config.schemas.training.motion import MotionConfig


def test_fixed_trajectory_seed_defaults_to_meta_training() -> None:
    assert MotionConfig().fixed_trajectory_seed is None


def test_fixed_trajectory_seed_accepts_a_seed() -> None:
    assert MotionConfig(fixed_trajectory_seed=42).fixed_trajectory_seed == 42


def test_negative_seed_and_unknown_keys_raise() -> None:
    with pytest.raises(ValidationError):
        MotionConfig(fixed_trajectory_seed=-1)
    with pytest.raises(ValidationError):
        MotionConfig(fixed_trajectory=True)
