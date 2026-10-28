"""Tests for ``AugmentationConfigSchema``.

Targets ``spectramr.config.schemas.augmentation``. Pydantic v2 schema for
TorchIO-style augmentation toggles + probabilities. Validates
[0, 1] probability bounds, ``ge=0`` for std/alpha/sigma, and
``frozen=True``.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from spectramr.config.schemas.augmentation import AugmentationConfigSchema


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_default_construction() -> None:
    """All augmentations disabled by default."""
    cfg = AugmentationConfigSchema()
    assert cfg.enabled is False
    assert cfg.probability == 0.5
    assert cfg.enable_rotation is False
    assert cfg.enable_flip is False
    assert cfg.enable_noise is False


def test_default_rotation_range() -> None:
    """Rotation range defaults to ±15°."""
    cfg = AugmentationConfigSchema()
    assert cfg.rotation_range == (-15, 15)


def test_default_brightness_range() -> None:
    """Brightness range defaults to ``(0.8, 1.2)``."""
    cfg = AugmentationConfigSchema()
    assert cfg.brightness_range == (0.8, 1.2)


# ---------------------------------------------------------------------------
# Probability bounds
# ---------------------------------------------------------------------------


def test_probability_must_be_in_unit_interval() -> None:
    """``probability`` ∈ ``[0, 1]``."""
    with pytest.raises(ValidationError):
        AugmentationConfigSchema(probability=-0.1)
    with pytest.raises(ValidationError):
        AugmentationConfigSchema(probability=1.1)


def test_per_augmentation_probabilities_in_unit_interval() -> None:
    """Each ``prob_*`` field ∈ ``[0, 1]``."""
    with pytest.raises(ValidationError):
        AugmentationConfigSchema(prob_rotate=1.5)
    with pytest.raises(ValidationError):
        AugmentationConfigSchema(prob_flip=-0.1)


def test_probability_at_boundaries_accepted() -> None:
    """Values exactly 0.0 and 1.0 are accepted."""
    cfg_low = AugmentationConfigSchema(probability=0.0)
    cfg_high = AugmentationConfigSchema(probability=1.0)
    assert cfg_low.probability == 0.0
    assert cfg_high.probability == 1.0


# ---------------------------------------------------------------------------
# Non-negative parameters
# ---------------------------------------------------------------------------


def test_elastic_alpha_non_negative() -> None:
    """``elastic_alpha >= 0``."""
    with pytest.raises(ValidationError):
        AugmentationConfigSchema(elastic_alpha=-1.0)


def test_elastic_sigma_non_negative() -> None:
    """``elastic_sigma >= 0``."""
    with pytest.raises(ValidationError):
        AugmentationConfigSchema(elastic_sigma=-0.1)


def test_noise_std_non_negative() -> None:
    """``noise_std >= 0``."""
    with pytest.raises(ValidationError):
        AugmentationConfigSchema(noise_std=-0.05)


# ---------------------------------------------------------------------------
# Frozen + extra='forbid'
# ---------------------------------------------------------------------------


def test_schema_is_frozen() -> None:
    """Cannot mutate after construction."""
    cfg = AugmentationConfigSchema()
    with pytest.raises(ValidationError):
        cfg.enabled = True


@pytest.mark.parametrize("typo", ["enable_flipp", "random_rotation_degrees", "p"])
def test_unknown_keys_raise(typo: str) -> None:
    """A misspelled or legacy flag raises instead of training without that augmentation."""
    with pytest.raises(ValidationError, match=typo):
        AugmentationConfigSchema(enabled=True, **{typo: True})


def test_renamed_custom_transforms_keeps_its_own_message() -> None:
    """The RENAMES record still answers first, naming the canonical home."""
    with pytest.raises(ValueError, match=r"processing\.transforms"):
        AugmentationConfigSchema(custom_transforms=[{"name": "x"}])


# ---------------------------------------------------------------------------
# Lists default to empty
# ---------------------------------------------------------------------------


def test_flip_axes_defaults_to_empty_list() -> None:
    """``flip_axes`` defaults to an empty list."""
    cfg = AugmentationConfigSchema()
    assert cfg.flip_axes == []


def test_explicit_flip_axes() -> None:
    """``flip_axes`` accepts a list of integers."""
    cfg = AugmentationConfigSchema(flip_axes=[0, 1])
    assert cfg.flip_axes == [0, 1]
