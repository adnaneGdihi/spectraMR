"""Tests for ``CheckpointConfigSchema``.

Targets ``spectramr.config.schemas.checkpoint``. Validates: defaults, ge
constraints, ``frozen=True``, ``extra='ignore'`` (lenient), and the refusal of
the retired ``checkpoint_dir`` / ``save_dir`` keys.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from spectramr.config.schemas.checkpoint import CheckpointConfigSchema


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_default_construction() -> None:
    """Default checkpoint config has expected values."""
    cfg = CheckpointConfigSchema()
    assert cfg.enabled is True
    assert cfg.save_interval == 1000
    assert cfg.keep_last_n == 5
    assert cfg.format == "safetensors"
    assert cfg.produced_by_arm is None


# ---------------------------------------------------------------------------
# Campaign-dependency declaration: produced_by_arm
# ---------------------------------------------------------------------------


def test_produced_by_arm_accepts_nonempty_name() -> None:
    """``produced_by_arm`` records the upstream campaign arm that builds the ckpt."""
    cfg = CheckpointConfigSchema(produced_by_arm="baseline_m4raw_unet_4x")
    assert cfg.produced_by_arm == "baseline_m4raw_unet_4x"


def test_produced_by_arm_rejects_empty_string() -> None:
    """An empty producer name is a meaningless declaration (min_length=1)."""
    with pytest.raises(ValidationError):
        CheckpointConfigSchema(produced_by_arm="")


def test_explicit_construction() -> None:
    """Explicit values are preserved."""
    cfg = CheckpointConfigSchema(
        enabled=False,
        save_interval=500,
        keep_last_n=10,
        format="pth",
    )
    assert cfg.enabled is False
    assert cfg.save_interval == 500
    assert cfg.keep_last_n == 10
    assert cfg.format == "pth"


# ---------------------------------------------------------------------------
# Retired: checkpoint_dir and its save_dir alias (#2842)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["checkpoint_dir", "save_dir"])
def test_a_checkpoint_path_key_is_refused_by_name(key: str) -> None:
    """Checkpoints land in ``<training.output_dir>/checkpoints``; no key moves them.

    The block is ``extra="ignore"``, so without the removal record the key would be
    dropped in silence and the arm would believe it had chosen a directory.
    """
    with pytest.raises(ValueError, match=rf"checkpoint\.{key}.*training\.output_dir"):
        CheckpointConfigSchema(**{key: "/scratch/checkpoints"})
    assert key not in CheckpointConfigSchema.model_fields


# ---------------------------------------------------------------------------
# Numeric constraints
# ---------------------------------------------------------------------------


def test_save_interval_ge_one() -> None:
    """``save_interval >= 1``."""
    with pytest.raises(ValidationError):
        CheckpointConfigSchema(save_interval=0)


def test_keep_last_n_ge_one() -> None:
    """``keep_last_n >= 1``."""
    with pytest.raises(ValidationError):
        CheckpointConfigSchema(keep_last_n=0)


def test_keep_best_n_is_refused_by_name() -> None:
    """#1972: no reader, nothing to bound; a declaration fails at load and says why."""
    with pytest.raises(ValidationError, match=r"`checkpoint\.keep_best_n` was removed"):
        CheckpointConfigSchema(keep_best_n=3)
    assert "keep_best_n" not in CheckpointConfigSchema.model_fields


def test_retention_descriptions_name_the_step_files() -> None:
    assert "step_{N:08d}" in CheckpointConfigSchema.model_fields["keep_last_n"].description


@pytest.mark.parametrize("fmt", ["safetensors", "pt", "pth"])
def test_format_accepts_the_three(fmt) -> None:
    assert CheckpointConfigSchema(format=fmt).format == fmt


@pytest.mark.parametrize("fmt", ["ckpt", "PT", "bin", ""])
def test_format_refuses_anything_else(fmt) -> None:
    with pytest.raises(ValidationError):
        CheckpointConfigSchema(format=fmt)


@pytest.mark.parametrize("key", ["pretrained_path", "resume_training", "resume_from"])
def test_retired_keys_are_refused_with_their_reason(key) -> None:
    with pytest.raises(ValidationError, match=f"checkpoint.{key}` was removed"):
        CheckpointConfigSchema(**{key: "x"})


# ---------------------------------------------------------------------------
# Frozen + extra='ignore'
# ---------------------------------------------------------------------------


def test_schema_is_frozen() -> None:
    """Cannot mutate after construction."""
    cfg = CheckpointConfigSchema()
    with pytest.raises(ValidationError):
        cfg.save_interval = 99


def test_extra_fields_silently_ignored() -> None:
    """``extra='ignore'`` accepts unknown fields without raising."""
    # Should NOT raise — extra='ignore' is lenient.
    cfg = CheckpointConfigSchema(unknown_legacy_field=42)
    assert not hasattr(cfg, "unknown_legacy_field")


# ---------------------------------------------------------------------------
# async_write
# ---------------------------------------------------------------------------


def test_async_write_defaults_on_and_is_frozen() -> None:
    cfg = CheckpointConfigSchema()
    assert cfg.async_write is True
    with pytest.raises(ValidationError):
        cfg.async_write = True  # type: ignore[misc]


def test_async_write_refuses_a_non_boolean() -> None:
    with pytest.raises(ValidationError):
        CheckpointConfigSchema(async_write="sometimes")


def test_async_write_is_in_the_resolved_config_dump() -> None:
    """``resolved_config.json`` is ``model_dump(mode="json")``, so the run records it."""
    assert CheckpointConfigSchema(async_write=True).model_dump(mode="json")["async_write"] is True
