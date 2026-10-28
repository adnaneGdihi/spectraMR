"""Removed config keys are refused at load, with the reason and the live key (#568 et al.)."""

from __future__ import annotations

import re

import pytest
from pydantic import ValidationError

from spectramr.config.schemas.removals import REMOVALS, RemovalRecord, removals_for_block
from spectramr.config.schemas.renames import RENAMES, reject_renamed_keys


def _minimal(**extra):
    return {
        "logging": {"identity": {"experiment": "t"}},
        "data": {"dataset_type": "kspace"},
        "model": {"model_type": "unet"},
        "optimization": {},
        **extra,
    }


def _declaring(dotted: str, value=True) -> dict:
    """A minimal settings document that declares ``dotted``."""
    doc = _minimal()
    *parents, leaf = dotted.split(".")
    node = doc
    for part in parents:
        node = node.setdefault(part, {})
    node[leaf] = value
    return doc


@pytest.mark.parametrize("dotted", sorted(REMOVALS))
def test_each_removed_key_is_refused_at_load(dotted: str) -> None:
    """Planted, one per record: the real settings class refuses the key by name."""
    from spectramr.config.settings import TrainingSettings

    value = {"persistent_root": "x"} if dotted == "artifacts" else True
    with pytest.raises(ValidationError, match=re.escape(f"`{dotted}` was ") + "(removed|retired)"):
        TrainingSettings(**_declaring(dotted, value))


@pytest.mark.parametrize(
    ("dotted", "live"),
    [
        ("training.enable_gradient_checkpointing", "optimization.gradient.enable_checkpointing"),
        ("training.num_workers", "data.loader.num_workers"),
        ("physics.coil_sensitivity.estimation_method", "physics.coil_processing.estimation.method"),
        ("artifacts", "training.output_dir"),
        ("early_stopping.check_interval", "validation.schedule.interval_steps"),
    ],
)
def test_the_refusal_names_the_live_key(dotted: str, live: str) -> None:
    assert f"`{live}`" in REMOVALS[dotted].message()


def test_resume_from_refusal_points_a_teacher_path_at_the_distillation_key() -> None:
    msg = REMOVALS["checkpoint.resume_from"].message()
    assert "Move the value to `training.distillation.teacher_checkpoint`" in msg
    assert "would change what the run does" not in msg


def test_the_live_checkpointing_key_still_loads() -> None:
    from spectramr.config.settings import TrainingSettings

    cfg = TrainingSettings(**_minimal(optimization={"gradient": {"enable_checkpointing": True}}))
    assert cfg.optimization.gradient.enable_checkpointing is True


def test_no_removed_key_is_also_a_rename() -> None:
    """A key is either moved or removed; both would give two messages for one fact."""
    assert not set(REMOVALS) & set(RENAMES)


@pytest.mark.parametrize("dotted", sorted(REMOVALS))
def test_no_removed_key_is_still_a_field_of_its_block(dotted: str) -> None:
    """Refused AND declared would be a field nothing can ever set."""
    from spectramr.config.schemas.base import DeepSpeedConfigSchema
    from spectramr.config.schemas.checkpoint import CheckpointConfigSchema
    from spectramr.config.schemas.early_stopping import EarlyStoppingConfigSchema
    from spectramr.config.schemas.metrics import MetricsConfigSchema
    from spectramr.config.schemas.physics import CoilSensitivityConfig, DataConsistencyConfig
    from spectramr.config.schemas.training.base import (
        BaseTrainingConfigSchema,
        DiffusionTrainingConfigSchema,
        TrainingStrategyConfigSchema,
    )
    from spectramr.config.schemas.training.bloch_manifold_dps import BlochManifoldDPSConfig
    from spectramr.config.schemas.training.equivariance_conformal import (
        EquivarianceConformalConfig,
    )
    from spectramr.config.schemas.training.phys_residual_conformal import (
        PhysResidualConformalConfig,
    )
    from spectramr.config.schemas.validation import ValidationConfigSchema
    from spectramr.config.settings import TrainingSettings

    owners = {
        "": [TrainingSettings],
        "validation": [ValidationConfigSchema],
        "checkpoint": [CheckpointConfigSchema],
        "early_stopping": [EarlyStoppingConfigSchema],
        "metrics": [MetricsConfigSchema],
        "training": [TrainingStrategyConfigSchema, BaseTrainingConfigSchema],
        "training.diffusion": [DiffusionTrainingConfigSchema],
        "training.bloch_manifold_dps": [BlochManifoldDPSConfig],
        "training.equivariance_conformal": [EquivarianceConformalConfig],
        "training.phys_residual_conformal": [PhysResidualConformalConfig],
        "physics.data_consistency": [DataConsistencyConfig],
        "physics.coil_sensitivity": [CoilSensitivityConfig],
        "parallel.deepspeed": [DeepSpeedConfigSchema],
    }
    rec = REMOVALS[dotted]
    for cls in owners[rec.mount_path]:
        assert rec.legacy_key not in cls.model_fields, cls.__name__


def test_a_mount_refuses_an_injected_removal_and_nothing_else() -> None:
    table = {"widget.old": RemovalRecord("widget.old", "2026-10-04", "Gone.", live="widget.new")}
    validator = reject_renamed_keys("widget", table={}, removals=table)
    assert validator(None, {"kept": 1}) == {"kept": 1}
    with pytest.raises(ValueError, match=r"`widget\.old` was removed.*`widget\.new`"):
        validator(None, {"old": 1})


def test_a_record_whose_mount_does_not_own_it_is_refused() -> None:
    with pytest.raises(ValueError, match="does not own"):
        RemovalRecord("a.b.c", "2026-10-04", "x", mount="a.z")


def test_a_nested_record_is_served_by_its_nested_mount() -> None:
    assert "degradation_dynamics" in removals_for_block("training.diffusion")
    assert "degradation_dynamics" not in removals_for_block("training")
    assert "artifacts" in removals_for_block("")


def test_strip_removed_keys_drops_only_listed_paths_and_leaves_the_input_alone() -> None:
    from spectramr.config.schemas.removals import strip_removed_keys

    doc = {
        "early_stopping": {"check_interval": 5, "patience": 4},
        "validation": {"schedule": {"interval_steps": 5}},
        "artifacts": {"persistent_root": "x"},
    }
    out, dropped = strip_removed_keys(doc)
    assert out == {
        "early_stopping": {"patience": 4},
        "validation": {"schedule": {"interval_steps": 5}},
    }
    assert sorted(r.legacy for r in dropped) == ["artifacts", "early_stopping.check_interval"]
    assert doc["early_stopping"]["check_interval"] == 5, "the caller's document was mutated"


def test_save_consolidated_best_is_refused_with_the_reason() -> None:
    from spectramr.config.schemas.base import DeepSpeedConfigSchema

    with pytest.raises(ValidationError, match="save_consolidated_best` was retired"):
        DeepSpeedConfigSchema(save_consolidated_best=True)


def test_a_retired_knob_that_was_read_does_not_claim_nothing_read_it() -> None:
    message = REMOVALS["parallel.deepspeed.save_consolidated_best"].message()
    assert "nothing read it" not in message
    assert "was retired (2026-10-10): it was read, and its behaviour is gone." in message
    assert "no separate consolidated file" in message


@pytest.mark.parametrize(
    "dotted", sorted(k for k, rec in REMOVALS.items() if not rec.was_read)
)
def test_unread_records_keep_their_message(dotted: str) -> None:
    rec = REMOVALS[dotted]
    assert rec.message().startswith(f"`{dotted}` was removed ({rec.since}): nothing read it. ")


def test_only_the_consolidated_best_knob_was_read() -> None:
    assert [k for k, rec in REMOVALS.items() if rec.was_read] == [
        "parallel.deepspeed.save_consolidated_best"
    ]


def test_strip_removed_keys_drops_a_retired_read_knob_from_a_snapshot() -> None:
    from spectramr.config.schemas.removals import strip_removed_keys

    doc = {"parallel": {"deepspeed": {"save_consolidated_best": False, "zero_stage": 3}}}
    out, dropped = strip_removed_keys(doc)
    assert out == {"parallel": {"deepspeed": {"zero_stage": 3}}}
    assert [r.legacy for r in dropped] == ["parallel.deepspeed.save_consolidated_best"]


@pytest.mark.parametrize(
    ("schema_path", "cls_name", "key"),
    [
        (
            "spectramr.config.schemas.training.bloch_manifold_dps",
            "BlochManifoldDPSConfig",
            "pretrained_score_checkpoint",
        ),
        (
            "spectramr.config.schemas.training.equivariance_conformal",
            "EquivarianceConformalConfig",
            "pretrained_reconstructor_checkpoint",
        ),
        (
            "spectramr.config.schemas.training.phys_residual_conformal",
            "PhysResidualConformalConfig",
            "pretrained_reconstructor_checkpoint",
        ),
    ],
)
def test_existence_only_prior_fields_are_refused_with_the_issue(
    schema_path: str, cls_name: str, key: str
) -> None:
    import importlib

    cls = getattr(importlib.import_module(schema_path), cls_name)
    with pytest.raises(ValidationError, match="#647"):
        cls(**{key: "x"})


@pytest.mark.parametrize("key", ["tissue_prior_checkpoint", "corruption_calibration_checkpoint"])
def test_pmps_prior_fields_are_refused_on_the_training_block(key: str) -> None:
    # `TrainingConfigCorruptionCalibration` / `TrainingConfigPairedSynthesis` are unreachable from
    # `TrainingSettings`; `training:` is validated by `TrainingStrategyConfigSchema`.
    from spectramr.config.schemas.training.base import TrainingStrategyConfigSchema

    with pytest.raises(ValidationError, match="#647"):
        TrainingStrategyConfigSchema(**{key: "x"})
