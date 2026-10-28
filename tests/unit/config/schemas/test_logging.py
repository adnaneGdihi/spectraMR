"""Tests for ``LoggingConfigSchema``.

Targets ``spectramr.config.schemas.logging``. Logging / experiment-tracking /
TensorBoard / debug-snapshot configuration. Validates documented
defaults, ge constraints, ``frozen=True``.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from spectramr.config.schemas.enums import LogLevel
from spectramr.config.schemas.logging import (
    LoggingConfigSchema,
    LoggingSnapshotsConfigSchema,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_default_construction() -> None:
    """Default config matches documented values."""
    cfg = LoggingConfigSchema()
    assert cfg.sinks.level == LogLevel.INFO
    assert cfg.sinks.silent is False
    assert cfg.sinks.dir == "./logs"
    assert cfg.sinks.to_file is True
    assert cfg.sinks.to_console is True
    assert cfg.identity.experiment == "default_experiment"
    assert cfg.identity.run is None
    assert cfg.tracking.service == "tensorboard"
    assert cfg.identity.tags == {}


def test_explicit_construction() -> None:
    """Custom values override defaults."""
    cfg = LoggingConfigSchema(
        level=LogLevel.DEBUG,
        silent=True,
        experiment_name="my_run",
    )
    assert cfg.sinks.level == LogLevel.DEBUG
    assert cfg.sinks.silent is True
    assert cfg.identity.experiment == "my_run"


# ---------------------------------------------------------------------------
# Constraints (ge=1)
# ---------------------------------------------------------------------------


def test_log_interval_ge_one() -> None:
    """``log_interval >= 1``."""
    with pytest.raises(ValidationError):
        LoggingConfigSchema(log_interval=0)


def test_anomaly_check_interval_ge_one() -> None:
    """``anomaly_check_interval >= 1``."""
    with pytest.raises(ValidationError):
        LoggingConfigSchema(anomaly_check_interval=0)


def test_max_images_per_batch_ge_one() -> None:
    """``max_images_per_batch >= 1``."""
    with pytest.raises(ValidationError):
        LoggingConfigSchema(max_images_per_batch=0)


def test_save_images_per_epoch_ge_zero() -> None:
    """``save_images_per_epoch >= 0`` (0 disables)."""
    cfg = LoggingConfigSchema(save_images_per_epoch=0)
    assert cfg.save_images_per_epoch == 0
    with pytest.raises(ValidationError):
        LoggingConfigSchema(save_images_per_epoch=-1)


def test_debug_snapshot_max_calls_ge_one() -> None:
    """``snapshots.max_calls >= 1``.

    The bound tightened from 0 with the block decomposition: a count of 0 used
    to be how an arm said "never snapshot", and that is now spelled
    ``snapshots.enabled: false``. Leaving 0 legal would have kept two ways to
    say it, one of which leaves `enabled` reading True.
    """
    cfg = LoggingConfigSchema(snapshots={"max_calls": 1})
    assert cfg.snapshots.max_calls == 1
    for illegal in (0, -1):
        # `match=` is load-bearing. The flat spelling was promoted to `raise`
        # (2026-08-18), so a bare `raises(ValidationError)` would start catching
        # the RENAME instead of the bound -- green, pinning nothing, and absent
        # from any failure diff.
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            LoggingConfigSchema(snapshots={"max_calls": illegal})


# ---------------------------------------------------------------------------
# Enum coercion from string
# ---------------------------------------------------------------------------


def test_level_coerced_from_string() -> None:
    """``level='debug'`` (string) is coerced to ``LogLevel.DEBUG``."""
    cfg = LoggingConfigSchema(level="debug")
    assert cfg.sinks.level == LogLevel.DEBUG


def test_level_invalid_string_raises() -> None:
    """Invalid level string raises."""
    with pytest.raises(ValidationError):
        LoggingConfigSchema(level="cosmic")


# ---------------------------------------------------------------------------
# Optional fields can be None
# ---------------------------------------------------------------------------


def test_optional_run_name_can_be_none() -> None:
    """``run_name`` is optional."""
    cfg = LoggingConfigSchema(run_name=None)
    assert cfg.identity.run is None


def test_optional_wandb_fields_can_be_none() -> None:
    """WandB project / entity are optional."""
    cfg = LoggingConfigSchema(wandb_project=None, wandb_entity=None)
    assert cfg.wandb_project is None
    assert cfg.wandb_entity is None


# ---------------------------------------------------------------------------
# W&B is deferred (#675) -- declaring it must raise, not be silently ignored
# ---------------------------------------------------------------------------


def test_wandb_project_is_refused_not_ignored() -> None:
    """W&B is deferred (2026-08-12). Declaring it must fail, not be accepted and dropped.

    `logging:` is extra="ignore", so a DELETED field would be silently discarded --
    the facade this refusal exists to remove. The field must stay and say no.
    """
    with pytest.raises(ValidationError, match="Weights & Biases is not implemented"):
        LoggingConfigSchema(wandb_project="spectramr_research")


def test_wandb_entity_is_refused_too() -> None:
    with pytest.raises(ValidationError, match="Weights & Biases is not implemented"):
        LoggingConfigSchema(wandb_entity="some-team")


def test_an_unset_wandb_field_still_constructs() -> None:
    """Absence is not a declaration. The 510 arms that never mention W&B are unaffected."""
    cfg = LoggingConfigSchema()
    assert cfg.wandb_project is None and cfg.wandb_entity is None


def test_explicit_null_is_accepted() -> None:
    """`wandb_project: null` declares nothing. Refusing it would break the reference template."""
    assert LoggingConfigSchema(wandb_project=None).wandb_project is None


# ---------------------------------------------------------------------------
# Debug snapshots
# ---------------------------------------------------------------------------


def test_debug_snapshots_defaults_on() -> None:
    """``debug_snapshots`` enabled by default."""
    cfg = LoggingConfigSchema()
    assert cfg.snapshots.enabled is True
    assert cfg.snapshots.max_calls == 8
    assert cfg.snapshots.save_images is True
    assert cfg.snapshots.save_json is True


def test_log_steps_does_not_drive_the_snapshot_cadence() -> None:
    """``interval_steps: 0`` does NOT fall back to ``log_steps``.

    The description said it did, the resolver never mapped ``log_steps``, and
    the due-check invented a third rule -- one knob, three incompatible
    meanings, which is the shape non-negotiable #8 forbids. ``log_steps`` in
    fact drives the diffusion anomaly LOG (``_log_diffusion_anomalies``), not
    the snapshot writer.

    Pinned behaviourally: the resolver is what the writer actually consults,
    so assert ``log_steps`` cannot reach it whatever the arm declares.
    """
    from spectramr.infrastructure.training.debug_snapshot import (
        _resolve_config,
        snapshot_step_is_due,
    )

    cfg = LoggingConfigSchema()
    assert cfg.snapshots.interval_steps == 0
    assert cfg.snapshots.log_steps == [0, 1, 2, 10]

    resolved = _resolve_config(cfg)
    assert resolved.image_interval_steps == 0
    assert not hasattr(resolved, "log_steps")

    # Step 3 is absent from the default `log_steps`; it is still due, because
    # the budget -- not that list -- is what bounds the writes.
    assert 3 not in cfg.snapshots.log_steps
    assert snapshot_step_is_due(cfg, 3) is True

    # And the description may not re-advertise the retired fallback.
    described = LoggingSnapshotsConfigSchema.model_fields["interval_steps"].description
    assert "uses `log_steps` instead" not in described


# ---------------------------------------------------------------------------
# Frozen + extra='ignore'
# ---------------------------------------------------------------------------


def test_schema_is_frozen() -> None:
    """Cannot mutate after construction."""
    cfg = LoggingConfigSchema()
    with pytest.raises(ValidationError):
        cfg.silent = True


def test_extra_fields_silently_ignored() -> None:
    """Unknown legacy fields don't raise (``extra='ignore'``)."""
    cfg = LoggingConfigSchema(legacy_field="anything")
    assert cfg.identity.experiment == "default_experiment"


# ---------------------------------------------------------------------------
# Report-case recording (Task 2.3)
# ---------------------------------------------------------------------------


def test_report_cases_knobs_default():
    s = LoggingConfigSchema()
    assert s.report_cases.enabled is True
    assert s.report_cases.subdir == "report_cases"


class TestDistributedAndProgressBlocks:
    """`logging.distributed`, `logging.progress` and `logging.sinks.console_stream`."""

    def test_defaults_are_rank0_only_split_and_auto(self):
        from spectramr.config.schemas.enums import ConsoleStream, LogLevel, ProgressMode, RankScope

        cfg = LoggingConfigSchema()
        assert cfg.distributed.console_ranks is RankScope.RANK0
        assert cfg.distributed.secondary_level is LogLevel.ERROR
        assert cfg.distributed.file_ranks is RankScope.RANK0
        assert cfg.sinks.console_stream is ConsoleStream.SPLIT
        assert cfg.progress.mode is ProgressMode.AUTO
        assert cfg.progress.log_every_n_iterations == 1500

    def test_values_load_from_yaml_shaped_dicts(self):
        cfg = LoggingConfigSchema(
            distributed={"console_ranks": "all", "secondary_level": "warning", "file_ranks": "all"},
            progress={"mode": "lines", "log_every_n_iterations": 30},
            sinks={"console_stream": "stderr"},
        )
        assert cfg.distributed.console_ranks.value == "all"
        assert cfg.progress.log_every_n_iterations == 30
        assert cfg.sinks.console_stream.value == "stderr"

    @pytest.mark.parametrize(
        "bad",
        [
            {"distributed": {"console_ranks": "some"}},
            {"distributed": {"secondary_level": "loud"}},
            {"distributed": {"typo": 1}},
            {"progress": {"mode": "fancy"}},
            {"progress": {"log_every_n_iterations": 0}},
            {"progress": {"log_every_n_iterations": -5}},
            {"progress": {"typo": 1}},
            {"sinks": {"console_stream": "stdin"}},
        ],
    )
    def test_a_bad_value_or_unknown_key_raises(self, bad):
        with pytest.raises(ValidationError):
            LoggingConfigSchema(**bad)
