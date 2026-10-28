"""Logging and experiment tracking configuration schema."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from .enums import ConsoleStream, LogLevel, ProgressMode, RankScope, TrackingService
from .renames import (
    fold_renamed_keys,
    folded_input_keys,
    folded_input_paths,
    reject_renamed_keys,
)

__all__ = [
    "LoggingConfigSchema",
    "LoggingDistributedConfigSchema",
    "LoggingIdentityConfigSchema",
    "LoggingImagesConfigSchema",
    "LoggingIntervalsConfigSchema",
    "LoggingProgressConfigSchema",
    "LoggingReportCasesConfigSchema",
    "LoggingSinksConfigSchema",
    "LoggingSnapshotsConfigSchema",
    "LoggingTrackingConfigSchema",
]

#: New sub-blocks are born strict even though the parent is `extra="ignore"`.
#: That asymmetry is the point: inside the parent a forgotten key VANISHES
#: (#550), so the strictness has to start somewhere.
_LOG_SUBBLOCK = ConfigDict(extra="forbid", frozen=True)


_UNREAD_1218 = (
    "NOT READ — nothing in src/ consumes it (KNOWN_UNCONSUMED ledger, "
    "tests/unit/config/test_schema_key_consumption.py). 1218 arms declare it "
    'and this block is extra="ignore", so deleting the field would silently drop the key rather than fail the arm. '
    "Wire it or retire it with a corpus migration; do not trust it today."
)
_UNREAD_893 = (
    "NOT READ — nothing in src/ consumes it (KNOWN_UNCONSUMED ledger, "
    "tests/unit/config/test_schema_key_consumption.py). 893 arms declare it "
    'and this block is extra="ignore", so deleting the field would silently drop the key rather than fail the arm. '
    "Wire it or retire it with a corpus migration; do not trust it today."
)


class LoggingIdentityConfigSchema(BaseModel):
    """What this run is called, for a human and for the tracking backend."""

    model_config = _LOG_SUBBLOCK

    experiment: str = Field(
        default="default_experiment",
        description="Experiment name; groups runs in the tracking backend.",
    )
    run: str | None = Field(
        default=None,
        description="Name of this particular run; None derives one.",
    )
    notes: str | None = Field(default=None, description="Free-text run notes.")
    tags: dict[str, Any] = Field(default_factory=dict, description="Key/value run tags.")


class LoggingSinksConfigSchema(BaseModel):
    """Where log lines go, and how much of them.

    `level` and `silent` are not destinations, but they decide what reaches
    one, so they read better beside the sinks than three blocks away.
    """

    model_config = _LOG_SUBBLOCK

    level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Logging level: debug, info, warning, error, critical.",
    )
    silent: bool = Field(default=False, description="Suppress logging output.")
    to_console: bool = Field(default=True, description="Log to the console.")
    console_stream: ConsoleStream = Field(
        default=ConsoleStream.SPLIT,
        description=(
            "Console stream: `split` (INFO and below to stdout, WARNING and above to "
            "stderr), `stdout`, or `stderr`."
        ),
    )
    to_file: bool = Field(default=True, description="Log to a file under `dir`.")
    dir: str = Field(default="./logs", description="Directory for log files.")


class LoggingDistributedConfigSchema(BaseModel):
    """What the ranks of a distributed launch write.

    By default only rank 0 logs. The other ranks still print ``secondary_level``
    and above, so a rank-specific failure (OOM, NaN) is not invisible; set
    ``console_ranks: all`` to give every rank the full level.
    """

    model_config = _LOG_SUBBLOCK

    console_ranks: RankScope = Field(
        default=RankScope.RANK0,
        description="`rank0`: other ranks are held to `secondary_level`. `all`: every rank logs at the full level.",
    )
    secondary_level: LogLevel = Field(
        default=LogLevel.ERROR,
        description="Console floor on ranks other than 0 when `console_ranks` is `rank0`.",
    )
    file_ranks: RankScope = Field(
        default=RankScope.RANK0,
        description=(
            "`rank0`: only rank 0 writes the text log. `all`: every rank writes its own "
            "`<name>.rank<N>.log` beside it, so ranks never interleave in one file."
        ),
    )


class LoggingProgressConfigSchema(BaseModel):
    """How progress is shown.

    A live tqdm bar is right on a terminal and wrong in a SLURM file, where every
    refresh becomes a line. `auto` therefore shows a bar on a TTY only, and the log
    gets one generic tracking line every `log_every_n_iterations` iterations. The bar and the lines are rank 0 only unless
    `logging.distributed.console_ranks` is `all`.
    """

    model_config = _LOG_SUBBLOCK

    mode: ProgressMode = Field(
        default=ProgressMode.AUTO,
        description="`auto` (bar on a TTY, lines otherwise), `bar`, `lines` or `off`.",
    )
    log_every_n_iterations: int = Field(
        default=1500,
        ge=1,
        description="Iterations between tracking lines in the log (tracking only; the bar is console-only).",
    )


class LoggingIntervalsConfigSchema(BaseModel):
    """Every-N-steps cadences. The block supplies the word `interval`."""

    model_config = _LOG_SUBBLOCK

    log: int = Field(default=100, ge=1, description="Log metrics every N steps.")
    save: int = Field(default=100, ge=1, description="Save artifacts every N steps.")
    validation_images: int = Field(
        default=1, ge=1, description="Save validation images every N validations."
    )
    anomaly_check: int = Field(
        default=100, ge=1, description="Run the anomaly check every N steps."
    )
    histogram: int = Field(
        default=1000,
        ge=1,
        description=(
            "Write weight/gradient histograms every N steps. Deliberately much "
            "coarser than `log`: `add_histogram` moves every parameter to host "
            "memory, which is a GPU sync per tensor (non-negotiable #9). Set it "
            "low only for a short debugging run."
        ),
    )


class LoggingImagesConfigSchema(BaseModel):
    """Which image panels are written, and how many.

    `save_images_per_epoch` is NOT here: nothing reads it (KNOWN_UNCONSUMED),
    and 882 arms set it. It stays flat on the parent rather than gaining a tidy
    home that would imply it works.
    """

    model_config = _LOG_SUBBLOCK

    log_input: bool = Field(default=False, description="Log input images.")
    log_validation: bool = Field(default=True, description="Log validation images.")
    log_difference: bool = Field(
        default=True, description="Log |prediction - target| difference images."
    )
    save_validation: bool = Field(default=True, description="Write validation images to disk.")
    max_per_batch: int = Field(default=4, ge=1, description="Cap on images logged per batch.")


class LoggingTrackingConfigSchema(BaseModel):
    """The experiment-tracking backend.

    `wandb_project` and `wandb_entity` are NOT here: both are inert
    (KNOWN_UNCONSUMED), so a W&B run configured through them is configured
    through nothing. They stay flat -- see the parent's note and issue #675.
    """

    model_config = _LOG_SUBBLOCK

    enabled: bool = Field(default=True, description="Enable experiment tracking at all.")
    service: TrackingService = Field(
        default=TrackingService.TENSORBOARD,
        description=(
            "Tracking backend. Closed enum: an unrecognised value used to fall "
            "off the `== 'tensorboard'` branch and disable tracking silently."
        ),
    )
    enable_tensorboard: bool = Field(
        default=True,
        description=(
            "Enable the TensorBoard writer. Read alongside `service`: either "
            "one off means no writer."
        ),
    )
    tensorboard_dir: str | None = Field(
        default=None,
        description=(
            "TensorBoard event directory. Resolved RELATIVE TO THE RUN "
            "DIRECTORY, so the default `None` and the corpus's `./tensorboard` "
            "both land on `<run_dir>/tensorboard`; an absolute path overrides "
            "it. Per-run isolation is the point -- a CWD-relative directory "
            "shared across runs interleaves event files."
        ),
    )


class LoggingSnapshotsConfigSchema(BaseModel):
    """Debug snapshots -- the per-step tensor/JSON dumps.

    Named `snapshots`, not `debug_snapshots`: that is the retired scalar's own
    name, and a destination block may not share it (a migrated arm would fold
    into itself). See `fold_renamed_keys`.
    """

    model_config = _LOG_SUBBLOCK

    enabled: bool = Field(default=True, description="Write debug snapshots.")
    interval_steps: int = Field(
        default=0,
        ge=0,
        description=(
            "Snapshot every N steps; 0 means every step, bounded only by "
            "`max_calls`. It does NOT fall back to `log_steps` -- that knob "
            "drives a diffusion anomaly LOG, not the snapshot writer."
        ),
    )
    max_calls: int = Field(
        default=8,
        ge=1,
        description=(
            "Cap on snapshot CALLS per (run, tag) -- a call budget, not a step "
            "bound. Counted per tag since #706, and reset per process, so a "
            "resumed run gets a fresh allowance at whatever step it restarts."
        ),
    )
    save_images: bool = Field(default=True, description="Include images.")
    save_json: bool = Field(default=True, description="Include a JSON dump.")
    log_steps: list[Any] = Field(
        default_factory=lambda: [0, 1, 2, 10],
        description=(
            "Step numbers that force the diffusion anomaly-check log "
            "(`_log_diffusion_anomalies`). Named for this block's era before "
            "the snapshot writer had its own cadence; it does not select which "
            "steps are snapshotted -- `interval_steps` and `max_calls` do."
        ),
    )


class LoggingReportCasesConfigSchema(BaseModel):
    """Per-case artifacts kept for the end-of-training report."""

    model_config = _LOG_SUBBLOCK

    enabled: bool = Field(default=True, description="Save per-case report inputs.")
    subdir: str = Field(
        default="report_cases", description="Subdirectory under the run's output dir."
    )


class LoggingConfigSchema(BaseModel):
    """Logging and experiment tracking configuration.

    Defines logging verbosity, output locations, and experiment tracking metadata.

    Example:
        >>> config = LoggingConfigSchema(
        ...     level=LogLevel.INFO,
        ...     silent=False,
        ...     experiment_name="my_experiment",
        ... )
    """

    model_config = {
        "protected_namespaces": (),
        "extra": "ignore",
        "frozen": True,
    }

    # ---- phase 10b: grouped sub-blocks -----------------------------------
    # The flat spellings still LOAD -- `fold_renamed_keys` moves each into its
    # sub-block before validation -- but they are gone from Python. This block
    # is `extra="ignore"`, so a key the fold table forgot does not raise, it
    # VANISHES and the arm runs on the default (#550). The totality pin is
    # `test_renames.py::TestPhase10bFoldTableIsTotal`.
    identity: LoggingIdentityConfigSchema = Field(default_factory=LoggingIdentityConfigSchema)
    sinks: LoggingSinksConfigSchema = Field(default_factory=LoggingSinksConfigSchema)
    intervals: LoggingIntervalsConfigSchema = Field(default_factory=LoggingIntervalsConfigSchema)
    images: LoggingImagesConfigSchema = Field(default_factory=LoggingImagesConfigSchema)
    tracking: LoggingTrackingConfigSchema = Field(default_factory=LoggingTrackingConfigSchema)
    snapshots: LoggingSnapshotsConfigSchema = Field(default_factory=LoggingSnapshotsConfigSchema)
    report_cases: LoggingReportCasesConfigSchema = Field(
        default_factory=LoggingReportCasesConfigSchema
    )
    distributed: LoggingDistributedConfigSchema = Field(
        default_factory=LoggingDistributedConfigSchema
    )
    progress: LoggingProgressConfigSchema = Field(default_factory=LoggingProgressConfigSchema)

    __folded_input_keys__ = folded_input_keys("logging")
    __folded_input_paths__ = folded_input_paths("logging")

    # Retired flat spellings, BOTH postures. `reject_renamed_keys` raises on
    # the ones already driven to zero; `fold_renamed_keys` moves the rest.
    #
    # The reject half is not optional here even while the raise set is small.
    # This block is `extra="ignore"`, so a raise-posture record with nothing
    # to refuse it is not "retired" -- it is SILENTLY DROPPED, which the
    # renames module docstring calls strictly worse than leaving the fold in
    # place: the key stops working AND stops being visible. Promoting a
    # drained record in a block without this validator converts a working
    # fold into exactly that.
    _reject_renamed = model_validator(mode="before")(
        classmethod(reject_renamed_keys("logging"))  # type: ignore[arg-type]
    )
    _fold_renamed = model_validator(mode="before")(
        classmethod(fold_renamed_keys("logging"))  # type: ignore[arg-type]
    )

    # ---- ungrouped: inert, and flatness is the tell ----------------------
    # All nine below are carried by KNOWN_UNCONSUMED in
    # tests/unit/config/test_schema_key_consumption.py -- nothing outside
    # config/schemas/ reads them. `log_gradients` is the odd one: it IS read,
    # but its two group-mates (`log_weights`, `log_activations`) are not, so
    # there is no group left to put it in. `wandb_project` / `wandb_entity`
    # are a second odd pair (#675): "carried by KNOWN_UNCONSUMED" is still
    # true of them for the rg-unreachable measurement, but unlike the other
    # seven they are no longer merely inert -- a non-null declaration RAISES
    # (see `_refuse_deferred_wandb` below), so "nothing reads them" describes
    # the census, not their runtime behaviour.

    # Experiment tracking

    # Tracking service
    # `wandb_project` / `wandb_entity` are unconsumed like their seven
    # siblings above -- but unlike them they do not stay silently inert.
    # `logging:` is extra="ignore", so DELETING these fields would silently
    # downgrade a real declaration to a discarded phantom key, joining the
    # ~420 arms that already declare phantom `enable_wandb` / `project_name`
    # (#675). W&B is deferred by owner decision (2026-08-12); TensorBoard is
    # the only tracking backend (`TrackingService`, `#932`). The fields are
    # retained ONLY so that declaring either one RAISES via
    # `_refuse_deferred_wandb` below, instead of vanishing.
    wandb_project: str | None = Field(
        default=None,
        description=(
            "DEFERRED -- Weights & Biases is not implemented. TensorBoard is "
            "the only tracking backend (see TrackingService). This field is "
            "retained ONLY so that declaring it RAISES: `logging:` is "
            'extra="ignore", so deleting it would silently discard the key '
            "instead of failing the arm."
        ),
    )
    wandb_entity: str | None = Field(
        default=None,
        description="DEFERRED -- see `wandb_project`. Declaring it raises.",
    )

    @field_validator("wandb_project", "wandb_entity")
    @classmethod
    def _refuse_deferred_wandb(cls, value: str | None, info: ValidationInfo) -> None:
        if value is None:
            return None
        raise ValueError(
            f"logging.{info.field_name}: Weights & Biases is not implemented and is "
            "deferred by owner decision (2026-08-12). TensorBoard is the only tracking "
            f"backend. Remove `{info.field_name}` from this arm."
        )

    # Logging intervals
    log_gradients: bool = Field(
        default=False,
        description="Log gradient statistics",
    )
    log_weights: bool = Field(
        default=False,
        description="Log model weight statistics",
    )
    log_activations: bool = Field(
        default=False,
        description="Log activation statistics",
    )

    # Progress bar settings
    progress_bar_enabled: bool = Field(
        default=True,
        description="Enable progress bars during training",
    )
    progress_bar_on_warning: bool = Field(
        default=True,
        description="Show progress bars when the logging level is WARNING or above; `false` turns them off there.",
    )
    progress_bar_no_progress: bool = Field(
        default=False,
        description="Disable progress bars (CLI flag --no-progress)",
    )

    # TensorBoard Configuration

    # Image logging during validation
    log_validation_graphs: bool = Field(
        default=True,
        description=("Log training/validation loss graphs to TensorBoard. " + _UNREAD_1218),
    )

    # Filesystem image saving (validation)
    save_images_per_epoch: int = Field(
        default=4,
        ge=0,
        description=("Number of image samples to save per epoch (0 = disabled). " + _UNREAD_893),
    )

    # ── Report-case recording (feeds the end-of-training figure pipeline) ──

    # ── Per-experiment debug snapshots (cross-paradigm) ────────────────


__all__ = ["LoggingConfigSchema"]
