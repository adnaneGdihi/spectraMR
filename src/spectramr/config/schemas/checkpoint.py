"""Checkpoint management configuration schema."""

from pydantic import BaseModel, Field, model_validator

from spectramr.core.checkpoint_formats import DEFAULT_CHECKPOINT_FORMAT, CheckpointFormat

from .renames import reject_renamed_keys


class CheckpointConfigSchema(BaseModel):
    """Checkpoint saving and loading configuration.

    Defines how and when model checkpoints are saved and loaded. Where they are
    saved is not a knob of this block: training writes every checkpoint to
    ``<training.output_dir>/checkpoints``.

    Example:
        >>> config = CheckpointConfigSchema(
        ...     enabled=True,
        ...     save_interval=1000,
        ... )
    """

    model_config = {
        "protected_namespaces": (),
        "extra": "ignore",
        "frozen": True,
        "populate_by_name": True,
    }

    # Removed keys (removals.py) are refused with the reason, not a bare extra error.
    # pydantic accepts the classmethod at runtime; its protocol types cannot describe one.
    _reject_renamed = model_validator(mode="before")(
        classmethod(reject_renamed_keys("checkpoint"))  # type: ignore[arg-type]
    )

    enabled: bool = Field(
        default=True,
        description="Enable checkpoint saving",
    )
    save_interval: int = Field(
        default=1000,
        ge=1,
        description="Save checkpoint every N steps",
    )
    keep_last_n: int = Field(
        default=5,
        ge=1,
        description="After each save, keep the N highest-step periodic checkpoints "
        "(step_{N:08d}.<format>) in <training.output_dir>/checkpoints and "
        "delete older ones; under DeepSpeed each pruned file's DeepSpeed step tag directory goes "
        "with it, and the tag DeepSpeed's latest file names is kept. The best file and "
        "the best tag are never pruned.",
    )
    load_optimizer_state: bool = Field(
        default=True,
        description="Load optimizer state from checkpoint",
    )
    load_scheduler_state: bool = Field(
        default=True,
        description="Load scheduler state from checkpoint",
    )
    allow_parallel_strategy_change: bool = Field(
        default=False,
        description="Permit resuming a checkpoint written under a different parallel "
        "strategy when either side is fsdp or deepspeed. Off by default: an FSDP "
        "checkpoint carries an unsharded optimizer state keyed by parameter name and a "
        "DeepSpeed one carries sharded per-rank state, neither of which a plain optimizer "
        "(none, dp, ddp) can load, and the reverse is not loadable either. none, dp and ddp "
        "write interchangeable checkpoints and are never refused.",
    )
    allow_world_size_change: bool = Field(
        default=False,
        description="Permit resuming a checkpoint written at a different world size. "
        "Off by default: training.max_iterations is a per-rank bound and the LR "
        "schedule is indexed by iteration, so resuming 1 -> 4 GPUs silently "
        "multiplies the samples seen per iteration, and the data epoch and shuffle "
        "seed (derived from the per-rank loader length) are wrong afterwards.",
    )
    format: CheckpointFormat = Field(
        default=DEFAULT_CHECKPOINT_FORMAT,
        description="File format of every checkpoint the run writes: safetensors "
        "(default; no pickle), pt or pth (the same torch.save container). Read by "
        "checkpointing.store; a resume reads whatever format each file has.",
    )
    async_write: bool = Field(
        default=True,
        description="Write checkpoint files on a background thread. The training thread "
        "still captures the state, copies it to CPU and runs any native (DeepSpeed) save; "
        "the file write and keep_last_n pruning run on the writer, at most one in flight. "
        "What changes when on: a failed periodic, yield or final write stops the run at "
        "the next save or at the end of the loop instead of at the save itself; a failed "
        "best write is reported in best_checkpoint_error at the end of the loop instead "
        "of when it happened; a failed loss-schedule write is logged, as before. The CPU "
        "copy costs one extra copy of the run state in host memory on rank 0. Read by "
        "the training loop, which drains the writer after a wall-clock yield save and "
        "after the final save, before the best-weights restore.",
    )
    produced_by_arm: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Name of the upstream campaign arm that produces this arm's declared "
            "checkpoint(s). When set, check_checkpoint_existence treats a "
            "still-absent, campaign-artefact-rooted checkpoint (under "
            "experiments/active/ or experiments/results/) as a deferred campaign "
            "dependency — info-passing the standalone --strict pre-flight rather "
            "than hard-failing it — because the campaign runner builds the "
            "producer first. The real existence gate still fires at actual "
            "checkpoint-load time. It does NOT defer non-artefact paths or "
            "precomputed data artefacts (marker basis/template tensors)."
        ),
    )


__all__ = ["CheckpointConfigSchema"]
