"""Phase 2: Checkpoint Director

Orchestrates checkpoint save/load for training.

Handles model state, optimizer state, and scheduler state persistence.
"""

import logging
import os
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from spectramr.config.settings import TrainingSettings
from spectramr.core.module_utils import strip_wrapper_prefixes, unwrap_model
from spectramr.core.rng_state import capture_rng_state, restore_rng_state
from spectramr.infrastructure.builders.context import (
    BuilderContext,
    accepts_builder_context,
)
from spectramr.infrastructure.builders.core import DirectorBuilder
from spectramr.infrastructure.distributed.distributed_training import RankUtility
from spectramr.shared.utils.safe_io import atomic_save_torch

if TYPE_CHECKING:  # annotation only — no runtime import, so no cycle
    from spectramr.infrastructure.training.builders.environment import (
        TrainingEnvironment,
    )

logger = logging.getLogger(__name__)


def _native_tag_for(checkpoint_file: Path) -> str:
    """The strategy-native tag corresponding to a generic checkpoint filename.

    The writers are the source of truth. ``save_best`` writes tag ``best``
    beside ``checkpoint_best.pt`` (and its ``best.pt`` alias); ``save`` writes
    tag ``epoch_XXXX_step_YYYYYY`` beside
    ``checkpoint_epoch_XXXX_step_YYYYYY.pt``. So the tag is the stem with the
    ``checkpoint_`` prefix removed, which folds the alias onto the canonical
    name for free.

    The expression this replaces -- ``"best" if "best" in name else "latest"``
    -- was only ever correct on the ``best`` branch, and only ever exercised
    there, because ``load_from`` logged the native result and then discarded
    it. ``latest`` is not a tag DeepSpeed writes: it is the name of the FILE
    recording which tag is newest, so ``load_checkpoint(dir, tag="latest")``
    looks for a directory that cannot exist. Now that a successful native load
    decides whether the generic parse is skipped, a wrong tag is a failed
    restore rather than a discarded log line.
    """
    return checkpoint_file.stem.removeprefix("checkpoint_")


def _publish_best_alias(canonical_path: Path) -> None:
    """Publish a 'best.<ext>' alias next to a 'checkpoint_best.<ext>' file.

    Writers produce ``checkpoint_best.pt`` (asserted by
    ``tests/unit/services/test_early_stopping_best_checkpoint.py``) but
    every consumer — campaign orchestrator, SLURM scripts, inference CLI,
    YAML defaults — historically reads ``best.pt``. To honor both contracts
    without forcing a repo-wide rename, publish a symlink alongside the
    canonical file. Falls back to a hardlink, then a copy, on filesystems
    that don't support symlinks.
    """
    if "checkpoint_best" not in canonical_path.name:
        return
    alias = canonical_path.with_name(canonical_path.name.replace("checkpoint_best", "best", 1))
    if alias == canonical_path:
        return
    try:
        if alias.is_symlink() or alias.exists():
            alias.unlink()
        # Relative target keeps the alias valid if the directory is moved.
        alias.symlink_to(canonical_path.name)
    except (OSError, NotImplementedError):
        try:
            if alias.exists():
                alias.unlink()
            os.link(canonical_path, alias)
        except OSError:
            shutil.copy2(canonical_path, alias)


@dataclass
class CheckpointState:
    """Container for checkpoint data."""

    epoch: int
    global_step: int
    generator_state: dict[str, Any]
    discriminator_state: dict[str, Any] | None
    optimizer_g_state: dict[str, Any]
    optimizer_d_state: dict[str, Any] | None
    scheduler_g_state: dict[str, Any] | None
    scheduler_d_state: dict[str, Any] | None
    metrics: dict[str, float]
    scaler_state: dict[str, Any] | None = None
    counter_state: dict[str, Any] | None = None
    ema_state: dict[str, Any] | None = None  # [FIX] Added EMA state tracker

    def get_total_size_mb(self) -> float:
        """Get approximate checkpoint size in MB.

        Returns:
            Size in megabytes
        """
        total_bytes = 0

        # Estimate state dict sizes
        for state_dict in [
            self.generator_state,
            self.discriminator_state,
            self.optimizer_g_state,
            self.optimizer_d_state,
        ]:
            if state_dict:
                for tensor in state_dict.values():
                    if isinstance(tensor, torch.Tensor):
                        total_bytes += tensor.element_size() * tensor.numel()

        return total_bytes / (1024 * 1024)


class CheckpointDirector(DirectorBuilder[Path | CheckpointState]):
    """Director for checkpoint management.

    Handles saving and loading training state including models, optimizers,
    schedulers, and metrics.

    Example:
        >>> director = CheckpointDirector(config)
        >>> checkpoint_path = (director
        ...     .with_checkpoint_dir("experiments/exp1/checkpoints")
        ...     .with_pipeline(pipeline)
        ...     .with_epoch(10)
        ...     .with_metrics({"loss": 0.5, "psnr": 30.2})
        ...     .validate()
        ...     .save())

        >>> # Load checkpoint
        >>> director2 = CheckpointDirector(config)
        >>> pipeline = (director2
        ...     .with_pipeline(pipeline)
        ...     .load_from("experiments/exp1/checkpoints/checkpoint_epoch_10.pt"))
    """

    @accepts_builder_context
    def __init__(self, ctx: BuilderContext) -> None:
        """Initialize checkpoint director.

        Args:
            config: Training configuration
        """
        config: TrainingSettings = ctx.config
        super().__init__()
        self._config = config
        self._checkpoint_dir: Path | None = None
        self._pipeline: TrainingEnvironment | None = None
        self._epoch: int = 0
        self._global_step: int = 0
        self._metrics: dict[str, float] = {}
        self._checkpoint_path: Path | None = None
        self._scaler: Any | None = None
        self._counter_state: dict[str, Any] | None = None
        self._strategy: Any | None = None
        self._parallel: Any | None = None

        logger.info("CheckpointDirector initialized")

    def with_checkpoint_dir(self, directory: str) -> "CheckpointDirector":
        """Set checkpoint directory.

        Args:
            directory: Directory for saving checkpoints

        Returns:
            self for chaining
        """
        self._checkpoint_dir = Path(directory)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return self

    def with_pipeline(self, pipeline: "TrainingEnvironment") -> "CheckpointDirector":
        """Set training pipeline.

        Args:
            pipeline: Training pipeline to checkpoint

        Returns:
            self for chaining
        """
        self._pipeline = pipeline
        return self

    def with_strategy(self, strategy: Any) -> "CheckpointDirector":
        """Set the training strategy so its OWNED learnable state is checkpointed.

        A strategy is not an ``nn.Module``; modules/parameters it builds itself
        (e.g. ``adaptive_sfc_hssc``'s ``BeltramiSFCBlock``, ``spin_sde``'s
        diffusion ``nn.Parameter``, ib_vf critics, ``mri_slam`` deltas) live on
        the strategy, not on ``pipeline.generator``. Without this they are NOT
        saved, so a resume re-inits them randomly while their optimizer state is
        restored by-index — a silent corrupt resume. When set, the director saves
        ``strategy.strategy_state_dict()`` under the ``strategy_state`` key and
        restores it on ``load_from``. See
        ``docs/strategy_owned_state_checkpointing_design_2026_06.rst``.

        Args:
            strategy: BaseTrainingStrategy instance (or ``None`` to disable).

        Returns:
            self for chaining
        """
        self._strategy = strategy
        return self

    def with_parallel_runtime(self, parallel: Any) -> "CheckpointDirector":
        """Set the resolved ``ParallelRuntime`` so sharded strategies save correctly.

        Two things change when this is set to an FSDP/DeepSpeed runtime:

        * ``state_dict()`` is read inside the adapter's gather context, so what
          lands on disk is the FULL model rather than this rank's shard;
        * the strategy's native artifact (DeepSpeed's tag directory) is written
          in addition to, or instead of, the generic ``torch.save`` payload.

        Both of those are **collectives**, so the caller must have entered this
        code path on every rank -- see ``ParallelRuntime.checkpoints_require_all_ranks``.
        Leaving it unset yields ``DefaultCheckpointAdapter``, which is exactly
        the pre-existing behaviour.

        Args:
            parallel: ``ParallelRuntime`` (or ``None`` for single-process).

        Returns:
            self for chaining
        """
        self._parallel = parallel
        return self

    @property
    def _adapter(self) -> Any:
        from spectramr.infrastructure.distributed.checkpoint_adapters import (
            resolve_checkpoint_adapter,
        )

        # getattr, not self._parallel: tests in this area construct the director
        # via __new__ to skip the heavy __init__ (the same reason _strategy is
        # read defensively below). A hard attribute access here turns every one
        # of them into an AttributeError inside the save path.
        return resolve_checkpoint_adapter(getattr(self, "_parallel", None))

    def _resolve_scaler(self) -> Any | None:
        """The scaler that ACTUALLY scaled this run's losses.

        ``OptimizationBuilder.build_grad_scaler`` used to hand the pipeline a
        plain ``GradScaler("cuda")`` that nothing ever called: the live one is
        built by ``MixedPrecisionIntegrationHelper`` and is a ``NativeScaler``
        or ``ComplexGradScaler`` depending on the arm. So the checkpoint stored
        a scale factor and growth-tracker of an object that had never seen a
        gradient, while the real dynamic scale -- the thing an fp16 resume needs
        in order not to re-converge its scale from 65536 -- was never written.

        A fabricated field is worse than a missing one: ``scaler_state`` was
        present, well-formed, and restored on resume, so nothing looked wrong.
        """
        helper = getattr(getattr(self, "_strategy", None), "amp_helper", None)
        live = getattr(helper, "scaler", None)
        if live is not None:
            return live
        return getattr(self, "_scaler", None)

    def _collect_models(self) -> dict[str, Any]:
        """The wrapped modules, by canonical key, for the adapter.

        Deliberately NOT unwrapped: DeepSpeed's ``save_checkpoint`` lives on the
        engine and FSDP's gather context needs the FSDP root, so handing either
        the bare module underneath would be a silent no-op.
        """
        models: dict[str, Any] = {}
        if self._pipeline is None:
            return models
        for key in ("generator", "discriminator"):
            module = getattr(self._pipeline, key, None)
            if module is not None:
                models[key] = module
        return models

    @staticmethod
    def _is_writer_rank() -> bool:
        """Whether THIS rank should touch the filesystem.

        Distinct from "should this rank run at all": under a collective
        strategy every rank executes the save, gathers, and then only rank 0
        writes. With ``rank0_only=True`` the others hold empty dicts, so
        letting them write would produce a directory of near-empty files that
        look like valid checkpoints.
        """
        from spectramr.infrastructure.distributed.distributed_training import RankUtility

        return RankUtility.is_main_rank()

    def _maybe_add_strategy_state(self, checkpoint_data: dict[str, Any]) -> None:
        """Attach strategy-owned learnable state to ``checkpoint_data`` (in place)
        when a strategy is set and actually owns state. Stored under
        ``strategy_state`` (a sibling key — NOT merged into ``generator`` — so
        strict inference loads of the generator are unaffected)."""
        # getattr (not self._strategy): some call sites / tests build the director
        # via __new__ and never run __init__, so the attribute may be absent.
        strategy = getattr(self, "_strategy", None)
        if strategy is None or not hasattr(strategy, "strategy_state_dict"):
            return
        state = strategy.strategy_state_dict()
        if state.get("modules") or state.get("params"):
            checkpoint_data["strategy_state"] = state

    def with_epoch(self, epoch: int) -> "CheckpointDirector":
        """Set current epoch.

        Args:
            epoch: Current training epoch

        Returns:
            self for chaining
        """
        if epoch < 0:
            raise ValueError(f"Invalid epoch: {epoch}")
        self._epoch = epoch
        return self

    def with_global_step(self, step: int) -> "CheckpointDirector":
        """Set global training step.

        Args:
            step: Current training step

        Returns:
            self for chaining
        """
        if step < 0:
            raise ValueError(f"Invalid step: {step}")
        self._global_step = step
        return self

    def with_metrics(self, metrics: dict[str, float]) -> "CheckpointDirector":
        """Add training metrics to checkpoint.

        Args:
            metrics: Dictionary of metric names and values

        Returns:
            self for chaining
        """
        self._metrics.update(metrics)
        return self

    def with_scaler(self, scaler: Any) -> "CheckpointDirector":
        """Set AMP GradScaler for state persistence.

        Args:
            scaler: torch.cuda.amp.GradScaler instance (or None)

        Returns:
            self for chaining
        """
        self._scaler = scaler
        return self

    def with_counter_state(self, counter_state: dict[str, Any]) -> "CheckpointDirector":
        """Set iteration counter state for deterministic resume.

        Args:
            counter_state: Dictionary from IterationCounterService.get_state()
                           or a lightweight {"current_step": ..., "current_epoch": ...} dict.

        Returns:
            self for chaining
        """
        self._counter_state = counter_state
        return self

    def validate(self) -> "CheckpointDirector":
        """Validate director configuration.

        Returns:
            self for chaining

        Raises:
            ValueError: If invalid configuration
        """
        super().validate()

        if not self._checkpoint_dir:
            raise ValueError("Checkpoint directory must be specified")

        if not self._pipeline:
            raise ValueError("Pipeline must be specified")

        return self

    def build(self) -> CheckpointState:
        """Build checkpoint state (without saving).

        Returns:
            CheckpointState object

        Raises:
            ValueError: If validation fails
        """
        self.validate()

        # Unwrap before reading state_dict(). torch.compile prefixes every key
        # with "_orig_mod.", DP/DDP with "module.", FSDP with
        # "_fsdp_wrapped_module.". This is the LIVE checkpoint writer, so without
        # this a `compile_model: true` or DDP run produced checkpoints that no
        # inference path could read: infer/predict/evaluate all build a BARE model
        # and load with strict=True, and under strict=False nothing matches,
        # nothing loads, and the load reports success (#619 F3).
        #
        # ModelEma is unwrapped for the same reason -- it holds a deepcopy under
        # `.module`, so its keys were ALWAYS prefixed, which is why
        # gan_inference_strategy's EMA load could never have worked.
        #
        # The gather context is what makes a SHARDED strategy's state_dict() the
        # full model instead of this rank's slice. It is a collective, so every
        # rank must reach it; `unwrap_model` is deliberately NOT applied to the
        # module handed to the adapter, because FSDP.state_dict_type needs the
        # FSDP root itself, not the module underneath it.
        adapter = self._adapter
        generator = self._pipeline.generator
        discriminator = self._pipeline.discriminator
        with adapter.gather_full_state_dict(generator):
            generator_state = unwrap_model(generator).state_dict()
        if discriminator is not None:
            with adapter.gather_full_state_dict(discriminator):
                discriminator_state = unwrap_model(discriminator).state_dict()
        else:
            discriminator_state = None

        checkpoint_state = CheckpointState(
            epoch=self._epoch,
            global_step=self._global_step,
            generator_state=generator_state,
            discriminator_state=discriminator_state,
            optimizer_g_state=self._pipeline.optimizer_g.state_dict(),
            optimizer_d_state=(
                self._pipeline.optimizer_d.state_dict() if self._pipeline.optimizer_d else None
            ),
            scheduler_g_state=(
                self._pipeline.scheduler_g.state_dict() if self._pipeline.scheduler_g else None
            ),
            scheduler_d_state=(
                self._pipeline.scheduler_d.state_dict() if self._pipeline.scheduler_d else None
            ),
            metrics=self._metrics.copy(),
            scaler_state=(scaler.state_dict() if (scaler := self._resolve_scaler()) else None),
            counter_state=self._counter_state,
            ema_state=(
                # NOT unwrap_model(...).state_dict(): unwrap peels ModelEma's
                # own ``.module``, so the override that injects the warmup
                # counter never runs and every resume restarts the decay ramp
                # at 0. Weight keys are identical either way (#2172).
                self._pipeline.ema.shadow_state_dict()
                if hasattr(self._pipeline, "ema") and self._pipeline.ema is not None
                else None
            ),
        )

        self._product = None
        return checkpoint_state

    def save(self) -> Path:
        """Build and save checkpoint.

        Returns:
            Path to saved checkpoint file

        Raises:
            ValueError: If validation fails
        """
        self.validate()

        try:
            # Create checkpoint state
            checkpoint_state = self.build()

            # ✅ SSOT: Checkpoint naming includes iteration if provided via .with_global_step()
            # This enables iteration-based checkpoint recovery (user request: iterations not epochs)
            # Format: checkpoint_epoch_XXXX_step_YYYYYY.pt if step available
            # Format: checkpoint_epoch_XXXX.pt if step not available (legacy)
            if self._global_step is not None:
                checkpoint_filename = (
                    f"checkpoint_epoch_{self._epoch:04d}_step_{self._global_step:06d}.pt"
                )
            else:
                checkpoint_filename = f"checkpoint_epoch_{self._epoch:04d}.pt"
            checkpoint_path = self._checkpoint_dir / checkpoint_filename

            # Prepare checkpoint data
            checkpoint_data = {
                "epoch": checkpoint_state.epoch,
                "global_step": checkpoint_state.global_step,
                "generator": checkpoint_state.generator_state,
                "optimizer_g": checkpoint_state.optimizer_g_state,
                "metrics": checkpoint_state.metrics,
            }

            # Add optional states
            if checkpoint_state.discriminator_state:
                checkpoint_data["discriminator"] = checkpoint_state.discriminator_state
            if checkpoint_state.optimizer_d_state:
                checkpoint_data["optimizer_d"] = checkpoint_state.optimizer_d_state
            if checkpoint_state.scheduler_g_state:
                checkpoint_data["scheduler_g"] = checkpoint_state.scheduler_g_state
            if checkpoint_state.scheduler_d_state:
                checkpoint_data["scheduler_d"] = checkpoint_state.scheduler_d_state
            if checkpoint_state.scaler_state:
                checkpoint_data["scaler_state"] = checkpoint_state.scaler_state
            if checkpoint_state.counter_state:
                checkpoint_data["counter_state"] = checkpoint_state.counter_state
            if checkpoint_state.ema_state:
                checkpoint_data["ema_state"] = checkpoint_state.ema_state

            # Strategy-owned learnable state (sfc heads, spin_sde diffusion param,
            # ib_vf critics, ...) — see with_strategy / section R design doc.
            self._maybe_add_strategy_state(checkpoint_data)

            # The stochastic half of the run. Without it a requeued 120 h chain
            # restores the right weights at the right iteration and then draws a
            # DIFFERENT sequence of diffusion timesteps, masks and dropout from
            # the one the interrupted job would have drawn — a divergence that
            # trains and reports success. `CheckpointService` has captured this
            # since 2026-05 but is only this class's exception fallback, so on
            # every normal save it was never called (non-negotiable 16).
            #
            # No collective here: under plain DDP `may_checkpoint` lets ONLY
            # rank 0 into this method, so gathering the other ranks' streams
            # would hang the job. Rank 0's is what is replayed.
            checkpoint_data["rng_state"] = capture_rng_state()

            # Strategy-native artifact first (DeepSpeed's sharded tag directory).
            # COLLECTIVE: every rank must call it, which is why the training loop
            # gates on `is_main_rank() or checkpoints_require_all_ranks`.
            self._adapter.save_native(
                models=self._collect_models(),
                checkpoint_dir=self._checkpoint_dir,
                tag=f"epoch_{self._epoch:04d}_step_{self._global_step:06d}",
                client_state={
                    "epoch": checkpoint_state.epoch,
                    "global_step": checkpoint_state.global_step,
                    "metrics": checkpoint_state.metrics,
                },
            )

            if not self._is_writer_rank():
                # Gathered above with rank0_only=True, so this rank holds empty
                # tensors. Writing them would litter the run directory with
                # files that pass every existence check and load as noise.
                self._checkpoint_path = checkpoint_path
                self._product = checkpoint_path
                return checkpoint_path

            # Save checkpoint atomically: a truncated checkpoint passes every
            # existence check and only fails at load time (#1352).
            atomic_save_torch(checkpoint_data, checkpoint_path)

            size_mb = checkpoint_state.get_total_size_mb()
            logger.info(f"Checkpoint saved: {checkpoint_path} ({size_mb:.2f} MB)")

            self._checkpoint_path = checkpoint_path
            self._product = checkpoint_path

            return checkpoint_path

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise RuntimeError(f"Failed to save checkpoint: {e}") from e

    def save_best(
        self,
        metric_name: str = "",
        metric_value: float = 0.0,
    ) -> Path:
        """Save checkpoint tagged as the best model.

        Overwrites the previous ``checkpoint_best.pt`` so there is always
        exactly one best checkpoint on disk.

        Args:
            metric_name: Name of the monitored metric (e.g. ``val_psnr``).
            metric_value: Value of the metric at the time of saving.

        Returns:
            Path to ``checkpoint_best.pt``.

        Raises:
            ValueError: If director is not fully configured.
            RuntimeError: If saving fails.
        """
        self.validate()

        try:
            checkpoint_state = self.build()
            checkpoint_path = self._checkpoint_dir / "checkpoint_best.pt"

            checkpoint_data: dict[str, Any] = {
                "epoch": checkpoint_state.epoch,
                "global_step": checkpoint_state.global_step,
                "generator": checkpoint_state.generator_state,
                "optimizer_g": checkpoint_state.optimizer_g_state,
                "metrics": checkpoint_state.metrics,
                "is_best": True,
                "best_metric_name": metric_name,
                "best_metric_value": metric_value,
            }

            if checkpoint_state.discriminator_state:
                checkpoint_data["discriminator"] = checkpoint_state.discriminator_state
            if checkpoint_state.optimizer_d_state:
                checkpoint_data["optimizer_d"] = checkpoint_state.optimizer_d_state
            if checkpoint_state.scheduler_g_state:
                checkpoint_data["scheduler_g"] = checkpoint_state.scheduler_g_state
            if checkpoint_state.scheduler_d_state:
                checkpoint_data["scheduler_d"] = checkpoint_state.scheduler_d_state
            if checkpoint_state.scaler_state:
                checkpoint_data["scaler_state"] = checkpoint_state.scaler_state
            if checkpoint_state.counter_state:
                checkpoint_data["counter_state"] = checkpoint_state.counter_state
            if checkpoint_state.ema_state:
                checkpoint_data["ema_state"] = checkpoint_state.ema_state

            self._maybe_add_strategy_state(checkpoint_data)

            # COLLECTIVE (see save()). For DeepSpeed this also publishes the
            # consolidated single-file copy that discover_best_checkpoint,
            # campaign evaluation and `spectramr infer` actually open.
            self._adapter.save_native(
                models=self._collect_models(),
                checkpoint_dir=self._checkpoint_dir,
                tag="best",
                client_state={
                    "epoch": checkpoint_state.epoch,
                    "global_step": checkpoint_state.global_step,
                    "metrics": checkpoint_state.metrics,
                    "best_metric_name": metric_name,
                    "best_metric_value": metric_value,
                },
            )

            if not self._is_writer_rank():
                self._checkpoint_path = checkpoint_path
                return checkpoint_path

            if self._adapter.writes_native_artifact and checkpoint_path.exists():
                # The adapter's consolidated write already produced this file;
                # overwriting it with the generic payload would replace gathered
                # ZeRO-3 weights with the engine's local partition.
                _publish_best_alias(checkpoint_path)
                self._checkpoint_path = checkpoint_path
                return checkpoint_path

            atomic_save_torch(checkpoint_data, checkpoint_path)
            _publish_best_alias(checkpoint_path)

            size_mb = checkpoint_state.get_total_size_mb()
            logger.info(
                f"Best checkpoint saved: {checkpoint_path} ({size_mb:.2f} MB) "
                f"[{metric_name}={metric_value:.6f}]"
            )

            self._checkpoint_path = checkpoint_path
            return checkpoint_path

        except Exception as e:
            logger.error(f"Failed to save best checkpoint: {e}")
            raise RuntimeError(f"Failed to save best checkpoint: {e}") from e

    def load_from(self, checkpoint_path: str) -> bool:
        """Load checkpoint into pipeline.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            True if load successful

        Raises:
            FileNotFoundError: If checkpoint not found
        """
        checkpoint_file = Path(checkpoint_path)
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        if not self._pipeline:
            raise ValueError("Pipeline must be set before loading")

        try:
            logger.info(f"Loading checkpoint: {checkpoint_file}")

            # A strategy that writes its own artifact must also read it: the
            # generic file carries model weights but NOT the sharded ZeRO
            # optimizer state, so restoring from it alone would resume with
            # freshly-initialised moments while reporting a successful resume.
            # COLLECTIVE -- every rank calls, same as the save side.
            adapter = self._adapter
            native_state: dict[str, Any] | None = None
            if adapter.writes_native_artifact:
                native_state = adapter.load_native(
                    models=self._collect_models(),
                    checkpoint_dir=checkpoint_file.parent,
                    tag=_native_tag_for(checkpoint_file),
                )
                if native_state is not None:
                    # `is not None`, not truthiness: load_native returns None when
                    # there was no engine to load into, and {} when the engines
                    # loaded from a tag that carried no client state. Those mean
                    # opposite things and the second one is a success.
                    logger.info(
                        "[%s] restored sharded state (epoch=%s step=%s)",
                        adapter.name,
                        native_state.get("epoch"),
                        native_state.get("global_step"),
                    )

            # Load checkpoint data
            # weights_only=False needed: checkpoints contain non-tensor metadata
            # (dicts, ints, floats for epoch/step/metrics/counter_state/scaler_state)
            checkpoint_data = torch.load(
                checkpoint_file,
                map_location=self._pipeline.device,
                weights_only=False,
            )

            # The mirror image of save_best's writes_native_artifact early
            # return. When the strategy wrote its own artifact, THIS FILE is the
            # consolidated export -- a bare parameter-keyed state_dict -- and not
            # the generic payload, which save_best deliberately never wrote.
            # Parsing it as one raised KeyError('generator') and threw away the
            # best weights of a completed run.
            #
            # Discriminate on the PAYLOAD, not on the adapter alone: `save`
            # (unlike `save_best`) writes both artifacts, and its generic half
            # carries ema/scaler/counter/strategy state that the tag directory
            # does not. Keying off the adapter would silently drop all of it on
            # every resume.
            if "generator" not in checkpoint_data:
                if native_state is None:
                    raise RuntimeError(
                        f"{checkpoint_file} carries no 'generator' key, and no "
                        "strategy-native artifact was loaded to supply the "
                        f"weights (adapter={adapter.name!r}). A consolidated "
                        "DeepSpeed export can only be restored by the strategy "
                        "that wrote it: attach the run's ParallelRuntime via "
                        "with_parallel_runtime() before calling load_from()."
                    )
                # Weights and sharded optimizer state came from the tag
                # directory. The consolidated file carries no metadata at all,
                # so client_state is the only source for it -- which makes an
                # absent key unknown, never zero.
                #
                # Defaulting these to 0 restored the WEIGHTS of a week-long run
                # while resetting its schedule position, then logged
                # "epoch=0, step=0" as though measured and returned True. Both
                # this branch and the `native_state is None` raise above end with
                # counters that are unknown; substituting 0 is the quiet half of
                # that pair, and it is undetectable downstream because 0 is also
                # a legitimate value (a genuine epoch-0 checkpoint). The
                # LR-schedule position, the early-stopping counter and any
                # step-based validation gate all read these.
                missing = [k for k in ("epoch", "global_step") if k not in native_state]
                if missing:
                    raise RuntimeError(
                        f"{checkpoint_file}: {adapter.name} restored sharded weights "
                        f"from tag {_native_tag_for(checkpoint_file)!r}, but its client "
                        f"state is missing {missing!r}, so the run position is unknown. "
                        "Both writers in this class record those keys on every save "
                        "(see `save` and `save_best`), so a tag without them was written "
                        "by another tool or an older version. Resuming would restart the "
                        "LR schedule and the early-stopping counter from zero while "
                        "reporting a successful restore."
                    )
                self._epoch = native_state["epoch"]
                self._global_step = native_state["global_step"]
                # Unlike the two above, these absences are honest rather than
                # fabricated: {} means "no metrics were recorded" and None means
                # "no counter state", and neither is mistakable for a measured
                # value. The guard deliberately does not extend to them.
                self._metrics = native_state.get("metrics", {})
                self._counter_state = native_state.get("counter_state")
                logger.info(
                    "Checkpoint loaded from %s native artifact: epoch=%s, step=%s",
                    adapter.name,
                    self._epoch,
                    self._global_step,
                )
                return True

            # Load model states.
            #
            # Unwrap the live module AND strip prefixes off the stored keys. Both
            # halves are needed: the first makes the load work when the running
            # model is wrapped (compile/DDP/FSDP) while the checkpoint is clean;
            # the second makes it work for checkpoints written BEFORE the save
            # side unwrapped, whose keys still carry "_orig_mod."/"module.".
            # Without the second half, every checkpoint produced by an existing
            # compiled or DDP run becomes unloadable the moment the save side is
            # fixed.
            unwrap_model(self._pipeline.generator).load_state_dict(
                strip_wrapper_prefixes(checkpoint_data["generator"])
            )
            if "discriminator" in checkpoint_data and self._pipeline.discriminator:
                unwrap_model(self._pipeline.discriminator).load_state_dict(
                    strip_wrapper_prefixes(checkpoint_data["discriminator"])
                )

            # Load optimizer states
            self._pipeline.optimizer_g.load_state_dict(checkpoint_data["optimizer_g"])
            if "optimizer_d" in checkpoint_data and self._pipeline.optimizer_d:
                self._pipeline.optimizer_d.load_state_dict(checkpoint_data["optimizer_d"])

            # [FIX] Load EMA state securely
            if (
                "ema_state" in checkpoint_data
                and hasattr(self._pipeline, "ema")
                and self._pipeline.ema is not None
            ):
                # Restores the warmup counter as well as the weights; strips
                # prefixes internally, so pre-fix checkpoints still load.
                self._pipeline.ema.load_shadow_state_dict(checkpoint_data["ema_state"])
                logger.info("Restored EMA shadow weights securely via director")

            # Load scheduler states (optional, gated by the advertised
            # checkpoint.load_scheduler_state knob — pitfall #15). Default True
            # preserves the previous unconditional-restore behavior.
            checkpoint_cfg = getattr(self._config, "checkpoint", None)
            load_scheduler = getattr(checkpoint_cfg, "load_scheduler_state", True)
            if load_scheduler:
                if "scheduler_g" in checkpoint_data and self._pipeline.scheduler_g:
                    self._pipeline.scheduler_g.load_state_dict(checkpoint_data["scheduler_g"])
                if "scheduler_d" in checkpoint_data and self._pipeline.scheduler_d:
                    self._pipeline.scheduler_d.load_state_dict(checkpoint_data["scheduler_d"])

            # Restore AMP GradScaler state into the scaler that will actually be
            # USED (see _resolve_scaler) -- restoring into pipeline.scaler put the
            # saved dynamic scale on an object no backward pass ever touched.
            live_scaler = self._resolve_scaler()
            if "scaler_state" in checkpoint_data and live_scaler is not None:
                live_scaler.load_state_dict(checkpoint_data["scaler_state"])
                logger.info("AMP GradScaler state restored from checkpoint")

            # Restore strategy-owned learnable state (the counterpart to
            # _maybe_add_strategy_state). No-op for older checkpoints lacking the
            # key, or when no strategy is attached.
            strategy = getattr(self, "_strategy", None)
            if (
                "strategy_state" in checkpoint_data
                and strategy is not None
                and hasattr(strategy, "load_strategy_state_dict")
            ):
                strategy.load_strategy_state_dict(checkpoint_data["strategy_state"])
                logger.info("Restored strategy-owned state from checkpoint")

            # Counterpart to the capture in `save`. Rank-gated: the file holds
            # ONE process's streams (rank 0's, the only rank that reaches the
            # save under plain DDP), so restoring it on every rank would give
            # them all an identical augmentation sequence and quietly divide the
            # run's stochastic diversity by the world size — the exact outcome
            # train.py's rank-offset seeding exists to prevent. Non-main ranks
            # keep the rank-distinct stream they were seeded with.
            rng_state = checkpoint_data.get("rng_state")
            if rng_state and RankUtility.is_main_rank():
                restore_rng_state(rng_state)
                logger.info("Restored RNG state (torch + cuda + numpy + python)")
            elif rng_state:
                logger.info(
                    "[RNG] rank %d keeps its own stream; the checkpoint carries "
                    "rank 0's only, so this rank continues rather than replays.",
                    RankUtility.get_rank(),
                )

            # Extract epoch, step, metrics, and counter state
            self._epoch = checkpoint_data.get("epoch", 0)
            self._global_step = checkpoint_data.get("global_step", 0)
            self._metrics = checkpoint_data.get("metrics", {})
            self._counter_state = checkpoint_data.get("counter_state")

            logger.info(f"Checkpoint loaded: epoch={self._epoch}, step={self._global_step}")
            if self._counter_state:
                logger.info(
                    f"Counter state restored: step={self._counter_state.get('current_step')}, "
                    f"epoch={self._counter_state.get('current_epoch')}"
                )

            return True

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise RuntimeError(f"Failed to load checkpoint: {e}") from e

    # `_resolve_monitor_metric` was deleted here, not wired. It had zero callers
    # and the WRONG precedence: it tried `metrics.best_metric_name` before
    # `early_stopping.metric`, so wiring it -- the obvious repair for an inert
    # knob -- would have silently flipped the selection metric on the 75 arms
    # where the two disagree, and defaulted the rest to a magic "loss" that is
    # not a registered metric name. The one owner of checkpoint selection is
    # `early_stopping.metric`, resolved by `EarlyStoppingService` and consumed
    # at `training_loop.py`'s `save_best` (non-negotiable 17).

    def cleanup_old_checkpoints(self, keep_last_n: int = 3) -> int:
        """Deprecated. Remove old checkpoints, keeping only the last n.

        **Nothing calls this**, and it is not the retention policy the corpus
        configures. `checkpoint.keep_last_n` / `keep_best_n` are read by
        :meth:`CheckpointService._apply_retention_policy`, which is the one
        owner (non-negotiable 17) -- this method takes `keep_last_n` as an
        argument and so cannot see either knob. Wiring it here would create a
        second retention policy rather than fix the first: it keeps by filename
        recency only, where the service protects the most-recent `keep_last_n`
        **union** the best `keep_best_n` by recorded metric, so a run switched
        to this one would start deleting its own best checkpoints.

        Why it is deprecated rather than deleted: the director is the primary
        writer on the training path, and a retention method on it reads as the
        one that runs. That both retention paths are currently dead is #1972 /
        #1973, and whichever fix those take should have one owner at the end of
        it, not three.

        Args:
            keep_last_n: Number of recent checkpoints to keep. **`0` removes
                nothing**, because `checkpoints[:-0]` is `checkpoints[:0]`, the
                empty list -- so "keep zero" silently means "keep all". The
                service's policy has no such edge: it builds a protected set
                rather than slicing a complement.

        Returns:
            Number of checkpoints removed
        """
        warnings.warn(
            "CheckpointDirector.cleanup_old_checkpoints is deprecated and has no "
            "callers. Checkpoint retention is owned by "
            "CheckpointService._apply_retention_policy, which reads "
            "checkpoint.keep_last_n and checkpoint.keep_best_n; this method reads "
            "neither and keeps by filename recency alone. See #1972 and #1973 for "
            "why neither path currently prunes on a normal run.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not self._checkpoint_dir:
            return 0

        checkpoints = sorted(self._checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        removed_count = 0

        for checkpoint_path in checkpoints[:-keep_last_n]:
            try:
                checkpoint_path.unlink()
                logger.info(f"Removed old checkpoint: {checkpoint_path}")
                removed_count += 1
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint {checkpoint_path}: {e}")

        return removed_count


__all__ = ["CheckpointDirector", "CheckpointState"]
