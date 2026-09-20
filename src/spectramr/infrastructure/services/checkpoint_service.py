import json
import logging
import os
import shutil
import tempfile
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from spectramr.config.schemas.checkpoint import CheckpointConfigSchema
from spectramr.core.module_utils import strip_wrapper_prefixes, unwrap_model
from spectramr.core.rng_state import (
    RNG_STATE_SAFE_GLOBALS,
    capture_rng_state,
    restore_rng_state,
)
from spectramr.domain.interfaces.checkpoint_service_interface import ICheckpointService
from spectramr.infrastructure.services.iteration_counter_service import (
    IterationCounterService,
)

logger = logging.getLogger(__name__)


def _publish_best_alias(canonical_path: Path) -> None:
    """Publish a 'best.<ext>' alias next to a 'checkpoint_best.<ext>' file.

    Writers produce ``checkpoint_best.<fmt>`` (asserted by
    ``tests/unit/services/test_early_stopping_best_checkpoint.py``) but
    every consumer — campaign orchestrator, SLURM scripts, inference CLI,
    YAML defaults — historically reads ``best.<fmt>``. To honor both
    contracts without a repo-wide rename, publish a symlink alongside the
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
        alias.symlink_to(canonical_path.name)
    except (OSError, NotImplementedError):
        try:
            if alias.exists():
                alias.unlink()
            os.link(canonical_path, alias)
        except OSError:
            shutil.copy2(canonical_path, alias)


# Extensions the two checkpoint writers use, in resolution priority. ``.pt`` is
# the CheckpointDirector default; ``.safetensors`` is the CheckpointService
# default. ``.pth`` is accepted for legacy configs.
_CHECKPOINT_EXTS = ("pt", "safetensors", "pth")


def discover_best_checkpoint(checkpoints_dir: Path | str) -> Path | None:
    """Find the best (or latest) checkpoint under ``checkpoints_dir``.

    Probes the ACTUAL conventions the two writers emit, in priority order:

    1. ``best.<ext>`` — the alias both writers publish next to their best file.
    2. ``checkpoint_best.<ext>`` — the canonical best file if the alias is absent.
    3. the newest ``checkpoint_step_*.<ext>`` / ``checkpoint_epoch_*.<ext>``.

    The previous discovery hard-coded ``best.pt`` then globbed ``model_iter_*.pt``
    — a name NEITHER writer produces, so the fallback was dead and safetensors
    runs (``best.safetensors``) were never found. Returns ``None`` if nothing
    matches.
    """
    ckpt_dir = Path(checkpoints_dir)
    if not ckpt_dir.exists():
        return None

    for ext in _CHECKPOINT_EXTS:
        alias = ckpt_dir / f"best.{ext}"
        if alias.exists():
            return alias
    for ext in _CHECKPOINT_EXTS:
        canonical = ckpt_dir / f"checkpoint_best.{ext}"
        if canonical.exists():
            return canonical

    candidates: list[Path] = []
    for ext in _CHECKPOINT_EXTS:
        candidates.extend(ckpt_dir.glob(f"checkpoint_step_*.{ext}"))
        candidates.extend(ckpt_dir.glob(f"checkpoint_epoch_*.{ext}"))
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    return None


@dataclass
class CheckpointMetadata:
    """Metadata about a saved checkpoint."""

    epoch: int
    step: int
    timestamp: str
    loss: float
    metric_name: str | None = None
    metric_value: float | None = None
    is_best: bool = True
    is_last: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class CheckpointService(ICheckpointService):
    """Robust checkpoint service with atomic writes and retention policy.

    Features:
    - Atomic writes using temp files and rename
    - Configurable retention policy (keep_last_n, keep_best_n)
    - Metadata tracking (epoch, step, timestamp, metrics)
    - Multiple formats supported (safetensors, pth)
    - Thread-safe operations with simple locking

    .. mermaid::

        sequenceDiagram
            participant Client
            participant Service
            participant Counter
            participant OS
            participant Metadata

            Client->>Service: save_checkpoint(model, optimizer...)
            opt Counter
                Service->>Counter: get_state()
            end
            Service->>OS: write_temp_file(state)
            Service->>OS: atomic_rename(temp, final)
            Service->>Metadata: update_index()
            Service->>Service: apply_retention_policy()
            Service-->>Client: file_path
    """

    def __init__(self, config: CheckpointConfigSchema, logger_service=None):
        """Initialize checkpoint service.

        Args:
            config: CheckpointConfigSchema with checkpoint settings
            logger_service: Optional logging service for detailed logging
        """
        self.config = config
        self.logger_service = logger_service
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # Use torch.save format for full checkpoints (includes optimizer state, etc.)
        # safetensors is better for model weights only
        # Allow format override from config, default to safetensors
        self.format = config.format
        self._lock = threading.Lock()  # Thread safety lock
        self._metadata: dict[str, CheckpointMetadata] = {}
        self._load_metadata_index()

    def should_save_checkpoint(self, counter: IterationCounterService) -> bool:
        """Check if checkpoint should be saved based on interval.

        Queries the IterationCounterService to determine if the checkpoint save
        interval has been reached and we haven't exceeded max_iterations.

        Args:
            counter: IterationCounterService instance (SSOT for step tracking)

        Returns:
            True if checkpoint interval reached and within limits, False otherwise
        """
        return counter.should_checkpoint(self.config.save_interval)

    def _load_metadata_index(self) -> None:
        """Load metadata index from manifest file."""
        manifest_path = self.checkpoint_dir / "manifest.json"
        if manifest_path.exists():
            try:
                with open(manifest_path) as f:
                    data = json.load(f)
                    for name, meta in data.items():
                        self._metadata[name] = CheckpointMetadata(**meta)
                logger.info(f"Loaded {len(self._metadata)} checkpoints from manifest")
            except Exception as e:
                logger.warning(f"Failed to load manifest: {e}")

    def _save_metadata_index(self) -> None:
        """Save metadata index to manifest file.

        [FORENSIC FIX] Uses atomic_save_json to prevent corruption on crash.
        """
        from spectramr.shared.utils.safe_io import atomic_save_json

        manifest_path = self.checkpoint_dir / "manifest.json"
        try:
            data = {name: meta.to_dict() for name, meta in self._metadata.items()}
            atomic_save_json(data, manifest_path)
        except Exception as e:
            logger.warning(f"Failed to save manifest: {e}")

    def _get_checkpoint_filename(
        self,
        epoch: int | None = None,
        step: int | None = None,
        is_best: bool = True,
        is_last: bool = False,
    ) -> str:
        """Generate checkpoint filename based on type and config."""
        if is_best:
            return f"checkpoint_best.{self.format}"
        elif is_last:
            return f"checkpoint_last.{self.format}"
        # Always use step-based naming as requested by user
        # "instead of epochs and number of epochs, put iterations"
        step = step or 0  # Default to 0 if step is None
        return f"checkpoint_step_{step}.{self.format}"

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: Any,
        epoch: int,
        loss: float,
        file_path: str | None = None,
        counter: IterationCounterService | None = None,
        **kwargs: Any,
    ) -> str:
        """Save a model checkpoint with atomic writes and retention.

        Args:
            model: Model to save
            optimizer: Optimizer to save
            epoch: Current epoch
            loss: Training loss value
            file_path: Optional custom file path (overrides standard naming)
            counter: Optional IterationCounterService instance (will save counter state)
            **kwargs: Additional arguments (step, is_best, is_last, metric_name,
                      metric_value, extra_optimizers=dict for multi-optimizer GANs)

        Returns:
            Path to saved checkpoint
        """
        step = kwargs.get("step")
        is_best = kwargs.get("is_best", False)
        is_last = kwargs.get("is_last", False)
        metric_name = kwargs.get("metric_name")
        metric_value = kwargs.get("metric_value")
        extra_optimizers = kwargs.get("extra_optimizers")  # NEW: FIX #3 support
        ema_state_dict = kwargs.get("ema_state_dict")  # [FIX] Added EMA support
        # Strategy-owned learnable state (sfc heads, spin_sde diffusion param, ...)
        # that lives on the training strategy, not on the model. See section R
        # design doc; mirrors the rng_state pickle-blob handling below.
        strategy_state = kwargs.get("strategy_state")
        # LR scheduler state — without this a resumed run restarts the warmup /
        # decay schedule at step 0, so the LR jumps discontinuously. Optional:
        # many callers train without a scheduler, in which case the key is simply
        # absent from the checkpoint (load tolerates that).
        scheduler = kwargs.get("scheduler")
        extra_schedulers = kwargs.get("extra_schedulers")
        # Sub-phase 2.9 (data-layer unification plan): persist the
        # training-time transform signature so inference can verify
        # the chain hasn't drifted (modes.infer.strict_train_parity).
        transform_signature = kwargs.get("transform_signature")

        # Use counter step if step is not provided
        if step is None and counter is not None:
            step = counter.current_step

        # Determine checkpoint path
        if file_path is None:
            filename = self._get_checkpoint_filename(epoch, step, is_best, is_last)
            file_path = str(self.checkpoint_dir / filename)

        # Create state dict.
        #
        # Unwrap first: torch.compile prefixes every key with "_orig_mod.",
        # DP/DDP with "module.", FSDP with "_fsdp_wrapped_module.". A prefixed
        # checkpoint cannot be loaded into the bare model that every inference
        # path builds, and with strict=False it loads NOTHING while reporting
        # success (#619 F3). Also applies to the safetensors branch below, which
        # flattens this dict and would otherwise emit
        # "model_state_dict__orig_mod.conv.weight".
        state = {
            "epoch": epoch,
            "step": step or 0,
            "model_state_dict": unwrap_model(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "loss": loss,
            "metric_name": metric_name,
            "metric_value": metric_value,
            # RNG state — without this, resuming training diverges on every
            # stochastic op (dropout, augmentation, diffusion noise sampling).
            # See findings booklet 2026-05-05 T-1.
            "rng_state": capture_rng_state(),
        }

        # Sub-phase 2.9: record transform signature (sha256 hex) when the
        # caller passes it. Older callers that don't pass it produce
        # checkpoints without the key, which inference must tolerate
        # (diff_signatures handles None gracefully).
        if transform_signature is not None:
            state["transform_signature"] = transform_signature

        # [FIX] Save EMA weights natively
        if ema_state_dict:
            state["ema_state_dict"] = ema_state_dict

        # NEW: FIX #3 - Save additional optimizers (for GANs with discriminator, etc.)
        if extra_optimizers:
            state["extra_optimizer_states"] = {
                key: opt.state_dict() for key, opt in extra_optimizers.items() if opt is not None
            }

        # Strategy-owned learnable state (section R). torch.save serializes the
        # nested dict natively; the safetensors path pickles it (see below).
        if strategy_state:
            state["strategy_state"] = strategy_state

        # LR scheduler state. Plain dict (base_lrs / last_epoch / _last_lr / ...);
        # torch.save stores it natively, the safetensors path pickles it (below).
        if scheduler is not None and hasattr(scheduler, "state_dict"):
            state["scheduler_state_dict"] = scheduler.state_dict()
        if extra_schedulers:
            state["extra_scheduler_states"] = {
                key: sch.state_dict()
                for key, sch in extra_schedulers.items()
                if sch is not None and hasattr(sch, "state_dict")
            }

        # Include iteration counter state if provided (SSOT pattern)
        if counter is not None:
            if hasattr(counter, "get_state"):
                state["counter_state"] = counter.get_state()
            elif isinstance(counter, dict):
                state["counter_state"] = counter
            else:
                logger.warning(f"Invalid counter object passed to save_checkpoint: {type(counter)}")

        # Atomic write using temp file
        try:
            # Write to temporary file first
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=self.checkpoint_dir,
                delete=False,
                suffix=f".{self.format}.tmp",
            ) as tmp_file:
                tmp_path = tmp_file.name

            # Save based on format
            if self.format == "safetensors":
                self._save_safetensors(state, tmp_path)
            else:  # .pth format
                torch.save(state, tmp_path)

            with self._lock:
                # Atomic rename
                os.replace(tmp_path, file_path)
                logger.info(f"Saved checkpoint to {file_path}")

                if is_best:
                    _publish_best_alias(Path(file_path))

                # Update metadata — detach any autograd-tracked tensors
                # before casting to float so a leaf with requires_grad=True
                # doesn't fire a UserWarning (or, in stricter modes, raise).
                metadata = CheckpointMetadata(
                    epoch=epoch,
                    step=step or 0,
                    timestamp=datetime.now().isoformat(),
                    loss=(float(loss.detach()) if torch.is_tensor(loss) else loss),
                    metric_name=metric_name,
                    metric_value=(
                        float(metric_value.detach())
                        if metric_value is not None and torch.is_tensor(metric_value)
                        else metric_value
                    ),
                    is_best=is_best,
                    is_last=is_last,
                )
                checkpoint_name = Path(file_path).name
                self._metadata[checkpoint_name] = metadata
                self._save_metadata_index()

                # Apply retention policy if not best/last
                if not is_best and not is_last:
                    self._apply_retention_policy()

            return file_path

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            # Clean up temp file if it exists
            if "tmp_path" in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    def _save_safetensors(self, state: dict[str, Any], path: str) -> None:
        """Save checkpoint in safetensors format."""
        try:
            from safetensors.torch import save_file

            # Convert to format safetensors expects
            # Only tensors can be saved, so we flatten and save only those
            save_dict = {}
            for key, value in state.items():
                if key == "optimizer_state_dict" and value is not None:
                    # Handle main optimizer state specially
                    self._flatten_optimizer_state(save_dict, value, prefix="optimizer_state_dict")

                elif key == "extra_optimizer_states" and isinstance(value, dict):
                    # NEW: Handle extra optimizers (FIX #3 support)
                    for opt_name, opt_state in value.items():
                        if opt_state is not None:
                            prefix = f"extra_optimizer_states__{opt_name}"
                            self._flatten_optimizer_state(save_dict, opt_state, prefix=prefix)

                elif key == "rng_state" and isinstance(value, dict):
                    # RNG state contains a mix of tensors (torch), lists of
                    # tensors (cuda), and tuples (numpy, python random).
                    # Serialize the whole dict via pickle into a uint8 tensor
                    # so safetensors can store it. Without this special-case,
                    # the generic "isinstance(v, torch.Tensor)" branch below
                    # would silently drop everything except the torch tensor.
                    # See findings booklet 2026-05-05 INF-1.
                    import pickle

                    import numpy as np

                    pkl_bytes = pickle.dumps(value, protocol=4)
                    save_dict["_pickle_rng_state"] = torch.from_numpy(
                        np.frombuffer(pkl_bytes, dtype=np.uint8).copy()
                    )

                elif key == "strategy_state" and isinstance(value, dict):
                    # Strategy-owned state (section R) is a nested dict of
                    # {"modules": {name: state_dict}, "params": {name: tensor}}.
                    # safetensors only stores flat tensors, so pickle the whole
                    # dict into a uint8 tensor exactly like rng_state above.
                    import pickle

                    import numpy as np

                    pkl_bytes = pickle.dumps(value, protocol=4)
                    save_dict["_pickle_strategy_state"] = torch.from_numpy(
                        np.frombuffer(pkl_bytes, dtype=np.uint8).copy()
                    )

                elif key == "counter_state" and isinstance(value, dict):
                    # Resume-critical: the iteration counter (global step /
                    # epoch) plus its nested RNG snapshot. These are plain
                    # ints + numpy/python random tuples, not tensors, so the
                    # generic dict branch below would silently drop them and a
                    # resumed run would restart at step 0. Pickle the whole dict
                    # into a uint8 tensor exactly like rng_state / strategy_state.
                    import pickle

                    import numpy as np

                    pkl_bytes = pickle.dumps(value, protocol=4)
                    save_dict["_pickle_counter_state"] = torch.from_numpy(
                        np.frombuffer(pkl_bytes, dtype=np.uint8).copy()
                    )

                elif key in (
                    "scheduler_state_dict",
                    "extra_scheduler_states",
                ) and isinstance(value, dict):
                    # LR scheduler state is a plain dict (base_lrs, last_epoch,
                    # _last_lr, ...), not tensors, so the generic dict branch
                    # below would silently drop it and a resumed run would reset
                    # the schedule. Pickle it like rng_state / strategy_state.
                    import pickle

                    import numpy as np

                    pkl_bytes = pickle.dumps(value, protocol=4)
                    save_dict[f"_pickle_{key}"] = torch.from_numpy(
                        np.frombuffer(pkl_bytes, dtype=np.uint8).copy()
                    )

                elif isinstance(value, dict):
                    # Flatten nested dicts - add prefix to nested keys
                    for k, v in value.items():
                        if isinstance(v, torch.Tensor):
                            prefixed_key = f"{key}_{k}"  # Use underscore to avoid dots in keys
                            save_dict[prefixed_key] = v
                elif isinstance(value, torch.Tensor):
                    save_dict[key] = value
                elif value is not None and not isinstance(value, dict):
                    # For non-tensor scalar values, convert to tensor
                    try:
                        save_dict[f"_meta_{key}"] = torch.tensor(value)
                    except (TypeError, ValueError):
                        # Skip if can't convert
                        pass

            save_file(save_dict, path)
        except ImportError:
            logger.warning("safetensors not installed, falling back to torch.save")
            torch.save(state, path)

    def _flatten_optimizer_state(
        self, save_dict: dict[str, torch.Tensor], opt_state: dict[str, Any], prefix: str
    ) -> None:
        """Helper to flatten an optimizer state dict into save_dict."""
        import json

        # Save param_groups as metadata
        if "param_groups" in opt_state:
            try:
                # Save as bytes tensor because safetensors only supports tensors
                json_bytes = json.dumps(opt_state["param_groups"]).encode("utf-8")
                save_dict[f"_meta_{prefix}_param_groups"] = torch.from_numpy(
                    np.frombuffer(json_bytes, dtype=np.uint8).copy()
                )
            except Exception as e:
                logger.warning(f"Failed to serialize param_groups for {prefix}: {e}")

        # Save state tensors
        if "state" in opt_state:
            for param_id, param_state in opt_state["state"].items():
                for k, v in param_state.items():
                    if isinstance(v, torch.Tensor):
                        # Format: <prefix>__state__<param_id>__<key>
                        save_dict[f"{prefix}__state__{param_id}__{k}"] = v

    def _apply_retention_policy(self) -> None:
        """Apply retention policy: keep only the specified number of checkpoints."""
        checkpoints = list(self.checkpoint_dir.glob(f"checkpoint_epoch_*.{self.format}"))
        checkpoints.extend(self.checkpoint_dir.glob(f"checkpoint_step_*.{self.format}"))

        def get_sort_key(path):
            # Sort by mtime (primary) and step count (secondary) to handle fast training intervals
            """get_sort_key.

            Args:
                path (Any): Description.
            Returns:
                Any: Description.
            """
            mtime = path.stat().st_mtime

            # Extract step/epoch number for secondary sort
            name = path.name
            num = 0
            if "step_" in name:
                try:
                    num = int(name.split("step_")[1].split(".")[0])
                except (ValueError, IndexError) as _exc:
                    logger.debug("Suppressed exception: %s", _exc)
            elif "epoch_" in name:
                try:
                    num = int(name.split("epoch_")[1].split(".")[0])
                except (ValueError, IndexError) as _exc:
                    logger.debug("Suppressed exception: %s", _exc)

            return (mtime, num)

        checkpoints.sort(key=get_sort_key, reverse=True)

        # Protected set = the most-recent ``keep_last_n`` (mtime order above) UNION
        # the best ``keep_best_n`` by recorded metric. Without the best-N union a
        # user setting ``keep_best_n: 3`` silently kept only one checkpoint
        # (pitfall #15: an advertised knob must actually do something).
        keep_n = self.config.keep_last_n
        protected: set[str] = {ckpt.name for ckpt in checkpoints[:keep_n]}

        keep_best_n = getattr(self.config, "keep_best_n", 0) or 0
        if keep_best_n > 0:
            protected |= self._best_checkpoint_names(checkpoints, keep_best_n)

        for checkpoint in checkpoints:
            if checkpoint.name in protected:
                continue
            try:
                checkpoint.unlink()
                if checkpoint.name in self._metadata:
                    del self._metadata[checkpoint.name]
                logger.info(f"Removed old checkpoint: {checkpoint.name}")
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint {checkpoint}: {e}")

        self._save_metadata_index()

    def _best_checkpoint_names(self, checkpoints: list[Path], n: int) -> set[str]:
        """Return the names of the ``n`` best periodic checkpoints by metric.

        Ranks the on-disk periodic checkpoints by their recorded
        ``metric_value`` (direction resolved via the metric SSOT; ``loss`` /
        missing metric falls back to lower-is-better). Checkpoints with no
        recorded metric are not eligible to be "best" and are left to the
        recency policy.
        """
        from spectramr.core.metrics.metric_directions import metric_higher_is_better

        scored: list[tuple[str, float, bool]] = []
        for ckpt in checkpoints:
            meta = self._metadata.get(ckpt.name)
            if meta is None or meta.metric_value is None:
                continue
            higher_better = metric_higher_is_better(meta.metric_name or "loss")
            scored.append((ckpt.name, float(meta.metric_value), higher_better))

        if not scored:
            return set()

        # All periodic checkpoints in a run share one metric, so a single
        # direction governs the ranking; sort best-first accordingly.
        higher_better = scored[0][2]
        scored.sort(key=lambda t: t[1], reverse=higher_better)
        return {name for name, _, _ in scored[:n]}

    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: Any,
        file_path: str,
        counter: IterationCounterService | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        """Load a model checkpoint and optionally restore counter state.

        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into
            file_path: Path to checkpoint file
            counter: Optional IterationCounterService instance (will restore counter state)
            **kwargs: Additional arguments (strict, load_optimizer, load_scheduler,
                      extra_optimizers=dict for multi-optimizer GANs)

        Returns:
            Checkpoint state dict or None if failed
        """
        strict = kwargs.get("strict", True)
        load_optimizer = self.config.load_optimizer_state
        # Honor the advertised load_scheduler_state knob (pitfall #15). The
        # scheduler objects to restore into are passed by the caller; when a run
        # trains without a scheduler these are simply absent.
        load_scheduler = self.config.load_scheduler_state
        scheduler = kwargs.get("scheduler")
        extra_schedulers = kwargs.get("extra_schedulers")
        extra_optimizers = kwargs.get("extra_optimizers")  # NEW: FIX #3 support
        ema_model = kwargs.get("ema_model")  # [FIX] Add support to load EMA weights

        # Use lock to prevent loading while retention policy is deleting
        with self._lock:
            try:
                if not os.path.exists(file_path):
                    logger.error(f"Checkpoint file not found: {file_path}")
                    return None

                if file_path.endswith(".safetensors"):
                    state = self._load_safetensors(file_path)
                else:
                    # [FORENSIC FIX] Security: try weights_only=True first to prevent pickle RCE
                    try:
                        # Our own rng_state is the one thing weights_only=True
                        # cannot reconstruct unaided, so allowlist exactly the
                        # four data-only globals it needs. Before this, the
                        # weights-only attempt failed on *every* checkpoint this
                        # service writes and the full unpickler ran instead —
                        # the security guard was unreachable on its own format,
                        # and each load emitted a fallback warning (pitfall #10).
                        with torch.serialization.safe_globals(RNG_STATE_SAFE_GLOBALS):
                            state = torch.load(file_path, map_location="cpu", weights_only=True)
                    except Exception as e:
                        # Any weights_only=True failure → fall back to the full
                        # unpickler. Catch broadly: a checkpoint may carry other
                        # non-tensor payloads (a strategy's own state), and the
                        # previously-named ``torch.serialization.UnpicklingError``
                        # does not exist in current torch — referencing it here
                        # raised AttributeError *during* exception handling and
                        # silently nulled every .pth load (2026-06).
                        logger.warning(
                            f"Failed to load checkpoint with weights_only=True: {e}. "
                            f"Falling back to weights_only=False for compatibility."
                        )
                        state = torch.load(file_path, map_location="cpu", weights_only=False)

                # Load model state.
                #
                # Symmetric with the save side: unwrap the live module (it may be
                # compiled/DDP-wrapped while the checkpoint is clean) AND strip
                # prefixes off the stored keys (the checkpoint may predate the
                # save-side fix). Without the strip, every checkpoint an existing
                # compiled or DDP run already produced would become unloadable.
                if "model_state_dict" in state:
                    unwrap_model(model).load_state_dict(
                        strip_wrapper_prefixes(state["model_state_dict"]),
                        strict=strict,
                    )

                # [FIX] Load EMA state
                if ema_model and "ema_state_dict" in state:
                    # ModelEma holds its shadow under `.module`, so an EMA state
                    # dict was ALWAYS prefixed even on a single-GPU eager run.
                    # Prefer ModelEma's own loader, which also restores the
                    # warmup counter; unwrapping first peels past the override
                    # and silently restarts the decay ramp at 0 (#2172). The
                    # attribute check is for callers that pass a bare module.
                    if hasattr(ema_model, "load_shadow_state_dict"):
                        ema_model.load_shadow_state_dict(state["ema_state_dict"], strict=strict)
                    else:
                        unwrap_model(ema_model).load_state_dict(
                            strip_wrapper_prefixes(state["ema_state_dict"]), strict=strict
                        )
                    logger.info("Restored EMA shadow weights perfectly")

                # Load optimizer state if requested
                if load_optimizer and optimizer and "optimizer_state_dict" in state:
                    optimizer.load_state_dict(state["optimizer_state_dict"])

                # NEW: FIX #3 - Load additional optimizers for multi-optimizer models
                if load_optimizer and extra_optimizers and "extra_optimizer_states" in state:
                    for key, opt in extra_optimizers.items():
                        if opt is not None and key in state["extra_optimizer_states"]:
                            opt.load_state_dict(state["extra_optimizer_states"][key])
                            logger.info(f"Restored optimizer state for '{key}'")

                # Restore LR scheduler state (gated by load_scheduler_state) so
                # the warmup / decay schedule resumes instead of restarting at 0.
                if load_scheduler and scheduler is not None and state.get("scheduler_state_dict"):
                    scheduler.load_state_dict(state["scheduler_state_dict"])
                    logger.info("Restored LR scheduler state")

                if load_scheduler and extra_schedulers and state.get("extra_scheduler_states"):
                    for key, sch in extra_schedulers.items():
                        if sch is not None and key in state["extra_scheduler_states"]:
                            sch.load_state_dict(state["extra_scheduler_states"][key])
                            logger.info(f"Restored scheduler state for '{key}'")

                # Restore counter state if provided (SSOT pattern)
                if counter is not None and "counter_state" in state:
                    counter.restore_state(state["counter_state"])
                    logger.info(
                        f"Restored counter state: step={counter.current_step}, "
                        f"epoch={counter.current_epoch}"
                    )

                # Restore RNG state for deterministic resume across all
                # stochastic ops. See findings booklet 2026-05-05 T-1.
                if "rng_state" in state and state["rng_state"] is not None:
                    restore_rng_state(state["rng_state"])
                    logger.info("Restored RNG state (torch + cuda + numpy + python)")

                logger.info(f"Loaded checkpoint from {file_path}")
                return state

            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
                return None

    def _load_safetensors(self, file_path: str) -> dict[str, Any]:
        """Load checkpoint from safetensors format.

        Raises:
            Exception: If loading fails
        """
        try:
            import json

            from safetensors.torch import load_file

            tensors = load_file(file_path)
            # Unflatten nested structures
            state = {
                "model_state_dict": {},
                "ema_state_dict": {},  # [FIX] Initialize ema_state_dict
                "optimizer_state_dict": {},  # Initialize optimizer_state_dict
                "extra_optimizer_states": {},  # Initialize extra_optimizer_states
                "epoch": None,
                "step": None,
                "loss": None,
                "metric_name": None,
                "metric_value": None,
            }

            # Temporary storage for optimizer state reconstruction
            opt_states_tensors = {"optimizer_state_dict": {}}

            for key, value in tensors.items():
                if key == "_pickle_rng_state":
                    # Counterpart to the rng_state pickle done in
                    # _save_safetensors. See findings booklet 2026-05-05 INF-1.
                    import pickle

                    try:
                        pkl_bytes = bytes(value.tolist())
                        state["rng_state"] = pickle.loads(pkl_bytes)
                    except Exception as exc:
                        logger.warning(
                            "Failed to deserialize rng_state from safetensors checkpoint: %s",
                            exc,
                        )

                elif key == "_pickle_strategy_state":
                    # Counterpart to the strategy_state pickle in _save_safetensors
                    # (section R).
                    import pickle

                    try:
                        pkl_bytes = bytes(value.tolist())
                        state["strategy_state"] = pickle.loads(pkl_bytes)
                    except Exception as exc:
                        logger.warning(
                            "Failed to deserialize strategy_state from safetensors checkpoint: %s",
                            exc,
                        )

                elif key == "_pickle_counter_state":
                    # Counterpart to the counter_state pickle in
                    # _save_safetensors. Without re-hydrating this key,
                    # load_checkpoint never sees "counter_state" and the
                    # resumed run silently restarts at step 0.
                    import pickle

                    try:
                        pkl_bytes = bytes(value.tolist())
                        state["counter_state"] = pickle.loads(pkl_bytes)
                    except Exception as exc:
                        logger.warning(
                            "Failed to deserialize counter_state from safetensors checkpoint: %s",
                            exc,
                        )

                elif key in (
                    "_pickle_scheduler_state_dict",
                    "_pickle_extra_scheduler_states",
                ):
                    # Counterpart to the scheduler pickle in _save_safetensors.
                    # Without re-hydrating these keys, load_checkpoint never sees
                    # the scheduler state and the resumed run resets its LR schedule.
                    import pickle

                    target_key = key[len("_pickle_") :]
                    try:
                        state[target_key] = pickle.loads(bytes(value.tolist()))
                    except Exception as exc:
                        logger.warning(
                            "Failed to deserialize %s from safetensors checkpoint: %s",
                            target_key,
                            exc,
                        )

                elif key.startswith("model_state_dict_"):
                    # Restore nested key
                    nested_key = key[17:]  # Remove "model_state_dict_" prefix (17 chars)
                    state["model_state_dict"][nested_key] = value

                elif key.startswith("ema_state_dict_"):
                    # [FIX] Restore EMA shadow weights nested key
                    nested_key = key[15:]  # Remove "ema_state_dict_" prefix (15 chars)
                    state["ema_state_dict"][nested_key] = value

                elif (
                    key == "_meta_optimizer_param_groups"
                    or key == "_meta_optimizer_state_dict_param_groups"
                ):
                    # Restore main optimizer param_groups
                    try:
                        json_str = bytes(value.tolist()).decode("utf-8")
                        state["optimizer_state_dict"]["param_groups"] = json.loads(json_str)
                    except Exception as e:
                        logger.warning(f"Failed to deserialize param_groups: {e}")

                elif key.startswith("_meta_extra_optimizer_states__") and key.endswith(
                    "_param_groups"
                ):
                    # Restore extra optimizer param_groups
                    # Format: _meta_extra_optimizer_states__<opt_name>_param_groups
                    parts = key.split("__")
                    if len(parts) >= 2:
                        opt_name = parts[1].replace("_param_groups", "")
                        if opt_name not in state["extra_optimizer_states"]:
                            state["extra_optimizer_states"][opt_name] = {}
                        try:
                            json_str = bytes(value.tolist()).decode("utf-8")
                            state["extra_optimizer_states"][opt_name]["param_groups"] = json.loads(
                                json_str
                            )
                        except Exception as e:
                            logger.warning(
                                f"Failed to deserialize param_groups for {opt_name}: {e}"
                            )

                elif key.startswith("optimizer_state_dict__state__"):
                    # Restore main optimizer state tensors
                    parts = key.split("__")
                    if len(parts) == 4:
                        try:
                            param_id = int(parts[2])
                        except ValueError:
                            param_id = parts[2]
                        state_key = parts[3]

                        if param_id not in opt_states_tensors["optimizer_state_dict"]:
                            opt_states_tensors["optimizer_state_dict"][param_id] = {}
                        opt_states_tensors["optimizer_state_dict"][param_id][state_key] = value

                elif key.startswith("extra_optimizer_states__"):
                    # Restore extra optimizer state tensors
                    # Format: extra_optimizer_states__<opt_name>__state__<param_id>__<key>
                    parts = key.split("__")
                    if len(parts) == 5 and parts[2] == "state":
                        opt_name = parts[1]
                        try:
                            param_id = int(parts[3])
                        except ValueError:
                            param_id = parts[3]
                        state_key = parts[4]

                        if opt_name not in opt_states_tensors:
                            opt_states_tensors[opt_name] = {}
                        if param_id not in opt_states_tensors[opt_name]:
                            opt_states_tensors[opt_name][param_id] = {}
                        opt_states_tensors[opt_name][param_id][state_key] = value

                elif key.startswith("optimizer_state_dict_"):
                    # Fallback for old/other format
                    nested_key = key[21:]
                    state["optimizer_state_dict"][nested_key] = value

                elif key.startswith("_meta_"):
                    # Restore metadata values from tensors
                    meta_key = key[6:]  # Remove "_meta_" prefix
                    # Convert tensor back to scalar
                    scalar_value = value.item() if value.numel() == 1 else value
                    if meta_key in state:
                        state[meta_key] = scalar_value
                elif key in ["epoch", "step", "loss", "metric_name", "metric_value"]:
                    state[key] = value

            # Finalize optimizer states
            state["optimizer_state_dict"]["state"] = opt_states_tensors["optimizer_state_dict"]
            for opt_name, tensors_dict in opt_states_tensors.items():
                if opt_name != "optimizer_state_dict":
                    if opt_name not in state["extra_optimizer_states"]:
                        state["extra_optimizer_states"][opt_name] = {}
                    state["extra_optimizer_states"][opt_name]["state"] = tensors_dict

            return state
        except ImportError:
            logger.warning("safetensors not installed, falling back to torch.load")
            # [FORENSIC FIX] Security: weights_only=True prevents pickle RCE
            return torch.load(file_path, map_location="cpu", weights_only=True)
        except Exception as e:
            logger.error(f"Failed to load safetensors file {file_path}: {e}")
            raise

    def find_latest_checkpoint(self, checkpoint_dir: str) -> str | None:
        """Find the latest checkpoint in a directory.

        Args:
            checkpoint_dir: Directory to search

        Returns:
            Path to latest checkpoint or None if none found
        """
        dir_path = Path(checkpoint_dir)
        # Both suffixes, always. `self.format` describes what THIS service
        # writes, but the writer on the production training path is
        # `CheckpointDirector.save()`, which spells `.pt` unconditionally — so
        # globbing the declared format alone returned None against every
        # checkpoint a real run had just produced, and `--resume auto` then
        # raised "No checkpoint found" on an arm whose directory was full of
        # them (#2070). Ordering below is by step, not by suffix, so
        # a mixed directory resolves to the newest state either writer saved.
        suffixes = dict.fromkeys((self.format, "pt"))
        checkpoints: list[Path] = []
        for suffix in suffixes:
            checkpoints.extend(dir_path.glob(f"checkpoint_epoch_*.{suffix}"))
            checkpoints.extend(dir_path.glob(f"checkpoint_step_*.{suffix}"))
        if not checkpoints:
            for suffix in suffixes:
                last = dir_path / f"checkpoint_last.{suffix}"
                if last.exists():
                    return str(last)
            for suffix in suffixes:
                best = dir_path / f"checkpoint_best.{suffix}"
                if best.exists():
                    return str(best)
            return None

        def extract_step(path: Path) -> int:
            """Extract step/epoch number from filename for deterministic ordering.

            [FORENSIC FIX] Uses embedded step number instead of st_mtime
            to avoid non-deterministic behavior on parallel filesystems.
            """
            name = path.stem
            try:
                if "step_" in name:
                    return int(name.split("step_")[1].split(".")[0])
                if "epoch_" in name:
                    # Epochs are conceptually larger units, multiply to sort after steps
                    return int(name.split("epoch_")[1].split(".")[0]) * 100000
            except (ValueError, IndexError) as _exc:
                logger.debug("Suppressed exception: %s", _exc)
            return -1

        # Sort by step (primary), mtime as tiebreaker only
        latest = max(checkpoints, key=lambda x: (extract_step(x), x.stat().st_mtime))
        # Ensure we return a string, resolving any Path objects or mocks
        return str(latest)

    def save_last(
        self, model: nn.Module, optimizer: Any, epoch: int, step: int, loss: float
    ) -> str:
        """Save the last checkpoint (overrides previous last).

        Args:
            model: Model to save
            optimizer: Optimizer to save
            epoch: Current epoch
            step: Current step
            loss: Current loss

        Returns:
            Path to saved checkpoint
        """
        return self.save_checkpoint(model, optimizer, epoch, loss, step=step, is_last=True)

    def list_checkpoints(self) -> list[tuple[str, CheckpointMetadata]]:
        """List all saved checkpoints with their metadata.

        Returns:
            List of (filename, metadata) tuples
        """
        with self._lock:
            return [(name, meta) for name, meta in self._metadata.items()]

    def get_last_checkpoint(self) -> tuple[str, CheckpointMetadata] | None:
        """Get the last checkpoint.

        Returns:
            (filename, metadata) tuple or None
        """
        with self._lock:
            last_files = [(name, meta) for name, meta in self._metadata.items() if meta.is_last]
            return last_files[0] if last_files else None

    def save_if_needed(
        self,
        counter: IterationCounterService,
        model: nn.Module,
        optimizer: Any,
        loss: float,
        **kwargs: Any,
    ) -> str | None:
        """Save checkpoint only if interval reached (SSOT pattern).

        This method queries the IterationCounterService to determine if the
        checkpoint save interval has been reached. If so, saves the checkpoint
        with counter state included automatically.

        Args:
            counter: IterationCounterService instance (SSOT for step tracking)
            model: Model to save
            optimizer: Optimizer to save
            loss: Current training loss
            **kwargs: Additional arguments (passed to save_checkpoint)

        Returns:
            Path to saved checkpoint if saved, None if interval not reached

        Example:
            >>> checkpoint_service.save_if_needed(
            ...     counter, model, optimizer, loss,
            ...     epoch=epoch, step=counter.current_step
            ... )
        """
        if not self.should_save_checkpoint(counter):
            return None

        # Save with counter state
        return self.save_checkpoint(
            model=model,
            optimizer=optimizer,
            loss=loss,
            counter=counter,
            epoch=counter.current_epoch,
            step=counter.current_step,
            **kwargs,
        )


class CheckpointServiceFactory:
    """Factory for creating CheckpointService instances."""

    @staticmethod
    def create(config: CheckpointConfigSchema) -> CheckpointService:
        """Create a CheckpointService instance."""
        return CheckpointService(config)


__all__ = ["CheckpointMetadata", "CheckpointService", "CheckpointServiceFactory"]
