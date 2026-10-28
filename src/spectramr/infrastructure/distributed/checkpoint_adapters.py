"""How each parallel strategy wants its checkpoints written.

Three facts differ per strategy, and getting any of them wrong is silent:

1. **Who calls.** FSDP's ``state_dict()`` and DeepSpeed's ``save_checkpoint()``
   are *collectives*. The training loop gates every checkpoint on
   ``RankUtility.is_main_rank()``, so rank 0 would enter an all-gather that
   ranks 1..N never enter: the job HANGS with no error and no log line, and
   SLURM eventually kills it at walltime. That is the most expensive possible
   failure mode -- far worse than a crash -- which is why
   :attr:`requires_all_ranks` exists and why the loop derives ONE predicate
   from it rather than editing each call site.

2. **What ``state_dict()`` returns.** Under FSDP the default is the local
   *shard*. Saving that produces a file with the right key names, the right
   dtypes, and 1/N of the values -- so it loads into a bare model without
   complaint and reconstructs a model that was never trained.

3. **What the artifact IS.** DeepSpeed writes a tag directory of per-rank
   shards. Resume reads it; every other reader opens the RunState file beside it,
   whose models ``full_model_state`` gathered.

``rank0_only=True`` controls who *receives* the gathered tensors, never who
*participates* -- a distinction that reads as an optimisation and is actually
the deadlock.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple, Protocol, runtime_checkable

from spectramr.infrastructure.distributed.state_io import FSDPStateIO, PassthroughStateIO

logger = logging.getLogger(__name__)

__all__ = [
    "DeepSpeedCheckpointAdapter",
    "DefaultCheckpointAdapter",
    "FSDPCheckpointAdapter",
    "IParallelCheckpointAdapter",
    "NativeLoadResult",
    "resolve_checkpoint_adapter",
]

#: Subdirectory name for DeepSpeed's sharded tag directories.
DEEPSPEED_TAG_PREFIX = "deepspeed"


class NativeLoadResult(NamedTuple):
    """What a strategy-native restore produced, and which models it covered.

    ``client_state`` is ``None`` when nothing was loaded and ``{}`` when the
    tag carried no client state; the two mean opposite things. ``handled`` names
    the model keys whose weights AND optimizer state came from the native
    artifact, so the generic parse must not load them a second time: a ZeRO
    optimizer cannot read the generic dict (#2482), and at world size > 1 the
    writer rank's copy would overwrite every rank's partition (#1309).
    """

    client_state: dict[str, Any] | None
    handled: frozenset[str]


_NOTHING_LOADED = NativeLoadResult(None, frozenset())


@runtime_checkable
class IParallelCheckpointAdapter(Protocol):
    """The checkpoint contract one parallel strategy imposes."""

    name: str

    #: True when save/load must be entered by EVERY rank because the strategy's
    #: state-gathering is collective. The training loop ORs this with
    #: ``is_main_rank()``; the adapter itself decides who touches the disk.
    requires_all_ranks: bool

    #: True when the strategy writes its own artifact and the generic
    #: ``torch.save`` payload would be shard-shaped nonsense.
    writes_native_artifact: bool

    #: Reads and restores model / optimizer state. The one place a strategy's
    #: sharding is allowed to change *how* state is read, so the director never
    #: calls ``state_dict()`` on a model or optimizer itself.
    state_io: PassthroughStateIO

    def full_model_state(self, model: Any) -> dict[str, Any]:
        """The model's WHOLE state dict on the writer rank, ``{}`` elsewhere; collective."""
        ...

    def engine_owned_keys(self, models: Mapping[str, Any]) -> frozenset[str]:
        """Model keys whose optimizer and scheduler state live in the native artifact."""
        ...

    def ema_is_whole(self, models: Mapping[str, Any]) -> bool:
        """Whether the EMA shadow is whole-parameter, so the best checkpoint carries it."""
        ...

    def save_native(
        self,
        *,
        models: Mapping[str, Any],
        checkpoint_dir: Path,
        tag: str,
        client_state: Mapping[str, Any],
    ) -> Path | None:
        """Write the strategy-native artifact. ``None`` = nothing extra to write."""
        ...

    def load_native(
        self,
        *,
        models: Mapping[str, Any],
        checkpoint_dir: Path,
        tag: str,
        optimizer: bool = True,
        scheduler: bool = True,
        module_only: bool = False,
    ) -> NativeLoadResult:
        """Restore from the strategy-native artifact.

        ``optimizer``/``scheduler`` carry the run's load gates; ``module_only`` restores
        weights alone, for the best-weights restore. Returns the client state and the
        model keys the restore covered.
        """
        ...

    def native_tag_roots(
        self, *, models: Mapping[str, Any], checkpoint_dir: Path
    ) -> tuple[Path, ...]:
        """The directories :meth:`save_native` writes one ``<tag>/`` into per save.

        Retention prunes a tag's directory under each root together with its ``.pt``;
        the adapter owns the layout so the writer and the pruner cannot disagree on it.
        """
        ...


class DefaultCheckpointAdapter:
    """No wrapper, DataParallel, DDP: ``state_dict()`` is already complete.

    DP and DDP replicate rather than shard, so rank 0's copy IS the model.
    Prefix stripping is handled by :mod:`spectramr.core.module_utils` at the
    save site and is not this adapter's concern.
    """

    name = "default"
    requires_all_ranks = False
    writes_native_artifact = False
    state_io = PassthroughStateIO()

    def full_model_state(self, model: Any) -> dict[str, Any]:
        return self.state_io.model_state(model)

    def engine_owned_keys(self, models: Mapping[str, Any]) -> frozenset[str]:
        return frozenset()

    def ema_is_whole(self, models: Mapping[str, Any]) -> bool:
        return True

    def save_native(self, **_kwargs: Any) -> Path | None:
        return None

    def load_native(self, **_kwargs: Any) -> NativeLoadResult:
        return _NOTHING_LOADED

    def native_tag_roots(self, **_kwargs: Any) -> tuple[Path, ...]:
        return ()


class FSDPCheckpointAdapter:
    """Gather the full model and optimizer state on the FSDP root before they are read.

    A shard saved as-is has plausible key names and 1/N of the values, so a bare
    model loads it cleanly at world_size==1 (``no_shard``, a single-GPU debug run)
    and raises a shape error only at world_size>1 -- which is how this class of
    bug survives local testing. :class:`FSDPStateIO` does the gathering
    (``offload_to_cpu``: the gathered dict is the FULL model on one device, on top
    of that rank's shard).
    """

    name = "fsdp"
    #: state_dict() under FSDP all-gathers -- every rank must call it.
    requires_all_ranks = True
    writes_native_artifact = False
    state_io = FSDPStateIO()

    def full_model_state(self, model: Any) -> dict[str, Any]:
        return self.state_io.model_state(model)

    def engine_owned_keys(self, models: Mapping[str, Any]) -> frozenset[str]:
        return frozenset()

    def ema_is_whole(self, models: Mapping[str, Any]) -> bool:
        return True

    def save_native(self, **_kwargs: Any) -> Path | None:
        return None

    def load_native(self, **_kwargs: Any) -> NativeLoadResult:
        return _NOTHING_LOADED

    def native_tag_roots(self, **_kwargs: Any) -> tuple[Path, ...]:
        return ()


class DeepSpeedCheckpointAdapter:
    """A sharded tag directory for resume; gathered weights for the RunState file beside it.

    ``engine.save_checkpoint`` writes per-rank ZeRO shards, the only thing that can
    restore optimizer state at stage 2/3. Every other reader opens the RunState file,
    whose models :meth:`full_model_state` gathered, so no second weights file is written.
    """

    name = "deepspeed"
    #: save_checkpoint/load_checkpoint are collective.
    requires_all_ranks = True
    writes_native_artifact = True
    #: Models the engine does not own (a plain discriminator) go through the
    #: generic payload, so they read and restore like any unwrapped module.
    state_io = PassthroughStateIO()

    def full_model_state(self, model: Any) -> dict[str, Any]:
        """Gathered weights for every capture, so a step or best file loads without DeepSpeed.

        At ZeRO-3 the unwrapped module holds partitions, so the 16-bit consolidation runs on
        every rank (it is collective) and only rank 0 receives the dict. Below stage 3 every
        rank holds whole parameters and the passthrough read is complete.
        """
        partitioned = getattr(model, "zero_optimization_partition_weights", None)
        if callable(partitioned) and partitioned():
            return model._zero3_consolidated_16bit_state_dict() or {}
        return self.state_io.model_state(model)

    def engine_owned_keys(self, models: Mapping[str, Any]) -> frozenset[str]:
        """The engines' keys: their ZeRO optimizer and scheduler are saved in the tag directory.

        The ZeRO optimizer's ``state_dict()`` holds DeepSpeed's ``LossScaler`` object, which
        no checkpoint codec stores, and it is sharded per rank anyway.
        """
        return frozenset(self._engines(models))

    def ema_is_whole(self, models: Mapping[str, Any]) -> bool:
        """Whether the EMA shadow is whole-parameter: true only when every engine is at ZeRO <= 2.

        At stage 3 the shadow is built from local partitions, so a strict load of it into
        the full model raises. An engine that cannot report its stage is treated as
        partitioned rather than assumed whole.
        """
        for engine in self._engines(models).values():
            stage = getattr(engine, "zero_optimization_stage", None)
            if not callable(stage) or stage() > 2:
                logger.warning(
                    "[DeepSpeed] the EMA shadow is partitioned, not whole-parameter: the "
                    "engine is at ZeRO stage 3 or does not report its stage."
                )
                return False
        return True

    @staticmethod
    def _engines(models: Mapping[str, Any]) -> dict[str, Any]:
        """The values that are DeepSpeed engines, by model key.

        Duck-typed on ``save_checkpoint`` rather than ``isinstance`` so this
        module never imports deepspeed -- keeping it importable, and testable,
        on a host without the extra.
        """
        return {
            key: model
            for key, model in models.items()
            if model is not None and hasattr(model, "save_checkpoint")
        }

    @staticmethod
    def _tag_root(checkpoint_dir: Path, key: str) -> Path:
        return checkpoint_dir / DEEPSPEED_TAG_PREFIX / key

    def native_tag_roots(
        self, *, models: Mapping[str, Any], checkpoint_dir: Path
    ) -> tuple[Path, ...]:
        return tuple(self._tag_root(checkpoint_dir, key) for key in self._engines(models))

    def save_native(
        self,
        *,
        models: Mapping[str, Any],
        checkpoint_dir: Path,
        tag: str,
        client_state: Mapping[str, Any],
    ) -> Path | None:
        engines = self._engines(models)
        if not engines:
            return None

        for key, engine in engines.items():
            # COLLECTIVE: every rank calls, DeepSpeed decides who writes what. It ends in
            # a barrier, so on return every rank's shards of this tag have landed.
            engine.save_checkpoint(
                str(self._tag_root(checkpoint_dir, key)), tag=tag, client_state=dict(client_state)
            )
        logger.info(
            "[DeepSpeed] wrote sharded checkpoint tag %r under %s",
            tag,
            checkpoint_dir / DEEPSPEED_TAG_PREFIX,
        )
        return None

    def load_native(
        self,
        *,
        models: Mapping[str, Any],
        checkpoint_dir: Path,
        tag: str,
        optimizer: bool = True,
        scheduler: bool = True,
        module_only: bool = False,
    ) -> NativeLoadResult:
        engines = self._engines(models)
        if not engines:
            return _NOTHING_LOADED
        client_state: dict[str, Any] = {}
        for key, engine in engines.items():
            tag_root = self._tag_root(checkpoint_dir, key)
            # COLLECTIVE, same as the save side.
            load_path, state = engine.load_checkpoint(
                str(tag_root),
                tag=tag,
                load_optimizer_states=optimizer,
                load_lr_scheduler_states=scheduler,
                load_module_only=module_only,
            )
            if load_path is None:
                # DeepSpeed reports a miss by RETURNING (None, None), not by
                # raising. Swallowing that yields an empty dict indistinguishable
                # from "loaded, but this tag carried no client state" -- and the
                # caller's next move is to treat the sharded load as authoritative
                # and stop parsing the file beside it. The run would then report a
                # successful restore while still holding the weights it started
                # with, which is the silent-fallback shape non-negotiable 3 exists
                # to prevent.
                raise FileNotFoundError(
                    f"[{self.name}] no checkpoint for tag {tag!r} under "
                    f"{tag_root}. engine.load_checkpoint returned no path, "
                    "which is how DeepSpeed reports a missing or incomplete tag "
                    "directory."
                )
            if state:
                client_state.update(state)
        return NativeLoadResult(client_state, frozenset(engines))


def resolve_checkpoint_adapter(parallel: Any) -> IParallelCheckpointAdapter:
    """The adapter carried by a resolved :class:`ParallelRuntime` (or ``None``).

    The *plugin* builds its own adapter during ``adopt`` -- it is the only thing
    that knows how the model was wrapped and what the arm declared -- and this
    just reads it back, defaulting for the single-process and scripting-API
    paths where there is no runtime at all.
    """
    if parallel is None:
        return DefaultCheckpointAdapter()
    adapter = getattr(parallel, "checkpoint_adapter", None)
    if adapter is None:
        return DefaultCheckpointAdapter()
    return adapter
