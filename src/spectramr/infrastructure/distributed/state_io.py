"""How a model's and optimizer's state is read for a checkpoint and put back.

Under FSDP a checkpoint has to be assembled by *collectives* on the FSDP root:

* ``model.module.state_dict()`` -- what ``unwrap_model(...)`` hands back -- is the
  inner module's flat shard views (a ``(54,)`` slice where a ``(8, 8)`` weight
  belongs), even inside ``FSDP.state_dict_type``. Only the root's own
  ``state_dict()`` gathers.
* ``optimizer.state_dict()`` holds this rank's slice of the Adam moments, so rank
  0 would write one Nth of the optimizer and every rank would load it back.
  ``FSDP.optim_state_dict`` / ``optim_state_dict_to_load`` convert to and from the
  unsharded form.

Every method here is entered by every rank. ``rank0_only`` decides who *receives*
the gathered tensors, never who participates.
"""

from __future__ import annotations

import contextlib
from collections.abc import Generator
from typing import Any

from spectramr.core.module_utils import strip_wrapper_prefixes, unwrap_model


class PassthroughStateIO:
    """Replicated or unwrapped models: ``state_dict()`` of the unwrapped module is complete."""

    def model_state(self, model: Any) -> dict[str, Any]:
        return unwrap_model(model).state_dict()

    def optimizer_state(self, model: Any, optimizer: Any) -> dict[str, Any]:
        return optimizer.state_dict()

    def load_model_state(self, model: Any, state: dict[str, Any]) -> None:
        unwrap_model(model).load_state_dict(strip_wrapper_prefixes(state))

    def load_optimizer_state(self, model: Any, optimizer: Any, state: dict[str, Any]) -> None:
        optimizer.load_state_dict(state)

    @contextlib.contextmanager
    def full_params(self, model: Any) -> Generator[None]:
        yield

    @contextlib.contextmanager
    def swap_in(self, model: Any, shadow: Any) -> Generator[None]:
        """Run the block with the EMA ``shadow`` module's weights installed in ``model``."""
        from spectramr.infrastructure.optimization.ema_swap import ema_weights_swapped_in

        with ema_weights_swapped_in(unwrap_model(model), shadow):
            yield


def state_io_of(parallel: Any) -> PassthroughStateIO:
    """The ``state_io`` of a resolved ``ParallelRuntime``, or the passthrough without one."""
    adapter = getattr(parallel, "checkpoint_adapter", None)
    return getattr(adapter, "state_io", None) or PassthroughStateIO()


def _is_fsdp_root(model: Any) -> bool:
    import torch.distributed.fsdp as fsdp

    return isinstance(model, fsdp.FullyShardedDataParallel)


class FSDPStateIO(PassthroughStateIO):
    """Gather on save, scatter on load, all on the FSDP root.

    A model that is not an FSDP root (a plain discriminator beside a wrapped
    generator) takes the passthrough path, because the FSDP context managers
    raise on it rather than degrade.
    """

    def _save_contexts(self, model: Any):
        import torch.distributed.fsdp as fsdp
        from torch.distributed.fsdp import (
            FullOptimStateDictConfig,
            FullStateDictConfig,
            StateDictType,
        )

        return fsdp.FullyShardedDataParallel.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
        )

    def model_state(self, model: Any) -> dict[str, Any]:
        if not _is_fsdp_root(model):
            return super().model_state(model)
        with self._save_contexts(model):
            # The ROOT's state_dict(), not the unwrapped module's: only the root's
            # hooks gather. Non-zero ranks receive an empty dict (rank0_only).
            return strip_wrapper_prefixes(model.state_dict())

    def optimizer_state(self, model: Any, optimizer: Any) -> dict[str, Any]:
        if not _is_fsdp_root(model):
            return super().optimizer_state(model, optimizer)
        import torch.distributed.fsdp as fsdp

        with self._save_contexts(model):
            return fsdp.FullyShardedDataParallel.optim_state_dict(model, optimizer)

    def load_model_state(self, model: Any, state: dict[str, Any]) -> None:
        if not _is_fsdp_root(model):
            return super().load_model_state(model, state)
        import torch.distributed.fsdp as fsdp
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType

        # Every rank holds the full dict (each reads the file), so rank0_only is
        # False: FSDP slices out this rank's shard from what it is given.
        with fsdp.FullyShardedDataParallel.state_dict_type(
            model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(rank0_only=False)
        ):
            model.load_state_dict(strip_wrapper_prefixes(state))

    def load_optimizer_state(self, model: Any, optimizer: Any, state: dict[str, Any]) -> None:
        if not _is_fsdp_root(model):
            return super().load_optimizer_state(model, optimizer, state)
        import torch.distributed.fsdp as fsdp

        optimizer.load_state_dict(
            fsdp.FullyShardedDataParallel.optim_state_dict_to_load(model, optimizer, state)
        )

    @contextlib.contextmanager
    def full_params(self, model: Any) -> Generator[None]:
        """Unsharded parameters on every rank, for code that reads whole tensors (EMA)."""
        if not _is_fsdp_root(model):
            yield
            return
        import torch.distributed.fsdp as fsdp

        with fsdp.FullyShardedDataParallel.summon_full_params(model, writeback=False):
            yield

    @contextlib.contextmanager
    def swap_in(self, model: Any, shadow: Any) -> Generator[None]:
        """EMA swap through the full-state-dict path, restoring the live weights on exit.

        ``summon_full_params`` cannot be held across a validation forward (FSDP
        only runs ``forward`` from its idle state), so the shadow is loaded the
        way a checkpoint is -- gathered, then re-sharded -- and the live weights
        are put back the same way. Strict on both loads: a partial swap would
        grade a blend while reporting EMA.
        """
        if not _is_fsdp_root(model):
            with super().swap_in(model, shadow):
                yield
            return
        import torch.distributed.fsdp as fsdp
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType

        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
        with fsdp.FullyShardedDataParallel.state_dict_type(
            model, StateDictType.FULL_STATE_DICT, cfg
        ):
            live = {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(shadow.state_dict())
        try:
            yield
        finally:
            with fsdp.FullyShardedDataParallel.state_dict_type(
                model, StateDictType.FULL_STATE_DICT, cfg
            ):
                model.load_state_dict(live)
