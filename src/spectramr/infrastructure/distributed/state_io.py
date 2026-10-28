"""How a model's and optimizer's state is read for a checkpoint and put back.

Under FSDP2 a parameter is a DTensor holding this rank's shard, so
``model.state_dict()`` and ``optimizer.state_dict()`` are 1/N of the truth. The
distributed-checkpoint state-dict API (``get_model_state_dict`` /
``set_model_state_dict`` and the optimizer pair) gathers to, and scatters from, the
full unsharded form.

Every method here is entered by every rank: the gathers are collectives.
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
    from torch.distributed.fsdp import FSDPModule

    return isinstance(model, FSDPModule)


def _full(*, to_cpu: bool) -> Any:
    from torch.distributed.checkpoint.state_dict import StateDictOptions

    # cpu_offload with a full state dict also makes the gather rank-0-only.
    return StateDictOptions(full_state_dict=True, cpu_offload=to_cpu)


class FSDPStateIO(PassthroughStateIO):
    """Gather on save, scatter on load, on an FSDP2 module.

    A model that is not sharded (a plain discriminator beside a sharded generator)
    takes the passthrough path.
    """

    def model_state(self, model: Any) -> dict[str, Any]:
        if not _is_fsdp_root(model):
            return super().model_state(model)
        from torch.distributed.checkpoint.state_dict import get_model_state_dict

        # Non-zero ranks receive an empty dict; rank 0 writes.
        return get_model_state_dict(model, options=_full(to_cpu=True))

    def optimizer_state(self, model: Any, optimizer: Any) -> dict[str, Any]:
        if not _is_fsdp_root(model):
            return super().optimizer_state(model, optimizer)
        from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict

        return get_optimizer_state_dict(model, optimizer, options=_full(to_cpu=True))

    def load_model_state(self, model: Any, state: dict[str, Any]) -> None:
        if not _is_fsdp_root(model):
            return super().load_model_state(model, state)
        from torch.distributed.checkpoint.state_dict import set_model_state_dict

        # Every rank read the file, so every rank holds the full dict to slice from.
        set_model_state_dict(model, strip_wrapper_prefixes(state), options=_full(to_cpu=False))

    def load_optimizer_state(self, model: Any, optimizer: Any, state: dict[str, Any]) -> None:
        if not _is_fsdp_root(model):
            return super().load_optimizer_state(model, optimizer, state)
        from torch.distributed.checkpoint.state_dict import set_optimizer_state_dict

        set_optimizer_state_dict(model, optimizer, state, options=_full(to_cpu=False))

    @contextlib.contextmanager
    def swap_in(self, model: Any, shadow: Any) -> Generator[None]:
        """EMA swap shard against shard; the shadow must be sharded like ``model``.

        A plain shadow beside a sharded model would need a full gather per swap and
        a full-size copy per rank, which is what sharding exists to avoid, so it is
        refused instead (``shard_like`` builds the matching shadow at Stage A).
        """
        if _is_fsdp_root(model) and not _is_fsdp_root(shadow):
            raise RuntimeError(
                "EMA swap under FSDP needs a shadow sharded like the live model; got an "
                f"unsharded {type(shadow).__name__}. The FSDP plugin's prepare_ema shards it."
            )
        with super().swap_in(model, shadow):
            yield
