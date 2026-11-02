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
        state = optimizer.state_dict()
        if not _tensor_parallel(model):
            return state
        from spectramr.infrastructure.distributed.tensor_parallel.optim_state import (
            rewrite_packed_optimizer_state,
        )

        # Every tensor rank enters (a mesh makes checkpoints collective); rank 0 writes.
        return rewrite_packed_optimizer_state(model, optimizer, state, gather=True)

    def load_model_state(self, model: Any, state: dict[str, Any]) -> None:
        unwrap_model(model).load_state_dict(strip_wrapper_prefixes(state))

    def load_optimizer_state(self, model: Any, optimizer: Any, state: dict[str, Any]) -> None:
        if _tensor_parallel(model):
            from spectramr.infrastructure.distributed.tensor_parallel.optim_state import (
                rewrite_packed_optimizer_state,
            )

            state = rewrite_packed_optimizer_state(model, optimizer, state, gather=False)
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


def _tensor_parallel(model: Any) -> bool:
    from spectramr.infrastructure.distributed.tensor_parallel.optim_state import (
        has_tensor_parallel_plan,
    )

    return has_tensor_parallel_plan(model)


def _is_fsdp_root(model: Any) -> bool:
    from torch.distributed.fsdp import FSDPModule

    return isinstance(model, FSDPModule)


def _wrapper_prefix(model: Any) -> str:
    """The key prefix a depth-split root adds above the unsplit model's keys, else ``""``."""
    from spectramr.infrastructure.distributed.spatial.wrapper import SpatialModel

    return "module." if isinstance(model, SpatialModel) else ""


def _rekey_optimizer(state: dict[str, Any], rename: Any) -> dict[str, Any]:
    """Apply ``rename`` to the parameter FQNs of a full optimizer state dict."""
    groups = [{**g, "params": [rename(k) for k in g["params"]]} for g in state["param_groups"]]
    return {"state": {rename(k): v for k, v in state["state"].items()}, "param_groups": groups}


def _full(*, to_cpu: bool) -> Any:
    from torch.distributed.checkpoint.state_dict import StateDictOptions

    # cpu_offload with a full state dict also makes the gather rank-0-only.
    return StateDictOptions(full_state_dict=True, cpu_offload=to_cpu)


def _to_cpu(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _to_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_cpu(v) for v in value]
    return value.cpu() if hasattr(value, "cpu") else value


class FSDPStateIO(PassthroughStateIO):
    """Gather on save, scatter on load, on an FSDP2 module.

    A model that is not sharded (a plain discriminator beside a sharded generator)
    takes the passthrough path.
    """

    def model_state(self, model: Any) -> dict[str, Any]:
        if not _is_fsdp_root(model):
            return super().model_state(model)
        from torch.distributed.checkpoint.state_dict import get_model_state_dict

        # Non-zero ranks receive an empty dict; rank 0 writes. The unsplit model's keys.
        prefix = _wrapper_prefix(model)
        state = get_model_state_dict(model, options=_full(to_cpu=True))
        return {k.removeprefix(prefix): v for k, v in state.items()}

    def optimizer_state(self, model: Any, optimizer: Any) -> dict[str, Any]:
        if not _is_fsdp_root(model):
            return super().optimizer_state(model, optimizer)
        from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict

        prefix = _wrapper_prefix(model)
        if not _tensor_parallel(model):
            state = get_optimizer_state_dict(model, optimizer, options=_full(to_cpu=True))
        else:
            import torch.distributed as dist

            from spectramr.infrastructure.distributed.tensor_parallel.optim_state import (
                gather_split_optimizer_state,
            )

            # Gathered over dp on every rank, then over tp, which a rank-0-only gather would
            # deadlock; the transient full copy per rank is the price of that order.
            state = get_optimizer_state_dict(model, optimizer, options=_full(to_cpu=False))
            gather_split_optimizer_state(model, state["state"])
            state = _to_cpu(state) if dist.get_rank() == 0 else {}
        if not prefix or not state:
            return state
        return _rekey_optimizer(state, lambda k: k.removeprefix(prefix))

    def load_model_state(self, model: Any, state: dict[str, Any]) -> None:
        if not _is_fsdp_root(model):
            return super().load_model_state(model, state)
        from torch.distributed.checkpoint.state_dict import set_model_state_dict

        # Every rank read the file, so every rank holds the full dict to slice from.
        prefix = _wrapper_prefix(model)
        bare = strip_wrapper_prefixes(state)
        set_model_state_dict(
            model, {prefix + k: v for k, v in bare.items()}, options=_full(to_cpu=False)
        )

    def load_optimizer_state(self, model: Any, optimizer: Any, state: dict[str, Any]) -> None:
        if not _is_fsdp_root(model):
            return super().load_optimizer_state(model, optimizer, state)
        from torch.distributed.checkpoint.state_dict import set_optimizer_state_dict

        if _tensor_parallel(model):
            from spectramr.infrastructure.distributed.tensor_parallel.optim_state import (
                slice_split_optimizer_state,
            )

            state = {**state, "state": dict(state["state"])}
            slice_split_optimizer_state(model, state["state"])
        prefix = _wrapper_prefix(model)
        if prefix:
            state = _rekey_optimizer(state, lambda k: prefix + k)
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
