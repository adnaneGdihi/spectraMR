"""Backend decision for state a training strategy builds for itself (#2629).

The parallel wrappers reach only ``env.models``, and a strategy builds its own heads
after the optimizers exist, so nothing can wrap them in Stage A. This module is the one
owner of the answer to "may a strategy-built parameter run on backend X": the runtime
helpers (:func:`wrap_owned_module`, ``add_owned_param_group``) and the static gate
(``strategy_compat``) both call :func:`owned_verdict`.

Four mechanisms exist. A ``MODULE`` is registered with ``register_owned_module`` and is
DDP-wrapped under ``ddp``. A ``PARAM_GROUP`` is bare parameters added to the generator
optimizer, which DDP never reduces, so ``ddp`` refuses it. A ``CALL`` is a non-forward
method a strategy calls on a wrapped ``env`` model, which skips the reducer, so every
multi-process backend refuses it. ``BUFFERS`` are state a strategy adds to a wrapped
``env`` model in place (PnP's spectral-norm vectors): DDP broadcasts only the buffers it saw
at construction, so under ``ddp`` they are broadcast from rank 0 once, when added. Under ``fsdp`` and every ``deepspeed`` stage all three
are refused, because the optimizer is initialised before the module exists and ZeRO never
sees its parameters. The ZeRO stage does not change the answer, so it is not an input.

A strategy may declare a ``MODULE`` only if every rank reaches its construction, and every
gradient-carrying call, at the same step whatever its batch holds: in ``__init__``, or on
a step all ranks run unconditionally. DDP construction is a collective, so a build or
rebuild gated on a batch key or a batch shape can leave one rank in it while another
moves on. A size read from the first batch is allowed, because DDP checks parameter
shapes across ranks at construction and raises. A module that fails this test is
declared a ``PARAM_GROUP``, which ``ddp`` refuses.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn as nn

from spectramr.domain.exceptions import ConfigurationError

OWNED_MODULE = "module"
OWNED_PARAM_GROUP = "param_group"
OWNED_CALL = "call"
OWNED_BUFFERS = "buffers"
OwnedMechanism = Literal["module", "param_group", "call", "buffers"]

Verdict = Literal["wrap", "bare", "refuse"]


class OwnedParameter(nn.Module):
    """Holds one bare parameter a strategy owns, so DDP can wrap it; forward returns it.

    The strategy keeps the parameter as its own attribute and does not store the holder,
    so ``strategy_state_dict`` still saves it under ``params.<name>``.
    """

    def __init__(self, param: nn.Parameter) -> None:
        super().__init__()
        self.value = param

    def forward(self) -> torch.Tensor:
        return self.value


def backend_of(parallel: Any) -> str:
    """The ``parallel.strategy`` name, ``"none"`` for a single-process run."""
    return getattr(parallel, "strategy", "none") if parallel is not None else "none"


def owned_verdict(mechanism: OwnedMechanism, backend: str) -> Verdict:
    """Decide how a strategy-built thing runs under ``backend``.

    Raises:
        ConfigurationError: ``backend`` is not a known ``parallel.strategy``.
    """
    if backend in ("none", "dp"):
        return "bare"
    if backend == "ddp":
        return "wrap" if mechanism in (OWNED_MODULE, OWNED_BUFFERS) else "refuse"
    if backend in ("fsdp", "deepspeed"):
        return "refuse"
    raise ConfigurationError(f"unknown parallel.strategy {backend!r} for an owned module")


def refusal_reason(mechanism: OwnedMechanism, backend: str) -> str:
    """Why ``backend`` refuses ``mechanism``, for the audit and the runtime messages."""
    if mechanism == OWNED_CALL:
        return (
            "the call bypasses the wrapper's forward, so the gradient all-reduce (or the "
            "sharded all-gather) never runs for it"
        )
    if mechanism == OWNED_BUFFERS:
        return (
            "it reparameterises the wrapped model's weights in place after the engine took "
            "them over, which the engine does not see"
        )
    if backend == "fsdp":
        return "FSDP shards parameters and this module is built after the sharding wrap"
    if backend == "deepspeed":
        return (
            "DeepSpeed initialises the optimizer before this module exists, so ZeRO "
            "never sees its parameters"
        )
    return "DDP never reduces a bare parameter group, so each rank would train its own copy"


def refuse_outside_forward_call(parallel: Any, *, owner: str, name: str) -> None:
    """Raise if ``backend`` cannot run the non-forward call ``name`` on a wrapped model.

    Raises:
        ConfigurationError: ``parallel.strategy`` is ddp, fsdp or deepspeed, or unknown.
    """
    backend = backend_of(parallel)
    if owned_verdict(OWNED_CALL, backend) == "refuse":
        raise ConfigurationError(
            f"{owner} calls {name!r} outside the wrapped forward under "
            f"parallel.strategy={backend!r}: {refusal_reason(OWNED_CALL, backend)}. "
            "Turn the feature off, or use parallel.strategy: none or dp."
        )


def sync_owned_buffers(
    buffers: list[torch.Tensor], *, owner: str, name: str, parallel: Any
) -> None:
    """Make ``buffers`` added to a wrapped model identical on every rank, or raise.

    Every rank must call this at the same point, since under ``ddp`` it broadcasts.

    Raises:
        ConfigurationError: ``parallel.strategy`` is fsdp or deepspeed, or unknown.
    """
    backend = backend_of(parallel)
    verdict = owned_verdict(OWNED_BUFFERS, backend)
    if verdict == "refuse":
        raise ConfigurationError(
            f"{owner} adds {name!r} to the wrapped model under parallel.strategy={backend!r}: "
            f"{refusal_reason(OWNED_BUFFERS, backend)}. Use parallel.strategy: none, dp or "
            "ddp, or turn the feature off."
        )
    if verdict == "wrap":
        import torch.distributed as dist

        for buf in buffers:
            dist.broadcast(buf, src=0)


def sync_frozen_module(module: nn.Module) -> None:
    """Give a frozen module a strategy built outside ``env.models`` rank 0's values.

    A frozen module is never reduced, so any backend may hold one, but every rank must
    run the same one. A partial checkpoint load leaves the unmatched tensors at each
    rank's own seeded initialisation. A no-op without a multi-rank process group. Every
    rank must call it at the same point.
    """
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() < 2:
        return
    with torch.no_grad():
        for tensor in [*module.parameters(), *module.buffers()]:
            dist.broadcast(tensor.data, src=0)


def wrap_owned_module(
    module: nn.Module, *, owner: str, name: str, parallel: Any, device: Any
) -> nn.Module | None:
    """Return the DDP wrapper for ``module``, ``None`` if it runs unwrapped, or raise.

    Args:
        module: The module the strategy just built.
        owner: The strategy class name, for the message.
        name: The attribute the module is registered under.
        parallel: ``config.parallel``, or ``None`` for a single-process run.
        device: The strategy's device.

    Raises:
        ConfigurationError: ``parallel.strategy`` is fsdp or deepspeed, or unknown.
    """
    backend = backend_of(parallel)
    verdict = owned_verdict(OWNED_MODULE, backend)
    if verdict == "bare":
        return None
    if verdict == "wrap":
        from spectramr.infrastructure.distributed.strategies import wrap_with_ddp

        return wrap_with_ddp(module, parallel, device)
    raise ConfigurationError(
        f"{owner} builds {name!r} itself, outside env.models, after the parallel wrap "
        f"({refusal_reason(OWNED_MODULE, backend)}). parallel.strategy={backend!r} would leave it unsynced or sharded. "
        "Use parallel.strategy: ddp or none, or turn the feature off."
    )


def refuse_param_group(strategy: Any, owner: str) -> None:
    """Raise if ``strategy``'s backend cannot reduce a bare parameter group.

    Raises:
        ConfigurationError: ``parallel.strategy`` is ddp, fsdp or deepspeed.
    """
    backend = backend_of(getattr(getattr(strategy, "config", None), "parallel", None))
    if owned_verdict(OWNED_PARAM_GROUP, backend) == "refuse":
        raise ConfigurationError(
            f"{type(strategy).__name__} builds {owner} itself, outside env.models, so "
            f"parallel.strategy={backend!r} would not reduce its gradients and each rank would "
            "train its own copy. Turn the feature off, or use parallel.strategy: none or dp."
        )
