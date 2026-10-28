"""The one place a strategy hands bare parameters it built itself to the generator optimizer (#2697).

A module a strategy creates (a conditioning encoder, uncertainty weights) is not in
``env.models``, so DDP, FSDP and ZeRO never wrap it: each rank trains its own copy, or a
shard is read where a whole tensor is expected, and the run reports success. The backend
decision is owned by :mod:`spectramr.infrastructure.distributed.owned_modules`; this helper
asks it at the moment the parameters exist.

Modules that can be DDP-wrapped should use ``register_owned_module`` instead.
"""

from __future__ import annotations

from typing import Any

from spectramr.infrastructure.distributed.owned_modules import refuse_param_group


def add_owned_param_group(
    strategy: Any, optimizer: Any, group: dict[str, Any], *, owner: str
) -> None:
    """``optimizer.add_param_group(group)``, refused where the backend cannot reduce it.

    Args:
        strategy: The strategy registering the parameters; its ``config.parallel`` names
            the backend.
        optimizer: The generator optimizer receiving the group.
        group: The ``add_param_group`` argument.
        owner: What is being registered, for the error message.

    Raises:
        ConfigurationError: ``parallel.strategy`` is ddp, fsdp or deepspeed.
    """
    refuse_param_group(strategy, owner)
    optimizer.add_param_group(group)
