"""Client-blocked batch order for federated rounds, and per-site validation.

:class:`FederatedBlocksBalancer` yields every batch of client ``k`` for
``local_epochs`` passes before moving to client ``k + 1``; one pass over all
clients is one round, and one round is one training epoch. Each item is the
client's own collated batch plus two host-side fields the loop keeps in batch
metadata: ``federated_client`` (its name) and ``federated_block_end`` (its last
local batch), so the round controller swaps state without a device read.

:class:`ChainedDomainLoader` runs each site's validation loader in turn and calls
``on_site(name)`` before a site's first batch, so every batch is scored with that
site's model. Iterating it with no ``on_site`` consumer raises: unswapped, the
"per-site" pass would silently score one model on the pooled data.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import Any

from torch.utils.data import DataLoader

from spectramr.data.builders.site_balancer import Balancer, _weights_are_uniform

__all__ = ["ChainedDomainLoader", "FederatedBlocksBalancer", "federated_blocks"]


class FederatedBlocksBalancer(Balancer):
    """One contiguous block per client per round, in declaration order."""

    #: The validation half must be per-site too (see ``ChainedDomainLoader``).
    per_domain_validation = True

    def __init__(self, loaders: dict[str, DataLoader], local_epochs: int) -> None:
        super().__init__(loaders)
        if int(local_epochs) < 1:
            raise ValueError(f"local_epochs must be >= 1, got {local_epochs}")
        for name, loader in loaders.items():
            if len(loader) < 1:
                raise ValueError(f"client {name!r} has an empty train loader")
        self.local_epochs = int(local_epochs)

    def block_length(self, name: str) -> int:
        return self.local_epochs * len(self.loaders[name])

    def client_sizes(self) -> dict[str, int]:
        """Samples per client, the FedAvg weight ``n_k``."""
        return {name: len(loader.dataset) for name, loader in self.loaders.items()}

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for name in self.names:
            remaining = self.block_length(name)
            for _ in range(self.local_epochs):
                for batch in self.loaders[name]:
                    if not isinstance(batch, Mapping):
                        raise TypeError(
                            f"client {name!r} yields {type(batch).__name__}; "
                            "federated_blocks tags collated dict batches"
                        )
                    remaining -= 1
                    yield {
                        **batch,
                        "federated_client": name,
                        "federated_block_end": remaining == 0,
                    }

    def __len__(self) -> int:
        return sum(self.block_length(name) for name in self.names)


class ChainedDomainLoader:
    """Each site's validation loader in turn, every batch tagged with its site."""

    def __init__(self, loaders: dict[str, DataLoader]) -> None:
        if not loaders:
            raise ValueError("ChainedDomainLoader needs at least one site loader")
        self.loaders = loaders
        self.on_site: Callable[[str], None] | None = None

    def __iter__(self) -> Iterator[Any]:
        if self.on_site is None:
            raise RuntimeError(
                "per-site validation loader iterated with no on_site consumer; the "
                "strategy must swap each site's model in (FederatedRoundController.use_site)"
            )
        for name, loader in self.loaders.items():
            self.on_site(name)
            yield from loader

    def __len__(self) -> int:
        return sum(len(loader) for loader in self.loaders.values())


def federated_blocks(
    loaders: dict[str, DataLoader], weights: Any, local_epochs: int | None
) -> FederatedBlocksBalancer:
    """The ``federated_blocks`` balancing factory: clients weigh by ``n_k``, not ``weight``."""
    if not _weights_are_uniform(weights):
        raise ValueError(
            "federated_blocks weighs clients by sample count (FedAvg n_k); "
            f"per-domain weights {weights} are not read here"
        )
    if local_epochs is None:
        raise ValueError("balancing 'federated_blocks' needs training.federated_config")
    return FederatedBlocksBalancer(loaders, local_epochs)
