"""FedAvg / FedBN rounds over one model, simulated client by client.

The batch order comes from :class:`~spectramr.data.builders.federated_balancer.FederatedBlocksBalancer`:
each client trains a contiguous block of local steps, starting from the global
weights, and one pass over every client is one round. After a client's last local
step its weights are kept; after the last client's, the round is aggregated
(McMahan et al. 2017):

    theta_global = sum_k n_k theta_k / sum_k n_k

FedBN (Li et al. 2021) leaves every normalisation layer's parameters and running
statistics out of the average; each client keeps its own, and validation scores
site ``k`` with site ``k``'s.

State swaps happen at block boundaries only (once per client per round), driven by
an optimizer step post-hook, so a block ends after its last step has been applied.
Validation loads the global model and leaves it loaded, so a checkpoint written
after validation holds the global weights; the in-progress client state comes back
at the next training step.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

__all__ = ["AGGREGATIONS", "FederatedRoundController", "build_round_controller", "norm_state_keys"]

#: The aggregation rules this controller implements.
AGGREGATIONS = ("fed_avg", "fed_bn")

_NORM_TYPES = (nn.modules.batchnorm._NormBase, nn.GroupNorm, nn.LayerNorm)


def norm_state_keys(model: nn.Module) -> frozenset[str]:
    """State-dict keys (parameters and buffers) of every normalisation layer."""
    keys: set[str] = set()
    for prefix, module in model.named_modules():
        if isinstance(module, _NORM_TYPES):
            for name, _ in [
                *module.named_parameters(recurse=False),
                *module.named_buffers(recurse=False),
            ]:
                keys.add(f"{prefix}.{name}" if prefix else name)
    return frozenset(keys)


def _snapshot(state: dict[str, torch.Tensor], keys: Any = None) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in state.items() if keys is None or k in keys}


class FederatedRoundController:
    """Swaps the model between global and per-client state at block boundaries."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        client_sizes: dict[str, int],
        aggregation: str,
    ) -> None:
        if aggregation not in AGGREGATIONS:
            raise ValueError(
                f"aggregation {aggregation!r} is not implemented; one of {AGGREGATIONS}"
            )
        if len(client_sizes) < 2 or any(n < 1 for n in client_sizes.values()):
            raise ValueError(f"need >= 2 non-empty clients, got {client_sizes}")
        self.model, self.optimizer = model, optimizer
        self.clients = list(client_sizes)
        total = float(sum(client_sizes.values()))
        self.weights = {k: n / total for k, n in client_sizes.items()}
        self.local_keys = norm_state_keys(model) if aggregation == "fed_bn" else frozenset()
        if aggregation == "fed_bn" and not self.local_keys:
            raise ValueError("fed_bn: the model has no normalisation layer to keep local")
        self.aggregation = aggregation
        self.global_state = _snapshot(model.state_dict())
        self.local_state: dict[str, dict[str, torch.Tensor]] = {}
        self.results: dict[str, dict[str, torch.Tensor]] = {}
        self.current: str | None = None
        self.rounds_completed = 0
        self.steps_into_round = 0
        self._block_pending = False
        self._held: dict[str, torch.Tensor] | None = None
        self.val_site: str | None = None
        optimizer.register_step_post_hook(self._after_step)

    # -- training ------------------------------------------------------------
    def before_step(self, client: str, block_end: bool) -> None:
        """Called by the strategy with each training batch's client tag."""
        if self._held is not None:  # validation left the global model loaded
            self.model.load_state_dict(self._held)
            self._held = None
        if self._block_pending:  # an AMP-skipped last step never fired the hook
            self._close_block()
        if client not in self.weights:
            raise KeyError(f"batch from unknown client {client!r}; clients: {self.clients}")
        if client != self.current:
            self._open_block(client)
        self._block_pending = block_end
        self.steps_into_round += 1

    def _after_step(self, _optimizer: Any, _args: Any, _kwargs: Any) -> None:
        if self._block_pending:
            self._close_block()

    def _open_block(self, client: str) -> None:
        state = dict(self.global_state)
        state.update(self.local_state.get(client, {}))
        self.model.load_state_dict(state)
        self.optimizer.state.clear()  # local optimiser state starts fresh each block
        self.current = client

    def _close_block(self) -> None:
        self._block_pending = False
        state = self.model.state_dict()
        self.results[self.current] = _snapshot(state)
        if self.local_keys:
            self.local_state[self.current] = _snapshot(state, self.local_keys)
        if len(self.results) == len(self.clients):
            self._aggregate()

    def _aggregate(self) -> None:
        new = {}
        for key, ref in self.global_state.items():
            if not ref.is_floating_point():  # num_batches_tracked and other counters
                new[key] = self.results[self.clients[0]][key].clone()
                continue
            new[key] = sum(self.weights[c] * self.results[c][key] for c in self.clients)
        self.global_state = new  # FedBN: the norm keys hold the n_k-weighted mean
        self.results.clear()
        self.rounds_completed += 1
        self.steps_into_round = 0
        self.current = None
        self.model.load_state_dict(self.global_state)

    # -- validation ------------------------------------------------------------
    def begin_validation(self) -> None:
        """Load the global model; it stays loaded until the next training step."""
        if self._held is None:
            self._held = _snapshot(self.model.state_dict())
        self.model.load_state_dict(self.global_state)

    def use_site(self, client: str) -> None:
        """FedBN: score ``client``'s validation batches with its own norm layers."""
        if client not in self.weights:
            raise KeyError(f"validation batch from unknown site {client!r}; sites: {self.clients}")
        if not self.local_keys:
            return
        self.val_site = client
        # Before a site's first block its norm layers are still the shared initial ones.
        self.model.load_state_dict({**self.global_state, **self.local_state.get(client, {})})

    def end_validation(self) -> None:
        self.val_site = None
        self.model.load_state_dict(self.global_state)

    # -- checkpointing ---------------------------------------------------------
    def refuse_mid_round(self) -> None:
        """A checkpoint only replays at a round boundary; the balancer cannot seek."""
        if self.steps_into_round:
            raise RuntimeError(
                f"federated checkpoint taken {self.steps_into_round} steps into round "
                f"{self.rounds_completed + 1}; resume needs a round boundary"
            )

    def state_dict(self) -> dict[str, Any]:
        """Round-boundary state: the global weights, the per-site norms and the round count."""
        self.refuse_mid_round()
        return {
            "rounds_completed": self.rounds_completed,
            "global_state": self.global_state,
            "local_state": self.local_state,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if set(state.get("local_state", {})) - set(self.clients):
            raise ValueError(f"checkpoint sites {sorted(state['local_state'])} != {self.clients}")
        self.rounds_completed = int(state["rounds_completed"])
        self.global_state = _snapshot(state["global_state"])
        self.local_state = {k: _snapshot(v) for k, v in state["local_state"].items()}
        self.model.load_state_dict(self.global_state)


def build_round_controller(
    config: Any, model: nn.Module, optimizer: torch.optim.Optimizer, train_loader: Any
) -> FederatedRoundController:
    """Check the arm describes one round per epoch over its declared sites, then build."""
    fed = config.training.federated_config
    md = getattr(config.data, "multi_domain", None)
    if md is None or not md.enabled or md.balancing != "federated_blocks":
        raise ValueError(
            "training.federated_config needs data.multi_domain.enabled with balancing "
            "'federated_blocks': the rounds are the balancer's client blocks"
        )
    if fed.num_clients != len(md.domains):
        raise ValueError(
            f"federated_config.num_clients={fed.num_clients} but multi_domain declares "
            f"{len(md.domains)} sites"
        )
    planned = config.training.max_iterations
    if planned is None or planned <= 0:
        planned = (config.training.epochs or 0) * len(train_loader)
    if planned != fed.num_rounds * len(train_loader):
        raise ValueError(
            f"{fed.num_rounds} rounds of {len(train_loader)} steps is "
            f"{fed.num_rounds * len(train_loader)} steps, but the run is planned for "
            f"{planned}; set training.epochs to num_rounds and leave max_iterations unset"
        )
    if getattr(getattr(config, "ema", None), "enabled", False):
        raise ValueError("ema.enabled: an EMA over block-swapped weights mixes clients; disable it")
    dist = torch.distributed
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        raise ValueError("federated rounds are simulated in one process; run on one rank")
    return FederatedRoundController(
        model, optimizer, train_loader.client_sizes(), fed.aggregation_strategy
    )
