"""Routes a DP strategy's steps through FedAvg / FedBN rounds (``training.federated_config``).

Inactive unless the arm declares ``training.federated_config``. Then each training batch
names its client, :class:`~..federated_rounds.FederatedRoundController` swaps the model
at block boundaries, DP-SGD is per client (one accountant per site at
``q_k = batch / n_k``, advanced only on that site's steps, ``epsilon = max_k epsilon_k``),
and validation scores each site with its own model and reports ``<metric>/site_<name>``.
The host class supplies the DP state (``DPMixin``) and ``_dp_batch_size``.
"""

from __future__ import annotations

from typing import Any

import torch

from spectramr.data.batch_types import read_batch_field
from spectramr.domain.exceptions import ConfigurationError
from spectramr.infrastructure.services.privacy.rdp_accountant import RDPAccountant

from ..federated_rounds import FederatedRoundController, build_round_controller

__all__ = ["FederatedRoundsMixin"]


class FederatedRoundsMixin:
    """Federated rounds over a DP reconstruction strategy."""

    _fed: FederatedRoundController | None = None
    _fed_accountants: dict[str, RDPAccountant] | None = None
    _fed_sample_rates: dict[str, float] | None = None

    def _federated(self) -> FederatedRoundController | None:
        """The round controller, built on first use (it needs the optimizer and loaders)."""
        training = getattr(self.config, "training", None)
        if getattr(training, "federated_config", None) is None:
            return None
        if self._fed is None:
            loaders = self.env.data_loaders
            train = loaders["train"]
            self._fed = build_round_controller(
                self.config, self.env.generator, self.env.opt_g, train
            )
            sizes = train.client_sizes()
            self._fed_sample_rates = {
                c: min(self._dp_batch_size / n, 1.0) for c, n in sizes.items()
            }
            self._fed_accountants = {c: RDPAccountant() for c in sizes}
            val = loaders.get("val")
            if not hasattr(val, "on_site"):
                raise ConfigurationError("federated validation needs the per-site val loader")
            val.on_site = self._fed.use_site
        return self._fed

    def train_step(
        self,
        batch: Any,
        epoch: int,
        input_batch: torch.Tensor | None = None,
        target_batch: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        fed = self._federated()
        if fed is not None:
            client = read_batch_field(batch, "federated_client")
            if client is None:
                raise RuntimeError("federated round: the batch carries no federated_client tag")
            fed.before_step(client, bool(read_batch_field(batch, "federated_block_end")))
            if self._dp_enabled:
                self._accountant = self._fed_accountants[client]
                self._sample_rate = self._dp_sample_rate = self._fed_sample_rates[client]
        return super().train_step(batch, epoch, input_batch, target_batch, **kwargs)

    def validation_step(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        *,
        batch_data: Any = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        metrics = super().validation_step(
            input_batch, target_batch, batch_data=batch_data, **kwargs
        )
        site = self._fed.val_site if self._fed is not None else None
        if site is None:
            return metrics
        # Per-site columns beside the pooled ones; the loop weighs each key by its own samples.
        return {**metrics, **{f"{k}/site_{site}": v for k, v in metrics.items()}}

    def on_validation_start(self) -> None:
        super().on_validation_start()
        fed = self._federated()
        if fed is not None:
            fed.begin_validation()

    def on_validation_end(self, metrics: dict[str, float]) -> None:
        if self._fed is not None:
            self._fed.end_validation()
        super().on_validation_end(metrics)

    def get_epsilon(self) -> float:
        """Federated: the largest site budget, since each site's data is protected separately."""
        if not self._fed_accountants:
            return super().get_epsilon()
        return max(a.get_epsilon(delta=self._target_delta) for a in self._fed_accountants.values())

    def _resolve_dataset_size(self, priv: Any, data: Any) -> int:
        balancer = (getattr(getattr(self, "env", None), "data_loaders", None) or {}).get("train")
        if hasattr(balancer, "client_sizes"):  # the per-site q is set per block
            return int(sum(balancer.client_sizes().values()))
        return super()._resolve_dataset_size(priv, data)

    def check_strategy_state(self, state: dict[str, Any] | None) -> None:
        if getattr(getattr(self.config, "training", None), "federated_config", None) is not None:
            raise RuntimeError(
                "federated_dp_conformal: resuming a federated run is not supported; the "
                "per-site accountants and the round position are not checkpointed"
            )
        super().check_strategy_state(state)
