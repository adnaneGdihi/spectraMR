"""Client-blocked federated batch order and the per-site validation chain."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from spectramr.data.builders.federated_balancer import ChainedDomainLoader, FederatedBlocksBalancer
from spectramr.data.builders.site_balancer import build_balancer


class _Dicts(torch.utils.data.Dataset):
    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict:
        return {"input": torch.tensor([float(i)]), "target": torch.tensor([float(i)])}


def _loaders(**sizes: int) -> dict[str, DataLoader]:
    return {k: DataLoader(_Dicts(n), batch_size=2) for k, n in sizes.items()}


def test_each_client_runs_one_contiguous_block_per_round() -> None:
    bal = FederatedBlocksBalancer(_loaders(a=4, b=6), local_epochs=2)
    items = list(bal)
    order = [it["federated_client"] for it in items]
    assert order == ["a"] * 4 + ["b"] * 6
    assert [it["federated_block_end"] for it in items].count(True) == 2
    assert items[3]["federated_block_end"] and items[9]["federated_block_end"]
    assert len(bal) == len(items) == 10
    assert bal.client_sizes() == {"a": 4, "b": 6}
    assert "input" in items[0] and bal.per_domain_validation


def test_blocks_refuse_bad_inputs() -> None:
    with pytest.raises(ValueError, match="local_epochs"):
        FederatedBlocksBalancer(_loaders(a=2, b=2), local_epochs=0)
    tensors = {k: DataLoader(TensorDataset(torch.zeros(2, 1)), batch_size=1) for k in "ab"}
    with pytest.raises(TypeError, match="collated dict"):
        list(FederatedBlocksBalancer(tensors, local_epochs=1))


def test_build_balancer_routes_local_epochs_only_to_federated_blocks() -> None:
    bal = build_balancer(_loaders(a=2, b=2), "federated_blocks", {"a": 1.0, "b": 1.0}, 3)
    assert isinstance(bal, FederatedBlocksBalancer) and bal.local_epochs == 3
    with pytest.raises(ValueError, match="only 'federated_blocks' reads it"):
        build_balancer(_loaders(a=2, b=2), "round_robin", None, 2)
    with pytest.raises(ValueError, match="n_k"):
        build_balancer(_loaders(a=2, b=2), "federated_blocks", {"a": 1.0, "b": 3.0}, 1)
    with pytest.raises(ValueError, match="federated_config"):
        build_balancer(_loaders(a=2, b=2), "federated_blocks", None, None)


def test_chained_validation_announces_each_site_before_its_batches() -> None:
    chain = ChainedDomainLoader(_loaders(a=2, b=4))
    with pytest.raises(RuntimeError, match="on_site"):
        next(iter(chain))
    events = []
    chain.on_site = lambda name: events.append(("site", name))
    for _batch in chain:
        events.append(("batch", None))
    assert events == [
        ("site", "a"),
        ("batch", None),
        ("site", "b"),
        ("batch", None),
        ("batch", None),
    ]
    assert len(chain) == 3
