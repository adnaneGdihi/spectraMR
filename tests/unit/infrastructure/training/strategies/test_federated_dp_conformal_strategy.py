"""Federated rounds inside FederatedDPConformalULFStrategy: routing, per-site DP, per-site validation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from spectramr.config.schemas.training.federated import FederatedConfig
from spectramr.data.builders.federated_balancer import ChainedDomainLoader, FederatedBlocksBalancer
from spectramr.infrastructure.training.strategies import reconstruction
from spectramr.infrastructure.training.strategies.federated_dp_conformal_strategy import (
    FederatedDPConformalULFStrategy,
)


class _Pairs(torch.utils.data.Dataset):
    def __init__(self, n: int, shift: float) -> None:
        self.n, self.shift = n, shift

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict:
        x = torch.randn(3) + self.shift
        return {"input": x, "target": x}


def _loaders(sizes: dict[str, int]) -> dict[str, DataLoader]:
    return {k: DataLoader(_Pairs(n, i), batch_size=2) for i, (k, n) in enumerate(sizes.items())}


def _strategy(monkeypatch, aggregation="fed_bn", sizes=None, rounds=1):
    sizes = sizes or {"a": 4, "b": 8}
    train = FederatedBlocksBalancer(_loaders(sizes), local_epochs=1)
    val = ChainedDomainLoader(_loaders(dict.fromkeys(sizes, 2)))
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4), nn.Linear(4, 3))
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    s = FederatedDPConformalULFStrategy.__new__(FederatedDPConformalULFStrategy)
    s.device = torch.device("cpu")
    s.config = SimpleNamespace(
        training=SimpleNamespace(
            federated_config=FederatedConfig(
                num_clients=len(sizes), aggregation_strategy=aggregation, num_rounds=rounds
            ),
            max_iterations=None,
            epochs=rounds,
        ),
        data=SimpleNamespace(
            multi_domain=SimpleNamespace(
                enabled=True, balancing="federated_blocks", domains=list(sizes)
            )
        ),
        ema=SimpleNamespace(enabled=False),
    )
    s.env = SimpleNamespace(generator=model, opt_g=opt, data_loaders={"train": train, "val": val})
    s._dp_batch_size = 2
    s.configure_dp(noise_multiplier=1.0, max_grad_norm=1.0, sample_rate=0.5)

    # The parent step builds the reconstruction closures; here one SGD step stands in.
    def _step(self, batch, *args, **kwargs):
        opt.zero_grad(set_to_none=True)
        model(batch["input"]).pow(2).mean().backward()
        opt.step()
        self.record_dp_step()
        return []

    monkeypatch.setattr(reconstruction.ReconstructionTrainingStrategy, "train_step", _step)
    monkeypatch.setattr(
        reconstruction.ReconstructionTrainingStrategy,
        "validation_step",
        lambda self, i, t, **kw: {"psnr": float(model(i).mean())},
    )
    return s, train, val, model


def test_each_site_spends_its_own_dp_budget(monkeypatch) -> None:
    s, train, _val, _model = _strategy(monkeypatch)
    for batch in train:
        s.train_step(batch, 0)
    assert s._fed.rounds_completed == 1
    steps = {k: a.steps for k, a in s._fed_accountants.items()}
    assert steps == {"a": train.block_length("a"), "b": train.block_length("b")}
    assert s._fed_sample_rates == {"a": 0.5, "b": 0.25}
    eps = {k: a.get_epsilon(delta=s._target_delta) for k, a in s._fed_accountants.items()}
    assert s.get_epsilon() == max(eps.values())


def test_validation_scores_each_site_with_its_own_norms(monkeypatch) -> None:
    s, train, val, model = _strategy(monkeypatch)
    for batch in train:
        s.train_step(batch, 0)
    monkeypatch.setattr(
        reconstruction.ReconstructionTrainingStrategy, "on_validation_start", lambda self: None
    )
    monkeypatch.setattr(
        reconstruction.ReconstructionTrainingStrategy, "on_validation_end", lambda self, m: None
    )
    model.eval()  # the validation loop scores in eval mode
    s.on_validation_start()
    seen = {}
    for batch in val:
        out = s.validation_step(batch["input"], batch["target"])
        seen[s._fed.val_site] = out
        bn_mean = model.state_dict()["1.running_mean"]
        torch.testing.assert_close(bn_mean, s._fed.local_state[s._fed.val_site]["1.running_mean"])
    s.on_validation_end({})
    assert set(seen) == {"a", "b"}
    assert set(seen["a"]) == {"psnr", "psnr/site_a"}


def test_resume_is_refused_and_a_non_federated_arm_is_untouched(monkeypatch) -> None:
    s, *_ = _strategy(monkeypatch)
    monkeypatch.setattr(
        reconstruction.ReconstructionTrainingStrategy, "check_strategy_state", lambda self, st: None
    )
    with pytest.raises(RuntimeError, match="not supported"):
        s.check_strategy_state({"counters": {}})
    s.config.training.federated_config = None
    assert s._federated() is None


def test_a_round_count_that_disagrees_with_the_run_length_raises(monkeypatch) -> None:
    s, *_ = _strategy(monkeypatch, rounds=2)
    s.config.training.epochs = 3
    with pytest.raises(ValueError, match="rounds"):
        s._federated()
