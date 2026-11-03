"""FedAvg / FedBN round controller on a two-layer model, driven by real optimizer steps."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from spectramr.infrastructure.training.strategies.federated_rounds import (
    FederatedRoundController,
    norm_state_keys,
)


def _model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4), nn.Linear(4, 1))


def _step(model: nn.Module, opt: torch.optim.Optimizer, x: torch.Tensor) -> None:
    opt.zero_grad(set_to_none=True)
    model(x).pow(2).mean().backward()
    opt.step()


def _run_block(ctl, model, opt, client, x, steps=2):
    for i in range(steps):
        ctl.before_step(client, block_end=i == steps - 1)
        _step(model, opt, x)


def _setup(aggregation="fed_avg", sizes=None):
    model = _model().train()
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    ctl = FederatedRoundController(model, opt, sizes or {"a": 1, "b": 3}, aggregation)
    return model, opt, ctl


def test_round_aggregates_the_weighted_mean_of_client_results() -> None:
    model, opt, ctl = _setup()
    xa, xb = torch.randn(8, 3), torch.randn(8, 3) + 2.0
    start = {k: v.clone() for k, v in model.state_dict().items()}
    _run_block(ctl, model, opt, "a", xa)
    after_a = {k: v.clone() for k, v in model.state_dict().items()}
    _run_block(ctl, model, opt, "b", xb)
    # b started from the global (initial) weights, not from a's result
    ref, ref_opt = _model().train(), None
    ref.load_state_dict(start)
    ref_opt = torch.optim.SGD(ref.parameters(), lr=0.1, momentum=0.9)
    for _ in range(2):
        _step(ref, ref_opt, xb)
    after_b = ref.state_dict()
    assert ctl.rounds_completed == 1 and ctl.steps_into_round == 0
    for k, v in model.state_dict().items():
        if v.is_floating_point():
            torch.testing.assert_close(v, 0.25 * after_a[k] + 0.75 * after_b[k])


def test_the_optimizer_state_is_reset_at_each_block() -> None:
    model, opt, ctl = _setup()
    _run_block(ctl, model, opt, "a", torch.randn(8, 3))
    ctl.before_step("b", block_end=False)
    assert not opt.state


def test_fed_bn_keeps_each_sites_norm_layers_local() -> None:
    model, opt, ctl = _setup("fed_bn")
    keys = norm_state_keys(model)
    assert keys == {
        f"1.{n}" for n in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked")
    }
    _run_block(ctl, model, opt, "a", torch.randn(8, 3))
    _run_block(ctl, model, opt, "b", torch.randn(8, 3) + 5.0)
    a_mean, b_mean = ctl.local_state["a"]["1.running_mean"], ctl.local_state["b"]["1.running_mean"]
    assert not torch.allclose(a_mean, b_mean)
    # the next block of a starts from the global weights with a's own norms
    ctl.before_step("a", block_end=False)
    torch.testing.assert_close(model.state_dict()["1.running_mean"], a_mean)
    torch.testing.assert_close(model.state_dict()["0.weight"], ctl.global_state["0.weight"])


def test_validation_scores_the_global_model_and_leaves_it_loaded() -> None:
    model, opt, ctl = _setup("fed_bn")
    _run_block(ctl, model, opt, "a", torch.randn(8, 3))
    _run_block(ctl, model, opt, "b", torch.randn(8, 3) + 5.0)
    ctl.before_step("a", block_end=False)
    _step(model, opt, torch.randn(8, 3))  # one local step into round 2
    local = {k: v.clone() for k, v in model.state_dict().items()}
    ctl.begin_validation()
    ctl.use_site("b")
    torch.testing.assert_close(
        model.state_dict()["1.running_mean"], ctl.local_state["b"]["1.running_mean"]
    )
    ctl.end_validation()
    # a checkpoint written now holds the global weights
    torch.testing.assert_close(model.state_dict()["0.weight"], ctl.global_state["0.weight"])
    ctl.before_step("a", block_end=False)  # the in-progress client state comes back
    torch.testing.assert_close(model.state_dict()["0.weight"], local["0.weight"])


def test_checkpoint_refuses_a_mid_round_state_and_round_trips_at_a_boundary() -> None:
    model, opt, ctl = _setup("fed_bn")
    _run_block(ctl, model, opt, "a", torch.randn(8, 3))
    with pytest.raises(RuntimeError, match="round boundary"):
        ctl.state_dict()
    _run_block(ctl, model, opt, "b", torch.randn(8, 3))
    state = ctl.state_dict()
    model2, _opt2, ctl2 = _setup("fed_bn")
    ctl2.load_state_dict(state)
    assert ctl2.rounds_completed == 1
    torch.testing.assert_close(model2.state_dict()["0.weight"], ctl.global_state["0.weight"])


def test_unknown_client_and_unimplemented_rules_raise() -> None:
    model, opt, ctl = _setup()
    with pytest.raises(KeyError, match="unknown client"):
        ctl.before_step("c", block_end=False)
    with pytest.raises(ValueError, match="not implemented"):
        FederatedRoundController(model, opt, {"a": 1, "b": 1}, "scaffold")
    with pytest.raises(ValueError, match="no normalisation"):
        lin = nn.Linear(2, 2)
        FederatedRoundController(
            lin, torch.optim.SGD(lin.parameters(), lr=0.1), {"a": 1, "b": 1}, "fed_bn"
        )
