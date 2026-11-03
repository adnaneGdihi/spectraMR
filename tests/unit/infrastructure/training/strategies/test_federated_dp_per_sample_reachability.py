"""Reachability: the promoted arm resolves to the DP strategy and a real step clips per sample.

Drives ``exp_promoted_federated_dp.yaml`` -> ``TrainingStrategyFactory`` ->
``FederatedDPConformalULFStrategy`` -> ``StepExecutor.execute_step`` on a tiny CPU
model. Only the parent's loss is replaced with an L1 on a 1x1 conv, so no
generator is built.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from spectramr.config.settings import TrainingSettings
from spectramr.domain.exceptions import ConfigurationError
from spectramr.infrastructure.training.optimizers import AMPPolicy
from spectramr.infrastructure.training.step_executor import StepExecutor
from spectramr.infrastructure.training.strategies import federated_dp_conformal_strategy as fdp
from spectramr.infrastructure.training.strategies.reconstruction import (
    ReconstructionTrainingStrategy,
)
from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory

ARM = Path(__file__).parents[5] / "experiments/inprogress/promoted/exp_promoted_federated_dp.yaml"


def _strategy(settings: TrainingSettings) -> fdp.FederatedDPConformalULFStrategy:
    cls = TrainingStrategyFactory().get_strategy_class(settings)
    assert cls is fdp.FederatedDPConformalULFStrategy
    s = cls.__new__(cls)
    s.device = torch.device("cpu")
    s.config = settings
    gen = torch.nn.Conv2d(1, 1, 1, bias=False)
    with torch.no_grad():
        gen.weight.fill_(1.0)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.zeros(400, 1)))
    s.env = SimpleNamespace(
        generator=gen, opt_g=torch.optim.SGD(gen.parameters(), lr=0.1), train_loader=loader
    )
    s._configure_from_settings()
    return s


def _l1_loss(self, input_batch, target_batch, epoch, **kwargs):
    pred = self.env.generator(input_batch)
    loss = (pred - target_batch).abs().mean()
    return {"g_total_loss": loss, "l1": loss}, {}


def test_arm_step_clips_each_example_then_noises_and_accounts(monkeypatch) -> None:
    settings = TrainingSettings.from_yaml(ARM)
    assert settings.metadata.status != "needs_implementation"
    s = _strategy(settings)
    # YAML knobs reached the strategy.
    assert (s._noise_multiplier, s._max_grad_norm, s._dp_batch_size) == (1.1, 1.0, 2)

    monkeypatch.setattr(ReconstructionTrainingStrategy, "_compute_losses", _l1_loss)
    seen: list[torch.Tensor] = []

    def _record_noise(g, **kw):
        seen.append(g.clone())
        assert kw["batch_size"] == 2
        return g

    monkeypatch.setattr(fdp, "add_gaussian_noise", _record_noise)

    # Example 0 input 50 -> grad 50 (clipped to C=1); example 1 input 0.5 -> grad 0.5 (kept).
    inp = torch.stack([torch.full((1, 2, 2), 50.0), torch.full((1, 2, 2), 0.5)])
    tgt = torch.full((2, 1, 2, 2), -1000.0)
    helper = MagicMock(device_type="cpu", enabled=False, scaler=None)
    helper.get_autocast_context = contextlib.nullcontext
    executor = StepExecutor(
        amp_helper=helper, amp_policy=AMPPolicy(max_grad_norm=1.0, enable_gradient_clipping=False)
    )
    executor.execute_step(
        [
            {
                "optimizer": s.env.opt_g,
                "closure": lambda: s._compute_losses(inp, tgt, 0)[0]["g_total_loss"],
                "model": s.env.generator,
                "name": "generator",
            }
        ],
        epoch=0,
        global_step=0,
        clip_and_log_fn=s._clip_and_log_gradients,
    )

    # The hook sees the mean of per-sample clipped grads, (1 + 0.5) / 2, not a batch-level clip (1.0).
    assert len(seen) == 1 and torch.allclose(seen[0], torch.tensor(0.75).view(1, 1, 1, 1))
    assert s.accountant.steps == 1
    assert torch.allclose(s.env.generator.weight, torch.tensor(1.0 - 0.1 * 0.75).view(1, 1, 1, 1))


def _executor() -> StepExecutor:
    helper = MagicMock(device_type="cpu", enabled=False, scaler=None)
    helper.get_autocast_context = contextlib.nullcontext
    return StepExecutor(
        amp_helper=helper, amp_policy=AMPPolicy(max_grad_norm=1.0, enable_gradient_clipping=False)
    )


def _real_step(s: fdp.FederatedDPConformalULFStrategy) -> None:
    """``train_step`` builds the closure, ``execute_step`` runs it and the noise hook."""
    s.env.discriminator = None
    s.adapter_chains = {}
    s.save_debug_snapshot = lambda *a, **k: None
    inp = torch.stack([torch.full((1, 2, 2), 50.0), torch.full((1, 2, 2), 0.5)])
    configs = s.train_step(None, 0, input_batch=inp, target_batch=torch.zeros(2, 1, 2, 2))
    _executor().execute_step(
        configs, epoch=0, global_step=0, clip_and_log_fn=s._clip_and_log_gradients
    )


def test_real_step_reports_the_budget_in_the_step_metrics(monkeypatch) -> None:
    s = _strategy(TrainingSettings.from_yaml(ARM))
    monkeypatch.setattr(ReconstructionTrainingStrategy, "_compute_losses", _l1_loss)
    for _ in range(2):
        _real_step(s)
    metrics = s.get_last_metrics()
    # The loop's CSV header keeps only declared keys, so every stamped dp/* key must be declared.
    dp_keys = {k for k in metrics if k.startswith("dp/")}
    assert dp_keys == s.declared_metric_keys() and "g_total_loss" in metrics
    assert metrics["dp/steps"] == 2.0 and metrics["dp/epsilon"] == pytest.approx(s.get_epsilon())
    assert metrics["dp/epsilon"] > 0 and metrics["dp/sample_rate"] == 2 / 400


def test_checkpoint_round_trip_continues_the_accountant(monkeypatch) -> None:
    s = _strategy(TrainingSettings.from_yaml(ARM))
    monkeypatch.setattr(ReconstructionTrainingStrategy, "_compute_losses", _l1_loss)
    for _ in range(3):
        _real_step(s)
    state = s.strategy_state_dict()
    assert state["counters"]["_dp_accountant_steps"] == 3

    resumed = _strategy(TrainingSettings.from_yaml(ARM))
    resumed.check_strategy_state(state)
    resumed.load_strategy_state_dict(state)
    assert resumed.accountant.steps == 3
    assert resumed.get_epsilon() == pytest.approx(s.get_epsilon(), rel=1e-12)

    # A checkpoint without the ledger, or with a different q, would misreport epsilon.
    for bad in ({}, {**state["counters"], "_dp_dataset_size": 800}):
        with pytest.raises(RuntimeError, match="ledger"):
            _strategy(TrainingSettings.from_yaml(ARM)).check_strategy_state({"counters": bad})


def test_large_example_is_clipped_alone(monkeypatch) -> None:
    s = _strategy(TrainingSettings.from_yaml(ARM))
    monkeypatch.setattr(ReconstructionTrainingStrategy, "_compute_losses", _l1_loss)
    # Example 0 input 50 -> grad 50 (clipped to 1); example 1 input 0.5 -> grad 0.5 (kept).
    inp = torch.stack([torch.full((1, 1, 1), 50.0), torch.full((1, 1, 1), 0.5)])
    tgt = torch.full((2, 1, 1, 1), -1000.0)
    losses = s._compute_losses(inp, tgt, 0)[0]
    loss = losses["g_total_loss"]
    loss.backward()
    # Logged components are batch means, not the last example's value.
    assert torch.isclose(losses["l1"], torch.tensor((1050.0 + 1000.5) / 2))
    # Per-sample: (1 + 0.5) / 2 = 0.75. Batch-level clip of the mean (25.25) would give 1.0.
    assert torch.allclose(s.env.generator.weight.grad, torch.tensor(0.75).view(1, 1, 1, 1))
    assert torch.isclose(loss, torch.tensor((1050.0 + 1000.5) / 2))
    assert s._dp_per_sample_clipped


def test_validation_path_is_untouched(monkeypatch) -> None:
    s = _strategy(TrainingSettings.from_yaml(ARM))
    monkeypatch.setattr(ReconstructionTrainingStrategy, "_compute_losses", _l1_loss)
    with torch.no_grad():
        s._compute_losses(torch.ones(2, 1, 1, 1), torch.zeros(2, 1, 1, 1), 0)
    assert not s._dp_per_sample_clipped


@pytest.mark.parametrize(
    "patch",
    [
        {"optimization": {"precision": {"enabled": True}}},
        {"optimization": {"gradient": {"accumulation_steps": 2}}},
    ],
)
def test_settings_that_break_per_sample_accounting_raise(patch) -> None:
    settings = TrainingSettings.from_yaml(ARM)
    block, inner = next(iter(patch["optimization"].items()))
    opt = settings.optimization
    new_block = getattr(opt, block).model_copy(update=inner)
    s = fdp.FederatedDPConformalULFStrategy.__new__(fdp.FederatedDPConformalULFStrategy)
    s.config = SimpleNamespace(optimization=opt.model_copy(update={block: new_block}))
    with pytest.raises(ConfigurationError):
        s._require_per_sample_compatible_optimisation(s.config)
