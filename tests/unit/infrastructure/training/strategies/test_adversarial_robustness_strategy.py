"""Tests for AdversarialRobustnessStrategy loss-dict key normalisation.

Regression for the A-8.3 review BLOCKER (2026-06-16): the merge read ``loss_total``
from the parent's dict, but ReconstructionTrainingStrategy emits ``g_total_loss``
(+ a ``loss`` alias), so ``loss_total`` was always None — the merged dict ended up
with NO grad-carrying total, which (a) would raise "No g_total_loss" downstream and
(b) made the CertRob Lipschitz penalty inert (#16). The fix reads the canonical
``g_total_loss`` and writes it back.
"""

from __future__ import annotations

import types

import pytest
import torch
from torch import nn

from spectramr.infrastructure.training.strategies.adversarial_robustness_strategy import (
    AdversarialRobustnessStrategy,
)


def _make_strategy(gen: nn.Module) -> AdversarialRobustnessStrategy:
    strat = object.__new__(AdversarialRobustnessStrategy)
    strat.env = types.SimpleNamespace(generator=gen)
    strat.clean_weight = 0.1
    strat.adv_weight = 0.9
    strat.epsilon = 0.01
    strat.attack_steps = 2
    strat.attack_norm = "linf"
    return strat


def test_merge_emits_grad_carrying_g_total_loss(monkeypatch) -> None:
    import spectramr.infrastructure.training.strategies.adversarial_robustness_strategy as mod

    # The grandparent (ReconstructionTrainingStrategy) emits g_total_loss, NOT loss_total.
    def _fake_super(self, input_batch=None, target_batch=None, epoch=0, **k):
        return {"g_total_loss": (input_batch.sum() * 0.0 + 1.0).requires_grad_(True), "l1": torch.tensor(0.5)}

    monkeypatch.setattr(mod.ReconstructionTrainingStrategy, "_compute_losses_impl", _fake_super)
    # Skip the actual PGD attack — return the input unchanged.
    monkeypatch.setattr(mod.AdversarialRobustnessStrategy, "pgd_attack", lambda self, x, loss_fn: x)

    gen = nn.Conv2d(1, 1, 1)
    strat = _make_strategy(gen)
    batch = {"input": torch.rand(2, 1, 8, 8, requires_grad=True), "target": torch.rand(2, 1, 8, 8)}

    out = strat._compute_losses_impl(input_batch=None, target_batch=None, epoch=0, batch=batch)

    # The canonical key must exist, be a tensor, and carry gradient (for backward).
    assert isinstance(out.get("g_total_loss"), torch.Tensor)
    assert out["g_total_loss"].requires_grad
    # No stale loss_total key (the old broken contract).
    assert "loss_total" not in out
    # Components are forwarded with clean_/adv_ prefixes (detached), not the totals.
    assert "clean_l1" in out and "adv_l1" in out
    assert "clean_g_total_loss" not in out  # the total is merged, not duplicated as a component


# --------------------------------------------------------------------------- #
# Attack-time validation pass, the clean-trained control and the knob block
# --------------------------------------------------------------------------- #


def _resolved_class() -> type:
    from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory

    cfg = types.SimpleNamespace(
        training=types.SimpleNamespace(strategy_class=None, training_mode="adversarial_robustness")
    )
    return TrainingStrategyFactory().get_strategy_class(cfg)


def test_validation_step_attacks_through_the_production_class(monkeypatch) -> None:
    import spectramr.infrastructure.training.strategies.adversarial_robustness_strategy as mod

    cls = _resolved_class()
    assert cls is AdversarialRobustnessStrategy
    monkeypatch.setattr(
        mod.ReconstructionTrainingStrategy, "validation_step", lambda self, *a, **k: {"val_psnr": 1.0}
    )
    torch.manual_seed(0)
    gen = nn.Conv2d(1, 1, 3, padding=1)
    with torch.no_grad():
        gen.weight.mul_(20.0)  # high gain: a small input perturbation moves the output
    strat = _make_strategy(gen)
    strat.adapter_chains = {}
    strat.sigma_smooth, strat.n_smooth_samples = 0.25, 4
    strat._validation_forward = lambda z, ctx, **k: gen(z)
    x = torch.rand(2, 1, 8, 8)
    target = gen(x).detach()
    with torch.no_grad():  # the validation loop's context
        out = strat.validation_step(x, target, batch_data=None)
    assert out["val_psnr"] == 1.0
    assert set(out) >= mod.ATTACK_VALIDATION_KEYS
    assert out["val_attack_psnr_drop"] > 0.0
    assert out["val_attack_psnr"] < out["val_attack_clean_psnr"]
    assert torch.isfinite(torch.tensor(out["val_attack_certified_radius"]))


def test_emitted_metrics_declared_and_parent_tags_kept() -> None:
    from spectramr.infrastructure.training.strategies.certified_robustness_strategy import (
        CertifiedRobustnessStrategy,
    )
    from spectramr.infrastructure.training.strategies.reconstruction import (
        ReconstructionTrainingStrategy,
    )

    caps = AdversarialRobustnessStrategy.capabilities
    assert caps.emitted_metrics == {
        "val_attack_clean_psnr",
        "val_attack_psnr",
        "val_attack_psnr_drop",
        "val_attack_certified_radius",
    }
    assert caps.workflows == ReconstructionTrainingStrategy.capabilities.workflows
    child = CertifiedRobustnessStrategy.capabilities.emitted_metrics
    assert caps.emitted_metrics < child
    assert "val_composed_spectral_norm_bound" in child


def test_adv_weight_zero_is_the_clean_control(monkeypatch) -> None:
    import spectramr.infrastructure.training.strategies.adversarial_robustness_strategy as mod

    def _fake_super(self, input_batch=None, target_batch=None, epoch=0, **k):
        return {"g_total_loss": (input_batch.sum() * 0.0 + 2.0).requires_grad_(True)}

    def _no_attack(self, x, loss_fn):
        raise AssertionError("the clean control must not run PGD")

    monkeypatch.setattr(mod.ReconstructionTrainingStrategy, "_compute_losses_impl", _fake_super)
    monkeypatch.setattr(mod.AdversarialRobustnessStrategy, "pgd_attack", _no_attack)
    strat = _make_strategy(nn.Conv2d(1, 1, 1))
    strat.clean_weight, strat.adv_weight = 1.0, 0.0
    batch = {"input": torch.rand(2, 1, 8, 8), "target": torch.rand(2, 1, 8, 8)}
    out = strat._compute_losses_impl(input_batch=None, target_batch=None, epoch=0, batch=batch)
    assert float(out["g_total_loss"].detach()) == 2.0
    assert out["g_total_loss"].requires_grad
    assert not any(k.startswith("adv_") for k in out)


def test_training_batch_from_the_loop_is_attacked_and_weighted(monkeypatch) -> None:
    """The loop hands ``batch=TrainingBatch``, not a dict: PGD must fire and weights apply."""
    import spectramr.infrastructure.training.strategies.adversarial_robustness_strategy as mod
    from spectramr.data.batch_types import TrainingBatch

    seen: list[tuple[torch.Tensor, object]] = []

    def _fake_super(self, input_batch=None, target_batch=None, epoch=0, **k):
        seen.append((input_batch, k.get("batch")))
        return {"g_total_loss": (input_batch.sum() * 0.0 + 3.0).requires_grad_(True)}

    real_pgd = mod.AdversarialRobustnessStrategy.pgd_attack
    calls: list[int] = []

    def _counting_pgd(self, x, loss_fn):
        calls.append(1)
        return real_pgd(self, x, loss_fn)

    monkeypatch.setattr(mod.ReconstructionTrainingStrategy, "_compute_losses_impl", _fake_super)
    monkeypatch.setattr(mod.AdversarialRobustnessStrategy, "pgd_attack", _counting_pgd)
    strat = _make_strategy(nn.Conv2d(1, 1, 1))
    x, t = torch.rand(2, 1, 8, 8), torch.rand(2, 1, 8, 8)
    batch = TrainingBatch(input=x, target=t, metadata={"slice": 3})
    out = strat._compute_losses_impl(input_batch=x, target_batch=t, epoch=0, batch=batch)

    assert calls == [1]
    assert float(out["g_total_loss"].detach()) == pytest.approx(0.1 * 3.0 + 0.9 * 3.0)
    adv_x, adv_batch = seen[1]
    assert isinstance(adv_batch, TrainingBatch)
    assert adv_batch.input is adv_x and adv_batch.metadata == {"slice": 3}
    assert not torch.equal(adv_x, x)
    assert float((adv_x - x).abs().max()) <= strat.epsilon + 1e-6
    assert seen[0][1] is batch  # the clean pass keeps the loop's batch


def test_training_batch_clean_control_is_weighted_without_attack(monkeypatch) -> None:
    import spectramr.infrastructure.training.strategies.adversarial_robustness_strategy as mod
    from spectramr.data.batch_types import TrainingBatch

    def _fake_super(self, input_batch=None, target_batch=None, epoch=0, **k):
        return {"g_total_loss": (input_batch.sum() * 0.0 + 3.0).requires_grad_(True)}

    def _no_attack(self, x, loss_fn):
        raise AssertionError("the clean control must not run PGD")

    monkeypatch.setattr(mod.ReconstructionTrainingStrategy, "_compute_losses_impl", _fake_super)
    monkeypatch.setattr(mod.AdversarialRobustnessStrategy, "pgd_attack", _no_attack)
    strat = _make_strategy(nn.Conv2d(1, 1, 1))
    strat.clean_weight, strat.adv_weight = 0.1, 0.0
    x, t = torch.rand(2, 1, 8, 8), torch.rand(2, 1, 8, 8)
    out = strat._compute_losses_impl(
        input_batch=x, target_batch=t, epoch=0, batch=TrainingBatch(input=x, target=t)
    )
    assert float(out["g_total_loss"].detach()) == pytest.approx(0.3)


def test_missing_target_raises_instead_of_training_clean() -> None:
    strat = _make_strategy(nn.Conv2d(1, 1, 1))
    with pytest.raises(ValueError, match="needs a generator, an input and a target"):
        strat._compute_losses_impl(input_batch=torch.rand(1, 1, 4, 4), target_batch=None, epoch=0)


def _configured(**block):
    from spectramr.config.schemas.training.certified_robustness import CertifiedRobustnessConfig

    strat = object.__new__(AdversarialRobustnessStrategy)
    strat._configure_adversarial_knobs(CertifiedRobustnessConfig(**block))
    return strat


def test_knobs_read_from_block_and_stamped() -> None:
    strat = _configured(epsilon=0.02, attack_steps=3, adv_weight=0.0, clean_weight=1.0)
    assert (strat.epsilon, strat.attack_steps, strat.adv_weight) == (0.02, 3, 0.0)
    assert strat.adversarial_provenance["epsilon"] == 0.02
    assert strat.adversarial_provenance["n_smooth_samples"] == 32


@pytest.mark.parametrize(
    "block",
    [
        {"lambda_lipschitz": 0.1},  # read only by CertifiedRobustnessStrategy
        {"attack_norm": "l2"},  # validation scores an L-inf attack
        {"clean_weight": 0.0, "adv_weight": 0.0},  # no loss at all
    ],
)
def test_knobs_refuse_what_the_strategy_cannot_honour(block) -> None:
    with pytest.raises(ValueError):
        _configured(**block)


def test_attack_pass_refuses_a_forward_that_reads_the_context_measurement() -> None:
    strat = _make_strategy(nn.Conv2d(1, 1, 1))
    strat.validation_reads_context_measurement = True
    with pytest.raises(ValueError):
        strat._attack_validation_metrics(torch.rand(1, 1, 4, 4), torch.rand(1, 1, 4, 4), None)
