"""``SCASStrategy``: the scout-conditioned per-sample mask reaches the reconstruction input (#2405).

The strategy used to index the parent's ``(image, kwargs)`` tuple as a dict and write
an ``acquisition_mask`` nothing read, and its hypernet emitted per-row logits while
LOUPE masks columns. These tests run the strategy on a shell (no model) and capture
what ``ReconstructionTrainingStrategy`` is handed.
"""

from __future__ import annotations

import types

import pytest
import torch

from spectramr.infrastructure.physics.fft_ops import fft2c
from spectramr.infrastructure.training.strategies import reconstruction as recon_mod
from spectramr.infrastructure.training.strategies.scas_strategy import (
    SCASStrategy,
    ScoutHypernet,
)

W = 16


def _adaptive_cfg(budget: str = "hard_topk"):
    return types.SimpleNamespace(
        enabled=True,
        target_rate=0.25,
        density_penalty_lambda=2.0,
        budget=budget,
        mask_learning_rate=1e-3,
        annealing=types.SimpleNamespace(tau_init=1.0, tau_final=0.1, anneal_steps=10),
    )


def _strategy(training: bool = True) -> SCASStrategy:
    s = object.__new__(SCASStrategy)
    dummy = torch.nn.Parameter(torch.zeros(1))
    s.env = types.SimpleNamespace(
        opt_g=torch.optim.SGD([dummy], lr=0.1),
        generator=types.SimpleNamespace(training=training),
    )
    s.device = torch.device("cpu")
    s._adaptive_cfg = _adaptive_cfg()
    s._mask_logits = None
    s._hypernet = None
    s._steps_seen = 0
    s._owned_modules = {}
    s._owned_wrappers = {}
    return s


@pytest.fixture
def captured(monkeypatch):
    seen: dict[str, torch.Tensor] = {}

    def _loss(self, *, input_batch=None, target_batch=None, epoch=0, **kw):
        seen["input"] = input_batch
        return {"g_total_loss": input_batch.sum() * 0.0}

    def _val(self, input_batch, batch_context, **kw):
        seen["val_input"] = input_batch
        return input_batch

    monkeypatch.setattr(recon_mod.ReconstructionTrainingStrategy, "_compute_losses_impl", _loss)
    monkeypatch.setattr(recon_mod.ReconstructionTrainingStrategy, "_validation_forward", _val)
    return seen


def _batch(b: int = 3):
    torch.manual_seed(0)
    # Interleaved (Re, Im): the mask round-trip is linear, so its kept columns are visible.
    x = torch.randn(b, 2, W, W)
    return x, {"scout": torch.randn(b, 1, W, W, dtype=torch.complex64)}


def _kept_columns(image: torch.Tensor) -> torch.Tensor:
    return fft2c(image).abs().sum(dim=(1, 2)) > 1e-4  # (B, W)


def test_hypernet_emits_one_logit_per_column() -> None:
    logits = ScoutHypernet(in_channels=1, init_logit=-1.0)(torch.randn(2, 1, 8, W))
    assert logits.shape == (2, W)


def test_training_masks_each_sample_on_its_own_columns(captured) -> None:
    s = _strategy()
    x, raw = _batch()
    s._compute_losses_impl(input_batch=x, target_batch=x, epoch=0, batch=raw)
    kept = _kept_columns(captured["input"])
    assert kept.any(dim=1).all()
    assert not kept.all(dim=1).any(), "every sample must lose columns"
    assert s._mask_logits is None, "LOUPE's shared mask must not be applied on top"


def test_single_channel_input_keeps_no_mirrored_column() -> None:
    s = _strategy()
    x = torch.randn(1, 1, W, W)
    mask = torch.zeros(1, 1, 1, W)
    mask[..., 3] = 1.0
    out = s._apply_learned_mask(x, mask)
    assert out.dtype == x.dtype
    zero_filled = torch.zeros_like(x)
    assert not torch.allclose(out, zero_filled)
    # |ifft(M F x)| has the support of one column's sinusoid: constant magnitude along W.
    assert torch.allclose(out[0, 0], out[0, 0, :, :1].expand(-1, W), atol=1e-5)


def test_density_penalty_is_added_and_reaches_the_hypernet(captured) -> None:
    s = _strategy()
    x, raw = _batch()
    out = s._compute_losses_impl(input_batch=x, target_batch=x, epoch=0, batch=raw)
    assert "loss_scas_density" in out
    out["g_total_loss"].backward()
    assert s._hypernet is not None
    assert s._hypernet.head.bias.grad is not None
    assert s._steps_seen == 1
    hyper_ids = {id(p) for p in s._hypernet.parameters()}
    assert hyper_ids <= {id(p) for g in s.env.opt_g.param_groups for p in g["params"]}


def test_eval_pass_uses_the_topk_budget_and_no_penalty(captured) -> None:
    s = _strategy(training=False)
    x, raw = _batch()
    out = s._compute_losses_impl(input_batch=x, target_batch=x, epoch=0, batch=raw)
    assert "loss_scas_density" not in out
    assert s._steps_seen == 0
    assert (_kept_columns(captured["input"]).sum(dim=1) == round(0.25 * W)).all()


def test_validation_forward_grades_the_hypernet_mask(captured) -> None:
    s = _strategy(training=False)
    x, raw = _batch()
    context = s.validation_batch_context(x, x, raw)
    s._validation_forward(x, context)
    first = captured["val_input"]
    s._validation_forward(x, context)
    assert torch.equal(first, captured["val_input"])
    assert (_kept_columns(first).sum(dim=1) == round(0.25 * W)).all()


def test_missing_scout_raises_and_names_the_producer(captured) -> None:
    s = _strategy()
    x = torch.randn(2, 1, W, W)
    with pytest.raises(ValueError, match="scout_acquisition") as exc:
        s._compute_losses_impl(input_batch=x, target_batch=x, epoch=0, batch={"input": x})
    assert "input" in str(exc.value)
    with pytest.raises(ValueError, match="scout_acquisition"):
        s._validation_forward(x, {"use_dc": False})


def test_scout_that_does_not_match_the_input_raises(captured) -> None:
    s = _strategy()
    x = torch.randn(2, 1, W, W)
    with pytest.raises(ValueError, match="scout"):
        s._compute_losses_impl(
            input_batch=x, target_batch=x, epoch=0, batch={"scout": torch.randn(2, 1, W, 8)}
        )


def test_scas_declares_no_loupe_mask_module() -> None:
    cfg = types.SimpleNamespace(undersampling=types.SimpleNamespace(adaptive=_adaptive_cfg()))
    assert "_mask_logits" not in dict(SCASStrategy.owned_module_declarations(cfg))


def test_scas_keeps_its_own_ddp_refusal_over_its_migrated_parent():
    """LOUPE's mask logits are an owned module now; SCAS's bare hypernet group is not."""
    from spectramr.infrastructure.distributed.strategy_compat import evaluate_parallel_compat
    from spectramr.infrastructure.training.strategies.loupe_strategy import LOUPEStrategy

    assert not evaluate_parallel_compat(LOUPEStrategy, "ddp").refused
    assert evaluate_parallel_compat(SCASStrategy, "ddp").refused


def test_reconstruction_loss_alone_trains_the_scout_encoder(monkeypatch) -> None:
    # #2405 point 2: the mask must carry gradient from the reconstruction, not only the penalty.
    monkeypatch.setattr(
        recon_mod.ReconstructionTrainingStrategy,
        "_compute_losses_impl",
        lambda self, *, input_batch=None, **kw: {"g_total_loss": input_batch.pow(2).sum()},
    )
    s = _strategy()
    s._adaptive_cfg.density_penalty_lambda = 0.0
    x, raw = _batch()
    s._compute_losses_impl(input_batch=x, target_batch=x, epoch=0, batch=raw)[
        "g_total_loss"
    ].backward()
    grad = s._hypernet.encoder[0].weight.grad
    assert grad is not None and grad.abs().sum() > 0


def test_the_eval_mask_follows_the_scout(captured) -> None:
    s = _strategy(training=False)
    x, raw = _batch(b=2)
    raw["scout"][1] = raw["scout"][0]
    s._compute_losses_impl(input_batch=x, target_batch=x, epoch=0, batch=raw)
    same = _kept_columns(captured["input"])
    assert torch.equal(same[0], same[1])
    torch.manual_seed(1)
    raw["scout"][1] = torch.randn(1, W, W, dtype=torch.complex64) * 10
    s._hypernet.encoder[0].weight.data.normal_()
    s._compute_losses_impl(input_batch=x, target_batch=x, epoch=0, batch=raw)
    differ = _kept_columns(captured["input"])
    assert not torch.equal(differ[0], differ[1])


def test_a_hypernet_built_before_the_optimizer_joins_it_on_the_next_step(captured) -> None:
    s = _strategy(training=False)
    opt = s.env.opt_g
    s.env.opt_g = None
    x, raw = _batch()
    s._compute_losses_impl(input_batch=x, target_batch=x, epoch=0, batch=raw)
    s.env.opt_g = opt
    s.env.generator.training = True
    s._compute_losses_impl(input_batch=x, target_batch=x, epoch=0, batch=raw)
    hyper_ids = {id(p) for p in s._hypernet.parameters()}
    assert hyper_ids <= {id(p) for g in opt.param_groups for p in g["params"]}


def test_resume_rebuilds_and_restores_the_hypernet(captured) -> None:
    saved = _strategy()
    x, raw = _batch()
    saved._compute_losses_impl(input_batch=x, target_batch=x, epoch=0, batch=raw)
    state = saved.strategy_state_dict()
    assert "_hypernet" in state["modules"]

    resumed = _strategy()
    resumed.load_strategy_state_dict(state)
    for a, b in zip(resumed._hypernet.parameters(), saved._hypernet.parameters(), strict=True):
        assert torch.equal(a, b)
    assert {id(p) for p in resumed._hypernet.parameters()} <= {
        id(p) for g in resumed.env.opt_g.param_groups for p in g["params"]
    }
