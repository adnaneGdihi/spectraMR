"""Regression: the LOUPE learnable-acquisition mask must actually be wired.

2026-05-31 audit found ``LOUPEStrategy._compute_losses_impl`` called the
parent reconstruction loss on the RAW input and never invoked
``_ensure_mask`` / ``_gumbel_sigmoid_mask`` / ``_current_tau`` — so the learned
sampling mask was never created, sampled, or applied. The learnable
acquisition (the whole point of the paradigm) was dead.

These tests assert the KEY properties of the fix:

* ``self._mask_logits`` (the learnable acquisition parameter) is registered
  on ``self.env.opt_g`` so it actually trains.
* The mask is applied in k-space: the input the recon backbone receives
  DIFFERS from the raw input.
* Gradient flows from the recon path back to ``self._mask_logits`` (the
  Gumbel-sigmoid straight-through estimator is differentiable).

The strategy is constructed via ``object.__new__`` with only the attributes
the tested method touches, and the parent ``_compute_losses_impl`` is
monkeypatched to capture the input it is handed.
"""

from __future__ import annotations

import types

import pytest
import torch

from spectramr.infrastructure.physics.fft_ops import fft2c
from spectramr.infrastructure.training.strategies import reconstruction as recon_mod
from spectramr.infrastructure.training.strategies.loupe_strategy import LOUPEStrategy


def _adaptive_cfg(enabled: bool = True, budget: str = "soft"):
    annealing = types.SimpleNamespace(tau_init=5.0, tau_final=0.1, anneal_steps=1000)
    return types.SimpleNamespace(
        enabled=enabled,
        target_rate=0.125,
        density_penalty_lambda=1.0,
        budget=budget,
        mask_learning_rate=1e-3,
        annealing=annealing,
    )


def _config(model_kwargs=None):
    # model_kwargs is a real dict[str, Any] at runtime (config.schemas.model
    # defines ``model_kwargs: dict[str, Any]``). _ensure_mask reads
    # ``model_kwargs.get("share_across_coils", True)`` — using a dict here
    # exercises that read path. A SimpleNamespace would mask the bug AND crash
    # on ``.get``, so the helper MUST hand back a dict.
    if model_kwargs is None:
        model_kwargs = {}  # no share_across_coils -> default True
    model = types.SimpleNamespace(model_kwargs=model_kwargs)
    return types.SimpleNamespace(model=model)


def _env_with_opt():
    dummy = torch.nn.Parameter(torch.zeros(1))
    generator = types.SimpleNamespace(training=True)
    return types.SimpleNamespace(opt_g=torch.optim.SGD([dummy], lr=0.1), generator=generator)


def _param_ids(opt):
    return {id(p) for g in opt.param_groups for p in g["params"]}


def _make_strategy(config=None):
    s = object.__new__(LOUPEStrategy)
    s.env = _env_with_opt()
    s.device = torch.device("cpu")
    s.config = config if config is not None else _config()
    s._adaptive_cfg = _adaptive_cfg(enabled=True)
    s._mask_logits = None
    s._steps_seen = 0
    s._owned_modules = {}
    s._owned_wrappers = {}
    return s


def test_mask_logits_registered_on_opt_g(monkeypatch):
    """_mask_logits land in opt_g.param_groups after a compute_losses call."""
    s = _make_strategy()
    monkeypatch.setattr(
        recon_mod.ReconstructionTrainingStrategy,
        "_compute_losses_impl",
        lambda self, **kw: {"g_total_loss": torch.tensor(0.0, requires_grad=True)},
    )

    before = _param_ids(s.env.opt_g)
    x = torch.randn(2, 2, 8, 8)
    s._compute_losses_impl(input_batch=x, target_batch=x, epoch=0)
    after = _param_ids(s.env.opt_g)

    assert s._mask_logits is not None, "mask logits were never created"
    assert id(s._mask_logits) in after, "mask logits not registered on opt_g"
    assert after - before, "no new params added to opt_g"


def test_learned_mask_is_applied_to_input(monkeypatch):
    """The input handed to the recon backbone differs from the raw input."""
    captured: dict[str, torch.Tensor] = {}

    def _capture(self, *, input_batch=None, target_batch=None, epoch=0, **kw):
        captured["input_batch"] = input_batch
        return {"g_total_loss": torch.tensor(0.0, requires_grad=True)}

    s = _make_strategy()
    monkeypatch.setattr(
        recon_mod.ReconstructionTrainingStrategy, "_compute_losses_impl", _capture
    )

    x = torch.randn(2, 2, 16, 16)
    s._compute_losses_impl(input_batch=x, target_batch=x, epoch=0)

    masked = captured["input_batch"]
    assert isinstance(masked, torch.Tensor)
    assert masked.shape == x.shape, "mask round-trip must preserve [B,C,H,W] layout"
    # Heavy undersampling (target_rate 0.125) MUST perturb the input.
    assert not torch.allclose(masked, x), "learned mask was never applied to input"


def test_gradient_flows_to_mask_logits(monkeypatch):
    """Density penalty + masked-input path back-propagate into the mask logits."""

    def _recon_loss(self, *, input_batch=None, target_batch=None, epoch=0, **kw):
        # A trivial loss that depends on the (masked) input so gradient can
        # reach the mask logits through the differentiable k-space round-trip.
        return {"g_total_loss": input_batch.pow(2).mean()}

    s = _make_strategy()
    monkeypatch.setattr(
        recon_mod.ReconstructionTrainingStrategy, "_compute_losses_impl", _recon_loss
    )

    x = torch.randn(1, 2, 8, 8)
    losses = s._compute_losses_impl(input_batch=x, target_batch=x, epoch=0)

    assert "loss_density_penalty" in losses, "density/rate penalty missing"
    total = losses["g_total_loss"]
    total.backward()
    assert s._mask_logits.grad is not None, "no gradient reached the mask logits"
    assert s._mask_logits.grad.abs().sum().item() > 0.0, "mask-logit gradient is zero"


def test_disabled_adaptive_is_passthrough(monkeypatch):
    """With adaptive disabled, the raw input is forwarded and no mask is built."""
    captured: dict[str, torch.Tensor] = {}

    def _capture(self, *, input_batch=None, target_batch=None, epoch=0, **kw):
        captured["input_batch"] = input_batch
        return {"g_total_loss": torch.tensor(0.0, requires_grad=True)}

    s = _make_strategy()
    s._adaptive_cfg = _adaptive_cfg(enabled=False)
    monkeypatch.setattr(
        recon_mod.ReconstructionTrainingStrategy, "_compute_losses_impl", _capture
    )

    x = torch.randn(2, 2, 8, 8)
    out = s._compute_losses_impl(input_batch=x, target_batch=x, epoch=0)

    assert captured["input_batch"] is x, "raw input must pass through unchanged"
    assert s._mask_logits is None, "mask must not be built when disabled"
    assert "loss_density_penalty" not in out


def test_per_coil_branch_reachable_from_config():
    """share_across_coils=False must produce a per-coil (n_coils, w) logit grid.

    Regression (pitfall #15 silent no-op knob): the strategy previously read the
    flag via ``getattr(self.config.model.model_kwargs, "share_across_coils",
    True)``. Because ``model_kwargs`` is a plain ``dict`` at runtime,
    ``getattr`` always returned the ``True`` default and the per-coil branch was
    unreachable from YAML. The fixed code uses ``mk.get("share_across_coils",
    True)``. On the pre-fix code this asserts (6,) and FAILS.
    """
    s = _make_strategy(_config(model_kwargs={"share_across_coils": False}))
    s._ensure_mask(h=8, w=6, n_coils=4)
    assert s._mask_logits is not None
    assert tuple(s._mask_logits.shape) == (4, 6)


def test_shared_branch_default_single_mask():
    """Default (share_across_coils=True) yields a single (w,) mask."""
    s = _make_strategy(_config(model_kwargs={"share_across_coils": True}))
    s._ensure_mask(h=8, w=6, n_coils=4)
    assert s._mask_logits is not None
    assert tuple(s._mask_logits.shape) == (6,)


def test_missing_key_defaults_to_shared():
    """An empty model_kwargs dict falls back to the shared single mask."""
    s = _make_strategy(_config(model_kwargs={}))
    s._ensure_mask(h=8, w=6, n_coils=4)
    assert tuple(s._mask_logits.shape) == (6,)


def test_non_bool_share_across_coils_raises():
    """A non-bool share_across_coils must raise (no silent coercion, pitfall #9)."""
    s = _make_strategy(_config(model_kwargs={"share_across_coils": "yes"}))
    with pytest.raises(TypeError):
        s._ensure_mask(h=8, w=6, n_coils=2)


# ── the learned mask survives a resume (#2820) ──────────────────────────────


def _trained_strategy():
    s = _make_strategy(_config(model_kwargs={"share_across_coils": False}))
    s.env.opt_g = torch.optim.SGD(s.env.opt_g.param_groups[0]["params"], lr=0.1, momentum=0.9)
    logits = s._ensure_mask(h=8, w=6, n_coils=2)
    (logits * torch.arange(logits.numel()).reshape(logits.shape)).sum().backward()
    s.env.opt_g.step()
    return s


def test_a_resume_restores_the_learned_mask_and_its_optimizer_state():
    saved = _trained_strategy()
    resumed = _make_strategy(_config(model_kwargs={"share_across_coils": False}))
    resumed.env.opt_g = torch.optim.SGD(
        resumed.env.opt_g.param_groups[0]["params"], lr=0.1, momentum=0.9
    )

    state = saved.strategy_state_dict()
    # The DDP holder is not an attribute, so the checkpoint key is unchanged (#2629).
    assert sorted(state["params"]) == ["_mask_logits"] and state["modules"] == {}
    resumed.load_strategy_state_dict(state)
    resumed.env.opt_g.load_state_dict(saved.env.opt_g.state_dict())

    assert torch.equal(resumed._mask_logits, saved._mask_logits)
    assert resumed.owned_modules()["_mask_logits"].value is resumed._mask_logits
    assert resumed.env.opt_g.param_groups[-1]["params"] == [resumed._mask_logits]
    assert torch.equal(
        resumed.env.opt_g.state[resumed._mask_logits]["momentum_buffer"],
        saved.env.opt_g.state[saved._mask_logits]["momentum_buffer"],
    )


def test_a_resume_does_not_rebuild_the_mask_when_none_was_saved():
    resumed = _make_strategy()
    resumed.load_strategy_state_dict({"modules": {}, "params": {}, "generators": {}})
    assert resumed._mask_logits is None


def test_mask_in_image_domain_keeps_only_the_chosen_phase_encode_column() -> None:
    # cohort5 §2.1: the mask is applied in k-space of an IMAGE input (LOUPE's
    # pre_model chain delivers one); one kept column must survive as one column.
    s = _make_strategy()
    x = torch.randn(1, 2, 16, 16)
    line = torch.zeros(16)
    line[5] = 1.0
    out = s._apply_learned_mask(x, line)
    k = fft2c(out)
    dropped = [c for c in range(16) if c != 5]
    assert k[..., dropped].abs().max().item() < 1e-4
    assert k[..., 5].abs().max().item() > 1e-3


def test_validation_grades_the_deterministic_topk_mask(monkeypatch) -> None:
    # cohort5 §2.0/§2.3: validation applies the eval mask, not the fully sampled input,
    # and it draws no Gumbel noise, so two passes see the same input.
    captured: list[torch.Tensor] = []

    def _capture(self, input_batch, batch_context, **kw):
        captured.append(input_batch.clone())
        return input_batch

    s = _make_strategy()
    s._adaptive_cfg = _adaptive_cfg(enabled=True, budget="hard_topk")
    monkeypatch.setattr(recon_mod.ReconstructionTrainingStrategy, "_validation_forward", _capture)
    x = torch.randn(1, 2, 16, 16)
    s._validation_forward(x, {})
    s._validation_forward(x, {})

    assert not torch.allclose(captured[0], x)
    assert torch.equal(captured[0], captured[1])
    k = fft2c(captured[0])
    kept = (k.abs().sum(dim=(0, 1, 2)) > 1e-4).sum().item()
    assert kept == round(0.125 * 16)


def test_eval_pass_does_not_advance_the_schedule(monkeypatch) -> None:
    monkeypatch.setattr(
        recon_mod.ReconstructionTrainingStrategy,
        "_compute_losses_impl",
        lambda self, **kw: {"g_total_loss": torch.tensor(0.0, requires_grad=True)},
    )
    s = _make_strategy()
    s.env.generator.training = False
    s._compute_losses_impl(input_batch=torch.randn(1, 2, 8, 8), target_batch=None, epoch=0)
    assert s._steps_seen == 0


def test_unknown_budget_raises() -> None:
    s = _make_strategy()
    s._adaptive_cfg = _adaptive_cfg(enabled=True, budget="greedy")
    with pytest.raises(ValueError, match="budget"):
        s._eval_mask(torch.zeros(16))
