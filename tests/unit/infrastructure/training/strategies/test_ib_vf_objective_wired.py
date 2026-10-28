"""Regression: the IB-VF InfoNCE objective must actually enter training.

Before this fix the IB-VF strategy (``IBVFTrainingStrategy``) built two
:class:`InfoNCECritic` modules in ``__init__`` but:

* never moved them to ``self.device`` nor registered their parameters on the
  generator optimizer (``self.env.opt_g``) — so even if the IB term had been
  computed, its gradient would have updated nothing;
* never overrode ``_compute_losses_impl`` to call ``_ib_term`` — so the IB
  objective was dead code (no caller) and never reached the total loss.

These two tests pin the wiring:

1. ``test_critics_registered_on_opt_g`` — both critics' params land in
   ``opt_g.param_groups`` (and are on the strategy device).
2. ``test_compute_losses_adds_ib_term`` — the override forwards to the parent
   (mocked to return a base dict), adds a ``loss_ib`` component, folds the live
   IB tensor into the total, and that total carries a gradient back to the
   critic parameters.

Construction is via ``object.__new__`` with only the attributes the tested
methods touch (matching the surrounding aux-param-registration regression
tests), so neither the full DI container nor a real simulator is needed.
"""

from __future__ import annotations

import types

import torch

from spectramr.infrastructure.training.strategies import ib_vf_strategy
from spectramr.infrastructure.training.strategies.ib_vf_strategy import (
    IBVFTrainingStrategy,
)
from spectramr.models.encoders.nav_encoder_1d import NavEncoder1D
from spectramr.models.losses.infonce_critic import InfoNCECritic


def _env_with_opt() -> types.SimpleNamespace:
    dummy = torch.nn.Parameter(torch.zeros(1))
    return types.SimpleNamespace(opt_g=torch.optim.SGD([dummy], lr=0.1))


def _param_ids(opt: torch.optim.Optimizer) -> set[int]:
    return {id(p) for group in opt.param_groups for p in group["params"]}


def _bare_strategy(env: types.SimpleNamespace) -> IBVFTrainingStrategy:
    s = object.__new__(IBVFTrainingStrategy)
    s._owned_modules, s._owned_wrappers = {}, {}  # set by the base __init__ (#2629)
    s.env = env
    s.device = torch.device("cpu")
    s.beta_plus = 1.0
    s.beta_minus = 0.1
    s.infonce_negatives_K = 8
    d = 16
    s.critic_max = InfoNCECritic(critic_hidden_dim=32, feature_dim=d)
    s.critic_min = InfoNCECritic(critic_hidden_dim=32, feature_dim=d)
    # The navigator encoder is a strategy-owned module (not the main generator);
    # _register_ib_modules moves it to device and registers its params on opt_g.
    s.nav_encoder = NavEncoder1D(in_channels=2, nav_latent_dim=d, depth=2)
    # _compute_losses_impl reads config.data.dataset_type to decide the k-space
    # iFFT branch; provide a minimal frozen-config stand-in (image domain).
    s.config = types.SimpleNamespace(data=types.SimpleNamespace(dataset_type="image"))
    return s


class TestIBVFObjectiveWired:
    def test_critics_registered_on_opt_g(self) -> None:
        env = _env_with_opt()
        s = _bare_strategy(env)
        before = _param_ids(env.opt_g)

        s._register_ib_modules()

        after = _param_ids(env.opt_g)
        assert after - before, "IB module params were not added to opt_g"
        # Both InfoNCE critics AND the strategy-owned navigator encoder must be
        # registered (the nav encoder is no longer the main generator, so its
        # params would otherwise train nothing — CLAUDE.md #9).
        for module in (s.critic_max, s.critic_min, s.nav_encoder):
            assert all(id(p) in after for p in module.parameters())
            assert all(p.device.type == "cpu" for p in module.parameters())

    def test_compute_losses_adds_ib_term(self, monkeypatch) -> None:
        env = _env_with_opt()
        s = _bare_strategy(env)
        s._register_ib_modules()

        b, d = 4, 16
        # Stub the navigator-feature derivation so the test does not depend on
        # the simulator / generator; the wiring under test is that _ib_term is
        # called and its result reaches the total loss with a live gradient.
        z_m = torch.randn(b, d)
        theta_hat = torch.randn(b, d)
        anatomy = torch.randn(b, d)
        monkeypatch.setattr(
            s, "_ib_features", lambda target_complex: (z_m, theta_hat, anatomy)
        )

        base_total = torch.tensor(2.0, requires_grad=True)

        def _fake_parent(self, *, input_batch, target_batch, epoch, **kwargs):
            return {"g_total_loss": base_total, "loss_l1": torch.tensor(2.0)}

        monkeypatch.setattr(
            ib_vf_strategy.ConcreteVirtualFiducialStrategy,
            "_compute_losses_impl",
            _fake_parent,
        )

        target = torch.randn(b, 2, 8, 8)
        out = s._compute_losses_impl(input_batch=None, target_batch=target, epoch=0)

        # IB term entered the dict and was folded into the total.
        assert "loss_ib" in out, "IB loss component never reached the loss dict"
        assert "g_total_loss" in out
        assert out["g_total_loss"] is not base_total, "total was not augmented"
        assert out["g_total_loss"].requires_grad

        # The augmented total carries a gradient back to the critic params,
        # proving the IB objective trains the InfoNCE critics (not a no-op).
        env.opt_g.zero_grad(set_to_none=True)
        out["g_total_loss"].backward()
        critic_params = [
            p
            for critic in (s.critic_max, s.critic_min)
            for p in critic.parameters()
            if p.requires_grad
        ]
        assert any(
            p.grad is not None and torch.any(p.grad != 0) for p in critic_params
        ), "IB term produced no gradient on any InfoNCE critic parameter"


def test_ib_vf_block_coerces_from_raw_dict():
    """Regression: the mode-dispatch leaves ``config.training.ib_vf`` as a raw
    dict, so IBVFTrainingStrategy.__init__ must coerce it to IBVFConfig before
    ``cfg.beta_plus`` — otherwise it raised "'dict' object has no attribute
    'beta_plus'" once the misplaced ``infonce_critic`` image-loss was removed
    (cluster smoke 20260605, job 7095209). This pins that the loader-shaped
    dict round-trips through the SSOT schema the strategy coerces to."""
    from spectramr.config.schemas.training.vf_advanced import IBVFConfig

    raw = {
        "beta_plus": 1.0,
        "beta_minus": 1.0,
        "infonce_temperature": 0.1,
        "infonce_negatives_K": 32,
        "critic_hidden_dim": 128,
        "memory_bank_size": 0,
        "nav_latent_dim": 64,
        "nav_encoder_depth": 4,
        "navigator_source": "dc_kspace",
    }
    cfg = IBVFConfig(**raw)
    assert cfg.beta_plus == 1.0
    assert cfg.nav_latent_dim == 64
