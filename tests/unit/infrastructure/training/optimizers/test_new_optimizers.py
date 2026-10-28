"""LARS / LAMB / Lion / Lookahead / extra-backed optimizers.

``lars`` and ``lamb`` were members of ``OptimizerType`` with **no implementation
anywhere** for the entire life of the enum; the only record was an xfail reading
"needs apex / timm". These tests exist so that never silently recurs.

Assertions are deliberately behavioural — "does one step reduce a quadratic",
"does the hyper-parameter reach ``param_groups``", "does state round-trip" —
rather than shape/dtype checks, which pass for a wrong operator. Where an
algorithm has a distinguishing identity (Lion's sign update, LARS's trust ratio)
that identity is asserted directly, because every one of these is a small
variation on SGD/Adam and a subtly wrong one still trains.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

from spectramr.infrastructure.training.optimizer_registry import (  # noqa: E402
    OptimizerRegistry,
)
from spectramr.infrastructure.training.optimizers import (  # noqa: E402
    LAMB,
    LARS,
    Lion,
    Lookahead,
)

IN_REPO = ("lars", "lamb", "lion")


def _descends(factory, x0: float = 3.0, steps: int = 200) -> float:
    """Minimise ``x**2`` from ``x0`` and return the final |x|."""
    p = nn.Parameter(torch.tensor([x0]))
    opt = factory([p])
    for _ in range(steps):
        opt.zero_grad()
        (p**2).sum().backward()
        opt.step()
    return abs(p.item())


class TestRegistration:
    @pytest.mark.parametrize("name", IN_REPO)
    def test_is_registered(self, name: str) -> None:
        assert OptimizerRegistry.get(name) is not None

    def test_registry_covers_the_whole_advertised_vocabulary(self) -> None:
        """The lockstep invariant. A ``@register_optimizer`` in a module nothing
        imports is dead, and the symptom -- 'unknown optimizer' -- reads as a
        typo rather than a missing import."""
        from spectramr.config.schemas.enums import OPTIMIZER_NAMES

        assert not OPTIMIZER_NAMES - set(OptimizerRegistry.list_available())

    def test_importing_the_registry_first_still_registers_everything(self) -> None:
        """Guards the circular-import shape: ``optimizer_registry`` imports the
        ``optimizers`` package at the END of its own module, and those modules
        import ``register_optimizer`` back from it. Either import order must
        yield the same table."""
        import importlib

        import spectramr.infrastructure.training.optimizer_registry as reg

        before = set(reg.OptimizerRegistry.list_available())
        importlib.import_module("spectramr.infrastructure.training.optimizers")
        assert set(reg.OptimizerRegistry.list_available()) == before

    def test_lookahead_is_not_registered_as_a_name(self) -> None:
        """It wraps a base optimizer; standalone it would be unconstructible."""
        assert "lookahead" not in OptimizerRegistry.list_available()

    def test_sam_is_absent_until_its_stepper_exists(self) -> None:
        """SAM needs two forward+backward passes. Registering it against the
        current single-backward seam would perturb the weights and never restore
        them -- the run would train on w + rho*g and report success."""
        assert "sam" not in OptimizerRegistry.list_available()
        from spectramr.config.schemas.enums import OPTIMIZER_NAMES

        assert "sam" not in OPTIMIZER_NAMES


class TestConvergence:
    @pytest.mark.parametrize(
        "factory",
        [
            lambda p: LARS(p, lr=0.1, eta=1.0, momentum=0.0),
            lambda p: LAMB(p, lr=0.1),
            lambda p: Lion(p, lr=0.05),
        ],
        ids=IN_REPO,
    )
    def test_minimises_a_quadratic(self, factory) -> None:
        assert _descends(factory) < 0.5


class TestLARS:
    def test_trust_ratio_scales_the_step_by_weight_over_gradient_norm(self) -> None:
        """The defining property. With momentum off and eta=1, one step is
        ``lr * (||w|| / ||g||) * g``; a plain-SGD implementation would give
        ``lr * g``."""
        p = nn.Parameter(torch.tensor([[3.0, 4.0]]))  # ||w|| = 5
        opt = LARS([p], lr=1.0, eta=1.0, momentum=0.0, weight_decay=0.0, epsilon=0.0)
        p.grad = torch.tensor([[0.6, 0.8]])  # ||g|| = 1
        opt.step()
        # trust = 5/1 = 5  =>  delta = -1.0 * 5 * g
        assert torch.allclose(p.detach(), torch.tensor([[3.0, 4.0]]) - 5.0 * p.grad)

    def test_excluded_1d_parameters_get_plain_sgd(self) -> None:
        bias = nn.Parameter(torch.tensor([2.0]))
        opt = LARS([bias], lr=0.5, eta=1.0, momentum=0.0, exclude_1d=True)
        bias.grad = torch.tensor([1.0])
        opt.step()
        assert bias.item() == pytest.approx(2.0 - 0.5 * 1.0)

    def test_a_zero_gradient_layer_does_not_produce_nan(self) -> None:
        """0/0 on the trust ratio is what an unguarded division gives on the
        first step of a zero-initialised layer."""
        p = nn.Parameter(torch.zeros(2, 2))
        opt = LARS([p], lr=0.1, eta=1.0)
        p.grad = torch.zeros(2, 2)
        opt.step()
        assert torch.isfinite(p).all()

    def test_hyperparameters_reach_param_groups(self) -> None:
        opt = LARS([nn.Parameter(torch.zeros(1))], lr=0.2, eta=0.5, momentum=0.7)
        group = opt.param_groups[0]
        assert (group["lr"], group["eta"], group["momentum"]) == (0.2, 0.5, 0.7)

    @pytest.mark.parametrize("bad", [{"lr": 0.0}, {"eta": 0.0}, {"momentum": -1.0}])
    def test_rejects_invalid_hyperparameters(self, bad: dict) -> None:
        with pytest.raises(ValueError):
            LARS([nn.Parameter(torch.zeros(1))], **bad)


class TestLAMB:
    def test_adam_mode_disables_the_trust_ratio(self) -> None:
        """An exact within-optimizer ablation: same code path, ratio pinned to 1,
        so 'was it the trust ratio?' does not require swapping optimizer_type and
        perturbing everything else."""
        torch.manual_seed(0)
        a = nn.Parameter(torch.tensor([[1.0, 2.0]]))
        b = nn.Parameter(torch.tensor([[1.0, 2.0]]))
        for param, adam in ((a, False), (b, True)):
            opt = LAMB([param], lr=0.1, adam=adam)
            param.grad = torch.tensor([[0.3, -0.4]])
            opt.step()
        assert not torch.allclose(a.detach(), b.detach())

    def test_weight_decay_is_decoupled(self) -> None:
        """Coupling it into the gradient would make the trust ratio a function of
        the decay strength."""
        p = nn.Parameter(torch.tensor([[1.0]]))
        opt = LAMB([p], lr=0.1, weight_decay=0.5, adam=True)
        p.grad = torch.tensor([[0.0]])
        opt.step()
        # Zero gradient + decoupled decay still moves the weight.
        assert p.item() != pytest.approx(1.0)

    @pytest.mark.parametrize("steps", [1, 10])
    def test_the_trust_ratio_sets_the_step_to_lr_times_the_weight_norm(self, steps: int) -> None:
        """The ratio rescales the update to ``||w||``, so a bias correction left
        outside it survived as a warmup: 0.32 of this step at t=1, 0.15 at t=10."""
        torch.manual_seed(0)
        p = nn.Parameter(torch.randn(8, 8))
        opt = LAMB([p], lr=1e-3, clamp_value=1e9)
        for _ in range(steps):
            before = p.detach().clone()
            p.grad = torch.randn(8, 8)
            opt.step()
        ratio = (p.detach() - before).norm() / (1e-3 * before.norm())
        assert ratio.item() == pytest.approx(1.0, rel=1e-4)

    def test_a_complex_parameter_moves_along_its_gradient(self) -> None:
        """``addcmul_(g, g)`` squares a complex gradient: a purely imaginary one
        gave a negative second moment and moved the real part."""
        p = nn.Parameter(torch.tensor([1.0 + 1.0j]))
        opt = LAMB([p], lr=0.1)
        p.grad = torch.tensor([1.0j])
        opt.step()
        assert p.detach().real.item() == pytest.approx(1.0)
        assert p.detach().imag.item() < 1.0
        assert (opt.state[p]["exp_avg_sq"] >= 0).all()

    def test_rejects_sparse_gradients_loudly(self) -> None:
        p = nn.Parameter(torch.zeros(3))
        opt = LAMB([p])
        p.grad = torch.sparse_coo_tensor(torch.tensor([[0]]), torch.tensor([1.0]), (3,))
        with pytest.raises(RuntimeError, match="sparse"):
            opt.step()


class TestLion:
    def test_update_is_a_pure_sign_so_every_element_moves_by_lr(self) -> None:
        """Lion's defining property, and the reason its LR must be 3-10x smaller
        than AdamW's. A magnitude-carrying implementation would move elements by
        different amounts."""
        p = nn.Parameter(torch.zeros(4))
        opt = Lion([p], lr=0.1, weight_decay=0.0)
        p.grad = torch.tensor([100.0, -0.001, 5.0, -20.0])
        opt.step()
        assert torch.allclose(p.detach().abs(), torch.full((4,), 0.1))
        assert torch.equal(p.detach().sign(), -p.grad.sign())

    def test_uses_distinct_betas_for_update_and_buffer(self) -> None:
        """Collapsing beta1 and beta2 into one is the usual way Lion gets
        reimplemented wrongly; with beta1 != beta2 the second step differs."""
        p = nn.Parameter(torch.zeros(1))
        opt = Lion([p], lr=0.1, betas=(0.0, 0.9), weight_decay=0.0)
        p.grad = torch.tensor([1.0])
        opt.step()
        # beta1=0 => first update is sign(g). Buffer now holds 0.1*g.
        assert opt.state[p]["exp_avg"].item() == pytest.approx(0.1)

    def test_defaults_match_the_paper_not_adam(self) -> None:
        opt = Lion([nn.Parameter(torch.zeros(1))])
        assert opt.param_groups[0]["betas"] == (0.9, 0.99)

    def test_a_complex_parameter_is_signed_per_component(self) -> None:
        """``sign_()`` raised NotImplementedError on a complex tensor."""
        p = nn.Parameter(torch.tensor([1.0 + 1.0j]))
        opt = Lion([p], lr=0.1, weight_decay=0.0)
        p.grad = torch.tensor([1.0j])
        opt.step()
        assert torch.allclose(p.detach(), torch.tensor([1.0 + 0.9j]))


class TestLookahead:
    def _wrapped(self):
        model = nn.Linear(4, 2)
        return model, Lookahead(
            torch.optim.SGD(model.parameters(), lr=0.1), la_steps=3, la_alpha=0.5
        )

    def test_state_dict_round_trips_including_the_sync_counter(self) -> None:
        """The counter matters: a resumed Lookahead that resets it to 0 syncs one
        step early and corrupts the periodic cadence."""
        model, opt = self._wrapped()
        for _ in range(4):
            opt.zero_grad()
            model(torch.randn(2, 4)).sum().backward()
            opt.step()
        state = opt.state_dict()
        assert "counters" in state

        _, opt2 = self._wrapped()
        opt2.load_state_dict(state)
        assert opt2.state_dict()["counters"] == state["counters"]

    def test_is_still_importable_from_its_historical_module(self) -> None:
        """``gradient_stability`` keeps a re-export, not a second copy."""
        from spectramr.infrastructure.training.gradient_stability import (
            Lookahead as Shim,
        )

        assert Shim is Lookahead

    def test_reached_through_the_config_sub_block_not_optimizer_type(self) -> None:
        from spectramr.config.schemas.optimization import OptimizationConfigSchema
        from spectramr.infrastructure.training.optimizer_resolution import (
            build_optimizer_from_spec,
            resolve_optimizer_spec,
        )

        model = nn.Linear(4, 2)
        cfg = OptimizationConfigSchema(
            optimizer={
                "type": "sgd",
                "lookahead": {"enabled": True, "sync_period": 4, "alpha": 0.3},
            },
        )
        opt = build_optimizer_from_spec(resolve_optimizer_spec(cfg, model=model), model)
        assert isinstance(opt, Lookahead)
        assert opt.la_steps == 4 and opt.la_alpha == pytest.approx(0.3)

    def test_absent_sub_block_leaves_the_base_optimizer_unwrapped(self) -> None:
        from spectramr.config.schemas.optimization import OptimizationConfigSchema
        from spectramr.infrastructure.training.optimizer_resolution import (
            build_optimizer_from_spec,
            resolve_optimizer_spec,
        )

        model = nn.Linear(4, 2)
        cfg = OptimizationConfigSchema(optimizer_type="sgd")
        opt = build_optimizer_from_spec(resolve_optimizer_spec(cfg, model=model), model)
        assert not isinstance(opt, Lookahead)


class TestOptionalExtras:
    """Registered unconditionally; construction raises when the extra is absent.

    Skipping registration would make the name vanish from ``list_available()``,
    so the user would get "unknown optimizer" -- a message that sends them to fix
    their YAML instead of their environment.
    """

    EXTRA_BACKED = ("adam8bit", "adamw8bit", "schedulefree_adamw", "schedulefree_sgd")

    @pytest.mark.parametrize("name", EXTRA_BACKED)
    def test_name_is_registered_whether_or_not_the_extra_is_installed(
        self, name: str
    ) -> None:
        assert OptimizerRegistry.get(name) is not None

    @pytest.mark.parametrize("name", EXTRA_BACKED)
    def test_construction_either_works_or_raises_with_an_install_command(
        self, name: str
    ) -> None:
        cls = OptimizerRegistry.get(name)
        assert cls is not None
        try:
            cls([nn.Parameter(torch.zeros(2))], lr=1e-3)
        except ImportError as exc:
            message = str(exc)
            assert "pip install" in message
            assert "Refusing to fall back" in message

    @pytest.mark.parametrize("name", EXTRA_BACKED)
    def test_accepted_kwargs_are_declared_not_introspected(self, name: str) -> None:
        """The accepted set must not depend on whether the extra happens to be
        installed on the machine doing the resolving."""
        assert OptimizerRegistry.accepted_kwargs(name)
        assert "lr" in OptimizerRegistry.accepted_kwargs(name)

    def test_schedule_free_mode_detection_is_duck_typed(self) -> None:
        from spectramr.infrastructure.training.optimizers import (
            supports_schedule_free_modes,
        )

        assert not supports_schedule_free_modes(
            torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=0.1)
        )

        class _Fake:
            def train(self): ...
            def eval(self): ...

            # `step` is part of the predicate: train/eval alone do not identify
            # an optimizer, because every nn.Module has both. A stand-in without
            # it was never a faithful one.
            def step(self): ...

        assert supports_schedule_free_modes(_Fake())

        class _Module:
            """An nn.Module-shaped object is NOT a schedule-free optimizer."""

            def train(self): ...
            def eval(self): ...

        assert not supports_schedule_free_modes(_Module())


class TestInRepoOptimizersDoNotSyncTheHost:
    """Non-negotiable #9: no ``.item()``/``.cpu()`` inside the training loop.

    ``LAMB`` computed its trust ratio as
    ``1.0 if (w_norm == 0 or u_norm == 0) else (w_norm / u_norm).item()`` --
    three device->host round-trips per parameter tensor per step (two
    tensor-to-bool conversions plus the explicit ``.item()``). ``LARS`` did the
    same with its zero-norm guard. Measured on CUDA over 80 parameter tensors,
    that cost 19.1x (LAMB) and 9.4x (LARS) an AdamW step.

    Patching the sync entry points to raise is device-independent, so this runs
    on CPU CI where ``torch.cuda.set_sync_debug_mode`` cannot. Both norms stay
    on-device via ``torch.where``, which is what makes the guard free.
    """

    @staticmethod
    def _model():
        torch.manual_seed(0)
        return nn.Sequential(nn.Linear(6, 5), nn.LayerNorm(5), nn.Linear(5, 3))

    @staticmethod
    def _backward(model):
        (model(torch.randn(4, 6)) ** 2).mean().backward()

    def _assert_no_sync(self, optimizer, model, monkeypatch):
        self._backward(model)

        def _boom(self, *a, **kw):  # noqa: ANN001
            raise AssertionError(
                "optimizer.step() synchronised the host: a tensor was converted "
                "to a Python scalar inside the training loop (non-negotiable #9)"
            )

        monkeypatch.setattr(torch.Tensor, "item", _boom, raising=True)
        monkeypatch.setattr(torch.Tensor, "__bool__", _boom, raising=True)
        monkeypatch.setattr(torch.Tensor, "__float__", _boom, raising=True)
        optimizer.step()

    def test_lamb_step_does_not_sync(self, monkeypatch) -> None:
        model = self._model()
        self._assert_no_sync(LAMB(model.parameters(), lr=1e-3), model, monkeypatch)

    def test_lars_step_does_not_sync(self, monkeypatch) -> None:
        model = self._model()
        self._assert_no_sync(
            LARS(model.parameters(), lr=1e-3, momentum=0.9), model, monkeypatch
        )

    def test_lion_step_does_not_sync(self, monkeypatch) -> None:
        """Lion was already clean -- pinned so it stays that way."""
        model = self._model()
        self._assert_no_sync(Lion(model.parameters(), lr=1e-4), model, monkeypatch)

    def test_lamb_zero_norm_still_falls_back_to_plain_adam(self) -> None:
        """The on-device guard must keep the numerics it replaced: a zero-norm
        parameter takes the un-scaled step rather than emitting nan."""
        p = nn.Parameter(torch.zeros(4))
        opt = LAMB([p], lr=1e-2, weight_decay=0.0)
        p.grad = torch.ones(4)
        opt.step()
        assert torch.isfinite(p).all()
        # w_norm == 0 -> trust pinned to 1.0 -> the plain Adam step.
        assert p.detach().abs().max() > 0

    def test_lars_zero_gradient_does_not_produce_nan(self) -> None:
        p = nn.Parameter(torch.ones(4, 4))
        opt = LARS([p], lr=1e-2, momentum=0.0, weight_decay=0.0)
        p.grad = torch.zeros(4, 4)
        opt.step()
        assert torch.isfinite(p).all()


class TestStepReturnsTheClosureLoss:
    """``step(closure)`` evaluates the closure once and returns its loss, as the
    typed ``Optimizer.step`` overloads promise; ``step()`` returns ``None``."""

    @pytest.mark.parametrize(
        "factory",
        [
            lambda p: LAMB(p, lr=0.1),
            lambda p: Lion(p, lr=0.1),
            lambda p: Lookahead(torch.optim.SGD(p, lr=0.1), la_steps=2),
        ],
        ids=["lamb", "lion", "lookahead"],
    )
    def test_closure_loss_is_returned(self, factory) -> None:
        p = nn.Parameter(torch.tensor([2.0]))
        opt = factory([p])
        calls: list[int] = []

        def closure() -> float:
            calls.append(1)
            opt.zero_grad()
            loss = (p**2).sum()
            loss.backward()
            return float(loss)

        assert opt.step(closure) == pytest.approx(4.0)
        assert calls == [1]
        p.grad = torch.ones(1)
        assert opt.step() is None


class TestStepRecordsNoAutogradHistory:
    """``step()`` updates parameters and state under ``torch.no_grad()``; a
    graph recorded there would grow every step and hold every old tensor."""

    @pytest.mark.parametrize(
        "factory", [lambda p: LAMB(p, lr=0.1), lambda p: Lion(p, lr=0.1)], ids=["lamb", "lion"]
    )
    def test_state_and_parameters_carry_no_graph(self, factory) -> None:
        p = nn.Parameter(torch.tensor([2.0, -1.0]))
        opt = factory([p])
        p.grad = torch.tensor([0.5, 0.5])
        opt.step()
        assert all(not t.requires_grad for t in opt.state[p].values() if torch.is_tensor(t))
        assert p.grad_fn is None


def test_lion_multi_tensor_step_matches_per_parameter_reference() -> None:
    """The foreach path equals the per-parameter update, incl. a complex parameter."""
    torch.manual_seed(0)
    lr, (b1, b2), wd = 0.01, (0.9, 0.99), 0.1
    base = [torch.randn(4, 3), torch.randn(5), torch.randn(2, 2, dtype=torch.complex64)]
    ref = [p.clone() for p in base]
    ref_avg = [torch.zeros_like(torch.view_as_real(p) if p.is_complex() else p) for p in ref]
    live = [p.clone().requires_grad_() for p in base]
    opt = Lion(live, lr=lr, betas=(b1, b2), weight_decay=wd)
    for _ in range(3):
        grads = [torch.randn_like(p) for p in base]
        for p, g in zip(live, grads, strict=True):
            p.grad = g.clone()
        opt.step()
        for p, g, avg in zip(ref, grads, ref_avg, strict=True):
            pr = torch.view_as_real(p) if p.is_complex() else p
            gr = torch.view_as_real(g) if g.is_complex() else g
            pr.mul_(1 - lr * wd)
            pr.add_(avg.mul(b1).add(gr, alpha=1 - b1).sign_(), alpha=-lr)
            avg.mul_(b2).add_(gr, alpha=1 - b2)
    for a, b in zip(live, ref, strict=True):
        torch.testing.assert_close(a.detach(), b)
