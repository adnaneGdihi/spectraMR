"""Lookahead wrapper: the scheduler must reach the INNER optimizer.

This is the test plan-risk R6 called for and that was never written, which is
why the defect shipped. The failure mode is entirely silent: parameters keep
updating, the wrapper reports a textbook decay curve, and the inner optimizer
steps at the initial LR for the whole run.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from spectramr.infrastructure.training.optimizers.lookahead import (  # noqa: E402
    Lookahead,
)


def _tiny_model() -> "torch.nn.Module":
    torch.manual_seed(0)
    return torch.nn.Linear(4, 2)


def _one_step(model, optimizer) -> None:
    x = torch.randn(8, 4)
    optimizer.zero_grad()
    (model(x) ** 2).mean().backward()
    optimizer.step()


class TestLookaheadSchedulerReachesInnerOptimizer:
    """R6: a scheduler is attached to the WRAPPER, not the inner optimizer.

    ``OptimizationBuilder`` hands the scheduler ``self._optimizers[name]``,
    which IS the Lookahead wrapper. ``Lookahead.__init__`` copies the inner
    optimizer's ``param_groups``, so without an explicit sync the scheduler
    decays a dict nothing reads.
    """

    def test_inner_lr_tracks_the_wrapper_under_a_schedule(self) -> None:
        model = _tiny_model()
        inner = torch.optim.AdamW(model.parameters(), lr=0.1)
        wrapper = Lookahead(inner, la_steps=5, la_alpha=0.5)
        scheduler = torch.optim.lr_scheduler.StepLR(wrapper, step_size=1, gamma=0.5)

        observed = []
        for _ in range(4):
            _one_step(model, wrapper)
            observed.append(
                (wrapper.param_groups[0]["lr"], inner.param_groups[0]["lr"])
            )
            scheduler.step()

        # The regression: inner stayed at 0.1 for every entry.
        assert [round(i, 6) for _, i in observed] == [0.1, 0.05, 0.025, 0.0125]
        for wrapper_lr, inner_lr in observed:
            assert wrapper_lr == pytest.approx(inner_lr), (
                "wrapper and inner LR diverged; the scheduler is decaying a "
                "copy the inner optimizer never reads"
            )

    def test_inner_lr_actually_moves(self) -> None:
        """Guards the degenerate pass: equal-but-frozen would satisfy a
        naive equality assertion if the schedule never fired."""
        model = _tiny_model()
        inner = torch.optim.SGD(model.parameters(), lr=1.0)
        wrapper = Lookahead(inner, la_steps=2, la_alpha=0.5)
        scheduler = torch.optim.lr_scheduler.StepLR(wrapper, step_size=1, gamma=0.1)

        first = inner.param_groups[0]["lr"]
        for _ in range(3):
            _one_step(model, wrapper)
            scheduler.step()
        _one_step(model, wrapper)  # sync happens inside step()

        assert inner.param_groups[0]["lr"] < first

    def test_sync_carries_non_lr_hyperparameters_too(self) -> None:
        """The sync copies whatever the scheduler wrote, not an enumerated
        allow-list of keys -- a `weight_decay` schedule must land as well."""
        model = _tiny_model()
        inner = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.01)
        wrapper = Lookahead(inner, la_steps=5, la_alpha=0.5)

        wrapper.param_groups[0]["weight_decay"] = 0.5
        _one_step(model, wrapper)

        assert inner.param_groups[0]["weight_decay"] == pytest.approx(0.5)

    def test_lookahead_private_bookkeeping_does_not_leak_inward(self) -> None:
        """``counter`` is Lookahead's own state; pushing it onto the inner
        optimizer's group would put a meaningless key in its state_dict."""
        model = _tiny_model()
        inner = torch.optim.AdamW(model.parameters(), lr=0.1)
        wrapper = Lookahead(inner, la_steps=3, la_alpha=0.5)

        _one_step(model, wrapper)

        assert "counter" in wrapper.param_groups[0]
        assert "counter" not in inner.param_groups[0]

    def test_params_list_is_not_overwritten_by_the_sync(self) -> None:
        """`params` holds tensor identity; copying the wrapper's list over the
        inner one would repoint the optimizer at different objects."""
        model = _tiny_model()
        inner = torch.optim.AdamW(model.parameters(), lr=0.1)
        before = list(inner.param_groups[0]["params"])
        wrapper = Lookahead(inner, la_steps=3, la_alpha=0.5)

        _one_step(model, wrapper)

        after = list(inner.param_groups[0]["params"])
        assert all(a is b for a, b in zip(before, after))


class TestLookaheadResumePreservesParamGroups:
    """``load_state_dict`` installs the INNER optimizer's groups, whose ``params``
    are integer INDICES, then repairs them positionally from a FLAT list of
    ``self.state`` keys.

    Both halves of that repair are wrong, and both fail silently:

    * the flat list is indexed by the position *within* a group, so group 1
      slot 0 resolves to group 0's first tensor;
    * a parameter that never entered ``slow_state`` (``_update_slow`` skips
      ``p.grad is None``) has no entry to repair from, so the raw ``int``
      survives into ``param_groups``.

    Multi-group is the common case here -- ~25 strategies call
    ``add_param_group`` on ``opt_g`` -- and the checkpoint path reaches this
    (``checkpointing.capture`` and ``resume`` save and load the same wrapper object).
    """

    @staticmethod
    def _two_group_wrapper():
        torch.manual_seed(0)
        model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 2))
        inner = torch.optim.SGD(
            [
                {"params": list(model[0].parameters()), "lr": 0.1},
                {"params": list(model[1].parameters()), "lr": 0.9},
            ]
        )
        return model, Lookahead(inner, la_steps=2, la_alpha=0.5)

    def test_each_group_keeps_its_own_parameters_across_a_resume(self) -> None:
        model, wrapper = self._two_group_wrapper()
        _one_step(model, wrapper)

        before = [[id(p) for p in g["params"]] for g in wrapper.param_groups]
        wrapper.load_state_dict(wrapper.state_dict())
        after = [[id(p) for p in g["params"]] for g in wrapper.param_groups]

        assert after == before, (
            "resume repointed a group at another group's tensors; the second "
            "module stops receiving the outer update and the first is "
            "slow-synced twice"
        )

    def test_resume_does_not_alias_the_second_group_onto_the_first(self) -> None:
        """Sharper than identity: the concrete corruption is duplication."""
        model, wrapper = self._two_group_wrapper()
        _one_step(model, wrapper)

        wrapper.load_state_dict(wrapper.state_dict())

        flat = [id(p) for g in wrapper.param_groups for p in g["params"]]
        assert len(set(flat)) == len(
            flat
        ), "the same tensor appears in two param groups after resume"

    def test_resume_keeps_tensors_when_a_parameter_has_no_gradient(self) -> None:
        """A frozen parameter never enters ``slow_state``, so the positional
        repair runs out of entries and leaves the raw index behind."""
        torch.manual_seed(0)
        model = torch.nn.Linear(4, 2)
        frozen = torch.nn.Parameter(torch.randn(2), requires_grad=False)
        inner = torch.optim.SGD([{"params": [*model.parameters(), frozen], "lr": 0.1}])
        wrapper = Lookahead(inner, la_steps=2, la_alpha=0.5)
        _one_step(model, wrapper)

        wrapper.load_state_dict(wrapper.state_dict())

        kinds = [type(p).__name__ for p in wrapper.param_groups[0]["params"]]
        assert "int" not in kinds, f"resume left index placeholders: {kinds}"

    def test_stepping_after_a_resume_with_a_frozen_parameter_does_not_raise(
        self,
    ) -> None:
        """The user-visible symptom: every SLURM requeue of a model with a
        frozen encoder dies on the step where the sync counter wraps."""
        torch.manual_seed(0)
        model = torch.nn.Linear(4, 2)
        frozen = torch.nn.Parameter(torch.randn(2), requires_grad=False)
        inner = torch.optim.SGD([{"params": [*model.parameters(), frozen], "lr": 0.1}])
        wrapper = Lookahead(inner, la_steps=2, la_alpha=0.5)
        _one_step(model, wrapper)

        wrapper.load_state_dict(wrapper.state_dict())

        for _ in range(4):  # long enough for `counter` to wrap to 0
            _one_step(model, wrapper)

    def test_resume_restores_the_slow_weights_it_saved(self) -> None:
        """Identity alone is not enough -- the restored slow weights must be
        the saved ones, on the tensors they were saved for."""
        model, wrapper = self._two_group_wrapper()
        for _ in range(3):  # past one sync so slow weights exist and differ
            _one_step(model, wrapper)

        # Keep hard references to the ORIGINALS. Iterating ``param_groups``
        # after the resume would silently pass on corrupted state, because the
        # duplicated tensors it then contains are all present in ``saved``.
        originals = [p for g in wrapper.param_groups for p in g["params"]]
        saved = {
            id(p): wrapper.state[p]["slow_param"].clone()
            for p in originals
            if "slow_param" in wrapper.state[p]
        }
        assert len(saved) == len(originals), (
            "not every parameter recorded a slow weight; the test would not "
            "discriminate"
        )

        wrapper.load_state_dict(wrapper.state_dict())

        for p in originals:
            assert (
                "slow_param" in wrapper.state[p]
            ), "a parameter lost its slow weight across the resume"
            assert torch.equal(wrapper.state[p]["slow_param"], saved[id(p)])


class TestLookaheadResumeRejectsAShapeMismatch:
    """A group-count mismatch is a changed model, not a resumable checkpoint.

    Truncating to the shorter list would restore some groups' hyperparameters
    and silently leave the rest at their construction-time values -- a
    partially resumed optimizer that reports success. The inner optimizer's own
    ``load_state_dict`` rejects this first; the wrapper's ``strict=True`` keeps
    the invariant from depending on that ordering.
    """

    def test_resume_with_fewer_saved_groups_raises(self) -> None:
        torch.manual_seed(0)
        one = torch.nn.Linear(4, 2)
        wrapper = Lookahead(torch.optim.SGD(one.parameters(), lr=0.1), 2, 0.5)
        _one_step(one, wrapper)
        saved = wrapper.state_dict()

        two = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 2))
        wider = Lookahead(
            torch.optim.SGD(
                [
                    {"params": list(two[0].parameters()), "lr": 0.1},
                    {"params": list(two[1].parameters()), "lr": 0.9},
                ]
            ),
            2,
            0.5,
        )
        with pytest.raises(ValueError, match="number of parameter groups"):
            wider.load_state_dict(saved)


class TestScheduleFreeModeApiReachesTheInnerOptimizer:
    """Same family as the scheduler trap: a wrapper drops what it forgets to relay.

    Schedule-free optimizers keep an averaged sequence separate from the iterate
    the gradient is taken at, and ``train()``/``eval()`` swap between them.
    ``training_loop._set_optimizer_eval_mode`` finds them by duck-typing the
    object in ``pipeline.optimizers`` -- which, with Lookahead enabled, is the
    WRAPPER. ``Optimizer`` defines neither method, so the hook found nothing and
    validation silently graded the un-averaged iterate.
    """

    class _ScheduleFreeSGD(torch.optim.SGD):
        """Minimal stand-in: the real one needs the optional extra."""

        def __init__(self, *a, **kw) -> None:
            super().__init__(*a, **kw)
            self.mode = "train"

        def train(self) -> None:
            self.mode = "train"

        def eval(self) -> None:
            self.mode = "eval"

    def _wrapped(self) -> Lookahead:
        model = torch.nn.Linear(4, 2)
        return Lookahead(self._ScheduleFreeSGD(model.parameters(), lr=0.1), 2, 0.5)

    def test_the_wrapper_advertises_the_inner_mode_api(self) -> None:
        from spectramr.infrastructure.training.optimizers import (
            supports_schedule_free_modes,
        )

        assert supports_schedule_free_modes(self._wrapped())

    def test_toggling_the_wrapper_toggles_the_inner_optimizer(self) -> None:
        wrapper = self._wrapped()
        wrapper.eval()
        assert wrapper.optimizer.mode == "eval"
        wrapper.train()
        assert wrapper.optimizer.mode == "train"

    def test_a_plain_inner_optimizer_does_not_gain_a_phantom_mode_api(self) -> None:
        """The forward must be conditional, not a blanket ``__getattr__``.

        If Lookahead advertised ``train``/``eval`` unconditionally, every
        wrapped arm would report as schedule-free and the predicate would stop
        meaning anything.
        """
        from spectramr.infrastructure.training.optimizers import (
            supports_schedule_free_modes,
        )

        model = torch.nn.Linear(4, 2)
        plain = Lookahead(torch.optim.SGD(model.parameters(), lr=0.1), 2, 0.5)
        assert not hasattr(plain, "train")
        assert not supports_schedule_free_modes(plain)

    def test_an_unrelated_missing_attribute_still_raises_attributeerror(self) -> None:
        """A catch-all forward would mask real errors and make hasattr lie."""
        with pytest.raises(AttributeError):
            _ = self._wrapped().no_such_attribute

    def test_getattr_does_not_recurse_before_init_binds_the_inner(self) -> None:
        """``__getattr__`` fires for ``optimizer`` itself during construction.

        Reading ``self.optimizer`` there (rather than ``self.__dict__``) is an
        infinite recursion, so the guard is load-bearing rather than stylistic.
        """
        bare = Lookahead.__new__(Lookahead)
        with pytest.raises(AttributeError):
            _ = bare.train


class _CountingSGD(torch.optim.SGD):
    """SGD that advances per-group state in ``step`` and has a train/eval mode.

    Stands in for schedule-free, which keeps ``k``, ``weight_sum`` and
    ``train_mode`` in its param groups and refuses to step in eval mode.
    """

    def __init__(self, params, lr: float = 0.1) -> None:
        super().__init__(params, lr=lr)
        for group in self.param_groups:
            group["k"] = 0
            group["train_mode"] = False

    def train(self) -> None:
        for group in self.param_groups:
            group["train_mode"] = True

    def eval(self) -> None:
        for group in self.param_groups:
            group["train_mode"] = False

    def step(self, closure=None):
        for group in self.param_groups:
            if not group["train_mode"]:
                raise RuntimeError("not in train mode")
            group["k"] += 1
        return super().step(closure)


class TestLookaheadKeepsTheInnerOptimizerWhole:
    """The sync, a late parameter group and a save must not desynchronise the pair."""

    def test_a_group_added_after_construction_is_trained(self) -> None:
        model = _tiny_model()
        extra = torch.nn.Parameter(torch.ones(3))
        inner = torch.optim.SGD(model.parameters(), lr=0.1)
        wrapper = Lookahead(inner, la_steps=5, la_alpha=0.5)

        wrapper.add_param_group({"params": [extra], "lr": 0.1})
        extra.grad = torch.ones(3)
        wrapper.step()

        assert any(extra is p for g in inner.param_groups for p in g["params"])
        assert torch.allclose(extra.detach(), torch.full((3,), 0.9))
        assert len(wrapper.param_groups) == len(inner.param_groups) == 2

    def test_group_state_the_inner_optimizer_advances_is_not_reset(self) -> None:
        model = _tiny_model()
        inner = _CountingSGD(model.parameters(), lr=0.1)
        wrapper = Lookahead(inner, la_steps=5, la_alpha=0.5)

        wrapper.train()
        for _ in range(3):
            _one_step(model, wrapper)

        assert inner.param_groups[0]["train_mode"] is True
        assert inner.param_groups[0]["k"] == 3

    def test_a_scheduler_write_still_reaches_the_inner_optimizer(self) -> None:
        model = _tiny_model()
        inner = _CountingSGD(model.parameters(), lr=0.1)
        wrapper = Lookahead(inner, la_steps=5, la_alpha=0.5)
        scheduler = torch.optim.lr_scheduler.StepLR(wrapper, step_size=1, gamma=0.5)

        wrapper.train()
        _one_step(model, wrapper)
        scheduler.step()
        _one_step(model, wrapper)

        assert inner.param_groups[0]["lr"] == pytest.approx(0.05)
        assert inner.param_groups[0]["k"] == 2

    def test_state_dict_carries_the_latest_scheduler_write(self) -> None:
        model = _tiny_model()
        inner = torch.optim.AdamW(model.parameters(), lr=0.1)
        wrapper = Lookahead(inner, la_steps=5, la_alpha=0.5)
        scheduler = torch.optim.lr_scheduler.StepLR(wrapper, step_size=1, gamma=0.5)

        _one_step(model, wrapper)
        scheduler.step()

        assert wrapper.state_dict()["param_groups"][0]["lr"] == pytest.approx(0.05)

    def test_lookahead_defaults_do_not_leak_into_the_inner_optimizer(self) -> None:
        inner = torch.optim.SGD(_tiny_model().parameters(), lr=0.1)
        Lookahead(inner, la_steps=5, la_alpha=0.5)

        assert "la_steps" not in inner.defaults
        assert "la_alpha" not in inner.defaults
