"""OptimizationBuilder: scheduler + optimizer knobs must reach torch.

Regression suite for issue #533 (scheduler block discarded) and for the same
silent-discard hole on ``optimization.optimizer.kwargs``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn

from spectramr.domain.exceptions import ConfigurationError
from spectramr.infrastructure.training.builders.optimization_builder import (
    OptimizationBuilder,
)
from spectramr.infrastructure.training.optimizer_resolution import (
    OptimizerConfigurationError,
    resolve_optimizer_spec,
)


def _registry_names() -> set[str]:
    """Canonical registered optimizer names, aliases excluded.

    Read from the registry at collection time rather than hand-listed: the
    factory's bug survived because the coverage suite next door enumerated six
    names, all of which happened to be handled. Aliases are dropped because
    ``OPTIMIZER_ALIASES`` normalises them onto a canonical entry, so testing
    both spellings tests the same class twice.
    """
    from spectramr.config.schemas.enums import OPTIMIZER_NAMES
    from spectramr.infrastructure.training.optimizer_registry import OptimizerRegistry

    return set(OptimizerRegistry.list_available()) & OPTIMIZER_NAMES


#: Knobs that phase 8 moved onto the `optimizer:` sub-block, and the three that
#: were also renamed on the way. The stand-in FOLDS them exactly as the real
#: schema does, so every call site below still reads `_Opt(eps=..., ...)` -- the
#: fixture mirrors production instead of forcing 20 call sites to spell the
#: nesting out.
_OPTIMIZER_KNOBS = {
    "learning_rate", "optimizer_type", "weight_decay", "beta1", "beta2", "betas",
    "momentum", "eps", "nesterov", "amsgrad", "optimizer_kwargs", "lookahead",
    "generator_learning_rate", "discriminator_learning_rate",
    "param_group_overrides",
}
_OPTIMIZER_RENAMES = {
    "optimizer_type": "type",
    "optimizer_kwargs": "kwargs",
    "param_group_overrides": "param_groups",
}


class _Optimizer:
    """Stand-in for ``optimization.optimizer``."""

    def __init__(self, **kw: Any) -> None:
        self.type = "adamw"
        self.learning_rate = 5e-5
        self.weight_decay = 1e-4
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.betas = None
        self.momentum = 0.9
        self.eps = 1e-8
        self.nesterov = None
        self.amsgrad = None
        self.kwargs: dict[str, Any] = {}
        self.param_groups = None
        self.lookahead = None
        self.generator_learning_rate = None
        self.discriminator_learning_rate = None
        #: The resolver reads this to tell "declared" from "left at default";
        #: an empty set means "not introspectable", which is the documented
        #: stand-in behaviour.
        self.model_fields_set: set[str] = set()
        for k, v in kw.items():
            setattr(self, k, v)
            self.model_fields_set.add(k)


class _Opt:
    """Minimal stand-in for the frozen optimization block."""

    def __init__(self, **kw: Any) -> None:
        self.precision = SimpleNamespace(enabled=False, dtype=None)
        # The scheduler's default period counts optimizer steps, so it reads this.
        self.gradient = SimpleNamespace(accumulation_steps=1)
        self.lr_scheduler_strategy = "cosine"
        self.lr_scheduler_kwargs: dict[str, Any] = {}
        self.scheduler: dict[str, Any] | None = None
        self.scheduler_type = None
        self.T_max = None
        self.eta_min = None
        self.warmup_steps = 0

        opt_kw = {
            _OPTIMIZER_RENAMES.get(k, k): v
            for k, v in kw.items()
            if k in _OPTIMIZER_KNOBS
        }
        self.optimizer = _Optimizer(**opt_kw)
        for k, v in kw.items():
            if k not in _OPTIMIZER_KNOBS:
                setattr(self, k, v)


class _Training:
    def __init__(self, max_iterations: int | None = 30000) -> None:
        self.max_iterations = max_iterations


class _Cfg:
    def __init__(self, optimization: _Opt, max_iterations: int | None = 30000) -> None:
        self.optimization = optimization
        self.training = _Training(max_iterations)


def _builder(opt: _Opt, max_iterations: int | None = 30000) -> OptimizationBuilder:
    model = nn.Linear(2, 2)
    b = OptimizationBuilder(_Cfg(opt, max_iterations), models={"generator": model})
    b._optimizers = {"opt_g": torch.optim.AdamW(model.parameters(), lr=5e-5)}
    return b


class TestSchedulerWiring:
    def test_flat_block_reaches_torch(self):
        """The exp_11 block must produce warm restarts with T_0=50000."""
        b = _builder(
            _Opt(
                scheduler={
                    "warmup_steps": 3500,
                    "T_0": 50000,
                    "T_mult": 2,
                    "eta_min": 1e-6,
                    "warmup_start_lr": 1e-6,
                }
            )
        )
        b.build_schedulers()
        spec = b.scheduler_spec
        assert spec is not None
        assert spec.name == "cosine_annealing_warm_restarts"
        assert spec.params["T_0"] == 50000
        assert spec.warmup_steps == 3500
        # warmup is applied immediately, not one step later
        assert b._optimizers["opt_g"].param_groups[0]["lr"] == pytest.approx(1e-6)

    def test_t_max_comes_from_max_iterations(self):
        b = _builder(_Opt(scheduler={"type": "cosine", "eta_min": 1e-6}))
        b.build_schedulers()
        assert b.scheduler_spec.params["T_max"] == 30000

    def test_no_scheduler_block_builds_nothing(self):
        # The stand-in's default lr_scheduler_strategy counts as a declared family.
        b = _builder(_Opt(scheduler=None, lr_scheduler_strategy=None))
        b.build_schedulers()
        assert b._schedulers == {}
        assert b.scheduler_spec is None

    def test_unroutable_scheduler_knob_raises(self):
        b = _builder(_Opt(scheduler={"type": "cosine", "T_max": 100, "gamma": 0.5}))
        with pytest.raises(ConfigurationError, match="gamma"):
            b.build_schedulers()

    def test_spec_is_provenance_serialisable(self):
        b = _builder(_Opt(scheduler={"type": "step_lr", "step_size": 10, "gamma": 0.5}))
        b.build_schedulers()
        rec = b.scheduler_spec.as_provenance()
        assert rec == {"scheduler": "step_lr", "step_size": 10, "gamma": 0.5}


class TestOptimizerKwargs:
    """The #533 optimizer-kwargs invariants, now asserted on the resolver.

    ``_unconsumed_optimizer_kwargs`` used to own this. It validated
    ``optimizer_kwargs`` against the optimizer signature — the right idea — but
    first stripped ``betas``/``eps``/``momentum`` from the declared set on the
    assumption that ``_create_optimizer`` forwarded those itself. It only partly
    did, so ``optimizer_kwargs: {betas: ...}`` was stripped here AND dropped
    there. ``resolve_optimizer_spec`` now owns the whole decision, so these
    assertions moved onto it rather than being deleted with the helper.
    """

    def test_extra_kwarg_is_forwarded_not_dropped(self):
        spec = resolve_optimizer_spec(_Opt(optimizer_kwargs={"amsgrad": True}))
        assert spec.params["amsgrad"] is True

    def test_a_kwarg_duplicating_a_typed_field_is_allowed_when_it_agrees(self):
        spec = resolve_optimizer_spec(_Opt(optimizer_kwargs={"eps": 1e-8}))
        assert spec.params["eps"] == 1e-8

    def test_a_kwarg_contradicting_a_typed_field_raises(self):
        """Was silently stripped, so the typed field won and the YAML lied."""
        opt = _Opt(eps=1e-8, optimizer_kwargs={"eps": 1e-4})
        with pytest.raises(OptimizerConfigurationError, match="eps"):
            resolve_optimizer_spec(opt)

    def test_unknown_kwarg_raises_rather_than_being_dropped(self):
        opt = _Opt(optimizer_kwargs={"nesterov_momentum_typo": 0.9})
        with pytest.raises(OptimizerConfigurationError, match="nesterov_momentum_typo"):
            resolve_optimizer_spec(opt)

    def test_empty_kwargs_is_a_noop(self):
        spec = resolve_optimizer_spec(_Opt())
        assert "amsgrad" not in spec.params

    def test_forwarded_kwarg_lands_on_the_built_optimizer(self):
        b = _builder(_Opt(optimizer_kwargs={"amsgrad": True}))
        built = b._create_optimizer(nn.Linear(2, 2), b._config.optimization)
        assert built.defaults["amsgrad"] is True

    def test_betas_reach_the_adam_family_beyond_adam_and_adamw(self):
        """The headline regression: a hardcoded ``("Adam", "AdamW")`` tuple meant
        beta1/beta2 were silently dropped for adamax, nadam and radam — all of
        which accept ``betas``."""
        for name in ("adam", "adamw", "adamax", "nadam", "radam"):
            spec = resolve_optimizer_spec(
                _Opt(optimizer_type=name, beta1=0.85, beta2=0.995)
            )
            assert spec.params["betas"] == (0.85, 0.995), name

    def test_weight_decay_is_dropped_for_optimizers_that_cannot_take_it(self):
        """lbfgs/sparseadam accept no weight_decay; it used to be forwarded
        unconditionally and TypeError'd at construction."""
        spec = resolve_optimizer_spec(_Opt(optimizer_type="lbfgs"))
        assert "weight_decay" not in spec.params
        assert "weight_decay" in spec.dropped_defaults


class TestGradScalerIsNoLongerFabricated:
    """The builder used to hand the pipeline a scaler nothing ever called.

    Not a leaked allocation: the pre-1.0 checkpoint writer persisted ITS state, so
    every fp16 checkpoint carried an untouched scale of 65536 and a zero
    growth-tracker while the live ``NativeScaler`` / ``ComplexGradScaler`` on
    the strategy was never written. A fabricated field is worse than a missing
    one -- ``scaler_state`` was present, well-formed, and restored on resume,
    so nothing ever looked wrong.
    """

    def test_it_is_a_chainable_no_op(self):
        from spectramr.infrastructure.training.builders.optimization_builder import (
            OptimizationBuilder,
        )

        builder = OptimizationBuilder.__new__(OptimizationBuilder)
        builder._scaler = None
        assert builder.build_grad_scaler() is builder
        assert builder._scaler is None

    def test_it_does_not_touch_an_explicitly_set_scaler(self):
        from spectramr.infrastructure.training.builders.optimization_builder import (
            OptimizationBuilder,
        )

        sentinel = object()
        builder = OptimizationBuilder.__new__(OptimizationBuilder)
        builder._scaler = sentinel
        builder.build_grad_scaler()
        assert builder._scaler is sentinel

    def test_the_single_reader_is_capture_resolve_scaler(self):
        """Guards the SSOT: if a second resolver appears, the two disagree
        about which scaler a resume restores."""
        import re
        from pathlib import Path

        from spectramr.infrastructure import checkpointing

        definer = re.compile(r"^\s*def _?resolve_scaler\b", re.MULTILINE)
        package = Path(checkpointing.__file__).parent
        definers = sorted(
            str(p.relative_to(package))
            for p in package.rglob("*.py")
            if definer.search(p.read_text(encoding="utf-8"))
        )
        assert definers == ["capture.py"]


class TestTheStandInDoesNotDivergeFromTheSchema:
    """`_Opt` folds its kwargs exactly as the schema does, which keeps ~20 call
    sites readable but means none of them exercises the NESTED spelling. If the
    real fold broke, `_Opt` would keep agreeing with itself -- the same
    "second resolver that agrees until it doesn't" failure the fold exists to
    remove. These two go through the real schema instead.
    """

    def test_the_real_schema_resolves_the_same_spec_as_the_stand_in(self) -> None:
        from spectramr.config.schemas.optimization import OptimizationConfigSchema

        real = OptimizationConfigSchema(
            optimizer={"type": "adamw", "learning_rate": 5e-5, "eps": 1e-4}
        )
        assert resolve_optimizer_spec(real).as_provenance() == (
            resolve_optimizer_spec(_Opt(optimizer_type="adamw", eps=1e-4))
        ).as_provenance()

    def test_the_stand_in_mirrors_where_the_schema_puts_each_knob(self) -> None:
        """Anti-drift: a knob added to `_OPTIMIZER_KNOBS` that the schema does
        NOT carry on `optimizer:` would make the stand-in a fiction."""
        from spectramr.config.schemas.optimization import (
            OptimizationConfigSchema,
            OptimizerConfigSchema,
        )

        real_names = set(OptimizerConfigSchema.model_fields)
        mirrored = {_OPTIMIZER_RENAMES.get(k, k) for k in _OPTIMIZER_KNOBS}
        assert mirrored <= real_names, (
            f"the stand-in routes {sorted(mirrored - real_names)} onto "
            "`optimizer:`, but the schema does not declare them there"
        )
        # And nothing it routes is still a top-level field.
        assert not mirrored & set(OptimizationConfigSchema.model_fields)


class TestCreateSingleOptimizerHonoursTheRegistryTable:
    """The static factory must forward the same knobs the config path does.

    ``resolve_optimizer_spec`` was converted to signature-driven forwarding;
    ``create_single_optimizer`` was not, and kept the hand-maintained family
    tuple (``betas`` for adam/adamw/nadam/radam, ``weight_decay`` always). So
    the two dispatch paths disagreed for 13 of the 21 registered names --
    ``lbfgs``/``rprop``/``sparseadam`` raised ``TypeError`` at construction,
    and ten more silently trained on library defaults for a knob the caller
    had asked for.

    Parametrised over the WHOLE registry rather than a hand-listed set: a
    curated list is what let the gap survive (``TestOptimizerTypeCoverage``
    enumerated six names and every one of them happened to be handled).
    """

    @staticmethod
    def _params() -> list[torch.nn.Parameter]:
        return list(nn.Linear(4, 2).parameters())

    @staticmethod
    def _build_or_skip(name: str, **kw: Any):
        """Construct, skipping names whose optional extra is absent.

        Derived from the raised hint, not from a hardcoded ``{bnb, schedulefree}``
        set -- those wrappers register unconditionally *precisely* so a missing
        extra is distinguishable from an unknown name, and a hand-listed skip
        set here would go stale the moment one is installed in CI.
        """
        try:
            return OptimizationBuilder.create_single_optimizer(
                TestCreateSingleOptimizerHonoursTheRegistryTable._params(), **kw
            )
        except ImportError as exc:  # optional extra not installed
            pytest.skip(f"{name}: {exc}")

    @pytest.mark.parametrize("name", sorted(_registry_names()))
    def test_every_registered_optimizer_is_constructible(self, name: str) -> None:
        """No registered name may raise TypeError on the default call.

        ``lbfgs``/``rprop``/``sparseadam`` did, because ``weight_decay`` was
        forwarded to constructors that do not take it.
        """
        built = self._build_or_skip(name, optimizer_type=name)
        assert isinstance(built, torch.optim.Optimizer)
        assert built.param_groups[0]["params"]

    @pytest.mark.parametrize("name", sorted(_registry_names()))
    def test_factory_forwards_exactly_what_the_registry_accepts(
        self, name: str
    ) -> None:
        """Every portable knob the optimizer accepts must actually arrive.

        This is the silent half of the bug: ``adamax``/``lamb``/``lion`` accept
        ``betas`` and never saw them, ``rmsprop``/``lars`` accept ``momentum``
        and never saw it. A dropped knob leaves the arm on library defaults
        while the call site reads as though it configured them.
        """
        from spectramr.infrastructure.training.optimizer_registry import (
            accepted_optimizer_kwargs,
        )

        accepted = accepted_optimizer_kwargs(name)
        probes = {"betas": (0.85, 0.995), "weight_decay": 0.017, "momentum": 0.77}
        # Ask only for what this optimizer takes: passing an unaccepted knob
        # EXPLICITLY is the case the sibling test below pins, and it raises.
        asked = {k: v for k, v in probes.items() if k in accepted}
        built = self._build_or_skip(name, optimizer_type=name, **asked)

        for knob, value in probes.items():
            if knob in accepted:
                assert (
                    built.defaults[knob] == value
                ), f"{name} accepts {knob!r} but the factory did not forward it"
            else:
                assert knob not in built.defaults, f"{name} was handed {knob!r}"

    def test_an_explicit_unaccepted_knob_raises_rather_than_being_dropped(
        self,
    ) -> None:
        """Pitfall #15: the caller asked, so silence is a broken promise.

        Mirrors ``_partition_kwargs`` -- a knob left at its default is dropped
        quietly (the caller never asked), an explicit one raises.
        """
        with pytest.raises(ValueError, match="does not accept"):
            OptimizationBuilder.create_single_optimizer(
                self._params(), optimizer_type="lbfgs", weight_decay=0.01
            )

    def test_the_same_default_is_dropped_quietly(self) -> None:
        built = OptimizationBuilder.create_single_optimizer(
            self._params(), optimizer_type="lbfgs"
        )
        assert "weight_decay" not in built.defaults

    def test_alpha_stays_rmsprop_scoped_because_asgd_means_something_else(
        self,
    ) -> None:
        """``alpha`` is the one name here whose MEANING is not portable.

        RMSprop's alpha is a squared-gradient smoothing constant (0.99); ASGD's
        is the power for its eta update (0.75). A signature-driven forward
        matches on the name and would hand ASGD the wrong quantity -- so this
        knob must refuse rather than generalise.
        """
        rms = OptimizationBuilder.create_single_optimizer(
            self._params(), optimizer_type="rmsprop", alpha=0.9
        )
        assert rms.defaults["alpha"] == 0.9

        with pytest.raises(ValueError, match="RMSprop-only"):
            OptimizationBuilder.create_single_optimizer(
                self._params(), optimizer_type="asgd", alpha=0.9
            )
        # ...and ASGD keeps torch's own default, not RMSprop's.
        assert (
            OptimizationBuilder.create_single_optimizer(
                self._params(), optimizer_type="asgd"
            ).defaults["alpha"]
            == 0.75
        )

    def test_unknown_name_still_raises_with_the_available_list(self) -> None:
        with pytest.raises(ValueError, match=r"(?i)unknown optimizer"):
            OptimizationBuilder.create_single_optimizer(
                self._params(), optimizer_type="nonexistent_xyz"
            )


class TestParameterFreeGenerator:
    """A declared parameter-free solver gets no optimizer; anything else still needs one."""

    class _Solver(nn.Module):
        parameter_free = True

    def test_declared_solver_builds_no_optimizer_and_validates(self):
        b = OptimizationBuilder(_Cfg(_Opt()), models={"generator": self._Solver()})
        optimizers, schedulers, _ = b.build_optimizers().build_schedulers().validate().build()
        assert optimizers == {} and schedulers == {}

    def test_an_undeclared_empty_model_still_raises(self):
        b = OptimizationBuilder(_Cfg(_Opt()), models={"generator": nn.Sequential()})
        with pytest.raises(ValueError, match="does not declare parameter_free"):
            b.build_optimizers()

    def test_a_network_still_requires_its_optimizer(self):
        b = OptimizationBuilder(_Cfg(_Opt()), models={"generator": nn.Linear(2, 2)})
        b._optimizers = {}
        with pytest.raises(ValueError, match="No optimizers created"):
            b.validate()
