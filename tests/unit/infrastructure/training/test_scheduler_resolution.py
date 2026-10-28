"""Scheduler-spec resolution: the flat ``scheduler:`` block must be honoured.

Regression suite for issue #533. Before the fix,
``OptimizationBuilder.build_schedulers`` read ``scheduler["type"]`` and
``scheduler["kwargs"]``; 226/226 experiment configs declare neither, so every
declared scheduler parameter was discarded and every arm silently ran
``CosineAnnealingLR(T_max=100, eta_min=1e-6)`` -- an LR that oscillates
5e-5 <-> 1e-6 with a 400-iteration period instead of annealing.
"""

from __future__ import annotations

from itertools import pairwise
from typing import Any

import pytest
import torch

from spectramr.config.schemas.optimization import OptimizationConfigSchema
from spectramr.domain.exceptions import ConfigurationError
from spectramr.infrastructure.training.scheduler_resolution import (
    SCHEDULER_PARAM_KEYS,
    SchedulerSpec,
    resolve_scheduler_spec,
)


def _opt(**kwargs) -> OptimizationConfigSchema:
    base: dict[str, Any] = {"learning_rate": 5e-5}
    base.update(kwargs)
    return OptimizationConfigSchema(**base)


class TestFlatSchedulerBlock:
    """The form every config in the corpus actually uses."""

    def test_flat_block_is_not_discarded(self):
        """The exp_11 block: flat keys, no ``type``, no ``kwargs``."""
        spec = resolve_scheduler_spec(
            _opt(
                lr_scheduler_strategy="cosine",
                scheduler={
                    "warmup_steps": 3500,
                    "T_0": 50000,
                    "T_mult": 2,
                    "eta_min": 1.0e-06,
                    "warmup_start_lr": 1.0e-06,
                },
            ),
            max_iterations=30000,
        )
        assert spec is not None
        # T_0/T_mult belong to warm restarts, not to plain cosine annealing.
        assert spec.name == "cosine_annealing_warm_restarts"
        assert spec.params == {"T_0": 50000, "T_mult": 2, "eta_min": 1.0e-06}
        assert spec.warmup_steps == 3500
        assert spec.warmup_start_lr == 1.0e-06

    def test_lr_scheduler_strategy_is_wired(self):
        """146 configs declare the type here; it used to be read by nobody."""
        spec = resolve_scheduler_spec(
            _opt(
                lr_scheduler_strategy="cosine_annealing_warm_restarts",
                scheduler={"T_0": 10000, "T_mult": 2},
            ),
            max_iterations=30000,
        )
        assert spec.name == "cosine_annealing_warm_restarts"
        assert spec.params["T_0"] == 10000

    def test_t_max_defaults_to_max_iterations_not_100(self):
        """The 100 default is what caused the 400-iteration LR oscillation."""
        spec = resolve_scheduler_spec(
            _opt(lr_scheduler_strategy="cosine", scheduler={"eta_min": 1e-6}),
            max_iterations=30000,
        )
        assert spec.name == "cosine_annealing"
        assert spec.params["T_max"] == 30000

    def test_explicit_t_max_wins_over_max_iterations(self):
        spec = resolve_scheduler_spec(
            _opt(scheduler={"type": "cosine", "T_max": 1234}),
            max_iterations=30000,
        )
        assert spec.params["T_max"] == 1234

    def test_legacy_nested_kwargs_form_still_works(self):
        spec = resolve_scheduler_spec(
            _opt(
                scheduler={"type": "step_lr", "kwargs": {"step_size": 10, "gamma": 0.5}}
            ),
            max_iterations=30000,
        )
        assert spec.name == "step_lr"
        assert spec.params == {"step_size": 10, "gamma": 0.5}

    def test_warm_up_epochs_alias(self):
        """17 configs spell warmup as ``warm_up_epochs``/``warm_up_start_lr``."""
        spec = resolve_scheduler_spec(
            _opt(
                scheduler={
                    "type": "cosine",
                    "T_0": 100,
                    "T_mult": 2,
                    "warm_up_epochs": 50,
                    "warm_up_start_lr": 1e-7,
                }
            ),
            max_iterations=30000,
        )
        assert spec.warmup_steps == 50
        assert spec.warmup_start_lr == 1e-7

    def test_no_scheduler_block_returns_none(self):
        assert resolve_scheduler_spec(_opt(scheduler=None), max_iterations=100) is None


class TestRaisesInsteadOfDiscarding:
    """Pitfall #15: an unread knob must fail loudly, never be dropped."""

    def test_unroutable_key_raises(self):
        with pytest.raises(ConfigurationError, match="step_size"):
            resolve_scheduler_spec(
                _opt(scheduler={"type": "cosine", "T_max": 100, "step_size": 7}),
                max_iterations=30000,
            )

    def test_typo_raises_rather_than_silently_defaulting(self):
        with pytest.raises(ConfigurationError, match="eta_minn"):
            resolve_scheduler_spec(
                _opt(scheduler={"type": "cosine", "eta_minn": 1e-6}),
                max_iterations=30000,
            )

    def test_unknown_scheduler_name_raises(self):
        with pytest.raises(ConfigurationError, match="nosuchsched"):
            resolve_scheduler_spec(
                _opt(lr_scheduler_strategy="nosuchsched", scheduler={}),
                max_iterations=30000,
            )

    def test_conflicting_flat_and_nested_value_raises(self):
        """``optimization.T_max`` disagreeing with ``scheduler.T_max``."""
        with pytest.raises(ConfigurationError, match="T_max"):
            resolve_scheduler_spec(
                _opt(T_max=500, scheduler={"type": "cosine", "T_max": 900}),
                max_iterations=30000,
            )

    def test_agreeing_flat_and_nested_value_is_accepted(self):
        spec = resolve_scheduler_spec(
            _opt(T_max=900, scheduler={"type": "cosine", "T_max": 900}),
            max_iterations=30000,
        )
        assert spec.params["T_max"] == 900

    def test_flat_optimization_level_knobs_are_promoted(self):
        """Flat ``optimization.T_max``/``eta_min`` were documented as inert."""
        spec = resolve_scheduler_spec(
            _opt(scheduler_type="cosine", T_max=777, eta_min=1e-7, scheduler={}),
            max_iterations=30000,
        )
        assert spec.name == "cosine_annealing"
        assert spec.params == {"T_max": 777, "eta_min": 1e-7}

    def test_cosine_without_resolvable_t_max_raises(self):
        with pytest.raises(ConfigurationError, match="T_max"):
            resolve_scheduler_spec(
                _opt(scheduler={"type": "cosine"}), max_iterations=None
            )


class TestNameAliases:
    @pytest.mark.parametrize(
        ("declared", "expected"),
        [
            ("cosine", "cosine_annealing"),
            ("cosine_annealing", "cosine_annealing"),
            ("step", "step_lr"),
            ("step_lr", "step_lr"),
            ("linear_decay", "linear"),
            ("cosine_annealing_warm_restarts", "cosine_annealing_warm_restarts"),
        ],
    )
    def test_alias_resolves(self, declared, expected):
        spec = resolve_scheduler_spec(
            _opt(lr_scheduler_strategy=declared, scheduler={}), max_iterations=1000
        )
        assert spec.name == expected

    @pytest.mark.parametrize(
        ("declared", "expected"),
        [
            ("warmup_cosine", "cosine_annealing"),
            ("warmup", "constant"),
            ("linear_warmup", "constant"),
        ],
    )
    def test_warmup_bearing_alias_resolves_only_with_a_warmup(self, declared, expected):
        """Split out of the table above: these names carry a second promise.

        The alias maps them onto a decay family; the warmup half comes from the
        wrapper ``warmup_steps`` selects. Resolving one without a warmup length
        used to drop that half silently, so the assertion now has to supply it
        -- and :class:`TestWarmupBearingAliases` covers the omission case.
        """
        spec = resolve_scheduler_spec(
            _opt(lr_scheduler_strategy=declared, scheduler={"warmup_steps": 50}),
            max_iterations=1000,
        )
        assert spec.name == expected
        assert spec.warmup_steps == 50

    @pytest.mark.parametrize("declared", ["plateau", "reduce_on_plateau", "reduce_lr_on_plateau"])
    def test_the_plateau_scheduler_is_rejected_with_its_reason(self, declared):
        """It was built and never stepped with a metric, so it held the LR (#2301)."""
        with pytest.raises(ConfigurationError, match="#2301"):
            resolve_scheduler_spec(_opt(lr_scheduler_strategy=declared, scheduler={}), 1000)

    def test_no_registered_scheduler_needs_a_metric(self):
        """The loop steps every scheduler with no argument; a metric-driven one
        can only be added together with the validation-time call that feeds it."""
        import inspect

        from spectramr.infrastructure.training.scheduler_system import SCHEDULER_REGISTRY

        p = [torch.nn.Parameter(torch.zeros(1))]
        for name, factory in SCHEDULER_REGISTRY.items():
            sched = factory(torch.optim.SGD(p, lr=0.1), {})
            required = [
                a
                for a in inspect.signature(sched.step).parameters.values()
                if a.default is inspect.Parameter.empty and a.kind is a.POSITIONAL_OR_KEYWORD
            ]
            assert not required, f"{name}.step requires {required}"

    def test_every_registry_name_has_a_param_key_set(self):
        """A factory with no declared param set cannot be validated."""
        from spectramr.infrastructure.training.scheduler_system import SCHEDULER_REGISTRY

        missing = set(SCHEDULER_REGISTRY) - set(SCHEDULER_PARAM_KEYS)
        assert not missing, f"factories with no declared param keys: {sorted(missing)}"


class TestBuiltSchedulerBehaviour:
    """The end-to-end property the bug violated: the LR must anneal, not cycle."""

    def _lr_trace(self, spec: SchedulerSpec, iters: int, accum: int) -> list[float]:
        from spectramr.infrastructure.training.scheduler_resolution import (
            build_scheduler_from_spec,
        )

        p = [torch.nn.Parameter(torch.zeros(1))]
        opt = torch.optim.AdamW(p, lr=5e-5)
        sched = build_scheduler_from_spec(spec, opt)
        trace = []
        for it in range(1, iters + 1):
            # Mirror the real loop: the LR in force at iteration ``it`` is the
            # one present before the scheduler advances.
            trace.append(opt.param_groups[0]["lr"])
            if it % accum == 0:
                sched.step()
        return trace

    def test_cosine_anneals_monotonically_over_the_run(self):
        spec = resolve_scheduler_spec(
            _opt(
                scheduler={"type": "cosine", "eta_min": 1e-6},
                gradient={"accumulation_steps": 2},
            ),
            max_iterations=4000,
        )
        trace = self._lr_trace(spec, iters=4000, accum=2)
        # One full cosine over the run's 2000 optimizer steps.
        assert trace[0] > trace[-1]
        assert trace[-1] == pytest.approx(1e-6, rel=0.01)
        # and it must never come back up (the oscillation signature)
        assert all(b <= a + 1e-12 for a, b in pairwise(trace))

    def test_regression_no_400_iteration_oscillation(self):
        """The exact exp_11 config must not reproduce the logged LR cycle."""
        spec = resolve_scheduler_spec(
            _opt(
                lr_scheduler_strategy="cosine",
                scheduler={
                    "warmup_steps": 3500,
                    "T_0": 50000,
                    "T_mult": 2,
                    "eta_min": 1.0e-06,
                    "warmup_start_lr": 1.0e-06,
                },
            ),
            max_iterations=30000,
        )
        trace = self._lr_trace(spec, iters=20000, accum=2)
        # Pre-fix the logged LRs were min/max/min/max at these iterations.
        sampled = [trace[i - 1] for i in (5000, 10000, 15000, 20000)]
        assert not (
            sampled[0] < 2e-6 and sampled[1] > 4e-5
        ), f"LR still oscillating between eta_min and base: {sampled}"
        # warmup ramps from warmup_start_lr, which used to be dropped entirely
        assert trace[0] < 1e-5

    def test_warmup_is_applied_when_declared(self):
        spec = resolve_scheduler_spec(
            _opt(
                scheduler={
                    "type": "cosine",
                    "warmup_steps": 100,
                    "warmup_start_lr": 1e-7,
                }
            ),
            max_iterations=1000,
        )
        trace = self._lr_trace(spec, iters=400, accum=1)
        assert trace[0] == pytest.approx(1e-7, rel=0.5)
        assert trace[99] > trace[0]


class TestDefaultPeriodIsTheRunInSchedulerSteps:
    """The default period counts the steps the main scheduler actually takes.

    The loop steps schedulers once per optimizer step, and a warmup wrapper
    holds the main scheduler for its first ``warmup_steps``. A period of
    ``max_iterations`` micro-batches left the 11 ``accumulation_steps: 4``
    arms at 85 % of their base LR when the run ended, and ``max_iterations:
    -1`` (epoch-bounded) produced ``T_max=-1``.
    """

    def test_period_counts_optimizer_steps_under_accumulation(self):
        spec = resolve_scheduler_spec(
            _opt(lr_scheduler_strategy="cosine", gradient={"accumulation_steps": 4}),
            max_iterations=100_000,
        )
        assert spec.params["T_max"] == 25_000

    def test_period_excludes_the_warmup(self):
        spec = resolve_scheduler_spec(
            _opt(scheduler={"type": "cosine", "warmup_steps": 100}),
            max_iterations=1000,
        )
        assert spec.params["T_max"] == 900

    def test_linear_total_iters_uses_the_same_run_length(self):
        spec = resolve_scheduler_spec(
            _opt(
                lr_scheduler_strategy="linear_decay",
                gradient={"accumulation_steps": 2},
                warmup_steps=50,
            ),
            max_iterations=1000,
        )
        assert spec.params["total_iters"] == 450

    def test_accumulated_run_ends_annealed(self):
        """The observable the defect broke: the LR at the end of the run."""
        from spectramr.infrastructure.training.scheduler_resolution import (
            build_scheduler_from_spec,
        )

        spec = resolve_scheduler_spec(
            _opt(
                lr_scheduler_strategy="cosine",
                scheduler={"eta_min": 1e-6},
                gradient={"accumulation_steps": 4},
            ),
            max_iterations=4000,
        )
        p = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.AdamW([p], lr=1e-4)
        sched = build_scheduler_from_spec(spec, opt)
        for it in range(4000):
            opt.step()
            if (it + 1) % 4 == 0:
                sched.step()
        assert opt.param_groups[0]["lr"] == pytest.approx(1e-6, rel=1e-3)

    @pytest.mark.parametrize("max_iterations", [-1, 0])
    def test_epoch_bounded_run_needs_a_declared_period(self, max_iterations):
        with pytest.raises(ConfigurationError, match="T_max"):
            resolve_scheduler_spec(
                _opt(scheduler={"warmup_steps": 1000}), max_iterations=max_iterations
            )

    def test_declared_period_is_honoured_on_an_epoch_bounded_run(self):
        spec = resolve_scheduler_spec(
            _opt(scheduler={"type": "cosine", "T_max": 5000}), max_iterations=-1
        )
        assert spec.params["T_max"] == 5000

    def test_warmup_that_consumes_the_run_raises(self):
        with pytest.raises(ConfigurationError, match="no optimizer steps"):
            resolve_scheduler_spec(
                _opt(
                    scheduler={"type": "cosine", "warmup_steps": 300},
                    gradient={"accumulation_steps": 4},
                ),
                max_iterations=1000,
            )


class TestDeclaredStrategyWithoutSchedulerBlock:
    """Issue #662: a declared family with no ``scheduler:`` dict must resolve.

    ``resolve_scheduler_spec`` returned ``None`` -- no scheduler at all -- the
    moment ``optimization.scheduler`` was absent, *before* it ever read
    ``lr_scheduler_strategy``. 531 corpus arms named a family and declared no
    dict, so they trained at a constant LR while their config said ``cosine``.
    Nothing warned: a ``None`` spec is also how "no scheduler wanted" is spelled.
    """

    def test_declared_strategy_alone_resolves(self):
        spec = resolve_scheduler_spec(
            _opt(lr_scheduler_strategy="cosine"), max_iterations=100_000
        )
        assert spec is not None, "a declared family with no dict must still resolve"
        assert spec.name == "cosine_annealing"
        # The period falls back to the run length, never to torch's default 100.
        assert spec.params == {"T_max": 100_000}

    def test_declared_scheduler_type_alone_resolves(self):
        spec = resolve_scheduler_spec(
            _opt(scheduler_type="step", scheduler={"step_size": 10}),
            max_iterations=1000,
        )
        assert spec is not None
        assert spec.name == "step_lr"

    def test_defaulted_strategy_still_means_no_scheduler(self):
        """The guard that keeps this fix from scheduling 305 unrelated arms.

        ``lr_scheduler_strategy`` carries ``default="cosine"``, so a resolved
        config ALWAYS presents a family name. Honouring the default rather than
        an explicit declaration would silently start annealing every arm that
        asked for nothing -- the failure this fix exists to remove, inverted.
        """
        opt = _opt()  # declares neither `scheduler` nor a strategy
        assert opt.lr_scheduler_strategy == "cosine", "precondition: the default is set"
        assert "lr_scheduler_strategy" not in opt.model_fields_set
        assert resolve_scheduler_spec(opt, max_iterations=100_000) is None

    def test_non_pydantic_standin_falls_back_to_presence(self):
        """Test doubles carry no ``model_fields_set``; a present name is the signal."""
        from types import SimpleNamespace

        stub = SimpleNamespace(
            scheduler=None,
            lr_scheduler_strategy="cosine",
            scheduler_type=None,
            gradient=SimpleNamespace(accumulation_steps=1),
        )
        spec = resolve_scheduler_spec(stub, max_iterations=500)
        assert spec is not None and spec.name == "cosine_annealing"

        silent = SimpleNamespace(
            scheduler=None, lr_scheduler_strategy=None, scheduler_type=None
        )
        assert resolve_scheduler_spec(silent, max_iterations=500) is None


class TestWarmupBearingAliases:
    """A name that promises warmup must not resolve to a schedule without it.

    ``_NAME_ALIASES`` maps ``warmup_cosine`` onto plain ``cosine_annealing``
    because the warmup is the wrapper's job. With no warmup length declared the
    alias therefore discarded the only thing the name asked for -- 16 corpus
    arms named ``warmup_cosine`` and silently got plain cosine.
    """

    def test_warmup_cosine_without_warmup_raises(self):
        with pytest.raises(ConfigurationError, match="names a warmup"):
            resolve_scheduler_spec(
                _opt(lr_scheduler_strategy="warmup_cosine"), max_iterations=1000
            )

    def test_warmup_cosine_with_warmup_resolves(self):
        spec = resolve_scheduler_spec(
            _opt(lr_scheduler_strategy="warmup_cosine", warmup_steps=250),
            max_iterations=1000,
        )
        assert spec.name == "cosine_annealing"
        assert spec.warmup_steps == 250

    def test_warmup_steps_from_the_scheduler_block_also_satisfies_it(self):
        spec = resolve_scheduler_spec(
            _opt(lr_scheduler_strategy="warmup_cosine", scheduler={"warmup_steps": 40}),
            max_iterations=1000,
        )
        assert spec.warmup_steps == 40

    def test_plain_cosine_needs_no_warmup(self):
        """The check keys on the DECLARED name, not on the resolved family."""
        spec = resolve_scheduler_spec(
            _opt(lr_scheduler_strategy="cosine"), max_iterations=1000
        )
        assert spec.name == "cosine_annealing"
        assert spec.warmup_steps == 0

    @pytest.mark.parametrize("declared", ["warmup", "linear_warmup"])
    def test_warmup_only_names_resolve_to_a_flat_base(self, declared):
        """``warmup`` is the wrapper alone: warm up, then hold the base LR."""
        spec = resolve_scheduler_spec(
            _opt(lr_scheduler_strategy=declared, warmup_steps=10_000),
            max_iterations=500_000,
        )
        assert spec.name == "constant"
        assert spec.warmup_steps == 10_000


class TestResolvedScheduleRecord:
    """The one record of the built schedule, stamped by provenance and the dump (#2760)."""

    @staticmethod
    def _config(opt: OptimizationConfigSchema) -> Any:
        from types import SimpleNamespace

        return SimpleNamespace(optimization=opt, training=SimpleNamespace(max_iterations=100))

    def test_an_undeclared_scheduler_records_none_despite_the_default(self):
        from spectramr.infrastructure.training.scheduler_resolution import (
            resolved_schedule_record,
        )

        opt = _opt()
        assert opt.lr_scheduler_strategy == "cosine", "precondition: the default is set"
        assert resolved_schedule_record(self._config(opt)) == {"scheduler": None}

    def test_a_declared_scheduler_is_recorded_as_built(self):
        from spectramr.infrastructure.training.scheduler_resolution import (
            resolved_schedule_record,
        )

        record = resolved_schedule_record(self._config(_opt(lr_scheduler_strategy="cosine")))
        assert record == {"scheduler": "cosine_annealing", "T_max": 100}
