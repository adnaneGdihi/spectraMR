"""Scheduler Coverage Tests.

PURPOSE:
Verifies the gap between `lr_scheduler_strategy` (a real schema field) and
`SCHEDULER_REGISTRY` (where schedulers actually live):

  1. Every SCHEDULER_REGISTRY key can be invoked and returns a valid scheduler.
  2. The `lr_scheduler_strategy` field in OptimizationConfigSchema is documented
     as ignored by OptimizationBuilder.build_schedulers() — this test makes the
     gap VISIBLE and AUDITABLE.
  3. If the gap is ever fixed, the "ghost field" test must be updated.
"""

from __future__ import annotations

from typing import ClassVar

import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def dummy_optimizer():
    model = nn.Linear(4, 2)
    return torch.optim.Adam(model.parameters(), lr=1e-3)


# ---------------------------------------------------------------------------
# 1. SCHEDULER_REGISTRY coverage: all registered types are callable
# ---------------------------------------------------------------------------


class TestSchedulerRegistryCoverage:
    """Every entry in SCHEDULER_REGISTRY must produce a valid LR scheduler."""

    def test_all_registered_schedulers_are_callable(self, dummy_optimizer):
        from spectramr.infrastructure.training.scheduler_system import SCHEDULER_REGISTRY

        assert SCHEDULER_REGISTRY, "SCHEDULER_REGISTRY is empty!"

        failed = []
        for name, factory_fn in SCHEDULER_REGISTRY.items():
            try:
                scheduler = factory_fn(dummy_optimizer, {})
                assert hasattr(scheduler, "step"), f"Scheduler '{name}' has no .step() method."
            except Exception as exc:
                failed.append((name, exc))
            finally:
                # Reset optimizer lr for next scheduler
                for g in dummy_optimizer.param_groups:
                    g["lr"] = 1e-3

        if failed:
            lines = "\n".join(f"  {name}: {exc}" for name, exc in failed)
            pytest.fail(f"The following schedulers raised during creation:\n{lines}")

    @pytest.mark.parametrize(
        "scheduler_name",
        [
            "cosine_annealing",
            "cosine_annealing_warm_restarts",
            "step_lr",
            "exponential",
        ],
    )
    def test_known_scheduler_is_registered(self, scheduler_name: str):
        """Core scheduler types must be present in the registry."""
        from spectramr.infrastructure.training.scheduler_system import SCHEDULER_REGISTRY

        assert scheduler_name in SCHEDULER_REGISTRY, (
            f"Expected '{scheduler_name}' in SCHEDULER_REGISTRY but it's missing.\n"
            f"Registered: {sorted(SCHEDULER_REGISTRY.keys())}"
        )

    def test_scheduler_factory_create_scheduler_works(self, dummy_optimizer):
        """SchedulerFactory.create_scheduler() high-level API must produce a valid scheduler."""
        from spectramr.infrastructure.training.scheduler_system import SchedulerFactory

        scheduler = SchedulerFactory.create_scheduler(
            optimizer=dummy_optimizer,
            scheduler_type="cosine_annealing",
            scheduler_params={"T_max": 10},
            warmup_steps=0,
        )
        assert hasattr(scheduler, "step")


# ---------------------------------------------------------------------------
# 2. Ghost-field audit: lr_scheduler_strategy is NOT consumed by the builder
# ---------------------------------------------------------------------------


class TestLrSchedulerStrategyGhostField:
    """``lr_scheduler_strategy`` is no longer a ghost field (issue #533).

    This class used to PIN the gap: the field existed in
    ``OptimizationConfigSchema`` but ``build_schedulers`` read only
    ``optimization.scheduler["type"]``, so declaring the family here silently did
    nothing (146 configs in the corpus do exactly that). Its own docstring said
    the assertion "WILL FAIL when the bug is fixed (update the assertion then)".

    It is fixed: ``resolve_scheduler_spec`` reads the field, and the assertions
    below are inverted to cover the live path instead of the gap.
    """

    def test_lr_scheduler_strategy_field_exists_in_schema(self):
        """The schema declares lr_scheduler_strategy — it's a real YAML key."""
        from spectramr.config.schemas.optimization import OptimizationConfigSchema

        assert "lr_scheduler_strategy" in OptimizationConfigSchema.model_fields, (
            "lr_scheduler_strategy was removed from OptimizationConfigSchema.\n"
            "Update this test and the documentation accordingly."
        )

    def test_lr_scheduler_strategy_selects_the_family(self):
        """The declared family takes effect (the inverted ghost-field assertion).

        Asserted at the seam that decides behaviour — the resolved spec — rather
        than by grepping the builder's source, which said nothing about whether
        the value reached torch.
        """
        from spectramr.config.schemas.optimization import OptimizationConfigSchema
        from spectramr.infrastructure.training.scheduler_resolution import (
            resolve_scheduler_spec,
        )

        cfg = OptimizationConfigSchema(
            lr_scheduler_strategy="step_lr", scheduler={"step_size": 10, "gamma": 0.5}
        )
        spec = resolve_scheduler_spec(cfg, max_iterations=1000)
        assert spec is not None
        assert spec.name == "step_lr"
        assert spec.params == {"step_size": 10, "gamma": 0.5}

    def test_lr_scheduler_strategy_is_not_silently_ignored_when_unknown(self):
        """An unknown family raises rather than degrading to cosine (pitfall #9)."""
        from spectramr.config.schemas.optimization import OptimizationConfigSchema
        from spectramr.domain.exceptions import ConfigurationError
        from spectramr.infrastructure.training.scheduler_resolution import (
            resolve_scheduler_spec,
        )

        cfg = OptimizationConfigSchema(lr_scheduler_strategy="not_a_scheduler", scheduler={})
        with pytest.raises(ConfigurationError, match="not_a_scheduler"):
            resolve_scheduler_spec(cfg, max_iterations=1000)

    def test_lr_scheduler_strategy_default_is_cosine(self):
        """The schema default is 'cosine' — verify it matches what users expect."""
        from spectramr.config.schemas.optimization import OptimizationConfigSchema

        config = OptimizationConfigSchema()
        assert config.lr_scheduler_strategy == "cosine", (
            f"Expected default 'cosine', got '{config.lr_scheduler_strategy}'.\n"
            "If the default changed, update YAML templates accordingly."
        )


# ---------------------------------------------------------------------------
# 3. build_schedulers() works when optimization.scheduler dict is set
# ---------------------------------------------------------------------------


class TestBuildSchedulersWithDict:
    """When optimization.scheduler is set as a dict, the builder must create a scheduler."""

    #: Family -> params that family actually consumes. Passing ``T_max`` to every
    #: family (what this fixture used to do) is precisely the wrong-family knob
    #: the resolver now rejects, so each entry declares its own.
    FAMILY_PARAMS: ClassVar[dict[str, dict[str, float]]] = {
        "cosine_annealing": {"T_max": 10},
        "step_lr": {"step_size": 10, "gamma": 0.5},
        "exponential": {"gamma": 0.9},
    }

    def _make_settings_with_scheduler(self, scheduler_type: str, params: dict):
        from spectramr.config.schemas.data import DataConfigSchema
        from spectramr.config.schemas.logging import LoggingConfigSchema
        from spectramr.config.schemas.metrics import MetricsConfigSchema
        from spectramr.config.schemas.model import ModelConfigSchema
        from spectramr.config.schemas.optimization import OptimizationConfigSchema
        from spectramr.config.settings import TrainingSettings

        return TrainingSettings(
            model=ModelConfigSchema(),
            data=DataConfigSchema(),
            optimization=OptimizationConfigSchema(scheduler={"type": scheduler_type, **params}),
            logging=LoggingConfigSchema(),
            metrics=MetricsConfigSchema(),
        )

    @pytest.mark.parametrize("scheduler_type", ["cosine_annealing", "step_lr", "exponential"])
    def test_scheduler_dict_produces_scheduler_object(self, scheduler_type: str):
        """optimizer.scheduler dict config flows to an actual scheduler object."""
        import torch.nn as nn

        from spectramr.infrastructure.training.builders.optimization_builder import (
            OptimizationBuilder,
        )

        settings = self._make_settings_with_scheduler(
            scheduler_type, self.FAMILY_PARAMS[scheduler_type]
        )
        model = nn.Linear(4, 2)
        builder = OptimizationBuilder(config=settings, models={"generator": model})
        builder.build_optimizers().build_schedulers()
        _, schedulers, _ = builder.build()

        assert schedulers, (
            f"No schedulers created for scheduler_type='{scheduler_type}'.\n"
            "optimization.scheduler dict should produce at least one scheduler."
        )
        assert builder.scheduler_spec.name == scheduler_type

    def test_wrong_family_knob_raises_instead_of_being_dropped(self):
        """``T_max`` on a step_lr schedule is a silent no-op; it must raise.

        This is the generalisation of issue #533: the declared parameter reached
        no factory and nothing said so.
        """
        import torch.nn as nn

        from spectramr.domain.exceptions import ConfigurationError
        from spectramr.infrastructure.training.builders.optimization_builder import (
            OptimizationBuilder,
        )

        settings = self._make_settings_with_scheduler("step_lr", {"T_max": 10})
        builder = OptimizationBuilder(config=settings, models={"generator": nn.Linear(4, 2)})
        builder.build_optimizers()
        with pytest.raises(ConfigurationError, match="T_max"):
            builder.build_schedulers()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
