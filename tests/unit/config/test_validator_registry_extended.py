"""Extended unit tests for spectramr.config.schemas.validator_registry.

PART A — Task III.3: Config-validator unit tests.

Covers four archetypes per the master plan:
  1. canary  — a valid config produces zero error-severity failures.
  2. parametrized — multiple paradigm variants.
  3. edge   — boundary values and optional sub-blocks present/absent.
  4. expected-failure — CLAUDE.md pitfall violations (flat access, frozen
     mutation, illegal enum, cross-field violations).

All tests are marked @pytest.mark.unit.
The existing tests/config/test_validator_registry.py covers the low-level
ValidatorRegistry plumbing; this file targets the *registered default rules*
and the pitfall-#1/#4/#9 failure paths.
"""

from __future__ import annotations

from typing import Any

import pytest

from spectramr.config.schemas.validator_registry import (
    ValidationRule,
    ValidatorRegistry,
    _validate_batch_size_positive,
    _validate_checkpoint_config,
    _validate_dataset_config,
    _validate_device_config,
    _validate_epochs,
    _validate_learning_rate,
    _validate_logging_config,
    _validate_loss_weights_positive,
    _validate_metric_domain_compatibility,
    _validate_model_in_out_channels,
    _validate_num_workers,
    _validate_training_mode_specified,
    get_validator_registry,
)

# ---------------------------------------------------------------------------
# 1. CANARY — minimal valid dict produces zero *error*-severity failures
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCanaryValidConfig:
    """A well-formed config dict should produce no error-severity results."""

    @pytest.fixture
    def valid_v6_dict(self) -> dict[str, Any]:
        """Minimal dict that satisfies every default registered rule."""
        return {
            "data": {"batch_size": 4, "num_workers": 2},
            "optimization": {"learning_rate": 1e-4, "epochs": 100},
            "training": {
                "strategy_class": (
                    "spectramr.infrastructure.training.strategies."
                    "reconstruction.ReconstructionTrainingStrategy"
                )
            },
            "model": {"in_channels": 1, "out_channels": 1},
        }

    def test_no_errors_on_valid_config(self, valid_v6_dict: dict[str, Any]) -> None:
        registry = get_validator_registry()
        issues = registry.validate(valid_v6_dict)
        error_names = [
            name
            for name, _ in issues
            if registry.get(name) and registry.get(name).severity == "error"
        ]
        assert error_names == [], f"Unexpected errors: {error_names}"

    def test_batch_size_edge_one_is_valid(self, valid_v6_dict: dict[str, Any]) -> None:
        """batch_size=1 is the boundary minimum and should be accepted."""
        valid_v6_dict["data"]["batch_size"] = 1
        errors = _validate_batch_size_positive(valid_v6_dict)
        assert errors == []

    def test_learning_rate_very_small_is_valid(
        self, valid_v6_dict: dict[str, Any]
    ) -> None:
        """LR=1e-7 is tiny but positive — should be accepted."""
        valid_v6_dict["optimization"]["learning_rate"] = 1e-7
        errors = _validate_learning_rate(valid_v6_dict)
        assert errors == []


# ---------------------------------------------------------------------------
# 2. PARAMETRIZED — multiple paradigm configs
# ---------------------------------------------------------------------------


PARADIGM_DICTS: list[tuple[str, dict[str, Any]]] = [
    (
        "reconstruction",
        {
            "data": {"batch_size": 8, "num_workers": 4},
            "optimization": {"learning_rate": 1e-4, "epochs": 50},
            "training": {
                "strategy_class": (
                    "spectramr.infrastructure.training.strategies."
                    "reconstruction.ReconstructionTrainingStrategy"
                )
            },
            "model": {"in_channels": 1, "out_channels": 1},
        },
    ),
    (
        "gan",
        {
            "data": {"batch_size": 4, "num_workers": 2},
            "optimization": {"learning_rate": 2e-4, "epochs": 200},
            "training": {
                "strategy_class": (
                    "spectramr.infrastructure.training.strategies.gan.GANTrainingStrategy"
                ),
                "training_mode": "gan",
            },
            "objectives": {"gan": {"lambda_adversarial": 1.0}},
            "model": {"in_channels": 1, "out_channels": 1},
        },
    ),
    (
        "diffusion",
        {
            "data": {"batch_size": 2, "num_workers": 0},
            "optimization": {"learning_rate": 2e-4},
            "training": {
                "strategy_class": (
                    "spectramr.infrastructure.training.strategies.diffusion.DiffusionTrainingStrategy"
                ),
                "max_iterations": 100,
            },
            "model": {"in_channels": 2, "out_channels": 2},
        },
    ),
]


@pytest.mark.unit
@pytest.mark.parametrize(
    "paradigm,config_dict", PARADIGM_DICTS, ids=[p for p, _ in PARADIGM_DICTS]
)
class TestParadigmVariants:
    """All paradigm variants should pass error-severity validation."""

    def test_no_error_severity_failures(
        self, paradigm: str, config_dict: dict[str, Any]
    ) -> None:
        registry = get_validator_registry()
        issues = registry.validate(config_dict)
        error_names = [
            name
            for name, _ in issues
            if registry.get(name) and registry.get(name).severity == "error"
        ]
        assert (
            error_names == []
        ), f"Paradigm '{paradigm}' raised unexpected errors: {error_names}"


# ---------------------------------------------------------------------------
# 3. EDGE — boundary values and optional sub-blocks present / absent
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEdgeCases:
    """Boundary values and optional-block presence / absence."""

    # -- Boundary values ---------------------------------------------------

    def test_lambda_zero_accepted(self) -> None:
        """lambda_l1=0.0 is explicitly allowed (CLAUDE.md pitfall #1 footprint).

        Reads `losses.*` -- the canonical v1.0 block (issue #933). `objectives`
        is a v4.0 spelling the root schema `extra="forbid"`s and 0/647 arms
        declare, so a dict keyed on it would test a question no config asks.
        """
        config = {"losses": {"reconstruction": {"lambda_l1": 0.0}}}
        errors = _validate_loss_weights_positive(config)
        assert errors == [], f"lambda_l1=0.0 should be accepted, got: {errors}"

    def test_batch_size_one_accepted(self) -> None:
        errors = _validate_batch_size_positive({"data": {"batch_size": 1}})
        assert errors == []

    def test_epochs_one_accepted(self) -> None:
        errors = _validate_epochs({"optimization": {"epochs": 1}})
        assert errors == []

    def test_num_workers_zero_accepted(self) -> None:
        errors = _validate_num_workers({"data": {"num_workers": 0}})
        assert errors == []

    def test_patch_size_boundary_16_accepted(self) -> None:
        """patch_size=16 is the minimum allowed by _validate_dataset_config."""
        errors = _validate_dataset_config({"data": {"patch_size": 16}})
        assert errors == []

    def test_patch_size_below_16_warns(self) -> None:
        """patch_size=8 should produce a warning message."""
        errors = _validate_dataset_config({"data": {"patch_size": 8}})
        assert len(errors) > 0

    # -- Optional sub-blocks absent ----------------------------------------

    def test_absent_checkpoint_block_passes(self) -> None:
        """Absent checkpoint block should produce no failures (it's optional)."""
        errors = _validate_checkpoint_config({})
        assert errors == []

    def test_absent_logging_block_passes(self) -> None:
        errors = _validate_logging_config({})
        assert errors == []

    def test_absent_acceleration_block_passes(self) -> None:
        errors = _validate_device_config({})
        assert errors == []

    # -- Optional sub-blocks present (valid values) ------------------------

    def test_checkpoint_block_valid(self) -> None:
        errors = _validate_checkpoint_config(
            {"checkpoint": {"save_interval": 5, "keep_top_k": 3}}
        )
        assert errors == []

    def test_logging_block_valid(self) -> None:
        errors = _validate_logging_config(
            {"logging": {"log_interval": 10, "log_level": "INFO"}}
        )
        assert errors == []

    def test_strategy_class_only_no_training_mode(self) -> None:
        """strategy_class alone (no training_mode) should satisfy training_mode_specified."""
        config = {
            "training": {
                "strategy_class": (
                    "spectramr.infrastructure.training.strategies."
                    "reconstruction.ReconstructionTrainingStrategy"
                )
            }
        }
        errors = _validate_training_mode_specified(config)
        assert errors == []

    def test_training_mode_only_no_strategy_class(self) -> None:
        """training_mode alone should also satisfy the check."""
        errors = _validate_training_mode_specified(
            {"training": {"training_mode": "reconstruction"}}
        )
        assert errors == []


# ---------------------------------------------------------------------------
# 4. EXPECTED-FAILURE — CLAUDE.md pitfall violations
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExpectedFailures:
    """Configurations that MUST raise / produce errors.

    These tests directly encode CLAUDE.md pitfalls #1, #4, #9 as unit tests.
    """

    # Pitfall #1: flat config access (top-level lr / lambda_l1)
    # The validator_registry checks nested paths.  A flat top-level `lr` key
    # means `optimization.optimizer.learning_rate` is absent → learning_rate_positive fires.

    def test_missing_learning_rate_raises_error(self) -> None:
        """CLAUDE.md pitfall #1: flat `lr` instead of `optimization.optimizer.learning_rate`."""
        config_flat = {
            "lr": 1e-4,  # wrong — flat access, not nested
            "optimization": {},  # nested path missing
        }
        errors = _validate_learning_rate(config_flat)
        assert (
            len(errors) > 0
        ), "Flat 'lr' instead of 'optimization.optimizer.learning_rate' must produce an error"

    def test_missing_batch_size_raises_error(self) -> None:
        """CLAUDE.md pitfall #1: `batch_size` not under `data:` block."""
        config_flat = {"batch_size": 4, "data": {}}
        errors = _validate_batch_size_positive(config_flat)
        assert (
            len(errors) > 0
        ), "Flat 'batch_size' (not under data:) must produce an error"

    # Pitfall #4: mutating a frozen TrainingSettings object
    # We test this via the Pydantic schema, not the dict-level validator.

    def test_frozen_settings_raises_on_mutation(self, tmp_path) -> None:
        """CLAUDE.md pitfall #4: TrainingSettings is frozen=True; mutation must fail."""
        import yaml
        from pydantic import ValidationError as PydanticValidationError

        # Build a minimal valid v6.0 YAML via the dummy_basic template
        cfg = {
            "config_version": "1.0",
            "training": {
                "strategy_class": (
                    "spectramr.infrastructure.training.strategies."
                    "reconstruction.ReconstructionTrainingStrategy"
                ),
                "max_iterations": 10,
                "task": "reconstruction",
                "output_dir": "experiments/results/test_freeze",
            },
            "data": {
                "coil_processing_mode": "rss",
                "dataset_type": "synthetic",
                "patch_size": [32, 32, 1],
                "batch_size": 4,
                "num_workers": 0,
            },
            "model": {
                "model_type": "standard_unet",
                "in_channels": 1,
                "out_channels": 1,
            },
            "optimization": {
                "learning_rate": 1e-4,
                "optimizer_type": "adam",
            },
            "logging": {
                "log_interval": 5,
                "level": "info",
            },
            "losses": {
                "output_domain": "image",
                "image_losses": [{"name": "mse", "weight": 1.0, "enabled": True}],
                "kspace_losses": [],
                "complex_losses": [],
            },
        }
        # `seed` and `device` live on `run:` since phase 4b; declaring them under
        # `training:` is now a hard rename error, not a legacy spelling.
        cfg["run"] = {"seed": 42, "device": "cuda"}
        yaml_path = tmp_path / "freeze_test.yaml"
        yaml_path.write_text(yaml.dump(cfg))

        from spectramr.config.settings import TrainingSettings

        settings = TrainingSettings.from_yaml(yaml_path)
        assert settings.run.seed == 42
        # Pydantic v2 frozen model raises ValidationError on attribute set.
        # Assert against a REAL declared field: setting an attribute the model
        # does not have raises for the wrong reason and would keep passing after
        # the field moved again.
        with pytest.raises((TypeError, PydanticValidationError)):
            settings.run.seed = 999  # type: ignore[misc]

    # Pitfall #9: illegal enum string → ValidationError

    def test_illegal_training_mode_reports_error(self) -> None:
        """CLAUDE.md pitfall #9: unknown training_mode must produce an error.

        Ownership moved 2026-07-19. This rule no longer decides *existence* --
        it would have to consult STRATEGY_CLASS_PATHS, and config/ may not import
        infrastructure/ (non-negotiable #5). Existence is now enforced by
        ``infrastructure.validation.config_validation._validate_training_mode_dispatchable``;
        see tests/unit/infrastructure/validation/test_config_validation.py.

        The pitfall-#9 guarantee is unchanged -- an unknown mode still hard-fails
        the load. This test pins that the guarantee is still *somewhere*.
        """
        from spectramr.infrastructure.validation.config_validation import (
            _validate_training_mode_dispatchable,
        )

        config = {
            "training": {"training_mode": "NONEXISTENT_MODE_XYZ"},
            "objectives": {},
        }

        # The mode is rejected by the rule that owns existence.
        assert (
            len(_validate_training_mode_dispatchable(config)) > 0
        ), "An unknown training_mode must still be flagged as an error"

    def test_batch_size_zero_is_invalid(self) -> None:
        """batch_size=0 must fail validation."""
        errors = _validate_batch_size_positive({"data": {"batch_size": 0}})
        assert len(errors) > 0

    def test_batch_size_negative_is_invalid(self) -> None:
        errors = _validate_batch_size_positive({"data": {"batch_size": -4}})
        assert len(errors) > 0

    def test_learning_rate_zero_is_invalid(self) -> None:
        errors = _validate_learning_rate({"optimization": {"learning_rate": 0}})
        assert len(errors) > 0

    def test_learning_rate_negative_is_invalid(self) -> None:
        errors = _validate_learning_rate({"optimization": {"learning_rate": -1e-3}})
        assert len(errors) > 0

    def test_epochs_zero_is_invalid(self) -> None:
        """epochs=0 with no max_iterations must fail."""
        errors = _validate_epochs({"optimization": {"epochs": 0}})
        assert len(errors) > 0

    def test_missing_training_mode_and_strategy_class_is_invalid(self) -> None:
        """Neither training_mode nor strategy_class → must fail."""
        errors = _validate_training_mode_specified({"training": {}})
        assert len(errors) > 0

    def test_invalid_log_level_is_flagged(self) -> None:
        """An unrecognized log_level must produce a warning message."""
        errors = _validate_logging_config({"logging": {"log_level": "VERBOSE_CUSTOM"}})
        assert len(errors) > 0

    def test_invalid_device_is_flagged(self) -> None:
        errors = _validate_device_config({"acceleration": {"device": "tpu_pod"}})
        assert len(errors) > 0

    def test_checkpoint_save_interval_zero_is_invalid(self) -> None:
        errors = _validate_checkpoint_config({"checkpoint": {"save_interval": 0}})
        assert len(errors) > 0


# ---------------------------------------------------------------------------
# 5. Registry-wide contract tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRegistryContracts:
    """Registry-level invariants that must always hold."""

    def test_global_registry_has_expected_categories(self) -> None:
        """The singleton must expose all documented categories.

        "objectives" -> "losses" (issue #933): the only rule that used the
        "objectives" category (`loss_weights`) was repointed to read the
        `losses.*` block its validator_fn actually reads, and the category
        was renamed to match rather than continuing to advertise a v4.0
        block the schema rejects.
        """
        expected = {
            "data",
            "optimization",
            "training",
            "model",
            "losses",
            "physics",
            "metrics",
            "checkpoint",
            "logging",
            "device",
        }
        registry = get_validator_registry()
        actual = set(registry.list_categories())
        missing = expected - actual
        assert not missing, f"Registry is missing categories: {missing}"

    def test_every_registered_rule_has_non_empty_description(self) -> None:
        registry = get_validator_registry()
        for rule in registry.list_validators():
            assert rule.description, f"Rule '{rule.name}' has an empty description"

    def test_every_registered_rule_severity_is_valid(self) -> None:
        registry = get_validator_registry()
        for rule in registry.list_validators():
            assert rule.severity in {
                "error",
                "warning",
            }, f"Rule '{rule.name}' has invalid severity '{rule.severity}'"

    def test_registry_check_rule_returns_true_for_non_applicable_rule(self) -> None:
        """A rule that doesn't apply (wrong training mode) returns True (passing)."""
        registry = ValidatorRegistry()
        rule = ValidationRule(
            name="gan_only",
            category="model",
            description="Only for GAN",
            validator_fn=lambda c: ["always fail"],
            applies_to=["gan"],
        )
        registry.register(rule)
        # Config has training_mode=reconstruction → rule doesn't apply → passes
        config = {"training": {"training_mode": "reconstruction"}}
        assert registry.check_rule(config, "gan_only") is True

    def test_registry_stats_counts_match_registered(self) -> None:
        registry = ValidatorRegistry()
        for i in range(4):
            registry.register(
                ValidationRule(
                    name=f"rule_{i}",
                    category="data" if i < 2 else "opt",
                    description=f"Rule {i}",
                    severity="error" if i % 2 == 0 else "warning",
                    validator_fn=lambda c: [],
                )
            )
        stats = registry.get_stats()
        assert stats["total_validators"] == 4
        assert stats["categories"]["data"] == 2
        assert stats["categories"]["opt"] == 2
        assert stats["severity"]["error"] == 2
        assert stats["severity"]["warning"] == 2


class TestLpipsMetricDomainCompatibility:
    """F7c (smoke audit 2026-06-03): LPIPS routes through
    ``adapt_to_rgb(ChannelMode.AUTO)``, which RSS-collapses EVEN channels to a
    1-channel magnitude (then replicates to 3) — so even-channel output is
    handled and must NOT warn. AUTO only *raises* on odd C != 1, 3, so the
    warning is legitimate only for odd channels > 3.
    """

    @staticmethod
    def _cfg(out_channels: int) -> dict[str, Any]:
        return {
            "validation": {"metrics": ["lpips"]},
            "model": {"out_channels": out_channels, "model_domain": "image"},
            "data": {"patch_size": [256, 256, 1]},
        }

    @pytest.mark.parametrize("ch", [2, 4, 8])
    def test_even_channel_lpips_does_not_warn(self, ch: int) -> None:
        msgs = _validate_metric_domain_compatibility(self._cfg(ch))
        assert not any(
            "lpips" in m for m in msgs
        ), f"even out_channels={ch} is RSS-adapted by LPIPS; it must not warn"

    @pytest.mark.parametrize("ch", [5, 7])
    def test_odd_channel_gt3_lpips_warns(self, ch: int) -> None:
        msgs = _validate_metric_domain_compatibility(self._cfg(ch))
        assert any(
            "lpips" in m for m in msgs
        ), f"odd out_channels={ch} cannot be RSS-adapted (AUTO raises); warn"

    def test_grayscale_single_channel_does_not_warn(self) -> None:
        assert not any(
            "lpips" in m for m in _validate_metric_domain_compatibility(self._cfg(1))
        )

    def test_kspace_model_never_warns(self) -> None:
        cfg = self._cfg(8)
        cfg["model"]["model_domain"] = "kspace"
        assert not any("lpips" in m for m in _validate_metric_domain_compatibility(cfg))


class TestVfAdvancedTrainingModesRegistered:
    """2026-06-04 dispatch (tasks 49/50): exp_vf_ib_infonce_v2 / exp_vf_twin_dps_v2
    passed Tier-0/1 audit but ``main`` rejected them at startup with "Unknown
    training mode: ib_vf" / "twin_dps". The strategies are registered in
    strategy_factory.py (IBVFTrainingStrategy / TwinLikelihoodDPSStrategy) with
    schemas in vf_advanced.py; only the validator allow-list was missing them.
    """

    @pytest.mark.parametrize("mode", ["ib_vf", "twin_dps"])
    def test_mode_in_allow_list(self, mode: str) -> None:
        from spectramr.config.validation_constants import VALID_TRAINING_MODES

        assert mode in VALID_TRAINING_MODES

    def test_allowlist_consistent_with_strategy_factory(self) -> None:
        """Every dispatchable training mode must be in the validator allow-list
        (this is the invariant that the ib_vf/twin_dps gap violated)."""
        from spectramr.config.validation_constants import VALID_TRAINING_MODES
        from spectramr.infrastructure.training.strategy_factory import (
            TrainingStrategyFactory,
        )

        paths = TrainingStrategyFactory.STRATEGY_CLASS_PATHS
        for mode in ("ib_vf", "twin_dps"):
            assert mode in paths, f"{mode} not dispatchable"
            assert (
                mode in VALID_TRAINING_MODES
            ), f"{mode} dispatchable but not allow-listed"


@pytest.mark.unit
class TestAccelerationFactorsNoFlatLrAlias:
    """Regression WS1-core-05: ``_validate_acceleration_factors`` fell back to the
    flat ``optimization.lr`` alias when ``optimization.optimizer.learning_rate`` was absent.
    The flat alias was removed in v6.0 (NN#1) and is rejected by extra="forbid"
    before this validator runs, so the fallback was dead code that quietly
    "accepted" a forbidden key shape. Removing it must not surface a spurious
    warning for a config that only carries the (already-impossible) flat alias.
    """

    def test_flat_lr_alias_not_read(self) -> None:
        from spectramr.config.schemas.validator_registry import (
            _validate_acceleration_factors,
        )

        cfg = {"acceleration": {"amp_enabled": True}, "optimization": {"lr": 0.5}}
        msgs = _validate_acceleration_factors(cfg)
        # No fallback → 'lr' is never read → no AMP-stability warning fires.
        assert msgs == []

    def test_canonical_lr_still_warns_when_high(self) -> None:
        from spectramr.config.schemas.validator_registry import (
            _validate_acceleration_factors,
        )

        cfg = {
            "acceleration": {"amp_enabled": True},
            "optimization": {"learning_rate": 0.5},
        }
        msgs = _validate_acceleration_factors(cfg)
        assert any("AMP" in m for m in msgs)


class TestValidatorRegistryDuplicateRegistration:
    """WS-1 round-2: ``register`` honours its docstring — a duplicate name now
    RAISES instead of silently overwriting (pitfall #9), with an ``override``
    escape hatch."""

    def _rule(self, name: str):
        from spectramr.config.schemas.validator_registry import ValidationRule

        return ValidationRule(
            name=name,
            validator_fn=lambda cfg: [],
            category="test",
            description="test rule",
        )

    def test_duplicate_registration_raises(self) -> None:
        import pytest

        from spectramr.config.schemas.validator_registry import ValidatorRegistry

        reg = ValidatorRegistry()
        reg.register(self._rule("dup_rule"))
        with pytest.raises(ValueError, match="already registered"):
            reg.register(self._rule("dup_rule"))

    def test_override_allows_replacement(self) -> None:
        from spectramr.config.schemas.validator_registry import ValidatorRegistry

        reg = ValidatorRegistry()
        reg.register(self._rule("dup_rule"))
        replacement = self._rule("dup_rule")
        reg.register(replacement, override=True)  # must not raise
        assert reg.get("dup_rule") is replacement

@pytest.mark.parametrize(
    "cfg",
    [
        {"validation": None},
        {"validation": {"metrics": ["fid", "lpips"]}, "data": None, "model": None},
    ],
)
def test_metric_domain_compatibility_tolerates_null_blocks(cfg: dict[str, Any]) -> None:
    assert isinstance(_validate_metric_domain_compatibility(cfg), list)
