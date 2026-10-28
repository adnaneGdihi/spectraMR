"""Unit tests for CollationStrategySelector logic.

Tests the centralized collation strategy selection logic in isolation.
Verifies auto-selection, user overrides, parameter derivation, and validation.

Author: spectraMR Team
Date: 2026-02-15
"""

import pydantic
import pytest
import torch

from spectramr.config.schemas.collation import CollationConfigSchema
from spectramr.data.collation.strategies import ImageCollateStrategy
from spectramr.data.collation.strategy_selector import CollationStrategySelector


class TestCollationStrategyAutoSelection:
    """Test automatic strategy selection when user doesn't specify."""

    def test_auto_select_image_for_kspace(self):
        """AUTO: dataset_type='kspace' → strategy='image'"""
        config = CollationConfigSchema(strategy=None)  # User didn't specify

        strategy_name, kwargs = CollationStrategySelector.select_strategy(
            config=config,
            dataset_type="kspace",
            enable_slab_mode=False,
        )

        assert strategy_name == "image"
        assert "padding_mode" in kwargs
        assert kwargs["padding_mode"] == "zeros"  # Default

    def test_auto_select_image_for_various_types(self):
        """AUTO: Multiple dataset types should auto-select 'image'"""
        config = CollationConfigSchema(strategy=None)

        for dataset_type in ["kspace", "nifti", "fastmri", "image", "m4raw"]:
            strategy_name, kwargs = CollationStrategySelector.select_strategy(
                config=config,
                dataset_type=dataset_type,
                enable_slab_mode=False,
            )

            assert strategy_name == "image", f"Failed for dataset_type={dataset_type}"

    def test_auto_select_slab_when_enabled(self):
        """AUTO: enable_slab_mode=True → strategy='slab'"""
        config = CollationConfigSchema(strategy=None)

        strategy_name, kwargs = CollationStrategySelector.select_strategy(
            config=config,
            dataset_type="nifti",
            enable_slab_mode=True,  # Slab mode enabled
            patch_size=(256, 256, 1),
        )

        assert strategy_name == "slab"
        assert kwargs.get("flat_input") is True
        assert kwargs.get("target_mode") == "middle"

    def test_auto_select_robust_for_unknown_type(self):
        """AUTO: Unknown dataset_type → fallback to 'robust'"""
        config = CollationConfigSchema(strategy=None)

        strategy_name, kwargs = CollationStrategySelector.select_strategy(
            config=config,
            dataset_type="unknown_type_12345",
            enable_slab_mode=False,
        )

        assert strategy_name == "robust"

    def test_auto_select_graph_for_graph_type(self):
        """AUTO: dataset_type='graph' → strategy='graph'"""
        config = CollationConfigSchema(strategy=None)

        strategy_name, kwargs = CollationStrategySelector.select_strategy(
            config=config,
            dataset_type="graph",
            enable_slab_mode=False,
        )

        assert strategy_name == "graph"


class TestUserSpecifiedStrategy:
    """Test that user-specified strategy overrides auto-selection."""

    def test_user_specified_overrides_auto(self):
        """USER CHOICE: User specifies strategy → use it, not auto-selection"""
        config = CollationConfigSchema(strategy="robust")  # User specified

        strategy_name, kwargs = CollationStrategySelector.select_strategy(
            config=config,
            dataset_type="nifti",  # Would normally auto-select "image"
            enable_slab_mode=False,
        )

        assert strategy_name == "robust"  # User's choice, not auto-selected

    def test_user_specified_overrides_slab_mode(self):
        """USER CHOICE: User strategy overrides enable_slab_mode"""
        config = CollationConfigSchema(strategy="image")  # User wants image

        strategy_name, kwargs = CollationStrategySelector.select_strategy(
            config=config,
            dataset_type="nifti",
            enable_slab_mode=True,  # Would auto-select "slab", but user specified "image"
        )

        assert strategy_name == "image"  # User's explicit choice wins


class TestParameterDerivation:
    """Test automatic parameter derivation (squeeze_depth from patch_size)."""

    def test_squeeze_depth_auto_derived_from_patch_size_2d(self):
        """squeeze_depth_dim auto-derived when config=None and patch_size is 2D"""
        config = CollationConfigSchema(strategy="image", squeeze_depth_dim=None)

        # 2D patch: patch_size[-1] == 1
        strategy_name, kwargs = CollationStrategySelector.select_strategy(
            config=config,
            dataset_type="nifti",
            enable_slab_mode=False,
            patch_size=(256, 256, 1),  # 2D: D=1
        )

        assert kwargs["squeeze_depth_dim"] is True  # Auto-derived from patch_size

    def test_squeeze_depth_auto_derived_from_patch_size_3d(self):
        """squeeze_depth_dim=False when patch_size is 3D"""
        config = CollationConfigSchema(strategy="image", squeeze_depth_dim=None)

        # 3D patch: patch_size[-1] > 1
        strategy_name, kwargs = CollationStrategySelector.select_strategy(
            config=config,
            dataset_type="nifti",
            enable_slab_mode=False,
            patch_size=(256, 256, 16),  # 3D: D=16
        )

        assert kwargs["squeeze_depth_dim"] is False  # Auto-derived: no squeeze for 3D

    def test_squeeze_depth_user_specified(self):
        """squeeze_depth_dim respects user specification"""
        config = CollationConfigSchema(
            strategy="image", squeeze_depth_dim=False
        )  # User explicit

        strategy_name, kwargs = CollationStrategySelector.select_strategy(
            config=config,
            dataset_type="nifti",
            enable_slab_mode=False,
            patch_size=(256, 256, 1),  # Would be True if auto-derived
        )

        assert kwargs["squeeze_depth_dim"] is False  # User's explicit choice

    def test_squeeze_depth_defaults_false_without_patch_size(self):
        """squeeze_depth_dim=False when patch_size is None"""
        config = CollationConfigSchema(strategy="image", squeeze_depth_dim=None)

        strategy_name, kwargs = CollationStrategySelector.select_strategy(
            config=config,
            dataset_type="image",
            enable_slab_mode=False,
            patch_size=None,  # No patch size
        )

        assert kwargs["squeeze_depth_dim"] is False  # Default to False


class TestStrategyParameters:
    """Test that all config parameters are passed to kwargs."""

    def test_padding_parameters_passed_through(self):
        """All padding parameters should be in kwargs for strategy instantiation"""
        config = CollationConfigSchema(
            strategy="image",
            padding_mode="reflect",
            padding_value=99.0,
            allow_variable_shapes=True,
        )

        strategy_name, kwargs = CollationStrategySelector.select_strategy(
            config=config,
            dataset_type="image",
        )

        assert kwargs["padding_mode"] == "reflect"
        assert kwargs["padding_value"] == 99.0
        assert kwargs["allow_variable_shapes"] is True

    def test_default_padding_parameters(self):
        """Default padding parameters should be in kwargs"""
        config = CollationConfigSchema(strategy="image")

        strategy_name, kwargs = CollationStrategySelector.select_strategy(
            config=config,
            dataset_type="image",
        )

        assert kwargs["padding_mode"] == "zeros"  # Default
        assert kwargs["padding_value"] == 0.0  # Default
        assert kwargs["allow_variable_shapes"] is False  # Default

    def test_slab_specific_parameters(self):
        """Slab strategy should get target_mode and flat_input"""
        config = CollationConfigSchema(strategy="slab")

        strategy_name, kwargs = CollationStrategySelector.select_strategy(
            config=config,
            dataset_type="nifti",
            enable_slab_mode=True,
        )

        assert strategy_name == "slab"
        assert kwargs.get("target_mode") == "middle"
        assert kwargs.get("flat_input") is True


class TestCompatibilityValidation:
    """Test validation warnings for incompatible strategy/dataset combinations."""

    def test_incompatible_strategy_raises(self):
        """An incompatible pair must RAISE, not warn-and-continue.

        This asserted "warn but allow" and returned ``"graph"`` for a
        ``kspace`` dataset. A warning the run proceeds past is the
        passed-with-warnings outcome pitfall #10 names as the most dangerous
        one: the collate would then fail deep in the loader, or silently
        mis-batch. The selector now refuses at selection time and names the
        compatible set.
        """
        config = CollationConfigSchema(strategy="graph")

        with pytest.raises(ValueError, match="not compatible with dataset_type"):
            CollationStrategySelector.select_strategy(
                config=config,
                dataset_type="kspace",
            )

    def test_compatible_strategy_no_warning(self, caplog):
        """No warning when strategy is compatible"""
        import logging

        with caplog.at_level(logging.WARNING):
            config = CollationConfigSchema(strategy="image")

            strategy_name, kwargs = CollationStrategySelector.select_strategy(
                config=config,
                dataset_type="kspace",  # Compatible
            )

        assert strategy_name == "image"
        # Should NOT have warnings
        assert "may not be compatible" not in caplog.text.lower()

    def test_universal_is_rejected_at_config_validation(self):
        """``universal`` was advertised, validated, and unbuildable.

        ``CollateStrategyFactory`` has no such class and ``COMPATIBILITY_MAP``
        dropped it (audit A8) -- but the schema ``Literal``, the vocabulary a
        YAML is actually checked against, kept accepting it. So a config could
        pass validation and die in construction. Zero arms declared it, which is
        why it stayed latent.

        Rejected at the earliest point now, rather than "always compatible".
        """
        with pytest.raises(pydantic.ValidationError):
            CollationConfigSchema(strategy="universal")

    def test_all_three_strategy_vocabularies_agree(self):
        """Schema Literal, COMPATIBILITY_MAP and the factory name one set.

        ``_assert_vocabularies_agree`` compared only the last two, so the schema
        could drift alone -- and did. Pinned here as a set equality, which is
        stronger than the guard's one-directional subset check and fails
        whichever of the three moves first.
        """
        import typing

        from spectramr.data.collation.strategies import CollateStrategyFactory

        literal = {
            a
            for a in typing.get_args(
                typing.get_args(
                    CollationConfigSchema.model_fields["strategy"].annotation
                )[0]
            )
            if isinstance(a, str)
        }
        buildable = set(CollateStrategyFactory.STRATEGIES)
        assert literal == buildable
        assert set(CollationStrategySelector.COMPATIBILITY_MAP) == buildable


class TestInvalidConfigurations:
    """Test error handling for invalid configurations."""

    def test_invalid_strategy_raises_error(self):
        """Unknown strategy should raise ValidationError during config validation"""
        from pydantic import ValidationError

        # Pydantic raises ValidationError for invalid Literal values
        with pytest.raises(ValidationError, match="Input should be"):
            config = CollationConfigSchema(strategy="invalid_strategy_123")

    def test_invalid_padding_mode_raises_error(self):
        """Invalid padding_mode should raise ValidationError during config validation"""
        from pydantic import ValidationError

        # Pydantic raises ValidationError for invalid Literal values
        with pytest.raises(ValidationError, match="Input should be"):
            config = CollationConfigSchema(
                strategy="image",
                padding_mode="invalid_mode",  # type: ignore
            )

    def test_nan_padding_value_raises_error(self):
        """NaN padding_value should raise ValueError"""
        with pytest.raises(ValueError, match="must be finite"):
            config = CollationConfigSchema(strategy="image", padding_value=float("nan"))

    def test_inf_padding_value_raises_error(self):
        """Inf padding_value should raise ValueError"""
        with pytest.raises(ValueError, match="must be finite"):
            config = CollationConfigSchema(strategy="image", padding_value=float("inf"))


class TestLoggingBehavior:
    """Test logging configuration and behavior."""

    def test_logging_enabled_by_default(self, caplog):
        """Strategy selection should be logged by default"""
        import logging

        with caplog.at_level(logging.INFO):
            config = CollationConfigSchema(strategy="image")

            CollationStrategySelector.select_strategy(
                config=config,
                dataset_type="kspace",
            )

        # Should have INFO logs
        assert "user-specified" in caplog.text.lower()
        assert "final selection" in caplog.text.lower()

    def test_logging_disabled_when_configured(self, caplog):
        """Logging can be disabled via config"""
        import logging

        with caplog.at_level(logging.INFO):
            config = CollationConfigSchema(
                strategy="image", log_strategy_selection=False
            )

            CollationStrategySelector.select_strategy(
                config=config,
                dataset_type="kspace",
            )

        # Should have minimal or no logs when disabled
        # (The selector respects config.log_strategy_selection)
        # Note: Currently the selector always logs, so this test documents expected behavior
        # TODO: Update selector to respect log_strategy_selection flag


class TestValidateNansKnob:
    """CWS-001: the ``collation.validate_nans`` YAML knob must be wired through
    the selector into ImageCollateStrategy (pitfall #15 — no unread knobs)."""

    def test_selector_forwards_validate_nans_true(self):
        """Default config (validate_nans=True) is forwarded into kwargs."""
        config = CollationConfigSchema(strategy="image", validate_nans=True)
        _, kwargs = CollationStrategySelector.select_strategy(
            config=config,
            dataset_type="kspace",
        )
        assert kwargs["validate_nans"] is True

    def test_selector_forwards_validate_nans_false(self):
        """validate_nans=False is forwarded (not silently dropped)."""
        config = CollationConfigSchema(strategy="image", validate_nans=False)
        _, kwargs = CollationStrategySelector.select_strategy(
            config=config,
            dataset_type="kspace",
        )
        assert kwargs["validate_nans"] is False

    def test_validate_nans_false_passes_through(self):
        """validate_nans=False must NOT replace NaNs with zeros."""
        strategy = ImageCollateStrategy(validate_nans=False)
        batch = [{"x": torch.tensor([float("nan")])}]
        result = strategy.collate(batch)
        assert torch.isnan(
            result["x"]
        ).any(), "validate_nans=False must not replace NaNs"

    def test_validate_nans_true_raises_on_nan(self):
        """validate_nans=True (default) raises; it no longer repairs to zeros.

        Updated for finding M3: the guard used to log at ERROR and then
        ``nan_to_num(value, nan=0.0)`` and return the batch, so training carried
        on over fabricated background. Full contract in
        ``tests/unit/data/collation/test_strategies.py``.
        """
        strategy = ImageCollateStrategy(validate_nans=True)
        batch = [{"x": torch.tensor([float("nan")])}]
        with pytest.raises(ValueError, match="non-finite"):
            strategy.collate(batch)


class TestSelectedStrategiesCanBeBuilt:
    """What the selector hands the factory must be what the class accepts.

    The selector tests stopped at the kwargs, and the strategy tests built the classes
    by hand, so the pairing in between was never run: every slab arm raised
    ``got multiple values for keyword argument 'allow_variable_shapes'`` at loader build.
    """

    @pytest.mark.parametrize(
        ("slab_mode", "expected"), [(False, "ImageCollateStrategy"), (True, "SlabCollateStrategy")]
    )
    def test_the_strategy_selected_for_a_2d_patch_is_constructible(self, slab_mode, expected):
        from spectramr.data.collation.strategies import CollateStrategyFactory

        name, kwargs = CollationStrategySelector.select_strategy(
            config=CollationConfigSchema(),
            dataset_type="m4raw",
            enable_slab_mode=slab_mode,
            patch_size=(256, 256, 1),
        )
        assert type(CollateStrategyFactory.create(name, **kwargs)).__name__ == expected

    def test_slab_refuses_variable_shapes_it_does_not_implement(self):
        config = CollationConfigSchema(strategy="slab", allow_variable_shapes=True)
        with pytest.raises(ValueError, match="allow_variable_shapes"):
            CollationStrategySelector.select_strategy(
                config=config, dataset_type="nifti", patch_size=(64, 64, 5)
            )

    def test_slab_refuses_a_depth_squeeze_it_does_not_implement(self):
        config = CollationConfigSchema(strategy="slab", squeeze_depth_dim=True)
        with pytest.raises(ValueError, match="squeeze_depth_dim"):
            CollationStrategySelector.select_strategy(
                config=config, dataset_type="nifti", patch_size=(64, 64, 5)
            )
