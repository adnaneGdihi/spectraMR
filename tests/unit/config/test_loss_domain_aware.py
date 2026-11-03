"""Unit tests for domain-aware list-based loss configuration.

Validates:
1. Schema parsing with new fields (output_domain, kspace/image/complex_losses)
2. Backward compatibility (old YAMLs without lists still work)
3. Validator: lists require output_domain
4. get_enabled_losses() includes list-based losses
"""

import pytest
from pydantic import ValidationError

from spectramr.config.schemas.loss import (
    LossComponentConfig,
    LossConfigSchema,
    PhysicsLossesConfig,
    ReconstructionLossesConfig,
)
from spectramr.models.losses.build_set import declared_build_set
from spectramr.models.losses.weights import build_loss_weight_table


def _declared(losses):
    """Weights the author declared (the table owns this, #1966); the build set is narrower."""
    return {
        k: s.weight
        for k, s in build_loss_weight_table(losses).items()
        if s.enabled and s.weight > 0
    }



class TestLossConfigSchemaListBased:
    """Tests for the new domain-aware list-based loss fields."""

    def test_default_schema_has_empty_lists(self):
        """Default schema should have empty loss lists and no output_domain."""
        schema = LossConfigSchema()
        assert schema.policy.output_domain == ""
        assert schema.kspace_losses == []
        assert schema.image_losses == []
        assert schema.complex_losses == []
        assert not schema.uses_list_based_losses

    def test_backward_compat_old_config_still_works(self):
        """Old-style config without list-based losses must parse and work."""
        schema = LossConfigSchema(
            reconstruction=ReconstructionLossesConfig(
                enable_l1=True,
                lambda_l1=10.0,
                enable_background_suppression=True,
                lambda_background_suppression=0.1,
                background_suppression_use_fourier_bridge=True,
            )
        )
        assert not schema.uses_list_based_losses
        enabled = _declared(schema)
        assert "l1" in enabled
        assert enabled["l1"] == 10.0
        assert "background_suppression" in enabled

    def test_list_based_config_parses(self):
        """List-based config with output_domain parses correctly."""
        schema = LossConfigSchema(
            output_domain="kspace",
            kspace_losses=[
                LossComponentConfig(name="sobolev_kspace", weight=1.0),
                LossComponentConfig(name="log_spectral", weight=0.08, kwargs={"skip_fft": False}),
            ],
            image_losses=[
                LossComponentConfig(name="background_suppression", weight=0.1, kwargs={"threshold_ratio": 1.5}),
            ],
            complex_losses=[
                LossComponentConfig(name="complex_spatial_gradient", weight=0.2),
            ],
        )
        assert schema.uses_list_based_losses
        assert schema.policy.output_domain == "kspace"
        assert len(schema.kspace_losses) == 2
        assert len(schema.image_losses) == 1
        assert len(schema.complex_losses) == 1

    def test_list_based_losses_appear_in_get_enabled_losses(self):
        """List-based losses must appear in get_enabled_losses()."""
        schema = LossConfigSchema(
            output_domain="kspace",
            kspace_losses=[
                LossComponentConfig(name="sobolev_kspace", weight=1.0),
            ],
            image_losses=[
                LossComponentConfig(name="l1", weight=5.0),
            ],
        )
        enabled = _declared(schema)
        assert "sobolev_kspace" in enabled
        assert enabled["sobolev_kspace"] == 1.0
        assert "l1" in enabled
        assert enabled["l1"] == 5.0

    def test_disabled_list_losses_excluded(self):
        """Disabled list-based losses (enabled=False or weight=0) must not appear."""
        schema = LossConfigSchema(
            output_domain="kspace",
            reconstruction=None,  # Disable default recon to isolate list-based behavior
            kspace_losses=[
                LossComponentConfig(name="sobolev_kspace", weight=1.0, enabled=True),
                LossComponentConfig(name="log_spectral", weight=0.0, enabled=True),
            ],
            image_losses=[
                LossComponentConfig(name="l1", weight=5.0, enabled=False),
            ],
        )
        enabled = _declared(schema)
        assert "sobolev_kspace" in enabled
        assert "log_spectral" not in enabled  # weight=0
        assert "l1" not in enabled  # enabled=False

    def test_validator_rejects_lists_without_output_domain(self):
        """Must reject list-based losses when output_domain is empty."""
        with pytest.raises(ValidationError, match="output_domain must be set"):
            LossConfigSchema(
                kspace_losses=[
                    LossComponentConfig(name="sobolev_kspace", weight=1.0),
                ],
            )

    def test_validator_rejects_invalid_output_domain(self):
        """Must reject unknown output_domain values."""
        with pytest.raises(ValidationError, match="invalid"):
            LossConfigSchema(
                output_domain="frequency",
                kspace_losses=[
                    LossComponentConfig(name="sobolev_kspace", weight=1.0),
                ],
            )

    def test_output_domain_alone_without_lists_is_valid(self):
        """Setting output_domain without lists is fine (no-op for backward compat)."""
        schema = LossConfigSchema(output_domain="kspace")
        assert not schema.uses_list_based_losses
        # No validator error — output_domain alone is allowed

    def test_mixed_list_and_paradigm_losses(self):
        """List-based losses can coexist with paradigm-specific sections."""
        schema = LossConfigSchema(
            output_domain="kspace",
            kspace_losses=[
                LossComponentConfig(name="sobolev_kspace", weight=1.0),
            ],
            physics=PhysicsLossesConfig(
                enable_physics_constraint=True,
                lambda_physics_constraint=0.1,
            ),
        )
        enabled = _declared(schema)
        assert "sobolev_kspace" in enabled
        assert "physics_constraint" in enabled

    def test_disable_default_losses_applies_to_lists(self):
        """disable_default_losses should also filter list-based losses."""
        schema = LossConfigSchema(
            output_domain="kspace",
            kspace_losses=[
                LossComponentConfig(name="sobolev_kspace", weight=1.0),
            ],
            image_losses=[
                LossComponentConfig(name="l1", weight=5.0),
            ],
            policy={"exclude_defaults": ["l1"]},
        )
        enabled = declared_build_set(schema)

        assert "sobolev_kspace" in enabled
        assert "l1" not in enabled  # disabled via disable_default_losses

    def test_kwargs_preserved_in_components(self):
        """Component kwargs must be preserved and accessible."""
        comp = LossComponentConfig(
            name="background_suppression",
            weight=0.1,
            kwargs={"threshold_ratio": 1.5, "use_fourier_bridge": False},
        )
        assert comp.kwargs["threshold_ratio"] == 1.5
        assert comp.kwargs["use_fourier_bridge"] is False
"""Tests for domain-aware list-based loss configuration."""
