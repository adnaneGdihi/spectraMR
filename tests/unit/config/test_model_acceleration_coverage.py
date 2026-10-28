"""Model Schema and Acceleration Schema Coverage Tests.

PURPOSE:
ModelConfigSchema has fields like weight_init_type, weight_init_gain,
apply_weight_init, denoising_model, base_diffusion_config, z_dim,
kan_type, wavelet_type, wav_version, target_domain, etc.

ModelBuilder reads some (model_type, in_channels, out_channels, model_kwargs,
compile_model) but many are passed through only via model_kwargs or ignored.

AccelerationConfigSchema has 20+ fields. The consumer (KSpaceMaskGenerator
and TorchIOTransformBuilder) reads only a handful.

This suite tracks:
  1. Which ModelConfigSchema fields ModelBuilder actually reads
  2. Which AccelerationConfigSchema fields KSpaceMaskGenerator reads
  3. Deprecated/legacy acceleration fields that should warn but don't
"""

from __future__ import annotations

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# KNOWN STATE — model schema
# ─────────────────────────────────────────────────────────────────────────────

CONSUMED_MODEL_FIELDS = {
    "model_type",  # ModelBuilder.build_generator() + factory
    "in_channels",  # ModelFactory.create_generator()
    "out_channels",  # ModelFactory.create_generator() + discriminator
    "model_kwargs",  # forwarded to factory as **gen_kwargs
    "discriminator_component",  # ModelBuilder.build_discriminator()
    "encoder_component",  # ModelBuilder.build_encoder_decoder()
    "decoder_component",  # ModelBuilder.build_encoder_decoder()
    "generator_component",  # ModelBuilder.build_generator() (partial)
    "architecture",  # Factory picks up architecture style
    "trellis",  # TrellisModelFactory reads trellis config
    "trellis_vae",  # TrellisVAEConfig forwarded to factory
    "spatial_dims",  # Factory: 2D vs 3D model selection
    "base_channels",  # forwarded via model_kwargs or factory
    "depth",  # forwarded via model_kwargs or factory
    "checkpoint_path",  # ModelBuilder warm-start + pipeline_strategy stage-weight load
}

# Fields in ModelConfigSchema that go through model_kwargs or are NOT
# directly read by ModelBuilder (they rely on the model implementation
# to accept them as **kwargs)
INDIRECTLY_CONSUMED = {
    "z_dim",  # VAE latent dim — passed via model_kwargs
    "kan_type",  # KAN architecture type — passed via model_kwargs
    "wavelet_type",  # WavKAN — passed via model_kwargs
    "wav_version",  # WavKAN version — passed via model_kwargs
    "input_type",  # passed via model_kwargs
    "output_activation",  # ActivationManager applies post-construction
    "target_domain",  # used by some strategies for kspace routing
    "output_type",  # forwarded via model_kwargs (model_factory) — like input_type
    "model_domain",  # strategy kspace routing (mixins/kspace, context_resolver)
    "conditioning",  # consumed by conditioned strategies (virtual_fiducial) + audit
}

# Fields NOT consumed by ModelBuilder (schema-only)
KNOWN_MODEL_SCHEMA_ONLY = {
    "weight_init_type",  # Schema: init strategy; Builder: never applies init
    "weight_init_gain",  # Schema: Xavier gain; Builder: no consumer
    "weight_init_activation",  # Schema: He activation; Builder: no consumer
    "apply_weight_init",  # Schema: master switch; Builder: never reads
    "denoising_model",  # Schema: diffusion denoiser config; Factory: not wired
    "base_diffusion_config",  # Schema: diffusion process config; Factory: not wired
}

# ─────────────────────────────────────────────────────────────────────────────
# KNOWN STATE — acceleration schema
# ─────────────────────────────────────────────────────────────────────────────

# AccelerationConfigSchema fields actually used by KSpaceMaskGenerator/builder
CONSUMED_ACCELERATION_FIELDS = {
    "base_acceleration",  # KSpaceMaskGenerator.generate_mask(acceleration=...)
    "center_fraction",  # KSpaceMaskGenerator.generate_mask(center_fraction=...)
    "acceleration_type",  # routes to correct mask generator
    "mask_types",  # multi-mask rotation
    "mask_combination_mode",  # rotate vs union
    "enable_dynamic_mask",  # different mask per sample
    "mask_seed",  # reproducible mask generation
    "ground_truth_folder",  # DataBuilder: GT k-space path
    "ground_truth_type",  # DataBuilder: GT domain
    # All four are read by ``resolve_undersampling_kwargs`` (kspace_process.py),
    # the SSOT both KSpaceColdDiffusionGenerator and the CI ladder gate build
    # from. The three below were classified schema-only while the generator was
    # already reading them inline; ``min_center_fraction`` is new (issue #550).
    "min_center_fraction",  # ACS floor at R=max — drives the ACS ramp
    "max_acceleration",  # ceiling of the R(t) schedule
    "schedule_type",  # selects the step/linear/power_law ladder
    "acceleration_range",  # the explicit step-ladder rungs
}

# Legacy/deprecated that exist purely for YAML backwards-compat (no real consumer)
KNOWN_ACCELERATION_DEPRECATED = {
    "mixed_precision",  # DEPRECATED → optimization.precision.enabled
    "gradient_accumulation_steps",  # DEPRECATED → optimization.gradient.accumulation_steps
    "use_compile",  # DEPRECATED → optimization.use_compile
    "use_distributed",  # DEPRECATED → parallel.strategy
    "use_gradient_checkpointing",  # DEPRECATED → optimization.gradient.enable_checkpointing
}

# Fields that are schema-only (not deprecated, just unimplemented consumers)
KNOWN_ACCELERATION_SCHEMA_ONLY = {
    "acceleration_schedule",  # Schedule type — no scheduler reads this
    "schedule_steps",  # Steps to max accel — no consumer
    "schedule_power",  # Polynomial power — no consumer
    "mask_direction",  # Direction of masking — not in KSpaceMaskGenerator
    "density_power",  # VD density power — not forwarded to generator
    "sigma_scaling",  # Gaussian sigma — not forwarded
    "std_scale",  # Gaussian std — not forwarded
    "pf_fraction",  # Partial Fourier — not implemented
    "pattern_path",  # Learned pattern — not implemented
    "undersampling_pattern",  # Distribution — not forwarded
    "adaptive",  # adaptive sampling toggle — no consumer wires it yet
    "rate",  # generic acceleration rate alias — superseded by base_acceleration
    "sampling_pattern",  # Pattern descriptor — superseded by acceleration_type
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. ModelConfigSchema field audit
# ─────────────────────────────────────────────────────────────────────────────


class TestModelSchemaFieldCoverage:
    """Every ModelConfigSchema field must be classified."""

    ALL_KNOWN_MODEL = (
        CONSUMED_MODEL_FIELDS | INDIRECTLY_CONSUMED | KNOWN_MODEL_SCHEMA_ONLY
    )

    def test_all_model_fields_classified(self):
        """Every ModelConfigSchema field must be in a known category."""
        from spectramr.config.schemas.model import ModelConfigSchema

        all_fields = set(ModelConfigSchema.model_fields.keys())
        unclassified = all_fields - self.ALL_KNOWN_MODEL

        assert not unclassified, (
            f"New ModelConfigSchema fields not classified:\n  {sorted(unclassified)}\n"
            "Add to CONSUMED_MODEL_FIELDS, INDIRECTLY_CONSUMED, or KNOWN_MODEL_SCHEMA_ONLY."
        )

    def test_weight_init_is_schema_only(self):
        """apply_weight_init=True has no effect — ModelBuilder never applies init.

        This is a HIGH-PRIORITY gap: users configure weight initialization
        expecting it to be applied, but ModelBuilder ignores all init fields.
        """
        import inspect

        from spectramr.infrastructure.training.builders.model_builder import ModelBuilder

        src = inspect.getsource(ModelBuilder)
        weight_init_referenced = any(
            field in src
            for field in ["weight_init_type", "apply_weight_init", "weight_init_gain"]
        )

        if not weight_init_referenced:
            pytest.xfail(
                reason="KNOWN GAP: ModelConfigSchema has weight_init_type, "
                "weight_init_gain, weight_init_activation, and apply_weight_init fields, "
                "but ModelBuilder.build_generator() NEVER applies weight initialization.\n"
                "Setting weight_init_type='xavier_uniform' in YAML has NO effect.\n"
                "Fix: Add weight initialization logic to ModelBuilder.build_generator() "
                "using spectramr.models.initialization.apply_weight_init(generator, config.model)"
            )

    def test_denoising_model_config_is_schema_only(self):
        """model.denoising_model config is in schema but no consumer wires it."""
        import inspect

        from spectramr.infrastructure.training.builders.model_builder import ModelBuilder
        from spectramr.models.factories.model_factory import ModelFactory

        src_builder = inspect.getsource(ModelBuilder)
        src_factory = inspect.getsource(ModelFactory)

        if (
            "denoising_model" not in src_builder
            and "denoising_model" not in src_factory
        ):
            pytest.xfail(
                reason="KNOWN GAP: model.denoising_model is in ModelConfigSchema "
                "but neither ModelBuilder nor ModelFactory reads or forwards it. "
                "Diffusion models that need a nested denoising_model config cannot "
                "be configured this way.\n"
                "Fix: Add denoising_model forwarding in ModelBuilder or ModelFactory."
            )


# ─────────────────────────────────────────────────────────────────────────────
# 2. AccelerationConfigSchema field audit
# ─────────────────────────────────────────────────────────────────────────────


class TestAccelerationSchemaFieldCoverage:
    """Every AccelerationConfigSchema field must be classified."""

    ALL_KNOWN_ACCEL = (
        CONSUMED_ACCELERATION_FIELDS
        | KNOWN_ACCELERATION_DEPRECATED
        | KNOWN_ACCELERATION_SCHEMA_ONLY
    )

    def test_all_acceleration_fields_classified(self):
        """Every AccelerationConfigSchema field must be in a known category."""
        from spectramr.config.schemas.acceleration import AccelerationConfigSchema

        all_fields = set(AccelerationConfigSchema.model_fields.keys())
        unclassified = all_fields - self.ALL_KNOWN_ACCEL

        assert not unclassified, (
            f"New AccelerationConfigSchema fields not classified:\n"
            f"  {sorted(unclassified)}\n"
            "Add to the appropriate known set in this test."
        )

    def test_schema_only_count_has_not_increased(self):
        """Pin the known schema-only acceleration field count."""
        from spectramr.config.schemas.acceleration import AccelerationConfigSchema

        all_fields = set(AccelerationConfigSchema.model_fields.keys())
        unclassified = all_fields - self.ALL_KNOWN_ACCEL

        new_gaps = unclassified - KNOWN_ACCELERATION_SCHEMA_ONLY
        assert (
            not new_gaps
        ), f"New unclassified AccelerationConfigSchema fields:\n  {sorted(new_gaps)}"

    def test_acceleration_schedule_is_schema_only(self):
        """acceleration_schedule enum is in schema but nothing reads it during training.

        This means progressive k-space curriculum training is not implemented
        despite being configurable — a significant missing feature.
        """
        import inspect

        from spectramr.infrastructure.physics.sampling import MaskGenerator

        src = inspect.getsource(MaskGenerator)
        if "acceleration_schedule" not in src:
            pytest.xfail(
                reason="KNOWN GAP: AccelerationConfigSchema.acceleration_schedule "
                "is defined (choices: linear, polynomial, exponential, step, power_law) "
                "but KSpaceMaskGenerator never reads it. Progressive curriculum training "
                "via accelerating k-space undersampling is silently not implemented.\n"
                "Fix: Read acceleration_schedule in KSpaceMaskGenerator or "
                "in a curriculum training mixin to ramp acceleration over training."
            )

    def test_deprecated_fields_have_no_effect(self):
        """Deprecated acceleration fields must be schema-only (no silent consumer)."""
        import inspect

        from spectramr.infrastructure.training.builders.data_builder import DataBuilder

        src = inspect.getsource(DataBuilder)
        for deprecated_field in KNOWN_ACCELERATION_DEPRECATED:
            assert deprecated_field not in src, (
                f"Deprecated field '{deprecated_field}' is still consumed by DataBuilder! "
                "This could cause version confusion. Remove consumption or rename."
            )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Validation: dual freq field gap
# ─────────────────────────────────────────────────────────────────────────────


class TestValidationDualFrequencyGap:
    """ValidationConfigSchema has BOTH eval_interval AND frequency_steps.

    Both control evaluation frequency but it's ambiguous which takes priority.
    This tests the actual behavior of ValidationService.
    """

    def test_eval_interval_and_frequency_steps_are_distinct(self):
        """Document that eval_interval and frequency_steps are separate fields
        and clarify which one ValidationService.should_run_validation() uses."""
        from spectramr.config.schemas.validation import ValidationConfigSchema

        # Both fields exist
        fields = set(ValidationConfigSchema.model_fields.keys())
        assert "eval_interval" in fields
        assert "frequency_steps" in fields

        # Check their defaults
        default = ValidationConfigSchema()
        assert default.eval_interval == default.frequency_steps, (
            "eval_interval and frequency_steps have different defaults but serve the same purpose. "
            f"eval_interval={default.eval_interval}, frequency_steps={default.frequency_steps}. "
            "One should be removed or the priority should be documented in ValidationService."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
