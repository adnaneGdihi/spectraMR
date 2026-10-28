"""Diffusion Schema Coverage Tests.

PURPOSE:
DiffusionTrainingConfigSchema fields (sampler, prediction_type, noise_schedule,
degradation, etc.) are user-facing but the consuming strategy must actually read
them. This test suite:
  1. Verifies all schema fields are accessible (no AttributeError)
  2. Verifies DiffusionTrainingStrategy reads key fields at setup time
  3. Uses importlib inspection to detect fields that are defined but never
     referenced in the strategy source code
"""

from __future__ import annotations

import inspect

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strategy_source() -> str:
    """Return the full source of DiffusionTrainingStrategy."""
    from spectramr.infrastructure.training.strategies.diffusion import (
        DiffusionTrainingStrategy,
    )

    return inspect.getsource(DiffusionTrainingStrategy)


# ---------------------------------------------------------------------------
# 1. DiffusionTrainingConfigSchema field accessibility
# ---------------------------------------------------------------------------


class TestDiffusionSchemaFieldAccessibility:
    """All DiffusionTrainingConfigSchema fields must be accessible via config."""

    EXPECTED_FIELDS = [
        "timesteps",
        "noise_schedule",
        "sampler",
        "sampling_steps",
        "cond_drop_prob",
        "guidance_scale",
        "type",
        "prediction_type",
        "degradation",
        "enable_diffusion_amp",
        "enforce_output_range",
        "lambda_mse",
        "condition_on_input",
    ]

    def test_condition_on_input_defaults_false(self):
        """The measurement-conditioning flag is a declared, validated bool that
        defaults off (every existing arm stays unconditional / byte-identical)."""
        from spectramr.config.schemas.training.base import DiffusionTrainingConfigSchema

        field = DiffusionTrainingConfigSchema.model_fields["condition_on_input"]
        assert field.default is False
        assert DiffusionTrainingConfigSchema().condition_on_input is False
        assert DiffusionTrainingConfigSchema(condition_on_input=True).condition_on_input is True

    @pytest.mark.parametrize("field_name", EXPECTED_FIELDS)
    def test_field_exists_in_schema(self, field_name: str):
        from spectramr.config.schemas.training.base import DiffusionTrainingConfigSchema

        assert field_name in DiffusionTrainingConfigSchema.model_fields, (
            f"DiffusionTrainingConfigSchema.{field_name} is missing.\n"
            "Either the field was removed from the schema (check consumer code still works) "
            "or this test's EXPECTED_FIELDS list needs updating."
        )

    @pytest.mark.parametrize("field_name", EXPECTED_FIELDS)
    def test_field_accessible_from_instance(self, field_name: str):
        from spectramr.config.schemas.training.base import DiffusionTrainingConfigSchema

        cfg = DiffusionTrainingConfigSchema()
        value = getattr(cfg, field_name, "_MISSING_")
        assert value != "_MISSING_", (
            f"DiffusionTrainingConfigSchema.{field_name} raises AttributeError on access."
        )

    def test_sampler_valid_values(self):
        """sampler field must accept the documented valid values."""
        from spectramr.config.schemas.training.base import DiffusionTrainingConfigSchema

        valid_samplers = ["ddpm", "ddim", "predictor_corrector", "dpm_solver"]
        for sampler in valid_samplers:
            cfg = DiffusionTrainingConfigSchema(sampler=sampler)
            assert cfg.sampler == sampler

    def test_prediction_type_valid_values(self):
        """prediction_type must accept documented values."""
        from spectramr.config.schemas.training.base import DiffusionTrainingConfigSchema

        valid = ["epsilon", "sample", "v_prediction"]
        for pred_type in valid:
            cfg = DiffusionTrainingConfigSchema(prediction_type=pred_type)
            assert cfg.prediction_type == pred_type


# ---------------------------------------------------------------------------
# 2. DiffusionTrainingStrategy SOURCE references key config fields
# ---------------------------------------------------------------------------


class TestDiffusionStrategyConsumesSchemaFields:
    """DiffusionTrainingStrategy source must reference each schema field it is expected to use.

    These tests fail if a field is in the schema as a user-facing parameter
    but the strategy never reads it.
    """

    # (field_name_in_schema, search_term_in_strategy_source)
    # Only fields that DiffusionTrainingStrategy *actually* reads from config.training.diffusion
    EXPECTED_CONSUMPTIONS = [
        ("timesteps", "timesteps"),
        ("noise_schedule", "noise_schedule"),
    ]

    # Fields defined in schema but not yet consumed by the strategy (technical debt)
    # AUDITED: DiffusionTrainingStrategy reads: cold_diffusion, noise_schedule, num_timesteps, timesteps
    # It does NOT read: sampler, prediction_type, lambda_mse, cond_drop_prob, guidance_scale, etc.
    KNOWN_SCHEMA_ONLY_DIFFUSION_FIELDS = {
        "sampler",  # SCHEMA_ONLY: not read in DiffusionTrainingStrategy
        "prediction_type",  # SCHEMA_ONLY: not read in DiffusionTrainingStrategy
        "lambda_mse",  # SCHEMA_ONLY: used in config but not in strategy source
        "cond_drop_prob",  # Classifier-free guidance dropout — not yet wired
        "guidance_scale",  # CFG scale — not yet implemented in training loop
        "enable_diffusion_amp",  # Separate AMP path — not wired
        "sampling_steps",  # Inference-only, not read during training
        "type",  # Used indirectly via cold_diffusion check
        "degradation",  # Read via cold_diffusion branch
        "enforce_output_range",  # Not read in training strategy
    }

    @pytest.mark.parametrize(
        "field_name,search_term",
        EXPECTED_CONSUMPTIONS,
        ids=[c[0] for c in EXPECTED_CONSUMPTIONS],
    )
    def test_field_referenced_in_strategy(self, field_name: str, search_term: str):
        """The diffusion strategy source must reference each expected field."""
        try:
            src = _strategy_source()
        except ImportError:
            pytest.skip("DiffusionTrainingStrategy not importable")

        assert search_term in src, (
            f"DiffusionTrainingStrategy does not reference '{search_term}' anywhere in its source.\n"
            f"Schema field '{field_name}' is user-facing but silently unused.\n"
            "Either implement it in the strategy or move it to KNOWN_SCHEMA_ONLY_DIFFUSION_FIELDS."
        )

    def test_known_schema_only_count_has_not_increased(self):
        """Track known unused diffusion fields to prevent silent proliferation."""
        try:
            src = _strategy_source()
        except ImportError:
            pytest.skip("DiffusionTrainingStrategy not importable")

        from spectramr.config.schemas.training.base import DiffusionTrainingConfigSchema

        all_fields = set(DiffusionTrainingConfigSchema.model_fields.keys())
        consumed_search_terms = {search for _, search in self.EXPECTED_CONSUMPTIONS}

        # Check which fields are NOT referenced in strategy source
        schema_only_fields = {
            field for field in all_fields if field not in consumed_search_terms and field not in src
        }

        new_schema_only = schema_only_fields - self.KNOWN_SCHEMA_ONLY_DIFFUSION_FIELDS
        assert not new_schema_only, (
            f"New schema-only diffusion fields detected (not referenced in strategy):\n"
            f"  {new_schema_only}\n"
            "Either implement them in DiffusionTrainingStrategy or add to "
            "KNOWN_SCHEMA_ONLY_DIFFUSION_FIELDS."
        )


# ---------------------------------------------------------------------------
# 3. LatentTrainingConfigSchema accessibility
# ---------------------------------------------------------------------------


class TestLatentSchemaFieldAccessibility:
    """LatentTrainingConfigSchema fields should all be accessible."""

    EXPECTED_FIELDS = [
        "latent_dim",
        "beta_kl",
        "commitment_weight",
        "codebook_weight",
        "n_embeddings",
        "embedding_dim",
        "anneal_kl_beta",
        "kl_beta_start",
        "kl_beta_end",
        "kl_anneal_steps",
        "lambda_kl",
        "lambda_vq",
        "lambda_commit",
        "temperature",
    ]

    @pytest.mark.parametrize("field_name", EXPECTED_FIELDS)
    def test_field_accessible(self, field_name: str):
        from spectramr.config.schemas.training.base import LatentTrainingConfigSchema

        cfg = LatentTrainingConfigSchema()
        value = getattr(cfg, field_name, "_MISSING_")
        assert value != "_MISSING_", (
            f"LatentTrainingConfigSchema.{field_name} missing or raises AttributeError"
        )


class TestTimestepSamplerValidator:
    """timestep_sampling_strategy enum (2026-06-08 high_t_emphasis addition).

    The valid set is duplicated in two SSOTs that must stay in sync:
    the schema ``VALID_TIMESTEP_SAMPLERS`` and the strategy's
    ``_VALID_SAMPLING_STRATEGIES``. An unknown value must RAISE (pitfall #9),
    never silently fall back."""

    def test_high_t_emphasis_in_literal_and_valid_set(self):
        import typing

        from spectramr.config.schemas.training.diffusion import (
            VALID_TIMESTEP_SAMPLERS,
            TrainingConfigDiffusion,
        )

        assert "high_t_emphasis" in VALID_TIMESTEP_SAMPLERS
        ann = TrainingConfigDiffusion.model_fields["timestep_sampling_strategy"].annotation
        assert "high_t_emphasis" in typing.get_args(ann)

    def test_validator_accepts_high_t_emphasis(self):
        # Call the field validator directly: full-model construction trips an
        # UNRELATED objectives.reconstruction v6.0 migration shim, so isolate the
        # validator that my change actually touches.
        from spectramr.config.schemas.training.diffusion import TrainingConfigDiffusion

        assert (
            TrainingConfigDiffusion._validate_timestep_sampler("high_t_emphasis")
            == "high_t_emphasis"
        )

    def test_validator_rejects_unknown(self):
        from spectramr.config.schemas.training.diffusion import TrainingConfigDiffusion

        with pytest.raises(ValueError, match="timestep_sampling_strategy must be one of"):
            TrainingConfigDiffusion._validate_timestep_sampler("definitely_not_a_sampler")

    def test_balanced_high_t_in_literal_and_valid_set(self):
        """balanced_high_t (2026-07-24 R2x accel-inversion fix) must be a
        selectable strategy in both SSOTs, or the cohort YAML flip is rejected
        at schema parse (dead knob, pitfall #15)."""
        import typing

        from spectramr.config.schemas.training.diffusion import (
            VALID_TIMESTEP_SAMPLERS,
            TrainingConfigDiffusion,
        )

        assert "balanced_high_t" in VALID_TIMESTEP_SAMPLERS
        ann = TrainingConfigDiffusion.model_fields["timestep_sampling_strategy"].annotation
        assert "balanced_high_t" in typing.get_args(ann)

    def test_validator_accepts_balanced_high_t(self):
        from spectramr.config.schemas.training.diffusion import TrainingConfigDiffusion

        assert (
            TrainingConfigDiffusion._validate_timestep_sampler("balanced_high_t")
            == "balanced_high_t"
        )

    def test_dds_in_inference_sampler_literal_and_valid_set(self):
        """SOTA plan step 2: 'dds' must be a selectable inference_sampler value.

        Previously the DDS sampler was registered but absent from the
        inference_sampler enum, so ``inference_sampler: dds`` was rejected at
        schema parse — a dead knob (pitfall #15)."""
        import typing

        from spectramr.config.schemas.training.diffusion import (
            VALID_INFERENCE_SAMPLERS,
            TrainingConfigDiffusion,
        )

        assert "dds" in VALID_INFERENCE_SAMPLERS
        ann = TrainingConfigDiffusion.model_fields["inference_sampler"].annotation
        assert "dds" in typing.get_args(ann)

    def test_validator_accepts_dds(self):
        from spectramr.config.schemas.training.diffusion import TrainingConfigDiffusion

        assert TrainingConfigDiffusion._validate_inference_sampler("dds") == "dds"

    def test_dds_inference_sampler_resolves_to_registered_sampler(self):
        """Every inference_sampler value must name a sampler the registry knows —
        otherwise the enum value is itself a dead knob. Guards 'dds' specifically."""
        from spectramr.config.schemas.training.diffusion import VALID_INFERENCE_SAMPLERS

        try:
            from spectramr.models.diffusion.samplers import SamplerRegistry
        except Exception:  # pragma: no cover - torch-less env
            pytest.skip("sampler registry not importable")

        assert "dds" in VALID_INFERENCE_SAMPLERS
        assert SamplerRegistry.is_registered("dds")

    def test_dynamic_dps_in_enum_and_registered(self):
        """SOTA plan Phase C: 'dynamic_dps' (3 dynamic-guidance variants) must be a
        selectable inference_sampler value resolving to a registered sampler."""
        import typing

        from spectramr.config.schemas.training.diffusion import (
            VALID_INFERENCE_SAMPLERS,
            TrainingConfigDiffusion,
        )

        assert "dynamic_dps" in VALID_INFERENCE_SAMPLERS
        ann = TrainingConfigDiffusion.model_fields["inference_sampler"].annotation
        assert "dynamic_dps" in typing.get_args(ann)
        assert TrainingConfigDiffusion._validate_inference_sampler("dynamic_dps") == "dynamic_dps"
        try:
            from spectramr.models.diffusion.samplers import SamplerRegistry
        except Exception:  # pragma: no cover - torch-less env
            pytest.skip("sampler registry not importable")
        assert SamplerRegistry.is_registered("dynamic_dps")

    def test_schema_and_strategy_sampler_sets_agree(self):
        """The only schema sampler the strategy does not implement is
        'cosine_warm' (pre-existing). Every other schema sampler MUST be handled
        by the strategy, and high_t_emphasis must be in BOTH."""
        from spectramr.config.schemas.training.diffusion import VALID_TIMESTEP_SAMPLERS

        try:
            from spectramr.infrastructure.training.strategies.diffusion import (
                _VALID_SAMPLING_STRATEGIES,
            )
        except Exception:  # pragma: no cover - torch-less env
            pytest.skip("diffusion strategy not importable")

        assert "high_t_emphasis" in VALID_TIMESTEP_SAMPLERS
        assert "high_t_emphasis" in _VALID_SAMPLING_STRATEGIES
        unimplemented = set(VALID_TIMESTEP_SAMPLERS) - set(_VALID_SAMPLING_STRATEGIES)
        assert unimplemented <= {"cosine_warm"}, (
            f"schema advertises samplers the strategy cannot run: {unimplemented}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
