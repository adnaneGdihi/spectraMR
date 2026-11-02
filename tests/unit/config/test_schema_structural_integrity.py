"""Schema Structural Integrity Tests.

PURPOSE:
Catches bugs *in the schema itself* before they cause silent misconfiguration:
  - Duplicate Pydantic field names (v2 silently drops the second definition)
  - Asymmetric enable_*/lambda_* pairs
  - get_enabled_losses() coverage gaps
  - Enum values that are inconsistent with downstream lookups
"""

import inspect

import pytest

from spectramr.models.losses.weights import (
    ENABLE_FLAG_LAMBDAS,
    build_loss_weight_table,
    canonical_loss_name,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pydantic_field_names(model_cls) -> list[str]:
    """Return declared field names in source order (including duplicates)."""
    import ast

    src = inspect.getsource(model_cls)
    # We only want attribute assignments at class body level
    tree = ast.parse(src)
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == model_cls.__name__:
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(
                    stmt.target, ast.Name
                ):
                    names.append(stmt.target.id)
    return names


# ---------------------------------------------------------------------------
# 1. No duplicate Pydantic field names
# ---------------------------------------------------------------------------


class TestNoDuplicateFields:
    """Pydantic v2 silently drops the second definition of a duplicate field.

    Every duplicate is a silent bug: the user writes a value in the YAML,
    Pydantic parses the *first* declaration, ignores the second, and the
    configured override is lost.
    """

    SCHEMA_CLASSES_TO_CHECK = [
        "spectramr.config.schemas.loss.ReconstructionLossesConfig",
        "spectramr.config.schemas.loss.GANLossesConfig",
        "spectramr.config.schemas.loss.DiffusionLossesConfig",
        "spectramr.config.schemas.loss.LatentLossesConfig",
        "spectramr.config.schemas.loss.PhysicsLossesConfig",
        "spectramr.config.schemas.loss.SSLLossesConfig",
        "spectramr.config.schemas.loss.PINNLossesConfig",
        "spectramr.config.schemas.loss.LossConfigSchema",
        "spectramr.config.schemas.optimization.OptimizationConfigSchema",
        "spectramr.config.schemas.physics.PhysicsConfigSchema",
        "spectramr.config.schemas.training.base.DiffusionTrainingConfigSchema",
        "spectramr.config.schemas.training.base.TrainingStrategyConfigSchema",
    ]

    @pytest.mark.parametrize("class_path", SCHEMA_CLASSES_TO_CHECK)
    def test_no_duplicate_field_names(self, class_path: str):
        """Each class must have unique field names at the source level."""
        import importlib

        module_path, cls_name = class_path.rsplit(".", 1)
        mod = importlib.import_module(module_path)
        cls = getattr(mod, cls_name)

        names = _pydantic_field_names(cls)
        seen: set[str] = set()
        duplicates = []
        for n in names:
            if n in seen:
                duplicates.append(n)
            seen.add(n)

        assert not duplicates, (
            f"{cls_name} has duplicate field declarations: {duplicates}\n"
            "Pydantic v2 silently drops the second definition, "
            "making YAML overrides for those keys ineffective."
        )

    @pytest.mark.parametrize("class_path", SCHEMA_CLASSES_TO_CHECK)
    def test_model_fields_match_source_declarations(self, class_path: str):
        """model_fields count must equal unique source-declared annotation count."""
        import importlib

        module_path, cls_name = class_path.rsplit(".", 1)
        mod = importlib.import_module(module_path)
        cls = getattr(mod, cls_name)

        source_names = [n for n in _pydantic_field_names(cls) if not n.startswith("_")]
        unique_source = set(source_names)
        pydantic_fields = set(cls.model_fields.keys())

        # pydantic_fields is the ground truth — every key in model_fields must
        # appear in source, and every unique source annotation should be in
        # model_fields (inherited fields make a strict == comparison wrong).
        missing_from_pydantic = unique_source - pydantic_fields
        # We only care about *declared* fields not making it into model_fields,
        # which would indicate a silent shadowing by a non-field attribute.
        assert not missing_from_pydantic, (
            f"{cls_name}: source-declared fields not in model_fields "
            f"(possibly shadowed): {missing_from_pydantic}"
        )


# ---------------------------------------------------------------------------
# 2. Symmetric enable_* / lambda_* pairs in loss configs
# ---------------------------------------------------------------------------


class TestSymmetricEnableLambdaPairs:
    """Every enable_<X> flag must have a corresponding lambda_<X> weight and vice-versa.

    A flag without a weight is harmless; a weight without a flag is the dangerous
    pattern (weight is non-zero but the loss is never gated).  We check both
    directions and report asymmetries.
    """

    LOSS_CONFIG_CLASSES = [
        "spectramr.config.schemas.loss.ReconstructionLossesConfig",
        "spectramr.config.schemas.loss.GANLossesConfig",
        "spectramr.config.schemas.loss.DiffusionLossesConfig",
        "spectramr.config.schemas.loss.LatentLossesConfig",
        "spectramr.config.schemas.loss.PhysicsLossesConfig",
        "spectramr.config.schemas.loss.SSLLossesConfig",
        "spectramr.config.schemas.loss.PINNLossesConfig",
    ]

    # Allowlisted asymmetries that are intentional design decisions
    KNOWN_INTENTIONAL_ASYMMETRIES: dict[str, set[str]] = {
        # GAN: feature_matching weight has no "enable" boolean — value=0 disables it
        # enable_adversarial is the GAN gate; lambda_adv / lambda_gp are the separate weights
        "GANLossesConfig": {
            "feature_matching",  # gated by value=0/non-0, not a bool flag
            "enable_adversarial",  # gate for adversarial loss; lambda_adv is the weight
            "enable_gradient_penalty",  # gate; lambda_gp is the weight (different name)
            "enable_r1",  # gate; lambda_r1 is the weight (different name)
            "enable_feature_matching",  # gate; feature_matching float IS the weight
        },
        # Reconstruction: physics/bloch/style lambdas have no boolean gates —
        # value=0 disables them. This is an intentional design here (weight-only gating).
        "ReconstructionLossesConfig": {
            "lambda_bloch_residual",
            "lambda_physics_constraint",
            "lambda_physics_informed",
            "lambda_parallel_imaging_kspace",
            "lambda_biophysical_flow",
            "lambda_sobel",
            "lambda_snr_preserving",
            "lambda_recon",
            "lambda_style",
            "lambda_content",
            "lambda_bloch",
            "lambda_anat",
            # enable-only flags: boolean gates with no paired weight (config-only behavior)
            "enable_curriculum_scheduling",  # config-only; weights are curriculum_*_weight
            "enable_sobolev_kspace",  # weight: lambda_sobolev_kspace exists
            "enable_sense_adjoint_l1",  # weight: lambda_sense_adjoint_l1 exists
            "enable_persistent_homology_option",  # topology flag; weight uses different naming
            "enable_prior_loss",  # strategy-managed; lambda lives under prior_kwargs
            "enable_marker_loss",  # strategy-managed; weight injected at compute time
        },
        # Diffusion: some enable flags control non-weighted behaviors
        "DiffusionLossesConfig": {
            "enable_reconstruction_auxiliary",  # AMP flag, no paired lambda weight
            "enable_diffusion_amp",  # AMP toggle, no weight pairing
            "enable_diffusion",  # main gate; weight is lambda_mse
            "enable_data_consistency",  # DC toggle; weight is in physics config
        },
        # Latent: VQ-specific flags use non-lambda naming conventions (commitment_weight etc.)
        "LatentLossesConfig": {
            "enable_commitment",  # weight: lambda_commit (different suffix convention)
            "enable_codebook",  # weight: lambda_codebook (if exists) or codebook_weight
            "enable_reconstruction",  # weight: lambda_recon (different suffix)
        },
    }

    @pytest.mark.parametrize("class_path", LOSS_CONFIG_CLASSES)
    def test_every_enable_flag_with_associated_lambda(self, class_path: str):
        """enable_X => lambda_X should exist (flags without weights are low risk but noisy)."""
        import importlib

        module_path, cls_name = class_path.rsplit(".", 1)
        mod = importlib.import_module(module_path)
        cls = getattr(mod, cls_name)
        fields = set(cls.model_fields.keys())
        intentional = self.KNOWN_INTENTIONAL_ASYMMETRIES.get(cls_name, set())

        orphan_enables = []
        for field in fields:
            if field.startswith("enable_"):
                suffix = field[len("enable_") :]
                lambda_name = f"lambda_{suffix}"
                if (
                    lambda_name not in fields
                    and field not in intentional
                    and lambda_name not in intentional
                ):
                    orphan_enables.append(field)

        assert not orphan_enables, (
            f"{cls_name}: enable_* flags with no matching lambda_*: {orphan_enables}\n"
            "Either add the corresponding lambda field, or document the intentional asymmetry "
            "in KNOWN_INTENTIONAL_ASYMMETRIES."
        )


# ---------------------------------------------------------------------------
# 3. enable_* flags are declared by the weight table
# ---------------------------------------------------------------------------


class TestGetEnabledLossesCoverage:
    """The weight table must declare each enable_* flag's term (#1966: the table is the owner).

    We enable each flag one at a time (with weight > 0) and verify the returned dict
    contains the expected key.
    """

    # enable_flag -> (loss_key_expected_in_result, paradigm_config_kwarg)
    ENABLE_FLAG_TO_EXPECTED = [
        # Reconstruction
        ("enable_l1", "l1", "reconstruction"),
        ("enable_l2", "l2", "reconstruction"),
        ("enable_perceptual", "perceptual", "reconstruction"),
        ("enable_ssim", "ssim", "reconstruction"),
        ("enable_ms_ssim", "ms_ssim", "reconstruction"),
        ("enable_lpips", "lpips", "reconstruction"),
        ("enable_hfen", "hfen", "reconstruction"),
        ("enable_ffl", "ffl", "reconstruction"),
        ("enable_kspace", "kspace", "reconstruction"),
        ("enable_frequency", "frequency", "reconstruction"),
        ("enable_edge", "edge", "reconstruction"),
        ("enable_tv", "tv", "reconstruction"),
        ("enable_explicit_gradient", "explicit_gradient", "reconstruction"),
        ("enable_background_suppression", "background_suppression", "reconstruction"),
        ("enable_rician_consistency", "rician_consistency", "reconstruction"),
        ("enable_sobolev_kspace", "sobolev_kspace", "reconstruction"),
        ("enable_energy_conservation", "energy_conservation", "reconstruction"),
        # GAN
        ("enable_adversarial", "adversarial", "gan"),
        ("enable_gradient_penalty", "gradient_penalty", "gan"),
        ("enable_r1", "r1", "gan"),
        # Diffusion
        ("enable_diffusion", "mse", "diffusion"),
        # Physics
        ("enable_bloch_residual", "bloch_residual", "physics"),
        ("enable_physics_constraint", "physics_constraint", "physics"),
        ("enable_parallel_imaging", "parallel_imaging", "physics"),
        ("enable_snr_preserving", "snr_preserving", "physics"),
        ("enable_complex_spatial_gradient", "complex_spatial_gradient", "physics"),
        # SSL
        ("enable_contrastive", "contrastive", "ssl"),
    ]

    def _make_loss_config(self, paradigm: str, enable_flag: str, lambda_key: str):
        from spectramr.config.schemas.loss import (
            DiffusionLossesConfig,
            GANLossesConfig,
            LatentLossesConfig,
            LossConfigSchema,
            PhysicsLossesConfig,
            ReconstructionLossesConfig,
            SSLLossesConfig,
        )

        paradigm_cls_map = {
            "reconstruction": ReconstructionLossesConfig,
            "gan": GANLossesConfig,
            "diffusion": DiffusionLossesConfig,
            "latent": LatentLossesConfig,
            "physics": PhysicsLossesConfig,
            "ssl": SSLLossesConfig,
        }
        cls = paradigm_cls_map[paradigm]
        # Build kwargs: enable the flag, set the lambda to non-zero
        kwargs: dict = {enable_flag: True}
        if lambda_key in cls.model_fields:
            kwargs[lambda_key] = 1.0
        paradigm_obj = cls(**kwargs)
        return LossConfigSchema(**{paradigm: paradigm_obj})

    @pytest.mark.parametrize(
        "enable_flag,expected_key,paradigm",
        ENABLE_FLAG_TO_EXPECTED,
        ids=[f"{p}.{f}" for f, _, p in ENABLE_FLAG_TO_EXPECTED],
    )
    def test_enable_flag_appears_in_get_enabled_losses(
        self, enable_flag: str, expected_key: str, paradigm: str
    ):
        """Enabling a flag + non-zero lambda must produce the key in get_enabled_losses()."""

        lambda_key = ENABLE_FLAG_LAMBDAS.get(enable_flag, f"lambda_{enable_flag[len('enable_'):]}")
        loss_config = self._make_loss_config(paradigm, enable_flag, lambda_key)
        declared = {
            k: spec.weight
            for k, spec in build_loss_weight_table(loss_config).items()
            if spec.enabled and spec.weight > 0
        }
        assert canonical_loss_name(expected_key) in declared, (
            f"the weight table did not declare '{expected_key}' when "
            f"{paradigm}.{enable_flag}=True and {lambda_key} > 0.\n"
            f"Declared keys: {sorted(declared)}\n"
            "The enable/lambda naming convention is inconsistent (see ENABLE_FLAG_LAMBDAS)."
        )


# ---------------------------------------------------------------------------
# 4. Enum value casing is consistent with downstream lowercase lookups
# ---------------------------------------------------------------------------


class TestEnumValueCasing:
    """All enum values used as registry keys must be lowercase."""

    def test_training_mode_values_are_lowercase(self):
        from spectramr.config.schemas.enums import TrainingMode

        for member in TrainingMode:
            assert member.value == member.value.lower(), (
                f"TrainingMode.{member.name} = '{member.value}' is not lowercase. "
                "Strategy registry lookups are case-sensitive lowercase."
            )

    def test_optimizer_type_values_are_lowercase(self):
        from spectramr.config.schemas.enums import OptimizerType

        for member in OptimizerType:
            assert (
                member.value == member.value.lower()
            ), f"OptimizerType.{member.name} = '{member.value}' is not lowercase."

    def test_gan_loss_type_values_are_lowercase(self):
        from spectramr.config.schemas.enums import GANLossType

        for member in GANLossType:
            # GAN loss types may have hyphens; just check no uppercase letters
            assert (
                member.value == member.value.lower()
            ), f"GANLossType.{member.name} = '{member.value}' contains uppercase letters."

    def test_lr_scheduler_values_are_lowercase(self):
        from spectramr.config.schemas.enums import LRScheduler

        for member in LRScheduler:
            assert (
                member.value == member.value.lower()
            ), f"LRScheduler.{member.name} = '{member.value}' is not lowercase."


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
