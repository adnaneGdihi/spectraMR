#!/usr/bin/env python
"""Configuration Defaults Provider
Provides default configuration values per environment.
Follows SOLID principles with single responsibility for defaults management.
"""

from typing import Any

from pydantic_core import PydanticUndefined

from spectramr.config.schemas.base import CANONICAL_CONFIG_VERSION


class ConfigDefaultsProvider:
    """Provides default configuration values for different environments."""

    def __init__(self, environment: str = "base"):
        """Initialize defaults provider for specific environment.

        Args:
            environment: Environment name ("base", "development",
                          "production", "testing")

        """
        self.environment = environment

    def get_defaults(self, training_mode: str | None = None) -> dict[str, Any]:
        """Get default configuration values for the current environment.

        Args:
            training_mode: Training paradigm (gan, diffusion, reconstruction, etc.)
                          If provided, only includes paradigm-specific defaults.

        Returns:
            Dictionary of default configuration values

        """
        # Base defaults that apply to all environments
        defaults = self._get_base_defaults()

        # Add paradigm-specific defaults only for reconstruction mode
        # This prevents validation errors when GAN/Diffusion schemas reject
        # reconstruction-specific fields like enable_data_consistency, etc.
        if training_mode == "reconstruction":
            defaults.update(self._get_reconstruction_specific_defaults())

        # Environment-specific overrides
        if self.environment == "production":
            defaults.update(self._get_production_defaults())
        elif self.environment == "testing":
            defaults.update(self._get_testing_defaults())
        elif self.environment == "development":
            defaults.update(self._get_development_defaults())
        # For "base" environment, use base defaults without overrides

        return defaults

    def _get_base_defaults(self) -> dict[str, Any]:
        """Get base defaults that apply to all environments."""
        return {
            "config_version": CANONICAL_CONFIG_VERSION,
            "metadata": {
                "name": "comprehensive_experiment_template",
                "description": None,
                "author": None,
                # The AUTHOR's experiment revision, not a schema tier -- and so
                # `None` ("the author did not say"), matching every other
                # free-form neighbour in this dict. It was hardcoded `"6.0"`
                # two lines under the symbolic `CANONICAL_CONFIG_VERSION`
                # above: a *retired, unloadable* config tier (`6.0` left
                # ACCEPTED_CONFIG_VERSIONS in PR #891, so declaring it now
                # raises) standing as the default for a free-form author field.
                #
                # This is LATENT, not live -- do not read it as the mechanism
                # that put 6.x into the corpus. `ConfigDefaultsProvider` has no
                # production call site (verified 2026-08-18: the only importers
                # are `tests/unit/config/**`, plus a prose mention in
                # `config_health_checker.py`), so this dict reaches no arm
                # today. The 467 of 647 `inprogress/` arms carrying a 6.x
                # `metadata.version` declare it in their own YAML, inherited
                # from a template -- that is a corpus edit, tracked separately.
                # Fixing the literal here keeps the provider from re-teaching
                # the retired spelling the day it *is* wired (an unwired
                # surface is capability to wire, not dead code to delete).
                #
                # Why it mattered at all: `TrainingSettings.metadata` is a raw
                # `dict[str, Any]` (ledger class_id `raw_dict_unvalidated`), so
                # nothing validates it and nothing reads it -- it survives only
                # into provenance, where a `6.0` sits beside a real
                # `config_version` and a real `_ledger.schema_version` and
                # makes the artifact read as self-contradictory.
                "version": None,
                "created_at": None,
                "tags": {},
                "timestamp": None,
            },
            # Core training parameters
            "epochs": 200,
            "training_mode": "reconstruction",
            "device": "cpu",  # Safe default; users must explicitly set to "cuda" or "auto"
            "seed": 42,
            "training_seed": 42,  # Global training seed (for reproducibility)
            "validation_seed": 12345,  # Validation mask seed (distinct from training)
            "motion_sim_seed": 42,  # Motion simulation seed
            "output_dir": "./training_output",
            # Model configuration
            "model": {
                "in_channels": 1,
                "out_channels": 1,
                "model_type": "unet",
                "output_activation": None,
                "model_kwargs": {},
                "z_dim": 256,
                "kan_type": "BSpline",
                "architecture": None,
            },
            # Data configuration
            "data": {
                "data_root": "./data",
                "synthetic_data_root": "./data_synthetic",
                "input_lr_dir": "./data_synthetic/A_LRSI",
                "input_hr_dir": "./data_synthetic/A_HRSI",
                "lr_volume_dir": "./data_volumetric/lr",
                "hr_volume_dir": "./data_volumetric/hr",
                "target_size_3d": (128, 128, 128),
                "cache_in_memory": False,
                "image_size": 256,
                "num_workers": 4,
                "batch_size": 4,
                "dataset_type": "2d",
                "synthetic_num_samples": 16,
                "synthetic_lr_shape": (1, 64, 64),
                "synthetic_upscale_factor": 2,
                "synthetic_noise_level": 0.0,
                "ssl_active": False,
                "datasets": [],
                "dataset_mix_strategy": "alternating",
            },
            # Optimization configuration
            "optimization": {
                "learning_rate": 1e-5,
                "generator_learning_rate": None,
                "discriminator_learning_rate": None,
                "optimizer_type": "adamw",
                "weight_decay": 1e-4,
                "use_amp": True,
                "use_gradient_checkpointing": False,
                "gradient_accumulation_steps": 1,
                "lr_scheduler_strategy": "cosine",
                "gradient_clip_value": 1.0,
                "gradient_clip_method": "norm",
                "enable_gradient_clipping": False,
                "beta1": 0.5,
                "beta2": 0.999,
            },
            # Objectives configuration
            # Note: Only include reconstruction defaults here. GAN and diffusion
            # objectives are added by paradigm-specific configuration, not base defaults.
            # This prevents warnings about paradigm-specific fields in other modes.
            "objectives": {
                "reconstruction": {
                    "lambda_l1": 10.0,
                    "lambda_perceptual": 10.0,
                    "lambda_ssim": 0.0,
                    "lambda_lpips": 0.0,
                    "lambda_frequency": 0.0,
                },
            },
            # Logging configuration
            "logging": {
                "log_dir": "./logs",
                "log_interval": 100,
                "save_interval": 10,
                "silent": True,
                "experiment_name": "default_experiment",
                "run_name": None,
                "tags": {},
            },
            # Checkpoint configuration
            "checkpoint": {
                "save_checkpoints": True,
                "checkpoint_frequency": 10,
                "save_interval": 10,
            },
            # Validation configuration
            "validation": {
                "val_interval": 1000,
                "val_frequency": 1,
                "enable_validation": True,
            },
            # Early stopping configuration
            "early_stopping": {
                "enable_early_stopping": False,
                "patience": 20,
                "min_delta": 0.0,
            },
            # Services configuration
            "services": {
                "enable_logging": True,
                "enable_checkpointing": True,
                "enable_profiling": False,
                "enable_metrics_tracking": True,
                "enable_visualization": True,
                "enable_early_stopping": False,
            },
            # EMA configuration
            # Must stay loadable by EMAConfigSchema (extra="forbid"), which this
            # block predates: it still carried the v5.1-removed alias
            # "enable_ema", and it turned on enable_adaptive_ema alongside
            # stability_threshold / adaptation_rate — a combination the schema
            # now refuses because no signal feeds that schedule (#1294).
            # Pinned by tests/unit/config/test_defaults_provider.py.
            "ema": {
                "enabled": True,
                "decay": 0.999,
                "update_frequency": 1,
                "enable_adaptive_ema": False,
                "warmup_steps": 1000,
                "initial_decay": 0.9,
                "final_decay": 0.999,
            },
            # Augmentation configuration
            "augmentation": {
                "enabled": False,
                "probability": 0.5,
                "enable_rotation": False,
                "enable_flip": False,
                "enable_noise": False,
                "enable_brightness": False,
                "enable_contrast": False,
                "enable_gamma": False,
                "enable_rician_noise": False,
                "enable_motion_blur": False,
                "enable_kspace_undersampling_augmentation": False,
                "enable_mixup": False,
                "enable_cutmix": False,
                "enable_elastic_deformation": False,
            },
            # Undersampling configuration (the block was `acceleration:`
            # until phase 11; the legacy spelling still folds).
            "undersampling": {
                "base_acceleration": 1.0,
                "max_acceleration": 32.0,  # Default max acceleration
                "acceleration_schedule": "linear",
                "acceleration_type": "variable_density",  # Default undersampling pattern
                "center_fraction": 0.08,  # Default center fraction for k-space
                "mask_seed": None,  # No default mask seed (use config or random)
            },
            # Physics configuration
            "physics": {
                "flip_angle": 90.0,  # Default 90° flip angle
                "enable_data_consistency": False,
                "data_consistency": {
                    "enabled": False,
                    "weight": 1.0,
                    "interval": 1,
                    "method": "projection_2d_consistency",
                    "train_noise_level": 0.01,
                    "eval_noise_level": 0.005,
                    "noise_type": "gaussian",
                },
                "ode_solver_rtol": 1e-4,  # ODE solver relative tolerance
                "ode_solver_atol": 1e-5,  # ODE solver absolute tolerance
            },
            # Metrics configuration
            "metrics": {
                "batch_size": 64,  # Default metrics evaluation batch size
                "save_images": False,
                "image_format": "png",
            },
            # Inference configuration
            "inference": {
                "batch_size": 4,  # Default inference batch size
                "device": "cpu",  # Default inference device
            },
            # Acceleration/CUDA configuration
            "acceleration_env": {
                "pytorch_cuda_alloc_conf": "expandable_segments:True,max_split_size_mb:512",
                "cuda_cache_maxsize": 2147483648,  # 2GB default
                "cublas_workspace_config": ":4096:8",
            },
            # Parallelism configuration
            "parallel": {
                "strategy": "none",
                "num_devices": 1,
            },
            # Task-based configuration
            "task": "gan_super_resolution",
            "task_settings": {
                "name": "image_to_3d",
                "training_type": "supervised",
                "objective": "gan",
                "target_representation": "volume",
            },
            # Metrics output
            "metrics_output_dir": None,
        }

    def _get_reconstruction_specific_defaults(self) -> dict[str, Any]:
        """Get defaults specific to reconstruction paradigm.

        These fields are defined in TrainingConfigReconstruction but NOT in
        TrainingConfigGAN or TrainingConfigDiffusion. Adding them to base defaults
        causes validation errors for non-reconstruction paradigms.

        Returns:
            Dictionary of reconstruction-specific defaults
        """
        return {
            # Data consistency
            "enable_data_consistency": False,
            "dc_weight": 1.0,
            "dc_interval": 1,
            # Coil sensitivity
            "enable_coil_sensitivity_estimation": False,
            # K-space reconstruction
            "enable_kspace_recon": False,
            # Iterative refinement
            "enable_iterative_refinement": False,
            "num_refinement_steps": 1,
            # Regularization
            "enable_tv_regularization": False,
            "tv_weight": 0.01,
            "enable_wavelets": False,
            "wavelet_weight": 0.01,
        }

    def _get_development_defaults(self) -> dict[str, Any]:
        """Get defaults optimized for development environment."""
        return {
            "epochs": 50,
            "logging": {
                "silent": False,
                "log_interval": 10,
                "save_interval": 5,
            },
            "services": {
                "enable_profiling": True,
            },
            "checkpoint": {
                "save_interval": 5,
            },
            "optimization": {
                "enable_gradient_monitoring": True,
            },
        }

    def _get_production_defaults(self) -> dict[str, Any]:
        """Get defaults optimized for production environment."""
        return {
            "epochs": 500,
            "logging": {
                "silent": True,
                "log_interval": 1000,
                "save_interval": 50,
            },
            "services": {
                "enable_profiling": False,
                "enable_early_stopping": True,
            },
            "checkpoint": {
                "save_interval": 50,
            },
            "optimization": {
                "enable_mixed_precision": True,
                "use_gradient_checkpointing": True,
            },
            "early_stopping": {
                "enable_early_stopping": True,
            },
        }

    def _get_testing_defaults(self) -> dict[str, Any]:
        """Get defaults optimized for testing environment."""
        return {
            "epochs": 1,
            "data": {
                "batch_size": 1,
            },
            "logging": {
                "silent": True,
                "log_interval": 1,
                "save_interval": 1,
            },
            "services": {
                "enable_profiling": False,
            },
            "checkpoint": {
                "save_interval": 1,
            },
            "optimization": {
                "enable_gradient_clipping": True,
                "gradient_clip_value": 0.1,
                "use_amp": False,
                "enable_mixed_precision": False,
            },
        }

    def get_default_for_field(self, field_name: str) -> Any:
        """Get default value for a specific field.

        Args:
            field_name: Name of the field

        Returns:
            Default value for the field, or None if not found

        """
        defaults = self.get_defaults()
        return defaults.get(field_name)

    def merge_with_defaults(
        self,
        config_dict: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge provided config with defaults.

        Args:
            config_dict: User-provided configuration

        Returns:
            Configuration with defaults filled in for missing values

        """
        defaults = self.get_defaults()
        merged = defaults.copy()
        merged.update(config_dict)
        return merged

    @staticmethod
    def extract_defaults_from_schema(schema_class: type) -> dict[str, Any]:
        """Extract default values from a Pydantic schema class.

        This provides the single source of truth for defaults by reading them
        from the Pydantic schema definitions rather than hardcoding them separately.

        Args:
            schema_class: A Pydantic BaseModel class

        Returns:
            Dictionary of field names to their default values
        """
        if not hasattr(schema_class, "model_fields"):
            return {}

        defaults = {}
        for field_name, field_info in schema_class.model_fields.items():
            # Get the default value from the field
            if field_info.default is not PydanticUndefined:
                defaults[field_name] = field_info.default
            elif field_info.default_factory is not None:
                defaults[field_name] = field_info.default_factory()
            else:
                # Field is required, skip it
                pass

        return defaults

    @staticmethod
    def extract_nested_defaults_from_schema(schema_class: type, prefix: str = "") -> dict[str, Any]:
        """Extract nested default values from a Pydantic schema recursively.

        Handles nested BaseModel classes by flattening them into a hierarchical dict.

        Args:
            schema_class: A Pydantic BaseModel class
            prefix: Prefix for nested keys (internal use)

        Returns:
            Nested dictionary of field names to their default values
        """
        if not hasattr(schema_class, "model_fields"):
            return {}

        defaults = {}
        for field_name, field_info in schema_class.model_fields.items():
            key = f"{prefix}.{field_name}" if prefix else field_name

            # Check if the field type is also a BaseModel (nested)
            if hasattr(field_info.annotation, "model_fields"):
                nested_defaults = ConfigDefaultsProvider.extract_nested_defaults_from_schema(
                    field_info.annotation, key
                )
                defaults.update(nested_defaults)
            else:
                # Regular field - get its default
                if field_info.default is not PydanticUndefined:
                    defaults[key] = field_info.default
                elif field_info.default_factory is not None:
                    defaults[key] = field_info.default_factory()

        return defaults
