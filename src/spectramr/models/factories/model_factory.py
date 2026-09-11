"""Factory helpers for registered generators, discriminators, and more.

This module centralizes alias resolution, validation, and instantiation for the
model families exposed through ``ModelFactory``. Component-specific helpers
delegate to :func:`_create_registered_model` so new registries and fallback
policies can be wired in one place.
"""

import importlib
import inspect
import logging
from typing import TYPE_CHECKING, Any, Optional, cast

if TYPE_CHECKING:
    from spectramr.config.schemas.training import TrainingConfigSchema as TrainingConfig

import torch
from torch import nn

from spectramr.models.errors import ModelCreationError

# NOTE (2026-07-02): the ``import spectramr.models.sota_registry`` block that
# lived here is gone — sota_registry is a deprecated no-op shim. SOTA models
# register via their own @register_model decorators, fired by
# ``spectramr.models.init_registry.populate_model_registry()``.
# Import diffusion implementations
# Presets imported from external file to reduce bloat
from spectramr.models.factories.presets import DIFFUSION_CONFIGS, DIFFUSION_PROCESSES
from spectramr.models.interfaces import IDiscriminator, IGenerator, IModel

#: Module logger. Its absence was not cosmetic: `create_model` and
#: `_create_diffusion_model_internal` each open with a bare `logger.debug`,
#: so both raised NameError on their FIRST statement, for every caller and
#: every config. That is why `spectramr predict` / `make` failed on every
#: arm -- not only the 12 that trip the kspace_log_scaled guard behind it.
#: ruff had been reporting all five sites as F821 the whole time.
logger = logging.getLogger(__name__)


# Component classes for ModelFactory (inlined for consolidation)
class ModelRegistry:
    """Registry for managing generator and discriminator classes."""

    def __init__(self):
        """__init__."""
        self._generators: dict[str, type[IGenerator]] = {}
        self._discriminators: dict[str, type[IDiscriminator]] = {}

        # Sync with global registry, bucketing on the role each registration
        # DECLARES rather than one guessed from its base classes.
        #
        # This used to read ``issubclass(cls, IDiscriminator)`` with an
        # ``else: # Default to generator for backward compatibility`` tail.
        # Only 10 of the framework's discriminators subclass that interface,
        # so 9 more -- domain_discriminator, frequency_domain_discriminator,
        # kan_discriminator, kspace_discriminator, ldm_latent_discriminator,
        # ldm_multiscale_latent_discriminator, ldm_patch_latent_discriminator,
        # stargan_v2_discriminator, wasserstein_discriminator -- fell through
        # to the generator bucket, and ``create_discriminator`` then raised
        # "Discriminator type '<x>' not registered" for each. A live build
        # failure for any GAN arm naming one of them, produced by a
        # classifier that could not report its own uncertainty (#1932).
        #
        # The import is NOT wrapped in try/except ImportError. The old bare
        # ``except ImportError: pass`` left BOTH buckets empty and silent, so
        # every create_generator and create_discriminator call reported its
        # name as unregistered -- a failure indistinguishable from an empty
        # registry (non-negotiable 3). spectramr.models.registry is a
        # first-party module; if it cannot import, that is not a condition to
        # degrade past.
        import spectramr.models.registry as global_registry

        for name, metadata in global_registry.list_models().items():
            cls = metadata.get("class")
            if cls is None:
                continue
            if metadata.get("role") == "discriminator":
                self._discriminators[name] = cls
            else:
                self._generators[name] = cls

    def register_generator(self, name: str, generator_class: type[IGenerator]) -> None:
        """Register a generator class."""
        self._generators[name] = generator_class

    def register_discriminator(self, name: str, discriminator_class: type[IDiscriminator]) -> None:
        """Register a discriminator class."""
        self._discriminators[name] = discriminator_class

    def has_generator(self, name: str) -> bool:
        """Check if a generator is registered."""
        return name in self._generators

    def has_discriminator(self, name: str) -> bool:
        """Check if a discriminator is registered."""
        return name in self._discriminators

    def get_generator_class(self, name: str) -> type | None:
        """Get a registered generator class."""
        return self._generators.get(name)

    def get_discriminator_class(self, name: str) -> type | None:
        """Get a registered discriminator class."""
        return self._discriminators.get(name)

    def get_available_generators(self) -> list[str]:
        """Get list of available generator names."""
        return list(self._generators.keys())

    def get_available_discriminators(self) -> list[str]:
        """Get list of available discriminator names."""
        return list(self._discriminators.keys())


class ParameterMapper:
    """Component for mapping parameters between different model types."""

    def __init__(self):
        """__init__."""
        self._model_specific_mappings = {
            "vit_kan": {
                "num_heads": "heads",  # Map num_heads -> heads
            },
            "enhanced_deep_unet": {
                # Map config params to EnhancedDeepDiffusionUNet params
                # Note: 'features' is not directly mappable - needs custom handling
            },
        }

        # Model-specific kwargs to filter (remove if present)
        self._model_filter_list = {
            "pimn": {"hidden_channels"},
            "diffusion_sr": {
                "acceleration_config",
                "use_mc_dropout",
                "mc_dropout_p",
                "num_res_blocks",
                "attention_resolutions",
                "xdiffusion_latent_bridge",
                "latent_context_dim",
                "cross_domain_encoder",
                "chi_square_guidance",
                "num_heads",
                "input_type",
                "z_dim",
                "weight_init_type",
                "weight_init_gain",
                "weight_init_activation",
                "apply_weight_init",
                "image_size",
                "features",
                "feature_dim",
                "num_layers",
                "use_attention",
                "straight_path",
            },
            "diffusion_sr_generator": {
                "acceleration_config",
                "features",
                "feature_dim",
                "num_layers",
                "use_attention",
                "straight_path",
            },
            "rectified_flow": {
                "acceleration_config",
                "features",
                "feature_dim",
                "num_layers",
                "use_attention",
                "straight_path",
            },
            "implicit_neural": {"pos_enc_levels", "features", "acceleration_config"},
            "unetr": {"features", "acceleration_config"},
            "unetr_generator": {"features", "acceleration_config"},
            "laplace_diffusion_generator": {"features", "acceleration_config"},
            "laplace_diffusion": {"features", "acceleration_config"},
            "rician_diffusion_generator": {"features", "acceleration_config"},
            "rician_diffusion": {"features", "acceleration_config"},
            "stable_diffusion_adapter": {"acceleration_config"},
            "score_based_diffusion": {
                "acceleration_config",
                "features",
                "score_matching",
                "sde_type",
                "sigma_min",
                "sigma_max",
            },
            "score_based_diffusion_generator": {
                "acceleration_config",
                "features",
                "score_matching",
                "sde_type",
                "sigma_min",
                "sigma_max",
            },
            "neural_ode": {
                "acceleration_config",
                "ode_solver",
                "latent_dim",
                "hidden_dim",
            },  # Filter hidden_dim if present
            "neural_ode_adaptive": {
                "hidden_dim",
                "in_channels",
                "out_channels",
                "acceleration_config",
            },
            "enhanced_unet": {
                "acceleration_config",
                "hidden_dim",
                "out_channels",
                "features",
                "num_layers",
                "solver",
                "rtol",
                "atol",
                "adaptive",
            },
            "enhanced_deep_unet": {
                "acceleration_config",
                # EnhancedDeepDiffusionUNet doesn't accept these config params
                "features",  # Uses channel_mult instead (handled by map_generator_params)
                "time_dim",  # Uses model_channels * 4 internally
                "hidden_dims",
                "num_resblocks",  # Uses num_res_blocks
                "param_group",
            },
            "latent_diffusion_sr": {
                "acceleration_config",
                "time_embedding_dim",
                "num_timesteps",
                "num_res_blocks",
                "attention_resolutions",
                "use_self_conditioning",
                "apply_latent_data_consistency",
                "kspace_operator",
                "mask_strategy",
                "consistency_start_step",
                "consistency_end_step",
                "consistency_schedule",
                "latent_consistency_weight",
                "reencode_after_consistency",
                "ema_decay",
                "ema_update_every",
                "input_type",
                "z_dim",
                "weight_init_type",
                "weight_init_gain",
                "weight_init_activation",
                "apply_weight_init",
            },
            "vit_kan": {
                "acceleration_config",
                "num_classes",
                "dropout_prob",
                "attention_dropout_prob",
                "use_cls_token",
                "use_mean_pooling",
                "use_stochastic_depth",
                "stochastic_depth_prob",
                "num_res_blocks",
                "use_mc_dropout",
                "mc_dropout_p",
                "attention_resolutions",
                "xdiffusion_latent_bridge",
                "latent_context_dim",
                "cross_domain_encoder",
                "chi_square_guidance",
            },
            "pin_nerf": {
                "acceleration_config",
                "weight_init_type",
                "apply_weight_init",
                "norm_type",
                "act_type",
                "dropout_p",
                "use_mc_dropout",
                "mc_dropout_p",
                "cold_schedule",
                "config",
                "pos_enc_levels",
                "features",
                "in_channels",
                "out_channels",
                "hidden_dim",
                "num_layers",
                "activation_type",
            },
            "fno": {
                "acceleration_config",
                "weight_init_type",
                "apply_weight_init",
                "norm_type",
                "act_type",
                "dropout_p",
            },
            "graph_unet": {
                "acceleration_config",
                "num_layers",
                "num_heads",
                "use_mc_dropout",
                "mc_dropout_p",
                "cold_schedule",
                "config",
            },
            "medgs": {
                "acceleration_config",
                "output_type",
                "in_channels",
                "out_channels",
                "embed_dim",
                "input_ch",
                "output_ch",
            },
            "physics_aware_gaussian_splatter": {
                "acceleration_config",
                "output_type",
                "in_channels",
                "out_channels",
                "embed_dim",
                "input_ch",
                "output_ch",
            },
        }

    def map_generator_params(self, model_type: str, **kwargs) -> dict[str, Any]:
        """Map parameters for generator creation.

        Applies model-specific parameter mappings and filters invalid kwargs.
        """
        result = dict(kwargs)

        # Apply model-specific mappings
        if model_type in self._model_specific_mappings:
            mapping = self._model_specific_mappings[model_type]
            for old_key, new_key in mapping.items():
                if old_key in result:
                    result[new_key] = result.pop(old_key)

        # Filter out model-specific invalid kwargs
        if model_type in self._model_filter_list:
            invalid_keys = self._model_filter_list[model_type]
            result = {k: v for k, v in result.items() if k not in invalid_keys}

        return result

    def map_discriminator_params(self, model_type: str, **kwargs) -> dict[str, Any]:
        """Map parameters for discriminator creation.

        Handles known parameter aliases between YAML configs and discriminator
        constructors (e.g., ``num_layers`` → ``n_layers`` for PatchGAN).
        """
        result = dict(kwargs)

        # PatchGAN uses 'n_layers' but YAML configs often specify 'num_layers'
        if "num_layers" in result and "n_layers" not in result:
            result["n_layers"] = result.pop("num_layers")

        return result


class ActivationManager:
    """Component for managing activation functions in models."""

    def __init__(self):
        """__init__."""
        pass

    def apply_activation(self, model: nn.Module, model_type: str) -> nn.Module:
        """Apply appropriate activation to model output."""
        return model


class ModelValidator:
    """Component for validating model creation parameters."""

    def __init__(self, registry: ModelRegistry):
        """__init__.

        Args:
            registry (ModelRegistry): Description.
        """
        self._registry = registry

    def validate_generator_params(self, model_type: str, **kwargs) -> None:
        """Validate generator parameters."""
        pass

    def validate_discriminator_params(self, model_type: str, **kwargs) -> None:
        """Validate discriminator parameters."""
        pass


def _filter_kwargs_to_signature(
    mapped: dict[str, Any],
    target_class: type | None,
    *,
    logger_: Any,
    site: str,
    role: str = "",
    declared: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Drop kwargs the target's ``__init__`` cannot accept, and RECORD the drop.

    The filtering itself is deliberate and stays: injecting ``model_kwargs`` and
    physics config into a constructor that never declared those parameters is a
    supported flexibility path, and removing it would break the corpus. What was
    not acceptable is that the drop was invisible. This function computed
    ``rejected``, logged it at ``debug`` — a level no production run enables —
    and discarded it, in three separate copies of the same block.

    That is how ``validation.sampler_steps`` was lost: it had a schema field, a
    resolver and a call site, and still died here, so the ldm cohort ran 1000 NFE
    instead of the declared 25 (#480). Nothing in any artifact said so.

    Now each dropped key becomes a ``DROPPED_UNCONSUMED_KWARG`` record naming the
    consumer that refused it, so ``resolved_config.json`` carries the evidence.
    Recording is separated from the decision to raise: ``scheduler_resolution``
    and ``optimization_builder`` must raise on an unroutable knob, this seam must
    not, but both must be visible.
    """
    from spectramr.core.component_signature import signature_contract
    from spectramr.core.execution_ledger import SubstitutionClass, unconsumed_keys

    if target_class is None:
        # A registry miss is the caller's error to report, not ours to mask.
        return mapped

    contract = signature_contract(target_class)
    label = f"{target_class.__name__}{f' {role}' if role else ''}"

    if contract.accepts_var_kwargs:
        # Nothing is dropped — but acceptance into ``**kwargs`` is not
        # consumption, and staying silent here is the same invisibility the
        # DROPPED record exists to break (#878).
        #
        # Scoped to AUTHOR-DECLARED keys only. `GeneratorBuilder.build()`
        # opportunistically forwards every top-level `model.*` field on the
        # chance the constructor wants it; recording those would bury the
        # `model_kwargs` entries that are the actual signal under one record per
        # schema field per arm. `declared=None` (every caller that cannot tell
        # the two apart) therefore records nothing.
        if declared:
            unconsumed_keys(
                {k: v for k, v in mapped.items() if k in declared},
                contract.accepted,
                site=site,
                stage="model_build",
                consumer=f"{label}.__init__(**kwargs)",
                class_id=SubstitutionClass.EXTRA_ALLOW_UNTYPED,
            )
        return mapped

    if not contract.owner:
        # No class in the MRO defines __init__ (only nn.Module/object do), so
        # there is no contract to filter against. Passing through unfiltered.
        #
        # The previous inline reading called `inspect.signature(cls.__init__)`,
        # which reports nn.Module's `(*args, **kwargs)` here and so took the
        # var-kwargs branch above. Same outcome — pass through — but for a
        # reason that was wrong, and that reason is the PR #718 bug class.
        return mapped

    rejected = unconsumed_keys(
        mapped,
        contract.accepted,
        site=site,
        stage="model_build",
        consumer=f"{label}.__init__",
    )
    if not rejected:
        return mapped
    logger_.debug(f"[ModelCreator] Stripping invalid kwargs for {label}: {rejected}")
    return {k: v for k, v in mapped.items() if k in contract.accepted}


class ModelCreator:
    """Component for creating model instances with proper parameter handling."""

    def __init__(
        self,
        registry: ModelRegistry,
        parameter_mapper: ParameterMapper,
        validator: ModelValidator,
        activation_manager: ActivationManager,
    ):
        """__init__.

        Args:
            registry (ModelRegistry): Description.
            parameter_mapper (ParameterMapper): Description.
            validator (ModelValidator): Description.
            activation_manager (ActivationManager): Description.
        """
        self._registry = registry
        self._parameter_mapper = parameter_mapper
        self._validator = validator
        self._validator = validator
        self._activation_manager = activation_manager
        self._logger = logging.getLogger(__name__)

        # Registry-based handler dispatcher for special model creation logic
        # O(1) lookup instead of O(n) if/elif chains
        self._special_handlers = {
            "pggs_pipeline": self._handle_pggs_pipeline,
            "uncertainty_wrapper": self._handle_uncertainty_wrapper,
            "adversarial_purification": self._handle_adversarial_purification,
        }

        # Diffusion wrappers that require denoising model
        self._diffusion_wrappers = {
            "laplace_diffusion",
            "laplace_diffusion_generator",
            "rician_diffusion",
            "rician_diffusion_generator",
            "score_based_diffusion",
            "score_based_diffusion_generator",
            "cold_diffusion",
            "chi_square_diffusion",
            "chi_square_diffusion_generator",
            "consistency_model",
            "consistency_model_generator",
        }
        # Register diffusion wrappers with handler
        for wrapper_type in self._diffusion_wrappers:
            self._special_handlers[wrapper_type] = self._handle_diffusion_wrapper

    def create_generator(
        self,
        model_type: str,
        *,
        _declared_keys: frozenset[str] | None = None,
        **kwargs,
    ) -> IGenerator:
        """Create a generator instance using registry-based handler dispatch.

        Uses O(1) handler lookup instead of O(n) if/elif chains.
        Special handlers registered in __init__ for complex model creation logic.

        ``_declared_keys`` names the subset of ``kwargs`` the ARM actually wrote
        (``model.model_kwargs``), as opposed to the top-level ``model.*`` fields
        ``GeneratorBuilder.build()`` forwards opportunistically. Only the former
        is worth reporting when a ``**kwargs`` constructor swallows it — see
        :func:`_filter_kwargs_to_signature`. Keyword-only and underscore-led so
        it cannot collide with a YAML-authored model parameter; ``None`` (every
        caller that cannot distinguish the two) reports nothing.
        """
        # Check if this model type has a special handler (O(1) dict lookup)
        if model_type in self._special_handlers:
            handler = self._special_handlers[model_type]
            return handler(model_type, **kwargs)

        # Generic path for all other models
        if not self._registry.has_generator(model_type):
            raise ModelCreationError(f"Generator type '{model_type}' not registered")

        generator_class = self._registry.get_generator_class(model_type)

        # [FIX] Sanitize input kwargs by converting Pydantic models to dicts
        # This prevents AttributeError: 'Schema' object has no attribute 'get' in older models
        clean_kwargs = {
            k: v.model_dump() if hasattr(v, "model_dump") else v for k, v in kwargs.items()
        }

        mapped_params = self._parameter_mapper.map_generator_params(model_type, **clean_kwargs)
        self._validator.validate_generator_params(model_type, **mapped_params)

        # [FIX] Filter kwargs against constructor signature to prevent
        # "unexpected keyword argument" when model_kwargs or physics
        # config inject keys (use_dc, bottleneck_only, num_grids, etc.)
        # that the generator's __init__ doesn't accept.
        mapped_params = _filter_kwargs_to_signature(
            mapped_params,
            generator_class,
            logger_=self._logger,
            site="spectramr.models.factories.model_factory:create_generator",
            declared=_declared_keys,
        )

        # [FIX] Coerce list-valued params to scalars ONLY when the constructor
        # actually expects int.  Many generators (U-Nets, etc.) use `features`
        # as a list of channel counts per encoder level.  Blindly taking [0]
        # destroys the architecture and causes downstream "'int' object is not
        # subscriptable / iterable / has no len()" errors.
        _MAYBE_INT_PARAMS = {  # noqa: N806 - in-function constant, pre-existing style
            "in_channels",
            "out_channels",
            "num_layers",
            "num_heads",
            "hidden_dim",
            "latent_dim",
            "embed_dim",
            "num_coils",
            "num_unrolls",
            "num_stages",
            "features",
        }
        try:
            _sig = inspect.signature(generator_class.__init__)
            _param_annotations = {
                name: p.annotation for name, p in _sig.parameters.items() if name != "self"
            }
        except (ValueError, TypeError):
            _param_annotations = {}

        for k in _MAYBE_INT_PARAMS:
            if k in mapped_params and isinstance(mapped_params[k], (list, tuple)):
                original = mapped_params[k]
                annotation = _param_annotations.get(k, inspect.Parameter.empty)

                # Determine if the constructor wants a scalar
                wants_scalar = False
                if annotation is not inspect.Parameter.empty:
                    ann_str = str(annotation)
                    # Check for int-like annotations
                    if annotation is int or ann_str in ("int", "<class 'int'>"):
                        wants_scalar = True
                    elif (
                        "list" in ann_str.lower()
                        or "tuple" in ann_str.lower()
                        or "sequence" in ann_str.lower()
                    ):
                        wants_scalar = False
                    elif "int" in ann_str and "list" not in ann_str.lower():
                        wants_scalar = True
                else:
                    # No annotation: only coerce single-element lists
                    wants_scalar = len(original) == 1

                if wants_scalar:
                    # F-COERCE-LOUD / 2026-05-20 — silently dropping
                    # ``original[1:]`` is a CLAUDE.md pitfall #9 silent
                    # fallback: a YAML that intended a 5-level UNet
                    # (``features: [32, 64, 128, 256, 512]``) became a
                    # 32-channel residual block while the YAML and
                    # logs both *looked* normal. The 2026-05-19 smoke
                    # surfaced this for ``NeuralODEGenerator`` whose
                    # constructor accepts ``features: int``. Refuse
                    # the silent narrowing if more than one element
                    # would be discarded; let the user pick whether
                    # to flatten or switch generator.
                    if len(original) > 1:
                        raise TypeError(
                            f"[ModelCreator] {generator_class.__name__}.__init__ "
                            f"declares ``{k}: int`` but the YAML passed "
                            f"{original!r}. Silent coercion to {original[0]!r} "
                            f"would discard {len(original) - 1} element(s) and "
                            f"violate CLAUDE.md pitfall #9 (no silent "
                            f"fallback). Either set the YAML key to a single "
                            f"int (e.g. ``{k}: {original[0]}``) or switch "
                            f"to a generator whose constructor accepts a "
                            f"list of channel widths."
                        )
                    mapped_params[k] = original[0]
                    self._logger.warning(
                        f"[ModelCreator] Coerced single-element list param "
                        f"'{k}' {original} → {mapped_params[k]} "
                        f"for {generator_class.__name__} (constructor expects int)"
                    )
                else:
                    self._logger.debug(
                        f"[ModelCreator] Preserving list param '{k}' {original} "
                        f"for {generator_class.__name__} (constructor expects list/sequence)"
                    )

        generator = generator_class(**mapped_params)
        generator = self._activation_manager.apply_activation(generator, model_type)
        self._logger.debug(f"ModelCreator created {model_type}: {type(generator)}")
        if generator is None:
            self._logger.error(
                f"CRITICAL: Generator {model_type} became None after creation/activation!"
            )
        return generator

    def _handle_pggs_pipeline(self, model_type: str, **kwargs) -> IGenerator:
        """Handler for PGGS Pipeline creation (instantiate sub-models)."""
        pags_config = kwargs.pop("pags_model", None)
        lpd_config = kwargs.pop("lpd_model", None)

        if not pags_config:
            raise ModelCreationError("PGGS Pipeline requires 'pags_model' config")

        if isinstance(pags_config, dict):
            pags_type = pags_config.pop("model_type", None) or pags_config.pop(
                "type", "physics_aware_gaussian_splatter"
            )
            # Clean nested config
            if "model_kwargs" in pags_config:
                pags_config.update(pags_config.pop("model_kwargs"))

            # Map alias-safe keys
            if "input_ch" in pags_config:
                pags_config["in_channels"] = pags_config.pop("input_ch")
            if "output_ch" in pags_config:
                pags_config["out_channels"] = pags_config.pop("output_ch")

            # Create PAGS
            kwargs["medgs_model"] = self.create_generator(pags_type, **pags_config)

        if lpd_config and isinstance(lpd_config, dict):
            lpd_type = lpd_config.pop("model_type", None) or lpd_config.pop(
                "type", "latent_gaussian_diffusion"
            )
            if "model_kwargs" in lpd_config:
                lpd_config.update(lpd_config.pop("model_kwargs"))

            # Map alias-safe keys
            if "input_ch" in lpd_config:
                lpd_config["in_channels"] = lpd_config.pop("input_ch")
            if "output_ch" in lpd_config:
                lpd_config["out_channels"] = lpd_config.pop("output_ch")

            # Create LPD
            kwargs["lpd_model"] = self.create_generator(lpd_type, **lpd_config)
        elif lpd_config is None:
            kwargs["lpd_model"] = None

        if not self._registry.has_generator(model_type):
            raise ModelCreationError(f"Generator type '{model_type}' not registered")

        generator_class = self._registry.get_generator_class(model_type)
        return generator_class(**kwargs)

    def _handle_uncertainty_wrapper(self, model_type: str, **kwargs) -> IGenerator:
        """Handler for uncertainty wrapper to instantiate base model."""
        base_model_type = kwargs.pop("base_model_type", None)
        if not base_model_type:
            raise ModelCreationError("uncertainty_wrapper requires 'base_model_type' argument")

        # Recursively create base model
        wrapper_keys = {
            "mc_dropout_p",
            "use_deep_ensemble",
            "n_ensemble",
            "use_epistemic_uncertainty",
            "use_aleatoric_uncertainty",
        }

        wrapper_kwargs = {k: v for k, v in kwargs.items() if k in wrapper_keys}
        base_kwargs = {k: v for k, v in kwargs.items() if k not in wrapper_keys}

        base_model = self.create_generator(base_model_type, **base_kwargs)

        generator_class = self._registry.get_generator_class(model_type)
        wrapper_kwargs["base_model"] = base_model

        generator = generator_class(**wrapper_kwargs)
        return generator

    def _handle_adversarial_purification(self, model_type: str, **kwargs) -> IGenerator:
        """Handler for adversarial purification with nested diffusion model."""
        base_config = kwargs.pop("base_diffusion_config", {})
        inner_type = base_config.get("base_model_type", "conditional_diffusion_generator")
        inner_kwargs = base_config.get("base_model_kwargs", {})

        # Remap inner channels if needed
        if "inner_in_channels" in inner_kwargs:
            inner_kwargs["in_channels"] = inner_kwargs.pop("inner_in_channels")
        if "inner_out_channels" in inner_kwargs:
            inner_kwargs["out_channels"] = inner_kwargs.pop("inner_out_channels")

        inner_model = self.create_generator(inner_type, **inner_kwargs)
        kwargs["base_model"] = inner_model

        if not self._registry.has_generator(model_type):
            raise ModelCreationError(f"Generator type '{model_type}' not registered")

        generator_class = self._registry.get_generator_class(model_type)
        return generator_class(**kwargs)

    def _handle_diffusion_wrapper(self, model_type: str, **kwargs) -> IGenerator:
        """Handler for diffusion wrappers that require a denoising model."""
        denoising_config = kwargs.pop("denoising_model", None)
        if not denoising_config or not isinstance(denoising_config, dict):
            raise ModelCreationError(
                f"{model_type} requires 'denoising_model' configuration dictionary"
            )

        # Extract inner model type and kwargs
        inner_type = denoising_config.pop("type", denoising_config.pop("model_type", None))
        if not inner_type:
            raise ModelCreationError("denoising_model config must specify 'type' or 'model_type'")

        # Map input_channels/output_channels back to in_channels/out_channels
        if "input_channels" in denoising_config:
            denoising_config["in_channels"] = denoising_config.pop("input_channels")
        if "output_channels" in denoising_config:
            denoising_config["out_channels"] = denoising_config.pop("output_channels")

        # Inject top-level channels if available and remove from kwargs
        # This prevents "unexpected keyword argument" for wrapper init
        # and ensures inner model gets correct channels.
        top_in_ch = kwargs.pop("in_channels", None)
        top_out_ch = kwargs.pop("out_channels", None)

        if top_in_ch is not None and "in_channels" not in denoising_config:
            denoising_config["in_channels"] = top_in_ch
        if top_out_ch is not None and "out_channels" not in denoising_config:
            denoising_config["out_channels"] = top_out_ch

        # Recursively create inner model
        inner_model = self.create_generator(inner_type, **denoising_config)

        if not self._registry.has_generator(model_type):
            raise ModelCreationError(f"Generator type '{model_type}' not registered")

        generator_class = self._registry.get_generator_class(model_type)

        # Filter kwargs using parameter mapper to remove invalid keys
        # that were meant for the inner model or unused.
        mapped_wrapper_params = self._parameter_mapper.map_generator_params(model_type, **kwargs)
        mapped_wrapper_params["denoising_model"] = inner_model

        # [FIX] Filter against the wrapper constructor's signature so that
        # top-level config.model.* fields forwarded by GeneratorBuilder.build()
        # (base_channels, depth, spatial_dims, …) don't raise
        # "unexpected keyword argument" when the wrapper's __init__ is narrow
        # (laplace/rician_diffusion_generator only accept denoising_model,
        # timesteps, beta_schedule, device).
        mapped_wrapper_params = _filter_kwargs_to_signature(
            mapped_wrapper_params,
            generator_class,
            logger_=self._logger,
            site="spectramr.models.factories.model_factory:diffusion_wrapper",
            role="wrapper",
        )

        generator = generator_class(**mapped_wrapper_params)
        return generator

    def create_discriminator(self, model_type: str, **kwargs) -> IDiscriminator:
        """Create a discriminator instance."""
        if not self._registry.has_discriminator(model_type):
            raise ModelCreationError(f"Discriminator type '{model_type}' not registered")

        discriminator_class = self._registry.get_discriminator_class(model_type)
        # [FIX] Sanitize input kwargs by converting Pydantic models to dicts
        clean_kwargs = {
            k: v.model_dump() if hasattr(v, "model_dump") else v for k, v in kwargs.items()
        }

        mapped_params = self._parameter_mapper.map_discriminator_params(model_type, **clean_kwargs)
        self._validator.validate_discriminator_params(model_type, **mapped_params)

        # [FIX] Filter kwargs against constructor signature to prevent
        # "unexpected keyword argument" errors (same guard as generators).
        try:
            sig = inspect.signature(discriminator_class.__init__)
            params = sig.parameters
            has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
            if not has_var_keyword:
                mapped_params = _filter_kwargs_to_signature(
                    mapped_params,
                    discriminator_class,
                    logger_=self._logger,
                    site="spectramr.models.factories.model_factory:create_discriminator",
                )
        except (ValueError, TypeError):
            pass

        # [LOGGING] Trace creation
        self._logger.info(
            f"Creating Discriminator: {model_type} | Class: {discriminator_class.__name__} |Params: {mapped_params}"
        )

        discriminator = discriminator_class(**mapped_params)
        return discriminator

    def inspect_generator_contract(self, model_type: str) -> dict[str, Any]:
        """Inspect the contract/parameters for a generator type."""
        if not self._registry.has_generator(model_type):
            return {}

        generator_class = self._registry.get_generator_class(model_type)
        sig = inspect.signature(generator_class.__init__)
        return {name: param.annotation for name, param in sig.parameters.items() if name != "self"}


class DiscriminatorPairer:
    """Component for creating balanced generator-discriminator pairs."""

    def __init__(self):
        """__init__."""
        pass

    def pair_discriminator(self, generator_type: str, discriminator_type: str | None = None) -> str:
        """Pair a discriminator with a generator."""
        if discriminator_type:
            return discriminator_type
        # Default pairing logic
        return "patchgan"

    def create_model_pair(
        self, model_type: str, **kwargs
    ) -> tuple[IGenerator, IDiscriminator | None]:
        """Create a generator-discriminator pair with balanced configuration.

        For non-GAN training modes (diffusion, reconstruction, VAE), the discriminator
        is not created to avoid unnecessary memory allocation and computation.
        """
        factory = get_model_factory()

        # Only create discriminator for GAN training
        requires_discriminator = kwargs.pop("requires_discriminator", True)
        generator = factory.create_generator(model_type, **kwargs)
        if not requires_discriminator:
            return generator, None

        # Filter out generator-specific kwargs for discriminator
        disc_kwargs = {k: v for k, v in kwargs.items() if k not in {"in_channels", "out_channels"}}
        discriminator = factory.create_discriminator(
            "patchgan", in_channels=kwargs.get("out_channels", 1), **disc_kwargs
        )
        return generator, discriminator


class CompositeModelFactory:
    """Factory for creating composite models."""

    def create_composite_model(self, config, **kwargs):
        """Create a composite model."""
        raise NotImplementedError("Composite models not yet implemented in consolidated factory")


def get_composite_model_factory():
    """Get the composite model factory instance."""
    return CompositeModelFactory()


class ModelFactory:
    """Concrete factory implementation following SOLID principles for model
    creation.

    This factory provides a centralized way to create generators and
    discriminators while maintaining clean architecture principles. It
    implements the Open/Closed principle by allowing new model types to
    be registered without modifying existing code.

    The factory handles:
    - Generator creation with automatic parameter mapping
    - Discriminator creation with balanced pairing for stable GAN training
    - Model type validation and error handling
    - Singleton pattern for global access

    Attributes:
        _registry (ModelRegistry): Registry component for managing
            available generator and discriminator classes.
        _parameter_mapper (ParameterMapper): Component for mapping
            parameters between different model types.
        _model_creator (ModelCreator): Component for creating
            model instances with proper parameter handling.
        _pairer (DiscriminatorPairer): Component for creating
            balanced generator-discriminator pairs.

    Example:
        >>> factory = ModelFactory()
        >>> gen = factory.create_generator('unet', in_channels=1, out_channels=1)
        >>> disc = factory.create_discriminator('patchgan', in_channels=1)
        >>> gen, disc = factory.create_model_pair('unet',
        ...                                       in_channels=1, out_channels=1)

    Note:
        The factory automatically applies Tanh activation to generators (except
        diffusion models) to ensure outputs are in the [-1, 1] range expected
        by GAN training pipelines.

    """

    _instances: dict[tuple[str, tuple[tuple[str, Any], ...]], "ModelFactory"] = {}
    _warned_unknown: set[tuple[str, str]] = set()
    _DISCRIMINATOR_ONLY_KEYS = {
        "spectral_norm",
        "disc_filters",
        "disc_depth",
        "disc_channels",
        "ndf",
        "n_layers",
    }

    def __init__(
        self,
        registry: ModelRegistry | None = None,
        parameter_mapper: ParameterMapper | None = None,
        activation_manager: ActivationManager | None = None,
        validator: ModelValidator | None = None,
        model_creator: ModelCreator | None = None,
        discriminator_pairer: DiscriminatorPairer | None = None,
        **kwargs: Any,
    ):
        """Initialize the ModelFactory with SOLID components.

        This constructor uses dependency injection to set up the factory's
        internal components, promoting loose coupling and testability.

        Constructing a factory is **not** deprecated. ``create_generator`` is
        the primitive the canonical builder chain itself calls
        (``GeneratorBuilder.build``), and model-internal composers reach it
        because ``check_layering.sh`` forbids ``models/ -> infrastructure/``.
        The deprecation lives on :meth:`create_model`, which is the surface
        being retired -- see the note there.
        """
        strict_kwargs_default = kwargs.pop("strict_kwargs_default", False)
        if kwargs:
            # ``**kwargs`` exists for the one legacy key above; anything else is
            # a misspelt or retired option (``unknown_model_policy`` was one),
            # and swallowing it is the silent fallback non-negotiable 3 forbids.
            raise TypeError(f"ModelFactory got unexpected keyword argument(s): {sorted(kwargs)}")

        # Initialize SOLID components using composition (with DI support)
        # Extract components from parameters or use defaults
        registry = registry or ModelRegistry()
        parameter_mapper = parameter_mapper or ParameterMapper()
        activation_manager = activation_manager or ActivationManager()
        validator = validator or ModelValidator(registry)

        self._registry = registry
        self._parameter_mapper = parameter_mapper
        self._activation_manager = activation_manager
        self._model_creator = model_creator or ModelCreator(
            registry=registry,
            parameter_mapper=parameter_mapper,
            validator=validator,
            activation_manager=activation_manager,
        )
        self._pairer = discriminator_pairer or DiscriminatorPairer()

        self._strict_kwargs_default = bool(strict_kwargs_default)
        # LAZY IMPORT: Avoid circular import with loss_cache
        from spectramr.infrastructure.training.loss_cache import LossModuleCache

        self._loss_cache = LossModuleCache.get_instance()
        self._available_loss_functions = [
            "l1",
            "mse",
            "smooth_l1",
            "bce",
            "bce_with_logits",
            "cross_entropy",
            "huber",
        ]

        self._adapter_ready_models: set[str] = {
            "x_diffusion_latent",
            "x_diffusion",
            "trellis",
        }

        # Initialize logger
        self._logger = logging.getLogger(__name__)

        # Initialize composite model factory lazily
        self._composite_factory = None

        # Register all available models
        self._register_all_models()

    @classmethod
    def create(cls, config: Any, **kwargs: Any) -> Any:
        """Static convenience method to create a model instance from config.

        This allows creating models without manually instantiating the factory
        in simple use cases like inference or standalone scripts.

        Args:
            config: Configuration object or dictionary.
            **kwargs: Extra arguments for model creation. ``model_type``
                overrides ``config.model.model_type``; everything else is
                merged over the fields read off ``config.model``.

        Returns:
            Model instance.

        Raises:
            TypeError: If ``config`` exposes no ``model`` block.
        """
        # ``cls(**kwargs)`` fed the CALLER'S model kwargs into the factory's own
        # dependency-injection constructor, where ``**kwargs: Any`` absorbed them
        # silently, and then passed the same kwargs on to ``create_generator``.
        # The factory takes no model arguments; construct it bare (#1511).
        instance = cls()

        # Extract creation params from config if possible
        creation_kwargs: dict[str, Any] = {}
        model_config = config.model
        if isinstance(model_config, dict):
            creation_kwargs.update(model_config)
        elif hasattr(model_config, "__dict__"):
            creation_kwargs.update(model_config.__dict__)

        # Override with explicit kwargs
        creation_kwargs.update(kwargs)

        # ``model_type`` is a POSITIONAL parameter of ``create_generator`` and is
        # also a field of ``config.model``, so splatting the merged mapping
        # passed it twice: this method raised
        # ``TypeError: create_generator() got multiple values for argument
        # 'model_type'`` on EVERY call, for every model, since it was written --
        # it has no callers, which is the only reason that went unnoticed
        # (#1511, sibling of #1277). Popped rather than dropped so an explicit
        # ``model_type=`` override is honoured instead of silently discarded
        # (non-negotiable 3).
        model_type = creation_kwargs.pop("model_type", None)
        if model_type is None:
            raise TypeError(
                "ModelFactory.create() needs a model type: config.model carries "
                "no 'model_type' and none was passed as a keyword argument."
            )

        return instance.create_generator(model_type, **creation_kwargs)

    def _register_all_models(self) -> None:
        """Register all available generator and discriminator models.

        Split into tiered import blocks to prevent single
        dependency failure from crashing entire model registry.
        """

        # --- Tier 1: Core Models (MUST succeed) ---
        # These are zero-dependency, essential models. Failure here is fatal.
        try:
            from ..discriminators.patchgan_discriminator import PatchGANDiscriminator
            from ..generators import (
                AttentionUNetGenerator,
                EnhancedUNetGenerator,
                StandardUNetGenerator,
                UNetGenerator,
            )
            from ..generators.convolution_variants import EnhancedUNet

            self._registry.register_generator("standard_unet", StandardUNetGenerator)
            self._registry.register_generator("unet", UNetGenerator)
            # ``EnhancedUNetGenerator`` IS ``UNet`` -- an import-compat alias
            # (reconstruction/unet.py:1177), not a distinct architecture. Kept
            # because 11 arms declare ``enhanced_unet`` and build this today.
            self._registry.register_generator("enhanced_unet", EnhancedUNetGenerator)
            # The squeeze-excitation UNet that used to claim ``enhanced_unet``
            # in MODEL_REGISTRY and was therefore buildable from nowhere (#977).
            self._registry.register_generator("se_residual_unet", EnhancedUNet)
            self._registry.register_generator("attention_unet", AttentionUNetGenerator)
            self._registry.register_generator("unet_attention", AttentionUNetGenerator)
            self._registry.register_discriminator("patchgan", PatchGANDiscriminator)
            # Alias: YAML configs often use 'patch_gan_discriminator' (underscore-separated)
            # while @register_model uses 'patchgan_discriminator' (no underscore)
            self._registry.register_discriminator("patch_gan_discriminator", PatchGANDiscriminator)

        except ImportError as e:
            self._logger.error(f"CRITICAL: Failed to load Core Models: {e}")
            raise  # Core models must load

        # --- Tier 2: Extended Models (Should succeed, warn on failure) ---
        self._register_extended_models()

        # --- Tier 3: Experimental/Optional Models (May fail silently) ---
        self._register_optional_models()

        # --- Tier 4: Paradigm Shifts (Phase 15/16) ---
        self._register_paradigm_shifts()

    def _register_paradigm_shifts(self) -> None:
        """Register paradigm shift models (Phase 15/16)."""

        # 1. Mamba Models (Critical for Reconstruction)
        try:
            from spectramr.models.generators.mamba_unet import (
                MambaReconstruction,
                MambaUNet,
            )

            self._registry.register_generator("mamba_reconstruction", MambaReconstruction)
            self._registry.register_generator("mambareconstruction", MambaReconstruction)
            self._registry.register_generator("mamba_unet", MambaUNet)
        except ImportError as e:
            self._logger.warning(f"Skipping Mamba models: {e}")

        # 2. Experimental Physics/Geometric Models (isolated blocks)
        try:
            from spectramr.models.experimental.implicit_kan import (
                ImplicitKAN,
                ImplicitMRIField,
            )

            self._registry.register_generator("implicit_kan", ImplicitKAN)
            self._registry.register_generator("implicit_mri_field", ImplicitMRIField)
        except ImportError as e:
            self._logger.debug(f"Skipping ImplicitKAN models: {e}")

        try:
            from spectramr.models.gaussian_splatting.medgs import MedGS

            self._registry.register_generator("gaussian_splatting", MedGS)
        except ImportError as e:
            self._logger.debug(f"Skipping gaussian_splatting alias: {e}")

        # 3. Advanced Generative Models (isolated blocks)
        try:
            from spectramr.models.experimental.clifford_gan import CliffordGANGenerator
            from spectramr.models.experimental.mk_recon import MKRecon
            from spectramr.models.experimental.world_model import GenerativeWorldModel

            self._registry.register_generator("mk_recon", MKRecon)
            self._registry.register_generator("clifford_gan", CliffordGANGenerator)
            self._registry.register_generator("world_model", GenerativeWorldModel)
        except ImportError as e:
            self._logger.debug(f"Skipping CliffordGAN/MKRecon/WorldModel: {e}")

        try:
            from spectramr.models.experimental.quantum_mri import QuantumUNet

            self._registry.register_generator("quantum_unet", QuantumUNet)
        except ImportError as e:
            self._logger.debug(f"Skipping QuantumUNet (no source): {e}")

        # 4. RL & Hardware Simulation
        try:
            from spectramr.models.active_learning.rl_scanner import DeepQScannerAgent
            from spectramr.models.hardware_simulation.photonic_fft import PhotonicFFT2d

            self._registry.register_generator("rl_scanner", DeepQScannerAgent)
            self._registry.register_generator("photonic_fft", PhotonicFFT2d)
        except ImportError as e:
            self._logger.warning(f"Skipping RL/Hardware models: {e}")

        # 5. Models with source code that were previously unregistered
        try:
            from spectramr.models.generators.cycle_gan import CycleGAN

            self._registry.register_generator("cycle_gan", CycleGAN)
        except ImportError as e:
            self._logger.debug(f"Skipping CycleGAN: {e}")

        try:
            from spectramr.models.generators.equivariant_diffusion_generator import (
                EquivariantDiffusionGenerator,
            )

            self._registry.register_generator(
                "equivariant_diffusion_generator", EquivariantDiffusionGenerator
            )
        except ImportError as e:
            self._logger.debug(f"Skipping EquivariantDiffusionGenerator: {e}")

        try:
            from spectramr.models.inr.motion import DualMotionINR

            self._registry.register_generator("motion_inr", DualMotionINR)
        except ImportError as e:
            self._logger.debug(f"Skipping DualMotionINR: {e}")

        try:
            from spectramr.models.generators.trellis_structured_vae import (
                TrellisStructuredVAEBase,
            )

            self._registry.register_generator("structured_vae", TrellisStructuredVAEBase)
        except ImportError as e:
            self._logger.debug(f"Skipping TrellisStructuredVAE: {e}")

        try:
            from spectramr.models.generators.trellis_generator import TRELLISGenerator

            self._registry.register_generator("trellis", TRELLISGenerator)
        except ImportError as e:
            self._logger.debug(f"Skipping TRELLISGenerator: {e}")

    def _register_extended_models(self) -> None:
        """Register extended models with isolated failure handling."""

        # 2a. Super-Resolution Models
        try:
            from ..generators import (
                DCSRNGenerator,
                EDSRGenerator,
                EfficientTransformerGenerator,
                ELANGenerator,
                HANGenerator,
                IMDNGenerator,
                NAFNetGenerator,
                RCANGenerator,
                RestormerGenerator,
                SwinIRGenerator,
            )

            mappings = {
                "edsr": EDSRGenerator,
                "edsr_generator": EDSRGenerator,
                "rcan": RCANGenerator,
                "rcan_generator": RCANGenerator,
                "swin_ir": SwinIRGenerator,
                "swinir_generator": SwinIRGenerator,
                "han": HANGenerator,
                "han_generator": HANGenerator,
                "nafnet": NAFNetGenerator,
                "nafnet_generator": NAFNetGenerator,
                "restormer": RestormerGenerator,
                "elan": ELANGenerator,
                "elan_generator": ELANGenerator,
                "imdn": IMDNGenerator,
                "dcsrn": DCSRNGenerator,
                "efficient_transformer": EfficientTransformerGenerator,
            }
            for name, cls in mappings.items():
                self._registry.register_generator(name, cls)
        except ImportError as e:
            self._logger.warning(f"Skipping Super-Resolution models: {e}")

        # 2b. Diffusion Models
        try:
            from ..generators import (
                AdversarialPurificationDiffusion,
                ChiSquareDiffusionGenerator,
                ColdDiffusionGenerator,
                ConditionalDiffusionGenerator,
                DiffusionSRGenerator,
                EquivariantDiffusionGenerator,
                HybridTransformerDiffusionGenerator,
                KSpaceColdDiffusionGenerator,
                LaplaceDiffusionGenerator,
                LatentDiffusionGenerator,
                LatentGaussianDiffusion,
                RicianDiffusionGenerator,
                ScoreBasedDiffusionGenerator,
            )

            mappings = {
                "diffusion_sr": DiffusionSRGenerator,
                "cold_diffusion": ColdDiffusionGenerator,
                "kspace_cold_diffusion": KSpaceColdDiffusionGenerator,
                "kspace_cold_diffusion_generator": KSpaceColdDiffusionGenerator,
                "latent_diffusion": LatentDiffusionGenerator,
                "latent_gaussian_diffusion": LatentGaussianDiffusion,
                "conditional_diffusion": ConditionalDiffusionGenerator,
                "conditional_diffusion_generator": ConditionalDiffusionGenerator,
                "score_based_diffusion": ScoreBasedDiffusionGenerator,
                "score_based_diffusion_generator": ScoreBasedDiffusionGenerator,
                "laplace_diffusion": LaplaceDiffusionGenerator,
                "laplace_diffusion_generator": LaplaceDiffusionGenerator,
                "rician_diffusion": RicianDiffusionGenerator,
                "rician_diffusion_generator": RicianDiffusionGenerator,
                "chi_square_diffusion": ChiSquareDiffusionGenerator,
                "hybrid_transformer_diffusion": HybridTransformerDiffusionGenerator,
                "equivariant_diffusion": EquivariantDiffusionGenerator,
                "equivariant_unet": EquivariantDiffusionGenerator,  # Alias for config compatibility
                "adversarial_purification": AdversarialPurificationDiffusion,
            }
            for name, cls in mappings.items():
                self._registry.register_generator(name, cls)
        except ImportError as e:
            self._logger.warning(f"Skipping Diffusion models: {e}")

        # 2a-extended: KAN and Enhanced Architecture Variants
        try:
            from ..diffusion.architectures import RefinedKANUNet

            # Register KAN variants with multiple names for compatibility
            self._registry.register_generator("kan_unet", RefinedKANUNet)
            self._registry.register_generator("enhanced_kan_unet", RefinedKANUNet)
            self._registry.register_generator("refined_kan_unet", RefinedKANUNet)
            self._registry.register_generator("kan_diffusion", RefinedKANUNet)
        except ImportError as e:
            self._logger.debug(f"Skipping KAN U-Net variants: {e}")

        # 2c. VAE/VQ Models
        try:
            from spectramr.models.vae.disentangled_vae import PhysicsConditionedVAE

            from ..generators import (
                SparseVAEGenerator,
                VAE3DGenerator,
                VAEGenerator,
                VQGANGenerator,
                VQVAEGenerator,
            )

            mappings = {
                "vae": VAEGenerator,
                "vae_3d": VAE3DGenerator,
                "vqgan": VQGANGenerator,
                "vqvae": VQVAEGenerator,
                "vq_vae": VQVAEGenerator,
                "sparse_vae": SparseVAEGenerator,
                "disentangled_vae": PhysicsConditionedVAE,
                "physics_vae": PhysicsConditionedVAE,
            }
            for name, cls in mappings.items():
                self._registry.register_generator(name, cls)
        except ImportError as e:
            self._logger.warning(f"Skipping VAE/VQ models: {e}")

        # 2d. Standard Transformer Models (NOT KAN-based)
        try:
            from ..generators import SwinTransformerGenerator, VisionTransformer

            mappings = {
                "vit": VisionTransformer,  # Standard ViT
                "vision_transformer": VisionTransformer,
                "swin": SwinTransformerGenerator,  # Standard Swin
                "swin_transformer": SwinTransformerGenerator,
            }
            for name, cls in mappings.items():
                self._registry.register_generator(name, cls)
        except ImportError as e:
            self._logger.warning(f"Skipping Standard Transformer models: {e}")

        # 2e. KAN/Transformer Models (KAN-based variants)
        try:
            from ..generators import KANGenerator, ViTKANGenerator

            mappings = {
                "kan": KANGenerator,
                "kan_gan": KANGenerator,  # Reuse KANGenerator for gan mode
                "vit_kan": ViTKANGenerator,
            }
            for name, cls in mappings.items():
                self._registry.register_generator(name, cls)
        except ImportError as e:
            self._logger.warning(f"Skipping KAN/Transformer models: {e}")

        # 2e. Physics/Medical Models
        try:
            from ..generators import (
                CoilSensitivityNetwork,
                EnhancedDeepUNetGenerator,
                MultiContrastFusionGenerator,
                PhysicsDrivenNetwork,
                UnrolledReconstructionGenerator,
            )

            # Import B0Estimator directly from its module to ensure registration
            from ..geometric.b0_estimator import B0Estimator

            mappings = {
                "unrolled_reconstruction": UnrolledReconstructionGenerator,
                "coil_sensitivity_network": CoilSensitivityNetwork,
                "physics_driven_network": PhysicsDrivenNetwork,
                "multi_contrast_fusion": MultiContrastFusionGenerator,
                "multicontrast_fusion": MultiContrastFusionGenerator,
                "multi_contrast_fusion_generator": MultiContrastFusionGenerator,
                "fusion_unet": MultiContrastFusionGenerator,
                "enhanced_deep_unet": EnhancedDeepUNetGenerator,
                "diffusion_unet": EnhancedDeepUNetGenerator,
                "b0_estimator": B0Estimator,
            }
            for name, cls in mappings.items():
                self._registry.register_generator(name, cls)
        except ImportError as e:
            self._logger.warning(f"Skipping Physics/Medical models: {e}")

        # 2f. 3D/Volume Models
        try:
            from ..generators import (
                AnisotropicVoxelGenerator,
                SliceToVolumeGenerator,
                UNet2_5DGenerator,
                UNet3DGenerator,
                UNETRGenerator,
                VNetGenerator,
            )

            mappings = {
                "unet3d": UNet3DGenerator,
                "unet_2_5d": UNet2_5DGenerator,
                "slice_to_volume": SliceToVolumeGenerator,
                "slice_to_volume_generator": SliceToVolumeGenerator,
                "anisotropic_voxel_generator": AnisotropicVoxelGenerator,
                "unetr_generator": UNETRGenerator,
                "vnet_generator": VNetGenerator,
            }
            for name, cls in mappings.items():
                self._registry.register_generator(name, cls)
        except ImportError as e:
            self._logger.warning(f"Skipping 3D/Volume models: {e}")

    def _register_optional_models(self) -> None:
        """Register experimental/optional models with graceful failure."""

        # 3a. Advanced Research Models - ISOLATED REGISTRATION
        # Each model is registered separately to prevent one failure from blocking others

        # GraphUNetGenerator - commonly used
        try:
            from ..generators import GraphUNetGenerator

            self._registry.register_generator("graph_unet", GraphUNetGenerator)
        except ImportError as e:
            self._logger.debug(f"Skipping GraphUNetGenerator: {e}")

        # FNO Generator
        try:
            from ..generators import FNOGenerator

            self._registry.register_generator("fno", FNOGenerator)
        except ImportError as e:
            self._logger.debug(f"Skipping FNOGenerator: {e}")

        # NCA Generator
        try:
            from ..generators import NCAGenerator

            self._registry.register_generator("nca_generator", NCAGenerator)
        except ImportError as e:
            self._logger.debug(f"Skipping NCAGenerator: {e}")

        # Neural ODE
        try:
            from ..generators import NeuralODEGenerator

            self._registry.register_generator("neural_ode", NeuralODEGenerator)
        except ImportError as e:
            self._logger.debug(f"Skipping NeuralODEGenerator: {e}")

        # GFlowNet
        try:
            from ..generators import GFlowNetGenerator

            self._registry.register_generator("gflownet_generator", GFlowNetGenerator)
        except ImportError as e:
            self._logger.debug(f"Skipping GFlowNetGenerator: {e}")

        # KSpace GPT
        try:
            from ..generators import KSpaceGPT

            self._registry.register_generator("kspace_gpt", KSpaceGPT)
        except ImportError as e:
            self._logger.debug(f"Skipping KSpaceGPT: {e}")

        # Trellis VAE variants
        try:
            from ..generators import TrellisStructuredGaussianVAEGenerator

            self._registry.register_generator(
                "trellis_gaussian_vae", TrellisStructuredGaussianVAEGenerator
            )
        except ImportError as e:
            self._logger.debug(f"Skipping TrellisStructuredGaussianVAEGenerator: {e}")

        try:
            from ..generators.trellis_structured_vae import (
                TrellisStructuredMeshVAEGenerator,
            )

            self._registry.register_generator("trellis_mesh_vae", TrellisStructuredMeshVAEGenerator)
        except ImportError as e:
            self._logger.debug(f"Skipping TrellisStructuredMeshVAEGenerator: {e}")

        # Optional: Mamba and MoE (may not be exported)
        try:
            from ..generators import MambaGenerator

            self._registry.register_generator("mamba", MambaGenerator)
        except ImportError as e:
            self._logger.debug(f"Skipping MambaGenerator: {e}")

        try:
            from ..generators import MoEKANGenerator

            self._registry.register_generator("moe_kan_generator", MoEKANGenerator)
        except ImportError as e:
            self._logger.debug(f"Skipping MoEKANGenerator: {e}")

        # Hilbert-Mamba ULF-to-HF reconstruction generators
        try:
            from ..generators.hilbert_mamba_generators import (
                CRMMamba,
                CTMamba,
                FEMamba,
                HilbertMambaUNet,
                HLDMamba,
                HWMamba,
                INRMamba,
                MDIMamba,
                MMMamba,
                TwoHalfDMamba,
            )

            hm_mappings = {
                "ct_mamba": CTMamba,
                "fe_mamba": FEMamba,
                "crm_mamba": CRMMamba,
                "hw_mamba": HWMamba,
                "mm_mamba": MMMamba,
                "mdi_mamba": MDIMamba,
                "two_half_d_mamba": TwoHalfDMamba,
                "hld_mamba": HLDMamba,
                "inr_mamba": INRMamba,
                "hilbert_mamba_unet": HilbertMambaUNet,
            }
            for name, cls in hm_mappings.items():
                self._registry.register_generator(name, cls)
        except ImportError as e:
            self._logger.debug(f"Skipping Hilbert-Mamba generators: {e}")

        # Rectified Flow Generator (Measurement-Conditional Flow Matching)
        try:
            from ..generators.rectified_flow_generator import RectifiedFlowGenerator

            self._registry.register_generator("rectified_flow", RectifiedFlowGenerator)
        except ImportError as e:
            self._logger.debug(f"Skipping RectifiedFlowGenerator: {e}")

        # Consistency Model Generator (1-2 step fast inference)
        try:
            from ..generators.consistency_model_generator import (
                ConsistencyModelGenerator,
            )

            self._registry.register_generator("consistency_model", ConsistencyModelGenerator)
        except ImportError as e:
            self._logger.debug(f"Skipping ConsistencyModelGenerator: {e}")

        # 3b. External Dependency Models (transformers, etc.)
        try:
            from ..generators import (
                CycleSRGenerator,
                FoundationModel,
                HypernetworkGenerator,
                HyperspectralGAN,
            )

            mappings = {
                "foundation_model": FoundationModel,
                "cyclesr": CycleSRGenerator,
                "hypernetwork": HypernetworkGenerator,
                "hyperspectral_gan": HyperspectralGAN,
            }
            for name, cls in mappings.items():
                self._registry.register_generator(name, cls)
        except ImportError as e:
            self._logger.debug(f"Skipping Foundation/External models: {e}")

        # 3c. Domain-Specific Models - ISOLATED CRITICAL MODELS FIRST
        # These commonly-used models are registered individually to prevent blocking

        try:
            from ..generators import PINNeRF

            self._registry.register_generator("pin_nerf", PINNeRF)
            self._registry.register_generator("pinn_nerf", PINNeRF)  # Alias
        except ImportError as e:
            self._logger.debug(f"Skipping PINNeRF: {e}")

        try:
            from ..generators import BlochCycleNetwork

            self._registry.register_generator("bloch_cycle_network", BlochCycleNetwork)
        except ImportError as e:
            self._logger.debug(f"Skipping BlochCycleNetwork: {e}")

        try:
            from ..generators import DeepImagePriorGenerator

            self._registry.register_generator("deep_image_prior", DeepImagePriorGenerator)
        except ImportError as e:
            self._logger.debug(f"Skipping DeepImagePriorGenerator: {e}")

        try:
            from ..generators import GNNCoilFusion

            self._registry.register_generator("gnn_coil_fusion", GNNCoilFusion)
        except ImportError as e:
            self._logger.debug(f"Skipping GNNCoilFusion: {e}")

        try:
            from ..generators import ScannerInvariantEmbedding

            self._registry.register_generator(
                "scanner_invariant_embedding", ScannerInvariantEmbedding
            )
        except ImportError as e:
            self._logger.debug(f"Skipping ScannerInvariantEmbedding: {e}")

        # DisentangledMRI - dedicated import to ensure it's registered even if other imports fail
        try:
            from ..generators.disentangled_mri import DisentangledMRI

            self._registry.register_generator("disentangled_mri", DisentangledMRI)
            self._registry.register_generator("disentangled_vae", DisentangledMRI)  # Alias
        except ImportError as e:
            self._logger.debug(f"Skipping DisentangledMRI: {e}")

        # B0Estimator - dedicated import
        try:
            from spectramr.models.geometric.b0_estimator import B0Estimator

            self._registry.register_generator("b0_estimator", B0Estimator)
            self._registry.register_generator("b0_mapping", B0Estimator)
        except ImportError as e:
            self._logger.debug(f"Skipping B0Estimator: {e}")

        # Grouped import for less commonly used models (single failure won't block others above)
        try:
            from ..generators import PIMN

            self._registry.register_generator("pimn", PIMN)
        except ImportError as e:
            self._logger.debug(f"Skipping PIMN: {e}")

        # Per-class registration: one missing import must not silently
        # drop a dozen unrelated models. Audit TODO/audit/02_application.md
        # F2 found the previous monolithic `try/except ImportError`
        # swallowing 16 registrations because the agents lived in
        # `src/application/agents/` (canonical home is now
        # `src/models/generators/` after the D14 move — they
        # self-register via `@register_model` and no longer need a
        # special factory entry). Combined with the
        # `stubs.py:rl_acquisition_agent → rl_mri` alias that silently
        # remapped the missing names to an unrelated model
        # (CLAUDE.md pitfall #9 — silent fallback × identity collision).
        domain_specific_mappings: list[tuple[str, str, str]] = [
            ("disentangled_mri", "..generators", "DisentangledMRI"),
            ("disentangled_vae", "..generators", "DisentangledMRI"),
            ("nesvor", "..generators", "NeSVoR"),
            ("padnet", "..generators", "PaDNet"),
            ("diffdeur", "..generators", "DiffDeuR"),
            ("qspace_translation", "..generators", "qSpaceTranslation"),
            ("digital_twin_simulator", "..generators", "DigitalTwinSimulator"),
            ("reflexion_reconstruction", "..generators", "ReflexionReconstruction"),
            ("continual_learning_unet", "..generators", "ContinualLearningUNet"),
            ("spiking_unet", "..generators", "SpikingUNet"),
            ("symbolic_regression", "..generators", "SymbolicRegressionWrapper"),
            (
                "uncertainty_wrapper",
                "spectramr.models.generators.uncertainty_wrappers",
                "EnhancedUncertaintyWrapper",
            ),
            (
                "split_client_encoder",
                "spectramr.models.generators.split_learning_unet",
                "SplitClientEncoder",
            ),
            (
                "split_server_decoder",
                "spectramr.models.generators.split_learning_unet",
                "SplitServerDecoder",
            ),
            # NB: ``rl_acquisition_agent`` and ``surgery_loop_agent``
            # used to be registered here from ``spectramr.application.agents``
            # (a CLAUDE.md layer-direction violation: models → application).
            # The 2026-05-15 D14 follow-up moved them to
            # ``src/models/generators/`` where they now self-register via
            # ``@register_model``. They are still imported below for the
            # legacy ``MODEL_REGISTRY[...]`` fast-path but no longer need
            # the explicit factory entry.
        ]
        for name, module_path, attr in domain_specific_mappings:
            try:
                module = importlib.import_module(module_path, package=__package__)
                cls = getattr(module, attr)
                self._registry.register_generator(name, cls)
            except ImportError as e:
                self._logger.debug("Skipping %s (%s): %s", name, attr, e)
            except AttributeError as e:
                self._logger.debug(
                    "Skipping %s (%s missing in %s): %s",
                    name,
                    attr,
                    module_path,
                    e,
                )

        # 3d. XDiffusion and Adapters
        try:
            from spectramr.models.generators.hobs_generator import HoBSGenerator

            self._registry.register_generator("hobs_generator", HoBSGenerator)
            self._registry.register_generator("hobs", HoBSGenerator)
        except ImportError as e:
            self._logger.debug(f"Skipping HoBS: {e}")

        try:
            from spectramr.models.generators.lans_generator import LaNSGenerator

            self._registry.register_generator("lans_generator", LaNSGenerator)
            self._registry.register_generator("lans", LaNSGenerator)
        except ImportError as e:
            self._logger.debug(f"Skipping LaNS: {e}")

        try:
            from spectramr.models.gaussian_splatting.medgs import MedGS

            self._registry.register_generator("medgs", MedGS)
            self._registry.register_generator("physics_aware_gaussian_splatter", MedGS)
        except ImportError as e:
            self._logger.debug(f"Skipping MedGS: {e}")

        try:
            from spectramr.models.generators.score_based_unet_generator import (
                ScoreBasedUNetGenerator,
            )

            self._registry.register_generator("score_based_unet", ScoreBasedUNetGenerator)
        except ImportError as e:
            self._logger.debug(f"Skipping ScoreBasedUNet: {e}")

        try:
            from spectramr.models.generators.stable_diffusion_adapter_generator import (
                StableDiffusionAdapterGenerator,
            )

            self._registry.register_generator(
                "stable_diffusion_adapter", StableDiffusionAdapterGenerator
            )
        except ImportError as e:
            self._logger.debug(f"Skipping StableDiffusionAdapter: {e}")

        try:
            from spectramr.models.generators.x_diffusion_generator import (
                XDiffusionGenerator,
            )

            self._registry.register_generator("x_diffusion", XDiffusionGenerator)
        except ImportError as e:
            self._logger.debug(f"Skipping XDiffusion: {e}")

        try:
            from spectramr.models.generators.x_diffusion_latent_generator import (
                XDiffusionLatentGenerator,
            )

            self._registry.register_generator("x_diffusion_latent", XDiffusionLatentGenerator)
        except ImportError as e:
            self._logger.debug(f"Skipping XDiffusionLatent: {e}")

        # 3e. MAE and Encoders
        try:
            from ..mae import MAEGenerator

            self._registry.register_generator("mae", MAEGenerator)
        except ImportError as e:
            self._logger.debug(f"Skipping MAE models: {e}")

        try:
            from ..encoders.medical_dino_encoder import MedicalDINOv2Encoder

            self._registry.register_generator("medical_dino", MedicalDINOv2Encoder)
        except ImportError as e:
            self._logger.debug(f"Skipping MedicalDINO: {e}")

        try:
            from ..transformer.stable_transformers import StableViTWithSkipConnections

            self._registry.register_generator("stable_vit", StableViTWithSkipConnections)
        except ImportError as e:
            self._logger.debug(f"Skipping Stable ViT models: {e}")

        # 3e.1. Missing inprogress models
        try:
            from spectramr.models.generators.hybrid_pinn import HybridQMapGenerator

            self._registry.register_generator("hybrid_qmap_generator", HybridQMapGenerator)
            self._registry.register_generator("hybrid_pinn", HybridQMapGenerator)
        except ImportError as e:
            self._logger.debug(f"Skipping HybridQMapGenerator: {e}")

        try:
            from spectramr.models.reconstruction.zerorf import ZeroRFReconstructor

            self._registry.register_generator("zerorf", ZeroRFReconstructor)
            # ``zero_shot_transfer`` retired 2026-08-12 (diffusion cleanup, phase
            # 1.3). It was a second name for this class here, while
            # init_registry bound the SAME name to RectifiedFlowGenerator — one
            # name, two classes, resolved by whichever registry the caller asked.
            # Zero arms declared it; experiment_88_zero_shot_transfer.yaml uses
            # ``zerorf``.
        except ImportError as e:
            self._logger.debug(f"Skipping ZeroRF: {e}")

        # 3f. Pipeline Models
        try:
            from spectramr.application.pipelines.pggs_pipeline import (
                PGGSReconstructionPipeline,
            )

            self._registry.register_generator("pggs_pipeline", PGGSReconstructionPipeline)
        except ImportError as e:
            self._logger.debug(f"Skipping Pipeline models: {e}")

        # 3g. Paradigm Shift Models (Phase 15)
        try:
            from spectramr.models.active_learning.rl_scanner import DeepQScannerAgent
            from spectramr.models.experimental.clifford_gan import CliffordGANGenerator
            from spectramr.models.experimental.mk_recon import MKRecon
            from spectramr.models.experimental.world_model import GenerativeWorldModel
            from spectramr.models.geometric.b0_estimator import B0Estimator

            self._registry.register_generator("world_model", GenerativeWorldModel)
            self._registry.register_generator("mk_recon", MKRecon)
            self._registry.register_generator("clifford_gan", CliffordGANGenerator)
            self._registry.register_generator("rl_scanner", DeepQScannerAgent)
            self._registry.register_generator("b0_estimator", B0Estimator)
        except ImportError as e:
            self._logger.debug(f"Skipping Paradigm Shift models: {e}")

    def get_available_types(self) -> dict[str, list[str]]:
        """Get list of available generator and discriminator types.

        Returns:
            dict[str, list]: Dictionary with 'generators' and 'discriminators'
                keys containing lists of available model type names.

        Example:
            >>> types = factory.get_available_types()
            >>> print(types['generators'])
            ['unet', 'standard_unet', 'enhanced_unet', ...]

        """
        return {
            "generators": self._registry.get_available_generators(),
            "discriminators": self._registry.get_available_discriminators(),
        }

    def _create_registered_model(
        self,
        kind: str,
        requested_type: str,
        has_registered: Any,
        fetch_class: Any,
        kwargs: dict[str, Any],
    ) -> tuple[str, Any]:
        """Generic helper to create a registered model instance."""
        if not has_registered(requested_type):
            # Try fallback logic if applicable (omitted for now to match basic behavior)
            raise ModelCreationError(f"{kind.capitalize()} type '{requested_type}' not registered")

        model_class = fetch_class(requested_type)
        # Instantiate directly (assuming kwargs are pre-validated/mapped or passed raw)
        return requested_type, model_class(**kwargs)

    def create_generator(self, model_type: str, **kwargs) -> IGenerator:
        """Create a generator instance."""
        if self._model_creator is None:
            raise ValueError("Model creator is not initialized")
        return self._model_creator.create_generator(model_type, **kwargs)

    def create_discriminator(self, model_type: str, **kwargs) -> IDiscriminator:
        """Create a discriminator instance."""
        if self._model_creator is None:
            raise ValueError("Model creator is not initialized")
        return self._model_creator.create_discriminator(model_type, **kwargs)

    def register_generator(
        self,
        name: str,
        generator_class: type[IGenerator],
    ) -> None:
        """Register a new generator type with the factory.

        This method implements the Open/Closed principle by allowing new
        generator types to be added without modifying the factory code.
        It delegates to the ModelRegistry for proper separation of concerns.

        Args:
            name (str): Name to register the generator under.
            generator_class (Type[IGenerator]): Generator class that implements
                the IGenerator interface.

        Raises:
            TypeError: If the generator_class doesn't implement IGenerator.

        Example:
            >>> factory.register_generator('my_generator', MyGeneratorClass)

        """
        self._registry.register_generator(name, generator_class)

    def register_discriminator(
        self,
        name: str,
        discriminator_class: type[IDiscriminator],
    ) -> None:
        """Register a new discriminator type with the factory.

        This method implements the Open/Closed principle by allowing new
        discriminator types to be added without modifying the factory code.
        It delegates to the ModelRegistry for proper separation of concerns.

        Args:
            name (str): Name to register the discriminator under.
            discriminator_class (Type[IDiscriminator]): Discriminator class
                that implements the IDiscriminator interface.

        Raises:
            TypeError: If the discriminator_class doesn't implement
                IDiscriminator.

        Example:
            >>> factory.register_discriminator('my_discriminator',
            ...                                MyDiscriminatorClass)

        """
        self._registry.register_discriminator(name, discriminator_class)

    def inspect_generator_contract(
        self,
        model_type: str,
    ) -> tuple[set[str], bool]:
        """inspect_generator_contract.

        Args:
            model_type (str): Description.
        Returns:
            tuple[set[str], bool]: Description.
        """
        if self._model_creator is None:
            raise ValueError("Model creator is not initialized")
        return self._model_creator.inspect_generator_contract(model_type)

    def create_model_pair(
        self,
        model_type: str,
        **kwargs,
    ) -> tuple[IGenerator, IDiscriminator]:
        """Create a generator-discriminator pair with balanced configuration.

        This method creates both a generator and discriminator, automatically
        selecting an appropriate discriminator type based on the generator's
        complexity to ensure stable GAN training. It delegates to the
        DiscriminatorPairer for proper separation of concerns.
        """

        # Delegate to DiscriminatorPairer for balanced pairing, or fallback to
        # simple pairing
        if self._pairer is not None:
            return self._pairer.create_model_pair(model_type, **kwargs)
        # Simple fallback: create generator and default discriminator
        generator = self.create_generator(model_type, **kwargs)
        discriminator = self.create_discriminator(
            "patchgan",
            in_channels=kwargs.get("out_channels", 1),
        )
        return generator, discriminator

    def create_encoder(self, model_type: str, **kwargs) -> IModel:
        """Create an encoder instance of the specified type.

        This method creates encoder models for latent space learning,
        such as VAE encoders and GAN encoders.

        Args:
            model_type (str): Type of encoder to create
                (e.g., 'vae_encoder', 'latent_gan_encoder').
            **kwargs: Model-specific parameters (latent_dim, in_channels,
                etc.).

        Returns:
            IModel: Instantiated encoder implementing the IModel interface.

        Raises:
            ValueError: If the model_type is not registered or invalid.

        Example:
            >>> encoder = factory.create_encoder('vae_encoder',
            ...                                  latent_dim=128)

        """
        _, encoder = self._create_registered_model(
            kind="encoder",
            requested_type=model_type,
            has_registered=self._registry.has_generator,
            fetch_class=self._registry.get_generator_class,
            kwargs=kwargs,
            registered_available_supplier=(self._registry.get_available_generators),
        )
        return cast(IModel, encoder)

    def create_decoder(self, model_type: str, **kwargs) -> IModel:
        """Create a decoder instance of the specified type.

        This method creates decoder models for latent space reconstruction,
        such as VAE decoders.

        Args:
            model_type (str): Type of decoder to create
                (e.g., 'vae_decoder').
            **kwargs: Model-specific parameters (latent_dim, out_channels,
                etc.).

        Returns:
            IModel: Instantiated decoder implementing the IModel interface.

        Raises:
            ValueError: If the model_type is not registered or invalid.

        Example:
            >>> decoder = factory.create_decoder('vae_decoder',
            ...                                  latent_dim=128)

        """
        _, decoder = self._create_registered_model(
            kind="decoder",
            requested_type=model_type,
            has_registered=self._registry.has_generator,
            fetch_class=self._registry.get_generator_class,
            kwargs=kwargs,
            registered_available_supplier=(self._registry.get_available_generators),
        )
        return cast(IModel, decoder)

    def create_model(
        self,
        config: "TrainingConfig",
    ) -> Any:
        """Create a generator (and optional discriminator) from a full config.

        .. deprecated::
            Prefer
            :class:`~spectramr.infrastructure.training.builders.model_builder.ModelBuilder`,
            which every training run already uses. This method resolves a
            strict subset of the config: it has no contract gating against the
            generator's ``__init__``, and never injects ``kspace_log_scaled``
            or the ``physics.data_consistency`` block. It is kept for callers
            outside ``src/`` that hold only a ``ModelFactory``.

        Args:
            config: A full ``TrainingSettings``. A bare ``ModelConfigSchema``
                raises -- see the comment below for why that branch is gone.

        Returns:
            tuple: (generator, discriminator)

        Raises:
            TypeError: If ``config`` is not a full training configuration.
            ValueError: If the configuration names no model type.
        """
        # The deprecation belongs HERE, not on ``__init__``.
        #
        # Warning on construction fired on the canonical path too -- every
        # ``GeneratorBuilder.build()`` -- which is why that builder was forced
        # to wrap the import in ``catch_warnings`` and mute it. A warning the
        # correct path must silence is noise, not a signal: it trained readers
        # to ignore the one surface that really is retired, and because
        # ``pyproject.toml`` promotes ``error::DeprecationWarning:spectramr.*``
        # while ``stacklevel=2`` attributes the warning to the *caller's*
        # module, whether it was enforced depended on who happened to call.
        # Moving it here makes the promote-to-error a real gate: it now fires
        # exactly on ``create_model`` and on nothing else.
        import warnings

        warnings.warn(
            "ModelFactory.create_model is deprecated and will be removed in "
            "version 6.0. Use ModelBuilder, which resolves every config->kwarg "
            "the model declares, contract-gated against the generator's own "
            "__init__:\n"
            "    from spectramr.infrastructure.training.builders.model_builder "
            "import ModelBuilder\n"
            "    model = ModelBuilder(config, device).build_generator()"
            ".validate().build()['generator']",
            DeprecationWarning,
            stacklevel=2,
        )
        logger.debug(f"DEBUG: ModelFactory.create_model called with config type {type(config)}")

        # Full TrainingSettings only.
        #
        # A bare ModelConfigSchema used to be accepted here through
        # `hasattr(config, "model")`, silently selecting a reduced path that
        # injected neither `acceleration_config` nor `kspace_log_scaled`. That
        # is how `infer`, `make` and the evaluation script each built a
        # *different model* from the same YAML that training built: the
        # acceleration block was dropped on every arm declaring it, and the 12
        # arms setting `output_kspace_clip_ratio` raised at construction
        # (#1306). The sniff itself was the defect -- a type test that quietly
        # selects degraded behaviour is the silent fallback CLAUDE.md #3
        # forbids -- so it is removed rather than repaired: ModelBuilder
        # already does this job completely, and is the replacement every
        # in-repo caller now uses.
        if not hasattr(config, "model"):
            raise TypeError(
                f"create_model() requires a full TrainingSettings, got "
                f"{type(config).__name__}. Build the model through the "
                "canonical builder instead:\n"
                "    from spectramr.infrastructure.training.builders."
                "model_builder import ModelBuilder\n"
                "    model = ModelBuilder(config, device).build_generator()"
                ".validate().build()['generator']\n"
                "It resolves every config->kwarg the model declares, "
                "contract-gated against the generator's own __init__ "
                "signature -- which the removed bare-schema branch did not."
            )

        model_config = config.model
        model_type = model_config.model_type
        # Start with model_kwargs, then merge in top-level model fields so the
        # constructor gets in_channels, out_channels, etc.
        kwargs = dict(model_config.model_kwargs) if model_config.model_kwargs else {}

        # Explicit attribute access instead of loop+getattr to satisfy the SSOT
        # compliance check.
        if "in_channels" not in kwargs:
            kwargs["in_channels"] = model_config.in_channels
        if "out_channels" not in kwargs:
            kwargs["out_channels"] = model_config.out_channels
        if "base_channels" not in kwargs:
            kwargs["base_channels"] = model_config.base_channels
        if "depth" not in kwargs:
            kwargs["depth"] = model_config.depth

        # Acceleration config for physics-informed models: parameters like
        # max_acceleration and schedule_type must reach the generator.
        if hasattr(config, "undersampling") and config.undersampling:
            accel_config = config.undersampling
            if hasattr(accel_config, "model_dump"):
                kwargs["acceleration_config"] = accel_config.model_dump()
            elif hasattr(accel_config, "__dict__"):
                kwargs["acceleration_config"] = accel_config.__dict__
            elif isinstance(accel_config, dict):
                kwargs["acceleration_config"] = accel_config

        # z_dim is only needed for VAE/GAN models -- models that need it
        # declare it in their model_kwargs explicitly.

        logger.debug(f"DEBUG: ModelFactory resolved model_type: {model_type}")
        logger.debug(f"DEBUG: ModelFactory final kwargs keys: {list(kwargs.keys())}")

        if not model_type:
            raise ValueError("Model type not specified in configuration")

        # Delegate to ModelCreator for parameter mapping and instantiation
        # Note: Only pass model-specific kwargs, NOT the full TrainingConfig.
        # The model_kwargs from config.model.model_kwargs contain all needed params.
        generator = self._model_creator.create_generator(model_type, **kwargs)
        discriminator = None
        if self._registry.get_discriminator_class(model_type) is not None:
            discriminator = self._model_creator.create_discriminator(model_type, **kwargs)

        return generator, discriminator

    def create_diffusion_model(
        self,
        model_type: str,
        in_channels: int = 1,
        out_channels: int = 1,
        device: str = "cuda",
        **kwargs,
    ) -> tuple[Any, Any]:
        """Create a diffusion model with its corresponding process.

        This method creates a diffusion model by combining a generator backbone
        with a diffusion process (noise scheduler).

        Args:
            model_type: Type of diffusion model to create
                (e.g., 'standard_unet_diffusion', 'vit_kan_cold').
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            device: Device to place the model on.
            **kwargs: Additional parameters for generator and diffusion process.

        Returns:
            tuple: (diffusion_process, model) where diffusion_process is the
                noise scheduler and model is the generator backbone.

        """
        return self._create_diffusion_model_internal(
            model_type, in_channels, out_channels, device, **kwargs
        )

    def create_time_conditioned_generator(
        self,
        model_type: str,
        time_embed_type: str = "sinusoidal",
        **kwargs,
    ) -> IGenerator:
        """Create a generator with time conditioning."""
        kwargs["time_embedding_type"] = time_embed_type
        return self.create_generator(model_type, **kwargs)

    def _create_diffusion_model_internal(
        self,
        model_type: str,
        in_channels: int = 1,
        out_channels: int = 1,
        device: str = "cuda",
        **kwargs,
    ) -> tuple[Any, Any]:
        """Internal method to create diffusion models."""
        # Handle backward compatibility: map in_channels to in_channels
        if "in_channels" in kwargs and "in_channels" not in locals():
            in_channels = kwargs.pop("in_channels")

        # Check if CUDA is available, fallback to CPU if not
        if device == "cuda" and not torch.cuda.is_available():
            self._logger.warning(
                "CUDA requested but not available. Using CPU instead.",
            )
            device = "cpu"

        # --- 1. Get Model Configuration ---
        lookup_key = model_type.lower()
        config = DIFFUSION_CONFIGS.get(lookup_key)

        if not config:
            available_types = ", ".join(sorted(DIFFUSION_CONFIGS.keys()))
            raise ValueError(
                f"Unknown model_type: {model_type}. Available types: {available_types}",
            )

        # --- 2. Instantiate Model using Model Factory ---
        generator_type = config["generator_type"]
        generator_params = {**config.get("generator_params", {}), **kwargs}

        # Filter out diffusion-specific parameters that don't belong to generators
        diffusion_specific_params = {
            "time_dim",
            "gradient_clip_val",
            "enable_gradient_checkpointing",
            "gradient_lambda",
            "entropy_lambda",
            "ssim_lambda",
            "min_dof",
            "max_dof",
            "dof_schedule",
            "gaussian_approximation_threshold",
            "v_min",
            "v_max",
            "degradation_type",
            "cold_schedule",
            "restoration_steps",
        }

        # Remove diffusion-specific parameters from generator params
        filtered_generator_params = {
            k: v for k, v in generator_params.items() if k not in diffusion_specific_params
        }

        # Create generator using the model factory
        try:
            model = self.create_generator(
                generator_type,
                in_channels=in_channels,
                out_channels=out_channels,
                **filtered_generator_params,
            )
            # Set device and time embedding flag if supported
            try:
                model = model.to(device)
            except AttributeError as _exc:
                logger.debug("Suppressed exception: %s", _exc)  # Model doesn't support .to() method

            try:
                model.has_time_embedding = True  # Flag for the trainer
            except (AttributeError, TypeError) as _exc:
                logger.debug(
                    "Suppressed exception: %s", _exc
                )  # Model doesn't support has_time_embedding attribute
        except Exception as e:
            self._logger.error(f"Failed to create generator {generator_type}: {e}")
            raise ValueError(
                f"Failed to create generator '{generator_type}' with "
                f"parameters: in_channels={in_channels}, out_channels={out_channels}, "
                f"additional_params={generator_params}",
            ) from e

        # --- 2b. Ensure Time Conditioning Support ---
        # If the model is intended for diffusion but doesn't accept 'timesteps' or 't',
        # wrap it in TimeConditionedWrapper.
        supports_time = False
        if hasattr(model, "forward"):
            sig = inspect.signature(model.forward)
            # Check for common time argument names
            if any(p in sig.parameters for p in ["t", "timestep", "timesteps"]):
                supports_time = True

        if not supports_time:
            self._logger.info(
                f"Model {generator_type} does not accept timesteps in forward(). "
                "Wrapping in TimeConditionedWrapper for diffusion training."
            )
            # Extract time embedding parameters
            time_dim = int(kwargs.get("time_dim", 256))
            max_timesteps = int(kwargs.get("num_timesteps", 1000))

            model = TimeConditionedWrapper(
                model, time_embed_dim=time_dim, max_timesteps=max_timesteps
            )

        # --- 3. Instantiate Diffusion Process ---
        diffusion_class = DIFFUSION_PROCESSES[config["diffusion_class"]]["class"]
        # Combine parameters from config and function call
        diffusion_params = {
            **DIFFUSION_PROCESSES[config["diffusion_class"]].get("diffusion_params", {}),
            **kwargs,
        }

        # Handle tuple-based kwarg mapping from the config
        # for backward compatibility
        for key, value in list(diffusion_params.items()):
            if isinstance(value, tuple) and len(value) == 2:
                kwarg_name, default_value = value
                diffusion_params[key] = kwargs.get(kwarg_name, default_value)

        # Filter parameters for the diffusion process constructor
        diffusion_sig = inspect.signature(diffusion_class.__init__)
        valid_diffusion_keys = {p.name for p in diffusion_sig.parameters.values()}
        filtered_diffusion_params = {
            k: v for k, v in diffusion_params.items() if k in valid_diffusion_keys
        }
        filtered_diffusion_params["device"] = device

        diffusion_process = diffusion_class(**filtered_diffusion_params)

        # Set noise type if specified in config
        if "noise_type" in config:
            diffusion_process.noise_type = config["noise_type"]

        return diffusion_process, model

    def get_available_diffusion_types(self) -> list[str]:
        """Get list of available diffusion model types.

        Returns:
            list: list of available diffusion model type strings

        """
        return sorted(DIFFUSION_CONFIGS.keys())

    def create_guidance_free_classifier(
        self,
        num_classes: int = 2,
        input_channels: int = 1,
        image_size: int = 256,
        backbone_type: str = "cnn",
        device: str | None = None,
    ) -> Any:
        """Create a guidance-free classifier for diffusion models.

        Args:
            num_classes: Number of classes to classify
            input_channels: Number of input channels
            image_size: Input image size (assumed square)
            backbone_type: Type of backbone network
            device: Device to run on

        Returns:
            GuidanceFreeClassifier instance

        """
        from spectramr.infrastructure.di.di_container import resolve_service
        from spectramr.models.classifiers.guidance_free_classifier import (
            GuidanceFreeClassifier,
        )

        classifier = GuidanceFreeClassifier(
            num_classes=num_classes,
            input_channels=input_channels,
            image_size=image_size,
            backbone_type=backbone_type,
            device=device,
        )

        # Lazy logger resolution to avoid import-time DI issues
        from spectramr.infrastructure.services import ILoggingService

        logger = resolve_service(ILoggingService)
        logger.log_info(
            f"Created {classifier.name} with {classifier.get_parameter_count():,} parameters",
        )

        return classifier

    def get_available_classifier_types(self) -> list[str]:
        """Get list of available classifier types.

        Returns:
            list of available classifier type strings

        """
        return ["guidance_free_cnn"]

    def create_classifier_from_config(self, config: dict[str, Any]) -> Any:
        """Create a classifier from configuration dictionary.

        Args:
            config: Configuration dictionary containing classifier parameters

        Returns:
            Configured classifier instance

        """
        return self.create_guidance_free_classifier(
            num_classes=config.get("num_classes", 2),
            input_channels=config.get("input_channels", 1),
            image_size=config.get("image_size", 256),
            backbone_type=config.get("backbone_type", "cnn"),
            device=config.get("device"),
        )


# Global factory instance
_policy_factory_instances: dict[tuple[str, bool], ModelFactory] = {}


def get_model_factory(
    config: Any | None = None,
    strict_kwargs: bool | None = None,
    # DI Injection points
    registry: Optional["ModelRegistry"] = None,
    parameter_mapper: Optional["ParameterMapper"] = None,
    validator: Optional["ModelValidator"] = None,
) -> ModelFactory:
    """Return a model factory honoring per-policy singleton semantics.

    If custom DI components (registry, parameter_mapper, validator) are provided,
    singleton caching is BYPASSED to ensure the specific components are used.
    """
    # The unknown-model policy and its fallback handler were deleted on
    # 2026-09-03 (#1338): every creation path already raises
    # ``ModelCreationError`` on a name the registry does not hold, the handler
    # returned ``None`` unconditionally, and the stored policy had no reader,
    # so the "warn" and "silent" fallbacks named a degradation that never
    # existed (pitfall #16). ``strict`` went with it: it selected the policy and
    # nothing else, and the ``allow_strict`` it produced reached a constructor
    # that never read it.

    if strict_kwargs is None:
        # F18: same None-safe lookup for ``strict_model_kwargs``.
        if config is None:
            strict_kwargs = False
        else:
            strict_kwargs = getattr(config, "strict_model_kwargs", False)
    strict_kwargs_flag = bool(strict_kwargs)

    # CHECK FOR DI INJECTION: If any custom component is passed, do not use cache.
    if registry is not None or parameter_mapper is not None or validator is not None:
        return ModelFactory(
            registry=registry,
            parameter_mapper=parameter_mapper,
            validator=validator,
            strict_kwargs_default=strict_kwargs_flag,
        )

    key = (strict_kwargs_flag,)

    if key not in _policy_factory_instances:
        _policy_factory_instances[key] = ModelFactory(
            strict_kwargs_default=strict_kwargs_flag,
        )
    return _policy_factory_instances[key]


from spectramr.shared.utils.metaclass import ModuleABCMeta


class TimeConditionedWrapper(nn.Module, IGenerator, metaclass=ModuleABCMeta):
    """Wrapper that adds time conditioning to any generator model.

    This class wraps a base generator and adds time conditioning capabilities,
    enabling it to be used as a denoising model in diffusion processes.
    """

    def __init__(
        self,
        model: nn.Module,
        time_embed_dim: int = 128,
        max_timesteps: int = 1000,
        time_embed_type: str = "sinusoidal",
    ):
        """Initialize the time-conditioned wrapper.

        Args:
            model: Base generator model to wrap
            time_embed_dim: Dimension of time embeddings
            max_timesteps: Maximum number of timesteps
            time_embed_type: Type of time embedding

        """
        super().__init__()
        self.model = model
        self.time_embed_dim = time_embed_dim
        self.max_timesteps = max_timesteps
        self.time_embed_type = time_embed_type

        self.time_embedding = self._create_time_embedding(
            time_embed_type, time_embed_dim, max_timesteps
        )

    def _create_time_embedding(self, embed_type: str, dim: int, max_timesteps: int) -> nn.Module:
        """Create time embedding module using a registry-based approach."""
        factories = {
            "sinusoidal": self._create_sinusoidal,
            "simple": self._create_simple,
            "kan": self._create_kan,
            "simple_kan": self._create_simple_kan,
        }

        factory = factories.get(embed_type)
        if factory is None:
            raise ValueError(f"Unknown time embedding type: {embed_type}")

        return factory(dim, max_timesteps)

    def _create_sinusoidal(self, dim: int, max_timesteps: int) -> nn.Module:
        """_create_sinusoidal.

        Args:
            dim (int): Description.
            max_timesteps (int): Description.
        Returns:
            nn.Module: Description.
        """
        from ..diffusion.simple_enhanced_diffusion import SimpleTimeEmbedding

        return SimpleTimeEmbedding(dim)

    def _create_simple(self, dim: int, max_timesteps: int) -> nn.Module:
        """_create_simple.

        Args:
            dim (int): Description.
            max_timesteps (int): Description.
        Returns:
            nn.Module: Description.
        """
        return nn.Embedding(max_timesteps, dim)

    def _create_kan(self, dim: int, max_timesteps: int) -> nn.Module:
        """_create_kan.

        Args:
            dim (int): Description.
            max_timesteps (int): Description.
        Returns:
            nn.Module: Description.
        """
        from ..blocks.embeddings import KANTimeEmbedding

        return KANTimeEmbedding(dim)

    def _create_simple_kan(self, dim: int, max_timesteps: int) -> nn.Module:
        """_create_simple_kan.

        Args:
            dim (int): Description.
            max_timesteps (int): Description.
        Returns:
            nn.Module: Description.
        """
        from ..diffusion.simple_enhanced_diffusion import SimpleKANTimeEmbedding

        return SimpleKANTimeEmbedding(dim)

    @property
    def name(self) -> str:
        """Returns the model name."""
        model_name = self.model.name
        return f"TimeConditioned{model_name}"

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with optional time conditioning.

        Args:
            x: Input tensor
            t: Time step tensor (optional)

        Returns:
            Output tensor

        forward method for TimeConditionedWrapper.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            t (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService.
        """
        if t is not None:
            # Ensure t is on the correct device
            device = (
                next(self.model.parameters()).device
                if hasattr(self.model, "parameters")
                else x.device
            )
            t = t.to(device)

            # Get time embeddings
            time_emb = self.time_embedding(t)
            # For simplicity, add time embedding as a bias to the output
            # In a full implementation, this would be injected into the model
            output = self.model(x)

            # Broadcast time embedding to match output shape
            # output shape: (batch, channels, spatial_dim1, spatial_dim2, ...)
            batch_size = output.shape[0]
            spatial_dims = output.ndim - 2

            # Reshape time_emb to (batch, time_dim, 1, 1, ...)
            view_shape = (batch_size, -1) + (1,) * spatial_dims
            time_emb_reshaped = time_emb.view(view_shape)

            # Expand to match output spatial dimensions
            # (batch, time_dim, spatial_dim1, spatial_dim2, ...)
            expand_shape = (batch_size, -1) + output.shape[2:]
            time_emb_expanded = time_emb_reshaped.expand(expand_shape)

            # Slice to match output channels
            if time_emb_expanded.shape[1] >= output.shape[1]:
                return output + time_emb_expanded[:, : output.shape[1]]
            else:
                # If time embedding is smaller, repeat it
                repeats = (output.shape[1] // time_emb_expanded.shape[1]) + 1
                time_emb_repeated = time_emb_expanded.repeat(1, repeats, *([1] * spatial_dims))
                return output + time_emb_repeated[:, : output.shape[1]]

        return self.model(x)

    def generate(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate samples from latent space."""
        return self.forward(z, **kwargs)

    def get_output_shape(
        self,
        input_shape: tuple[int, ...],
    ) -> tuple[int, ...]:
        """Returns the output shape for a given input shape."""
        # Try to infer from the model if it has this method
        if hasattr(self.model, "get_output_shape"):
            return self.model.get_output_shape(input_shape)
        # Otherwise, assume same shape
        return input_shape

    def get_parameter_count(self) -> int:
        """Returns the total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())


def list_available_models() -> dict[str, list[str]]:
    """Get lists of all available generator and discriminator types.

    Returns:
        dict[str, list]: Dictionary with 'generators' and 'discriminators'
            keys containing lists of available model type names.

    """
    factory = get_model_factory()
    return factory.get_available_types()


def create_model_pair(
    model_type: str,
    in_channels: int = 1,
    out_channels: int = 1,
    **kwargs: Any,
) -> tuple[torch.nn.Module | None, torch.nn.Module | None]:
    """Create a generator-discriminator pair for the given model type.

    This is a module-level convenience function that uses the default model factory.

    Args:
        model_type: Type of model to create
        in_channels: Number of input channels
        out_channels: Number of output channels
        **kwargs: Additional arguments for model creation

    Returns:
        Tuple of (generator, discriminator) models
    """
    factory = get_model_factory()
    return factory.create_model_pair(
        model_type, in_channels=in_channels, out_channels=out_channels, **kwargs
    )
