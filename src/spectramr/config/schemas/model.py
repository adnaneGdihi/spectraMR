"""Model configuration schema.

This module contains model architecture-related configuration including
model type, architecture parameters, and component definitions.

TRELLIS-specific configurations have been extracted to trellis.py for clarity.
"""

from typing import Any

from pydantic import BaseModel, Field, ValidationInfo, field_validator

from spectramr.config.validation_constants import (
    VALID_INPUT_TYPES,
    VALID_KAN_TYPES,
)

from .base import EncoderDecoderDefinitionSchema
from .conditioning import ConditioningConfig
from .trellis import TrellisConfigSchema, TrellisVAEConfigSchema

__all__ = [
    "ConditioningConfig",
    "ModelComponentSchema",
    "ModelConfigSchema",
    "TrellisConfigSchema",
    "TrellisVAEConfigSchema",
]


class ModelComponentSchema(BaseModel):
    """Generic component configuration (generator/discriminator/etc.)."""

    model_config = {"protected_namespaces": (), "extra": "ignore", "frozen": True}

    name: str = Field(default="")
    kwargs: dict[str, Any] = Field(default_factory=dict)


class ModelConfigSchema(BaseModel):
    """Pure data structure for model-specific configuration.

    No aliases are provided - use canonical field names only.

    Strictness (H4): unlike :class:`OptimizationConfigSchema` (which is
    ``extra="forbid"``), the ``model`` block stays ``extra="ignore"`` *for now*.
    The block is intentionally polymorphic — each generator architecture takes
    a different set of constructor knobs, and the canonical home for those is
    the open ``model_kwargs`` dict, which ``ModelBuilder.build_generator``
    forwards to the generator ``__init__``. Several live configs still place
    architecture knobs (``channel_multipliers``, ``num_res_blocks``, ``dropout``,
    ``attention_levels``, VAE ``latent_dim``, ...) at the *top level* of the
    ``model`` block, where ``build_generator`` silently drops them (they are NOT
    forwarded). Flipping to ``extra="forbid"`` would reject ~30 ``inprogress``
    configs at load time, and merely *moving* those keys into ``model_kwargs``
    changes runtime behaviour (the generator would suddenly receive kwargs it
    may not accept). That conform-then-forbid migration is tracked as a
    dedicated, cluster-auditable pass in
    ``TODO/backlog_model_block_strictness_2026_05_28.md`` — do NOT flip this to
    ``forbid`` until that pass lands. Architecture knobs belong in
    ``model_kwargs``; do not add new top-level fields for them.
    """

    model_config = {
        "protected_namespaces": (),
        # H4: kept "ignore" deliberately — see class docstring. The forbid
        # migration is deferred to TODO/backlog_model_block_strictness_2026_05_28.md.
        "extra": "ignore",
        "frozen": True,
    }

    in_channels: int = Field(default=1, gt=0)
    out_channels: int = Field(default=1, gt=0)
    base_channels: int = Field(default=64, gt=0, description="Base channel count")
    depth: int = Field(default=4, ge=1, description="Model depth (number of blocks)")
    spatial_dims: int = Field(default=2, description="Spatial dimensions (2 or 3)")
    model_type: str = Field(default="unet")
    output_activation: str | None = Field(default=None)
    target_domain: str | None = Field(
        default=None,
        description="Domain where model operates: 'kspace' for k-space models, 'image' for image-domain models",
    )
    model_domain: str | None = Field(
        default=None,
        description="Domain tag for the model ('image' / 'kspace' / "
        "'complex'). Widely used across experiment configs as a parallel to "
        "target_domain. Declared explicitly (rather than left to extra=ignore) "
        "so it receives typed validation instead of being passed through "
        "untyped.",
    )
    output_type: str | None = Field(
        default=None,
        description="Declared output representation of the model "
        "('image' / 'kspace' / 'complex' / 'magnitude'). Declared explicitly "
        "so it receives typed validation instead of being passed through "
        "untyped.",
    )
    model_kwargs: dict[str, Any] = Field(default_factory=dict)

    checkpoint_path: str | None = Field(
        default=None,
        description="Optional path to another model's checkpoint to warm-start "
        "(transfer-init / condition) THIS model from at build time. The "
        "sequential-campaign injection target (inject_as: model.checkpoint_path) "
        "— declared as a real field (not left to extra='ignore') so the injected "
        "value survives re-validation and is actually loaded by ModelBuilder. "
        "This is NOT the resume-training path (that is --resume / "
        "training.multi.stages.*.checkpoint_path): it loads weights only, not "
        "optimizer/scheduler state.",
    )

    z_dim: int = Field(default=256, gt=0)
    kan_type: str = Field(default="BSpline")
    wavelet_type: str | None = Field(
        default=None,
        description="Wavelet type for WavKAN: mexican_hat, morlet, dog, meyer, etc.",
    )
    wav_version: str | None = Field(
        default=None,
        description="Implementation version for WavKAN: base, fast, fast_plus_one",
    )
    input_type: str = Field(default="image")
    trellis: TrellisConfigSchema = Field(default_factory=TrellisConfigSchema)
    conditioning: ConditioningConfig = Field(
        default_factory=ConditioningConfig,
        description="Adaptive conditioning: which condition sources (severity, "
        "diffusion time, acquisition, contrast, motion pose) the model adapts to.",
    )
    trellis_vae: TrellisVAEConfigSchema | None = Field(default=None)
    generator_component: ModelComponentSchema | None = Field(default=None)
    discriminator_component: ModelComponentSchema | None = Field(
        default=None,
    )
    encoder_component: ModelComponentSchema | None = Field(default=None)
    decoder_component: ModelComponentSchema | None = Field(default=None)
    architecture: str | EncoderDecoderDefinitionSchema | None = Field(
        default=None,
    )

    # Weight initialization
    weight_init_type: str = Field(
        default="he_normal",
        description="Weight initialization strategy: he_normal, he_uniform, xavier_normal, xavier_uniform, orthogonal, normal",
    )
    weight_init_gain: float = Field(
        default=1.0,
        ge=0.0,
        description="Gain parameter for Xavier/orthogonal initialization",
    )
    weight_init_activation: str = Field(
        default="relu",
        description="Activation function for He initialization: relu, leaky_relu, tanh, sigmoid",
    )
    apply_weight_init: bool = Field(
        default=True,
        description="Apply weight initialization to model",
    )

    # Diffusion wrappers
    denoising_model: dict[str, Any] | None = Field(
        default=None,
        description="Configuration for denoising model inside diffusion wrapper",
    )
    base_diffusion_config: dict[str, Any] | None = Field(
        default=None,
        description="Base configuration for diffusion process",
    )

    # ==================== PHASE 5 ARCHITECTURE ENHANCEMENTS ====================
    # Phase 5 parameters are now passed via model_kwargs:
    # - attention_type: 'dual_domain' enables PhaseSafeDualAttention
    # - phase_safe_attention_num_heads / _reduction: the two keys
    #   kspace_cold_diffusion_generator.py reads for that block. Width is controlled
    #   by _reduction (hidden_dim = in_channels * 2 // reduction); the attention is
    #   linear at native resolution, so there is no token cap to configure.
    # - reflect_padding_bottleneck_layers: Number of reflect-padded layers (default: 2)
    # See model_kwargs for details.
    #
    # `phase_safe_attention_hidden_dim` used to be listed here as "Hidden dimension
    # for attention (default: 128)". It was never read by anything: no key of that
    # name is looked up, and PhaseSafeDualAttention has no hidden-dim parameter to
    # receive it. The similarly-named `phase_safe_dim` belongs to CrossContrastOLMPA,
    # a different block reached only via attention_type: cross_contrast_olmpa.
    # Documenting a default for an unread key is what kept 58 arms declaring it.

    @field_validator("model_type")
    @classmethod
    def validate_model_type(cls, value: str) -> str:
        """Validate ``model_type`` against the live MODEL_REGISTRY.

        The registry is the only authority here (#978). There used to be a
        second one: a fallback onto the static ``VALID_MODEL_TYPES`` whitelist,
        documented as covering "the bootstrap window before
        ``populate_model_registry()`` runs". Two measurements retired it.

        **It could not fire on the real load path.** ``TrainingSettings.
        from_yaml`` calls ``populate_model_registry()`` unconditionally
        whenever ``model_type`` is present, *before* these validators run
        (``config/settings.py``, step 1) — and its own comment explains why
        that call must not be gated on the registry looking empty. Measured:
        ``MODEL_REGISTRY`` is 0 on a bare import and 590 by the time this
        validator sees a value.

        **Where it could fire — direct ``ModelConfigSchema(...)`` construction
        with an unpopulated registry — it gave the wrong answer.** The
        whitelist is a strict *subset* of the registry (460 vs 590, with 0
        advertised-but-unregistered), so it never accepted anything the
        registry would have rejected; it only rejected 130 names that are
        legitimately registered. The old docstring called it "a permissive
        lower bound". It was a restrictive upper bound, in the only window
        where it applied.

        So this populates instead of falling back. ``populate_model_registry``
        is idempotent and its discovery walk is cached, making it a no-op on
        the ``from_yaml`` path where the loader has already run it.

        ``VALID_MODEL_TYPES`` itself is **not** dead and is deliberately left
        alone: it is the advertised-surface contract, enforced by
        ``scripts/audit/framework_coverage_audit.py`` and six test modules
        (every advertised name must resolve, rejected names must never appear,
        every wired alias must be advertised). What was dead was this branch.
        """
        from spectramr.models.init_registry import (
            populate_model_registry,
        )
        from spectramr.models.registry import MODEL_REGISTRY

        # Not gated on emptiness, for the reason config/settings.py records at
        # its own call site: eager ``@register_model`` import side-effects leave
        # the registry PARTIALLY populated, so an emptiness check would skip the
        # walk and silently drop the walk-only models and every alias.
        populate_model_registry()
        if value in MODEL_REGISTRY:
            return value
        raise ValueError(
            f"model_type {value!r} not found in MODEL_REGISTRY "
            f"({len(MODEL_REGISTRY)} entries). Add an @register_model "
            f"decorator or update src/models/init_registry.py."
        )

    @field_validator("kan_type")
    @classmethod
    def validate_kan_type(cls, value: str) -> str:
        """validate_kan_type.

        Args:
            value (str): Description.
        Returns:
            str: Description.
        """
        if value not in VALID_KAN_TYPES:
            raise ValueError(f"kan_type must be one of {VALID_KAN_TYPES}")
        return value

    @field_validator("input_type")
    @classmethod
    def validate_input_type(cls, value: str) -> str:
        """validate_input_type.

        Args:
            value (str): Description.
        Returns:
            str: Description.
        """
        if value not in VALID_INPUT_TYPES:
            raise ValueError(f"input_type must be one of {VALID_INPUT_TYPES}")
        return value

    @field_validator("trellis_vae", mode="after")
    @classmethod
    def ensure_trellis_vae_config(
        cls,
        value: TrellisVAEConfigSchema | None,
        info: ValidationInfo,
    ) -> TrellisVAEConfigSchema | None:
        """ensure_trellis_vae_config.

        Args:
            value (Optional[TrellisVAEConfigSchema]): Description.
            info (ValidationInfo): Description.
        Returns:
            Optional[TrellisVAEConfigSchema]: Description.
        """
        data = getattr(info, "data", {}) or {}
        model_type = data.get("model_type")
        if (
            model_type
            and model_type
            in {
                "trellis_structured_vae",
                "trellis_gaussian_vae",
                "trellis_mesh_vae",
            }
            and value is None
        ):
            return TrellisVAEConfigSchema()
        return value
