"""Model Builder

Creates neural network models (generators, discriminators, encoders, decoders)
based on training paradigm and configuration.
"""

import logging

import torch
import torch.nn as nn
import torch.utils.checkpoint  # [FIX] Added for gradient checkpointing

from spectramr.config.settings import TrainingSettings
from spectramr.core.module_utils import resolve_state_dict
from spectramr.infrastructure.builders.generator_kwargs import (
    apply_gradient_checkpointing,
    resolve_generator_kwargs,
)
from spectramr.infrastructure.builders.leaf.model_builders import (
    DecoderBuilder,
    DiscriminatorBuilder,
    EncoderBuilder,
    GeneratorBuilder,
)
from spectramr.infrastructure.optimization.ema import ModelEma  # [FIX] Added for EMA

from .base import Builder

logger = logging.getLogger(__name__)


class ModelBuilder(Builder):
    """Builds models based on training paradigm and config.

    This builder creates all neural network models required for a training
    paradigm. It delegates to ModelFactory for actual instantiation and
    handles device placement and validation.

    Attributes:
        _config: Training configuration
        _device: Target device for models
        _models: Dictionary of created models
        _factory: ModelFactory instance for creating models

    Example:
        >>> builder = ModelBuilder(config, torch.device("cuda"))
        >>> models = (builder
        ...     .build_generator()
        ...     .build_discriminator()
        ...     .validate()
        ...     .build())
        >>> generator = models["generator"]
    """

    def __init__(self, config: TrainingSettings, device: torch.device):
        """Initialize ModelBuilder.

        Args:
            config: Immutable training configuration
            device: Device where models will be placed
        """
        self._config = config
        self._device = device
        self._models: dict[str, nn.Module] = {}
        self._ema: ModelEma | None = None

    @property
    def ema(self) -> ModelEma | None:
        """Access the built EMA model if configured."""
        return self._ema

    def build_generator(self) -> "ModelBuilder":
        """Build generator/main reconstruction model.

        Creates the primary model for the training paradigm (generator for GAN,
        denoising model for diffusion, reconstruction model for supervised, etc.).

        Kwarg resolution is delegated wholesale to
        :func:`~spectramr.infrastructure.builders.generator_kwargs.resolve_generator_kwargs`
        and checkpointing to ``apply_gradient_checkpointing``, so that training
        and ``audit --probe`` construct the model from one answer. No
        resolution logic lives here by design -- reintroducing any would
        recreate the divergence those functions exist to prevent.

        Returns:
            self: For method chaining

        Raises:
            ValueError: If model creation fails
        """
        try:
            gen_kwargs = resolve_generator_kwargs(self._config, device=self._device)

            generator_builder = (
                GeneratorBuilder(self._config, self._device)
                .with_architecture(self._config.model.model_type)
                .with_input_channels(self._config.model.in_channels)
                .with_output_channels(self._config.model.out_channels)
            )
            for k, v in gen_kwargs.items():
                generator_builder.with_parameter(k, v)

            generator = generator_builder.validate().build()
            generator.train()  # Set to training mode

            # Warm-start from another model's checkpoint if model.checkpoint_path
            # is set (the sequential-campaign injection target — transfer-init /
            # conditioning, NOT a training resume).
            self._load_init_checkpoint(generator)

            apply_gradient_checkpointing(generator, self._config)

            self._models["generator"] = generator
            logger.info(f"Created generator: {self._config.model.model_type} on {self._device}")
        except Exception as e:
            raise ValueError(f"Failed to create generator: {e}") from e

        return self

    def _load_init_checkpoint(self, generator: nn.Module) -> None:
        """Warm-start ``generator`` from ``config.model.checkpoint_path`` if set.

        Loads ANOTHER model's weights into the freshly built generator as
        transfer-init / conditioning (the sequential-campaign injection target,
        ``inject_as: model.checkpoint_path``). Weights only — this is NOT a
        training resume (no optimizer/scheduler state).

        Uses ``strict=False`` so a partially-overlapping architecture (a cascade
        stage that grew a head) still transfers its shared trunk, but **raises**
        if NOTHING matched — a silently-dropped warm-start (``strict=False``
        swallowing every key) is the Method-C facade where the run trains from
        scratch while looking warm-started (CLAUDE.md pitfalls #9/#16).
        """
        model_cfg = getattr(self._config, "model", None)
        ckpt_path = getattr(model_cfg, "checkpoint_path", None)
        if not ckpt_path:
            return

        payload = torch.load(ckpt_path, map_location=self._device)
        model_keys = set(generator.state_dict().keys())
        # Envelope unwrap + wrapper-prefix strip + zero-overlap raise now live in
        # core.module_utils. This was the only correct reader of the six in the
        # repo, so it became the SSOT the others adopted rather than staying a
        # local fix (#1310). Prefix stripping is new here and only widens what
        # warm-starts: a checkpoint written by a compiled/DDP run shared zero
        # keys with a bare generator and raised below.
        state = resolve_state_dict(payload, model_keys, source=f"model.checkpoint_path={ckpt_path}")
        overlap = model_keys & set(state)

        missing, unexpected = generator.load_state_dict(state, strict=False)
        logger.info(
            "Warm-started generator from %s: %d/%d params loaded (%d missing, %d unexpected)",
            ckpt_path,
            len(overlap),
            len(model_keys),
            len(missing),
            len(unexpected),
        )

    def build_discriminator(self) -> "ModelBuilder":
        """Build discriminator (GAN paradigms only).

        Creates discriminator for adversarial training. Automatically skips
        if discriminator is not configured.

        Discriminator is created only if explicitly configured via:
        1. model.discriminator_component (preferred)
        2. training.gan.discriminator (backward compat)

        Returns:
            self: For method chaining

        Raises:
            ValueError: If discriminator creation fails
        """
        # Determine type & kwargs
        disc_type = None
        disc_kwargs = {}

        # Modern config schema mapping
        if (
            hasattr(self._config.model, "discriminator_component")
            and self._config.model.discriminator_component
        ):
            disc_type = self._config.model.discriminator_component.name
            disc_kwargs = dict(getattr(self._config.model.discriminator_component, "kwargs", {}))
            # Extract in_channels before it is popped at line 272 so the fallback
            # at line 278 doesn't silently use model.out_channels.
            disc_in_channels = disc_kwargs.get("in_channels")
        elif hasattr(self._config.model, "discriminator") and self._config.model.discriminator:
            disc_config = self._config.model.discriminator
            disc_type = (
                disc_config.model_type
                if hasattr(disc_config, "model_type")
                else "patch_gan_discriminator"
            )
            disc_kwargs = getattr(disc_config, "model_kwargs", {})

            # [FIX] Extract in_channels directly from config or kwargs
            disc_in_channels = getattr(disc_config, "in_channels", None)
            if disc_in_channels is None:
                disc_in_channels = disc_kwargs.pop("in_channels", self._config.model.out_channels)

        elif self._config.training.gan is not None and getattr(
            self._config.training.gan, "discriminator", None
        ):
            # GANSubConfigSchema has no ``discriminator`` field (extra='ignore'
            # drops it), so reading the attribute directly raised AttributeError
            # whenever ``training.gan`` was set without a top-level
            # ``model.discriminator_component``. Use getattr so this back-compat
            # branch is a safe no-op rather than a crash.
            gan_config = self._config.training.gan
            disc = gan_config.discriminator
            disc_type = getattr(disc, "type", "patchgan")
            disc_kwargs = getattr(disc, "kwargs", {})

        if disc_type is None:
            logger.debug("Skipping discriminator creation: No discriminator_component configured.")
            return self

        try:
            # [FIX] Ensure in_channels is not duplicated if present in disc_kwargs
            disc_kwargs.pop("in_channels", None)

            # Determine in_channels for discriminator
            # Default to model output channels for pixel-domain discrimination
            # (discriminators typically operate on the generator's output space)
            # Use previously extracted in_channels if it exists
            if "disc_in_channels" not in locals() or disc_in_channels is None:
                disc_in_ch = self._config.model.out_channels
            else:
                disc_in_ch = disc_in_channels

            disc_builder = (
                DiscriminatorBuilder(self._config, self._device)
                .with_architecture(disc_type)
                .with_input_channels(disc_in_ch)
            )
            for k, v in disc_kwargs.items():
                disc_builder.with_parameter(k, v)

            discriminator = disc_builder.validate().build()
            discriminator.train()
            self._models["discriminator"] = discriminator
            logger.info(f"Created discriminator: {disc_type} on {self._device}")
        except Exception as e:
            raise ValueError(f"Failed to create discriminator: {e}") from e

        return self

    def build_encoder_decoder(self) -> "ModelBuilder":
        """Build encoder/decoder pair (VAE/VQ-VAE paradigms).

        Creates encoder and decoder for variational autoencoder training.
        Automatically skips if not in VAE mode or if VAE config is not present.

        Returns:
            self: For method chaining

        Raises:
            ValueError: If VAE encoder/decoder creation fails
        """
        # Check if VAE config is present
        if not self._config.training or not hasattr(self._config.training, "latent"):
            logger.debug("Skipping encoder/decoder (no latent/VAE config found)")
            return self

        latent_config = self._config.training.latent
        if not latent_config or not hasattr(latent_config, "latent_dim"):
            logger.debug("Skipping encoder/decoder (latent_dim not configured)")
            return self

        # Create encoder if configured
        if hasattr(latent_config, "encoder_config") and latent_config.encoder_config:
            try:
                encoder_builder = (
                    EncoderBuilder(self._config, self._device)
                    .with_architecture("vae_encoder")
                    .with_input_channels(self._config.model.in_channels)
                    .with_latent_dim(latent_config.latent_dim)
                )
                for k, v in getattr(latent_config.encoder_config, "kwargs", {}).items():
                    encoder_builder.with_parameter(k, v)

                encoder = encoder_builder.validate().build()
                encoder.train()
                self._models["encoder"] = encoder
                logger.info(f"Created encoder on {self._device}")
            except Exception as e:
                logger.warning(f"Failed to create encoder: {e}")

        # Create decoder if configured
        if hasattr(latent_config, "decoder_config") and latent_config.decoder_config:
            try:
                decoder_builder = (
                    DecoderBuilder(self._config, self._device)
                    .with_architecture("vae_decoder")
                    .with_latent_dim(latent_config.latent_dim)
                    .with_output_channels(self._config.model.out_channels)
                )
                for k, v in getattr(latent_config.decoder_config, "kwargs", {}).items():
                    decoder_builder.with_parameter(k, v)

                decoder = decoder_builder.validate().build()
                decoder.train()
                self._models["decoder"] = decoder
                logger.info(f"Created decoder on {self._device}")
            except Exception as e:
                logger.warning(f"Failed to create decoder: {e}")

        return self

    def validate(self) -> "ModelBuilder":
        """Validate all required models are created.

        Checks that the generator model (essential for all paradigms) is present.
        Additional models (discriminator, encoder, decoder) are validated if configured.

        Returns:
            self: For method chaining

        Raises:
            ValueError: If the generator model is missing
        """
        # Minimum requirement: generator must always be created
        if "generator" not in self._models:
            raise ValueError(
                "Missing required model: 'generator' (essential for all training paradigms)"
            )

        # Validate optional models based on what was configured
        optional_models = []
        if (
            hasattr(self._config.model, "discriminator_component")
            and self._config.model.discriminator_component
        ):
            optional_models.append("discriminator")
        if (
            self._config.training
            and hasattr(self._config.training, "gan")
            and self._config.training.gan
        ):
            optional_models.append("discriminator")
        if (
            self._config.training
            and hasattr(self._config.training, "latent")
            and self._config.training.latent
        ):
            # VAE may optionally have encoder/decoder
            pass

        created_models = list(self._models.keys())
        logger.info(f"Model validation passed. Created models: {created_models}")
        return self

    def build_ema(self) -> "ModelBuilder":
        """Build Exponential Moving Average (EMA) shadow model.

        Evaluates config.ema.enabled and wraps the compiled/instantiated generator
        with the ModelEma tracker. Must be called after 'generator' is built.

        Consumes the whole EMA family: ``enabled``, ``decay``, ``warmup``, and
        the adaptive schedule (``enable_adaptive_ema``, ``warmup_steps``,
        ``initial_decay``, ``final_decay``). When the adaptive schedule is on
        it SUPERSEDES ``decay`` and ``warmup``; both are logged as ignored so
        the run log states which schedule actually ran.

        Returns:
            self: For method chaining

        Raises:
            ValueError: The EMA declaration cannot be honoured (e.g. adaptive
                with a zero-length ramp). Never silently disables EMA.
        """
        if (
            not hasattr(self._config, "ema")
            or self._config.ema is None
            or not self._config.ema.enabled
        ):
            logger.debug("Skipping EMA creation (disabled in config).")
            return self

        if "generator" not in self._models:
            logger.warning("Tried to build EMA before generator. Skipping.")
            return self

        try:
            generator = self._models["generator"]
            decay = getattr(self._config.ema, "decay", 0.9999)
            # Thread the warmup flag (pitfall #15): without it ModelEma always
            # used its own default (True), so config.ema.warmup=False was a
            # silent no-op — the cause of the byte-identical EMA-warmup
            # on/off Experiment-11 ablation. warmup ramps the effective decay
            # with num_updates so the shadow tracks the live model early
            # instead of validating a near-random-init shadow (EMA-lag).
            warmup = getattr(self._config.ema, "warmup", True)

            # Adaptive schedule (#1294). enable_adaptive_ema and its three
            # deterministic companions were schema-only for months: the
            # implementation was deleted in ff0efff9f and only its orphaned
            # caller in training/state.py survived, so an arm could declare
            # the whole family and get plain fixed-decay EMA. Thread them by
            # NAME (not a dict splat) so the schema-coverage source pin in
            # tests/unit/config/test_ema_schema_coverage.py can see them.
            adaptive = getattr(self._config.ema, "enable_adaptive_ema", False)
            warmup_steps = getattr(self._config.ema, "warmup_steps", 0)
            initial_decay = getattr(self._config.ema, "initial_decay", 0.0)
            final_decay = getattr(self._config.ema, "final_decay", 0.999)

            # Note: Create EMA. Note it deeply copies the generator.
            self._ema = ModelEma(
                generator,
                decay=decay,
                device=self._device,
                warmup=warmup,
                adaptive=adaptive,
                warmup_steps=warmup_steps,
                initial_decay=initial_decay,
                final_decay=final_decay,
            )
            if adaptive:
                # Log the SUPERSEDED knobs too: with adaptive on, decay and
                # warmup are inert, and a run log that only printed the ramp
                # would leave a reader unable to tell which schedule ran.
                logger.info(
                    "Initialized ModelEma for generator with the ADAPTIVE "
                    f"decay ramp {initial_decay} -> {final_decay} over "
                    f"{warmup_steps} updates (this supersedes decay={decay} "
                    f"and warmup={warmup}, which are ignored)."
                )
            else:
                logger.info(
                    f"Initialized ModelEma for generator with decay={decay}, warmup={warmup}"
                )
        except ValueError:
            # An unhonourable EMA declaration must be loud. Falling through to
            # ``self._ema = None`` here would disable EMA for the whole run
            # behind a WARNING while validation silently graded live weights
            # (non-negotiable 3).
            raise
        except Exception as e:
            logger.warning(f"Failed to build EMA model: {e}")
            self._ema = None

        return self

    def build(self) -> dict[str, nn.Module]:
        """Return validated, immutable model dict.

        v6.2 PR-14 — applies PEFT injection (LoRA / adapter / bitfit)
        and FSDP wrapping if either is enabled in
        ``settings.parallel.{peft, fsdp}``. Both default to disabled
        so single-GPU training is unaffected.

        Returns:
            Dict[str, nn.Module]: Copy of models dictionary

        Raises:
            ValueError: If validation hasn't been called or fails
        """
        if not self._models:
            raise ValueError("No models created. Call build methods before build().")

        # PEFT injection only.
        #
        # FSDP wrapping deliberately does NOT happen here any more. The parallel
        # strategy plugin owns it (Stage A, `FSDPStrategy.prepare_models`), and
        # this builder runs BEFORE that hook -- so doing it in both places gave
        # every `strategy: fsdp` arm an FSDP(FSDP(model)): sharding and
        # activation checkpointing applied twice, with the outer root managing
        # zero parameters. `maybe_wrap_with_fsdp` has no already-wrapped guard,
        # so nothing caught it. The schema pins `strategy == "fsdp"` to
        # `fsdp.enabled` (ParallelismConfigSchema._strategy_matches_subblocks),
        # which is exactly the case where the plugin is guaranteed to run, so
        # removing it here loses no coverage.
        parallel = getattr(self._config, "parallel", None)
        peft_cfg = getattr(parallel, "peft", None) if parallel else None

        if peft_cfg is not None and getattr(peft_cfg, "enabled", False):
            # No `except ImportError -> warn -> continue`: the user asked for
            # PEFT, and a run that silently trains every parameter instead of
            # the adapters is a different experiment reporting success
            # (CLAUDE.md #9). An unimportable dependency must be loud.
            from spectramr.infrastructure.distributed.peft_inject import inject_peft

            for name, model in self._models.items():
                self._models[name] = inject_peft(model, peft_cfg)

        return dict(self._models)  # Return copy for immutability
