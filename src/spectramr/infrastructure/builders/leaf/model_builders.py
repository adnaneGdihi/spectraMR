"""Phase 1: Model Builders for Neural Network Components

Implements fluent builders for creating neural network models:
- GeneratorBuilder: Creates generator/decoder models
- DiscriminatorBuilder: Creates discriminator/critic models
- EncoderBuilder: Creates encoder-only models
- DecoderBuilder: Creates decoder-only models

Each builder provides a fluent API for model configuration and instantiation.
"""

import logging
from typing import Any

import torch
import torch.nn as nn

from spectramr.config.settings import TrainingSettings
from spectramr.infrastructure.builders.core import FluentBuilder
from spectramr.infrastructure.builders.generator_kwargs import (
    apply_model_field_sweep,
)
from spectramr.infrastructure.checkpointing.restore import weights

logger = logging.getLogger(__name__)


class GeneratorBuilder(FluentBuilder[nn.Module]):
    """Builder for creating generator/decoder neural network models.

    Provides fluent API for configuring and instantiating generators
    with various architectures and configurations.

    Example:
        >>> builder = GeneratorBuilder(config, device)
        >>> generator = (builder
        ...     .with_architecture("standard_unet")
        ...     .with_input_channels(2)
        ...     .with_output_channels(2)
        ...     .with_checkpoint(checkpoint_path)
        ...     .validate()
        ...     .build())
    """

    def __init__(self, config: TrainingSettings, device: str = "cuda"):
        """Initialize generator builder.

        Args:
            config: Training configuration
            device: Device to place model on (cuda/cpu)
        """
        super().__init__()
        self._config = config
        self._device = torch.device(device)
        self._architecture: str | None = None
        self._in_channels: int | None = None
        self._out_channels: int | None = None
        self._checkpoint_path: str | None = None
        self._kwargs: dict[str, Any] = {}
        logger.info(f"GeneratorBuilder initialized on {self._device}")

    def with_architecture(self, architecture: str) -> "GeneratorBuilder":
        """Set model architecture.

        Args:
            architecture: Model type (standard_unet, swin_unet, etc)

        Returns:
            self for chaining
        """
        self._architecture = architecture
        return self

    def with_input_channels(self, channels: int) -> "GeneratorBuilder":
        """Set number of input channels.

        Args:
            channels: Input channel count

        Returns:
            self for chaining
        """
        self._in_channels = channels
        return self

    def with_output_channels(self, channels: int) -> "GeneratorBuilder":
        """Set number of output channels.

        Args:
            channels: Output channel count

        Returns:
            self for chaining
        """
        self._out_channels = channels
        return self

    def with_checkpoint(self, checkpoint_path: str) -> "GeneratorBuilder":
        """Load weights from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            self for chaining
        """
        self._checkpoint_path = checkpoint_path
        return self

    def with_parameter(self, key: str, value: Any) -> "GeneratorBuilder":
        """Set custom model parameter.

        Args:
            key: Parameter name
            value: Parameter value

        Returns:
            self for chaining
        """
        self._kwargs[key] = value
        return self

    def validate(self) -> "GeneratorBuilder":
        """Validate builder state.

        Returns:
            self for chaining

        Raises:
            ValueError: If required parameters missing
        """
        super().validate()

        if self._architecture is None:
            raise ValueError("Architecture not specified")

        if self._in_channels is None:
            self._in_channels = self._config.model.in_channels

        if self._out_channels is None:
            self._out_channels = self._config.model.out_channels

        return self

    def build(self) -> nn.Module:
        """Build and return generator model.

        Delegates to ModelFactory.create_generator() which has the full registry
        (191+ generators including manual registrations), parameter mapping,
        constructor signature filtering, and diffusion wrapper handling.

        Returns:
            Configured generator model

        Raises:
            ValueError: If validation fails
            ImportError: If model factory not available
        """
        self.validate()

        try:
            # No DeprecationWarning suppression here any more. The warning moved
            # off ``ModelFactory.__init__`` onto ``ModelFactory.create_model``,
            # the surface actually being retired. This builder calls the
            # *primitive* (``create_generator``/``create_discriminator``), which
            # is what the canonical path is supposed to use -- so there is
            # nothing to mute. A warning muted on the correct path is noise that
            # trains readers to ignore the real one; with the suppression gone,
            # pyproject's ``error::DeprecationWarning:spectramr.*`` becomes a live
            # gate on this line instead of something this builder defeats.
            from spectramr.models.factories.model_factory import ModelFactory

            factory = ModelFactory()

            # Second half of kwarg resolution: strip the explicit channel
            # args, snapshot declared_keys, then sweep top-level model fields.
            # Shared with ModelBuilder and the audit probe so the probed model
            # is the model this builder builds.
            resolved = apply_model_field_sweep(self._kwargs, self._config)
            clean_kwargs = resolved.kwargs
            declared_keys = resolved.declared_keys

            generator = factory.create_generator(
                model_type=self._architecture,
                in_channels=self._in_channels,
                out_channels=self._out_channels,
                _declared_keys=declared_keys,
                **clean_kwargs,
            )

            # Load checkpoint if specified
            if self._checkpoint_path:
                weights(self._checkpoint_path, "generator", generator, policy="exact")
                logger.info(f"Loaded generator checkpoint from {self._checkpoint_path}")

            # Move to device
            generator = generator.to(self._device)

            self._product = generator
            logger.info(
                f"Generator built: {self._architecture} "
                f"({self._in_channels}→{self._out_channels}) on {self._device}"
            )

            return generator

        except ImportError as e:
            raise ImportError(f"Failed to import ModelFactory: {e}") from e


class DiscriminatorBuilder(FluentBuilder[nn.Module]):
    """Builder for creating discriminator/critic neural network models.

    Provides fluent API for configuring and instantiating discriminators
    with various architectures and configurations.

    Example:
        >>> builder = DiscriminatorBuilder(config, device)
        >>> discriminator = (builder
        ...     .with_architecture("patch_gan_2d")
        ...     .with_input_channels(2)
        ...     .with_checkpoint(checkpoint_path)
        ...     .validate()
        ...     .build())
    """

    def __init__(self, config: TrainingSettings, device: str = "cuda"):
        """Initialize discriminator builder.

        Args:
            config: Training configuration
            device: Device to place model on (cuda/cpu)
        """
        super().__init__()
        self._config = config
        self._device = torch.device(device)
        self._architecture: str | None = None
        self._in_channels: int | None = None
        self._checkpoint_path: str | None = None
        self._kwargs: dict[str, Any] = {}
        logger.info(f"DiscriminatorBuilder initialized on {self._device}")

    def with_architecture(self, architecture: str) -> "DiscriminatorBuilder":
        """Set discriminator architecture.

        Args:
            architecture: Discriminator type (patch_gan_2d, patch_gan_3d, etc)

        Returns:
            self for chaining
        """
        self._architecture = architecture
        return self

    def with_input_channels(self, channels: int) -> "DiscriminatorBuilder":
        """Set number of input channels.

        Args:
            channels: Input channel count

        Returns:
            self for chaining
        """
        self._in_channels = channels
        return self

    def with_checkpoint(self, checkpoint_path: str) -> "DiscriminatorBuilder":
        """Load weights from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            self for chaining
        """
        self._checkpoint_path = checkpoint_path
        return self

    def with_parameter(self, key: str, value: Any) -> "DiscriminatorBuilder":
        """Set custom discriminator parameter.

        Args:
            key: Parameter name
            value: Parameter value

        Returns:
            self for chaining
        """
        self._kwargs[key] = value
        return self

    def validate(self) -> "DiscriminatorBuilder":
        """Validate builder state.

        Returns:
            self for chaining

        Raises:
            ValueError: If required parameters missing
        """
        super().validate()

        if self._architecture is None:
            raise ValueError("Architecture not specified")

        if self._in_channels is None:
            self._in_channels = self._config.model.in_channels

        return self

    def build(self) -> nn.Module:
        """Build and return discriminator model.

        Returns:
            Configured discriminator model

        Raises:
            ValueError: If validation fails
            ImportError: If model factory not available
        """
        self.validate()

        try:
            # No DeprecationWarning suppression here any more. The warning moved
            # off ``ModelFactory.__init__`` onto ``ModelFactory.create_model``,
            # the surface actually being retired. This builder calls the
            # *primitive* (``create_generator``/``create_discriminator``), which
            # is what the canonical path is supposed to use -- so there is
            # nothing to mute. A warning muted on the correct path is noise that
            # trains readers to ignore the real one; with the suppression gone,
            # pyproject's ``error::DeprecationWarning:spectramr.*`` becomes a live
            # gate on this line instead of something this builder defeats.
            from spectramr.models.factories.model_factory import ModelFactory

            factory = ModelFactory()

            discriminator = factory.create_discriminator(
                model_type=self._architecture,
                in_channels=self._in_channels,
                **self._kwargs,
            )

            # Load checkpoint if specified
            if self._checkpoint_path:
                weights(self._checkpoint_path, "discriminator", discriminator, policy="exact")
                logger.info(f"Loaded discriminator checkpoint from {self._checkpoint_path}")

            # Move to device
            discriminator = discriminator.to(self._device)

            self._product = discriminator
            logger.info(
                f"Discriminator built: {self._architecture} "
                f"({self._in_channels} channels) on {self._device}"
            )

            return discriminator

        except ImportError as e:
            raise ImportError(f"Failed to import ModelFactory: {e}") from e


class EncoderBuilder(FluentBuilder[nn.Module]):
    """Builder for creating encoder-only neural network models.

    Provides fluent API for configuring encoders used in VAE, autoencoders,
    and other latent-space models.

    Example:
        >>> builder = EncoderBuilder(config, device)
        >>> encoder = (builder
        ...     .with_architecture("vae_encoder")
        ...     .with_input_channels(2)
        ...     .with_latent_dim(64)
        ...     .validate()
        ...     .build())
    """

    def __init__(self, config: TrainingSettings, device: str = "cuda"):
        """Initialize encoder builder.

        Args:
            config: Training configuration
            device: Device to place model on (cuda/cpu)
        """
        super().__init__()
        self._config = config
        self._device = torch.device(device)
        self._architecture: str | None = None
        self._in_channels: int | None = None
        self._latent_dim: int | None = None
        self._kwargs: dict[str, Any] = {}
        logger.info(f"EncoderBuilder initialized on {self._device}")

    def with_architecture(self, architecture: str) -> "EncoderBuilder":
        """Set encoder architecture.

        Args:
            architecture: Encoder type (vae_encoder, residual_encoder, etc)

        Returns:
            self for chaining
        """
        self._architecture = architecture
        return self

    def with_input_channels(self, channels: int) -> "EncoderBuilder":
        """Set number of input channels.

        Args:
            channels: Input channel count

        Returns:
            self for chaining
        """
        self._in_channels = channels
        return self

    def with_latent_dim(self, dim: int) -> "EncoderBuilder":
        """Set latent space dimension.

        Args:
            dim: Latent dimension

        Returns:
            self for chaining
        """
        self._latent_dim = dim
        return self

    def with_parameter(self, key: str, value: Any) -> "EncoderBuilder":
        """Set custom encoder parameter.

        Args:
            key: Parameter name
            value: Parameter value

        Returns:
            self for chaining
        """
        self._kwargs[key] = value
        return self

    def validate(self) -> "EncoderBuilder":
        """Validate builder state.

        Returns:
            self for chaining

        Raises:
            ValueError: If required parameters missing
        """
        super().validate()

        if self._architecture is None:
            raise ValueError("Architecture not specified")

        if self._in_channels is None:
            self._in_channels = self._config.model.in_channels

        if self._latent_dim is None:
            self._latent_dim = self._config.model.latent_dim

        return self

    def build(self) -> nn.Module:
        """Build and return encoder model.

        Returns:
            Configured encoder model

        Raises:
            ValueError: If validation fails
        """
        self.validate()

        try:
            from spectramr.models.factories.model_factory import (
                ModelRegistry,
                ParameterMapper,
            )

            registry = ModelRegistry()
            # Encoders are typically registered as generators internally
            if not registry.has_generator(self._architecture):
                raise ValueError(f"Encoder '{self._architecture}' not registered.")

            encoder_class = registry.get_generator_class(self._architecture)

            mapper = ParameterMapper()
            mapped_params = mapper.map_generator_params(
                self._architecture,
                in_channels=self._in_channels,
                latent_dim=self._latent_dim,
                **self._kwargs,
            )

            encoder = encoder_class(**mapped_params)

            # Move to device
            encoder = encoder.to(self._device)

            self._product = encoder
            logger.info(
                f"Encoder built: {self._architecture} "
                f"({self._in_channels}→{self._latent_dim}) on {self._device}"
            )

            return encoder

        except ImportError as e:
            raise ImportError(f"Failed to import ModelFactory: {e}") from e


class DecoderBuilder(FluentBuilder[nn.Module]):
    """Builder for creating decoder-only neural network models.

    Provides fluent API for configuring decoders used in VAE, autoencoders,
    and other latent-space models.

    Example:
        >>> builder = DecoderBuilder(config, device)
        >>> decoder = (builder
        ...     .with_architecture("vae_decoder")
        ...     .with_latent_dim(64)
        ...     .with_output_channels(2)
        ...     .validate()
        ...     .build())
    """

    def __init__(self, config: TrainingSettings, device: str = "cuda"):
        """Initialize decoder builder.

        Args:
            config: Training configuration
            device: Device to place model on (cuda/cpu)
        """
        super().__init__()
        self._config = config
        self._device = torch.device(device)
        self._architecture: str | None = None
        self._latent_dim: int | None = None
        self._out_channels: int | None = None
        self._kwargs: dict[str, Any] = {}
        logger.info(f"DecoderBuilder initialized on {self._device}")

    def with_architecture(self, architecture: str) -> "DecoderBuilder":
        """Set decoder architecture.

        Args:
            architecture: Decoder type (vae_decoder, residual_decoder, etc)

        Returns:
            self for chaining
        """
        self._architecture = architecture
        return self

    def with_latent_dim(self, dim: int) -> "DecoderBuilder":
        """Set latent space dimension.

        Args:
            dim: Latent dimension

        Returns:
            self for chaining
        """
        self._latent_dim = dim
        return self

    def with_output_channels(self, channels: int) -> "DecoderBuilder":
        """Set number of output channels.

        Args:
            channels: Output channel count

        Returns:
            self for chaining
        """
        self._out_channels = channels
        return self

    def with_parameter(self, key: str, value: Any) -> "DecoderBuilder":
        """Set custom decoder parameter.

        Args:
            key: Parameter name
            value: Parameter value

        Returns:
            self for chaining
        """
        self._kwargs[key] = value
        return self

    def validate(self) -> "DecoderBuilder":
        """Validate builder state.

        Returns:
            self for chaining

        Raises:
            ValueError: If required parameters missing
        """
        super().validate()

        if self._architecture is None:
            raise ValueError("Architecture not specified")

        if self._latent_dim is None:
            self._latent_dim = self._config.model.latent_dim

        if self._out_channels is None:
            self._out_channels = self._config.model.out_channels

        return self

    def build(self) -> nn.Module:
        """Build and return decoder model.

        Returns:
            Configured decoder model

        Raises:
            ValueError: If validation fails
        """
        self.validate()

        try:
            from spectramr.models.factories.model_factory import (
                ModelRegistry,
                ParameterMapper,
            )

            registry = ModelRegistry()
            if not registry.has_generator(self._architecture):
                raise ValueError(f"Decoder '{self._architecture}' not registered.")

            decoder_class = registry.get_generator_class(self._architecture)

            mapper = ParameterMapper()
            mapped_params = mapper.map_generator_params(
                self._architecture,
                latent_dim=self._latent_dim,
                out_channels=self._out_channels,
                **self._kwargs,
            )

            decoder = decoder_class(**mapped_params)

            # Move to device
            decoder = decoder.to(self._device)

            self._product = decoder
            logger.info(
                f"Decoder built: {self._architecture} "
                f"({self._latent_dim}→{self._out_channels}) on {self._device}"
            )

            return decoder

        except ImportError as e:
            raise ImportError(f"Failed to import ModelFactory: {e}") from e


__all__ = [
    "DecoderBuilder",
    "DiscriminatorBuilder",
    "EncoderBuilder",
    "GeneratorBuilder",
]
