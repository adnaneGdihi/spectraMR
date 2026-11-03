"""Inference Strategy Factory

Factory for creating inference strategies based on configuration.
"""

from typing import Any

import torch
from torch import nn

from spectramr.infrastructure.inference.base_inference_strategy import BaseInferenceStrategy
from spectramr.infrastructure.inference.cascaded_diffusion_inference_strategy import (
    CascadedDiffusionInferenceStrategy,
)
from spectramr.infrastructure.inference.cold_diffusion_inference_strategy import (
    ColdDiffusionInferenceStrategy,
)
from spectramr.infrastructure.inference.diffusion_inference_strategy import (
    DiffusionInferenceStrategy,
)
from spectramr.infrastructure.inference.disentangled_inference_strategy import (
    DisentangledInferenceStrategy,
)
from spectramr.infrastructure.inference.domain_adaptation_inference_strategy import (
    DomainAdaptationInferenceStrategy,
)
from spectramr.infrastructure.inference.gan_inference_strategy import GANInferenceStrategy
from spectramr.infrastructure.inference.geomamba_ulf_inference_strategy import (
    GeoMambaULFInferenceStrategy,
)
from spectramr.infrastructure.inference.latent_diffusion_inference_strategy import (
    LatentDiffusionInferenceStrategy,
)
from spectramr.infrastructure.inference.latent_gan_inference_strategy import (
    LatentGANInferenceStrategy,
)
from spectramr.infrastructure.inference.mae_inference_strategy import MAEInferenceStrategy
from spectramr.infrastructure.inference.multi_inference_strategy import (
    MultiInferenceStrategy,
)
from spectramr.infrastructure.inference.physics_driven_inference_strategy import (
    PhysicsDrivenInferenceStrategy,
)
from spectramr.infrastructure.inference.reconstruction_inference_strategy import (
    ReconstructionInferenceStrategy,
)
from spectramr.infrastructure.inference.ssl_inference_strategy import SSLInferenceStrategy
from spectramr.infrastructure.inference.strategy_detector import create_default_detector
from spectramr.infrastructure.inference.vae_inference_strategy import VAEInferenceStrategy
from spectramr.infrastructure.inference.vqvae_inference_strategy import VQVAEInferenceStrategy


class InferenceStrategyFactory:
    """Factory for creating inference strategies based on configuration.

    Uses declarative StrategyDetector for type inference, reducing complexity
    from CC=31 (Grade E) to CC=2 (Grade A).
    """

    # Shared detector instance (initialized once)
    _detector = None

    @classmethod
    def _get_detector(cls):
        """Get or create the strategy detector singleton."""
        if cls._detector is None:
            cls._detector = create_default_detector()
        return cls._detector

    @staticmethod
    def create(
        model: nn.Module,
        device: torch.device,
        config: dict[str, Any],
        strategy_type: str | None = None,
    ) -> BaseInferenceStrategy:
        """Create an inference strategy instance.

        Args:
            model: The trained model
            device: Device to run on
            config: Configuration dictionary
            strategy_type: Optional explicit strategy type override

        Returns:
            Instantiated inference strategy
        """
        # Determine strategy type
        if strategy_type is None:
            strategy_type = InferenceStrategyFactory._infer_strategy_type(config)

        strategy_map = {
            "gan": GANInferenceStrategy,
            "diffusion": DiffusionInferenceStrategy,
            "cold_diffusion": ColdDiffusionInferenceStrategy,
            "cascaded_diffusion": CascadedDiffusionInferenceStrategy,
            "reconstruction": ReconstructionInferenceStrategy,
            "vae": VAEInferenceStrategy,
            "vqvae": VQVAEInferenceStrategy,
            "domain_adaptation": DomainAdaptationInferenceStrategy,
            "latent_gan": LatentGANInferenceStrategy,
            "latent_diffusion": LatentDiffusionInferenceStrategy,
            "mae": MAEInferenceStrategy,
            "disentangled": DisentangledInferenceStrategy,
            "physics_driven": PhysicsDrivenInferenceStrategy,
            "geomamba_ulf": GeoMambaULFInferenceStrategy,
            "ssl": SSLInferenceStrategy,
            "multi": MultiInferenceStrategy,
        }

        key = strategy_type.lower()
        strategy_class = strategy_map.get(key)
        if strategy_class is None:
            # CLAUDE.md pitfall #9: never silently fall back to a
            # different paradigm. Earlier this defaulted to
            # ReconstructionInferenceStrategy, masking typos in
            # ``strategy_type`` and producing wrong reconstructions
            # without warning. See TODO/audit/17_remaining_infrastructure.md F1.
            raise ValueError(
                f"Unknown inference strategy_type {strategy_type!r}. "
                f"Known types: {sorted(strategy_map)}. If you intended "
                f"a custom strategy, register it in "
                f"src/infrastructure/inference/inference_factory.py."
            )

        return strategy_class(model, device, config)

    @staticmethod
    def _infer_strategy_type(config: dict[str, Any]) -> str:
        """Infer strategy type from configuration using declarative detector.

        Delegates to StrategyDetector which uses priority-ordered rules
        to determine the appropriate strategy type.

        Args:
            config: Configuration dictionary.

        Returns:
            Strategy type name (e.g., 'gan', 'diffusion', 'reconstruction').
        """
        detector = InferenceStrategyFactory._get_detector()
        return detector.detect(config, default="reconstruction")
