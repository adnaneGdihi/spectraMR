"""Inference Strategies Module

This module contains inference strategies for different training paradigms,
including GAN, diffusion, VAE, VQVAE, disentangled, physics-driven, and
reconstruction inference strategies.
"""

from .base_inference_strategy import BaseInferenceStrategy
from .cold_diffusion_inference_strategy import ColdDiffusionInferenceStrategy
from .diffusion_inference_strategy import DiffusionInferenceStrategy
from .disentangled_inference_strategy import DisentangledInferenceStrategy
from .domain_adaptation_inference_strategy import DomainAdaptationInferenceStrategy
from .gan_inference_strategy import GANInferenceStrategy
from .latent_diffusion_inference_strategy import LatentDiffusionInferenceStrategy
from .latent_gan_inference_strategy import LatentGANInferenceStrategy
from .mae_inference_strategy import MAEInferenceStrategy
from .physics_driven_inference_strategy import PhysicsDrivenInferenceStrategy
from .reconstruction_inference_strategy import ReconstructionInferenceStrategy
from .ssl_inference_strategy import SSLInferenceStrategy
from .vae_inference_strategy import VAEInferenceStrategy
from .vqvae_inference_strategy import VQVAEInferenceStrategy

__all__ = [
    "BaseInferenceStrategy",
    "ColdDiffusionInferenceStrategy",
    "DiffusionInferenceStrategy",
    "DisentangledInferenceStrategy",
    "DomainAdaptationInferenceStrategy",
    "GANInferenceStrategy",
    "LatentDiffusionInferenceStrategy",
    "LatentGANInferenceStrategy",
    "MAEInferenceStrategy",
    "PhysicsDrivenInferenceStrategy",
    "ReconstructionInferenceStrategy",
    "SSLInferenceStrategy",
    "VAEInferenceStrategy",
    "VQVAEInferenceStrategy",
]

# `create_inference_strategy` lived here as a second dispatcher for "which
# inference strategy does this config want". `InferenceStrategyFactory`
# (inference_factory.py) is what `pipelines/infer.py` calls, maps a strict
# superset of the types, and raises on an unknown one; this copy had zero
# callers and detected cold diffusion by a different rule (non-negotiable 17).
