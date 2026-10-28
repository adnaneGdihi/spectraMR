# Diffusion implementations package
# Exports all diffusion process classes for easy importing

# Re-export SinusoidalPositionEmbeddings for backward compatibility
from spectramr.models.reconstruction.unet import SinusoidalPositionEmbeddings

# audit_plan_novel — novel diffusion model registrations
# (fires @register_model for "score_field_unet" and "riemannian_diffusion_qmap")
from . import (
    riemannian_bloch,  # noqa: F401
    score_field_unet,  # noqa: F401
)
from .base_diffusion import Diffusion
from .blurring_diffusion import BlurringDiffusion
from .chi_square_diffusion import ChiSquareDiffusion
from .classifier_free_guidance import ClassifierFreeGuidanceSampler
from .classifier_guidance import ClassifierGuidanceSampler
from .cold_diffusion import ColdDiffusion
from .cold_mri_sampler import ColdMRISampler

# Import components for backward compatibility
from .components import DiffusionPrior, StandardDiffusionUNet
from .diffusion_parts.kan_diffusion import ColdDiffusionKAN, DiffusionKAN
from .diffusion_parts.score_based_diffusion import ScoreBasedDiffusion
from .diffusion_parts.stable_diffusion import StableDiffusion
from .diffusion_parts.swin_diffusion import SwinScoreDiffusion
from .diffusion_parts.vit_diffusion import ViTScoreDiffusion
from .laplace_diffusion import LaplaceDiffusion
from .mrf_diph import (
    BlochDictionary,
    BlochProjector,
    MRFDiPhSampler,
    create_mrf_diph_sampler,
)
from .physics_guided_sampler import DCMode, PhysicsGuidedReverseSampler
from .pula_sampler import DPSMRISampler, MCGSampler, PULASampler
from .rician_diffusion import RicianDiffusion

# audit_plan_novel SFC §6: Teichmüller schedule head for cold diffusion
from .teichmuller_schedule import TeichmullerScheduleHead, teichmuller_mu_schedule  # noqa: F401

__all__ = [
    "BlurringDiffusion",
    "ChiSquareDiffusion",
    "ColdDiffusion",
    "ColdDiffusionKAN",
    "Diffusion",
    "DiffusionKAN",
    "DiffusionPrior",
    "LaplaceDiffusion",
    "RicianDiffusion",
    "ScoreBasedDiffusion",
    "StableDiffusion",
    "StandardDiffusionUNet",
    "SwinScoreDiffusion",
    "ViTScoreDiffusion",
    # MRF-DiPh
    "BlochDictionary",
    "BlochProjector",
    "MRFDiPhSampler",
    "create_mrf_diph_sampler",
    # Advanced Samplers
    "PULASampler",
    "MCGSampler",
    "DPSMRISampler",
    "ColdMRISampler",
    # Guidance samplers (composable reverse-step correctors)
    "ClassifierGuidanceSampler",
    "ClassifierFreeGuidanceSampler",
    "PhysicsGuidedReverseSampler",
    "DCMode",
]
